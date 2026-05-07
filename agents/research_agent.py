import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.tools import (
    get_funding_info,
    get_founder_info,
    get_company_overview,
    get_recent_news,
    get_sec_filing,
)

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"

_SYSTEM_PROMPT = """\
You are a research analyst at Lightning Capital, a seed-to-Series-A venture fund that writes \
$500K-$5M checks into AI, blockchain, and digital asset companies.

HARD DISQUALIFIERS — immediately score 0 and flag=false if any of these are true (only apply if you have a VERIFIED source for the number — do not apply based on estimated or unknown figures):
- Company valuation exceeds $500 million
- Company has raised more than $50 million total
- Company is a public company, bank, or financial institution (e.g. Morgan Stanley, JPMorgan, BlackRock)
- Company is a well-known large tech company (e.g. Apple, Uber, Anthropic, OpenAI, Google)
- Company has already had an exit (acquired, IPO'd)

If funding or valuation data is missing or unverified, do NOT apply hard disqualifiers. Score based on available signals (product, team, thesis fit) and note the data gap.

IDEAL PROFILE — score 70+ only if:
- Pre-seed, seed, or very early Series A
- Raised less than $20M total
- Building in AI, blockchain, web3, or digital asset infrastructure
- Has a small founding team with relevant background
- Has early traction but is not yet at scale

SCORING RUBRIC:
0        — hard disqualifier triggered, or complete research failure with no signal at all
10–30    — off-thesis, wrong stage, or established company
40–60    — on-thesis but key data missing (funding unknown, founders not found)
70–85    — on-thesis, early stage, credible signals found
85–100   — exceptional fit: strong team, right stage, clear traction, ideal investors

Be skeptical and conservative. It is better to miss a company than to waste the investment team's \
time on off-thesis candidates.

RESEARCH PROTOCOL — use these tools in order:
1. get_funding_info — find raise amount, round stage, investors
2. get_founder_info — find founder names, roles, and backgrounds
3. get_company_overview — scrape the company homepage for product description
4. get_recent_news — find recent launches, press, or traction signals
5. get_sec_filing — check for Form D filings if funding info is unclear

Only call a tool if the information is not already present in the profile.
Do not hallucinate results — if a tool returns nothing, note the gap in research_summary.

ANTI-HALLUCINATION RULES:
- Never guess, infer, or fill in plausible-sounding values for funding, investors, or founders.
- Every data point you return must be traceable to a real source you actually retrieved with a tool.
- If you cannot find verified information for a field after exhausting searches, return null for \
  that field. Do not fabricate.
- If funding or valuation information could not be found or verified, do not invent or estimate \
  numbers. Do not apply hard disqualifiers based on missing data. Note the gap in data_confidence \
  and score on the signals you did find (product, team, thesis fit).

PRE-RESEARCHED DATA: Fields already populated in the incoming profile are confirmed facts — do \
not re-search them. Only run tool calls for fields that are null or absent.

NAME CORRECTION: If company_name looks like a generic term (e.g. "Series", "Round", "Funding", \
"Seed", "Capital"), search the original article URL to find the actual company name and return it \
as "corrected_name" in your JSON.

OUTPUT — when all research is complete, end your final message with a JSON object in <json> tags.

Required fields:
  score          — integer 0–100
  flag           — true if the company meets the Ideal Profile criteria, else false
  rationale      — structured bullet points explaining the score
  research_summary — 1–2 sentence plain-English summary of what you found
  data_confidence — one of:
    "verified"   — funding/investor data came from a credible source (TechCrunch, Crunchbase, \
                   company blog, official press release)
    "unverified" — data came from a secondary aggregator or low-signal source
    "not_found"  — no funding/investor information found after exhausting all search steps

Optional fields (include when found; omit or set to null if not found after all steps):
  founders       — full names, comma-separated
  total_raised   — concise string, e.g. "$4M" or "$2.5M seed"
  investors      — VC/angel/accelerator names, comma-separated
  corrected_name — only if the original company_name was wrong

EXAMPLE RESEARCH PROCESS:

Profile received: {"company_name": "Clera", "one_line_description": "AI talent agent connecting candidates to hiring managers", "stage": "unknown", "verticals": ["AI"]}

Step 1 — get_funding_info("Clera")
Result: "Clera raises $3M pre-seed to build AI talent agent... investors include Soma Capital"

Step 2 — get_founder_info("Clera")
Result: "Founded by Alex Rivera (ex-LinkedIn) and Priya Patel (ex-Stripe)"

Step 3 — get_company_overview("clera.ai")
Result: "Clera represents candidates directly to hiring managers at venture-backed startups"

Step 4 — get_recent_news("Clera")
Result: "Clera launches private beta with 200 waitlist signups"

Conclusion: Early-stage AI startup, $3M pre-seed, strong founding team, on-thesis. Score 80.

<json>{"score": 80, "flag": true, "rationale": "- Sector: AI, future of work\\n- Stage: pre-seed\\n- Raised: $3M\\n- Founders: ex-LinkedIn, ex-Stripe\\n- Investors: Soma Capital\\n- Why: early-stage AI talent tool with credible team and early traction", "research_summary": "Clera raised a $3M pre-seed from Soma Capital. Founded by ex-LinkedIn and ex-Stripe engineers.", "founders": "Alex Rivera, Priya Patel", "total_raised": "$3M", "investors": "Soma Capital", "data_confidence": "verified"}</json>

---

EXAMPLE WITH MISSING DATA:

Profile received: {"company_name": "Brila", "one_line_description": "generates one-page websites from Google Maps reviews", "stage": "unknown", "verticals": ["AI"]}

Step 1 — get_funding_info("Brila")
Result: "No funding information found for Brila."

Step 2 — get_founder_info("Brila")
Result: "No founder information found for Brila."

Step 3 — get_company_overview scrape returns minimal content.

Step 4 — get_recent_news("Brila")
Result: "No recent news found for Brila."

Conclusion: On-thesis product concept but no verifiable data found. Score 40 — do not score 0 since no hard disqualifier was triggered.

<json>{"score": 40, "flag": false, "rationale": "- Sector: AI\\n- Stage: unknown\\n- Raised: unknown\\n- Founders: not found\\n- Why: on-thesis product but no verifiable data found after exhausting all searches", "research_summary": "No funding, founder, or traction data found for Brila after exhausting all search steps.", "founders": null, "total_raised": null, "investors": null, "data_confidence": "not_found"}</json>\
"""

_tools = [get_funding_info, get_founder_info, get_company_overview, get_recent_news, get_sec_filing]

_compiled_graph = None
_llm = None

_MAX_ITERATIONS = 20

_FALLBACK_PROMPT = (
    'You must now respond with ONLY a JSON object and nothing else. '
    'No text before or after. '
    'Format: {"score": 75, "flag": true, "rationale": "...", "research_summary": "brief summary", '
    '"founders": "Jane Smith, John Doe", "total_raised": "$4M", "investors": "a16z, YC", '
    '"data_confidence": "verified"} '
    '(data_confidence must be "verified", "unverified", or "not_found"; '
    'omit founders/total_raised/investors if unknown)'
)


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _llm


def _get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(_tools)

    def agent_node(state: MessagesState) -> dict:
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(_tools))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    _compiled_graph = builder.compile()
    return _compiled_graph


def research_and_score(profile: dict) -> dict:
    company_name = profile.get("company_name", "Unknown")
    try:
        app = _get_graph()
        user_content = (
            f"Please research and score the following startup:\n\n"
            f"{json.dumps(profile, indent=2)}"
        )
        iteration = 0
        last_agent_messages = None
        all_messages = [HumanMessage(content=user_content)]
        _t0 = time.time()
        print("      [Agent] Calling LLM (iteration 1)...", flush=True)

        for step in app.stream(
            {"messages": [HumanMessage(content=user_content)]},
            config={"recursion_limit": _MAX_ITERATIONS},
        ):
            if "agent" in step:
                iteration += 1
                elapsed = int((time.time() - _t0) * 1000)
                print(f"      [Agent] Response received ({elapsed}ms)", flush=True)
                logger.debug("Agent loop iteration %d of %d", iteration, _MAX_ITERATIONS)
                last_agent_messages = step["agent"]["messages"]
                all_messages.extend(last_agent_messages)
                for msg in last_agent_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            logger.info(
                                "Agent calling tool: %s | args: %s",
                                tc["name"],
                                tc["args"],
                            )
                        _t0 = time.time()
                        print(f"      [Agent] Calling LLM (iteration {iteration + 1})...", flush=True)
            elif "tools" in step:
                for msg in step["tools"]["messages"]:
                    result_preview = str(msg.content)[:150]
                    logger.info(
                        "Tool result: %s → %s...",
                        getattr(msg, "name", "tool"),
                        result_preview,
                    )
                all_messages.extend(step["tools"]["messages"])

        logger.info("Agent finished research after %d iterations", iteration)

        if not last_agent_messages:
            raise ValueError("Agent produced no output")

        last_message = last_agent_messages[-1]
        raw_text = last_message.content
        if isinstance(raw_text, list):
            raw_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_text
            )
        raw_text = raw_text.strip()
        logger.info("Agent raw final response: %s", raw_text)

        # Primary: extract JSON from <json>...</json> tags
        tag_match = re.search(r"<json>(.*?)</json>", raw_text, re.DOTALL)
        if tag_match:
            json_str = tag_match.group(1).strip()
        else:
            # Secondary: find first bare {...} object
            logger.warning("No <json> tags in final response — trying regex extraction")
            bare_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if bare_match:
                json_str = bare_match.group()
            else:
                # Fallback: one additional LLM call with strict prompt
                logger.warning("No JSON found in final response — attempting fallback LLM call")
                fallback_messages = all_messages + [HumanMessage(content=_FALLBACK_PROMPT)]
                fallback_response = _get_llm().invoke(fallback_messages)
                fallback_text = fallback_response.content
                if isinstance(fallback_text, list):
                    fallback_text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in fallback_text
                    )
                fallback_text = fallback_text.strip()
                logger.info("Fallback raw response: %s", fallback_text)
                fallback_match = re.search(r"\{.*\}", fallback_text, re.DOTALL)
                if not fallback_match:
                    raise ValueError(f"No JSON object found even after fallback. Original: {raw_text[:200]}")
                json_str = fallback_match.group()

        parsed = json.loads(json_str)
        raw_rationale = parsed.get("rationale", "")
        if isinstance(raw_rationale, list):
            raw_rationale = "\n".join(str(x) for x in raw_rationale)
        result = {
            "score": int(parsed.get("score", 0)),
            "flag": bool(parsed.get("flag", False)),
            "rationale": raw_rationale,
            "research_summary": parsed.get("research_summary", ""),
            "company_name": company_name,
        }
        for field in ("founders", "total_raised", "investors", "corrected_name", "data_confidence"):
            value = parsed.get(field)
            if value and isinstance(value, str) and value.strip():
                result[field] = value.strip()
        return result
    except Exception as exc:
        logger.warning("research_and_score failed for %s: %s", company_name, exc)
        return {
            "score": 0,
            "flag": False,
            "rationale": "",
            "error": str(exc),
            "company_name": company_name,
        }
