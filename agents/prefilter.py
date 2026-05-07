import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"
_llm: ChatOpenAI | None = None

_SYSTEM_PROMPT = """\
You are a strict filter for Lightning Capital, a seed-to-Series-A venture fund that writes \
$500K-$5M checks into early-stage AI, blockchain, and digital asset startups.
Answer with ONLY 'YES' or 'NO' — no explanation, no punctuation, nothing else.
Should this company be considered for seed-stage VC investment?

Answer NO if the company is any of the following:
- A large established corporation (any industry)
- A bank, financial institution, or asset manager
- A government body, regulator, nonprofit, or foundation
- A public company or a company valued over $100M
- A protocol, DAO, or decentralized network (not a company)
- A company that has already exited (acquired or IPO'd)

Answer YES only if the company appears to be a small, early-stage startup that could \
plausibly be at seed or Series A stage.

Examples:
Company: Draft AI
Description: AI-powered contract drafting tool for startups and SMBs
Verticals: AI, Legal Tech
Stage: Seed
Answer: YES

Company: Google DeepMind
Description: AI research lab and product division of Alphabet
Verticals: AI
Stage: Growth
Answer: NO

Company: Ethereum Foundation
Description: Nonprofit organization supporting the Ethereum protocol and ecosystem
Verticals: Blockchain
Stage: Unknown
Answer: NO

Company: Chainlink
Description: Decentralized oracle network connecting smart contracts to real-world data
Verticals: Blockchain, DeFi
Stage: Unknown
Answer: NO

Company: Mesh Protocol
Description: Early-stage blockchain infrastructure startup building cross-chain settlement rails
Verticals: Blockchain, Infrastructure
Stage: Series A
Answer: YES\
"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_MODEL,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=4,
        )
    return _llm


_BLACKLIST = {
    "anthropic", "anthropic labs", "openai", "google", "microsoft", "apple", "meta", "amazon",
    "doordash", "western union", "gamestop", "coinbase", "mastercard",
    "claude code", "claude design", "claude opus", "claude sonnet", "claude haiku",
}


def is_worth_scoring(profile: dict) -> tuple[bool, str]:
    """Returns (should_score, reason) using a single fast LLM triage call."""
    company_name = profile.get("company_name") or "Unknown"
    if company_name.lower().strip() in _BLACKLIST:
        return False, f"hard blacklist: {company_name}"

    description = profile.get("one_line_description") or profile.get("description") or ""
    verticals = profile.get("verticals") or ""
    if isinstance(verticals, list):
        verticals = ", ".join(verticals)
    stage = profile.get("stage") or ""

    user_msg = (
        f"Company: {company_name}\n"
        f"Description: {description}\n"
        f"Verticals: {verticals}\n"
        f"Stage: {stage}"
    )

    try:
        response = _get_llm().invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        answer = response.content.strip().upper()
        if answer.startswith("YES"):
            return True, "passed triage"
        return False, f"triage filtered: {response.content.strip()}"
    except Exception as exc:
        logger.warning("Prefilter LLM call failed for %s: %s — defaulting to pass", company_name, exc)
        return True, "triage error (defaulting to pass)"
