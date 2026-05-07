"""
scorer.py — Ranks startup profiles by relevance and VC-fit criteria.

Responsibilities:
  - Accept structured startup profiles from parser.py.
  - Apply a configurable scoring rubric that weighs factors such as:
      * Team signal (repeat founders, relevant background)
      * Market timing (AI/blockchain vertical alignment, trend fit)
      * Traction indicators (launch recency, community engagement)
      * Funding stage fit (pre-seed / seed preference)
  - Optionally use an LLM to produce a qualitative investment thesis per startup.
  - Attach a numeric score and short rationale to each profile.
  - Return profiles sorted by descending score.

Primary interface:
    score(profiles: list[dict]) -> list[dict]
        Returns profiles with 'score' and 'rationale' fields added, sorted
        highest-score first.
"""

import json
import logging
import os
import time

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

_client: openai.OpenAI | None = None

_MODEL = "meta-llama/llama-3.3-70b-instruct"

_SYSTEM_PROMPT = """\
You are a senior analyst at a venture capital fund focused on seed-to-Series-A investments \
in AI, blockchain, and digital asset companies. Evaluate early-stage startups against the \
fund's thesis.

HARD DISQUALIFICATION RULES — apply these before anything else:
Automatically assign a score of 20 or below if ANY of the following are true:
- The company has raised more than $20M in total funding.
- The company has a valuation above $100M.
- The company has more than $10M in ARR.
- The company has more than 100 employees.
- The company is publicly traded.
- The company is a large established institution (bank, fund, exchange, or foundation).
Lightning Capital only invests at pre-seed and seed stage. Early traction is a positive \
signal; scale is a disqualifier. If a company shows signs of significant scale or maturity, \
score it 20 or below regardless of other factors.

Scoring rubric (0–100) — only apply this after confirming none of the disqualifiers above are met:
- Team signal: repeat founders, domain expertise, relevant pedigree
- Market timing: alignment with AI/blockchain/digital asset trends, regulatory tailwinds
- Technology differentiation: moat, defensibility, genuine innovation vs. thin wrapper
- Traction indicators: launch recency, community engagement, early customers or revenue
- Stage fit: penalise late-stage or clearly off-thesis companies heavily

A score ≥ 70 warrants flag = true (worth the investment team reviewing).

Write rationale as plain-text bullet points, one per line, using exactly this format:
- Sector: <sector>
- Stage: <stage>
- Raised: <amount>
- Founders: <names>
- Investors: <investor names, funds, or accelerators>
- Why: <one sentence on why this is or is not a fit for a seed-stage AI/blockchain fund>

Rules for the rationale:
- Omit any field entirely if the information is not present in the context — do not write "unknown".
- The Why field is always required; all others are included only when the information is available.
- For Investors, include any VCs, angels, or accelerators mentioned in relation to the company.
- Use plain text only — no markdown, no bold, no extra punctuation around field names.\
"""


class _CompanyScore(BaseModel):
    score: int = Field(ge=0, le=100, description="VC-fit score, integer 0–100")
    rationale: str = Field(description="Exactly two sentences of investment rationale")
    flag: bool = Field(description="True if the company merits investment team review")


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set — add it to .env")
        print(f"[scorer] OPENROUTER_API_KEY prefix: {api_key[:10]}")
        _client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,  # disable auto-retry; time.sleep handles pacing
        )
    return _client


def score_company(company_name: str, context: str) -> dict:
    """Score a startup for VC fit using an LLM acting as a fund analyst.

    Args:
        company_name: Name of the company to evaluate.
        context: Free-text context — article summary, description, funding news, etc.

    Returns:
        Dict with keys:
            score (int, 0–100), rationale (str), flag (bool).
        On failure the same keys are zeroed/falsy and an extra 'error' key is set.
    """
    client = _get_client()

    time.sleep(15)  # let the rate limit window reset after the parse phase

    user_message = (
        f"Company: {company_name}\n\n"
        f"Context:\n{context}\n\n"
        "Score this company for our fund.\n\n"
        'You MUST respond with only a valid JSON object containing exactly these three fields: '
        'score (integer 0-100), flag (boolean), rationale (string, bullet points as instructed). '
        'No other text, no markdown, no code blocks. '
        'Example: {"score": 75, "flag": true, "rationale": "- Sector: Legal AI\\n- Stage: Seed\\n- Raised: $4M\\n- Founders: Max Junestrand\\n- Investors: a16z, YC, Naval Ravikant\\n- Why: Novel approach to contract automation in a large underserved market."}'
    )

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        result = _CompanyScore.model_validate(
            json.loads(response.choices[0].message.content)
        )
        return {
            "score": result.score,
            "rationale": result.rationale,
            "flag": result.flag,
        }

    except openai.BadRequestError as exc:
        logger.warning("Bad request scoring %s: %s", company_name, exc.message)
        return {"score": 0, "rationale": "", "flag": False, "error": str(exc.message)}
    except openai.AuthenticationError:
        logger.error("Invalid OpenRouter API key — check OPENROUTER_API_KEY env var")
        return {"score": 0, "rationale": "", "flag": False, "error": "authentication_error"}
    except openai.RateLimitError:
        logger.warning("Rate limited scoring %s", company_name)
        return {"score": 0, "rationale": "", "flag": False, "error": "rate_limit"}
    except openai.APIStatusError as exc:
        logger.warning("API error %d scoring %s: %s", exc.status_code, company_name, exc.message)
        return {"score": 0, "rationale": "", "flag": False, "error": f"api_error_{exc.status_code}"}
    except openai.APIConnectionError as exc:
        logger.warning("Network error scoring %s: %s", company_name, exc)
        return {"score": 0, "rationale": "", "flag": False, "error": "connection_error"}
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Failed to parse response for %s: %s", company_name, exc)
        return {"score": 0, "rationale": "", "flag": False, "error": "parse_error"}
    finally:
        time.sleep(8)
