"""
evaluator.py — Quality checks for parsed and enriched company profiles.

Two evaluation functions:

  evaluate_parse(profile, article_title, article_summary) -> dict
      LLM-as-judge: verifies the parser extracted the right company from the article.
      Returns: {"passed": bool, "confidence": int (0-100), "issues": list[str]}

  evaluate_research(profile) -> dict
      Structural completeness check (no LLM): scores how many key fields were found.
      Returns: {"completeness_score": float, "found_fields": list, "missing_fields": list, "passed": bool}

  log_eval(entry: dict) -> None
      Appends an evaluation entry to eval_log.jsonl with a timestamp.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"
_client: openai.OpenAI | None = None

_EVAL_LOG_PATH = Path("eval_log.jsonl")

_RESEARCH_REQUIRED_FIELDS = ["founders", "total_raised", "investors", "founded_date"]

_PARSE_EVAL_SYSTEM = """\
You are a quality-control reviewer for a startup data extraction pipeline.

You will be given an article (title and summary) and the structured company profile that was \
extracted from it by an LLM parser.

Evaluate whether the extraction is accurate. Check:
1. Does company_name match the primary startup being written about in the article?
2. Is the stage (pre-seed/seed/series-a/unknown) plausible given the article content?
3. Does the one_line_description accurately reflect what the company does?
4. Are the verticals (AI, blockchain, etc.) reasonable given the article?

Return a JSON object with exactly these three fields:
- passed: true if the extraction looks correct overall; false if there is a critical error \
(e.g. wrong company identified, description is completely off)
- confidence: integer 0-100 indicating extraction quality (100=perfect, 50=uncertain, 0=clearly wrong)
- issues: list of strings describing specific problems; empty list if no issues found\
"""


class _ParseEvalResult(BaseModel):
    passed: bool
    confidence: int = Field(ge=0, le=100)
    issues: list[str] = Field(default_factory=list)


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set — add it to .env")
        _client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            default_headers={"HTTP-Referer": "lightning-agent"},
        )
    return _client


def evaluate_parse(profile: dict, article_title: str, article_summary: str) -> dict:
    """LLM-as-judge: verify the parser extracted the right company from the article.

    Args:
        profile: Parsed profile dict (company_name, stage, verticals, one_line_description).
        article_title: Original article title.
        article_summary: Original article summary.

    Returns:
        {"passed": bool, "confidence": int, "issues": list[str]}
        On LLM failure, returns passed=True with confidence=-1 (eval unavailable, keep the profile).
    """
    article_text = f"Title: {article_title}\nSummary: {article_summary}"
    profile_text = json.dumps(
        {
            "company_name": profile.get("company_name"),
            "stage": profile.get("stage"),
            "verticals": profile.get("verticals"),
            "one_line_description": profile.get("one_line_description"),
        },
        indent=2,
    )
    user_message = (
        f"Article:\n{article_text}\n\n"
        f"Extracted profile:\n{profile_text}\n\n"
        "Evaluate whether this extraction is correct."
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _PARSE_EVAL_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            timeout=20,
        )
        content = response.choices[0].message.content
        if not content:
            return {"passed": True, "confidence": -1, "issues": ["eval skipped: empty response"]}
        result = _ParseEvalResult.model_validate(json.loads(content))
        return {"passed": result.passed, "confidence": result.confidence, "issues": result.issues}

    except openai.APITimeoutError:
        logger.warning("Parse eval timed out for %s", profile.get("company_name"))
        return {"passed": True, "confidence": -1, "issues": ["eval skipped: timeout"]}
    except openai.RateLimitError:
        logger.warning("Parse eval rate limited for %s", profile.get("company_name"))
        return {"passed": True, "confidence": -1, "issues": ["eval skipped: rate limited"]}
    except (openai.AuthenticationError, openai.APIConnectionError, openai.APIStatusError) as exc:
        logger.warning("Parse eval API error for %s: %s", profile.get("company_name"), exc)
        return {"passed": True, "confidence": -1, "issues": [f"eval skipped: {type(exc).__name__}"]}
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Parse eval parse error for %s: %s", profile.get("company_name"), exc)
        return {"passed": True, "confidence": -1, "issues": ["eval skipped: bad response format"]}


def evaluate_research(profile: dict) -> dict:
    """Structural completeness check: score how many key fields the research found.

    No LLM call — purely checks field presence in the enriched profile.

    Args:
        profile: Enriched profile dict.

    Returns:
        {
            "completeness_score": float (0-100),
            "found_fields": list[str],
            "missing_fields": list[str],
            "passed": bool  # True if at least 2 of 4 required fields are present
        }
    """
    found = [f for f in _RESEARCH_REQUIRED_FIELDS if profile.get(f)]
    missing = [f for f in _RESEARCH_REQUIRED_FIELDS if not profile.get(f)]
    completeness = (len(found) / len(_RESEARCH_REQUIRED_FIELDS)) * 100
    return {
        "completeness_score": completeness,
        "found_fields": found,
        "missing_fields": missing,
        "passed": len(found) >= 2,
    }


def log_eval(entry: dict) -> None:
    """Append an evaluation entry to eval_log.jsonl with an ISO timestamp."""
    entry_with_ts = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    try:
        with _EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_ts) + "\n")
    except OSError as exc:
        logger.warning("Could not write to eval_log.jsonl: %s", exc)
