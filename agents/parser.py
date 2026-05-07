"""
parser.py — Extracts structured startup profiles from raw scraped content.

Responsibilities:
  - Accept raw result dicts produced by scraper.py.
  - Use an LLM (e.g. Claude) or rule-based extraction to pull structured
    fields: startup name, URL, founding year, founders, description,
    funding stage, verticals (AI / blockchain / etc.), and source metadata.
  - Normalize values (e.g. funding stage labels, date formats).
  - Return a list of structured startup profile dicts ready for scoring.

Primary interface:
    parse(raw_results: list[dict]) -> list[dict]
        Parses raw scrape output into clean, structured startup profiles.
"""

import json
import logging
import os
import re
import time
from typing import Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

logger = logging.getLogger(__name__)

_client: openai.OpenAI | None = None

_MODEL = "meta-llama/llama-3.3-70b-instruct"

_SYSTEM_PROMPT = """\
You are an expert at reading startup and tech news articles and identifying the primary company \
being written about.
Your task is to extract structured information about the main startup or company featured in \
an article. Focus on the company that is the subject of the news — not investors, acquirers, \
or incidentally mentioned firms.

Rules:
- If the article does not clearly feature a specific startup or company (e.g. it is a roundup, \
  opinion piece, or macro analysis with no single subject), set company_name to null.
- If the primary company is a large established corporation, big tech company, or well-known \
  public company (e.g. Google, Apple, Anthropic, Figma, Mercedes-Benz), set company_name to null \
  — do not extract them as the subject.
- stage must be one of: pre-seed, seed, series-a, unknown. Use "unknown" if not mentioned.
- verticals must be a list containing only values from: AI, blockchain, web3, digital-assets, \
  infrastructure, other. Include every applicable vertical; use ["other"] if none fit.
- one_line_description should be a single sentence (max 150 characters) describing what the \
  company does. Set to null if you cannot determine it from the article.
- founded_date should be the founding date in 'YYYY-MM' format (e.g. '2023-03'). If both month \
  and year are mentioned, use them. If only the year is mentioned, use 'YYYY-01' as a fallback. \
  Set to null if not mentioned at all.
- investors should be a comma-separated string listing any VCs, angel investors, funds, or \
  accelerators mentioned as investors in the company (e.g. 'a16z, YC, Naval Ravikant'). \
  Set to null if none are mentioned.

Always respond with a valid JSON object. Even when company_name is null, return JSON — never plain text.

Examples:

Title: Plurai raises $4M to build multi-agent AI testing infrastructure
{"company_name": "Plurai", "stage": "seed", "verticals": ["AI", "infrastructure"], "one_line_description": "Multi-agent AI testing infrastructure.", "founded_date": null, "investors": null}

Title: Figma for Agents — a new tool that brings visual workflow design to AI agents
Summary: A new startup called Canvas AI has launched a Figma-like interface for designing agent workflows.
{"company_name": "Canvas AI", "stage": "unknown", "verticals": ["AI"], "one_line_description": "Visual workflow design tool for AI agents.", "founded_date": null, "investors": null}

Title: Plurai
Summary: Generate training data and evaluations for AI agents.
{"company_name": "Plurai", "stage": "unknown", "verticals": ["AI"], "one_line_description": "Generates training data and evaluations for AI agents.", "founded_date": null, "investors": null}

Title: Claude Opus 4.7 released by Anthropic with improved reasoning
Summary: Anthropic today announced Claude Opus 4.7, its latest frontier model.
{"company_name": null, "stage": "unknown", "verticals": [], "one_line_description": null, "founded_date": null, "investors": null}

Title: Ask Product Hunt AI — find your next favorite product
Summary: Product Hunt has launched an AI-powered search assistant to help users discover new tools.
{"company_name": null, "stage": "unknown", "verticals": [], "one_line_description": null, "founded_date": null, "investors": null}\
"""


# Matches titles like "21 startups to watch", "top 10 AI companies", "best startups of the week"
_LISTICLE_RE = re.compile(
    r"(\d+\s+(startups?|companies|founders?)"
    r"|startups?\s+of\s+the\s+(week|month|year)"
    r"|(best|top)\s+(\d+\s+)?(startups?|companies|founders?))",
    re.IGNORECASE,
)

_LISTICLE_USER_TEMPLATE = """\
This article appears to be a list of multiple companies. Extract ALL startup companies mentioned.

Return a JSON object with a single key "companies" containing an array. Each element must have:
- name: company name
- one_line_description: one sentence description if available, else null
- stage: funding stage if mentioned ('pre-seed', 'seed', 'series-a'), else 'unknown'
- verticals: list from ["AI", "blockchain", "web3", "digital-assets", "infrastructure", "other"]
- investors: comma-separated investor names if mentioned, else null

Article:
{article_text}\
"""


class _ListicleItem(BaseModel):
    name: str
    one_line_description: str | None = None
    stage: Literal["pre-seed", "seed", "series-a", "unknown"] = "unknown"
    verticals: list[Literal["AI", "blockchain", "web3", "digital-assets", "infrastructure", "other"]] = Field(default_factory=list)
    investors: str | None = None


class _ListicleResponse(BaseModel):
    companies: list[_ListicleItem]


class _CompanyProfile(BaseModel):
    company_name: str | None = Field(
        default=None,
        description="Name of the primary startup featured in the article, or null if none"
    )
    stage: Literal["pre-seed", "seed", "series-a", "unknown"] = Field(
        default="unknown",
        description="Funding stage of the company"
    )
    verticals: list[Literal["AI", "blockchain", "web3", "digital-assets", "infrastructure", "other"]] = Field(
        default_factory=list,
        description="Applicable verticals for the company"
    )
    one_line_description: str | None = Field(
        default=None,
        description="One sentence describing what the company does, or null if unclear"
    )
    founded_date: str | None = Field(
        default=None,
        description="Founding date as 'YYYY-MM' (e.g. '2023-03'); use 'YYYY-01' if only year is known; null if not mentioned"
    )
    investors: str | None = Field(
        default=None,
        description="Comma-separated investor names (VCs, angels, accelerators) mentioned in the article; null if none"
    )

    @field_validator("stage", mode="before")
    @classmethod
    def _coerce_stage(cls, v: object) -> str:
        return v if v in ("pre-seed", "seed", "series-a", "unknown") else "unknown"

    @field_validator("verticals", mode="before")
    @classmethod
    def _coerce_verticals(cls, v: object) -> list:
        return v if isinstance(v, list) else []


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set — add it to .env")
        print(f"[parser] OPENROUTER_API_KEY prefix: {api_key[:10]}")
        _client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,  # disable auto-retry; time.sleep handles pacing
            default_headers={"HTTP-Referer": "lightning-agent"},
        )
    return _client


def _extract_json(content: str) -> dict | list | None:
    """Parse JSON from content, falling back to regex extraction if direct parse fails.

    The model sometimes returns plain-text refusals or wraps JSON in explanation text.
    This handles both cases without raising.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "last_run_parse_log.json")


def _save_parse_log(entries: list[dict]) -> None:
    path = os.path.normpath(_DEBUG_LOG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        logger.debug("Parse log written to %s (%d entries)", path, len(entries))
    except OSError as exc:
        logger.warning("Could not write parse log: %s", exc)


def parse(raw_results: list[dict]) -> list[dict]:
    """Extract structured startup profiles from raw article dicts.

    Args:
        raw_results: List of article dicts from fetch_articles(), each with
                     keys: title, url, summary, published_date.

    Returns:
        List of structured profile dicts with keys:
            company_name, stage, verticals, one_line_description,
            url, published_date.
        Articles where no company is identified are silently skipped.
        A full decision log is written to data/last_run_parse_log.json.
    """
    client = _get_client()
    profiles: list[dict] = []
    log_entries: list[dict] = []

    # Deduplicate by URL so the same article from multiple RSS feeds is only parsed once
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for article in raw_results:
        u = article.get("url", "")
        if u and u in seen_urls:
            continue
        if u:
            seen_urls.add(u)
        deduped.append(article)
    if len(deduped) < len(raw_results):
        print(f"      Deduplicated {len(raw_results) - len(deduped)} duplicate URLs before parsing", flush=True)
    raw_results = deduped

    total = len(raw_results)
    print(f"      Processing {total} articles...", flush=True)

    for i, article in enumerate(raw_results, 1):
        title = article.get("title", "")
        summary = article.get("summary", "")
        url = article.get("url", "")
        published_date = article.get("published_date")
        source = article.get("source", "")

        base_entry = {"title": title, "url": url, "published_date": published_date, "source": source}

        article_text = f"Title: {title}\n\nSummary: {summary}".strip()
        if not article_text or article_text == "Title: \n\nSummary:":
            log_entries.append({**base_entry, "decision": "skipped", "reason": "empty article text"})
            continue

        is_listicle = bool(_LISTICLE_RE.search(title))

        if is_listicle:
            messages = [
                {"role": "system", "content": "You extract structured startup data from article text. Return only valid JSON."},
                {"role": "user", "content": _LISTICLE_USER_TEMPLATE.format(article_text=article_text)},
            ]
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract the primary startup from the following article.\n\n{article_text}"},
            ]

        try:
            label = "listicle" if is_listicle else "article"
            print(f"      Calling LLM for {label} {i}/{total}: {title[:50]}...", flush=True)
            _t0 = time.time()
            response = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=30,
            )
            print(f"      ✓ Response received ({int((time.time() - _t0) * 1000)}ms)", flush=True)
            try:
                if not response or not response.choices or len(response.choices) == 0:
                    logger.warning("No choices in response for article '%s' — skipping", title[:60])
                    log_entries.append({**base_entry, "decision": "llm_error", "reason": "no choices in response"})
                    continue
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    logger.warning("Empty response content for article '%s' — skipping", title[:60])
                    log_entries.append({**base_entry, "decision": "llm_error", "reason": "empty response content"})
                    continue
            except (AttributeError, IndexError) as exc:
                logger.warning("Unexpected response shape for article '%s': %s", title[:60], exc)
                log_entries.append({**base_entry, "decision": "llm_error", "reason": str(exc)})
                continue

            if is_listicle:
                raw = _extract_json(content)
                if raw is None:
                    logger.warning("No JSON found in listicle response for '%s'", title[:60])
                    log_entries.append({**base_entry, "decision": "llm_error", "reason": "no JSON in response"})
                    continue
                listicle = _ListicleResponse.model_validate(raw)
                batch = [
                    {
                        "company_name": item.name,
                        "stage": item.stage,
                        "verticals": item.verticals,
                        "one_line_description": item.one_line_description,
                        "founded_date": None,
                        "investors": item.investors,
                        "url": url,
                        "published_date": published_date,
                        "source": source,
                    }
                    for item in listicle.companies
                    if item.name and item.name.strip()
                ]
                profiles.extend(batch)
                log_entries.append({
                    **base_entry,
                    "decision": "listicle",
                    "reason": f"listicle ({len(batch)} companies)",
                    "extracted_companies": [p["company_name"] for p in batch],
                })
                logger.info("Listicle: extracted %d companies from '%s'", len(batch), title[:60])
            else:
                parsed_json = _extract_json(content)
                if parsed_json is None:
                    logger.warning("No JSON found in response for '%s'", title[:60])
                    log_entries.append({**base_entry, "decision": "llm_error", "reason": "no JSON in response"})
                    continue
                if isinstance(parsed_json, list):
                    if not parsed_json:
                        log_entries.append({**base_entry, "decision": "no_company", "reason": "LLM returned empty array"})
                        continue
                    parsed_json = parsed_json[0]
                profile = _CompanyProfile.model_validate(parsed_json)

                if not profile.company_name:
                    logger.debug("No company identified in article: %s", title[:60])
                    log_entries.append({**base_entry, "decision": "no_company", "reason": "LLM found no primary company"})
                    continue

                extracted = {
                    "company_name": profile.company_name,
                    "stage": profile.stage,
                    "verticals": profile.verticals,
                    "one_line_description": profile.one_line_description,
                    "founded_date": profile.founded_date,
                    "investors": profile.investors,
                    "url": url,
                    "published_date": published_date,
                    "source": source,
                }
                profiles.append(extracted)
                log_entries.append({**base_entry, "decision": "extracted", "extracted": extracted})
                logger.info("Parsed '%s' from article: %s", profile.company_name, title[:60])

        except openai.APITimeoutError:
            print("      ✗ Timeout after 30s — skipping", flush=True)
            logger.warning("Timeout parsing article '%s' — skipping", title[:60])
            log_entries.append({**base_entry, "decision": "error", "reason": "LLM timeout"})
        except openai.BadRequestError as exc:
            logger.warning("Bad request parsing article '%s': %s", title[:60], exc.message)
            log_entries.append({**base_entry, "decision": "error", "reason": f"bad_request: {exc.message}"})
        except openai.AuthenticationError:
            logger.error("Invalid OpenRouter API key — check OPENROUTER_API_KEY env var")
            log_entries.append({**base_entry, "decision": "error", "reason": "authentication_error"})
            break
        except openai.RateLimitError:
            logger.warning("Rate limited — waiting 20s then retrying '%s'", title[:60])
            time.sleep(20)
            try:
                response = client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=30,
                )
                content = (response.choices[0].message.content or "").strip()
                parsed_json = _extract_json(content)
                if parsed_json and not isinstance(parsed_json, list):
                    profile = _CompanyProfile.model_validate(parsed_json)
                    if profile.company_name:
                        extracted = {
                            "company_name": profile.company_name,
                            "stage": profile.stage,
                            "verticals": profile.verticals,
                            "one_line_description": profile.one_line_description,
                            "founded_date": profile.founded_date,
                            "investors": profile.investors,
                            "url": url,
                            "published_date": published_date,
                            "source": source,
                        }
                        profiles.append(extracted)
                        log_entries.append({**base_entry, "decision": "extracted", "extracted": extracted})
                        logger.info("Retry succeeded for '%s'", title[:60])
                        time.sleep(1)
                        continue
            except Exception:
                pass
            log_entries.append({**base_entry, "decision": "error", "reason": "rate_limited"})
        except openai.APIStatusError as exc:
            logger.warning("API error %d parsing article '%s': %s", exc.status_code, title[:60], exc.message)
            log_entries.append({**base_entry, "decision": "error", "reason": f"api_error_{exc.status_code}"})
        except openai.APIConnectionError as exc:
            logger.warning("Network error parsing article '%s': %s", title[:60], exc)
            log_entries.append({**base_entry, "decision": "error", "reason": f"network_error: {exc}"})
        except ValidationError as exc:
            logger.warning("Schema validation failed for article '%s': %s", title[:60], exc)
            log_entries.append({**base_entry, "decision": "error", "reason": f"parse_error: {exc}"})

        time.sleep(1)

    _save_parse_log(log_entries)
    logger.info("Parsed %d profiles from %d articles", len(profiles), len(raw_results))
    return profiles
