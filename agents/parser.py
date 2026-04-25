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
import time
from typing import Literal

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

_client: openai.OpenAI | None = None

_MODEL = "meta-llama/llama-3.3-70b-instruct"

_SYSTEM_PROMPT = """\
You are an expert at reading startup and tech news articles and identifying the primary company \
being written about.

Your task is to extract structured information about the main startup or company featured in an \
article. Focus on the company that is the subject of the news — not investors, acquirers, or \
incidentally mentioned firms.

Rules:
- If the article does not clearly feature a specific startup or company (e.g. it is a roundup, \
  opinion piece, or macro analysis with no single subject), set company_name to null.
- stage must be one of: pre-seed, seed, series-a, unknown. Use "unknown" if not mentioned.
- verticals must be a list containing only values from: AI, blockchain, web3, digital-assets, \
  infrastructure, other. Include every applicable vertical; use ["other"] if none fit.
- one_line_description should be a single sentence (max 150 characters) describing what the \
  company does. Set to null if you cannot determine it from the article.\
"""


class _CompanyProfile(BaseModel):
    company_name: str | None = Field(
        description="Name of the primary startup featured in the article, or null if none"
    )
    stage: Literal["pre-seed", "seed", "series-a", "unknown"] = Field(
        description="Funding stage of the company"
    )
    verticals: list[Literal["AI", "blockchain", "web3", "digital-assets", "infrastructure", "other"]] = Field(
        description="Applicable verticals for the company"
    )
    one_line_description: str | None = Field(
        description="One sentence describing what the company does, or null if unclear"
    )


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
    """
    client = _get_client()
    profiles: list[dict] = []

    for article in raw_results:
        title = article.get("title", "")
        summary = article.get("summary", "")
        url = article.get("url", "")
        published_date = article.get("published_date")

        article_text = f"Title: {title}\n\nSummary: {summary}".strip()
        if not article_text or article_text == "Title: \n\nSummary:":
            continue

        user_message = (
            f"Extract the primary startup from the following article.\n\n{article_text}"
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
            try:
                if not response or not response.choices or len(response.choices) == 0:
                    logger.warning("No choices in response for article '%s' — skipping", title[:60])
                    continue
                content = response.choices[0].message.content
                if not content:
                    logger.warning("Empty response content for article '%s' — skipping", title[:60])
                    continue
                profile = _CompanyProfile.model_validate(json.loads(content))
            except (AttributeError, IndexError) as exc:
                logger.warning("Unexpected response shape for article '%s': %s", title[:60], exc)
                continue

            if not profile.company_name:
                logger.debug("No company identified in article: %s", title[:60])
                continue

            profiles.append({
                "company_name": profile.company_name,
                "stage": profile.stage,
                "verticals": profile.verticals,
                "one_line_description": profile.one_line_description,
                "url": url,
                "published_date": published_date,
            })
            logger.info("Parsed '%s' from article: %s", profile.company_name, title[:60])

        except openai.BadRequestError as exc:
            logger.warning("Bad request parsing article '%s': %s", title[:60], exc.message)
        except openai.AuthenticationError:
            logger.error("Invalid OpenRouter API key — check OPENROUTER_API_KEY env var")
            break  # no point continuing if auth is broken
        except openai.RateLimitError:
            logger.warning("Rate limited parsing article '%s'", title[:60])
        except openai.APIStatusError as exc:
            logger.warning("API error %d parsing article '%s': %s", exc.status_code, title[:60], exc.message)
        except openai.APIConnectionError as exc:
            logger.warning("Network error parsing article '%s': %s", title[:60], exc)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Failed to parse response for article '%s': %s", title[:60], exc)

        time.sleep(6)

    logger.info("Parsed %d profiles from %d articles", len(profiles), len(raw_results))
    return profiles
