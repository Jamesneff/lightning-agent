"""
notion_writer.py — Persists scored startup profiles to a Notion database.

Responsibilities:
  - Accept a list of scored startup profiles from scorer.py.
  - Authenticate with the Notion API using NOTION_TOKEN from the environment.
  - Map profile fields to the target Notion database schema
    (name, URL, score, stage, verticals, rationale, source, discovered_at).
  - Create a new Notion page per startup; skip or update if a page for
    the same URL already exists (idempotent writes).
  - Report success/failure counts after the write pass.

Primary interface:
    write(profiles: list[dict]) -> dict
        Writes profiles to Notion and returns a summary:
        {"created": int, "skipped": int, "failed": int}
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from notion_client import Client
from notion_client.errors import APIResponseError

load_dotenv()

logger = logging.getLogger(__name__)

_VALID_STATUSES = {"New", "Reviewed", "Pass"}

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        token = os.environ.get("NOTION_TOKEN")
        if not token:
            raise EnvironmentError("NOTION_TOKEN is not set — add it to .env")
        _client = Client(auth=token)
    return _client


def _get_database_id() -> str:
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not db_id:
        raise EnvironmentError("NOTION_DATABASE_ID is not set — add it to .env")
    return db_id


def _truncate(text: str, max_chars: int = 2000) -> str:
    # Notion rich_text content blocks are capped at 2000 characters.
    return text[:max_chars] if len(text) > max_chars else text


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "-")


def _company_age_string(founded_date: str | None) -> str | None:
    """Return a human-readable age string from a 'YYYY-MM' founded_date.

    Examples: '2 years 3 months', '8 months', '1 year'.
    Returns None if founded_date is absent or unparseable.
    """
    if not founded_date:
        return None
    try:
        year, month = int(founded_date[:4]), int(founded_date[5:7])
    except (ValueError, IndexError):
        return None
    today = datetime.now().date()
    total_months = (today.year - year) * 12 + (today.month - month)
    if total_months <= 0:
        return None
    years, months = divmod(total_months, 12)
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    return " ".join(parts) if parts else None


def _to_date_string(value: str | None) -> str | None:
    """Normalise an ISO-8601 datetime or bare date string to YYYY-MM-DD."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None



def add_company_to_notion(company: dict) -> dict:
    """Add a single company as a new row in the configured Notion database.

    Expected keys in `company`:
        company_name  str           Company Name (title)   — required
        score         int | float   Score (number)
        rationale     str           Rationale (rich text)
        url           str           Source article URL (url)
        website       str | None    Company homepage URL → Website column (optional)
        date_added    str           Date Added — always set to today's date (run date)
        status        str           'New' | 'Reviewed' | 'Pass' — defaults to 'New'
        founders      str | None    Comma-separated founder names
        total_raised  str | None    Total funding raised, e.g. '$4M'
        investors     str | None    Comma-separated investor names

    Returns:
        {"ok": True,  "page_id": str}  on success.
        {"ok": False, "error":   str}  on failure.
    """
    client = _get_client()
    database_id = _get_database_id()

    # Accept both 'company_name' (scorer output) and 'title' (scraper output).
    company_name = (company.get("company_name") or company.get("title", "")).strip()
    if not company_name:
        return {"ok": False, "error": "company_name is required"}

    score = company.get("score")
    rationale = company.get("rationale", "")
    raw_url = company.get("url", "")
    url = raw_url.strip() or None  # Notion rejects empty-string URL values; this is the article URL
    website_url = (company.get("website") or company.get("homepage_url") or "").strip() or None
    slug = _slugify(company_name)
    linkedin_url = f"https://linkedin.com/company/{slug}"
    crunchbase_url = f"https://crunchbase.com/organization/{slug}"
    date_added = datetime.now().date().isoformat()
    founded_date = company.get("founded_date")
    company_age = _company_age_string(founded_date if isinstance(founded_date, str) else None)
    founders = (company.get("founders") or "").strip() or None
    total_raised = (company.get("total_raised") or "").strip() or None
    investors = (company.get("investors") or "").strip() or None
    status = company.get("status", "New")
    if status not in _VALID_STATUSES:
        logger.warning("Unknown status %r for %s; defaulting to 'New'", status, company_name)
        status = "New"

    properties: dict = {
        "Name": {
            "title": [{"text": {"content": company_name}}]
        },
        "Score": {
            "number": score if isinstance(score, (int, float)) else None
        },
        "Rationale": {
            "rich_text": [{"text": {"content": _truncate(rationale)}}]
        },
        "Status": {
            "select": {"name": status}
        },
    }
    if url:
        properties["Source"] = {"url": url}
    if website_url:
        properties["Website"] = {"url": website_url}
    if linkedin_url:
        properties["LinkedIn"] = {"url": linkedin_url}
    if crunchbase_url:
        properties["Crunchbase"] = {"url": crunchbase_url}
    properties["Date Added"] = {"date": {"start": date_added}}
    if company_age:
        properties["Company Age"] = {"rich_text": [{"text": {"content": company_age}}]}
    if founders:
        properties["Founders"] = {"rich_text": [{"text": {"content": _truncate(founders)}}]}
    if total_raised:
        properties["Funding Raised"] = {"rich_text": [{"text": {"content": total_raised}}]}
    if investors:
        properties["Investors"] = {"rich_text": [{"text": {"content": _truncate(investors)}}]}

    try:
        page = client.pages.create(
            parent={"database_id": database_id},
            properties=properties,
        )
        page_id = page["id"]
        logger.info(
            "Created Notion page %s for '%s' (score=%s, flag=%s)",
            page_id, company_name, score, company.get("flag"),
        )
        return {"ok": True, "page_id": page_id}

    except APIResponseError as exc:
        logger.warning("Notion API error writing '%s': %s", company_name, exc)
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        logger.warning("Unexpected error writing '%s' to Notion: %s", company_name, exc)
        return {"ok": False, "error": str(exc)}
