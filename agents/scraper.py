"""
scraper.py — Discovers and fetches raw startup data from external sources.

Responsibilities:
  - Query configured sources (e.g. Product Hunt, AngelList, Crunchbase,
    Twitter/X search, RSS feeds, newsletters) for recent AI/blockchain startups.
  - Handle pagination, rate-limiting, and retries for each source.
  - Return a list of raw result objects (dicts) with at minimum a URL and
    raw text/HTML blob for downstream parsing.

Primary interface:
    scrape() -> list[dict]
        Runs all configured scrapers and returns combined raw results.
"""

import logging
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "techcrunch_startups": "https://techcrunch.com/category/startups/feed/",
    "the_block": "https://www.theblock.co/rss.xml",
}

_RATE_LIMIT_DELAY = 2.0  # seconds between feed fetches


def _parse_date(entry: Any) -> str | None:
    """Return an ISO-8601 date string from a feedparser entry, or None."""
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return raw


def _entry_to_dict(entry: Any) -> dict:
    summary = getattr(entry, "summary", None) or ""
    # feedparser may return HTML in summary; strip tags for a plain-text preview
    if "<" in summary:
        summary = re.sub(r"<[^>]+>", "", summary).strip()

    return {
        "title": getattr(entry, "title", "").strip(),
        "url": getattr(entry, "link", "").strip(),
        "summary": summary,
        "published_date": _parse_date(entry),
    }


def fetch_articles() -> list[dict]:
    """Fetch startup articles from TechCrunch (startups) and The Block RSS feeds.

    Returns a list of dicts with keys: title, url, summary, published_date.
    Skips feeds that fail; logs all errors so the caller always gets a list.
    """
    results: list[dict] = []

    headers = {
        "User-Agent": "lightning-agent/1.0 (startup research bot; contact: hello@example.com)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        for source, url in RSS_FEEDS.items():
            try:
                logger.info("Fetching %s from %s", source, url)
                response = client.get(url)
                response.raise_for_status()
                feed = feedparser.parse(response.text)

                if feed.bozo and not feed.entries:
                    # bozo flag means malformed XML; skip only if no entries were recovered
                    raise ValueError(f"Malformed feed ({feed.bozo_exception})")

                entries = [_entry_to_dict(e) for e in feed.entries]
                # Drop entries missing both title and url
                entries = [e for e in entries if e["title"] or e["url"]]
                logger.info("  → %d articles from %s", len(entries), source)
                results.extend(entries)

            except httpx.HTTPStatusError as exc:
                logger.warning("HTTP %s fetching %s: %s", exc.response.status_code, source, url)
            except httpx.RequestError as exc:
                logger.warning("Network error fetching %s (%s): %s", source, url, exc)
            except Exception as exc:
                logger.warning("Unexpected error fetching %s: %s", source, exc)

            # Rate-limit: pause between requests regardless of success/failure
            time.sleep(_RATE_LIMIT_DELAY)

    return results
