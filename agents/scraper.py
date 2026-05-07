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
import os
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "techcrunch_startups": "https://techcrunch.com/category/startups/feed/",
    "techcrunch_ai": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "techcrunch_crypto": "https://techcrunch.com/tag/cryptocurrency/feed/",
    "hackernews": "https://news.ycombinator.com/rss",
    "decrypt": "https://decrypt.co/feed",
"the_block": "https://www.theblock.co/rss.xml",
}

_RATE_LIMIT_DELAY = 2.0  # seconds between feed fetches

# Canonical source names for each RSS feed key.
_SOURCE_NAMES: dict[str, str] = {
    "hackernews": "hacker_news",
}
# Keys not in this mapping keep their own name (e.g. techcrunch_ai, decrypt, the_block).

# How far back to look. Override with LOOKBACK_HOURS env var (e.g. 48 for a weekend catch-up run).
_LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))


def _parse_date(entry: Any) -> str | None:
    """Return an ISO-8601 date string from a feedparser entry, or None."""
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return raw


def _is_within_lookback(published_date: str | None, cutoff: datetime) -> bool:
    """Return True if the article is newer than cutoff, or if the date is unknown."""
    if not published_date:
        return True  # no date → keep it, can't tell
    try:
        dt = datetime.fromisoformat(published_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= cutoff
    except (ValueError, TypeError):
        return True  # unparseable date → keep it


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


_PH_QUERY = """
{
  posts(first: 20, order: VOTES) {
    edges {
      node {
        name
        tagline
        description
        url
        votesCount
        createdAt
        topics {
          edges {
            node { name }
          }
        }
      }
    }
  }
}
"""


def fetch_producthunt() -> list[dict]:
    """Fetch today's top posts from the Product Hunt GraphQL API."""
    api_key = os.getenv("PRODUCTHUNT_API_KEY", "").strip()
    print(f"      [PH Debug] Token prefix: {api_key[:10] if api_key else 'NOT SET'}", flush=True)
    if not api_key:
        logger.warning("PRODUCTHUNT_API_KEY not set — skipping Product Hunt")
        return []

    print("      Fetching producthunt...", flush=True)
    _t0 = time.time()
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        masked = {**headers, "Authorization": f"Bearer {api_key[:6]}..."}
        print(f"      [PH Debug] Headers: {masked}", flush=True)
        response = httpx.post(
            "https://api.producthunt.com/v2/api/graphql",
            headers=headers,
            json={"query": _PH_QUERY},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

        edges = data.get("data", {}).get("posts", {}).get("edges", [])
        articles = []
        for edge in edges:
            node = edge.get("node", {})
            tagline = node.get("tagline") or ""
            description = node.get("description") or ""
            summary = (tagline + " — " + description).strip() if description else tagline
            articles.append({
                "title": node.get("name", "").strip(),
                "url": node.get("url", "").strip(),
                "summary": summary,
                "published_date": node.get("createdAt"),
                "source": "product_hunt",
            })

        elapsed = int((time.time() - _t0) * 1000)
        print(f"      ✓ producthunt done ({elapsed}ms, {len(articles)} posts)", flush=True)
        logger.info("  → %d posts from producthunt", len(articles))
        return articles

    except httpx.TimeoutException:
        print("      ✗ producthunt timed out after 15s — skipping", flush=True)
        logger.warning("Timeout fetching Product Hunt")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s fetching Product Hunt", exc.response.status_code)
        return []
    except Exception as exc:
        logger.warning("Unexpected error fetching Product Hunt: %s", exc)
        return []


def fetch_crunchbase(cutoff: datetime) -> list[dict]:
    """Fetch recently funded seed/angel orgs from the Crunchbase Basic API."""
    api_key = os.getenv("CRUNCHBASE_API_KEY", "").strip()
    if not api_key:
        logger.warning("CRUNCHBASE_API_KEY not set — skipping Crunchbase")
        return []

    print("      Fetching crunchbase...", flush=True)
    _t0 = time.time()
    cb_body = {
        "field_ids": [
            "identifier",
            "short_description",
            "website_url",
            "founded_on",
            "funding_stage",
            "last_funding_type",
            "last_funding_at",
        ],
        "query": [
            {
                "type": "predicate",
                "field_id": "funding_stage",
                "operator_id": "includes",
                "values": ["seed", "pre_seed", "angel"],
            },
            {
                "type": "predicate",
                "field_id": "last_funding_at",
                "operator_id": "gte",
                "values": [cutoff.strftime("%Y-%m-%d")],
            },
        ],
        "sort": [{"field_id": "last_funding_at", "order": "desc"}],
        "limit": 25,
    }
    try:
        response = httpx.post(
            "https://api.crunchbase.com/api/v4/searches/organizations",
            params={"user_key": api_key},
            json=cb_body,
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

        articles = []
        for entity in data.get("entities", []):
            props = entity.get("properties", {})
            name = (props.get("identifier") or {}).get("value", "").strip()
            articles.append({
                "title": name,
                "url": (props.get("website_url") or "").strip(),
                "summary": (props.get("short_description") or "").strip(),
                "published_date": props.get("last_funding_at"),
                "source": "crunchbase",
            })

        elapsed = int((time.time() - _t0) * 1000)
        print(f"      ✓ crunchbase done ({elapsed}ms, {len(articles)} orgs)", flush=True)
        logger.info("  → %d orgs from crunchbase", len(articles))
        return articles

    except httpx.TimeoutException:
        print("      ✗ crunchbase timed out after 15s — skipping", flush=True)
        logger.warning("Timeout fetching Crunchbase")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s fetching Crunchbase", exc.response.status_code)
        return []
    except Exception as exc:
        logger.warning("Unexpected error fetching Crunchbase: %s", exc)
        return []


def fetch_articles() -> list[dict]:
    """Fetch startup articles from all configured sources within the lookback window.

    Returns a list of dicts with keys: title, url, summary, published_date.
    Articles older than LOOKBACK_HOURS are dropped. Articles with no date are kept.
    Skips sources that fail; logs all errors so the caller always gets a list.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_LOOKBACK_HOURS)
    print(f"      Lookback window: {_LOOKBACK_HOURS}h (cutoff: {cutoff.strftime('%Y-%m-%d %H:%M')} UTC)", flush=True)

    results: list[dict] = []

    headers = {
        "User-Agent": "lightning-agent/1.0 (startup research bot; contact: hello@example.com)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        for source, url in RSS_FEEDS.items():
            try:
                print(f"      Fetching {source}...", flush=True)
                logger.info("Fetching %s from %s", source, url)
                _t0 = time.time()
                response = client.get(url)
                response.raise_for_status()
                feed = feedparser.parse(response.text)

                if feed.bozo and not feed.entries:
                    raise ValueError(f"Malformed feed ({feed.bozo_exception})")

                source_name = _SOURCE_NAMES.get(source, source)
                entries = [_entry_to_dict(e) for e in feed.entries]
                for e in entries:
                    e["source"] = source_name
                entries = [e for e in entries if e["title"] or e["url"]]
                before = len(entries)
                entries = [e for e in entries if _is_within_lookback(e["published_date"], cutoff)]
                elapsed = int((time.time() - _t0) * 1000)
                print(f"      ✓ {source} done ({elapsed}ms, {len(entries)}/{before} within {_LOOKBACK_HOURS}h)", flush=True)
                logger.info("  → %d/%d articles from %s within lookback", len(entries), before, source)
                results.extend(entries)

            except httpx.TimeoutException:
                print(f"      ✗ {source} timed out after 15s — skipping", flush=True)
                logger.warning("Timeout fetching %s (%s)", source, url)
            except httpx.HTTPStatusError as exc:
                logger.warning("HTTP %s fetching %s: %s", exc.response.status_code, source, url)
            except httpx.RequestError as exc:
                logger.warning("Network error fetching %s (%s): %s", source, url, exc)
            except Exception as exc:
                logger.warning("Unexpected error fetching %s: %s", source, exc)

            time.sleep(_RATE_LIMIT_DELAY)

    results.extend(fetch_producthunt())
    results.extend(fetch_crunchbase(cutoff))
    return results
