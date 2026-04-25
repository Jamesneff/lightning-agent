"""
main.py — Entry point for the lightning-agent pipeline.

Orchestrates the full startup discovery workflow:
  1. Scraper   — fetches raw startup data from configured sources
  2. Parser    — extracts structured fields from raw content
  3. Deduplicator — filters out startups already seen or duplicated
  4. Scorer    — ranks remaining startups by VC-fit criteria
  5. NotionWriter — persists top-ranked startups to a Notion database

Usage:
    python main.py

Environment variables are loaded from .env (see .env.example).
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from agents.deduplicator import deduplicate, load_seen, save_seen
from agents.notion_writer import add_company_to_notion
from agents.parser import parse
from agents.scraper import fetch_articles
from agents.scorer import score_company

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCORE_THRESHOLD = 60  # minimum score to write to Notion (flag=True is also required)


def main() -> None:
    # ── 1. Scrape ────────────────────────────────────────────────────────────
    print("\n[1/5] Fetching articles from RSS feeds…")
    articles = fetch_articles()
    print(f"      → {len(articles)} articles fetched")

    if not articles:
        print("No articles fetched — exiting.")
        return

    # ── 2. Parse ─────────────────────────────────────────────────────────────
    print("\n[2/5] Parsing company profiles from articles…")
    profiles = parse(articles)
    print(f"      → {len(profiles)} company profiles extracted")

    if not profiles:
        print("No profiles parsed — exiting.")
        return

    # ── 3. Deduplicate ───────────────────────────────────────────────────────
    print("\n[3/5] Deduplicating against previously seen companies…")
    seen_before = load_seen()
    new_profiles = deduplicate(profiles)
    print(f"      → {len(new_profiles)} new profiles (skipped {len(profiles) - len(new_profiles)})")

    if not new_profiles:
        print("All companies already seen — nothing to score.")
        return

    # ── 4. Score ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Scoring {len(new_profiles)} companies with Claude…")
    scored: list[dict] = []
    for i, profile in enumerate(new_profiles, 1):
        name = profile["company_name"]
        context_parts = []
        if profile.get("one_line_description"):
            context_parts.append(profile["one_line_description"])
        if profile.get("verticals"):
            context_parts.append("Verticals: " + ", ".join(profile["verticals"]))
        if profile.get("stage") and profile["stage"] != "unknown":
            context_parts.append(f"Stage: {profile['stage']}")
        if profile.get("url"):
            context_parts.append(f"Source: {profile['url']}")
        context = "\n".join(context_parts) or name

        print(f"      [{i}/{len(new_profiles)}] Scoring {name}…", end=" ", flush=True)
        result = score_company(name, context)

        if "error" in result:
            print(f"ERROR ({result['error']})")
        else:
            flag_marker = " ★" if result["flag"] else ""
            print(f"score={result['score']}{flag_marker}")

        scored.append({**profile, **result})

    # ── 5. Write to Notion ───────────────────────────────────────────────────
    flagged = [c for c in scored if c.get("flag") and c.get("score", 0) >= _SCORE_THRESHOLD]
    print(f"\n[5/5] Writing {len(flagged)} flagged companies (score ≥ {_SCORE_THRESHOLD}) to Notion…")

    created = skipped = failed = 0
    for company in flagged:
        name = company["company_name"]
        print(f"      Writing {name}…", end=" ", flush=True)
        result = add_company_to_notion(company)
        if result.get("ok"):
            print(f"ok (page {result['page_id'][:8]}…)")
            created += 1
        else:
            print(f"FAILED ({result.get('error', 'unknown')})")
            failed += 1

    # ── Save seen set ────────────────────────────────────────────────────────
    newly_seen = seen_before | {c["company_name"].lower() for c in new_profiles if c.get("company_name")}
    save_seen(newly_seen)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Run complete ──────────────────────────────────────────────────")
    print(f"   Articles fetched : {len(articles)}")
    print(f"   Profiles parsed  : {len(profiles)}")
    print(f"   New (deduped)    : {len(new_profiles)}")
    print(f"   Scored           : {len(scored)}")
    print(f"   Flagged          : {len(flagged)}")
    print(f"   Notion created   : {created}")
    print(f"   Notion failed    : {failed}")
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
