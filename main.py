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
from agents.evaluator import evaluate_parse, evaluate_research, log_eval
from agents.notion_writer import add_company_to_notion
from agents.parser import parse
from agents.prefilter import is_worth_scoring
from agents.scraper import fetch_articles
from agents.research_agent import research_and_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCORE_THRESHOLD = 40


def main() -> None:
    # ── 1. Scrape ────────────────────────────────────────────────────────────
    print("\n[1/6] Fetching articles from RSS feeds…")
    articles = fetch_articles()
    print(f"      → {len(articles)} articles fetched")

    if not articles:
        print("No articles fetched — exiting.")
        return

    # ── 2. Parse ─────────────────────────────────────────────────────────────
    print("\n[2/6] Parsing company profiles from articles…")
    profiles = parse(articles)
    print(f"      → {len(profiles)} company profiles extracted")

    if not profiles:
        print("No profiles parsed — exiting.")
        return

    # ── 2b. Evaluate parsed profiles ─────────────────────────────────────────
    import time as _time
    print(f"\n[2b] Evaluating {len(profiles)} parsed profiles…")
    article_by_url = {a.get("url"): a for a in articles}
    eval_passed_profiles: list[dict] = []
    parse_eval_flagged = parse_eval_dropped = 0
    for profile in profiles:
        name = profile.get("company_name", "?")
        article = article_by_url.get(profile.get("url"), {})
        eval_result = evaluate_parse(
            profile,
            article.get("title", ""),
            article.get("summary", ""),
        )
        log_eval({"stage": "parse", "company_name": name, "article_url": profile.get("url"), **eval_result})

        confidence = eval_result["confidence"]
        passed = eval_result["passed"]
        issues = eval_result["issues"]

        if not passed and confidence != -1 and confidence < 50:
            print(f"      DROP  {name}  (confidence={confidence}) — {'; '.join(issues)}")
            parse_eval_dropped += 1
        else:
            if issues and confidence != -1 and confidence < 70:
                print(f"      WARN  {name}  (confidence={confidence}) — {'; '.join(issues)}")
                parse_eval_flagged += 1
            else:
                print(f"      OK    {name}  (confidence={confidence if confidence != -1 else 'n/a'})")
            eval_passed_profiles.append(profile)

        _time.sleep(4)

    print(f"      → {len(eval_passed_profiles)} passed, {parse_eval_flagged} flagged, {parse_eval_dropped} dropped")
    profiles = eval_passed_profiles

    if not profiles:
        print("All profiles dropped by parse evaluation — exiting.")
        return

    # ── 3. Deduplicate ───────────────────────────────────────────────────────
    print("\n[3/6] Deduplicating against previously seen companies…")
    seen_before = load_seen()
    new_profiles = deduplicate(profiles)
    print(f"      → {len(new_profiles)} new profiles (skipped {len(profiles) - len(new_profiles)})")

    if not new_profiles:
        print("All companies already seen — nothing to score.")
        return

    # ── 5a. Prefilter ────────────────────────────────────────────────────────
    import time
    print(f"\n[5/6] Pre-filtering {len(new_profiles)} companies…")
    triaged: list[dict] = []
    triage_passed = triage_failed = 0
    for i, profile in enumerate(new_profiles, 1):
        name = profile.get("company_name", "?")
        worth, reason = is_worth_scoring(profile)
        if worth:
            print(f"      [{i}/{len(new_profiles)}] PASS  {name}")
            triaged.append(profile)
            triage_passed += 1
        else:
            print(f"      [{i}/{len(new_profiles)}] SKIP  {name}  ({reason})")
            triage_failed += 1
        time.sleep(1)
    print(f"      → {triage_passed} passed, {triage_failed} filtered out")

    if not triaged:
        print("All companies filtered by triage — nothing to score.")
        return

    # ── 5b. Score ────────────────────────────────────────────────────────────
    print(f"\n      Scoring {len(triaged)} companies…")
    scored: list[dict] = []
    for i, profile in enumerate(triaged, 1):
        name = profile["company_name"]

        print(f"      [{i}/{len(triaged)}] Scoring {name}…", end=" ", flush=True)
        result = research_and_score(profile)

        if "error" in result:
            print(f"ERROR ({result['error']})")
        else:
            flag_marker = " ★" if result["flag"] else ""
            summary = f" — {result['research_summary']}" if result.get("research_summary") else ""
            print(f"score={result['score']}{flag_marker}{summary}")

        scored.append({**profile, **result})

    # ── Debug dump ───────────────────────────────────────────────────────────
    import json as _json, pathlib as _pathlib
    _data_dir = _pathlib.Path(__file__).parent / "data"
    _data_dir.mkdir(exist_ok=True)
    _scored_fields = ("company_name", "score", "flag", "stage", "verticals",
                      "one_line_description", "founders", "total_raised", "investors",
                      "founded_date", "rationale", "research_summary", "data_confidence", "url")
    _scored_log = [{k: c.get(k) for k in _scored_fields} for c in scored]
    (_data_dir / "last_run_scored.json").write_text(
        _json.dumps(_scored_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      → Debug logs: data/last_run_parse_log.json  data/last_run_scored.json")

    # ── 6. Write to Notion ───────────────────────────────────────────────────
    flagged = [c for c in scored if c.get("score", 0) >= _SCORE_THRESHOLD]
    print(f"\n[6/6] Writing {len(flagged)} flagged companies (score ≥ {_SCORE_THRESHOLD}) to Notion…")

    created = skipped = failed = 0
    for company in flagged:
        name = company.get("corrected_name") or company["company_name"]
        notion_payload = {**company, "company_name": name}
        print(f"      Writing {name}…", end=" ", flush=True)
        result = add_company_to_notion(notion_payload)
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
    print(f"   Articles fetched     : {len(articles)}")
    print(f"   → LLM extracted      : {len(profiles)}  (no company in {len(articles) - len(profiles)} articles)")
    print(f"   → Deduped (new)      : {len(new_profiles)}  (skipped {len(profiles) - len(new_profiles)} already seen)")
    print(f"   → Prefilter passed   : {triage_passed}  (dropped {triage_failed})")
    print(f"   → Scored             : {len(scored)}")
    print(f"   → Flagged (≥{_SCORE_THRESHOLD})     : {len(flagged)}")
    print(f"   → Notion created     : {created}  (failed {failed})")
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
