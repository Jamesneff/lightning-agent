# Lightning Agent — CLAUDE.md

## What this project does

Lightning Agent is an automated deal-flow pipeline for Lightning Capital, a seed-to-Series-A venture fund focused on AI, blockchain, and digital asset startups. It runs on a schedule, scrapes startup news from multiple sources, extracts company profiles, deduplicates, triages, researches each company with a LangGraph agent, scores them against the fund thesis, and writes qualifying companies to a Notion database.

**End users:** Non-technical Lightning Capital analysts who use the Flask dashboard at `localhost:5000`. They click "Run Agent", watch the live output, and review scored companies. They do not touch the CLI or read logs. Frontend changes must be polished and self-explanatory — do not add technical jargon, raw JSON, or internal error codes to the UI.

## Cost sensitivity

This project runs on paid APIs. Every unnecessary LLM call or search query has a real cost. When making changes:
- Do not add extra LLM calls without a clear reason
- Do not switch to more expensive models (GPT-4o, Claude Opus) — the current `meta-llama/llama-3.3-70b-instruct` via OpenRouter is intentional
- Do not propose paid third-party APIs or services unless explicitly asked
- Prefer prompt improvements over adding more tool calls
- The parser `time.sleep(1)` is a pacing delay between LLM calls. A `RateLimitError` triggers a 20-second backoff and one retry before skipping — do not reduce the retry sleep

## Architecture

```
fetch_articles()         scraper.py        RSS feeds + Product Hunt + Crunchbase
      ↓
parse()                  parser.py         LLM extracts structured company profiles
      ↓
deduplicate()            deduplicator.py   Filters companies seen in prior runs
      ↓
is_worth_scoring()       prefilter.py      Fast LLM triage (YES/NO) + hard blacklist
      ↓
research_and_score()     research_agent.py LangGraph agent with 5 search tools
      ↓
add_company_to_notion()  notion_writer.py  Writes qualifying companies to Notion DB
```

There are two entry points for the same pipeline:
- `main.py` — CLI runner, used for local development and testing
- `app.py` — Flask + SocketIO web app with a live dashboard at `localhost:5000`

**Two-entry-point rule:** These files duplicate the pipeline logic. If you are making a pipeline change (scoring threshold, stage order, filtering logic, new agent step), you must update both files. If the change is non-trivial, extract it to `agents/pipeline.py` first and have both callers use it — do not add more duplication.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Flask web app, SocketIO live dashboard, pipeline thread runner |
| `main.py` | CLI entry point for the same pipeline |
| `agents/scraper.py` | RSS feeds (TechCrunch, HackerNews, Decrypt, The Block) + Product Hunt GraphQL + Crunchbase API |
| `agents/parser.py` | LLM call per article to extract company name, stage, verticals, investors, founding date |
| `agents/prefilter.py` | Fast LLM YES/NO triage + hard `_BLACKLIST` for known large companies |
| `agents/research_agent.py` | LangGraph agent that orchestrates 5 tools to research and score each company |
| `agents/tools.py` | 5 Serper-based search tools: `get_funding_info`, `get_founder_info`, `get_company_overview`, `get_recent_news`, `get_sec_filing` |
| `agents/deduplicator.py` | Deduplicates against `data/seen_companies.json` and within the current batch |
| `agents/notion_writer.py` | Maps scored profiles to Notion database properties and creates pages |
| `data/runs.db` | SQLite database storing run history and per-run company results |
| `data/seen_companies.json` | Flat JSON list of lowercased company names seen in prior runs |

## Environment variables

Required (pipeline will fail without these):
```
OPENROUTER_API_KEY     LLM calls (parser, prefilter, research agent) via OpenRouter
NOTION_TOKEN           Notion integration token
NOTION_DATABASE_ID     Target Notion database UUID
SERPER_API_KEY         Google search via Serper (used by all 5 research tools)
```

Optional:
```
PRODUCTHUNT_API_KEY    Product Hunt GraphQL API (skipped if not set)
CRUNCHBASE_API_KEY     Crunchbase Basic API (skipped if not set)
LOOKBACK_HOURS         How far back to fetch articles (default: 24)
FLASK_SECRET           Flask session secret (default: dev key)
```

## Running the project

```bash
# Web dashboard (recommended)
python app.py
# → opens at http://localhost:5000

# CLI
python main.py
```

Always restart the Flask server after changing Python files — it runs with `use_reloader=False`.

## Models

All LLM calls go through OpenRouter:
- Parser: `meta-llama/llama-3.3-70b-instruct`
- Prefilter: `meta-llama/llama-3.3-70b-instruct`
- Research agent: `meta-llama/llama-3.3-70b-instruct`

## Pipeline tuning constants

| Constant | Location | Current value | Purpose |
|----------|----------|---------------|---------|
| `_SCORE_THRESHOLD` | `app.py`, `main.py` | `40` | Minimum score to write to Notion |
| `_LOOKBACK_HOURS` | `scraper.py` | `24` (env override) | How far back to fetch articles |
| `_MAX_ITERATIONS` | `research_agent.py` | `20` | Max LLM iterations per research agent run |
| `time.sleep(1)` | `parser.py` | 1 second | Pacing delay between parser LLM calls |
| `_RATE_LIMIT_DELAY` | `scraper.py` | 2 seconds | Delay between RSS feed fetches |
| `_BLACKLIST` | `prefilter.py` | see file | Hard-coded known large companies to skip |

## Scoring rubric (research_agent.py)

```
0        — hard disqualifier triggered (verified >$50M raised, >$500M valuation, public co, exit)
10–30    — off-thesis, wrong stage, or established company
40–60    — on-thesis but key data missing (funding unknown, founders not found)
70–85    — on-thesis, early stage, credible signals found
85–100   — exceptional fit: strong team, right stage, clear traction, ideal investors
```

Hard disqualifiers only apply when there is a **verified source** for the number. Missing funding data is NOT a disqualifier — score on product and team signals instead.

## Never do — recurring bug patterns

These mistakes have already happened. Do not repeat them.

**HTTP client:** Use `httpx` throughout. Do not introduce `requests`. The project is already on `httpx` and mixing clients causes subtle issues with timeout handling and async compatibility.

**JSON from LLM responses:** Never call `json.loads()` directly on raw LLM output. Models frequently wrap JSON in markdown fences, add explanation text, or return malformed objects. Always use the existing `_extract_json()` helper in `parser.py` (regex fallback after direct parse) or the `<json>` tag extraction pattern in `research_agent.py`. A bare `json.loads(response)` will crash in production.

**SQLite and lists:** SQLite columns are typed. Do not store Python lists directly in a TEXT column — they serialize as `"['a', 'b']"` and cannot be queried or deserialized cleanly. Either join to a comma-separated string, use a JSON column with `json.dumps`, or add a child table.

**Notion property names:** The Notion API rejects writes if a property name doesn't exactly match the database schema (case-sensitive). The field is `"Funding Raised"`, not `"Total Raised"`. Before adding a new Notion property write, verify the exact column name in the schema table at the bottom of this file.

**Flask server must be restarted:** The server runs with `use_reloader=False`. Code changes are not picked up until the process is restarted. If a bug appears to persist after a fix, the server is probably still running old code.

**Stage numbers in app.py must match run.html:** `_emit_stage(n, ...)` in `app.py` maps to `id="stage-n"` in `run.html`. The pipeline currently has 5 stages (1–5). If you add or remove a stage, update both the Python emit calls and the HTML stage list.

## Known issues / areas for improvement

1. **Duplicate pipeline** — `main.py` and `app.py` both implement the full pipeline. Extract to a shared `agents/pipeline.py` module.

2. **`seen_companies.json` is fragile** — flat file with no TTL or fuzzy matching. Name variations (`"Clera"` vs `"Clera AI"`) will cause the same company to be re-scored across runs. Should migrate to a `seen_companies` table in the existing `data/runs.db` SQLite database.

3. **Parser sleep is slow** — `time.sleep(8)` between every article means 93 articles takes 12+ minutes in the parse stage. 1–2 seconds is likely sufficient.

4. **Fake LinkedIn/Crunchbase URLs** — `notion_writer.py` constructs `linkedin.com/company/{slug}` and `crunchbase.com/organization/{slug}` from slugified names. These are almost always wrong. Remove them until the research agent can surface the actual URLs.

5. **No startup env var validation** — missing API keys surface as errors deep in the pipeline after minutes of work. Add validation at startup.

6. **No hard timeout on research agent** — `_MAX_ITERATIONS` caps loop count but not wall-clock time. A slow run could block for 3+ minutes per company.

7. **Product Hunt signal quality** — PH posts are product listings, not necessarily funded companies. They frequently return `not_found` research results and waste Serper API calls.

## Notion database schema

The Notion database expects these property names (casing matters):

| Property | Type | Source field |
|----------|------|-------------|
| Name | title | `company_name` |
| Score | number | `score` |
| Rationale | rich_text | `rationale` |
| Status | select | `"New"` (default) |
| Source | url | `url` (article URL) |
| Website | url | `website` / `homepage_url` |
| LinkedIn | url | constructed from slug |
| Crunchbase | url | constructed from slug |
| Date Added | date | today |
| Company Age | rich_text | derived from `founded_date` |
| Founders | rich_text | `founders` |
| Funding Raised | rich_text | `total_raised` |
| Investors | rich_text | `investors` |
