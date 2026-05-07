import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "lightning-capital-dev-secret")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

DB_PATH = Path("data/runs.db")
_is_running = False

current_run: dict = {
    "active": False,
    "run_id": None,
    "stage": 0,
    "stage_name": "",
    "stages": {},        # str(n) -> {name, status}
    "log_buffer": [],    # [{text, level}, ...]  capped at 500
    "companies": [],     # company_scored payloads
    "started_at": None,
    "stats": {},
}


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at       TEXT NOT NULL,
                completed_at     TEXT,
                duration_seconds REAL,
                articles_fetched INTEGER DEFAULT 0,
                profiles_parsed  INTEGER DEFAULT 0,
                new_companies    INTEGER DEFAULT 0,
                flagged_count    INTEGER DEFAULT 0,
                notion_written   INTEGER DEFAULT 0,
                status           TEXT DEFAULT 'running'
            );
            CREATE TABLE IF NOT EXISTS run_companies (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           INTEGER NOT NULL,
                company_name     TEXT,
                score            INTEGER DEFAULT 0,
                flag             INTEGER DEFAULT 0,
                rationale        TEXT,
                research_summary TEXT,
                notion_written   INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
        """)
        # Mark any runs still stuck as 'running' from a previous crashed/killed process
        conn.execute("UPDATE runs SET status='interrupted' WHERE status='running'")


def _save_run(run_id: int, stats: dict, duration: float, status: str, scored: list | None = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """UPDATE runs SET completed_at=?, duration_seconds=?, status=?,
               articles_fetched=?, profiles_parsed=?, new_companies=?,
               flagged_count=?, notion_written=? WHERE id=?""",
            (datetime.utcnow().isoformat(), round(duration, 1), status,
             stats["articles_fetched"], stats["profiles_parsed"], stats["new_companies"],
             stats["flagged_count"], stats["notion_written"], run_id),
        )
        if scored:
            def _to_str(v) -> str:
                if isinstance(v, list):
                    return "\n".join(str(x) for x in v)
                return str(v) if v is not None else ""

            conn.executemany(
                """INSERT INTO run_companies
                   (run_id, company_name, score, flag, rationale, research_summary, notion_written)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(run_id, s.get("company_name"), s.get("score", 0),
                  int(bool(s.get("flag"))), _to_str(s.get("rationale")),
                  _to_str(s.get("research_summary")), int(bool(s.get("notion_written"))))
                 for s in scored],
            )


# ── Run-state helpers ─────────────────────────────────────────────────────────

def _buffer_log(text: str, level: str):
    current_run["log_buffer"].append({"text": text, "level": level})
    if len(current_run["log_buffer"]) > 500:
        current_run["log_buffer"] = current_run["log_buffer"][-500:]


def _emit_stage(n: int, name: str, status: str):
    current_run["stage"] = n
    current_run["stage_name"] = name
    current_run["stages"][str(n)] = {"name": name, "status": status}
    socketio.emit("stage_update", {"n": n, "name": name, "status": status})


# ── SocketIO helpers ──────────────────────────────────────────────────────────

class _SocketIOLogHandler(logging.Handler):
    def emit(self, record):
        try:
            text = self.format(record)
            level = record.levelname.lower()
            _buffer_log(text, level)
            socketio.emit("log_line", {"text": text, "level": level})
        except Exception:
            pass


class _TeeStream:
    """Passes writes to the original stdout AND the SocketIO log stream."""
    def __init__(self, original):
        self._orig = original

    def write(self, text):
        self._orig.write(text)
        stripped = text.strip()
        if stripped:
            try:
                _buffer_log(stripped, "print")
                socketio.emit("log_line", {"text": stripped, "level": "print"})
            except Exception:
                pass

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(run_id: int):
    global _is_running

    current_run.update({
        "active": True,
        "run_id": run_id,
        "stage": 0,
        "stage_name": "",
        "stages": {},
        "log_buffer": [],
        "companies": [],
        "started_at": datetime.utcnow().isoformat(),
        "stats": {},
    })

    log_handler = _SocketIOLogHandler()
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    orig_stdout = sys.stdout
    sys.stdout = _TeeStream(orig_stdout)

    start_time = time.time()
    stats = dict(articles_fetched=0, profiles_parsed=0, new_companies=0,
                 flagged_count=0, notion_written=0)
    scored: list[dict] = []

    try:
        from agents.deduplicator import deduplicate, load_seen, save_seen
        from agents.notion_writer import add_company_to_notion
        from agents.parser import parse
        from agents.prefilter import is_worth_scoring
        from agents.research_agent import research_and_score
        from agents.scraper import fetch_articles

        # ── 1. Scrape ────────────────────────────────────────────────────────
        _emit_stage(1, "Fetching articles", "running")
        articles = fetch_articles()
        stats["articles_fetched"] = len(articles)
        _emit_stage(1, "Fetching articles", "complete")

        if not articles:
            _save_run(run_id, stats, time.time() - start_time, "complete")
            socketio.emit("run_failed", {"error": "No articles fetched from RSS feeds."})
            return

        # ── 2. Parse ─────────────────────────────────────────────────────────
        _emit_stage(2, "Parsing company profiles", "running")
        profiles = parse(articles)
        stats["profiles_parsed"] = len(profiles)
        _emit_stage(2, "Parsing company profiles", "complete")

        if not profiles:
            _save_run(run_id, stats, time.time() - start_time, "complete")
            socketio.emit("run_failed", {"error": "No company profiles could be parsed."})
            return

        # ── 3. Deduplicate ───────────────────────────────────────────────────
        _emit_stage(3, "Deduplicating", "running")
        seen_before = load_seen()
        new_profiles = deduplicate(profiles)
        stats["new_companies"] = len(new_profiles)
        _emit_stage(3, "Deduplicating", "complete")

        if not new_profiles:
            _save_run(run_id, stats, time.time() - start_time, "complete")
            socketio.emit("run_complete", {**stats, "message": "All companies already seen."})
            return

        # ── 4. Prefilter + Research & Score ──────────────────────────────────
        _emit_stage(4, f"Triaging {len(new_profiles)} companies", "running")
        triaged = []
        for profile in new_profiles:
            worth, reason = is_worth_scoring(profile)
            name = profile.get("company_name", "?")
            if worth:
                triaged.append(profile)
                logging.getLogger(__name__).info("Triage PASS: %s", name)
            else:
                logging.getLogger(__name__).info("Triage SKIP: %s — %s", name, reason)
            time.sleep(1)

        stats["new_companies"] = len(triaged)
        _emit_stage(4, f"Researching & scoring {len(triaged)} companies", "running")

        for profile in triaged:
            result = research_and_score(profile)
            company = {**profile, **result}
            scored.append(company)
            event = {
                "name": company.get("company_name", "?"),
                "score": company.get("score", 0),
                "flag": bool(company.get("flag", False)),
                "summary": company.get("research_summary", ""),
                "error": company.get("error", ""),
            }
            current_run["companies"].append(event)
            socketio.emit("company_scored", event)
        _emit_stage(4, f"Researching & scoring {len(triaged)} companies", "complete")

        # Write debug scored log so /last-run "Scored Companies" tab is populated
        import json as _json
        _scored_fields = ("company_name", "score", "flag", "stage", "verticals",
                          "one_line_description", "founders", "total_raised", "investors",
                          "founded_date", "rationale", "research_summary", "data_confidence", "url")
        try:
            Path("data").mkdir(exist_ok=True)
            Path("data/last_run_scored.json").write_text(
                _json.dumps([{k: c.get(k) for k in _scored_fields} for c in scored],
                            indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

        # ── 5. Write to Notion ───────────────────────────────────────────────
        _SCORE_THRESHOLD = 40
        flagged = [c for c in scored if c.get("score", 0) >= _SCORE_THRESHOLD]
        stats["flagged_count"] = len(flagged)
        _emit_stage(5, f"Writing {len(flagged)} companies to Notion", "running")

        for company in flagged:
            result = add_company_to_notion(company)
            if result.get("ok"):
                stats["notion_written"] += 1
                company["notion_written"] = True
        _emit_stage(5, f"Writing {len(flagged)} companies to Notion", "complete")

        newly_seen = seen_before | {
            c["company_name"].lower() for c in new_profiles if c.get("company_name")
        }
        save_seen(newly_seen)

        current_run["stats"] = stats
        _save_run(run_id, stats, time.time() - start_time, "complete", scored)
        socketio.emit("run_complete", stats)

    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline error: %s", exc)
        _save_run(run_id, stats, time.time() - start_time, "failed", scored or None)
        socketio.emit("run_failed", {"error": str(exc)})

    finally:
        root_logger.removeHandler(log_handler)
        sys.stdout = orig_stdout
        current_run["active"] = False
        _is_running = False


# ── Context processor ─────────────────────────────────────────────────────────

@app.context_processor
def inject_run_state():
    return {"run_active": current_run["active"], "current_run": current_run}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 10").fetchall()
    last_run = runs[0] if runs else None
    notion_url = "https://notion.so/{}".format(
        os.getenv("NOTION_DATABASE_ID", "").replace("-", "")
    )
    return render_template("dashboard.html", runs=runs, last_run=last_run, notion_url=notion_url)


@app.route("/run")
def run_page():
    return render_template("run.html")


@app.route("/run/start", methods=["POST"])
def start_run():
    global _is_running
    if _is_running:
        return jsonify({"error": "A pipeline run is already in progress."}), 409
    _is_running = True

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (started_at, status) VALUES (?, 'running')",
            (datetime.utcnow().isoformat(),),
        )
        run_id = cursor.lastrowid

    thread = threading.Thread(target=_run_pipeline, args=(run_id,))
    thread.daemon = True
    thread.start()
    return jsonify({"run_id": run_id})


@app.route("/run/status")
def run_status():
    return jsonify(current_run)


@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    return render_template("history.html", runs=runs)


@app.route("/last-run")
def last_run_log():
    from pathlib import Path
    import json as _json

    parse_log, scored_log = [], []
    parse_path = Path("data/last_run_parse_log.json")
    scored_path = Path("data/last_run_scored.json")
    if parse_path.exists():
        try:
            parse_log = _json.loads(parse_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if scored_path.exists():
        try:
            scored_log = _json.loads(scored_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return render_template("last_run_log.html", parse_log=parse_log, scored_log=scored_log)


@app.route("/history/<int:run_id>")
def run_detail(run_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        companies = conn.execute(
            "SELECT * FROM run_companies WHERE run_id=? ORDER BY score DESC", (run_id,)
        ).fetchall()
    if not run:
        return "Run not found", 404
    return render_template("run_detail.html", run=run, companies=companies)


@socketio.on("connect")
def on_connect():
    socketio.emit("connected", {"message": "Socket connected"})


if __name__ == "__main__":
    init_db()
    socketio.run(app, debug=True, port=5000, use_reloader=False, allow_unsafe_werkzeug=True)
