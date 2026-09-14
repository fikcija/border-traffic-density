"""
SQLite storage for the serving API: user accounts and a prediction log.

One file, two tables, no ORM. The API runs as a single uvicorn process, so each
operation opens a short-lived connection rather than using a pool.

    SERVE_DB_PATH   where the file lives (default ./serve.db)
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("SERVE_DB_PATH", "serve.db"))

# SQLite serialises writers itself; this keeps insert-then-read paths from
# interleaving.
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    username      TEXT,
    camera        TEXT    NOT NULL,
    prediction    TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    probabilities TEXT    NOT NULL,
    latency_ms    REAL    NOT NULL,
    model_version TEXT,
    model_source  TEXT,
    filename      TEXT,
    batch_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts     ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_predictions_camera ON predictions(camera);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    """Create the file and schema if absent. Called once from the app's lifespan."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        # WAL lets reads run while a prediction is being written.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    print(f"serve db at {DB_PATH.resolve()}", flush=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- users
def create_user(username: str, password_hash: str) -> int:
    """Insert a user. Raises sqlite3.IntegrityError when the name is taken."""
    with _write_lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, _utcnow()))
        return cur.lastrowid


def get_user(username: str):
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchone()


# ---------------------------------------------------------------- predictions
def log_prediction(*, username, camera, prediction, confidence, probabilities,
                   latency_ms, model_version=None, model_source=None,
                   filename=None, batch_id=None):
    """Record one served prediction. Callers treat a failure here as non-fatal."""
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO predictions (ts, username, camera, prediction, confidence,"
            " probabilities, latency_ms, model_version, model_source, filename,"
            " batch_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_utcnow(), username, camera, prediction, float(confidence),
             json.dumps(probabilities), float(latency_ms), model_version,
             model_source, filename, batch_id))


def recent_predictions(limit=50, offset=0, camera=None, prediction=None):
    sql = "SELECT * FROM predictions"
    where, params = [], []
    if camera:
        where.append("camera = ?")
        params.append(camera)
    if prediction:
        where.append("prediction = ?")
        params.append(prediction)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["probabilities"] = json.loads(d["probabilities"])
        out.append(d)
    return out


def prediction_stats():
    """Counts and mean confidence, overall and split by class and camera.

    Same shape as the training pipeline's summary, so the two are comparable.
    """
    with _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n, AVG(confidence) AS mean_conf,"
            " AVG(latency_ms) AS mean_latency FROM predictions").fetchone()
        by_class = conn.execute(
            "SELECT prediction, COUNT(*) AS n, AVG(confidence) AS mean_conf"
            " FROM predictions GROUP BY prediction ORDER BY n DESC").fetchall()
        by_camera = conn.execute(
            "SELECT camera, COUNT(*) AS n, AVG(confidence) AS mean_conf"
            " FROM predictions GROUP BY camera ORDER BY n DESC").fetchall()
        span = conn.execute(
            "SELECT MIN(ts) AS first, MAX(ts) AS last FROM predictions").fetchone()

    def _round(v, n=4):
        return round(v, n) if v is not None else None

    return {
        "total": total["n"],
        "mean_confidence": _round(total["mean_conf"]),
        "mean_latency_ms": _round(total["mean_latency"], 1),
        "first_seen": span["first"],
        "last_seen": span["last"],
        "by_class": [
            {"prediction": r["prediction"], "n": r["n"],
             "mean_confidence": _round(r["mean_conf"]),
             "share": _round(r["n"] / total["n"]) if total["n"] else None}
            for r in by_class
        ],
        "by_camera": [
            {"camera": r["camera"], "n": r["n"],
             "mean_confidence": _round(r["mean_conf"])}
            for r in by_camera
        ],
    }
