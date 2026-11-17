"""Daily macro tracker stored in a local SQLite file.

Each log row keeps the per-unit nutrient values it was calculated from, the basis
(per_serving or per_100g) and the source. Consumed amounts are
    per serving:  quantity_servings * value_per_serving
    per 100 g:    grams / 100 * value_per_100g
Unknown values stay NULL and are reported as missing, not added as zero.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

from .config import APP_DB

MACROS = ["kcal", "protein_g", "carbs_g", "fat_g"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_date TEXT NOT NULL,
    item_type TEXT NOT NULL,          -- recipe or food
    item_id TEXT NOT NULL,
    item_name TEXT NOT NULL,
    source TEXT NOT NULL,
    basis TEXT NOT NULL,              -- per_serving or per_100g
    quantity REAL NOT NULL,           -- servings or grams, depending on basis
    unit_kcal REAL, unit_protein_g REAL, unit_carbs_g REAL, unit_fat_g REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL,
    value INTEGER NOT NULL,           -- 1 like, -1 dislike
    source TEXT NOT NULL DEFAULT 'app_feedback',
    created_at TEXT NOT NULL
);
"""


def connect(path=APP_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def consumed(basis: str, quantity: float, unit_value):
    if unit_value is None:
        return None
    if basis == "per_serving":
        return quantity * unit_value
    if basis == "per_100g":
        return quantity / 100.0 * unit_value
    raise ValueError(f"unknown basis {basis!r}")


def add_log(con, *, item_type, item_id, item_name, source, basis, quantity, unit_values: dict,
            log_date: date | None = None) -> int:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if basis not in ("per_serving", "per_100g"):
        raise ValueError("basis must be per_serving or per_100g")
    cur = con.execute(
        """INSERT INTO meal_logs (log_date, item_type, item_id, item_name, source, basis, quantity,
           unit_kcal, unit_protein_g, unit_carbs_g, unit_fat_g, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        ((log_date or date.today()).isoformat(), item_type, str(item_id), item_name, source, basis,
         float(quantity), *[unit_values.get(m) for m in MACROS], datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    return cur.lastrowid


def update_quantity(con, log_id: int, quantity: float):
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    con.execute("UPDATE meal_logs SET quantity = ? WHERE id = ?", (float(quantity), log_id))
    con.commit()


def delete_log(con, log_id: int):
    con.execute("DELETE FROM meal_logs WHERE id = ?", (log_id,))
    con.commit()


def logs_for_day(con, day: date) -> list[dict]:
    rows = con.execute("SELECT * FROM meal_logs WHERE log_date = ? ORDER BY id", (day.isoformat(),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for m in MACROS:
            d[m] = consumed(d["basis"], d["quantity"], d[f"unit_{m}"])
        out.append(d)
    return out


def daily_summary(con, day: date, targets: dict) -> dict:
    """Totals per macro, with the number of logged items whose value is unknown."""
    logs = logs_for_day(con, day)
    summary = {}
    for m in MACROS:
        known = [x[m] for x in logs if x[m] is not None]
        total = sum(known) if known else 0.0
        target = targets.get(m)
        summary[m] = {
            "consumed": total,
            "missing_items": len(logs) - len(known),
            "target": target,
            "remaining": (target - total) if target else None,
        }
    return {"logs": logs, "totals": summary}


# --- profile and feedback (single local user) -------------------------------------
def save_profile(con, profile: dict):
    con.execute("INSERT INTO profile (id, data) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (json.dumps(profile),))
    con.commit()


def load_profile(con) -> dict | None:
    row = con.execute("SELECT data FROM profile WHERE id = 1").fetchone()
    return json.loads(row["data"]) if row else None


def add_feedback(con, recipe_id: int, value: int):
    con.execute("INSERT INTO feedback (recipe_id, value, created_at) VALUES (?,?,?)",
                (int(recipe_id), int(value), datetime.now().isoformat(timespec="seconds")))
    con.commit()


def load_feedback(con) -> tuple[list[int], list[int]]:
    """Latest feedback per recipe wins. Returns (liked ids, disliked ids)."""
    rows = con.execute("SELECT recipe_id, value FROM feedback ORDER BY id").fetchall()
    last = {}
    for r in rows:
        last[r["recipe_id"]] = r["value"]
    return [k for k, v in last.items() if v > 0], [k for k, v in last.items() if v < 0]


def clear_feedback(con):
    con.execute("DELETE FROM feedback")
    con.commit()
