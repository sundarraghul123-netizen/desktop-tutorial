"""SQLite persistence. All queries are parameterised (no string-built SQL with user data).
Month/year are derived from transaction_date at read time, so they can never drift out of sync."""
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

if __package__:
    from .config import DB_PATH
else:
    from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, rows_total INTEGER, rows_added INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER DEFAULT 1, batch_id INTEGER, txn_hash TEXT,
  transaction_date TEXT, description TEXT, merchant TEXT, amount REAL, transaction_type TEXT,
  category TEXT, subcategory TEXT, payment_method TEXT, user_status TEXT DEFAULT '',
  category_confidence REAL, category_source TEXT, category_reason TEXT, created_at TEXT,
  UNIQUE(user_id, txn_hash));
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date);
CREATE TABLE IF NOT EXISTS merchant_categories (
  user_id INTEGER DEFAULT 1, merchant TEXT, category TEXT, subcategory TEXT, source TEXT DEFAULT 'user',
  created_at TEXT, PRIMARY KEY(user_id, merchant));
CREATE TABLE IF NOT EXISTS budgets (
  user_id INTEGER DEFAULT 1, category TEXT, monthly_limit REAL, PRIMARY KEY(user_id, category));
CREATE TABLE IF NOT EXISTS goals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER DEFAULT 1, goal_name TEXT, target_amount REAL,
  current_amount REAL, target_date TEXT, monthly_contribution REAL, created_at TEXT);
"""
TXN_COLS = ["txn_hash", "transaction_date", "description", "merchant", "amount", "transaction_type", "category",
            "subcategory", "payment_method", "category_confidence", "category_source", "category_reason"]
EDITABLE = {
    "amount",
    "category",
    "merchant",
    "payment_method",
    "subcategory",
    "transaction_type",
    "user_status",
}


def _now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def add_upload(filename, total):
    with conn() as c:
        return c.execute("INSERT INTO uploads(filename, rows_total, rows_added, created_at) VALUES (?,?,0,?)",
                         (filename, int(total), _now())).lastrowid


def insert_transactions(df, batch_id):
    """Insert rows, silently skipping hashes already stored. Returns number actually inserted."""
    now = _now()
    recs = [(batch_id, now, *[r[c] if c != "transaction_date" else r[c].strftime("%Y-%m-%d %H:%M:%S") for c in TXN_COLS])
            for r in df.to_dict("records")]
    with conn() as c:
        before = c.total_changes
        c.executemany(f"INSERT OR IGNORE INTO transactions(batch_id, created_at, {','.join(TXN_COLS)}) VALUES (?,?,{','.join('?' * len(TXN_COLS))})", recs)
        added = c.total_changes - before
        c.execute("UPDATE uploads SET rows_added=? WHERE id=?", (added, batch_id))
    return added


def load():
    with conn() as c:
        df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id=1 ORDER BY transaction_date, id", c)
    if not df.empty:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def uploads():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM uploads ORDER BY id DESC", c)


def update_transaction(txn_id, **fields):
    fields = {k: v for k, v in fields.items() if k in EDITABLE}
    if not fields:
        return
    classification_changed = bool(
        {"category", "merchant", "subcategory"} & fields.keys()
    )
    if "amount" in fields:
        try:
            amount = float(fields["amount"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Transaction amount must be a valid number.") from exc
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Transaction amount must be greater than zero.")
        fields["amount"] = round(amount, 2)
    if "transaction_type" in fields and fields["transaction_type"] not in {
        "Expense", "Income", "Refund", "Transfer"
    }:
        raise ValueError("Transaction type must be Expense, Income, Refund, or Transfer.")
    if "user_status" in fields and fields["user_status"] not in {"", "normal", "suspicious"}:
        raise ValueError("Review status must be blank, normal, or suspicious.")
    if classification_changed:
        fields.update(
            category_confidence=1.0,
            category_source="user",
            category_reason="Classification updated manually.",
        )
    with conn() as c:
        c.execute(f"UPDATE transactions SET {', '.join(k + '=?' for k in fields)} WHERE id=?", (*fields.values(), int(txn_id)))


def delete_transactions(ids):
    with conn() as c:
        c.executemany("DELETE FROM transactions WHERE id=?", [(int(i),) for i in ids])


def delete_batch(batch_id):
    with conn() as c:
        c.execute("DELETE FROM transactions WHERE batch_id=?", (int(batch_id),))
        c.execute("DELETE FROM uploads WHERE id=?", (int(batch_id),))


def delete_all():
    with conn() as c:
        for t in ("transactions", "uploads", "merchant_categories", "budgets", "goals"):
            c.execute(f"DELETE FROM {t}")


# ---- personalised learning
def merchant_map():
    with conn() as c:
        return {r["merchant"].lower(): (r["category"], r["subcategory"]) for r in c.execute("SELECT * FROM merchant_categories WHERE user_id=1")}


def set_merchant_category(merchant, category, subcategory, ttype=None):
    """Remember the mapping and apply it to every stored transaction of that merchant (same transaction type if given)."""
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO merchant_categories(user_id, merchant, category, subcategory, source, created_at) VALUES (1,?,?,?,'user',?)",
                  (merchant.lower(), category, subcategory, _now()))
        q = "UPDATE transactions SET category=?, subcategory=?, category_confidence=1.0, category_source='user', category_reason=? WHERE merchant=?"
        args = [category, subcategory, f"Classified as {category} because you confirmed this merchant ({merchant}).", merchant]
        if ttype:
            q += " AND transaction_type=?"
            args.append(ttype)
        c.execute(q, args)


def confirmed_pairs():
    with conn() as c:
        return [(r["merchant"], r["category"]) for r in c.execute("SELECT merchant, category FROM merchant_categories WHERE user_id=1")]


def reset_merchant_map():
    with conn() as c:
        c.execute("DELETE FROM merchant_categories")


# ---- budgets & goals
def budgets():
    with conn() as c:
        return {r["category"]: r["monthly_limit"] for r in c.execute("SELECT * FROM budgets WHERE user_id=1")}


def set_budget(category, limit):
    with conn() as c:
        if limit and limit > 0:
            c.execute("INSERT OR REPLACE INTO budgets(user_id, category, monthly_limit) VALUES (1,?,?)", (category, float(limit)))
        else:
            c.execute("DELETE FROM budgets WHERE user_id=1 AND category=?", (category,))


def goals():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM goals WHERE user_id=1 ORDER BY id", c)


def add_goal(name, target, current, target_date, monthly):
    with conn() as c:
        c.execute("INSERT INTO goals(user_id, goal_name, target_amount, current_amount, target_date, monthly_contribution, created_at) VALUES (1,?,?,?,?,?,?)",
                  (name, float(target), float(current), target_date, float(monthly or 0), _now()))


def update_goal_amount(goal_id, current):
    with conn() as c:
        c.execute("UPDATE goals SET current_amount=? WHERE id=?", (float(current), int(goal_id)))


def delete_goal(goal_id):
    with conn() as c:
        c.execute("DELETE FROM goals WHERE id=?", (int(goal_id),))
