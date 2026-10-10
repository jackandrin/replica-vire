"""Shared helpers: config, env, logging, TigerData (Postgres) access, upsert."""
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
for _d in ("data", "results", "models", "logs"):
    (ROOT / _d).mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def env(name: str, required: bool = True) -> str:
    v = os.environ.get(name, "").strip()
    if required and not v:
        raise RuntimeError(f"Missing env var {name}. Fill it in {ROOT / '.env'}")
    return v


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


@contextmanager
def tiger():
    """Yield a psycopg2 connection to TigerData; commit on success, rollback on error."""
    conn = psycopg2.connect(env("TIGER_DATABASE_URL"))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query_df(sql: str, params=None) -> pd.DataFrame:
    with tiger() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def upsert(conn, table: str, df: pd.DataFrame, keys: list) -> int:
    """INSERT ... ON CONFLICT (keys) DO UPDATE. Columns not present in df are left untouched."""
    if df.empty:
        return 0
    df = df.drop_duplicates(subset=keys, keep="last")
    df = df.astype(object).where(df.notna(), None)
    cols = list(df.columns)
    upd = [c for c in cols if c not in keys]
    if upd:
        action = "DO UPDATE SET " + ", ".join(f"{c}=EXCLUDED.{c}" for c in upd)
    else:
        action = "DO NOTHING"
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s ON CONFLICT ({', '.join(keys)}) {action}"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, df.values.tolist(), page_size=1000)
    return len(df)


def log_step(step: str, ok: bool, detail: str = "") -> None:
    try:
        with tiger() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO run_log (step, ok, detail) VALUES (%s,%s,%s)", (step, ok, detail[:2000]))
    except Exception:
        pass  # logging must never break the pipeline


def universe() -> list:
    return list(cfg()["tickers"])
