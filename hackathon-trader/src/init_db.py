"""Create tables.  python -m src.init_db --tiger [--snowflake]"""
import argparse

from .common import ROOT, env, get_logger, tiger

log = get_logger("init_db")


def split_sql(text: str) -> list:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def init_tiger():
    stmts = split_sql((ROOT / "sql/tiger_schema.sql").read_text())
    with tiger() as conn, conn.cursor() as cur:
        for s in stmts:
            cur.execute(s)
    log.info("TigerData schema ready (%d statements)", len(stmts))


def init_snowflake():
    import snowflake.connector
    conn = snowflake.connector.connect(
        account=env("SNOWFLAKE_ACCOUNT"), user=env("SNOWFLAKE_USER"), password=env("SNOWFLAKE_PASSWORD"),
        role=env("SNOWFLAKE_ROLE", False) or None)
    cur = conn.cursor()
    for s in split_sql((ROOT / "sql/snowflake_schema.sql").read_text()):
        cur.execute(s)
    conn.close()
    log.info("Snowflake warehouse/database/schema ready")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiger", action="store_true")
    ap.add_argument("--snowflake", action="store_true")
    a = ap.parse_args(argv)
    if a.tiger or not a.snowflake:
        init_tiger()
    if a.snowflake:
        init_snowflake()


if __name__ == "__main__":
    main()
