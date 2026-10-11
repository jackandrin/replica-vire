"""Copy SIGNALS, ALERTS, ORDERS, SENTIMENT_EVENTS from TigerData to Snowflake (for analysis in Snowflake).
Optional: if this fails the rest of the pipeline is unaffected.
python -m src.sync_snowflake"""
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from .common import env, get_logger, query_df

log = get_logger("sync_snowflake")

TABLES = {
    "SIGNALS": "SELECT * FROM signals",
    "ALERTS": "SELECT * FROM alerts",
    "ORDERS": "SELECT * FROM orders",
    "SENTIMENT_EVENTS": "SELECT * FROM sentiment_events WHERE published_at > now() - interval '30 days'",
}


def main(argv=None):
    conn = snowflake.connector.connect(
        account=env("SNOWFLAKE_ACCOUNT"), user=env("SNOWFLAKE_USER"), password=env("SNOWFLAKE_PASSWORD"),
        warehouse=env("SNOWFLAKE_WAREHOUSE"), database=env("SNOWFLAKE_DATABASE"), schema=env("SNOWFLAKE_SCHEMA"),
        role=env("SNOWFLAKE_ROLE", False) or None)
    for name, sql in TABLES.items():
        df = query_df(sql)
        if df.empty:
            log.info("%s: empty, skipped", name)
            continue
        df.columns = [c.upper() for c in df.columns]
        for col in df.columns:  # tz-aware timestamps -> naive UTC for write_pandas
            if str(df[col].dtype).startswith("datetime64") and getattr(df[col].dt, "tz", None) is not None:
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        ok, _, nrows, _ = write_pandas(conn, df, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
        log.info("%s: %d rows (ok=%s)", name, nrows, ok)
    conn.close()


if __name__ == "__main__":
    main()
