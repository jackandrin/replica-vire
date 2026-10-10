"""Score news.sentiment (-1..1). Default = VADER (local, always works).
--cortex uses Snowflake Cortex SENTIMENT instead (if your account has it).
python -m src.sentiment [--cortex]"""
import argparse

import psycopg2.extras

from .common import env, get_logger, query_df, tiger

log = get_logger("sentiment")


def _text(row) -> str:
    return f"{row.headline or ''}. {(row.summary or '')[:300]}".strip()


def score_vader(df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    return [an.polarity_scores(_text(r))["compound"] for r in df.itertuples()]


def score_cortex(df):
    import snowflake.connector
    conn = snowflake.connector.connect(account=env("SNOWFLAKE_ACCOUNT"), user=env("SNOWFLAKE_USER"),
                                       password=env("SNOWFLAKE_PASSWORD"), warehouse=env("SNOWFLAKE_WAREHOUSE"),
                                       database=env("SNOWFLAKE_DATABASE"), schema=env("SNOWFLAKE_SCHEMA"),
                                       role=env("SNOWFLAKE_ROLE", False) or None)
    cur = conn.cursor()
    cur.execute("CREATE OR REPLACE TEMPORARY TABLE news_tmp (id NUMBER, txt STRING)")
    rows = [(int(r.id), _text(r)) for r in df.itertuples()]
    cur.executemany("INSERT INTO news_tmp VALUES (%s, %s)", rows)
    cur.execute("SELECT id, SNOWFLAKE.CORTEX.SENTIMENT(txt) FROM news_tmp")
    out = dict(cur.fetchall())
    conn.close()
    return [float(out[int(i)]) for i in df.id]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortex", action="store_true")
    a = ap.parse_args(argv)
    df = query_df("SELECT id, headline, summary FROM news WHERE sentiment IS NULL")
    if df.empty:
        log.info("nothing to score")
        return
    try:
        scores = score_cortex(df) if a.cortex else score_vader(df)
    except Exception as e:  # fall back so the pipeline never blocks on Snowflake
        log.warning("cortex failed (%s); falling back to VADER", e)
        scores = score_vader(df)
    with tiger() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, "UPDATE news n SET sentiment = v.s FROM (VALUES %s) AS v(id, s) WHERE n.id = v.id",
            [(int(i), float(s)) for i, s in zip(df.id, scores)], page_size=2000)
    log.info("scored %d articles", len(df))


if __name__ == "__main__":
    main()
