"""News from Finnhub (company-news) and Alpaca (news) -> TigerData.news.
Finnhub free tier only serves ~1 year of history at ~60 calls/min; Alpaca news goes back years.
python -m src.ingest_news [--source both|finnhub|alpaca] [--days N] [--only AAPL,MSFT]"""
import argparse
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from .common import cfg, env, get_logger, tiger, universe, upsert

log = get_logger("ingest_news")


def finnhub(symbol: str, start: date, end: date) -> list:
    rows, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=29), end)
        r = None
        for attempt in range(4):
            r = requests.get("https://finnhub.io/api/v1/company-news", timeout=30,
                             params={"symbol": symbol, "from": cur.isoformat(), "to": nxt.isoformat(),
                                     "token": env("FINNHUB_API_KEY")})
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        for x in r.json():
            if x.get("url") and x.get("datetime"):
                rows.append({"symbol": symbol, "source_api": "finnhub",
                             "event_time": datetime.fromtimestamp(x["datetime"], tz=timezone.utc),
                             "headline": x.get("headline"), "summary": x.get("summary"),
                             "publisher": x.get("source"), "url": x["url"]})
        time.sleep(1.1)  # stay under 60 calls/min
        cur = nxt + timedelta(days=1)
    return rows


def alpaca(symbol: str, start: date, end: date) -> list:
    hdr = {"APCA-API-KEY-ID": env("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": env("ALPACA_API_SECRET")}
    rows, token = [], None
    while True:
        p = {"symbols": symbol, "start": start.isoformat(), "end": (end + timedelta(days=1)).isoformat(),
             "limit": 50, "sort": "asc", "include_content": "false"}
        if token:
            p["page_token"] = token
        r = requests.get("https://data.alpaca.markets/v1beta1/news", params=p, headers=hdr, timeout=30)
        r.raise_for_status()
        j = r.json()
        for x in j.get("news", []):
            if x.get("url"):
                rows.append({"symbol": symbol, "source_api": "alpaca",
                             "event_time": pd.Timestamp(x["created_at"]).to_pydatetime(),
                             "headline": x.get("headline"), "summary": x.get("summary"),
                             "publisher": x.get("source"), "url": x["url"]})
        token = j.get("next_page_token")
        if not token:
            return rows
        time.sleep(0.35)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="both", choices=["both", "finnhub", "alpaca"])
    ap.add_argument("--days", type=int, default=cfg()["history_days"])
    ap.add_argument("--only", help="comma-separated tickers (testing)")
    a = ap.parse_args(argv)
    end = date.today()
    start = end - timedelta(days=a.days)
    fh_start = max(start, end - timedelta(days=360))  # finnhub free-tier window
    syms = a.only.split(",") if a.only else universe()
    for s in syms:
        rows = []
        try:
            if a.source in ("both", "alpaca"):
                rows += alpaca(s, start, end)
            if a.source in ("both", "finnhub"):
                rows += finnhub(s, fh_start, end)
        except Exception as e:  # one bad ticker must not kill the backfill
            log.error("%s failed: %s", s, e)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(["symbol", "url"])
            with tiger() as conn:
                upsert(conn, "news", df, ["symbol", "url"])
        log.info("%s: %d articles", s, len(rows))


if __name__ == "__main__":
    main()
