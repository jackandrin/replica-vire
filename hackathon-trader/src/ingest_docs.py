"""v2 documents: Alpaca news (full content), Finnhub company-news (~1y free tier), EDGAR 8-K/10-Q/10-K text
-> TigerData.documents (dedupe on symbol+url). Replaces ingest_news for v2.
python -m src.ingest_docs [--source all|alpaca|finnhub|edgar] [--only AAPL,MSFT] [--since 2019-01-01] [--incremental]"""
import argparse
import hashlib
import re
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from . import edgar
from .common import cfg, env, get_logger, query_df, tiger, universe, upsert
from .sectors import SYMBOLS

log = get_logger("ingest_docs")
FLUSH = 500
DOC_FORMS = ("8-K", "10-Q", "10-K")


def content_hash(title, body) -> str:
    norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()  # noqa: E731
    return hashlib.sha1(f"{norm(title)}\n{norm(body)}".encode("utf-8")).hexdigest()


def make_doc(symbol, source, doc_type, published_at, title, body, url) -> dict:
    title = edgar.html_to_text(title or "")[:1000]
    body = (body or "")[:edgar.MAX_CHARS]
    return {"symbol": symbol, "source": source, "doc_type": doc_type, "published_at": edgar.utc(published_at),
            "title": title, "body": body, "url": url, "content_hash": content_hash(title, body)}


def alpaca_news(symbol: str, start: date, end: date):
    hdr = {"APCA-API-KEY-ID": env("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": env("ALPACA_API_SECRET")}
    token = None
    while True:
        p = {"symbols": symbol, "start": start.isoformat(), "end": (end + timedelta(days=1)).isoformat(),
             "limit": 50, "sort": "asc", "include_content": "true"}
        if token:
            p["page_token"] = token
        for attempt in range(4):
            r = requests.get("https://data.alpaca.markets/v1beta1/news", params=p, headers=hdr, timeout=60)
            if r.status_code != 429 and r.status_code < 500:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        j = r.json()
        for x in j.get("news", []):
            if not x.get("created_at"):
                continue
            body = edgar.html_to_text(x.get("content") or "") or edgar.html_to_text(x.get("summary") or "")
            yield make_doc(symbol, "alpaca", "news", x["created_at"], x.get("headline"), body,
                           x.get("url") or f"alpaca://news/{x.get('id')}")
        token = j.get("next_page_token")
        if not token:
            return
        time.sleep(0.35)  # free tier: 200 calls/min


def finnhub_news(symbol: str, start: date, end: date):
    cur = start
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
            break
        r.raise_for_status()
        for x in r.json() or []:
            if x.get("url") and x.get("datetime"):
                yield make_doc(symbol, "finnhub", "news", datetime.fromtimestamp(x["datetime"], tz=timezone.utc),
                               x.get("headline"), edgar.html_to_text(x.get("summary") or ""), x["url"])
        time.sleep(1.1)  # stay under 60 calls/min
        cur = nxt + timedelta(days=1)


def edgar_docs(symbol: str, since, skip_acc: set):
    cik = edgar.cik_of(symbol)
    if not cik:
        log.warning("%s: no CIK in SEC ticker map; skipping EDGAR", symbol)
        return
    company = SYMBOLS.get(symbol, (symbol,))[0]
    f = edgar.filings(cik, DOC_FORMS, since)
    f = f[~f.accessionNumber.str.replace("-", "").isin(skip_acc)]
    log.info("%s: %d EDGAR filings to fetch", symbol, len(f))
    for r in f.itertuples():
        try:
            text, url = edgar.filing_text(cik, r.accessionNumber, r.form, r.primaryDocument)
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s %s failed: %s", symbol, r.form, r.accessionNumber, str(e)[:200])
            continue
        title = f"{company} {r.form}" + (f" (Items {r.items})" if r.items else "") + f" filed {r.filingDate}"
        yield make_doc(symbol, "edgar", r.form, r.acceptanceDateTime, title, text, url)


def _last_published(source: str) -> dict:
    try:
        df = query_df("SELECT symbol, max(published_at) AS m FROM documents WHERE source=%s GROUP BY symbol", (source,))
        return {r.symbol: edgar.utc(r.m) for r in df.itertuples() if r.m is not None}
    except Exception as e:  # noqa: BLE001
        log.warning("could not read last %s documents (%s)", source, e)
        return {}


def _edgar_accessions(symbol: str) -> set:
    try:
        df = query_df("SELECT url FROM documents WHERE source='edgar' AND symbol=%s", (symbol,))
    except Exception:  # noqa: BLE001
        return set()
    return {m.group(1) for u in df.url for m in [re.search(r"/data/\d+/(\d{18})/", u or "")] if m}


def write(rows) -> int:
    """Upsert an iterator of docs in batches of FLUSH so memory stays bounded."""
    n, buf = 0, []
    for d in rows:
        buf.append(d)
        if len(buf) >= FLUSH:
            n += _flush(buf)
            buf = []
    return n + (_flush(buf) if buf else 0)


def _flush(buf) -> int:
    with tiger() as conn:
        return upsert(conn, "documents", pd.DataFrame(buf), ["symbol", "url"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["all", "alpaca", "finnhub", "edgar"])
    ap.add_argument("--only", help="comma-separated tickers")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (default config data.start)")
    ap.add_argument("--incremental", action="store_true", help="per symbol+source: from last document minus 3 days")
    a = ap.parse_args(argv)
    since = pd.Timestamp(a.since or cfg()["data"]["start"]).date()
    end = date.today()
    syms = [s.strip().upper() for s in a.only.split(",")] if a.only else universe()
    sources = ["alpaca", "finnhub", "edgar"] if a.source == "all" else [a.source]
    last = {src: _last_published(src) for src in sources} if a.incremental else {}
    if "edgar" in sources:
        try:
            edgar.ticker_cik_map()
        except Exception as e:  # noqa: BLE001
            log.error("EDGAR disabled for this run: %s", e)
            sources.remove("edgar")
    totals = {src: 0 for src in sources}
    for s in syms:
        for src in sources:
            st = since
            if src in last and s in last[src]:
                st = max(since, (last[src][s] - pd.Timedelta(days=3)).date())
            try:
                if src == "alpaca":
                    n = write(alpaca_news(s, st, end))
                elif src == "finnhub":
                    n = write(finnhub_news(s, max(st, end - timedelta(days=360)), end))  # free-tier window
                else:
                    skip = _edgar_accessions(s) if a.incremental else set()
                    n = write(edgar_docs(s, pd.Timestamp(st, tz="UTC"), skip))
            except Exception as e:  # noqa: BLE001  one bad ticker/source must not kill the run
                log.error("%s %s failed: %s", s, src, str(e)[:300])
                continue
            totals[src] += n
            log.info("%s %s: %d documents since %s", s, src, n, st)
    log.info("done: %s", totals)


if __name__ == "__main__":
    main()
