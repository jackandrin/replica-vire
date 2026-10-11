"""v2 events -> TigerData.events: earnings (8-K Item 2.02), insider (Form 4 P/S), analyst (Finnhub
recommendation trends), short_volume (FINRA Reg SHO daily CNMS files). Rules in docs/V2_CONTRACT.md.
python -m src.ingest_events [--kinds earnings,insider,analyst,short_volume] [--only AAPL] [--since 2019-01-01] [--incremental]"""
import argparse
import json
import math
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import edgar
from .common import cfg, env, get_logger, query_df, tiger, universe, upsert
from .sectors import CONTEXT_SYMBOLS

log = get_logger("ingest_events")
KINDS = ["earnings", "insider", "analyst", "short_volume"]
KEYS = ["symbol", "kind", "event_time", "key"]
ET_TZ = "America/New_York"
FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{:%Y%m%d}.txt"


def et(d, hhmm: str) -> pd.Timestamp:
    """Calendar date + US/Eastern wall time -> UTC timestamp."""
    return pd.Timestamp(f"{pd.Timestamp(d).date()} {hhmm}").tz_localize(ET_TZ).tz_convert("UTC")


def ev(symbol, kind, event_time, available_at, value, key, payload) -> dict:
    return {"symbol": symbol, "kind": kind, "event_time": event_time, "available_at": available_at,
            "value": None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value),
            "key": str(key), "payload": json.dumps(payload, default=str)}


# ---------------------------------------------------------------- pure parsers (unit-tested)
def analyst_score(r: dict) -> float:
    """(2*strongBuy + buy - sell - 2*strongSell) / total, total includes hold. NaN if no analysts."""
    c = {k: float(r.get(k) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
    total = sum(c.values())
    return (2 * c["strongBuy"] + c["buy"] - c["sell"] - 2 * c["strongSell"]) / total if total else float("nan")


def parse_finra_line(line: str):
    """'20240102|AAPL|123|0|456|B,Q,N' -> dict, or None for header/footer/malformed lines."""
    p = line.strip().split("|")
    if len(p) < 5 or not p[0].isdigit():
        return None
    try:
        short, total = float(p[2]), float(p[4])
    except ValueError:
        return None
    return {"date": pd.Timestamp(p[0]).date(), "symbol": p[1].strip().upper(), "short_volume": short,
            "total_volume": total, "ratio": short / total if total > 0 else float("nan")}


def short_volume_events(text: str, universe_set: set) -> list:
    out = []
    for line in text.splitlines():
        x = parse_finra_line(line)
        if x and x["symbol"] in universe_set:
            d = x["date"]
            out.append(ev(x["symbol"], "short_volume", et(d, "16:00"), et(d + timedelta(days=1), "08:00"), x["ratio"],
                          f"{d:%Y%m%d}", {"short_volume": x["short_volume"], "total_volume": x["total_volume"]}))
    return out


def insider_events(symbol: str, accession: str, accepted, parsed: dict) -> list:
    out = []
    for t in parsed["trades"]:
        if not t["date"]:
            continue
        sign = 1.0 if t["code"] == "P" else -1.0
        out.append(ev(symbol, "insider", et(t["date"][:10], "16:00"), accepted, sign * t["shares"] * t["price"],
                      f"{accession}-{t['row']}",
                      {"code": t["code"], "shares": t["shares"], "price": t["price"], "owner": parsed["owner"],
                       "officer": parsed["officer"], "director": parsed["director"]}))
    return out


# ---------------------------------------------------------------- fetchers
def earnings(symbol: str, since) -> list:
    cik = edgar.cik_of(symbol)
    if not cik:
        return []
    f = edgar.filings(cik, ["8-K"], since)  # amendments would double-count
    f = f[f["items"].map(edgar.has_item)]
    return [ev(symbol, "earnings", r.acceptanceDateTime, r.acceptanceDateTime, None, r.accessionNumber,
               {"accession": r.accessionNumber, "form": r.form, "items": r.items}) for r in f.itertuples()]


def insider(symbol: str, since, skip: set) -> list:
    cik = edgar.cik_of(symbol)
    if not cik:
        return []
    f = edgar.filings(cik, ["4"], since)
    f = f[~f.accessionNumber.isin(skip)]
    log.info("%s: %d Form 4 filings to parse", symbol, len(f))
    out = []
    for r in f.itertuples():
        try:
            p = edgar.form4_trades(cik, r.accessionNumber, r.primaryDocument)
        except Exception as e:  # noqa: BLE001
            log.warning("%s Form 4 %s failed: %s", symbol, r.accessionNumber, str(e)[:160])
            continue
        if p["issuer_cik"] and p["issuer_cik"] != cik:
            continue  # the company itself reporting holdings in another issuer
        out += insider_events(symbol, r.accessionNumber, r.acceptanceDateTime, p)
    return out


def analyst(symbol: str) -> list:
    r = None
    for attempt in range(4):
        r = requests.get("https://finnhub.io/api/v1/stock/recommendation", timeout=30,
                         params={"symbol": symbol, "token": env("FINNHUB_API_KEY")})
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        break
    r.raise_for_status()
    time.sleep(1.1)  # 60 calls/min
    out = []
    for x in r.json() or []:
        if not x.get("period"):
            continue
        t = et(x["period"], "00:00")
        counts = {k: x.get(k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
        out.append(ev(symbol, "analyst", t, t + pd.Timedelta(days=31), analyst_score(x), x["period"], counts))
    return out


def short_volume(syms: set, start: date, end: date) -> int:
    hdr = {"User-Agent": env("SEC_USER_AGENT", required=False) or "hackathon-trader research script"}
    n, d, missing = 0, start, 0
    while d <= end:
        if d.weekday() < 5:
            try:
                r = requests.get(FINRA_URL.format(d), headers=hdr, timeout=60)
                if r.status_code in (403, 404):
                    missing += 1  # holiday / not yet published
                else:
                    r.raise_for_status()
                    rows = short_volume_events(r.text, syms)
                    if rows:
                        with tiger() as conn:
                            n += upsert(conn, "events", pd.DataFrame(rows), KEYS)
            except Exception as e:  # noqa: BLE001
                log.warning("FINRA %s failed: %s", d, str(e)[:160])
            time.sleep(0.1)
        if d.day == 1:
            log.info("short_volume: at %s, %d rows so far", d, n)
        d += timedelta(days=1)
    log.info("short_volume: %d rows %s..%s (%d weekdays without a file)", n, start, end, missing)
    return n


def _last(kind: str) -> dict:
    try:
        df = query_df("SELECT symbol, max(available_at) AS m FROM events WHERE kind=%s GROUP BY symbol", (kind,))
        return {r.symbol: edgar.utc(r.m) for r in df.itertuples() if r.m is not None}
    except Exception as e:  # noqa: BLE001
        log.warning("could not read last %s events (%s)", kind, e)
        return {}


def _insider_accessions(symbol: str) -> set:
    try:
        df = query_df("SELECT DISTINCT split_part(key, '-', 1) || '-' || split_part(key, '-', 2) || '-' || "
                      "split_part(key, '-', 3) AS acc FROM events WHERE kind='insider' AND symbol=%s", (symbol,))
        return set(df.acc)
    except Exception:  # noqa: BLE001
        return set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--only", help="comma-separated tickers")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (default config data.start)")
    ap.add_argument("--incremental", action="store_true", help="from last stored event minus 3 days")
    a = ap.parse_args(argv)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    bad = set(kinds) - set(KINDS)
    if bad:
        ap.error(f"unknown kinds {bad}; choose from {KINDS}")
    since = pd.Timestamp(a.since or cfg()["data"]["start"], tz="UTC")
    syms = [s.strip().upper() for s in a.only.split(",")] if a.only else universe()
    if {"earnings", "insider"} & set(kinds):
        try:
            edgar.ticker_cik_map()
        except Exception as e:  # noqa: BLE001
            log.error("EDGAR kinds disabled for this run: %s", e)
            kinds = [k for k in kinds if k not in ("earnings", "insider")]
    for kind in [k for k in kinds if k != "short_volume"]:
        last = _last(kind) if a.incremental and kind != "analyst" else {}
        total = 0
        for s in syms:
            st = max(since, last[s] - pd.Timedelta(days=3)) if s in last else since
            try:
                if kind == "earnings":
                    rows = earnings(s, st)
                elif kind == "insider":
                    rows = insider(s, st, _insider_accessions(s) if a.incremental else set())
                else:
                    rows = analyst(s)  # endpoint returns the full available history every call
                if rows:
                    with tiger() as conn:
                        total += upsert(conn, "events", pd.DataFrame(rows), KEYS)
                log.info("%s %s: %d events", s, kind, len(rows))
            except Exception as e:  # noqa: BLE001  one bad ticker must not kill the run
                log.error("%s %s failed: %s", s, kind, str(e)[:300])
        log.info("%s done: %d events", kind, total)
    if "short_volume" in kinds:
        start = since.date()
        if a.incremental:
            try:
                m = query_df("SELECT max(event_time) AS m FROM events WHERE kind='short_volume'")["m"].iloc[0]
                if m is not None and not pd.isna(m):
                    start = max(start, (edgar.utc(m) - pd.Timedelta(days=3)).date())
            except Exception as e:  # noqa: BLE001
                log.warning("could not read last short_volume (%s)", e)
        uni = set(syms) | (set() if a.only else set(CONTEXT_SYMBOLS))
        short_volume(uni, start, date.today())


if __name__ == "__main__":
    main()
