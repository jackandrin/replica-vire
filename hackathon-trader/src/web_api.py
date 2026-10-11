"""Read-only JSON API for ReplicaVire's React frontend.

Run from hackathon-trader/: python -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
Uses the EXISTING TigerData tables and model evaluation artifacts, not demo values.
No Alpaca trading calls, API tokens, database URLs, or secrets are returned.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .common import ROOT, cfg, tiger

app = FastAPI(title="ReplicaVire Research API", version="0.1.0")

# Local development: Vite proxies /api, so CORS is normally unnecessary.
# If using a separate frontend origin, explicitly set FRONTEND_ORIGIN in the server environment.
if os.getenv("FRONTEND_ORIGIN"):
    app.add_middleware(CORSMiddleware, allow_origins=[os.environ["FRONTEND_ORIGIN"]],
                       allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])

COMPANIES = {
    "AAPL": ("Apple", "Technology"), "MSFT": ("Microsoft", "Technology"),
    "NVDA": ("NVIDIA", "Technology"), "AMZN": ("Amazon", "Consumer"),
    "GOOGL": ("Alphabet", "Communication"), "META": ("Meta", "Communication"),
    "TSLA": ("Tesla", "Consumer"), "JPM": ("JPMorgan Chase", "Financial"),
    "XOM": ("Exxon Mobil", "Energy"), "UNH": ("UnitedHealth", "Healthcare"),
    "JNJ": ("Johnson & Johnson", "Healthcare"), "V": ("Visa", "Financial"),
    "WMT": ("Walmart", "Consumer"), "PG": ("Procter & Gamble", "Consumer"),
    "HD": ("Home Depot", "Consumer"), "KO": ("Coca-Cola", "Consumer"),
    "PEP": ("PepsiCo", "Consumer"), "COST": ("Costco", "Consumer"),
    "AMD": ("AMD", "Technology"), "NFLX": ("Netflix", "Communication"),
    "DUOL": ("Duolingo", "Consumer"), "UBER": ("Uber", "Technology"),
    "SPOT": ("Spotify", "Communication"), "RBLX": ("Roblox", "Communication"),
}


def clean(value):
    """Convert dates/numpy-like floats to JSON-safe values (missing => null)."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    return value


def rows(cur, sql, args=()):
    """DB-API query with parameter placeholders, returning dictionaries."""
    cur.execute(sql, args)
    columns = [x[0] for x in cur.description]
    return [dict(zip(columns, r)) for r in cur.fetchall()]


def available_symbols():
    """Source of truth is the Python pipeline's configured universe."""
    return list(cfg()["tickers"])


def require_symbol(symbol: str):
    sym = symbol.upper().strip()
    if sym not in available_symbols():
        raise HTTPException(status_code=404, detail=f"{sym} is not in config.yaml's tracked universe")
    return sym


def db_error(exc: Exception):
    # Never include database URLs, SQL, login names, or connection strings in HTTP responses.
    if not os.getenv("TIGER_DATABASE_URL", "").strip():
        return HTTPException(status_code=503, detail="TigerData isn't configured on the API server")
    return HTTPException(status_code=503, detail="TigerData is unavailable or its tables are not initialized")


def load_snapshot():
    symbols = available_symbols()
    try:
        with tiger() as conn, conn.cursor() as cur:
            bar_rows = rows(cur, """
                SELECT symbol, ts, close, ingested_at FROM (
                    SELECT symbol, ts, close, ingested_at,
                           row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) AS n
                    FROM bars WHERE symbol = ANY(%s)
                ) x WHERE n <= 12 ORDER BY symbol, ts DESC
            """, (symbols,))
            feature_rows = rows(cur, """
                SELECT DISTINCT ON (symbol) symbol, d, sent_1d, sent_5d, news_cnt_5d, computed_at
                FROM features WHERE symbol = ANY(%s)
                ORDER BY symbol, d DESC
            """, (symbols,))
            prediction_rows = rows(cur, """
                SELECT symbol, asof, model, mu, sigma, created_at
                FROM predictions
                WHERE asof = (SELECT max(asof) FROM predictions)
                  AND symbol = ANY(%s)
            """, (symbols,))
    except Exception as exc:
        raise db_error(exc) from None

    prices = {}
    for rec in bar_rows:
        prices.setdefault(rec["symbol"], []).append(rec)
    features = {r["symbol"]: r for r in feature_rows}
    forecasts = {}
    for r in prediction_rows:
        forecasts.setdefault(r["symbol"], {})[r["model"]] = r
    stocks = []
    for sym in symbols:
        hist = prices.get(sym, [])
        last, prev = (hist + [None, None])[:2]
        f = features.get(sym)
        p = forecasts.get(sym, {})
        # A forecast is displayed only when BOTH models share an as-of date and
        # the latest feature date matches it (avoid silently presenting stale scores).
        ridge, mlp = p.get("ridge"), p.get("mlp")
        same_asof = bool(ridge and mlp and f and ridge["asof"] == mlp["asof"] == f["d"])
        ridge_mu = float(ridge["mu"]) if same_asof and ridge["mu"] is not None else None
        mlp_mu = float(mlp["mu"]) if same_asof and mlp["mu"] is not None else None
        mean_mu = (ridge_mu + mlp_mu) / 2 if ridge_mu is not None and mlp_mu is not None else None
        comp, sector = COMPANIES.get(sym, (sym, "Other"))
        stocks.append({
            "ticker": sym, "company": comp, "sector": sector,
            "price": float(last["close"]) if last and last["close"] is not None else None,
            "price_date": last["ts"].date() if last else None,
            "day_change_pct": 100 * (float(last["close"]) / float(prev["close"]) - 1)
            if last and prev and last["close"] is not None and prev["close"] else None,
            "sentiment": float(f["sent_5d"]) if f and f["sent_5d"] is not None else None,
            "sentiment_1d": float(f["sent_1d"]) if f and f["sent_1d"] is not None else None,
            "news_count_5d": int(f["news_cnt_5d"]) if f and f["news_cnt_5d"] is not None else None,
            "feature_date": f["d"] if f else None,
            "prediction_date": ridge["asof"] if same_asof else None,
            # Database stores fractions of EXCESS RETURN vs SPY, NOT raw stock-price returns.
            "ridge_excess_pct": 100 * ridge_mu if ridge_mu is not None else None,
            "mlp_excess_pct": 100 * mlp_mu if mlp_mu is not None else None,
            "mean_excess_pct": 100 * mean_mu if mean_mu is not None else None,
            "spread_pp": 100 * abs(ridge_mu - mlp_mu) if mean_mu is not None else None,
            "price_history": [{"date": r["ts"].date(), "close": float(r["close"])}
                              for r in reversed(hist) if r["close"] is not None],
        })
    return clean({"stocks": stocks, "benchmark": cfg()["benchmark"],
                  "sentiment_source": "financial_news", "prediction_target": "five-day excess return vs SPY",
                  "review_data_connected": False})


@app.get("/api/health")
def health():
    if not os.getenv("TIGER_DATABASE_URL", "").strip():
        return {"status": "unconfigured", "detail": "Set TIGER_DATABASE_URL in hackathon-trader/.env"}
    try:
        with tiger() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM bars LIMIT 1")
        return {"status": "connected"}
    except Exception:
        return {"status": "unavailable", "detail": "Database connection or schema unavailable"}


@app.get("/api/overview")
def overview():
    return load_snapshot()


@app.get("/api/stocks/{symbol}")
def stock_detail(symbol: str):
    sym = require_symbol(symbol)
    snap = load_snapshot()
    s = next(item for item in snap["stocks"] if item["ticker"] == sym)
    try:
        with tiger() as conn, conn.cursor() as cur:
            history = rows(cur, """
                SELECT ts, open, high, low, close, volume FROM bars
                WHERE symbol = %s ORDER BY ts DESC LIMIT 150
            """, (sym,))
            news = rows(cur, """
                SELECT event_time, ingested_at, source_api, headline, summary, publisher, url, sentiment
                FROM news WHERE symbol = %s ORDER BY event_time DESC LIMIT 30
            """, (sym,))
            sentiments = rows(cur, """
                SELECT d, sent_1d, sent_5d, news_cnt_5d FROM features
                WHERE symbol = %s ORDER BY d DESC LIMIT 100
            """, (sym,))
    except Exception as exc:
        raise db_error(exc) from None
    return clean({"stock": s, "history": list(reversed(history)),
                  "articles": news, "sentiment_history": list(reversed(sentiments)),
                  "source_note": "News from Finnhub/Alpaca scored by the backend (VADER or optional Cortex). Google Play reviews aren't loaded into this pipeline."})


@app.get("/api/performance")
def performance():
    path = ROOT / "results" / "metrics.json"
    if not path.is_file():
        return {"status": "unavailable", "models": [], "reason": "Run python -m src.train first"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "unavailable", "models": [], "reason": "Cannot read metrics.json"}
    models = []
    for key, name in (("mu_ridge", "Ridge regression"), ("mu_mlp", "Neural network"),
                      ("mu_avg", "Standardized model ensemble")):
        if key in data:
            m = data[key]
            models.append({"name": name, "hit_rate": m.get("hit_rate"), "ic_mean": m.get("ic_mean"),
                           "n_rows": m.get("n_rows"), "n_days": m.get("n_days")})
    # Non-overlapping, cost-aware backtest periods saved by src.train.
    # Only real computed results are displayed; no synthetic chart paths.
    series = []
    curve_path = ROOT / "results" / "backtest_curve.csv"
    if curve_path.is_file():
        try:
            with curve_path.open(newline="", encoding="utf-8") as handle:
                cumulative = {key: 1.0 for key in ("ridge", "mlp", "agreement", "spy")}
                for record in csv.DictReader(handle):
                    point = {"date": record.get("d")}
                    for key in cumulative:
                        value = float(record[key])
                        if not math.isfinite(value):
                            raise ValueError("non-finite return")
                        cumulative[key] *= 1.0 + value
                        point[key] = cumulative[key] - 1.0
                    series.append(point)
        except (OSError, KeyError, ValueError):
            series = []
    return clean({"status": "available", "models": models, "meta": data.get("meta"),
                  "backtest": data.get("backtest", {}), "curve": series})


@app.get("/api/replay")
def replay(symbol: str = Query(...), selected_date: date = Query(...)):
    sym = require_symbol(symbol)
    # The database does not retain immutable historical model/feature source snapshots,
    # so NO selected date can be presented as a verified point-in-time backtest.
    # We only report availability and whether a retrospective prediction exists.
    try:
        with tiger() as conn, conn.cursor() as cur:
            found = rows(cur, """
                SELECT asof, model, created_at FROM predictions
                WHERE symbol = %s AND asof <= %s
                ORDER BY asof DESC, created_at DESC LIMIT 2
            """, (sym, selected_date))
    except Exception as exc:
        raise db_error(exc) from None
    status = "retrospective_only" if found else "insufficient_data"
    return clean({"ticker": sym, "selected_date": selected_date, "status": status,
                  "latest_research_date": found[0]["asof"] if found else None,
                  "eligible_reviews": None, "verified_forecast": None, "realized_return": None,
                  "reason": ("Stored estimates lack a fully auditable point-in-time source and training history; historical performance is withheld."
                             if found else "No model predictions stored on or before the selected date.")})
