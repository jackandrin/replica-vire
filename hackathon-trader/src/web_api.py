"""JSON API for ReplicaVire's React frontend.

Run from hackathon-trader/: python -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
Uses the EXISTING TigerData tables and model evaluation artifacts, not demo values.
No Alpaca trading calls, API tokens, database URLs, or secrets are returned.
Write endpoints (settings, alert execute/dismiss) require the passphrase session cookie.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
from datetime import date, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .common import ROOT, cfg, tiger
from .sectors import GICS_SECTORS, SYMBOLS

app = FastAPI(title="ReplicaVire Research API", version="0.2.0")

# Local development: Vite proxies /api, so CORS is normally unnecessary.
# If using a separate frontend origin, explicitly set FRONTEND_ORIGIN in the server environment.
if os.getenv("FRONTEND_ORIGIN"):
    app.add_middleware(CORSMiddleware, allow_origins=[os.environ["FRONTEND_ORIGIN"]],
                       allow_credentials=True, allow_methods=["GET", "POST", "PUT"], allow_headers=["*"])

# v2 model: 15 per-family regressors -> fusion neural network -> (mu, sigma) per horizon (docs/V2_CONTRACT.md).
HORIZONS = ["6h", "12h", "18h", "1d", "3d", "5d"]
REGRESSOR_LABELS = {
    "reversal": "Short-term reversal", "overnight_intraday": "Overnight vs intraday",
    "intraday_seasonality": "Intraday seasonality", "market_momentum": "Market momentum",
    "sector_leadlag": "Sector lead-lag", "abnormal_volume": "Abnormal volume", "volatility": "Volatility state",
    "oscillators": "RSI / Bollinger", "vwap_ma_distance": "VWAP & moving averages",
    "trend": "Trend (MACD, momentum)", "high_52w": "52-week high", "earnings": "Earnings event",
    "analyst": "Analyst ratings", "short_selling": "Short selling", "insider": "Insider trading",
}
PREDICTION_TARGET = "Raw return over 6h–5d horizons (fusion neural network)"


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


def table_exists(cur, name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
    return bool(cur.fetchone()[0])


def parse_json(value):
    """JSON columns may arrive as text or already decoded (jsonb); bad/missing => None."""
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def day_of(value):
    return value.date() if isinstance(value, datetime) else value


def daily_bars_source(cur):
    """(table, time column): v2 bars_1d when present, else the legacy daily bars table."""
    return ("bars_1d", "d") if table_exists(cur, "bars_1d") else ("bars", "ts")


def ordered_keys(data: dict) -> list:
    """Canonical regressor-family order first, unknown keys after."""
    return [k for k in REGRESSOR_LABELS if k in data] + [k for k in data if k not in REGRESSOR_LABELS]


def regressor_list(raw):
    """{family: yhat_pct} -> [{key, label, yhat_pct}]."""
    data = parse_json(raw)
    if not isinstance(data, dict):
        return None
    return [{"key": k, "label": REGRESSOR_LABELS.get(k, k), "yhat_pct": num(data[k])} for k in ordered_keys(data)]


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
            table, col = daily_bars_source(cur)
            bar_rows = rows(cur, f"""
                SELECT symbol, ts, close FROM (
                    SELECT symbol, {col} AS ts, close,
                           row_number() OVER (PARTITION BY symbol ORDER BY {col} DESC) AS n
                    FROM {table} WHERE symbol = ANY(%s)
                ) x WHERE n <= 12 ORDER BY symbol, ts DESC
            """, (symbols,))
            # SELECT * so a signals table that predates the v2 JSON columns still works (missing => null).
            signal_rows = rows(cur, """
                SELECT * FROM signals WHERE asof = (SELECT max(asof) FROM signals) AND symbol = ANY(%s)
            """, (symbols,)) if table_exists(cur, "signals") else []
            has_sentiment = table_exists(cur, "sentiment_events")
            sentiment_rows = rows(cur, """
                WITH latest AS (
                    SELECT DISTINCT ON (symbol) symbol, published_at, level
                    FROM sentiment_events WHERE symbol = ANY(%s) AND level IS NOT NULL
                    ORDER BY symbol, published_at DESC
                )
                SELECT l.symbol, l.published_at, l.level,
                       (SELECT p.level FROM sentiment_events p
                        WHERE p.symbol = l.symbol AND p.level IS NOT NULL
                          AND p.published_at <= now() - interval '24 hours'
                        ORDER BY p.published_at DESC LIMIT 1) AS level_24h_ago
                FROM latest l
            """, (symbols,)) if has_sentiment else []
            count_rows = rows(cur, """
                SELECT symbol, count(*) AS n FROM sentiment_events
                WHERE symbol = ANY(%s) AND published_at >= now() - interval '5 days'
                GROUP BY symbol
            """, (symbols,)) if has_sentiment else []
    except Exception as exc:
        raise db_error(exc) from None

    prices = {}
    for rec in bar_rows:
        prices.setdefault(rec["symbol"], []).append(rec)
    signals = {r["symbol"]: r for r in signal_rows}
    sentiment = {r["symbol"]: r for r in sentiment_rows}
    counts = {r["symbol"]: int(r["n"]) for r in count_rows}
    stocks = []
    for sym in symbols:
        hist = prices.get(sym, [])
        last, prev = (hist + [None, None])[:2]
        comp, sector = SYMBOLS.get(sym, (sym, "Other"))
        sg = signals.get(sym) or {}
        se = sentiment.get(sym) or {}
        level, before = num(se.get("level")), num(se.get("level_24h_ago"))
        stocks.append({
            "ticker": sym, "company": comp, "sector": sector,
            "price": float(last["close"]) if last and last["close"] is not None else None,
            "price_date": day_of(last["ts"]) if last else None,
            "day_change_pct": 100 * (float(last["close"]) / float(prev["close"]) - 1)
            if last and prev and last["close"] is not None and prev["close"] else None,
            # LLM sentiment agent: level is 0-100; trend = latest level minus the level as of 24h ago.
            "sentiment_level": level,
            "sentiment_trend": level - before if level is not None and before is not None else None,
            "sentiment_date": se.get("published_at"),
            "news_count_5d": counts.get(sym, 0) if has_sentiment else None,
            # Fusion-NN decision layer: the chosen horizon's RAW expected return in percent.
            "direction": sg.get("direction"), "expected_pct": sg.get("expected_pct"),
            "timeframe": sg.get("timeframe"), "height": sg.get("height"),
            "suggested_usd": sg.get("suggested_usd"), "reason": sg.get("reason"),
            "signal_date": sg.get("asof"), "signal_ts": sg.get("ts"),
            "horizons": parse_json(sg.get("horizons")),
            "price_history": [{"date": day_of(r["ts"]), "close": float(r["close"])}
                              for r in reversed(hist) if r["close"] is not None],
        })
    return clean({"stocks": stocks, "sectors": GICS_SECTORS, "benchmark": cfg()["benchmark"],
                  "sentiment_source": "financial_news", "prediction_target": PREDICTION_TARGET,
                  "horizon_order": HORIZONS, "review_data_connected": False})


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
    events, levels, articles, latest = [], [], [], {}
    try:
        with tiger() as conn, conn.cursor() as cur:
            table, col = daily_bars_source(cur)
            history = rows(cur, f"""
                SELECT {col} AS ts, open, high, low, close, volume FROM {table}
                WHERE symbol = %s ORDER BY {col} DESC LIMIT 150
            """, (sym,))
            if table_exists(cur, "signals"):
                found = rows(cur, "SELECT * FROM signals WHERE symbol = %s ORDER BY asof DESC LIMIT 1", (sym,))
                latest = found[0] if found else {}
            has_docs = table_exists(cur, "documents")
            if table_exists(cur, "sentiment_events"):
                doc_cols = ("d.title AS doc_title, d.url AS doc_url, d.source AS doc_source, d.doc_type"
                            if has_docs else
                            "NULL AS doc_title, NULL AS doc_url, NULL AS doc_source, NULL AS doc_type")
                doc_join = "LEFT JOIN documents d ON d.doc_id = s.doc_id" if has_docs else ""
                events = rows(cur, f"""
                    SELECT s.published_at, s.direction, s.delta, s.level, s.novelty, s.materiality,
                           s.drivers, s.summary, {doc_cols}
                    FROM sentiment_events s {doc_join}
                    WHERE s.symbol = %s ORDER BY s.published_at DESC LIMIT 30
                """, (sym,))
                levels = rows(cur, """
                    SELECT published_at, level FROM sentiment_events
                    WHERE symbol = %s AND level IS NOT NULL AND published_at >= now() - interval '90 days'
                    ORDER BY published_at
                """, (sym,))
            if has_docs:
                articles = rows(cur, """
                    SELECT published_at, source, doc_type, title, left(body, 300) AS body, url, NULL AS publisher
                    FROM documents WHERE symbol = %s ORDER BY published_at DESC LIMIT 30
                """, (sym,))
            elif table_exists(cur, "news"):  # pre-v2 fallback
                articles = rows(cur, """
                    SELECT event_time AS published_at, source_api AS source, 'news' AS doc_type,
                           headline AS title, left(summary, 300) AS body, url, publisher
                    FROM news WHERE symbol = %s ORDER BY event_time DESC LIMIT 30
                """, (sym,))
    except Exception as exc:
        raise db_error(exc) from None
    for e in events:
        drivers = parse_json(e["drivers"])
        e["drivers"] = [str(x) for x in drivers] if isinstance(drivers, list) else []
    drivers = parse_json(latest.get("drivers"))
    return clean({"stock": s, "history": list(reversed(history)),
                  "horizons": parse_json(latest.get("horizons")),
                  "regressors": regressor_list(latest.get("regressors")),
                  "drivers": drivers if isinstance(drivers, list) else None,
                  "sentiment_events": events,
                  "sentiment_history": [{"published_at": r["published_at"], "level": num(r["level"])} for r in levels],
                  "articles": articles,
                  "source_note": "News from Finnhub/Alpaca plus SEC EDGAR filings, read by an LLM sentiment agent "
                                 "(Snowflake Cortex). Google Play reviews aren't loaded into this pipeline."})


def _block(value) -> dict:
    return value if isinstance(value, dict) else {}


def published_metrics():
    """Metrics of the newest run published to Tiger Data (src.publish_models), or None."""
    if not os.getenv("TIGER_DATABASE_URL", "").strip():
        return None
    try:
        with tiger() as conn, conn.cursor() as cur:
            if not table_exists(cur, "model_runs"):
                return None
            cur.execute("SELECT metrics FROM model_runs ORDER BY created_at DESC LIMIT 1")
            r = cur.fetchone()
    except Exception:  # noqa: BLE001  fall back to a local metrics.json
        return None
    if not r:
        return None
    return r[0] if isinstance(r[0], dict) else json.loads(r[0])


@app.get("/api/performance")
def performance():
    data = published_metrics()
    path = ROOT / "results" / "metrics.json"
    if data is None and not path.is_file():
        return {"status": "unavailable", "horizons": [], "regressors": [],
                "reason": "No published model yet: run python -m src.train, then python -m src.publish_models"}
    if data is None:  # local development without a published run
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
    if not isinstance(data, dict):
        return {"status": "unavailable", "horizons": [], "regressors": [], "reason": "Cannot read metrics.json"}
    found = _block(data.get("horizons"))
    horizons = [{"horizon": h, "nn": _block(_block(found[h]).get("nn")), "baseline": _block(_block(found[h]).get("baseline"))}
                for h in HORIZONS + [k for k in found if k not in HORIZONS] if h in found]
    regs = _block(data.get("regressors"))
    regressors = [{"key": k, "label": REGRESSOR_LABELS.get(k, k), "ic_mean": num(_block(regs[k]).get("ic_mean")),
                   "hit_rate": num(_block(regs[k]).get("hit_rate"))} for k in ordered_keys(regs)]
    pairs = [{"a": p[0], "b": p[1], "a_label": REGRESSOR_LABELS.get(p[0], p[0]),
              "b_label": REGRESSOR_LABELS.get(p[1], p[1]), "corr": num(p[2])}
             for p in data.get("redundant_pairs") or [] if isinstance(p, (list, tuple)) and len(p) >= 3]
    # src.train stores the curve as cumulative returns already; only real computed points are passed through.
    curve = [{"date": p.get("date"), "nn": num(p.get("nn")), "baseline": num(p.get("baseline")), "spy": num(p.get("spy"))}
             for p in data.get("curve") or [] if isinstance(p, dict)]
    return clean({"status": "available", "horizons": horizons, "regressors": regressors,
                  "redundant_pairs": pairs, "backtest": _block(data.get("backtest")),
                  "curve": curve, "meta": data.get("meta")})


@app.get("/api/replay")
def replay(symbol: str = Query(...), selected_date: date = Query(...)):
    sym = require_symbol(symbol)
    # The database does not retain immutable historical model/feature source snapshots,
    # so NO selected date can be presented as a verified point-in-time backtest.
    # We only report availability and whether a retrospective signal exists.
    try:
        with tiger() as conn, conn.cursor() as cur:
            found = rows(cur, """
                SELECT asof FROM signals WHERE symbol = %s AND asof <= %s
                ORDER BY asof DESC LIMIT 1
            """, (sym, selected_date)) if table_exists(cur, "signals") else []
    except Exception as exc:
        raise db_error(exc) from None
    status = "retrospective_only" if found else "insufficient_data"
    return clean({"ticker": sym, "selected_date": selected_date, "status": status,
                  "latest_research_date": found[0]["asof"] if found else None,
                  "eligible_reviews": None, "verified_forecast": None, "realized_return": None,
                  "reason": ("Stored signals lack a fully auditable point-in-time source and training history; historical performance is withheld."
                             if found else "No model signals stored on or before the selected date.")})


# ------------------------------------------------------------------ single-user auth (passphrase -> signed cookie)
# Every write endpoint below requires the session cookie. Set DASHBOARD_PASSPHRASE (and SESSION_SECRET) in the
# server environment. Writes only touch TigerData rows; the worker (src/autotrade.py) holds the Alpaca keys.
SESSION_COOKIE = "rv_session"
SESSION_HOURS = 12


def _secret() -> bytes:
    s = os.getenv("SESSION_SECRET", "").strip() or os.getenv("DASHBOARD_PASSPHRASE", "").strip()
    if not s:
        raise HTTPException(status_code=503, detail="DASHBOARD_PASSPHRASE isn't configured on the API server")
    return hashlib.sha256(("rv:" + s).encode()).digest()


def _sign(exp: int) -> str:
    return f"{exp}.{hmac.new(_secret(), str(exp).encode(), hashlib.sha256).hexdigest()}"


def is_authed(request: Request) -> bool:
    tok = request.cookies.get(SESSION_COOKIE, "")
    exp, _, _sig = tok.partition(".")
    if not exp.isdigit() or int(exp) < time.time():
        return False
    try:
        return hmac.compare_digest(tok, _sign(int(exp)))
    except HTTPException:
        return False


def require_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(status_code=401, detail="Log in on the Auto-Trading page first")


class Login(BaseModel):
    passphrase: str


class Settings(BaseModel):
    auto_trading: bool = False
    sectors: list[str] = Field(default_factory=list)
    min_usd: float = Field(50, ge=1)
    max_usd: float = Field(500, ge=1)
    min_height: float = Field(0.5, ge=0)
    direction_mode: Literal["both", "long_only", "short_only"] = "both"
    max_positions: int = Field(10, ge=1, le=100)
    day_trade_min_height: float = Field(2.5, ge=0)


@app.get("/api/session")
def session(request: Request):
    return {"authenticated": is_authed(request)}


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    expected = os.getenv("DASHBOARD_PASSPHRASE", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="DASHBOARD_PASSPHRASE isn't configured on the API server")
    if not hmac.compare_digest(body.passphrase.encode(), expected.encode()):
        time.sleep(1)  # slow down guessing
        raise HTTPException(status_code=401, detail="Wrong passphrase")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(SESSION_COOKIE, _sign(int(time.time()) + SESSION_HOURS * 3600), max_age=SESSION_HOURS * 3600,
                        httponly=True, samesite="strict", secure=secure, path="/")
    return {"authenticated": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}


def _defaults() -> dict:
    d = dict(cfg().get("autotrade", {}))
    return Settings(**{k: v for k, v in d.items() if k in Settings.model_fields}).model_dump()


@app.get("/api/settings")
def get_settings():
    out = _defaults()
    try:
        with tiger() as conn, conn.cursor() as cur:
            if table_exists(cur, "user_settings"):
                cur.execute("SELECT data FROM user_settings WHERE id = 1")
                r = cur.fetchone()
                if r:
                    out.update({k: v for k, v in r[0].items() if k in out})
    except Exception as exc:
        raise db_error(exc) from None
    return out


@app.put("/api/settings")
def put_settings(body: Settings, request: Request):
    require_auth(request)
    if body.min_usd > body.max_usd:
        raise HTTPException(status_code=422, detail="Min $ per trade must be <= max $ per trade")
    bad = [s for s in body.sectors if s not in GICS_SECTORS]
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown sectors: {bad}")
    data = body.model_dump()
    try:
        with tiger() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO user_settings (id, data, updated_at) VALUES (1, %s::jsonb, now()) "
                        "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()", (json.dumps(data),))
    except Exception as exc:
        raise db_error(exc) from None
    return data


@app.get("/api/alerts")
def alerts(since: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    try:
        with tiger() as conn, conn.cursor() as cur:
            if not table_exists(cur, "alerts"):
                return {"alerts": []}
            found = rows(cur, """
                SELECT id, created_at, symbol, action, expected_pct, timeframe, act_by, size_usd, height,
                       reason, mode, status, detail
                FROM alerts WHERE id > %s ORDER BY id DESC LIMIT %s
            """, (since, limit))
    except Exception as exc:
        raise db_error(exc) from None
    return clean({"alerts": found})


def _set_alert(alert_id: int, status: str, request: Request):
    require_auth(request)
    try:
        with tiger() as conn, conn.cursor() as cur:
            cur.execute("UPDATE alerts SET status = %s WHERE id = %s AND status = 'new' RETURNING id", (status, alert_id))
            ok = cur.fetchone()
    except Exception as exc:
        raise db_error(exc) from None
    if not ok:
        raise HTTPException(status_code=409, detail="Alert not found or already handled")
    return {"status": status}


@app.post("/api/alerts/{alert_id}/execute")
def execute_alert(alert_id: int, request: Request):
    # Queues the order; the worker (python -m src.autotrade) places it on Alpaca paper.
    return _set_alert(alert_id, "requested", request)


@app.post("/api/alerts/{alert_id}/dismiss")
def dismiss_alert(alert_id: int, request: Request):
    return _set_alert(alert_id, "dismissed", request)
