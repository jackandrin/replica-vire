"""Live paper portfolio logic and endpoints: offline, no TigerData / Alpaca needed."""
from contextlib import contextmanager
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import portfolio as pf
from src import web_api


def sig(sym, height, mu=1.0, sigma=0.8, direction="long", price=100.0):
    return {"symbol": sym, "direction": direction, "expected_pct": mu, "timeframe": "1d", "height": height,
            "horizons": {"1d": {"mu": mu, "sigma": sigma}}, "price": price, "reason": None}


STOCKS = [sig("AAPL", 2.0), sig("MSFT", 1.8), sig("NVDA", 1.6, sigma=3.0), sig("AMZN", 1.2), sig("TSLA", 1.0),
          sig("META", 0.9), sig("JPM", 0.8), sig("XOM", 1.9, direction="short", mu=-1.0), sig("KO", 0.0, direction="flat", mu=0.0)]


def test_profile_validation():
    ok = pf.normalize_profile(250, "Balanced", "5+ years", ["No preference"])
    assert ok == {"budget": 250.0, "risk": "Balanced", "horizon": "5+ years", "sectors": []}
    for bad in [(5, "Balanced", "5+ years", []), (10**7, "Balanced", "5+ years", []), (100, "YOLO", "5+ years", []),
                (100, "Growth", "forever", []), (100, "Growth", "5+ years", ["Crypto"]), (float("nan"), "Growth", "5+ years", [])]:
        with pytest.raises(pf.PortfolioError) as e:
            pf.normalize_profile(*bad)
        assert e.value.status == 422


def test_selection_is_long_only_and_weights_sum_to_one():
    for risk, count in pf.PICK_COUNT.items():
        sel = pf.select_portfolio(STOCKS, risk, [])
        syms = [t["symbol"] for t in sel["targets"]]
        assert len(syms) == count and "XOM" not in syms and "KO" not in syms
        assert sum(t["weight"] for t in sel["targets"]) == pytest.approx(1.0)


def test_weighting_styles_and_uncertainty_penalty():
    cons = pf.select_portfolio(STOCKS, "Conservative", [])["targets"]
    assert len({round(t["weight"], 6) for t in cons}) == 1  # equal weight
    growth = pf.select_portfolio(STOCKS, "Growth", [])["targets"]
    assert growth[0]["weight"] > 2 * growth[-1]["weight"]
    # NVDA is noisy (sigma 3): Conservative penalises it out of its picks, Growth ignores uncertainty
    assert "NVDA" not in [t["symbol"] for t in cons] and "NVDA" in [t["symbol"] for t in growth]


def test_sector_filter_and_fallback():
    sel = pf.select_portfolio(STOCKS, "Balanced", ["Information Technology", "Communication Services"])
    assert not sel["sector_fallback"]
    assert {t["symbol"] for t in sel["targets"]} <= {"AAPL", "MSFT", "NVDA", "META"}
    thin = pf.select_portfolio(STOCKS, "Balanced", ["Energy"])  # only a short signal there
    assert thin["sector_fallback"] and thin["targets"]


def test_no_long_signals_means_cash():
    flat = [sig("AAPL", 0.0, direction="flat", mu=0.0)]
    assert pf.select_portfolio(flat, "Balanced", [])["targets"] == []


def order(sym, side, notional=None, fq=0.0, fp=None, status="filled", at=0):
    return {"symbol": sym, "side": side, "notional": notional, "filled_qty": fq, "filled_avg_price": fp,
            "status": status, "created_at": datetime(2026, 1, 1, 12, at, tzinfo=timezone.utc)}


def test_ledger_tracks_fills_cash_and_reservations():
    led = pf.ledger([order("AAPL", "buy", 100, 0.5, 200.0), order("MSFT", "buy", 50, 0, None, "new"),
                     order("AAPL", "sell", None, 0.25, 220.0, at=1)], budget=500)
    assert led["qty"] == {"AAPL": pytest.approx(0.25)}
    assert led["cash"] == pytest.approx(500 - 100 + 55)
    assert led["reserved"] == pytest.approx(50)
    assert led["open_symbols"] == {"MSFT"}
    assert led["cost"]["AAPL"] == pytest.approx(50)  # half of the $100 basis was sold
    assert pf.total_value(led, {"AAPL": 240.0}) == pytest.approx(455 + 60)


def test_order_status_helpers():
    assert pf.is_open("new") and pf.is_open("pending_submit") and pf.is_open("partially_filled")
    assert not pf.is_open("filled") and not pf.is_open("error: boom") and not pf.is_open("canceled")


def run_rebalance(orders, targets_stocks, prices, budget=1000.0, risk="Conservative"):
    placed = []
    cur = MagicMock()
    with patch.object(pf, "_orders", return_value=orders), patch.object(pf, "latest_prices", return_value=prices), \
            patch.object(pf, "place_order", side_effect=lambda tc, c, pid, sym, side, **kw: placed.append((sym, side, kw)) or {"ok": 1}):
        n = pf.rebalance(cur, object(), {"id": "p1", "budget": budget, "risk": risk, "sectors": []},
                         date(2026, 1, 2), targets_stocks)
    return n, placed


def test_rebalance_buys_to_target_from_cash():
    stocks = [sig("AAPL", 2.0), sig("MSFT", 1.5)]
    n, placed = run_rebalance([], stocks, {"AAPL": 100.0, "MSFT": 100.0})
    assert n == 2
    assert {(s, side) for s, side, _ in placed} == {("AAPL", "buy"), ("MSFT", "buy")}
    assert sum(kw["notional"] for _, _, kw in placed) <= 1000
    assert all(kw["coid"].startswith("pf-p1-2026-01-02-") for _, _, kw in placed)


def test_rebalance_sells_dropped_names_and_skips_open_orders():
    held = [order("KO", "buy", 500, 5.0, 100.0), order("PG", "buy", 500, 5.0, 100.0)]
    held[1]["status"] = "new"  # PG has an order in flight: leave it alone
    n, placed = run_rebalance(held, [sig("AAPL", 2.0)], {"KO": 100.0, "PG": 100.0, "AAPL": 100.0}, budget=1000)
    sells = [(s, kw["qty"]) for s, side, kw in placed if side == "sell"]
    assert sells == [("KO", pytest.approx(5.0))]
    assert "PG" not in [s for s, _, _ in placed]
    assert any(s == "AAPL" and side == "buy" for s, side, _ in placed)


def test_rebalance_ignores_small_drift():
    held = [order("AAPL", "buy", 1000, 9.9, 100.0)]  # $990 held vs $1000 target: inside the 2% band
    n, placed = run_rebalance(held, [sig("AAPL", 2.0)], {"AAPL": 100.0}, budget=1000)
    assert n == 0 and placed == []


# ------------------------------------------------------------------------------------------------------ endpoints
@contextmanager
def fake_tiger():
    cur = MagicMock()
    cur.__enter__.return_value = cur
    conn = MagicMock()
    conn.cursor.return_value = cur
    yield conn


def client():
    return TestClient(web_api.app)


def test_create_returns_token_once_and_maps_errors():
    view = {"id": "abc", "summary": {"value": 250.0}, "targets": [], "positions": [], "orders": [], "history": [], "notes": []}
    with patch.object(web_api, "tiger", fake_tiger), \
            patch.object(web_api.portfolios, "create_portfolio", return_value=("abc", "tok")) as made, \
            patch.object(web_api.portfolios, "view", return_value=view):
        r = client().post("/api/portfolios", json={"budget": 250, "risk": "Balanced", "horizon": "5+ years", "sectors": []})
    assert r.status_code == 201 and r.json()["token"] == "tok" and r.headers["cache-control"] == "no-store"
    assert made.call_args.args[1]["budget"] == 250
    err = pf.PortfolioError(429, "slow down")
    with patch.object(web_api, "tiger", fake_tiger), patch.object(web_api.portfolios, "create_portfolio", side_effect=err):
        r = client().post("/api/portfolios", json={"budget": 250, "risk": "Balanced", "horizon": "5+ years"})
    assert r.status_code == 429 and r.json()["detail"] == "slow down"


def test_create_rejects_malformed_bodies():
    assert client().post("/api/portfolios", json={"risk": "Balanced"}).status_code == 422
    assert client().post("/api/portfolios", json={"budget": "lots", "risk": "x", "horizon": "y"}).status_code == 422


def test_read_requires_valid_bearer_token():
    assert client().get("/api/portfolios/abc").status_code == 401
    with patch.object(web_api, "tiger", fake_tiger), patch.object(web_api.portfolios, "token_ok", return_value=False):
        r = client().get("/api/portfolios/abc", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 404
    with patch.object(web_api, "tiger", fake_tiger), patch.object(web_api.portfolios, "token_ok", return_value=True), \
            patch.object(web_api.portfolios, "refresh"), \
            patch.object(web_api.portfolios, "view", return_value={"id": "abc", "created_at": datetime(2026, 1, 1)}):
        r = client().get("/api/portfolios/abc", headers={"Authorization": "Bearer right"})
    assert r.status_code == 200 and r.json()["id"] == "abc" and "token" not in r.json()


def test_close_needs_the_token_and_closes():
    assert client().post("/api/portfolios/abc/close").status_code == 401
    with patch.object(web_api, "tiger", fake_tiger), patch.object(web_api.portfolios, "token_ok", return_value=True), \
            patch.object(web_api.portfolios, "close") as closed, \
            patch.object(web_api.portfolios, "view", return_value={"id": "abc", "status": "closed"}):
        r = client().post("/api/portfolios/abc/close", headers={"Authorization": "Bearer right"})
    assert r.status_code == 200 and r.json()["status"] == "closed" and closed.called


def test_credentials_store_only_a_hash():
    pid, token, digest = pf.new_credentials()
    assert digest != token and len(digest) == 64 and len(token) >= 40 and pid
