"""Read-only dashboard API unit tests; no live TigerData or API keys needed."""
import json
from datetime import date, datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import web_api


def fake_snapshot():
    return {
        "stocks": [{"ticker": "NFLX", "company": "Netflix", "sector": "Communication",
                    "price": 100.0, "price_date": "2026-10-09", "day_change_pct": 1.0,
                    "sentiment_level": 62.0, "sentiment_trend": -4.0, "news_count_5d": 6,
                    "direction": "long", "expected_pct": 1.2, "timeframe": "1d", "height": 1.1,
                    "suggested_usd": 250.0, "reason": "Fusion net expects +1.2% over 1d",
                    "signal_date": "2026-10-09", "signal_ts": "2026-10-09T19:45:00+00:00",
                    "horizons": {"1d": {"mu": 1.2, "sigma": 1.1}},
                    "price_history": [{"date": "2026-10-09", "close": 100.0}]}],
        "benchmark": "SPY", "sentiment_source": "financial_news",
        "prediction_target": web_api.PREDICTION_TARGET, "review_data_connected": False,
    }


def test_overview_labels_target_and_source():
    with patch.object(web_api, "load_snapshot", side_effect=fake_snapshot):
        response = TestClient(web_api.app).get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_source"] == "financial_news"
    assert "fusion neural network" in data["prediction_target"]
    assert "excess" not in data["prediction_target"]
    assert not data["review_data_connected"]
    stock = data["stocks"][0]
    assert stock["sentiment_level"] == 62.0 and stock["horizons"]["1d"]["mu"] == 1.2
    assert "mean_excess_pct" not in stock


def test_parse_json_and_regressor_labels():
    assert web_api.parse_json('{"6h": {"mu": 0.1, "sigma": 0.5}}') == {"6h": {"mu": 0.1, "sigma": 0.5}}
    assert web_api.parse_json("not json") is None and web_api.parse_json(None) is None
    assert web_api.parse_json([1, 2]) == [1, 2]  # jsonb arrives already decoded
    regs = web_api.regressor_list('{"insider": 0.02, "reversal": -0.3, "custom": 0.1}')
    assert [r["key"] for r in regs] == ["reversal", "insider", "custom"]
    assert regs[0] == {"key": "reversal", "label": "Short-term reversal", "yhat_pct": -0.3}
    assert len(web_api.REGRESSOR_LABELS) == 15


class FakeCursor:
    """Answers to_regclass() from `tables` and returns daily bars; anything else returns no rows."""
    def __init__(self, tables):
        self.tables, self.result, self.description, self.sql = tables, [], [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, args=()):
        self.sql.append(sql)
        if "to_regclass" in sql:
            self.description, self.result = [("exists",)], [(args[0] in self.tables,)]
        elif "row_number()" in sql:
            self.description = [("symbol",), ("ts",), ("close",)]
            self.result = [("NFLX", date(2026, 10, 9), 101.0), ("NFLX", date(2026, 10, 8), 100.0)]
        else:
            self.description, self.result = [("x",)], []

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class FakeConn:
    def __init__(self, cur):
        self.cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur


def test_snapshot_degrades_without_v2_tables(monkeypatch):
    cur = FakeCursor(tables={"bars_1d"})
    monkeypatch.setattr(web_api, "tiger", lambda: FakeConn(cur))
    monkeypatch.setattr(web_api, "available_symbols", lambda: ["NFLX"])
    stock = web_api.load_snapshot()["stocks"][0]
    assert stock["price"] == 101.0 and stock["price_date"] == "2026-10-09"
    assert round(stock["day_change_pct"], 6) == 1.0
    assert stock["sentiment_level"] is None and stock["sentiment_trend"] is None and stock["news_count_5d"] is None
    assert stock["horizons"] is None and stock["direction"] is None
    joined = " ".join(cur.sql)
    assert "predictions" not in joined and "features" not in joined and "sentiment_events" not in joined


def test_performance_unavailable_without_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(web_api, "ROOT", tmp_path)
    data = TestClient(web_api.app).get("/api/performance").json()
    assert data["status"] == "unavailable" and data["horizons"] == []


def test_performance_reads_v2_metrics(monkeypatch, tmp_path):
    metrics = {
        "horizons": {
            "1d": {"nn": {"ic_mean": 0.03, "hit_rate": 0.52, "coverage_1sigma": 0.66, "n_rows": 900, "n_days": 60},
                   "baseline": {"ic_mean": 0.01, "hit_rate": 0.50, "n_rows": 900, "n_days": 60}},
            "6h": {"nn": {"ic_mean": 0.02, "hit_rate": 0.51, "coverage_1sigma": 0.70, "n_rows": 1000, "n_days": 60},
                   "baseline": {"ic_mean": 0.0, "hit_rate": 0.49}},
        },
        "regressors": {"reversal": {"ic_mean": 0.02, "hit_rate": 0.51}, "oscillators": {"ic_mean": float("nan"), "hit_rate": 0.5}},
        "redundant_pairs": [["oscillators", "reversal", 0.83]],
        "backtest": {"nn": {"total_return": 0.04, "sharpe": 0.9, "max_drawdown": -0.05, "n_periods": 12}},
        "curve": [{"date": "2025-01-02", "nn": 0.0, "baseline": 0.0, "spy": 0.0},
                  {"date": "2025-01-09", "nn": 0.01, "baseline": 0.005, "spy": -0.002}],
        "meta": {"train_dates": ["2020-01-02", "2024-06-28"], "test_dates": ["2025-01-02", "2025-06-30"],
                 "caveat": "Held-out test only.", "folds": 6},
    }
    (tmp_path / "results").mkdir()
    # json.dumps writes NaN, which train.py may emit; the API must turn it into null.
    (tmp_path / "results" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(web_api, "ROOT", tmp_path)
    data = TestClient(web_api.app).get("/api/performance").json()
    assert data["status"] == "available"
    assert [h["horizon"] for h in data["horizons"]] == ["6h", "1d"]  # canonical horizon order
    assert data["horizons"][1]["nn"]["coverage_1sigma"] == 0.66
    assert data["horizons"][1]["baseline"]["hit_rate"] == 0.50
    assert data["regressors"][0] == {"key": "reversal", "label": "Short-term reversal", "ic_mean": 0.02, "hit_rate": 0.51}
    assert data["regressors"][1]["label"] == "RSI / Bollinger" and data["regressors"][1]["ic_mean"] is None
    assert data["redundant_pairs"] == [{"a": "oscillators", "b": "reversal", "a_label": "RSI / Bollinger",
                                        "b_label": "Short-term reversal", "corr": 0.83}]
    assert data["curve"][1] == {"date": "2025-01-09", "nn": 0.01, "baseline": 0.005, "spy": -0.002}
    assert data["backtest"]["nn"]["sharpe"] == 0.9
    assert data["meta"]["caveat"] == "Held-out test only."


def test_performance_rejects_unreadable_metrics(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(web_api, "ROOT", tmp_path)
    data = TestClient(web_api.app).get("/api/performance").json()
    assert data["status"] == "unavailable"


def test_no_unsupported_symbol():
    response = TestClient(web_api.app).get("/api/stocks/ZZZZINVALID")
    assert response.status_code == 404  # Not present in the ticker universe


def test_clean_removes_nan_and_dates():
    result = web_api.clean({"value": float("nan"), "date": date(2026, 10, 10), "numbers": [1, float("inf")]})
    assert result == {"value": None, "date": "2026-10-10", "numbers": [1, None]}


def test_read_only_api_methods():
    response = TestClient(web_api.app).post("/api/overview")
    assert response.status_code == 405


def test_writes_require_login(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSPHRASE", "correct horse")
    c = TestClient(web_api.app)
    assert c.put("/api/settings", json={}).status_code == 401
    assert c.post("/api/alerts/1/execute").status_code == 401
    assert c.post("/api/login", json={"passphrase": "nope"}).status_code == 401
    assert c.get("/api/session").json() == {"authenticated": False}


def test_login_sets_session(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSPHRASE", "correct horse")
    c = TestClient(web_api.app)
    assert c.post("/api/login", json={"passphrase": "correct horse"}).status_code == 200
    assert c.get("/api/session").json() == {"authenticated": True}
    r = c.put("/api/settings", json={"min_usd": 600, "max_usd": 100})
    assert r.status_code == 422  # validated before touching the database


def test_performance_prefers_published_run(monkeypatch):
    published = {"horizons": {"1d": {"nn": {"ic_mean": 0.02}, "baseline": {"ic_mean": 0.01}}}, "regressors": {},
                 "curve": [], "meta": {"trained_at": "v-published"}}
    monkeypatch.setattr(web_api, "published_metrics", lambda: published)
    data = TestClient(web_api.app).get("/api/performance").json()
    assert data["status"] == "available" and data["meta"]["trained_at"] == "v-published"
    assert data["horizons"][0]["nn"]["ic_mean"] == 0.02
