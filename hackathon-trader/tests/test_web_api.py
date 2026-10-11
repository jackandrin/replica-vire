"""Read-only dashboard API unit tests; no live TigerData or API keys needed."""
from datetime import date, datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import web_api


def fake_snapshot():
    return {
        "stocks": [{"ticker": "NFLX", "company": "Netflix", "sector": "Communication",
                    "price": 100.0, "price_date": "2026-10-09", "day_change_pct": 1.0,
                    "sentiment": -0.1, "sentiment_1d": 0.0, "news_count_5d": 6,
                    "feature_date": "2026-10-09", "prediction_date": "2026-10-09",
                    "ridge_excess_pct": 2.0, "mlp_excess_pct": 3.0,
                    "mean_excess_pct": 2.5, "spread_pp": 1.0,
                    "price_history": [{"date": "2026-10-09", "close": 100.0}]}],
        "benchmark": "SPY", "sentiment_source": "financial_news",
        "prediction_target": "five-day excess return vs SPY", "review_data_connected": False,
    }


def test_overview_labels_target_and_source():
    with patch.object(web_api, "load_snapshot", side_effect=fake_snapshot):
        response = TestClient(web_api.app).get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_source"] == "financial_news"
    assert "excess return" in data["prediction_target"]
    assert not data["review_data_connected"]
    assert data["stocks"][0]["mean_excess_pct"] == 2.5


def test_no_unsupported_symbol():
    response = TestClient(web_api.app).get("/api/stocks/ZZZZINVALID")
    assert response.status_code == 404  # Not present in the ticker universe


def test_clean_removes_nan_and_dates():
    result = web_api.clean({"value": float("nan"), "date": date(2026, 10, 10), "numbers": [1, float("inf")]})
    assert result == {"value": None, "date": "2026-10-10", "numbers": [1, None]}


def test_read_only_api_methods():
    response = TestClient(web_api.app).post("/api/overview")
    assert response.status_code == 405
