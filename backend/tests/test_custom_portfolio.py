"""Unit tests for Custom Portfolio Rebalancer & Retail Investor Shield."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.portfolio.custom_engine import (
    calculate_health_score,
    get_sector,
    score_headline,
    simulate_life_stress_tests,
    run_custom_portfolio_analysis,
)
import numpy as np


def test_custom_portfolio_presets():
    with TestClient(app) as client:
        res = client.get("/api/portfolio/custom/presets")
        assert res.status_code == 200
        data = res.json()
        assert "presets" in data
        assert len(data["presets"]) >= 4
        assert any(p["id"] == "mag_seven" for p in data["presets"])


def test_custom_portfolio_analysis_endpoint():
    payload = {
        "positions": [
            {"ticker": "NVDA", "weight": 0.40},
            {"ticker": "AAPL", "weight": 0.35},
            {"ticker": "JPM", "weight": 0.25},
        ],
        "total_capital": 20000.0,
        "currency": "USD",
    }
    with TestClient(app) as client:
        res = client.post("/api/portfolio/custom/analyze", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["status"] == "success"
        assert data["currency"] == "USD"
        assert data["currency_symbol"] == "$"
        assert data["total_capital"] == 20000.0

        # Health score validation
        assert "health_score" in data
        assert 0 <= data["health_score"]["before"]["score"] <= 100
        assert 0 <= data["health_score"]["after"]["score"] <= 100

        # Orders validation
        orders = data["execution_orders"]
        assert len(orders) == 3
        total_target_pct = sum(o["target_weight_pct"] for o in orders)
        assert abs(total_target_pct - 100.0) < 1.0  # Normalized to ~100%

        # Stress testing validation
        stress = data["stress_scenarios"]
        assert len(stress) >= 4
        assert any("Recession" in s["name"] for s in stress)

        # Scored news feed validation
        assert "scored_news_feed" in data
        assert len(data["scored_news_feed"]) >= 3
        first_news = data["scored_news_feed"][0]
        assert "fud_type" in first_news
        assert "impact_decile" in first_news


def test_custom_portfolio_inr_currency():
    payload = {
        "positions": [
            {"ticker": "MSFT", "weight": 0.50},
            {"ticker": "KO", "weight": 0.50},
        ],
        "total_capital": 500000.0,
        "currency": "INR",
    }
    with TestClient(app) as client:
        res = client.post("/api/portfolio/custom/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["currency"] == "INR"
        assert data["currency_symbol"] == "₹"
        assert "₹" in data["volatility_comparison"]["daily_var_95_cash_before"]


def test_custom_portfolio_validation_errors():
    with TestClient(app) as client:
        # Less than 2 positions
        res = client.post("/api/portfolio/custom/analyze", json={
            "positions": [{"ticker": "AAPL", "weight": 1.0}],
            "total_capital": 1000.0
        })
        assert res.status_code == 422


def test_health_score_calculation():
    weights = np.array([0.5, 0.5])
    vol = 0.20
    signals = np.array([0.1, -0.1])
    sectors = ["Technology", "Financials"]
    health = calculate_health_score(weights, vol, signals, sectors)
    assert 0 <= health["score"] <= 100
    assert health["rating"] in ["Resilient & Shielded", "Moderate Risk", "High Hazard Zone"]


def test_score_headline_fud_logic():
    # Noise/Panic test
    res = score_headline("Random market commentary speculates on tech stocks falling slightly", "AAPL")
    assert "fud_type" in res
    assert "event_class" in res
    assert "sentiment_score" in res
