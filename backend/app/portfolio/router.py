"""FastAPI router for Custom Portfolio Rebalancer & Retail Investor Shield."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.portfolio.custom_engine import run_custom_portfolio_analysis

router = APIRouter(prefix="/api/portfolio/custom", tags=["Custom Portfolio"])


class PositionItem(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, NVDA, JPM)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Portfolio allocation weight between 0.0 and 1.0")


class CustomPortfolioRequest(BaseModel):
    positions: List[PositionItem] = Field(..., min_length=2, max_length=25, description="List of 2 to 25 stock positions")
    total_capital: Optional[float] = Field(default=10000.0, ge=100.0, description="Total capital invested in currency units")
    currency: Optional[str] = Field(default="USD", description="Currency symbol/code: USD, INR, EUR, GBP")


@router.get("/presets")
def get_portfolio_presets():
    """Return pre-configured institutional and retail portfolio baskets for quick exploration."""
    return {
        "presets": [
            {
                "id": "mag_seven",
                "name": "Big Tech & AI Leaders (Mag 7)",
                "description": "High growth, high beta tech giants powering generative AI and cloud infrastructure.",
                "total_capital": 10000.0,
                "currency": "USD",
                "positions": [
                    {"ticker": "NVDA", "weight": 0.25},
                    {"ticker": "MSFT", "weight": 0.20},
                    {"ticker": "AAPL", "weight": 0.20},
                    {"ticker": "AMZN", "weight": 0.15},
                    {"ticker": "GOOGL", "weight": 0.10},
                    {"ticker": "META", "weight": 0.10},
                ],
            },
            {
                "id": "balanced_core",
                "name": "All-Weather Balanced Core",
                "description": "Resilient cross-sector mix of technology, financials, consumer essentials, and healthcare.",
                "total_capital": 25000.0,
                "currency": "USD",
                "positions": [
                    {"ticker": "MSFT", "weight": 0.25},
                    {"ticker": "JPM", "weight": 0.20},
                    {"ticker": "PG", "weight": 0.20},
                    {"ticker": "JNJ", "weight": 0.15},
                    {"ticker": "XOM", "weight": 0.10},
                    {"ticker": "COST", "weight": 0.10},
                ],
            },
            {
                "id": "defensive_dividend",
                "name": "Defensive Dividend & Stability",
                "description": "Low-beta, high cash-flow companies designed to weather recessions and inflation spikes.",
                "total_capital": 50000.0,
                "currency": "USD",
                "positions": [
                    {"ticker": "KO", "weight": 0.25},
                    {"ticker": "PG", "weight": 0.25},
                    {"ticker": "JNJ", "weight": 0.20},
                    {"ticker": "PEP", "weight": 0.15},
                    {"ticker": "WMT", "weight": 0.15},
                ],
            },
            {
                "id": "india_global_tech",
                "name": "Global Tech & Indian Inflows",
                "description": "Top-tier institutional portfolio tailored for cross-border and emerging tech investors.",
                "total_capital": 100000.0,
                "currency": "INR",
                "positions": [
                    {"ticker": "AAPL", "weight": 0.30},
                    {"ticker": "MSFT", "weight": 0.25},
                    {"ticker": "NVDA", "weight": 0.25},
                    {"ticker": "GOOGL", "weight": 0.20},
                ],
            },
        ]
    }


@router.post("/analyze")
def analyze_custom_portfolio(payload: CustomPortfolioRequest, request: Request):
    """Run dynamic live news scoring, covariance calculation, convex rebalancing, and health analytics."""
    pipeline = getattr(request.app.state, "pipeline", None)
    positions_data = [p.model_dump() for p in payload.positions]
    
    try:
        results = run_custom_portfolio_analysis(
            positions=positions_data,
            total_capital=payload.total_capital or 10000.0,
            currency=payload.currency or "USD",
            pipeline=pipeline,
        )
        return results
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Custom portfolio analysis error: {exc}")
