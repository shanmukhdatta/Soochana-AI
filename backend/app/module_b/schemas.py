"""Strict, versioned schema for Module B hypothetical positions and shocks."""
from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from app.schemas import AnalyzeRequest

STRICT = ConfigDict(extra='forbid', allow_inf_nan=False)

class Loan(BaseModel):
    model_config = STRICT
    type: Literal['loan'] = 'loan'
    id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=150)
    sector: str = Field(min_length=1, max_length=80)
    ticker: str | None = None
    ead_usd: float = Field(gt=0, le=1e10)
    pd_12m: float = Field(ge=0, le=1)
    lgd: float = Field(ge=0, le=1)

class Bond(BaseModel):
    model_config = STRICT
    type: Literal['bond'] = 'bond'
    id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=150)
    sector: str = Field(min_length=1, max_length=80)
    ticker: str | None = None
    market_value_usd: float = Field(gt=0, le=1e10)
    modified_duration: float = Field(ge=0, le=50)
    spread_duration: float = Field(ge=0, le=50)
    convexity: float = Field(ge=0, le=2000)

class Swap(BaseModel):
    model_config = STRICT
    type: Literal['swap'] = 'swap'
    id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=150)
    sector: str = Field(min_length=1, max_length=80)
    ticker: str | None = None
    notional_usd: float = Field(gt=0, le=1e10)
    dv01_usd_per_bp: float = Field(gt=0, le=1e7)
    side: Literal['pay_fixed','receive_fixed']

class Option(BaseModel):
    model_config = STRICT
    type: Literal['option'] = 'option'
    id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    name: str = Field(min_length=1, max_length=150)
    sector: str = Field(min_length=1, max_length=80)
    ticker: str | None = None
    underlying_price_usd: float = Field(gt=0, le=1e6)
    strike_usd: float = Field(gt=0, le=1e6)
    time_years: float = Field(gt=0, le=30)
    implied_volatility: float = Field(gt=0, le=5)
    risk_free_rate: float = Field(ge=-0.2, le=1)
    dividend_yield: float = Field(ge=0, le=1)
    kind: Literal['call','put']
    side: Literal['long','short']
    contracts: float = Field(gt=0, le=1e6)
    contract_multiplier: float = Field(gt=0, le=1e4)

Position = Annotated[Union[Loan, Bond, Swap, Option], Field(discriminator='type')]

class Portfolio(BaseModel):
    model_config = STRICT
    name: str = Field(min_length=1, max_length=120)
    currency: Literal['USD'] = 'USD'
    portfolio_version: str = 'synthetic-v1'
    positions: list[Position] = Field(min_length=1, max_length=250)

    @model_validator(mode='after')
    def ids_unique(self):
        ids = [p.id for p in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError('Every position.id must be unique')
        return self

class Shocks(BaseModel):
    model_config = STRICT
    rate_bps: float = Field(default=0, ge=-2000, le=2000)
    spread_bps: float = Field(default=0, ge=-2000, le=3000)
    equity_pct: float = Field(default=0, ge=-99.9, le=500)
    volatility_points: float = Field(default=0, ge=-5, le=5)
    pd_multiplier: float = Field(default=1, ge=0, le=100)

class Scenario(BaseModel):
    model_config = STRICT
    name: str = Field(min_length=1, max_length=120)
    event_class: Literal['Geopolitical','Macroeconomic','Credit Event','Merger / Acquisition','Product Launch','Custom'] = 'Custom'
    severity: Literal['mild','moderate','severe','custom'] = 'custom'
    shocks: Shocks
    issuer_shocks: dict[str, Shocks] = Field(default_factory=dict)
    sector_shocks: dict[str, Shocks] = Field(default_factory=dict)
    assumption_source: str = 'Illustrative user-configurable stress assumptions; not ML-predicted shocks'

    @field_validator('issuer_shocks', 'sector_shocks')
    @classmethod
    def normalize_and_validate_override_keys(cls, v, info):
        if len(v)>100:
            raise ValueError('Too many overrides')
        out = {}
        for key, shock in v.items():
            # Issuer tickers are case-insensitive; sector names follow the exact
            # portfolio taxonomy. Never silently drop an override key.
            canonical = key.strip().upper() if info.field_name == 'issuer_shocks' else key.strip()
            if not canonical or len(canonical)>80:
                raise ValueError('Invalid override key')
            if canonical in out:
                raise ValueError(f'Duplicate override key after normalization: {canonical}')
            out[canonical] = shock
        return out

class StressRequest(BaseModel):
    model_config = STRICT
    scenario: Scenario
    portfolio: Portfolio | None = None

class StressFromNewsRequest(BaseModel):
    model_config = STRICT
    news: AnalyzeRequest
    severity: Literal['auto','mild','moderate','severe'] = 'auto'
    run_stress: bool = False
    portfolio: Portfolio | None = None

class ScaleChange(BaseModel):
    model_config = STRICT
    position_id: str = Field(min_length=1, max_length=80)
    scale: float = Field(ge=0, le=2)

class WhatIfRequest(BaseModel):
    model_config = STRICT
    scenario: Scenario
    changes: list[ScaleChange] = Field(min_length=1, max_length=250)
    portfolio: Portfolio | None = None

    @model_validator(mode='after')
    def unique_changes(self):
        ids=[x.position_id for x in self.changes]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate scale changes')
        return self

class ReverseStressRequest(BaseModel):
    model_config = STRICT
    scenario: Scenario
    loss_budget_usd: float = Field(gt=0, le=1e10)
    max_multiplier: float = Field(default=4.0, gt=0, le=10)
    steps: int = Field(default=81, ge=10, le=401)
    portfolio: Portfolio | None = None
