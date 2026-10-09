"""API contract shared by future Module A and Module B."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator

Source = Literal['financial_news', 'social_media', 'manual', 'other']


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=5, max_length=10000)
    ticker: str = Field(min_length=1, max_length=10)
    published_at: datetime = Field(description='An ISO-8601 timestamp WITH a timezone, e.g. 2026-10-07T14:30:00Z')
    source: Source = 'manual'

    @field_validator('ticker')
    @classmethod
    def valid_ticker(cls, ticker: str) -> str:
        import re
        ticker = ticker.strip().upper()
        if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', ticker):
            raise ValueError('Invalid ticker symbol')
        return ticker

    @field_validator('published_at')
    @classmethod
    def tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError('published_at must contain an explicit timezone')
        return v

    @field_validator('text')
    @classmethod
    def trimmed(cls, text: str) -> str:
        if len(text.strip()) < 5:
            raise ValueError('text must contain at least five non-whitespace characters')
        return text.strip()


class BatchRequest(BaseModel):
    items: list[AnalyzeRequest] = Field(min_length=1, max_length=20)


class SentimentResult(BaseModel):
    score: float = Field(ge=-1, le=1)
    label: Literal['positive', 'negative', 'neutral']


class EventResult(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class ImpactResult(BaseModel):
    predicted_realized_impact: float = Field(ge=0)
    score: int = Field(ge=1, le=10)
    target_definition: str = 'Estimated magnitude of historical abnormal market reaction, not a guaranteed future move'


class MarketContextResult(BaseModel):
    sector: str
    market_regime: str
    stock_feature_date: str
    spy_feature_date: str
    provider: str


class AnalyzeResponse(BaseModel):
    ticker: str
    published_at: datetime
    source: Source
    sentiment: SentimentResult
    event: EventResult
    impact: ImpactResult
    market_context: MarketContextResult
    model_version: str = 'risk-engine-v1'
    warnings: list[str] = Field(default_factory=list)


class BatchResponse(BaseModel):
    results: list[AnalyzeResponse]
    total: int


class HealthResponse(BaseModel):
    status: Literal['ready', 'not_ready']
    model_version: str = 'risk-engine-v1'
    detail: str | None = None
