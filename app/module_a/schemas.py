"""Module A API schemas. Incoming scores are *predictions*, never realized returns."""
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EventClass=Literal['Geopolitical','Macroeconomic','Credit Event','Merger / Acquisition','Product Launch']
Source=Literal['financial_news','social_media','manual','other']

class Signal(BaseModel):
    model_config=ConfigDict(extra='forbid')
    ticker: str
    published_at: datetime
    source: Source = 'financial_news'
    headline: str = Field(min_length=5,max_length=10000)
    event_id: str | None = Field(default=None,max_length=200)
    sentiment_score: float = Field(ge=-1,le=1,allow_inf_nan=False)
    impact_score: int = Field(ge=1,le=10)
    event_class: EventClass
    event_confidence: float = Field(ge=0,le=1,allow_inf_nan=False)

    @field_validator('ticker')
    @classmethod
    def normal_ticker(cls,v):
        return v.strip().upper()

    @field_validator('published_at')
    @classmethod
    def timezone_mandatory(cls,v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError('published_at needs timezone')
        return v.astimezone(timezone.utc)

class RebalanceRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    as_of: datetime
    current_weights: dict[str,float] | None = None
    signals: list[Signal]=Field(default_factory=list,max_length=200)
    strategy: Literal['sentiment_only','sentiment_impact','full_event_aware']='full_event_aware'

    @field_validator('as_of')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError('as_of needs timezone')
        return v.astimezone(timezone.utc)

class RebalanceFromNewsRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    as_of: datetime
    current_weights: dict[str,float] | None = None
    articles: list['AnalyzeRequest']=Field(min_length=1,max_length=20)
    strategy: Literal['sentiment_only','sentiment_impact','full_event_aware']='full_event_aware'

    @field_validator('as_of')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError('as_of needs timezone')
        return v.astimezone(timezone.utc)

from app.schemas import AnalyzeRequest
RebalanceFromNewsRequest.model_rebuild()

class WeightChange(BaseModel):
    ticker: str
    before: float
    after: float
    change_percentage_points: float
    signal: float
    explanation: str

class RebalanceResponse(BaseModel):
    as_of: datetime
    strategy: str
    current_weights: dict[str,float]
    target_weights: dict[str,float]
    one_way_turnover: float
    estimated_transaction_cost_fraction: float
    signal_counts: dict[str,int]
    changes: list[WeightChange]
    warnings: list[str]=Field(default_factory=list)

class BacktestRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    dataset_id: str = Field(default='holdout',pattern=r'^[a-zA-Z0-9_-]{1,40}$')
    start_date: str | None=None
    end_date: str | None=None
    strategy: Literal['full_event_aware','sentiment_impact','sentiment_only']='full_event_aware'

    @model_validator(mode='after')
    def check_dates(self):
        from datetime import date
        for x in [self.start_date,self.end_date]:
            if x:date.fromisoformat(x)
        if self.start_date and self.end_date and self.start_date>self.end_date:
            raise ValueError('start_date after end_date')
        return self
