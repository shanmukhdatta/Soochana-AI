"""Module A configuration. Validated at construction, never tuned on final holdout."""
from dataclasses import dataclass
from pathlib import Path
import json

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'module_a_config.json'

@dataclass(frozen=True)
class ModuleAConfig:
    tickers: tuple[str,...]
    sector_map: dict[str,str]
    position_deviation_limit: float
    sector_deviation_limit: float
    one_way_turnover_limit: float
    minimum_trade_turnover: float
    signal_tilt_coefficient: float
    news_lookback_days: int
    signal_half_life_days: float
    evidence_prior_weight: float
    transaction_cost_bps: float
    max_signals_per_request: int
    source_weights: dict[str,float]
    event_multipliers: dict[str,float]

    @classmethod
    def load(cls, path: Path | str=DEFAULT_CONFIG):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        keys=cls.__dataclass_fields__.keys()
        cfg=cls(**{k:(tuple(data[k]) if k=='tickers' else data[k]) for k in keys})
        if not (2<=len(cfg.tickers)<=20) or len(set(cfg.tickers))!=len(cfg.tickers):
            raise ValueError('Require between 2 and 20 distinct index tickers')
        if set(cfg.tickers)!=set(cfg.sector_map):
            raise ValueError('Sector mapping must cover exactly the index constituents')
        if not (0<cfg.position_deviation_limit<1 and 0<cfg.sector_deviation_limit<1):
            raise ValueError('Invalid position or sector limits')
        if not (0<cfg.one_way_turnover_limit<=1 and 0<=cfg.minimum_trade_turnover<cfg.one_way_turnover_limit):
            raise ValueError('Invalid turnover limits')
        if not (0<cfg.signal_tilt_coefficient<=0.2 and cfg.news_lookback_days>=1
                and cfg.signal_half_life_days>0 and 0<cfg.evidence_prior_weight<=10
                and cfg.transaction_cost_bps>=0
                and 1<=cfg.max_signals_per_request<=10000):
            raise ValueError('Invalid strategy configuration')
        if set(cfg.source_weights) != {'financial_news','social_media','manual','other'}:
            raise ValueError('Missing or unsupported source weight')
        if set(cfg.event_multipliers) != {'Geopolitical','Macroeconomic','Credit Event','Merger / Acquisition','Product Launch'}:
            raise ValueError('Missing event multiplier')
        import math
        if not all(math.isfinite(v) and v>0 for v in [*cfg.source_weights.values(), *cfg.event_multipliers.values()]):
            raise ValueError('Source/event multipliers must be finite and strictly positive')
        return cfg

    @property
    def baseline(self):
        return {t:1/len(self.tickers) for t in self.tickers}

    @property
    def sector_targets(self):
        sectors={}
        for t in self.tickers:
            s=self.sector_map[t]
            sectors[s]=sectors.get(s,0.0)+1/len(self.tickers)
        return sectors
