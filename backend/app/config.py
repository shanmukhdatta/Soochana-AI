"""Runtime paths and settings. No model weights are bundled in the source ZIP."""
from dataclasses import dataclass
import os
from pathlib import Path

_p = Path(__file__).resolve()
ROOT = _p.parents[2] if (_p.parents[2] / 'configs').exists() else _p.parents[1]


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


@dataclass(frozen=True)
class Settings:
    event_model_dir: Path
    sentiment_model_dir: Path
    impact_model_dir: Path
    config_dir: Path
    price_csv_dir: Path | None
    price_provider: str
    model_device: str
    nlp_max_length: int
    max_batch: int
    skip_model_load: bool

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv('PRICE_PROVIDER', 'yfinance').strip().lower()
        if provider not in {'yfinance', 'csv'}:
            raise ValueError('PRICE_PROVIDER must be yfinance or csv')
        max_length = int(os.getenv('NLP_MAX_LENGTH', '128'))
        if not 1 <= max_length <= 512:
            raise ValueError('NLP_MAX_LENGTH must be between 1 and 512')
        price_csv = os.getenv('PRICE_CSV_DIR')
        if provider == 'csv' and not price_csv:
            raise ValueError('PRICE_CSV_DIR required when PRICE_PROVIDER=csv')
        return cls(
            event_model_dir=_path_env('EVENT_MODEL_DIR', ROOT/'models/event_classifier'),
            sentiment_model_dir=_path_env('SENTIMENT_MODEL_DIR', ROOT/'models/sentiment_regressor'),
            impact_model_dir=_path_env('IMPACT_MODEL_DIR', ROOT/'models/impact_engine'),
            config_dir=_path_env('CONFIG_DIR', ROOT/'configs'),
            price_csv_dir=Path(price_csv).expanduser().resolve() if price_csv else None,
            price_provider=provider,
            model_device=os.getenv('MODEL_DEVICE', 'cpu').strip().lower(),
            nlp_max_length=max_length,
            max_batch=max(1, int(os.getenv('MAX_BATCH_SIZE', '20'))),
            skip_model_load=os.getenv('SKIP_MODEL_LOAD', '').strip().lower() in {'1', 'true', 'yes'},
        )
