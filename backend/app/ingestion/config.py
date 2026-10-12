"""Settings intentionally independent of app.config to avoid disturbing model loader."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path

_p = Path(__file__).resolve()
ROOT = _p.parents[3] if (_p.parents[3] / 'configs').exists() else _p.parents[2]


@dataclass(frozen=True)
class IngestionConfig:
    raw: dict
    aliases: dict[str, list[str]]
    db_path: Path
    admin_token: str

    @classmethod
    def load(cls, root: Path | None = None) -> 'IngestionConfig':
        root = root or ROOT
        conf = json.loads((root / 'configs' / 'ingestion_config.json').read_text(encoding='utf-8'))
        aliases = json.loads((root / 'configs' / 'ticker_aliases.json').read_text(encoding='utf-8'))
        allowed = json.loads((root / 'configs' / 'supported_tickers.json').read_text(encoding='utf-8'))
        if isinstance(allowed, dict):
            allowed = allowed.get('supported_tickers', allowed.get('tickers', []))
        if not isinstance(allowed, list) or not allowed:
            raise ValueError('Unable to verify supported tickers from Risk Engine config')
        try:
            from dotenv import load_dotenv
            load_dotenv(root / '.env')
        except Exception:
            pass
        if not set(aliases).issubset(set(allowed)):
            raise ValueError('Ingestion aliases include unsupported Risk Engine tickers')
        if not set(conf['enabled_sources']).issubset({'gdelt', 'bluesky', 'marketaux'}):
            raise ValueError('Unknown ingestion source')
        if not 60 <= conf['poll_interval_seconds'] <= 86400:
            raise ValueError('Invalid scheduler interval')
        if not 1 <= conf['max_inference_per_poll'] <= 50:
            raise ValueError('Invalid inference batch size')
        db_value = os.getenv('INGESTION_DB_PATH', str(root / 'data' / 'ingestion' / 'riskpulse.db'))
        path = Path(db_value)
        if not path.is_absolute():
            path = root / path
        return cls(conf, aliases, path.resolve(), os.getenv('INGESTION_ADMIN_TOKEN', ''))
