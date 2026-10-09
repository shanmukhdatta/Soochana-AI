"""A real-weights integration smoke test, separate from fast mock unit tests."""
import json
from datetime import datetime, timezone
from app.config import Settings
from app.engine.pipeline import RiskEngine
from app.schemas import AnalyzeRequest


def main():
    engine = RiskEngine.from_settings(Settings.from_env())
    request = AnalyzeRequest(
        text='The company announced a new product line for enterprise customers.',
        ticker='AAPL',
        published_at=datetime.now(timezone.utc),
        source='manual',
    )
    result = engine.analyze(request)
    print(json.dumps(result.model_dump(mode='json'), indent=2))

if __name__ == '__main__':
    main()
