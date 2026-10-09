import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from app.config import Settings
from app.engine.pipeline import RiskEngine
from app.schemas import AnalyzeRequest

engine = RiskEngine.from_settings(Settings.from_env())

booster = engine.impact.model.get_booster()
print('Initial booster base_score:', json.loads(booster.save_config())['learner']['learner_model_param']['base_score'])

scenarios = [
    ('Credit Event', 'INTC', datetime(2024, 8, 15, 14, 0, tzinfo=timezone.utc), "Moody's downgrades Intel credit rating following missed bond payment obligations and liquidity stress."),
    ('Merger / Acquisition', 'MSFT', datetime(2024, 6, 20, 10, 0, tzinfo=timezone.utc), "Microsoft announces definitive agreement to acquire enterprise cloud automation platform for $12 billion."),
    ('Product Launch', 'NVDA', datetime(2024, 5, 10, 16, 0, tzinfo=timezone.utc), "Nvidia officially launches next-generation enterprise GPU architecture with massive AI processing speedup."),
    ('Macroeconomic', 'AMZN', datetime(2024, 9, 19, 18, 30, tzinfo=timezone.utc), "Federal Reserve unexpected interest rate policy shifts trigger macroeconomic tightening across consumer sectors."),
    ('Geopolitical', 'AAPL', datetime(2024, 4, 15, 12, 0, tzinfo=timezone.utc), "Escalating international sanctions, trade embargoes, and military conflict disrupt global semiconductor export routes."),
]

report = []

for name, ticker, dt, text in scenarios:
    req = AnalyzeRequest(text=text, ticker=ticker, published_at=dt, source='financial_news')
    event_internal, sentiment = engine.nlp.analyze(req.text)
    raw_features, market_context, warnings = engine.features.build(
        req.ticker, req.published_at, event_internal, sentiment
    )
    
    row = pd.DataFrame([{f: raw_features[f] for f in engine.impact.features}], columns=engine.impact.features)
    for k in engine.impact.numeric:
        row[k] = pd.to_numeric(row[k], errors='raise')
    for k in engine.impact.categorical:
        row[k] = row[k].astype(str)
    X_enc = engine.impact.preprocessor.transform(row)
    enc_array = X_enc.toarray()[0] if hasattr(X_enc, 'toarray') else np.asarray(X_enc)[0]
    enc_names = engine.impact.preprocessor.get_feature_names_out()
    
    # Original prediction (base_score = 0.5)
    booster.set_param({'base_score': 0.5})
    pred_orig = engine.impact.predict(raw_features)
    
    # Fixed prediction (base_score = 0.021967076942109524)
    booster.set_param({'base_score': 0.021967076942109524})
    pred_fixed = engine.impact.predict(raw_features)
    
    entry = {
        'scenario': name,
        'ticker': ticker,
        'raw_17': {k: raw_features[k] for k in engine.impact.features},
        'encoded_31': {name: float(val) for name, val in zip(enc_names, enc_array)},
        'orig_pred': pred_orig,
        'fixed_pred': pred_fixed,
    }
    report.append(entry)

print(json.dumps(report, indent=2))
