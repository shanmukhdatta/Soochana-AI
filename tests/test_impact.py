from pathlib import Path
import numpy as np
from app.config import ROOT
from app.engine.impact import ImpactPredictor


class DummyPreprocessor:
    def transform(self, frame):
        return np.array([[float(frame['sentiment_score'].iloc[0])]])


class DummyRegressor:
    def __init__(self, value):
        self.value=value
    def predict(self,x):
        return np.array([self.value])


def make_fake(v):
    import json
    cfg=ROOT/'configs'
    manifest=json.loads((cfg/'impact_model_feature_manifest.json').read_text())
    deciles=json.loads((cfg/'impact_decile_thresholds.json').read_text())['quantiles']
    ip=ImpactPredictor.__new__(ImpactPredictor)
    ip.model=DummyRegressor(v)
    ip.preprocessor=DummyPreprocessor()
    ip.features=manifest['model_features']
    ip.numeric=manifest['numeric_features']
    ip.categorical=manifest['categorical_features']
    ip.deciles=np.array([deciles[f'p{i}'] for i in range(10,100,10)])
    return ip


def fake_inputs(ip):
    data={k:0.02 for k in ip.numeric}
    data.update({
        'event_predicted_class':'Credit Event',
        'sector':'Technology',
        'market_regime':'neutral_normal_vol',
    })
    return data


def test_impact_score_from_frozen_deciles():
    for value,score in [(0,1),(.006,2),(.046,10),(.025,8),(-1,1)]:
        engine=make_fake(value)
        got=engine.predict(fake_inputs(engine))
        assert got['score']==score,(value,got)
        assert got['predicted_realized_impact']>=0
