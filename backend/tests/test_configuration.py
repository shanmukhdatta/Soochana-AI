from pathlib import Path
import json
import numpy as np
from app.config import ROOT


def read(name):
    return json.loads((ROOT/'configs'/name).read_text())


def test_exact_model_feature_schema():
    schema = read('impact_model_feature_manifest.json')
    assert len(schema['numeric_features']) == 14
    assert len(schema['categorical_features']) == 3
    assert schema['model_features'] == schema['numeric_features'] + schema['categorical_features']
    assert schema['target_column'] == 'realized_impact'
    assert not set(schema['model_features']).intersection(schema['forbidden_future_features'])
    assert schema['model_features'][-3:] == ['event_predicted_class','sector','market_regime']


def test_event_label_order_and_vocabs():
    labels = read('event_labels.json')
    assert list(labels.values()) == [
        'Geopolitical','Macroeconomic','Credit Event',
        'Merger / Acquisition','Product Launch'
    ]
    e = json.loads((ROOT/'configs/reference/event_model_config_CHECK_ONLY.json').read_text())
    s = json.loads((ROOT/'configs/reference/sentiment_model_config_CHECK_ONLY.json').read_text())
    assert e['vocab_size'] == 30873
    assert s['vocab_size'] == 30522
    assert s['problem_type'] == 'regression'


def test_train_only_frozen_deciles():
    d = read('impact_decile_thresholds.json')
    vals = np.array([d['quantiles'][f'p{x}'] for x in range(10,100,10)])
    assert len(vals) == 9
    assert np.all(np.diff(vals)>0)
    assert d['calibration_source'] == 'training split only'
    assert abs(vals[-1] - 0.04484478192898283)<1e-14


def test_regime_thresholds_frozen():
    d=read('impact_step2_market_regime_thresholds.json')
    assert d['training_end_date'] == '2021-04-15'
    assert abs(d['spy_return_20d_q33']-(-0.0014017912708248485))<1e-14
    assert abs(d['spy_volatility_20d_ann_q75']-.19568399976553735)<1e-14
