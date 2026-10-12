"""Frozen sklearn preprocessor + XGBoost model; no retraining in the API."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class ImpactPredictor:
    def __init__(self, model_dir: Path, manifest_path: Path, decile_path: Path):
        model_dir = Path(model_dir)
        model_path = model_dir / 'impact_model_final.joblib'
        prep_path = model_dir / 'impact_preprocessor.joblib'
        for p in [model_path, prep_path, manifest_path, decile_path]:
            if not Path(p).is_file():
                raise FileNotFoundError(
                    f'Required Impact Engine artifact not found: {p}'
                )
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        with open(decile_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)['quantiles']

        self.features = manifest['model_features']
        forbidden = set(manifest['forbidden_future_features'])
        if forbidden.intersection(self.features) or len(self.features) != len(set(self.features)):
            raise ValueError('Invalid feature manifest: duplicates/future leakage')
        self.numeric = manifest['numeric_features']
        self.categorical = manifest['categorical_features']
        if self.features != self.numeric + self.categorical:
            raise ValueError('Feature order differs from training manifest')
        self.deciles = np.array([thresholds[f'p{i}'] for i in range(10,100,10)], dtype=float)
        if not np.isfinite(self.deciles).all() or not np.all(np.diff(self.deciles) > 0):
            raise ValueError('Invalid frozen deciles')
        # Only load your own trusted joblib files; joblib uses pickle internally.
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(prep_path)
        raw_feature_order = list(getattr(self.preprocessor, 'feature_names_in_', []))
        if raw_feature_order and raw_feature_order != self.features:
            raise ValueError('Saved preprocessor feature_names_in_ differs from manifest')
        model_in = getattr(self.model, 'n_features_in_', None)
        if model_in is not None and model_in != len(self.preprocessor.get_feature_names_out()):
            raise ValueError('XGBoost feature count does not match saved preprocessor')
        # Restore trained target base_score (fit mean = 0.021967) lost during pickle serialization
        base_score = manifest.get('base_score', 0.021967076942109524)
        if hasattr(self.model, 'get_booster'):
            self.model.get_booster().set_param({'base_score': float(base_score)})

    def predict(self, features: dict) -> dict:
        missing = sorted(set(self.features) - set(features))
        if missing:
            raise ValueError(f'Impact features missing: {missing}')
        row = pd.DataFrame([{f: features[f] for f in self.features}], columns=self.features)
        for k in self.numeric:
            row[k] = pd.to_numeric(row[k], errors='raise')
        for k in self.categorical:
            row[k] = row[k].astype(str)
        X = self.preprocessor.transform(row)
        pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if len(pred) != 1 or not np.isfinite(pred[0]):
            raise ValueError('Impact model returned invalid prediction')
        impact = max(float(pred[0]), 0.0)
        score = int(np.searchsorted(self.deciles, impact, side='right')) + 1
        return {'predicted_realized_impact': impact, 'score': score}
