"""The single callable interface Module A and Module B will consume."""
import json
from pathlib import Path

from app.config import Settings
from app.engine.impact import ImpactPredictor
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.prices import PriceProvider
from app.services.features import FeatureBuilder


class RiskEngine:
    def __init__(self, nlp, impact, features):
        self.nlp = nlp
        self.impact = impact
        self.features = features

    @classmethod
    def from_settings(cls, settings: Settings) -> 'RiskEngine':
        config_dir = settings.config_dir
        with (config_dir/'event_labels.json').open('r', encoding='utf-8') as f:
            labels = {int(k):v for k,v in json.load(f).items()}
        from app.engine.nlp import NLPModels
        nlp = NLPModels(
            settings.event_model_dir,
            settings.sentiment_model_dir,
            labels,
            device=settings.model_device,
            max_length=settings.nlp_max_length,
        )
        impact = ImpactPredictor(
            settings.impact_model_dir,
            config_dir/'impact_model_feature_manifest.json',
            config_dir/'impact_decile_thresholds.json',
        )
        prices = PriceProvider(settings.price_provider,settings.price_csv_dir)
        features = FeatureBuilder(prices,config_dir)
        if features.supported is None:
            raise RuntimeError('Supported ticker universe missing')
        return cls(nlp, impact, features)

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        event_internal, sentiment = self.nlp.analyze(req.text)
        raw_features, market_context, warnings = self.features.build(
            req.ticker, req.published_at, event_internal, sentiment
        )
        impact = self.impact.predict(raw_features)
        sentiment_label = 'positive' if sentiment > .15 else 'negative' if sentiment < -.15 else 'neutral'
        return AnalyzeResponse(
            ticker=req.ticker,
            published_at=req.published_at,
            source=req.source,
            sentiment={'score':sentiment,'label':sentiment_label},
            event={
                'class_name':event_internal['class_name'],
                'confidence':event_internal['confidence'],
                'probabilities':event_internal['probabilities'],
            },
            impact=impact,
            market_context=market_context,
            warnings=warnings,
        )
