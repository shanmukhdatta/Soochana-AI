from datetime import datetime,timezone
from app.engine.pipeline import RiskEngine
from app.schemas import AnalyzeRequest


class FakeNLP:
    def analyze(self, text):
        assert 'credit' in text.lower()
        return {
            'class_name':'Credit Event',
            'confidence':.8,
            'probabilities':{'Credit Event':.8},
            'probability_vector':[.05,.05,.8,.05,.05],
            'class_id':2,
        }, -.42


class FakeFeatures:
    def build(self, ticker, pub, event, sentiment):
        assert ticker == 'AAPL'
        assert event['class_name'] == 'Credit Event'
        assert sentiment == -.42
        return {'sentiment_score':-.42}, {
            'sector':'Technology', 'market_regime':'neutral_normal_vol',
            'stock_feature_date':'2026-10-06',
            'spy_feature_date':'2026-10-06',
            'provider':'csv',
        }, ['mock feature provider']


class FakeImpact:
    def predict(self, features):
        assert features['sentiment_score']==-.42
        return {'predicted_realized_impact':.03,'score':8}


def test_combined_risk_engine_contract_with_mocks():
    engine=RiskEngine(FakeNLP(), FakeImpact(), FakeFeatures())
    response=engine.analyze(AnalyzeRequest(
        ticker='AAPL', text='Company credit event has occurred',
        published_at=datetime(2026,10,8,tzinfo=timezone.utc),source='manual'
    ))
    assert response.sentiment.label == 'negative'
    assert response.event.class_name == 'Credit Event'
    assert response.impact.score == 8
    assert response.market_context.stock_feature_date == '2026-10-06'
    assert response.warnings == ['mock feature provider']
