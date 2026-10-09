import json
from datetime import datetime, timezone
from app.config import Settings
from app.engine.pipeline import RiskEngine
from app.schemas import AnalyzeRequest

def test_five_scenarios():
    settings = Settings.from_env()
    engine = RiskEngine.from_settings(settings)

    scenarios = [
        {
            "category": "Credit downgrade / missed debt payment",
            "text": "Moody's downgrades Intel credit rating following missed bond payment obligations and liquidity stress.",
            "ticker": "INTC",
            "published_at": datetime(2024, 8, 15, 14, 0, tzinfo=timezone.utc),
            "source": "financial_news",
        },
        {
            "category": "Merger / acquisition announcement",
            "text": "Microsoft announces definitive agreement to acquire enterprise cloud automation platform for $12 billion.",
            "ticker": "MSFT",
            "published_at": datetime(2024, 6, 20, 10, 0, tzinfo=timezone.utc),
            "source": "financial_news",
        },
        {
            "category": "Product launch",
            "text": "Nvidia officially launches next-generation enterprise GPU architecture with massive AI processing speedup.",
            "ticker": "NVDA",
            "published_at": datetime(2024, 5, 10, 16, 0, tzinfo=timezone.utc),
            "source": "financial_news",
        },
        {
            "category": "Macroeconomic interest-rate event",
            "text": "Federal Reserve unexpected interest rate policy shifts trigger macroeconomic tightening across consumer sectors.",
            "ticker": "AMZN",
            "published_at": datetime(2024, 9, 19, 18, 30, tzinfo=timezone.utc),
            "source": "financial_news",
        },
        {
            "category": "Geopolitical disruption",
            "text": "Escalating international sanctions, trade embargoes, and military conflict disrupt global semiconductor export routes.",
            "ticker": "AAPL",
            "published_at": datetime(2024, 4, 15, 12, 0, tzinfo=timezone.utc),
            "source": "financial_news",
        },
    ]

    results = []
    print("=" * 80)
    print("RUNNING 5 REPRESENTATIVE REAL-MODEL INFERENCE SCENARIOS")
    print("=" * 80)

    for i, s in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {s['category']} ---")
        print(f"Ticker: {s['ticker']} | Timestamp: {s['published_at'].isoformat()}")
        print(f"Text: {s['text']}")
        req = AnalyzeRequest(
            text=s["text"],
            ticker=s["ticker"],
            published_at=s["published_at"],
            source=s["source"],
        )
        res = engine.analyze(req)
        data = res.model_dump(mode="json")
        results.append({"scenario": s["category"], "result": data})

        # Assertions
        assert data["event"]["class_name"] in [
            "Geopolitical", "Macroeconomic", "Credit Event", "Merger / Acquisition", "Product Launch"
        ], f"Invalid class: {data['event']['class_name']}"
        assert 0.0 <= data["event"]["confidence"] <= 1.0
        assert sum(data["event"]["probabilities"].values()) > 0.999
        assert -1.0 <= data["sentiment"]["score"] <= 1.0
        assert data["impact"]["predicted_realized_impact"] >= 0.0
        assert 1 <= data["impact"]["score"] <= 10
        assert isinstance(data["impact"]["score"], int)
        assert data["market_context"]["stock_feature_date"] < s["published_at"].date().isoformat()
        assert data["market_context"]["spy_feature_date"] < s["published_at"].date().isoformat()

        print(f"-> Predicted Event: {data['event']['class_name']} (conf: {data['event']['confidence']:.4f})")
        print(f"-> Sentiment: {data['sentiment']['score']:.4f} ({data['sentiment']['label']})")
        print(f"-> Impact Realized: {data['impact']['predicted_realized_impact']:.4f} | Decile Score: {data['impact']['score']}/10")
        print(f"-> Market Context: Sector={data['market_context']['sector']}, Regime={data['market_context']['market_regime']}")
        print(f"-> Pre-publication stock date: {data['market_context']['stock_feature_date']} < {s['published_at'].date().isoformat()}")
        print("PASS")

    print("\n" + "=" * 80)
    print("ALL 5 REAL-MODEL SCENARIOS PASSED ALL VALIDATIONS SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_five_scenarios()
