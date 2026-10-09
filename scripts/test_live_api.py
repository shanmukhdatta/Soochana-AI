import json
import httpx
from datetime import datetime, timezone

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    print("=" * 80)
    print("TESTING LIVE FASTAPI SERVER AT http://127.0.0.1:8000")
    print("=" * 80)

    # 1. GET /health
    print("\n1. Testing GET /health...")
    resp = client.get("/health")
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["model_version"] == "risk-engine-v1"
    print("-> GET /health PASSED")

    # 2. POST /analyze
    print("\n2. Testing POST /analyze with real ticker and headline...")
    req_payload = {
        "text": "The company announced a major new product launch for business customers.",
        "ticker": "AAPL",
        "published_at": "2024-05-15T14:30:00Z",
        "source": "financial_news"
    }
    resp = client.post("/analyze", json=req_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert -1.0 <= data["sentiment"]["score"] <= 1.0
    assert data["sentiment"]["label"] in ["positive", "negative", "neutral"]
    assert data["event"]["class_name"] in [
        "Geopolitical", "Macroeconomic", "Credit Event", "Merger / Acquisition", "Product Launch"
    ]
    assert 0.0 <= data["event"]["confidence"] <= 1.0
    assert data["impact"]["predicted_realized_impact"] >= 0.0
    assert 1 <= data["impact"]["score"] <= 10
    assert data["market_context"]["sector"] == "Technology"
    assert data["market_context"]["stock_feature_date"] < "2024-05-15"
    print("-> POST /analyze PASSED")

    # 3. POST /analyze/batch
    print("\n3. Testing POST /analyze/batch with 2 items...")
    batch_payload = {
        "items": [
            {
                "text": "Microsoft announces strategic cloud enterprise partnership with multi-year expansion.",
                "ticker": "MSFT",
                "published_at": "2024-06-10T12:00:00Z",
                "source": "financial_news"
            },
            {
                "text": "Nvidia expands next-generation AI chip production with record enterprise demand.",
                "ticker": "NVDA",
                "published_at": "2024-06-10T12:00:00Z",
                "source": "financial_news"
            }
        ]
    }
    resp = client.post("/analyze/batch", json=batch_payload)
    print(f"Status Code: {resp.status_code}")
    assert resp.status_code == 200
    batch_data = resp.json()
    assert batch_data["total"] == 2
    assert len(batch_data["results"]) == 2
    for item in batch_data["results"]:
        assert 1 <= item["impact"]["score"] <= 10
        assert -1.0 <= item["sentiment"]["score"] <= 1.0
    print("-> POST /analyze/batch PASSED")

    # 4. Error testing: Malformed input (text too short)
    print("\n4. Testing malformed input (text too short)...")
    resp = client.post("/analyze", json={
        "text": "hi",
        "ticker": "AAPL",
        "published_at": "2024-05-15T14:30:00Z"
    })
    print(f"Status Code: {resp.status_code} (Expected 422)")
    assert resp.status_code == 422
    print("-> Validation rejection PASSED")

    # 5. Error testing: Missing timezone
    print("\n5. Testing missing timezone...")
    resp = client.post("/analyze", json={
        "text": "Company launches enterprise cloud service.",
        "ticker": "AAPL",
        "published_at": "2024-05-15T14:30:00"
    })
    print(f"Status Code: {resp.status_code} (Expected 422)")
    assert resp.status_code == 422
    print("-> Timezone enforcement PASSED")

    # 6. Error testing: Unsupported ticker
    print("\n6. Testing unsupported ticker...")
    resp = client.post("/analyze", json={
        "text": "Company launches enterprise cloud service.",
        "ticker": "XYZUNKNOWN",
        "published_at": "2024-05-15T14:30:00Z"
    })
    print(f"Status Code: {resp.status_code} (Expected 422)")
    print(f"Detail: {resp.json().get('detail')}")
    assert resp.status_code == 422
    assert "outside the" in str(resp.json()) or "Invalid ticker" in str(resp.json())
    print("-> Unsupported ticker rejection PASSED")

    # 7. Error testing: Stale / unavailable market data
    print("\n7. Testing unavailable / stale date...")
    resp = client.post("/analyze", json={
        "text": "Ancient news item from year 1990.",
        "ticker": "AAPL",
        "published_at": "1990-01-15T12:00:00Z"
    })
    print(f"Status Code: {resp.status_code} (Expected 503 or 422)")
    print(f"Detail: {resp.json().get('detail')}")
    assert resp.status_code in [503, 422]
    print("-> Market data unavailability handling PASSED")

    print("\n" + "=" * 80)
    print("ALL API ENDPOINTS AND ERROR HANDLERS VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
