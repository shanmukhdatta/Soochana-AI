"""Comprehensive verification script for all ingestion and existing API endpoints."""
import os
import sys
import json
import requests

BASE = "http://127.0.0.1:8000"
TOKEN = "test_ingestion_admin_token_secure_32ch"


def main():
    print("Testing Live API Endpoints at", BASE)
    results = {}

    # 1. Health and Existing routes
    r = requests.get(f"{BASE}/health")
    print(f"GET /health: {r.status_code}")
    results["/health"] = r.status_code

    r = requests.get(f"{BASE}/module-a/index")
    print(f"GET /module-a/index: {r.status_code}")
    results["/module-a/index"] = r.status_code

    r = requests.get(f"{BASE}/module-b/portfolio")
    print(f"GET /module-b/portfolio: {r.status_code}")
    results["/module-b/portfolio"] = r.status_code

    # 2. Existing /analyze route
    r = requests.post(f"{BASE}/analyze", json={
        "text": "Apple introduces breakthrough M5 chip architecture with higher efficiency",
        "ticker": "AAPL",
        "published_at": "2026-10-09T18:00:00Z",
        "source": "manual"
    })
    print(f"POST /analyze: {r.status_code}")
    results["/analyze"] = r.status_code

    # 3. Ingestion Health
    r = requests.get(f"{BASE}/ingestion/health")
    print(f"GET /ingestion/health: {r.status_code}, body: {r.json()}")
    results["/ingestion/health"] = r.status_code

    # 4. Signals Summary
    r = requests.get(f"{BASE}/signals/summary")
    print(f"GET /signals/summary: {r.status_code}, status_counts: {r.json().get('status_counts')}")
    results["/signals/summary"] = r.status_code

    # 5. List Signals
    r = requests.get(f"{BASE}/signals?limit=5")
    print(f"GET /signals: {r.status_code}, total: {r.json().get('total')}")
    results["/signals"] = r.status_code
    items = r.json().get("items", [])
    completed_item = next((i for i in items if i.get("status") == "completed"), items[0] if items else None)

    # 6. Get single signal
    if completed_item:
        eid = completed_item["id"]
        r = requests.get(f"{BASE}/signals/{eid}")
        print(f"GET /signals/{eid}: {r.status_code}, status: {r.json().get('status')}")
        results["/signals/{id}"] = r.status_code

        # 7. Module B suggestion from signal
        r = requests.get(f"{BASE}/signals/{eid}/module-b-suggestion")
        print(f"GET /signals/{eid}/module-b-suggestion: {r.status_code}, executed={r.json().get('executed')}")
        results["/signals/{id}/module-b-suggestion"] = r.status_code

    # 8. Module A recommend driven by saved signals
    r = requests.post(f"{BASE}/ingestion/module-a/recommend")
    print(f"POST /ingestion/module-a/recommend: {r.status_code}")
    results["/ingestion/module-a/recommend"] = r.status_code
    if r.status_code == 200:
        rec = r.json()
        weights = rec.get("recommendation", {}).get("target_weights", {})
        print(f"  Saved signals considered: {rec.get('saved_signals_considered')}, sum(weights)={round(sum(weights.values()), 4)}")

    # 9. Protected mutations: missing token -> 403
    r = requests.post(f"{BASE}/ingestion/collect-once")
    print(f"POST /ingestion/collect-once (no token): {r.status_code} (expect 403)")
    results["collect_no_token"] = r.status_code

    # 10. Protected mutations: wrong token -> 403
    r = requests.post(f"{BASE}/ingestion/collect-once", headers={"X-Ingestion-Token": "wrong_token"})
    print(f"POST /ingestion/collect-once (wrong token): {r.status_code} (expect 403)")
    results["collect_wrong_token"] = r.status_code

    # 11. Protected mutations: correct token -> 200
    r = requests.post(f"{BASE}/ingestion/collect-once", headers={"X-Ingestion-Token": TOKEN})
    print(f"POST /ingestion/collect-once (correct token): {r.status_code}")
    results["collect_correct_token"] = r.status_code

    r = requests.post(f"{BASE}/ingestion/process-once", headers={"X-Ingestion-Token": TOKEN})
    print(f"POST /ingestion/process-once (correct token): {r.status_code}, body: {r.json()}")
    results["process_correct_token"] = r.status_code

    print("\nALL VERIFICATION CHECKS COMPLETE:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
