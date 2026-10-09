"""Comprehensive end-to-end verification and benchmarking of live FastAPI endpoints.
Tests existing Risk Engine, Module A, and new Module B routes.
"""
import time
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"

def main():
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    results = {}

    print("==================================================================")
    print("RUNNING LIVE ENDPOINT AUDIT & BENCHMARKING ON 127.0.0.1:8000")
    print("==================================================================")

    # 1. Existing Endpoints: Health
    print("\n[1] GET /health")
    r = client.get("/health")
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text}")
    assert r.status_code == 200
    results['health'] = r.json()

    # 2. Existing Endpoints: Analyze
    print("\n[2] POST /analyze (Real model inference)")
    news_payload = {
        "text": "Apple bonds face downgrade following liquidity concerns.",
        "ticker": "AAPL",
        "published_at": "2024-05-15T14:30:00Z",
        "source": "financial_news"
    }
    r = client.post("/analyze", json=news_payload)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:300]}...")
    assert r.status_code == 200
    results['analyze'] = r.json()

    # 3. Existing Endpoints: Analyze Batch
    print("\n[3] POST /analyze/batch")
    batch_payload = {"items": [news_payload]}
    r = client.post("/analyze/batch", json=batch_payload)
    print(f"Status: {r.status_code}")
    assert r.status_code == 200
    results['analyze_batch_count'] = len(r.json()['results'])

    # 4. Existing Module A Endpoints
    print("\n[4] GET /module-a/index")
    r = client.get("/module-a/index")
    print(f"Status: {r.status_code}")
    print(f"Body: {r.json()['version']}")
    assert r.status_code == 200
    results['module_a_index'] = r.json()['version']

    # 5. Module B: GET /module-b/portfolio
    print("\n[5] GET /module-b/portfolio")
    r = client.get("/module-b/portfolio")
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Holdings count: {data['summary']['position_count']}")
    assert r.status_code == 200
    assert data['summary']['position_count'] == 30
    assert data['summary']['counts'] == {'loan': 12, 'bond': 10, 'swap': 4, 'option': 4}
    assert data['summary']['loan_ead_usd'] == 5000000.0
    assert data['summary']['bond_market_value_usd'] == 3500000.0
    assert data['summary']['derivative_reference_notional_usd'] == 1500000.0
    results['module_b_portfolio'] = data['summary']

    # 6. Module B: GET /module-b/scenarios
    print("\n[6] GET /module-b/scenarios")
    r = client.get("/module-b/scenarios")
    print(f"Status: {r.status_code}")
    data = r.json()
    assert r.status_code == 200
    assert len(data['scenario_library']) == 5
    results['module_b_scenario_count'] = sum(len(v) for v in data['scenario_library'].values())

    # 7. Module B: POST /module-b/stress (Custom scenario)
    print("\n[7] POST /module-b/stress")
    stress_payload = {
        "scenario": {
            "name": "Custom hypothetical macro shock",
            "event_class": "Macroeconomic",
            "severity": "custom",
            "shocks": {
                "rate_bps": 100,
                "spread_bps": 150,
                "equity_pct": -15,
                "volatility_points": 0.05,
                "pd_multiplier": 2.0
            }
        }
    }
    r = client.post("/module-b/stress", json=stress_payload)
    print(f"Status: {r.status_code}")
    print(f"Totals: {r.json()['totals']}")
    assert r.status_code == 200
    results['custom_stress_totals'] = r.json()['totals']

    # 8. Module B: POST /module-b/what-if
    print("\n[8] POST /module-b/what-if")
    whatif_payload = {
        "scenario": stress_payload["scenario"],
        "changes": [
            {"position_id": "L01", "scale": 0.5},
            {"position_id": "B01", "scale": 0.0}
        ]
    }
    r = client.post("/module-b/what-if", json=whatif_payload)
    print(f"Status: {r.status_code}")
    print(f"Net effect change: {r.json()['net_effect_change_usd']}")
    assert r.status_code == 200
    results['what_if_net_effect_change'] = r.json()['net_effect_change_usd']

    # 9. Module B: POST /module-b/reverse-stress
    print("\n[9] POST /module-b/reverse-stress")
    rev_payload = {
        "scenario": {
            "name": "Illustrative credit deterioration",
            "event_class": "Credit Event",
            "shocks": {
                "rate_bps": 25,
                "spread_bps": 100,
                "equity_pct": -10,
                "volatility_points": 0.05,
                "pd_multiplier": 1.8
            }
        },
        "loss_budget_usd": 100000,
        "max_multiplier": 2.0,
        "steps": 41
    }
    r = client.post("/module-b/reverse-stress", json=rev_payload)
    print(f"Status: {r.status_code}")
    rev_data = r.json()
    print(f"Found breach: {rev_data['found_breach']}, First breach: {rev_data['first_breach_on_grid']}")
    assert r.status_code == 200
    results['reverse_stress'] = rev_data['first_breach_on_grid']

    # 10. Module B: POST /module-b/stress-from-news with run_stress=False (Suggest-only)
    print("\n[10] POST /module-b/stress-from-news (run_stress=False)")
    news_stress_payload = {
        "news": news_payload,
        "severity": "auto",
        "run_stress": False
    }
    r = client.post("/module-b/stress-from-news", json=news_stress_payload)
    print(f"Status: {r.status_code}")
    suggest_data = r.json()
    print(f"Executed: {suggest_data['executed']}")
    print(f"Suggested Scenario: {suggest_data['suggested_scenario']['name']}")
    print(f"Severity Source: {suggest_data['severity_source']}")
    assert r.status_code == 200
    assert suggest_data['executed'] is False
    assert 'stress_result' not in suggest_data
    assert 'AAPL' in suggest_data['suggested_scenario']['issuer_shocks']
    results['stress_from_news_suggest'] = {
        "suggested_scenario": suggest_data['suggested_scenario']['name'],
        "event_class": suggest_data['suggested_scenario']['event_class'],
        "severity": suggest_data['suggested_scenario']['severity'],
        "executed": suggest_data['executed']
    }

    # 11. Module B: POST /module-b/stress-from-news with run_stress=True (Explicit approval)
    print("\n[11] POST /module-b/stress-from-news (run_stress=True)")
    news_stress_approved_payload = {
        "news": news_payload,
        "severity": "auto",
        "run_stress": True
    }
    r = client.post("/module-b/stress-from-news", json=news_stress_approved_payload)
    print(f"Status: {r.status_code}")
    exec_data = r.json()
    print(f"Executed: {exec_data['executed']}")
    print(f"Stress totals: {exec_data['stress_result']['totals']}")
    assert r.status_code == 200
    assert exec_data['executed'] is True
    assert 'stress_result' in exec_data
    assert len(exec_data['stress_result']['positions']) == 30
    results['stress_from_news_executed'] = exec_data

    # 12. Negative / Error Testing
    print("\n[12] Negative Test: Invalid published_at date format -> HTTP 422")
    bad_time_payload = {"news": dict(news_payload, published_at="bad-date-time"), "run_stress": False}
    r = client.post("/module-b/stress-from-news", json=bad_time_payload)
    print(f"Status: {r.status_code} (Expected 422)")
    assert r.status_code == 422

    print("\n[12b] Negative Test: Unknown ticker -> HTTP 422")
    bad_ticker_payload = {"news": dict(news_payload, ticker="UNKNOWN_XYZ"), "run_stress": False}
    r = client.post("/module-b/stress-from-news", json=bad_ticker_payload)
    print(f"Status: {r.status_code} (Expected 422)")
    assert r.status_code == 422

    print("\n[12c] Negative Test: Invalid shock magnitude -> HTTP 422")
    bad_shock_payload = {"scenario": {"name": "Bad", "shocks": {"equity_pct": -150.0}}}
    r = client.post("/module-b/stress", json=bad_shock_payload)
    print(f"Status: {r.status_code} (Expected 422)")
    assert r.status_code == 422

    print("\n[12d] Negative Test: Extreme historical date with missing price data -> HTTP 503")
    old_date_payload = {"news": dict(news_payload, published_at="1970-01-01T12:00:00Z"), "run_stress": False}
    r = client.post("/module-b/stress-from-news", json=old_date_payload)
    print(f"Status: {r.status_code} (Expected 503)")
    assert r.status_code == 503

    # 13. CPU Latency Benchmarking
    print("\n[13] CPU Latency Benchmarking...")
    # Benchmark /module-b/stress (pure CPU offline valuation)
    t_stress_list = []
    for _ in range(20):
        t0 = time.perf_counter()
        r = client.post("/module-b/stress", json=stress_payload)
        t_stress_list.append(time.perf_counter() - t0)
    avg_stress_ms = (sum(t_stress_list) / len(t_stress_list)) * 1000
    min_stress_ms = min(t_stress_list) * 1000

    # Benchmark /module-b/stress-from-news (Real FinBERT + FinBERT + XGBoost + yfinance market context + stress engine)
    t_news_list = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = client.post("/module-b/stress-from-news", json=news_stress_approved_payload)
        t_news_list.append(time.perf_counter() - t0)
    avg_news_ms = (sum(t_news_list) / len(t_news_list)) * 1000
    min_news_ms = min(t_news_list) * 1000

    print(f"Module B /stress latency (20 runs): avg={avg_stress_ms:.2f} ms, min={min_stress_ms:.2f} ms")
    print(f"Module B /stress-from-news latency (5 runs): avg={avg_news_ms:.2f} ms, min={min_news_ms:.2f} ms")

    results['benchmarks'] = {
        'stress_offline_avg_ms': avg_stress_ms,
        'stress_offline_min_ms': min_stress_ms,
        'stress_from_news_avg_ms': avg_news_ms,
        'stress_from_news_min_ms': min_news_ms
    }

    # Save results to a file for report generation
    with open("data/module_b/live_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n==================================================================")
    print("ALL LIVE ENDPOINT TESTS AND BENCHMARKS PASSED PERFECTLY!")
    print("==================================================================")

if __name__ == '__main__':
    main()
