"""Comprehensive verification script for live FastAPI endpoints."""
import json
import requests

BASE = 'http://127.0.0.1:8000'

results = {}

# 1. GET /health
r = requests.get(f'{BASE}/health')
print('1. GET /health ->', r.status_code, r.json())
assert r.status_code == 200
results['health'] = r.status_code

# 2. POST /analyze (Original endpoint)
payload_analyze = {
    'text': 'Apple unveils next-generation M-series chips with breakthrough performance.',
    'ticker': 'AAPL',
    'published_at': '2024-05-15T12:00:00Z',
    'source': 'financial_news'
}
r = requests.post(f'{BASE}/analyze', json=payload_analyze)
print('2. POST /analyze ->', r.status_code, 'sentiment:', r.json()['sentiment']['score'], 'event:', r.json()['event']['class_name'], 'impact:', r.json()['impact']['score'])
assert r.status_code == 200
results['analyze'] = r.status_code

# 3. POST /analyze/batch (Original endpoint)
payload_batch = {
    'items': [
        payload_analyze,
        {
            'text': 'Microsoft faces massive cloud outage impacting government agencies.',
            'ticker': 'MSFT',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/analyze/batch', json=payload_batch)
print('3. POST /analyze/batch ->', r.status_code, 'total:', r.json()['total'])
assert r.status_code == 200
results['analyze_batch'] = r.status_code

# 4. GET /module-a/index
r = requests.get(f'{BASE}/module-a/index')
print('4. GET /module-a/index ->', r.status_code, 'tickers:', len(r.json()['tickers']))
assert r.status_code == 200
results['module_a_index'] = r.status_code

# 5. POST /module-a/rebalance (Stateless precomputed signal)
payload_rebal = {
    'as_of': '2024-05-16T23:00:00Z',
    'strategy': 'full_event_aware',
    'signals': [
        {
            'ticker': 'NVDA',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news',
            'headline': 'Nvidia reports record earnings and soaring enterprise AI chip demand.',
            'sentiment_score': 0.75,
            'impact_score': 8,
            'event_class': 'Macroeconomic',
            'event_confidence': 0.92
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance', json=payload_rebal)
print('5. POST /module-a/rebalance ->', r.status_code, 'NVDA target weight:', r.json()['target_weights']['NVDA'])
assert r.status_code == 200
assert r.json()['target_weights']['NVDA'] > 1/15
results['module_a_rebalance'] = r.status_code

# 6. POST /module-a/backtest and GET /module-a/backtest/{run_id}
r = requests.post(f'{BASE}/module-a/backtest', json={'dataset_id': 'holdout', 'start_date': '2022-08-19', 'end_date': '2023-12-26'})
print('6. POST /module-a/backtest ->', r.status_code, 'run_id:', r.json()['run_id'])
assert r.status_code == 200
run_id = r.json()['run_id']
results['module_a_backtest_post'] = r.status_code

r_res = requests.get(f'{BASE}/module-a/backtest/{run_id}')
print('7. GET /module-a/backtest/{run_id} ->', r_res.status_code, 'sessions:', r_res.json()['session_count'])
assert r_res.status_code == 200
results['module_a_backtest_get'] = r_res.status_code

# 8. POST /module-a/rebalance-from-news (Real model inference path!)
# Test Case A: Positive sentiment headline
payload_pos = {
    'as_of': '2024-05-16T23:00:00Z',
    'strategy': 'full_event_aware',
    'articles': [
        {
            'ticker': 'AAPL',
            'text': 'Apple announces record quarterly iPhone revenue and major share repurchase authorization.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_pos)
print('8a. Positive sentiment news ->', r.status_code, 'AAPL target:', r.json()['target_weights']['AAPL'])
assert r.status_code == 200
assert r.json()['target_weights']['AAPL'] > 1/15
results['rebalance_from_news_positive'] = r.status_code
real_example_response = r.json()

# Test Case B: Negative sentiment headline
payload_neg = {
    'as_of': '2024-05-16T23:00:00Z',
    'strategy': 'full_event_aware',
    'articles': [
        {
            'ticker': 'NVDA',
            'text': 'Nvidia faces regulatory antitrust investigation and slashes revenue guidance amid severe chip export halts.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_neg)
print('8b. Negative sentiment news ->', r.status_code, 'NVDA target:', r.json()['target_weights']['NVDA'])
assert r.status_code == 200
assert r.json()['target_weights']['NVDA'] < 1/15
results['rebalance_from_news_negative'] = r.status_code

# Test Case C: Multiple contradictory headlines
payload_contra = {
    'as_of': '2024-05-16T23:00:00Z',
    'strategy': 'full_event_aware',
    'articles': [
        {
            'ticker': 'MSFT',
            'text': 'Microsoft Azure posts record growth and massive enterprise demand.',
            'published_at': '2024-05-15T10:00:00Z',
            'source': 'financial_news'
        },
        {
            'ticker': 'MSFT',
            'text': 'Microsoft hit with severe security breach and massive corporate liability.',
            'published_at': '2024-05-15T14:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_contra)
print('8c. Contradictory headlines ->', r.status_code, 'MSFT signal count:', r.json()['signal_counts']['MSFT'])
assert r.status_code == 200
results['rebalance_from_news_contradictory'] = r.status_code

# Test Case D: Duplicate headline across sources
payload_dup = {
    'as_of': '2024-05-16T23:00:00Z',
    'strategy': 'full_event_aware',
    'articles': [
        {
            'ticker': 'AAPL',
            'text': 'Apple unveils next generation M-series chips with breakthrough performance.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        },
        {
            'ticker': 'AAPL',
            'text': 'Apple unveils next generation M-series chips with breakthrough performance!',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'social_media'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_dup)
print('8d. Duplicate headlines ->', r.status_code, 'AAPL signal count:', r.json()['signal_counts']['AAPL'], 'warnings:', r.json()['warnings'])
assert r.status_code == 200
assert r.json()['signal_counts']['AAPL'] == 1
assert any('Duplicate' in w for w in r.json()['warnings'])
results['rebalance_from_news_duplicate'] = r.status_code

# Test Case E: No actionable signal (all headlines on or after as_of date)
payload_future = {
    'as_of': '2024-05-15T12:00:00Z',
    'strategy': 'full_event_aware',
    'articles': [
        {
            'ticker': 'AAPL',
            'text': 'Apple unveils next generation M-series chips today.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_future)
print('8e. No actionable signal (same-day excluded) ->', r.status_code, 'turnover:', r.json()['one_way_turnover'], 'warnings:', r.json()['warnings'])
assert r.status_code == 200
assert r.json()['one_way_turnover'] == 0.0
assert any('No recent usable signals' in w for w in r.json()['warnings'])
results['rebalance_from_news_no_actionable'] = r.status_code

# Test Case F: Unsupported ticker
payload_unsupported = {
    'as_of': '2024-05-16T23:00:00Z',
    'articles': [
        {
            'ticker': 'UNKNOWN_TICKER_XYZ',
            'text': 'Unknown company announces surprise dividend.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_unsupported)
print('8f. Unsupported ticker ->', r.status_code, r.json())
assert r.status_code == 422
results['rebalance_from_news_unsupported_ticker'] = r.status_code

# Test Case G: Invalid timestamp (naive timestamp)
payload_bad_ts = {
    'as_of': '2024-05-16T23:00:00',  # missing timezone
    'articles': [
        {
            'ticker': 'AAPL',
            'text': 'Apple announces strong sales growth.',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_bad_ts)
print('8g. Invalid timestamp ->', r.status_code)
assert r.status_code == 422
results['rebalance_from_news_invalid_timestamp'] = r.status_code

# Test Case H: Unavailable market data (date in far future / 2099)
payload_far_future = {
    'as_of': '2099-05-16T23:00:00Z',
    'articles': [
        {
            'ticker': 'AAPL',
            'text': 'Apple announces quantum computer in future.',
            'published_at': '2099-05-15T12:00:00Z',
            'source': 'financial_news'
        }
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_far_future)
print('8h. Unavailable market data ->', r.status_code, r.json())
assert r.status_code in (422, 503)
results['rebalance_from_news_unavailable_market_data'] = r.status_code

# Test Case I: Excessive request size (> 20 articles in rebalance-from-news)
payload_excessive = {
    'as_of': '2024-05-16T23:00:00Z',
    'articles': [
        {
            'ticker': 'AAPL',
            'text': f'Apple headline number {i}',
            'published_at': '2024-05-15T12:00:00Z',
            'source': 'financial_news'
        }
        for i in range(25)
    ]
}
r = requests.post(f'{BASE}/module-a/rebalance-from-news', json=payload_excessive)
print('8i. Excessive request size ->', r.status_code)
assert r.status_code == 422
results['rebalance_from_news_excessive_size'] = r.status_code

# Test Case J: Invalid portfolio weights in rebalance
payload_invalid_weights = {
    'as_of': '2024-05-16T23:00:00Z',
    'current_weights': {'AAPL': 0.5, 'MSFT': 0.3},  # doesn't sum to 1, doesn't contain all 15
    'signals': []
}
r = requests.post(f'{BASE}/module-a/rebalance', json=payload_invalid_weights)
print('8j. Invalid portfolio weights ->', r.status_code, r.json())
assert r.status_code == 422
results['rebalance_invalid_weights'] = r.status_code

# Test Case K: Missing historical backtest data
r = requests.post(f'{BASE}/module-a/backtest', json={'dataset_id': 'nonexistent_dataset'})
print('8k. Missing historical backtest data ->', r.status_code, r.json())
assert r.status_code == 422
results['backtest_missing_data'] = r.status_code

print('\nALL LIVE ENDPOINT CHECKS PASSED!')
print(json.dumps(results, indent=2))

with open('data/module_a/real_example_response.json', 'w') as f:
    json.dump(real_example_response, f, indent=2)
print('Saved genuine example to data/module_a/real_example_response.json')
