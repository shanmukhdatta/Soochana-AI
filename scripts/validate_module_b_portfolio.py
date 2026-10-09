"""From c2c root: python -m scripts.validate_module_b_portfolio"""
import json
from app.module_b.portfolio import load_portfolio,portfolio_summary
from app.module_b.scenarios import load_library

def main():
    p=load_portfolio()
    summary=portfolio_summary(p)
    scenarios=load_library()
    assert summary['position_count']==30,summary
    assert summary['counts']=={'loan':12,'bond':10,'swap':4,'option':4}
    assert abs(summary['loan_ead_usd']-5_000_000)<0.01
    assert abs(summary['bond_market_value_usd']-3_500_000)<0.01
    assert abs(summary['derivative_reference_notional_usd']-1_500_000)<0.01
    assert set(scenarios)=={'Credit Event','Macroeconomic','Geopolitical','Merger / Acquisition','Product Launch'}
    assert all(set(x)=={'mild','moderate','severe'} for x in scenarios.values())
    print(json.dumps({'result':'PASS','portfolio':summary,'event_families':len(scenarios),'scenario_count':sum(len(x) for x in scenarios.values())},indent=2))
if __name__=='__main__':main()
