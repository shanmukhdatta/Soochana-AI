import json
from pathlib import Path
import pytest
from app.module_b.schemas import Portfolio, Loan, ScaleChange, Shocks, Scenario
from app.module_b.portfolio import load_portfolio, portfolio_summary,scaled_portfolio
from app.module_b.scenarios import template,merge_shocks,load_library
from app.module_b.stress_engine import run_stress


def neutral():return Scenario(name='Neutral',shocks=Shocks())

def test_synthetic_portfolio_counts():
    p=load_portfolio();s=portfolio_summary(p)
    assert s['position_count']==30
    assert s['counts']=={'loan':12,'bond':10,'swap':4,'option':4}
    assert s['loan_ead_usd']==5_000_000
    assert s['bond_market_value_usd']==3_500_000
    assert s['derivative_reference_notional_usd']==1_500_000

def test_portfolio_rejects_duplicate_ids():
    p=load_portfolio()
    with pytest.raises(ValueError,match='unique'):
        Portfolio(name='Duplicate',positions=[p.positions[0],p.positions[0]])

def test_scenarios_complete():
    library=load_library()
    assert len(library)==5
    assert all(set(tiers)=={'mild','moderate','severe'} for tiers in library.values())

def test_issuer_template_substitution():
    s=template('Credit Event','moderate',issuer='AAPL')
    assert '__ISSUER__' not in s.issuer_shocks
    assert 'AAPL' in s.issuer_shocks
    with pytest.raises(ValueError,match='requires a ticker'):
        template('Credit Event','mild')

def test_shocks_additive_and_multiplicative():
    global_=Shocks(rate_bps=100,spread_bps=30,equity_pct=-10,volatility_points=.05,pd_multiplier=1.2)
    sector=Shocks(spread_bps=20,pd_multiplier=1.5)
    issuer=Shocks(equity_pct=-3,pd_multiplier=2)
    s=merge_shocks(global_,sector,issuer)
    assert s.rate_bps==100 and s.spread_bps==50 and s.equity_pct==-13
    assert s.pd_multiplier==pytest.approx(3.6)

def test_zero_shock_every_instrument():
    p=load_portfolio();out=run_stress(p,neutral())
    assert out['totals']['incremental_12m_expected_credit_loss_usd']==pytest.approx(0,abs=1e-9)
    assert out['totals']['instantaneous_mtm_pnl_usd']==pytest.approx(0,abs=1e-9)
    assert out['totals']['illustrative_mixed_horizon_net_effect_usd']==pytest.approx(0,abs=1e-9)

def test_reconcile_position_type_sector():
    p=load_portfolio();out=run_stress(p,template('Credit Event','moderate',issuer='AAPL'))
    positions=out['positions'];expected=sum(row['indicative_loss_contribution_usd'] for row in positions)
    assert expected==pytest.approx(sum(out['loss_contribution_by_type_usd'].values()))
    assert expected==pytest.approx(sum(out['loss_contribution_by_sector_usd'].values()))
    assert expected==pytest.approx(out['totals']['illustrative_mixed_horizon_loss_usd'])
    assert all(row['ticker'] or row['type']=='swap' for row in positions)

def test_custom_issuer_scope():
    p=Portfolio(name='Two',positions=[Loan(id='a',name='A',sector='Tech',ticker='AAPL',ead_usd=1000,pd_12m=.1,lgd=.5),Loan(id='b',name='B',sector='Tech',ticker='MSFT',ead_usd=1000,pd_12m=.1,lgd=.5)])
    s=Scenario(name='One issuer',shocks=Shocks(),issuer_shocks={'AAPL':Shocks(pd_multiplier=2)})
    out=run_stress(p,s)
    assert out['positions'][0]['incremental_ecl_12m_usd']==pytest.approx(50)
    assert out['positions'][1]['incremental_ecl_12m_usd']==pytest.approx(0)

def test_what_if_no_mutation():
    p=load_portfolio(); orig=p.model_dump_json()
    new=scaled_portfolio(p,[ScaleChange(position_id='L01',scale=.5),ScaleChange(position_id='OPT01',scale=1.25)])
    assert new.positions[0].ead_usd==p.positions[0].ead_usd*.5
    assert p.model_dump_json()==orig
    with pytest.raises(ValueError,match='Unknown position'):
        scaled_portfolio(p,[ScaleChange(position_id='bad',scale=0.5)])

def test_what_if_reject_duplicate_changes():
    from app.module_b.schemas import WhatIfRequest
    with pytest.raises(ValueError,match='Duplicate'):
        WhatIfRequest(scenario=neutral(), changes=[ScaleChange(position_id='L01',scale=1),ScaleChange(position_id='L01',scale=0.5)])


def test_unresolved_issuer_template_must_not_silently_ignore_target_shock():
    raw=load_library()['Credit Event']['moderate']
    with pytest.raises(ValueError,match='unresolved __ISSUER__'):
        run_stress(load_portfolio(),raw)
