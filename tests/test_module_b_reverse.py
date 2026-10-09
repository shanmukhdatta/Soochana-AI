import pytest
from app.module_b.schemas import Portfolio,Loan,Shocks,Scenario
from app.module_b.reverse_stress import reverse_stress
from app.module_b.what_if import compare
from app.module_b.schemas import ScaleChange


def test_reverse_known_threshold():
    p=Portfolio(name='One',positions=[Loan(id='L',name='L',sector='Technology',ead_usd=1_000_000,pd_12m=.02,lgd=.5)])
    s=Scenario(name='PD',shocks=Shocks(pd_multiplier=3))
    out=reverse_stress(p,s,loss_budget_usd=10_000,max_multiplier=2,steps=41)
    # target 1e6*.02*.5*(3-1)*factor=20000*factor -> threshold at 0.5
    assert out['first_breach_on_grid']['multiplier']==pytest.approx(.5)
    assert out['monotone_loss_on_grid']

def test_reverse_no_breach():
    p=Portfolio(name='One',positions=[Loan(id='L',name='L',sector='Tech',ead_usd=1000,pd_12m=.02,lgd=.5)])
    s=Scenario(name='Neutral',shocks=Shocks())
    out=reverse_stress(p,s,loss_budget_usd=1000,max_multiplier=2,steps=11)
    assert not out['found_breach'] and out['first_breach_on_grid'] is None

def test_what_if_portfolio_effect_and_preserved():
    p=Portfolio(name='One',positions=[Loan(id='L',name='L',sector='Tech',ead_usd=1e6,pd_12m=.02,lgd=.5)])
    s=Scenario(name='PD',shocks=Shocks(pd_multiplier=2))
    r=compare(p,s,[ScaleChange(position_id='L',scale=.5)])
    assert r['before']['totals']['illustrative_mixed_horizon_loss_usd']==pytest.approx(10000)
    assert r['after']['totals']['illustrative_mixed_horizon_loss_usd']==pytest.approx(5000)
    assert p.positions[0].ead_usd==1e6


def test_reverse_physically_invalid_shocks_explicitly_truncated():
    p=Portfolio(name='One',positions=[Loan(id='L',name='L',sector='Tech',ead_usd=1000,pd_12m=.02,lgd=.5)])
    s=Scenario(name='Equity crash',shocks=Shocks(equity_pct=-50))
    out=reverse_stress(p,s,loss_budget_usd=1,max_multiplier=4,steps=81)
    assert out['truncated_at_multiplier'] is not None
    assert len(out['trace']) < 81
    assert out['truncation_reason']
