"""Regression checks against silently ignored issuer/sector stress overrides."""
import pytest
from pydantic import ValidationError
from app.module_b.portfolio import load_portfolio
from app.module_b.schemas import Scenario, Shocks
from app.module_b.stress_engine import run_stress


def test_lowercase_issuer_override_matches_held_ticker():
    s=Scenario(name='Lowercase issuer',shocks=Shocks(),issuer_shocks={'aapl':Shocks(pd_multiplier=2)})
    assert 'AAPL' in s.issuer_shocks
    result=run_stress(load_portfolio(),s)
    assert result['override_coverage']['all_overrides_matched']
    assert result['positions'][0]['incremental_ecl_12m_usd']>0


def test_duplicate_issuer_after_normalization_is_rejected():
    with pytest.raises(ValidationError,match='Duplicate override'):
        Scenario(name='Ambiguous',shocks=Shocks(),issuer_shocks={
            'aapl':Shocks(pd_multiplier=2),'AAPL':Shocks(pd_multiplier=3)})


def test_unused_issuer_or_sector_is_reported_in_result():
    s=Scenario(name='Unmatched',shocks=Shocks(),issuer_shocks={'UNKNOWN':Shocks(pd_multiplier=2)},
               sector_shocks={'Technolgy':Shocks(spread_bps=100)})
    result=run_stress(load_portfolio(),s)
    assert result['override_coverage']['unmatched_issuer_shocks']==['UNKNOWN']
    assert result['override_coverage']['unmatched_sector_shocks']==['Technolgy']
    assert not result['override_coverage']['all_overrides_matched']
    assert any('matched no portfolio holdings' in w for w in result['warnings'])


def test_sector_override_exact_match():
    s=Scenario(name='Sector',shocks=Shocks(),sector_shocks={'Technology':Shocks(spread_bps=100)})
    result=run_stress(load_portfolio(),s)
    assert result['override_coverage']['all_overrides_matched']
    assert any(x['applied_shocks']['spread_bps']==100 for x in result['positions'] if x['sector']=='Technology')


def test_reverse_stress_does_not_hide_unused_override():
    from app.module_b.reverse_stress import reverse_stress
    scenario=Scenario(name='Missing issuer',shocks=Shocks(),issuer_shocks={'UNKNOWN':Shocks(pd_multiplier=2)})
    result=reverse_stress(load_portfolio(),scenario,loss_budget_usd=1000,max_multiplier=1,steps=11)
    assert result['override_coverage']['unmatched_issuer_shocks']==['UNKNOWN']
