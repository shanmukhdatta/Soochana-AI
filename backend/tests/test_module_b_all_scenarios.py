"""Regression-test every stored template; all 15 must run without invalid values."""
import pytest
from app.module_b.portfolio import load_portfolio
from app.module_b.scenarios import template,load_library
from app.module_b.stress_engine import run_stress

ALL=[(family,tier) for family in load_library() for tier in ('mild','moderate','severe')]

@pytest.mark.parametrize('family,tier',ALL)
def test_all_templates_produce_reconciled_finite_values(family,tier):
    p=load_portfolio()
    s=template(family,tier,issuer='AAPL')
    result=run_stress(p,s)
    assert len(result['positions'])==30
    assert result['totals']['illustrative_mixed_horizon_loss_usd']==pytest.approx(
        sum(x['indicative_loss_contribution_usd'] for x in result['positions']),abs=1e-6)
    assert result['scenario']['severity']==tier
