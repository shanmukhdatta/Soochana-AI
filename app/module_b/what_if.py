"""Scenario comparison preserving original synthetic portfolio in memory."""
from app.module_b.schemas import Portfolio,Scenario,ScaleChange
from app.module_b.portfolio import scaled_portfolio
from app.module_b.stress_engine import run_stress

def compare(portfolio:Portfolio,scenario:Scenario,changes:list[ScaleChange])->dict:
    before=run_stress(portfolio,scenario)
    changed=scaled_portfolio(portfolio,changes)
    after=run_stress(changed,scenario)
    key='illustrative_mixed_horizon_net_effect_usd'
    return {'before':before,'after':after,
            'net_effect_change_usd':after['totals'][key]-before['totals'][key],
            'changes':[x.model_dump() for x in changes],
            'warning':'Changing exposure sizes changes risk but does not account for trade costs, hedge cost or financing.'}
