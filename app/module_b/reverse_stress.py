"""Grid search: DO NOT assume monotonic scenario loss because derivatives may hedge."""
from app.module_b.schemas import Portfolio,Scenario
from app.module_b.scenarios import scale_scenario
from app.module_b.stress_engine import run_stress

def reverse_stress(portfolio:Portfolio,scenario:Scenario,loss_budget_usd:float,
                   max_multiplier:float=4,steps:int=81)->dict:
    samples=[];first=None;stopped_at=None;stop_reason=None;coverage=None
    for i in range(steps):
        factor=max_multiplier*i/(steps-1)
        try:
            scaled=scale_scenario(scenario,factor)
            result=run_stress(portfolio,scaled)
            if coverage is None:
                coverage=result['override_coverage']
        except ValueError as exc:
            # Never silently extrapolate past nonnegative underlying price/volatility
            # or permitted shock bounds; report the truncated feasible grid.
            stopped_at=factor
            stop_reason=str(exc)
            break
        loss=result['totals']['illustrative_mixed_horizon_loss_usd']
        entry={'multiplier':factor,'loss_usd':loss,'breaches_budget':loss>=loss_budget_usd}
        samples.append(entry)
        if first is None and entry['breaches_budget']:first=entry
    monotone=all(b['loss_usd']>=a['loss_usd']-1e-6 for a,b in zip(samples,samples[1:]))
    return {'loss_budget_usd':loss_budget_usd,'found_breach':first is not None,
            'first_breach_on_grid':first,'max_multiplier':max_multiplier,'steps':steps,
            'monotone_loss_on_grid':monotone,'trace':samples,
            'truncated_at_multiplier':stopped_at,'truncation_reason':stop_reason,
            'largest_evaluated_multiplier':samples[-1]['multiplier'] if samples else None,
            'override_coverage':coverage,
            'method':'First sampled threshold breach on deterministic grid; not an exact analytical or globally minimal solution.',
            'warning':'Signed derivatives and mixed risk factors can make loss non-monotonic. Use finer grid for exploration; mixed-horizon loss is not regulatory capital.'}
