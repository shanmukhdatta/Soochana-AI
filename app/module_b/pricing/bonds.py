"""Duration/CS01 and rate-convexity instantaneous sensitivity approximation."""
from app.module_b.schemas import Bond

def bond_effect(pos: Bond, rate_bps: float, spread_bps: float) -> dict:
    dr=rate_bps/10_000
    ds=spread_bps/10_000
    # convexity applied to risk-free yield only; spread moves linearized
    rate_delta=pos.market_value_usd*(-pos.modified_duration*dr + 0.5*pos.convexity*dr*dr)
    spread_delta=-pos.market_value_usd*pos.spread_duration*ds
    shocked_market_value=pos.market_value_usd+rate_delta+spread_delta
    if shocked_market_value < 0:
        raise ValueError(f'Bond {pos.id} sensitivity approximation produced a negative market value; reduce scenario size or use full repricing')
    return {'mtm_pnl_usd': rate_delta+spread_delta,
            'rate_pnl_usd': rate_delta, 'spread_pnl_usd': spread_delta,
            'dv01_usd_per_bp': pos.market_value_usd*pos.modified_duration/10_000,
            'cs01_usd_per_bp': pos.market_value_usd*pos.spread_duration/10_000}
