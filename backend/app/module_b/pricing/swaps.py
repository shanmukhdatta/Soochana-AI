"""Signed DV01 shock approximation, not full curve/cashflow repricing."""
from app.module_b.schemas import Swap

def swap_effect(pos: Swap, rate_bps: float) -> dict:
    sign=1 if pos.side=='pay_fixed' else -1
    return {'mtm_pnl_usd': sign*pos.dv01_usd_per_bp*rate_bps,
            'signed_dv01_usd_per_bp': sign*pos.dv01_usd_per_bp}
