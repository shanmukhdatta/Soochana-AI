"""12-month expected credit loss. NOT cashflow-discounted loan fair value."""
from app.module_b.schemas import Loan

def loan_effect(pos: Loan, pd_multiplier: float) -> dict:
    pd_stressed = min(1.0, pos.pd_12m * pd_multiplier)
    base_ecl = pos.ead_usd * pos.pd_12m * pos.lgd
    stress_ecl = pos.ead_usd * pd_stressed * pos.lgd
    return {'base_ecl_12m_usd': base_ecl, 'stress_ecl_12m_usd': stress_ecl,
            'incremental_ecl_12m_usd': stress_ecl-base_ecl,
            'base_pd_12m': pos.pd_12m, 'stress_pd_12m': pd_stressed}
