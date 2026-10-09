"""From c2c root: python -m scripts.verify_module_b_scenarios"""
import json
import math
from app.module_b.portfolio import load_portfolio
from app.module_b.scenarios import load_library,template
from app.module_b.stress_engine import run_stress


def main():
    p=load_portfolio()
    total=0
    for event in load_library():
        for tier in ['mild','moderate','severe']:
            s=template(event,tier,issuer='AAPL')
            result=run_stress(p,s)
            mtm=result['totals']['instantaneous_mtm_pnl_usd']
            ecl=result['totals']['incremental_12m_expected_credit_loss_usd']
            mixed=result['totals']['illustrative_mixed_horizon_loss_usd']
            assert len(result['positions'])==30
            assert all(math.isfinite(x) for x in [mtm,ecl,mixed])
            assert abs((ecl-mtm)-mixed)<1e-6
            assert abs(sum(result['loss_contribution_by_type_usd'].values())-mixed)<1e-6
            total+=1
            print(f'{event:23s} {tier:8s} instantaneous_mtm_usd={mtm:13.2f} delta_ecl_12m_usd={ecl:11.2f} mixed_screening_loss_usd={mixed:13.2f}')
    assert total==15
    print(f'PASS: all {total} synthetic scenario cases reconciled')
if __name__=='__main__':main()
