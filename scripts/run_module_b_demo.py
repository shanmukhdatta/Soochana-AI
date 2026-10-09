"""From c2c root: python -m scripts.run_module_b_demo (no models/network required)."""
import json
from pathlib import Path
from app.module_b.portfolio import load_portfolio
from app.module_b.scenarios import template
from app.module_b.stress_engine import run_stress
from app.module_b.reverse_stress import reverse_stress

def main():
    p=load_portfolio()
    s=template('Credit Event','moderate',issuer='AAPL')
    result=run_stress(p,s)
    rev=reverse_stress(p,s,loss_budget_usd=100_000,max_multiplier=2,steps=41)
    out=Path('data/module_b/results')
    out.mkdir(parents=True,exist_ok=True)
    (out/'credit_stress_demo.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (out/'reverse_stress_demo.json').write_text(json.dumps(rev,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'scenario':s.name,'totals':result['totals'],'top_loss_contributors':result['top_loss_contributors'][:3],
        'reverse_stress_first_breach':rev['first_breach_on_grid'],'files':[str(out/'credit_stress_demo.json'),str(out/'reverse_stress_demo.json')]},indent=2,allow_nan=False))
if __name__=='__main__':main()
