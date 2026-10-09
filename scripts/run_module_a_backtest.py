"""Run offline Module A research comparison without loading FinBERT models."""
import argparse,json
from datetime import date
from pathlib import Path
from app.module_a.config import ModuleAConfig
from app.module_a.backtester import read_prices,read_signals,run_backtest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-id',default='holdout')
    ap.add_argument('--start-date',default=None)
    ap.add_argument('--end-date',default=None)
    args=ap.parse_args()
    if not args.dataset_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid dataset ID')
    cfg=ModuleAConfig.load()
    root=Path(__file__).resolve().parents[1]
    data=root/'data'/'module_a'
    prices=read_prices(data/f'{args.dataset_id}_prices.csv',cfg)
    signals=read_signals(data/f'{args.dataset_id}_signals.csv',cfg)
    report=run_backtest(cfg,prices,signals,
        date.fromisoformat(args.start_date) if args.start_date else None,
        date.fromisoformat(args.end_date) if args.end_date else None)
    output=data/'results'/f'{args.dataset_id}_offline_comparison.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',output)
    print('Strategy                     Total Return    Drawdown   Turnover   Trades')
    for mode,rec in report['evaluations'].items():
        m=rec['metrics']
        print(f'{mode:29}{m["total_return"]:11.2%} {m["maximum_drawdown"]:11.2%} '
              f'{m["total_one_way_turnover"]:10.2%} {m["trade_count"]:7d}')

if __name__=='__main__':main()
