"""Event-informed heuristic portfolio recommendations, not price-direction forecasts."""
from datetime import datetime
import numpy as np
from .config import ModuleAConfig
from .schemas import Signal, RebalanceResponse, WeightChange
from .signals import aggregate
from .constraints import validate_weights,project


def rebalance(cfg:ModuleAConfig, signals:list[Signal], as_of:datetime,
              current_weights:dict[str,float]|None=None,
              strategy:str='full_event_aware')->RebalanceResponse:
    current_weights=current_weights if current_weights is not None else cfg.baseline
    current=validate_weights(current_weights,cfg)
    scores,counts,warnings=aggregate(signals,as_of,cfg,strategy)
    vec=np.array([scores[t] for t in cfg.tickers],dtype=float)
    centered=vec-float(vec.mean())
    desired=np.full(len(cfg.tickers),1/len(cfg.tickers))+cfg.signal_tilt_coefficient*centered
    if all(n==0 for n in counts.values()):
        # No new data means hold current holdings, not magically positive/negative news.
        desired=current.copy()
        warnings.append('No recent usable signals: current portfolio held')
    target=project(desired,current,cfg)
    turnover=float(.5*np.abs(target-current).sum())
    if turnover<cfg.minimum_trade_turnover:
        # Current may be outside permissible position bands following price drift;
        # only skip if retaining current is still inside all constraints.
        try:
            feasible_current=project(current,current,cfg)
            if np.allclose(feasible_current,current,rtol=0,atol=1e-9):
                target=current.copy();turnover=0.0
        except ValueError:
            pass
    cost=turnover*2*cfg.transaction_cost_bps/10000
    changes=[]
    for i,t in enumerate(cfg.tickers):
        delta=float(target[i]-current[i])
        reason=('no change' if abs(delta)<1e-7 else
                f'{"increase" if delta>0 else "decrease"} constrained by portfolio limits; '
                f'aggregated sentiment/importance signal {scores[t]:+.3f} from {counts[t]} distinct headline(s)')
        changes.append(WeightChange(ticker=t,before=float(current[i]),after=float(target[i]),
                                    change_percentage_points=delta*100,signal=scores[t],explanation=reason))
    return RebalanceResponse(as_of=as_of,strategy=strategy,current_weights=dict(zip(cfg.tickers,map(float,current))),
        target_weights=dict(zip(cfg.tickers,map(float,target))),one_way_turnover=turnover,
        estimated_transaction_cost_fraction=cost,signal_counts=counts,changes=changes,warnings=warnings)
