"""Daily-close portfolio performance metrics; no invented returns."""
import numpy as np


def performance(nav:list[float],turnovers:list[float],costs:list[float],trade_count:int)->dict:
    a=np.asarray(nav,dtype=float)
    if len(a)<2 or not np.all(np.isfinite(a)) or np.any(a<=0):
        raise ValueError('Need at least two positive daily net asset values')
    daily=a[1:]/a[:-1]-1.0
    v=float(daily.std(ddof=1)*np.sqrt(252)) if len(daily)>1 else 0.0
    years=len(daily)/252
    total=float(a[-1]/a[0]-1)
    annualized=float((a[-1]/a[0])**(1/years)-1)
    peak=np.maximum.accumulate(a)
    drawdown=float(np.min(a/peak-1))
    return {'initial_nav':float(a[0]),'final_nav':float(a[-1]),'total_return':total,
        'annualized_return':annualized,'annualized_volatility':v,
        'sharpe_zero_riskfree':float(np.mean(daily)*np.sqrt(252)/np.std(daily,ddof=1)) if v>0 else None,
        'maximum_drawdown':drawdown,'total_one_way_turnover':float(sum(turnovers)),
        'total_transaction_costs_nav_units':float(sum(costs)),'trade_count':trade_count}
