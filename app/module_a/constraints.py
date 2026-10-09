"""Convex bounded portfolio projection using SciPy SLSQP with trade auxiliaries.

Solves min ||w - raw_target||² given sum(w)=1, position and sector bands,
and 0.5*sum(abs(w-current)) <= one_way_turnover_limit.
This is a portfolio *simulator*, not an order execution algorithm.
"""
import numpy as np
from scipy.optimize import minimize
from .config import ModuleAConfig

class InfeasiblePortfolio(ValueError):
    pass


def validate_weights(weights:dict[str,float], cfg:ModuleAConfig)->np.ndarray:
    if set(weights)!=set(cfg.tickers):
        raise ValueError('Current weights must contain exactly the index constituents')
    x=np.array([weights[t] for t in cfg.tickers],dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x < -1e-10) or abs(float(x.sum())-1)>1e-6:
        raise ValueError('Current weights must be finite, nonnegative and sum to 1')
    return x


def project(desired:np.ndarray, current:np.ndarray, cfg:ModuleAConfig)->np.ndarray:
    n=len(cfg.tickers)
    desired=np.asarray(desired,dtype=float)
    current=np.asarray(current,dtype=float)
    if desired.shape!=(n,) or current.shape!=(n,) or not np.all(np.isfinite(desired)):
        raise ValueError('Invalid target/current vector shape or finiteness')
    base=np.full(n,1/n)
    lo=np.maximum(0.0,base-cfg.position_deviation_limit)
    hi=np.minimum(1.0,base+cfg.position_deviation_limit)
    if (np.allclose(desired,current,rtol=0,atol=1e-12)
        and np.all(current>=lo-1e-10) and np.all(current<=hi+1e-10)
        and all(abs(sum(current[i] for i,t in enumerate(cfg.tickers) if cfg.sector_map[t]==s)-b)
                <=cfg.sector_deviation_limit+1e-10 for s,b in cfg.sector_targets.items())):
        return current.copy()
    # Optimization variables: portfolio weights w and auxiliary absolute trades u.
    # u_i >= +/- (w_i - current_i); sum(u)<=2*one-way turnover.
    A=[]; lower=[]; upper=[]
    def row_constraint(a,lb,ub):
        A.append(a);lower.append(lb);upper.append(ub)
    eq=np.zeros(2*n);eq[:n]=1
    for sector,target in cfg.sector_targets.items():
        a=np.zeros(2*n)
        a[:n]=[float(cfg.sector_map[t]==sector) for t in cfg.tickers]
        if sum(cfg.sector_map[t]==sector for t in cfg.tickers)<n:
            row_constraint(a,max(0,target-cfg.sector_deviation_limit),min(1,target+cfg.sector_deviation_limit))
    for i in range(n):
        a=np.zeros(2*n);a[i]=-1;a[n+i]=1
        row_constraint(a,-current[i],np.inf) # u - w >= -current
        a=np.zeros(2*n);a[i]=1;a[n+i]=1
        row_constraint(a,current[i],np.inf) # u + w >= current
    a=np.zeros(2*n);a[n:]=1
    row_constraint(a,0,2*cfg.one_way_turnover_limit)
    mat=np.array(A);l=np.array(lower);u=np.array(upper)
    from scipy.optimize import LinearConstraint
    lin=[LinearConstraint(eq,1.0,1.0),LinearConstraint(mat,l,u)]
    bounds=[(float(lo[i]),float(hi[i])) for i in range(n)]+[(0.0,2.0) for _ in range(n)]
    # Initialize close to a feasible weight vector. Current is used if inside position bands.
    start_w=current.copy() if np.all(current>=lo-1e-10) and np.all(current<=hi+1e-10) else base.copy()
    start=np.r_[start_w,np.abs(start_w-current)]
    def fun(z):return float(np.dot(z[:n]-desired,z[:n]-desired))
    def jac(z):return np.r_[2*(z[:n]-desired),np.zeros(n)]
    opt=minimize(fun,start,jac=jac,method='SLSQP',bounds=bounds,constraints=lin,
                 options={'ftol':1e-11,'maxiter':500})
    w=np.array(opt.x[:n])
    tol=2e-5
    if (not opt.success or abs(w.sum()-1)>tol or np.any(w<lo-tol) or np.any(w>hi+tol)
        or .5*np.abs(w-current).sum()>cfg.one_way_turnover_limit+tol
        or any(abs(sum(w[i] for i,t in enumerate(cfg.tickers) if cfg.sector_map[t]==s)-v)
               >cfg.sector_deviation_limit+tol for s,v in cfg.sector_targets.items())):
        raise InfeasiblePortfolio(f'Constrained rebalance infeasible or optimizer failed: {opt.message}')
    # Keep full precision; only presentation rounds numbers.
    return w
