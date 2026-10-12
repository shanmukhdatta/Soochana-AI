"""Load/validate a synthetic portfolio; never read external real customer positions."""
from __future__ import annotations
from pathlib import Path
import json
from app.module_b.schemas import Portfolio, Loan, Bond, Swap, Option, ScaleChange

_p=Path(__file__).resolve()
ROOT=_p.parents[3] if (_p.parents[3]/'data').exists() else _p.parents[2]
DEFAULT_PORTFOLIO=ROOT/'data'/'module_b'/'synthetic_portfolio.json'

def load_portfolio(path:Path|str=DEFAULT_PORTFOLIO)->Portfolio:
    return Portfolio.model_validate_json(Path(path).read_text(encoding='utf-8'))

def portfolio_summary(portfolio:Portfolio)->dict:
    loans=[p for p in portfolio.positions if isinstance(p,Loan)]
    bonds=[p for p in portfolio.positions if isinstance(p,Bond)]
    swaps=[p for p in portfolio.positions if isinstance(p,Swap)]
    opts=[p for p in portfolio.positions if isinstance(p,Option)]
    return {'name':portfolio.name,'portfolio_version':portfolio.portfolio_version,
            'currency':portfolio.currency,'position_count':len(portfolio.positions),
            'counts':{'loan':len(loans),'bond':len(bonds),'swap':len(swaps),'option':len(opts)},
            'loan_ead_usd':sum(p.ead_usd for p in loans),
            'bond_market_value_usd':sum(p.market_value_usd for p in bonds),
            'derivative_reference_notional_usd':sum(p.notional_usd for p in swaps)+sum(p.underlying_price_usd*p.contracts*p.contract_multiplier for p in opts),
            'exposure_warning':'Loan EAD, bond market value and derivative reference notional use different measurement bases and are NOT additive portfolio NAV.'}

def scaled_portfolio(portfolio:Portfolio, changes:list[ScaleChange])->Portfolio:
    by_id={c.position_id:c.scale for c in changes}
    unknown=set(by_id)-{p.id for p in portfolio.positions}
    if unknown: raise ValueError('Unknown position ID(s): '+', '.join(sorted(unknown)))
    updated=[]
    for p in portfolio.positions:
        scale=by_id.get(p.id,1.0)
        if scale == 0: continue
        new=p.model_copy(deep=True)
        if isinstance(new,Loan): new.ead_usd*=scale
        elif isinstance(new,Bond): new.market_value_usd*=scale
        elif isinstance(new,Swap):
            new.notional_usd*=scale
            new.dv01_usd_per_bp*=scale
        elif isinstance(new,Option): new.contracts*=scale
        updated.append(type(new).model_validate(new.model_dump()))
    if not updated:raise ValueError('Cannot remove every position')
    return Portfolio(name=portfolio.name+' (what-if)',currency=portfolio.currency,
                     portfolio_version=portfolio.portfolio_version,positions=updated)
