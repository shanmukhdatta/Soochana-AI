"""Hand-documented adverse stress scenarios; NOT predictions of rates, defaults, or credit spreads."""
import json
from pathlib import Path
from app.module_b.schemas import Scenario, Shocks

_p=Path(__file__).resolve()
ROOT=_p.parents[3] if (_p.parents[3]/'data').exists() else _p.parents[2]
LIBRARY=ROOT/'data'/'module_b'/'scenario_library.json'

def load_library(path:Path|str=LIBRARY)->dict[str,dict[str,Scenario]]:
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    return {event:{severity:Scenario.model_validate(value) for severity,value in entries.items()} for event,entries in raw.items()}

def template(event_class:str,severity:str,issuer:str|None=None)->Scenario:
    lib=load_library()
    if event_class not in lib: raise ValueError('Unknown event family')
    if severity not in lib[event_class]: raise ValueError('Unknown severity tier')
    s=lib[event_class][severity].model_copy(deep=True)
    # Templates for issuer-specific events contain issuer_increment: apply it to the relevant ticker.
    # The placeholder "__ISSUER__" must NEVER be carried to calculations as a real ticker.
    if '__ISSUER__' in s.issuer_shocks:
        extra=s.issuer_shocks.pop('__ISSUER__')
        if not issuer: raise ValueError('Issuer-specific scenario requires a ticker')
        s.issuer_shocks[issuer.upper()]=extra
    return s

def merge_shocks(global_:Shocks,sector:Shocks|None,issuer:Shocks|None)->Shocks:
    """Sector/issuer fields are ADDITIVE deviations, except PD multipliers multiply.
    Unset override fields use schema defaults (0 for shocks, 1 for PD).
    """
    modifiers=[x for x in (sector,issuer) if x is not None]
    def add(name):return getattr(global_,name)+sum(getattr(x,name) for x in modifiers)
    pd=global_.pd_multiplier
    for x in modifiers:pd*=x.pd_multiplier
    # Strict bounds prevent mistakes such as 2000 bps + 2000 bps silently clipping.
    return Shocks(rate_bps=add('rate_bps'),spread_bps=add('spread_bps'),
                  equity_pct=add('equity_pct'),volatility_points=add('volatility_points'),pd_multiplier=pd)

def effective_shocks(s:Scenario,sector:str,ticker:str|None)->Shocks:
    return merge_shocks(s.shocks,s.sector_shocks.get(sector),s.issuer_shocks.get((ticker or '').upper()))

def scale_shocks(shocks:Shocks,factor:float)->Shocks:
    """Interpolates shocks from neutral factor=0 to input scenario factor=1."""
    return Shocks(rate_bps=shocks.rate_bps*factor,
                  spread_bps=shocks.spread_bps*factor,
                  equity_pct=shocks.equity_pct*factor,
                  volatility_points=shocks.volatility_points*factor,
                  pd_multiplier=1+(shocks.pd_multiplier-1)*factor)

def scale_scenario(s:Scenario,factor:float)->Scenario:
    return s.model_copy(update={'shocks':scale_shocks(s.shocks,factor),
        'sector_shocks':{k:scale_shocks(v,factor) for k,v in s.sector_shocks.items()},
        'issuer_shocks':{k:scale_shocks(v,factor) for k,v in s.issuer_shocks.items()},
        'name':s.name+f' [x{factor:.3f}]'})
