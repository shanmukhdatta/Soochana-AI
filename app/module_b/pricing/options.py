"""European Black-Scholes, continuous dividend yield; deterministic repricing."""
import math
from app.module_b.schemas import Option

def _norm_cdf(x:float)->float:
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def black_scholes(kind: str, spot: float, strike: float, t: float, sigma: float, rate: float, dividend: float=0) -> float:
    if not (spot>0 and strike>0 and t>0 and sigma>0):
        raise ValueError('Option spot/strike/time/vol must be positive')
    root=sigma*math.sqrt(t)
    d1=(math.log(spot/strike)+(rate-dividend+0.5*sigma*sigma)*t)/root
    d2=d1-root
    disc_s=spot*math.exp(-dividend*t)
    disc_k=strike*math.exp(-rate*t)
    if kind=='call': value=disc_s*_norm_cdf(d1)-disc_k*_norm_cdf(d2)
    elif kind=='put': value=disc_k*_norm_cdf(-d2)-disc_s*_norm_cdf(-d1)
    else: raise ValueError('Unknown option kind')
    return max(0.0,value)

def option_effect(pos: Option, equity_pct: float, volatility_points: float, rate_bps: float) -> dict:
    spot_new=pos.underlying_price_usd*(1+equity_pct/100)
    sigma_new=pos.implied_volatility+volatility_points
    rate_new=pos.risk_free_rate+rate_bps/10_000
    if spot_new <= 0 or sigma_new <= 0:
        raise ValueError(f'Invalid shocked spot or volatility for option {pos.id}; reduce scenario magnitude')
    base=black_scholes(pos.kind,pos.underlying_price_usd,pos.strike_usd,pos.time_years,
                       pos.implied_volatility,pos.risk_free_rate,pos.dividend_yield)
    stressed=black_scholes(pos.kind,spot_new,pos.strike_usd,pos.time_years,
                           sigma_new,rate_new,pos.dividend_yield)
    scale=pos.contracts*pos.contract_multiplier*(1 if pos.side=='long' else -1)
    return {'mtm_pnl_usd':(stressed-base)*scale,
            'base_position_value_usd':base*scale, 'stressed_position_value_usd':stressed*scale,
            'shocked_underlying_price_usd':spot_new,'shocked_volatility':sigma_new}
