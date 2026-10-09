"""Deterministic stress engine with explicit instantaneous MTM vs 12m expected-loss separation."""
from __future__ import annotations
from collections import defaultdict
import math
from app.module_b.schemas import Portfolio,Scenario,Loan,Bond,Swap,Option
from app.module_b.scenarios import effective_shocks
from app.module_b.pricing.loans import loan_effect
from app.module_b.pricing.bonds import bond_effect
from app.module_b.pricing.swaps import swap_effect
from app.module_b.pricing.options import option_effect
from app.module_b.portfolio import portfolio_summary

def run_stress(portfolio:Portfolio,scenario:Scenario)->dict:
    if '__ISSUER__' in scenario.issuer_shocks:
        raise ValueError('Scenario has unresolved __ISSUER__ placeholder. Substitute the relevant ticker before running the stress.')
    # Report any targeted shock that matches no holding rather than silently
    # ignoring a typo or inadvertently targeting a different issuer/sector.
    held_tickers={(p.ticker or '').upper() for p in portfolio.positions if p.ticker}
    held_sectors={p.sector for p in portfolio.positions}
    unmatched_issuers=sorted(set(scenario.issuer_shocks)-held_tickers)
    unmatched_sectors=sorted(set(scenario.sector_shocks)-held_sectors)
    results=[]
    totals={'loan':0.0,'bond':0.0,'swap':0.0,'option':0.0}
    by_sector=defaultdict(float)
    by_ticker=defaultdict(float)
    base_ecl=0.0; stress_ecl=0.0; mtm=0.0
    for pos in portfolio.positions:
        sh=effective_shocks(scenario,pos.sector,pos.ticker)
        if isinstance(pos,Loan):
            d=loan_effect(pos,sh.pd_multiplier)
            ecl_inc=d['incremental_ecl_12m_usd']
            base_ecl+=d['base_ecl_12m_usd'];stress_ecl+=d['stress_ecl_12m_usd']
            pnl=0.0
        elif isinstance(pos,Bond):
            d=bond_effect(pos,sh.rate_bps,sh.spread_bps);pnl=d['mtm_pnl_usd'];ecl_inc=0.0
        elif isinstance(pos,Swap):
            d=swap_effect(pos,sh.rate_bps);pnl=d['mtm_pnl_usd'];ecl_inc=0.0
        elif isinstance(pos,Option):
            d=option_effect(pos,sh.equity_pct,sh.volatility_points,sh.rate_bps)
            pnl=d['mtm_pnl_usd'];ecl_inc=0.0
        else:raise TypeError('Unsupported position type')
        mtm+=pnl
        # Positive contribution means risk loss; negative means offsetting gain.
        contribution=ecl_inc-pnl
        totals[pos.type]+=contribution
        by_sector[pos.sector]+=contribution
        if pos.ticker:by_ticker[pos.ticker]+=contribution
        results.append({'id':pos.id,'name':pos.name,'type':pos.type,'sector':pos.sector,
                        'ticker':pos.ticker,'applied_shocks':sh.model_dump(),
                        'incremental_ecl_12m_usd':ecl_inc,'mtm_pnl_usd':pnl,
                        'indicative_loss_contribution_usd':contribution,'calculation':d})
    combined=mtm-(stress_ecl-base_ecl)
    out={'module':'riskpulse-module-b-v1','portfolio':portfolio_summary(portfolio),
         'scenario':scenario.model_dump(),
         'override_coverage':{
             'unmatched_issuer_shocks':unmatched_issuers,
             'unmatched_sector_shocks':unmatched_sectors,
             'all_overrides_matched':not unmatched_issuers and not unmatched_sectors,
         },
         'totals':{'instantaneous_mtm_pnl_usd':mtm,
                   'base_12m_expected_credit_loss_usd':base_ecl,
                   'stressed_12m_expected_credit_loss_usd':stress_ecl,
                   'incremental_12m_expected_credit_loss_usd':stress_ecl-base_ecl,
                   'illustrative_mixed_horizon_net_effect_usd':combined,
                   'illustrative_mixed_horizon_loss_usd':-combined},
         'loss_contribution_by_type_usd':dict(totals),
         'loss_contribution_by_sector_usd':dict(sorted(by_sector.items())),
         'loss_contribution_by_ticker_usd':dict(sorted(by_ticker.items())),
         'top_loss_contributors':sorted(results,key=lambda x:x['indicative_loss_contribution_usd'],reverse=True)[:10],
         'positions':results,
         'warnings':[
          'Synthetic illustrative positions and shocks, not market forecasts or investment advice.',
          'MTM revaluation is instantaneous; loan ECL change is projected 12-month loss. Mixed-horizon indicator is illustrative, not economic NAV, regulatory capital, VaR or realized P&L.',
          'No credit migration term structures, default dependence, full swap curve repricing, options early exercise, collateral, liquidity, tax, basis risk or counterparty wrong-way risk.',
          'Options are non-dividend/continuous-dividend European Black-Scholes approximations; swaps use linear DV01; bonds use duration/spread approximation.'
         ] + (['WARNING: issuer shock overrides matched no portfolio holdings: '+', '.join(unmatched_issuers)]
              if unmatched_issuers else []) + (['WARNING: sector shock overrides matched no portfolio holdings: '+', '.join(unmatched_sectors)]
              if unmatched_sectors else [])}
    # Fail loudly on numerical issues; never return NaN/Infinity inside API JSON.
    def check(v):
        if isinstance(v,dict):
            for child in v.values():check(child)
        elif isinstance(v,list):
            for child in v:check(child)
        elif isinstance(v,(int,float)) and not isinstance(v,bool) and not math.isfinite(v):
            raise ValueError('Nonfinite valuation; check scenario and instruments')
    check(out)
    return out
