"""Add-on-only routes; no changes to existing /analyze or /module-a routes."""
from fastapi import APIRouter, Request, HTTPException
from pydantic import ValidationError
from app.module_b.schemas import (StressRequest,StressFromNewsRequest,WhatIfRequest,ReverseStressRequest)
from app.module_b.portfolio import load_portfolio,portfolio_summary
from app.module_b.scenarios import load_library
from app.module_b.stress_engine import run_stress
from app.module_b.event_mapper import propose_scenario
from app.module_b.what_if import compare
from app.module_b.reverse_stress import reverse_stress
from app.services.prices import MarketDataError
from app.services.features import UnsupportedTicker

router=APIRouter(prefix='/module-b',tags=['Module B: Synthetic Portfolio Stress Testing'])

@router.get('/portfolio')
def get_portfolio():
    p=load_portfolio()
    return {'summary':portfolio_summary(p),'positions':[x.model_dump() for x in p.positions],
            'disclaimer':'Synthetic research portfolio; no live bank positions or account data.'}

@router.get('/scenarios')
def get_scenarios():
    return {'scenario_library':{k:{j:v.model_dump() for j,v in tiers.items()} for k,tiers in load_library().items()},
            'disclaimer':'Shock templates are hypothetical parameters, not model predictions or calibrated regulatory scenarios.'}

@router.post('/stress')
def stress(req:StressRequest):
    try:return run_stress(req.portfolio or load_portfolio(),req.scenario)
    except (ValueError,ValidationError) as exc:raise HTTPException(422,str(exc)) from exc

@router.post('/stress-from-news')
def stress_from_news(req:StressFromNewsRequest,request:Request):
    engine=getattr(request.app.state,'pipeline',None)
    if engine is None:
        raise HTTPException(503,'Unified Risk Engine has not loaded its three real models')
    try:
        analyzed=engine.analyze(req.news)
        proposed=propose_scenario(analyzed,req.severity)
        scenario=proposed['suggested_scenario']
        # Explicit opt-in prevents automated severity heuristic being mistaken for forecasts.
        answer={'risk_signal':proposed['risk_signal'],
                'suggested_scenario':scenario.model_dump(),
                'justification':proposed['justification'],
                'severity_source':proposed['severity_source'],
                'requires_manual_review':proposed['requires_manual_review'],
                'executed':req.run_stress,
                'warning':'Issuer and sector stress shocks are user-reviewable hypothetical assumptions. XGBoost equity impact score does not predict loan defaults, bond spreads or interest-rate changes.'}
        if req.run_stress:
            answer['stress_result']=run_stress(req.portfolio or load_portfolio(),scenario)
        return answer
    except (UnsupportedTicker, ValueError, ValidationError) as exc:
        raise HTTPException(422,str(exc)) from exc
    except MarketDataError as exc:
        raise HTTPException(503,'Market data unavailable: '+str(exc)) from exc

@router.post('/what-if')
def what_if(req:WhatIfRequest):
    try:return compare(req.portfolio or load_portfolio(),req.scenario,req.changes)
    except (ValueError,ValidationError) as exc:raise HTTPException(422,str(exc)) from exc

@router.post('/reverse-stress')
def reverse_stress_route(req:ReverseStressRequest):
    try:return reverse_stress(req.portfolio or load_portfolio(),req.scenario,
                              req.loss_budget_usd,req.max_multiplier,req.steps)
    except (ValueError,ValidationError) as exc:raise HTTPException(422,str(exc)) from exc
