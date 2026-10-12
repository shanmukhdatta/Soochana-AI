"""Recommendation only: ML equity magnitude is NOT a calibrated multi-asset shock."""
import json
from pathlib import Path
from app.module_b.scenarios import template

_p=Path(__file__).resolve()
_root=_p.parents[3] if (_p.parents[3]/'configs').exists() else _p.parents[2]
CONFIG=_root/'configs'/'module_b_config.json'

def propose_scenario(analyzed, severity: str='auto')->dict:
    event=analyzed.event.class_name
    ticker=analyzed.ticker
    impact=analyzed.impact.score
    if severity=='auto':
        bands=json.loads(CONFIG.read_text(encoding='utf-8'))['equity_impact_to_severity_ui_heuristic']
        tier='mild' if impact<=bands['mild_max_score'] else 'moderate' if impact<=bands['moderate_max_score'] else 'severe'
    else: tier=severity
    s=template(event,tier,issuer=ticker)
    needs_review=(analyzed.event.confidence < 0.65)
    return {'suggested_scenario':s,'severity_source':
        'UNVALIDATED UI HEURISTIC: impact score -> stress tier. Not an empirical conversion of abnormal returns into loan PD, rates or spreads.' if severity=='auto'
        else 'User-selected illustrative scenario tier',
        'requires_manual_review':needs_review,
        'justification':f'Class={event}; impact score={impact}/10; proposed tier={tier}. All scenario shocks are configurable assumptions.',
        'risk_signal':analyzed.model_dump(mode='json')}
