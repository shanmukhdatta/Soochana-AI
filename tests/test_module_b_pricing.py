import math
import pytest
from app.module_b.schemas import Loan,Bond,Swap,Option,Shocks,Scenario
from app.module_b.pricing.loans import loan_effect
from app.module_b.pricing.bonds import bond_effect
from app.module_b.pricing.swaps import swap_effect
from app.module_b.pricing.options import black_scholes,option_effect


def loan(**kw):return Loan(id='L',name='Loan',sector='Technology',ead_usd=1_000_000,pd_12m=.02,lgd=.45,**kw)
def bond(**kw):return Bond(id='B',name='Bond',sector='Technology',market_value_usd=1_000_000,modified_duration=5,spread_duration=4,convexity=0,**kw)
def swap(side='pay_fixed'):return Swap(id='S',name='IRS',sector='Financials',notional_usd=1e6,dv01_usd_per_bp=150,side=side)
def opt(kind='call',side='long'):return Option(id='O',name='Option',sector='Technology',underlying_price_usd=100,strike_usd=100,time_years=1,implied_volatility=.2,risk_free_rate=.05,dividend_yield=0,kind=kind,side=side,contracts=1,contract_multiplier=100)

def test_loan_example():
    x=loan_effect(loan(),2.5)
    assert x['base_ecl_12m_usd']==pytest.approx(9000)
    assert x['stress_ecl_12m_usd']==pytest.approx(22500)
    assert x['incremental_ecl_12m_usd']==pytest.approx(13500)

def test_pd_cap():
    x=loan_effect(loan(),500)
    assert x['stress_pd_12m']==1.0

def test_bond_100bps():
    x=bond_effect(bond(),100,150)
    assert x['rate_pnl_usd']==pytest.approx(-50_000)
    assert x['spread_pnl_usd']==pytest.approx(-60_000)
    assert x['mtm_pnl_usd']==pytest.approx(-110_000)

def test_bond_zero_shock():
    assert bond_effect(bond(),0,0)['mtm_pnl_usd']==0

def test_swap_opposite_signs():
    assert swap_effect(swap('pay_fixed'),100)['mtm_pnl_usd']==pytest.approx(15000)
    assert swap_effect(swap('receive_fixed'),100)['mtm_pnl_usd']==pytest.approx(-15000)

def test_call_and_put_known_black_scholes():
    assert black_scholes('call',100,100,1,.2,.05)==pytest.approx(10.450583572185565,abs=1e-6)
    assert black_scholes('put',100,100,1,.2,.05)==pytest.approx(5.573526022256971,abs=1e-6)

def test_option_put_call_parity():
    call=black_scholes('call',100,95,2,.22,.03,.01)
    put=black_scholes('put',100,95,2,.22,.03,.01)
    assert call-put==pytest.approx(100*math.exp(-.01*2)-95*math.exp(-.03*2))

def test_option_zero_shock():
    assert option_effect(opt(),0,0,0)['mtm_pnl_usd']==pytest.approx(0)

def test_option_short_flip():
    a=option_effect(opt(side='long'),-10,.05,100)['mtm_pnl_usd']
    b=option_effect(opt(side='short'),-10,.05,100)['mtm_pnl_usd']
    assert a==pytest.approx(-b)

def test_invalid_option_vol_shock():
    with pytest.raises(ValueError,match='Invalid shocked'):
        option_effect(opt(),0,-.5,0)

def test_schema_nonfinite_rejected():
    with pytest.raises(ValueError):Shocks(rate_bps=float('nan'))
    with pytest.raises(ValueError):Shocks(rate_bps=float('inf'))


def test_bond_rejects_economically_impossible_negative_stressed_value():
    from app.module_b.schemas import Bond
    unrealistic=Bond(id='B_LOW',name='Long duration bond',sector='Technology',
                     market_value_usd=1000,modified_duration=50,spread_duration=0,convexity=0)
    with pytest.raises(ValueError,match='negative market value'):
        bond_effect(unrealistic,2000,0)
