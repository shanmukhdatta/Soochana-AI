"""Independent mathematical and structural verification script for RiskPulse Module B v1.1.
Verifies all items specified in Step 6 and Step 7 of the integration prompt.
"""
import math
from app.module_b.portfolio import load_portfolio, portfolio_summary, scaled_portfolio
from app.module_b.scenarios import load_library, template, merge_shocks, effective_shocks, scale_scenario
from app.module_b.pricing.loans import loan_effect
from app.module_b.pricing.bonds import bond_effect
from app.module_b.pricing.swaps import swap_effect
from app.module_b.pricing.options import black_scholes, option_effect
from app.module_b.stress_engine import run_stress
from app.module_b.what_if import compare
from app.module_b.reverse_stress import reverse_stress
from app.module_b.schemas import Loan, Bond, Swap, Option, Shocks, Scenario, ScaleChange
import pytest

def main():
    print("=== STARTING INDEPENDENT AUDIT OF MODULE B v1.1 ===")
    
    # 1. Holdings and exposure counts
    p = load_portfolio()
    summary = portfolio_summary(p)
    print(f"Total positions: {summary['position_count']}")
    print(f"Breakdown: {summary['counts']}")
    assert summary['position_count'] == 30
    assert summary['counts'] == {'loan': 12, 'bond': 10, 'swap': 4, 'option': 4}
    assert abs(summary['loan_ead_usd'] - 5_000_000.0) < 1e-6
    assert abs(summary['bond_market_value_usd'] - 3_500_000.0) < 1e-6
    assert abs(summary['derivative_reference_notional_usd'] - 1_500_000.0) < 1e-6
    print("[PASS] 30 synthetic holdings (12 loans, 10 bonds, 4 swaps, 4 options) and separate exposure totals verified.")

    # 2. 15 scenario templates
    lib = load_library()
    assert len(lib) == 5
    assert all(set(tiers.keys()) == {'mild', 'moderate', 'severe'} for tiers in lib.values())
    print("[PASS] 15 scenario templates (5 event families x 3 tiers) verified.")

    # 3. Zero-shock effects zero
    neutral_scenario = Scenario(name='Neutral', shocks=Shocks())
    zero_res = run_stress(p, neutral_scenario)
    assert abs(zero_res['totals']['instantaneous_mtm_pnl_usd']) < 1e-9
    assert abs(zero_res['totals']['incremental_12m_expected_credit_loss_usd']) < 1e-9
    assert abs(zero_res['totals']['illustrative_mixed_horizon_net_effect_usd']) < 1e-9
    print("[PASS] Zero-shock produces exact zero MTM and zero incremental 12m ECL.")

    # 4. Loan pricing arithmetic: bp / PD*LGD*EAD
    test_loan = Loan(id='L_TEST', name='Test Loan', sector='Tech', ticker='AAPL', ead_usd=1_000_000, pd_12m=0.02, lgd=0.45)
    le = loan_effect(test_loan, 2.5)
    # base ECL = 1M * 0.02 * 0.45 = 9,000
    # stress ECL = 1M * (0.02 * 2.5) * 0.45 = 22,500
    # incremental = 13,500
    assert abs(le['base_ecl_12m_usd'] - 9000.0) < 1e-6
    assert abs(le['stress_ecl_12m_usd'] - 22500.0) < 1e-6
    assert abs(le['incremental_ecl_12m_usd'] - 13500.0) < 1e-6
    # PD cap check
    le_cap = loan_effect(test_loan, 100.0)
    assert le_cap['stress_pd_12m'] == 1.0
    print("[PASS] Loan PD*LGD*EAD, 12m ECL difference, and PD cap at 1.0 verified.")

    # 5. Bond DV01 / CS01 signs and convexity
    test_bond = Bond(id='B_TEST', name='Test Bond', sector='Tech', ticker='AAPL', market_value_usd=1_000_000, modified_duration=5.0, spread_duration=4.0, convexity=20.0)
    # Rate up 100 bps (+1%), Spread up 150 bps (+1.5%)
    be = bond_effect(test_bond, 100.0, 150.0)
    # dr = 0.01: rate_delta = 1M * (-5*0.01 + 0.5*20*(0.01^2)) = 1M * (-0.05 + 0.001) = -49,000
    # ds = 0.015: spread_delta = -1M * 4 * 0.015 = -60,000
    # total mtm = -109,000
    assert abs(be['rate_pnl_usd'] - (-49000.0)) < 1e-6
    assert abs(be['spread_pnl_usd'] - (-60000.0)) < 1e-6
    assert abs(be['mtm_pnl_usd'] - (-109000.0)) < 1e-6
    assert abs(be['dv01_usd_per_bp'] - 500.0) < 1e-6
    assert abs(be['cs01_usd_per_bp'] - 400.0) < 1e-6
    print("[PASS] Bond DV01/CS01 signs, rate convexity, and basis point scaling verified.")

    # Extreme bond shock produces negative market value -> must raise ValueError
    try:
        bond_effect(Bond(id='B_EXT', name='Bond', sector='Tech', market_value_usd=1000, modified_duration=40, spread_duration=0, convexity=0), 3000.0, 0.0)
        assert False, "Should have raised ValueError on negative bond market value"
    except ValueError as e:
        assert "negative market value" in str(e)
    print("[PASS] Bond sensitivity approximation guard against negative market value verified.")

    # 6. Swap pay-fixed vs receive-fixed signs
    pay_swap = Swap(id='S_PAY', name='Pay IRS', sector='Financials', notional_usd=1_000_000, dv01_usd_per_bp=150.0, side='pay_fixed')
    rec_swap = Swap(id='S_REC', name='Rec IRS', sector='Financials', notional_usd=1_000_000, dv01_usd_per_bp=150.0, side='receive_fixed')
    # Rates rise by 50 bps: pay-fixed gains, receive-fixed loses
    assert abs(swap_effect(pay_swap, 50.0)['mtm_pnl_usd'] - 7500.0) < 1e-6
    assert abs(swap_effect(rec_swap, 50.0)['mtm_pnl_usd'] - (-7500.0)) < 1e-6
    print("[PASS] Swap pay-fixed (+gain when rates rise) vs receive-fixed (-loss when rates rise) signs verified.")

    # 7. Black-Scholes benchmark, put-call parity, and long-short symmetry
    bs_call = black_scholes('call', 100, 100, 1.0, 0.2, 0.05, 0.0)
    bs_put = black_scholes('put', 100, 100, 1.0, 0.2, 0.05, 0.0)
    assert abs(bs_call - 10.450583572185565) < 1e-6
    assert abs(bs_put - 5.573526022256971) < 1e-6
    # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    c_parity = black_scholes('call', 105, 95, 1.5, 0.25, 0.04, 0.02)
    p_parity = black_scholes('put', 105, 95, 1.5, 0.25, 0.04, 0.02)
    parity_diff = (c_parity - p_parity) - (105 * math.exp(-0.02 * 1.5) - 95 * math.exp(-0.04 * 1.5))
    assert abs(parity_diff) < 1e-6
    # Long-short symmetry
    opt_long = Option(id='O_L', name='Opt L', sector='Tech', ticker='AAPL', underlying_price_usd=100, strike_usd=100, time_years=1, implied_volatility=0.2, risk_free_rate=0.05, dividend_yield=0, kind='call', side='long', contracts=10, contract_multiplier=100)
    opt_short = Option(id='O_S', name='Opt S', sector='Tech', ticker='AAPL', underlying_price_usd=100, strike_usd=100, time_years=1, implied_volatility=0.2, risk_free_rate=0.05, dividend_yield=0, kind='call', side='short', contracts=10, contract_multiplier=100)
    res_l = option_effect(opt_long, -5.0, 0.02, 25.0)
    res_s = option_effect(opt_short, -5.0, 0.02, 25.0)
    assert abs(res_l['mtm_pnl_usd'] + res_s['mtm_pnl_usd']) < 1e-6
    print("[PASS] Black-Scholes benchmark (Call 10.4506, Put 5.5735), put-call parity, and long/short symmetry verified.")

    # 8. All position-type-sector-ticker contributions reconcile with separate totals
    for event_family in lib:
        for tier in ['mild', 'moderate', 'severe']:
            scen = template(event_family, tier, issuer='AAPL')
            st_res = run_stress(p, scen)
            pos_loss = sum(x['indicative_loss_contribution_usd'] for x in st_res['positions'])
            type_loss = sum(st_res['loss_contribution_by_type_usd'].values())
            sector_loss = sum(st_res['loss_contribution_by_sector_usd'].values())
            ticker_loss = sum(st_res['loss_contribution_by_ticker_usd'].values())
            # Note: ticker_loss only covers positions with tickers (swaps do not have tickers)
            swaps_contrib = st_res['loss_contribution_by_type_usd']['swap']
            total_loss = st_res['totals']['illustrative_mixed_horizon_loss_usd']
            assert abs(pos_loss - type_loss) < 1e-6
            assert abs(pos_loss - sector_loss) < 1e-6
            assert abs(pos_loss - (ticker_loss + swaps_contrib)) < 1e-6
            assert abs(pos_loss - total_loss) < 1e-6
            assert math.isfinite(pos_loss)
    print("[PASS] Reconciliation across all 15 scenarios (positions, types, sectors, tickers) verified.")

    # 9. __ISSUER__ placeholder unresolved rejection
    raw_credit = lib['Credit Event']['moderate']
    try:
        run_stress(p, raw_credit)
        assert False, "Should have rejected unresolved __ISSUER__"
    except ValueError as e:
        assert "__ISSUER__" in str(e)
    print("[PASS] Rejection of unresolved __ISSUER__ template placeholder verified.")

    # 10. Case-insensitive issuer normalization and duplicate key rejection
    scen_case = Scenario(name='Case test', shocks=Shocks(), issuer_shocks={'aapl': Shocks(pd_multiplier=2.0)})
    assert 'AAPL' in scen_case.issuer_shocks
    try:
        Scenario(name='Dup test', shocks=Shocks(), issuer_shocks={'aapl': Shocks(), 'AAPL': Shocks()})
        assert False, "Should have rejected duplicate issuer keys"
    except Exception as e:
        assert "Duplicate override key" in str(e)
    print("[PASS] Case-insensitive issuer ticker normalization and duplicate key rejection verified.")

    # 11. override_coverage warning for unmatched issuer / sector shocks
    scen_unmatched = Scenario(name='Unmatched test', shocks=Shocks(), issuer_shocks={'UNKNOWN': Shocks(pd_multiplier=2)}, sector_shocks={'NonExistentSector': Shocks(spread_bps=50)})
    unm_res = run_stress(p, scen_unmatched)
    assert unm_res['override_coverage']['unmatched_issuer_shocks'] == ['UNKNOWN']
    assert unm_res['override_coverage']['unmatched_sector_shocks'] == ['NonExistentSector']
    assert unm_res['override_coverage']['all_overrides_matched'] is False
    assert any('matched no portfolio holdings' in w for w in unm_res['warnings'])
    print("[PASS] override_coverage reporting unmatched issuer/sector shocks verified.")

    # 12. What-If preserves original portfolio without mutation
    orig_json = p.model_dump_json()
    wi_res = compare(p, template('Credit Event', 'mild', issuer='AAPL'), [ScaleChange(position_id='L01', scale=0.5)])
    assert p.model_dump_json() == orig_json
    assert wi_res['after']['portfolio']['loan_ead_usd'] < wi_res['before']['portfolio']['loan_ead_usd']
    print("[PASS] What-if portfolio immutability verified.")

    # 13. Reverse stress grid search, non-monotonicity handling, truncation reporting
    rev_res = reverse_stress(p, template('Credit Event', 'moderate', issuer='AAPL'), loss_budget_usd=100_000, max_multiplier=2.0, steps=41)
    assert rev_res['found_breach'] is True
    assert rev_res['first_breach_on_grid']['multiplier'] == pytest.approx(0.7)
    assert 'override_coverage' in rev_res
    
    # Truncation test with extreme equity shock
    trunc_scen = Scenario(name='Crash', shocks=Shocks(equity_pct=-50.0))
    trunc_res = reverse_stress(p, trunc_scen, loss_budget_usd=1.0, max_multiplier=4.0, steps=81)
    assert trunc_res['truncated_at_multiplier'] is not None
    assert trunc_res['truncation_reason'] is not None
    print("[PASS] Reverse stress grid search, first breach detection, and physical truncation reporting verified.")

    print("=== ALL INDEPENDENT AUDIT CHECKS PASSED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
