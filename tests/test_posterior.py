"""Posterior, Kippanteil, Verlust, Wiederholungen, drei Zustände: Handbeispiele in Brüchen (n = 10, k = 5: Posterior Beta(6, 6))."""

from fractions import Fraction as Fr
from math import comb

import pytest
from conftest import beta_cdf_poly, beta_moment_poly, beta_partial_mean_poly

import kps_constants as C
import kps_posterior as Q

PS = Fr(13, 30)
FLAT = C.PRIORS[C.PRIOR_FLAT]


def test_posterior_parameters_and_mean():
    a, b = Q.posterior(10, 5, FLAT)
    assert (a, b) == (6, 6) and Q.posterior_mean(a, b) == Fr(1, 2)
    assert Q.posterior(0, 0, FLAT) == (1, 1)                              # ohne Daten gleich dem Vorab
    assert Q.posterior(20, 10, C.PRIORS[C.PRIOR_TIGHT]) == (30, 30)       # Beta(20, 20) zählt wie 40 Beobachtungen
    assert Q.posterior(10, 5, C.PRIORS[C.PRIOR_JEFFREYS]) == (Fr(11, 2), Fr(11, 2))


def test_flip_share_n10_is_an_exact_fraction_on_two_ways():
    """P(p < 13/30) für Beta(6, 6): Binomialsumme P(Bin(11, 13/30) >= 6) gleich Polynomintegration, Bruch gleich Bruch."""
    by_sum = Q.beta_cdf_fraction(6, 6, PS)
    assert by_sum == beta_cdf_poly(6, 6, PS) == Fr(26634858184181, 82012500000000)
    assert abs(float(by_sum) - 0.324766) < 5e-6
    assert Q.flip_share(6, 6, PS)[0] == pytest.approx(float(by_sum), rel=1e-12)        # scipy-Weg gegen Bruch
    share, chosen = Q.flip_share(6, 6, PS)
    assert chosen == "G"


def test_info_value_n10_by_hand():
    """Verlust = 300 · E[(p* − p)+] = 300 (p* F(6, 6) − (6/12) F(7, 6)): zwei exakte Wege, dann der Gleitkommaweg."""
    formula = 300 * (PS * Q.beta_cdf_fraction(6, 6, PS) - Fr(6, 12) * Q.beta_cdf_fraction(7, 6, PS))
    poly = 300 * (PS * beta_cdf_poly(6, 6, PS) - beta_partial_mean_poly(6, 6, PS))
    assert formula == poly
    assert abs(float(formula) - 8.697273) < 5e-5
    assert Q.info_value(6, 6, PS, 300) == pytest.approx(float(formula), rel=1e-12)
    assert Q.info_value(6, 6, PS, 300) < 20                                # kleiner als der Vorsprung 20 GE


def test_linearity_expected_advantage_equals_advantage_at_the_mean():
    for a, b in ((6, 6), (11, 11), (13, 9), (3, 29)):
        assert 300 * beta_moment_poly(a, b, 1) - 130 == 300 * Fr(a, a + b) - 130 == 300 * (Fr(a, a + b) - PS)


def test_flip_share_without_data_is_the_threshold_itself():
    """Ohne Daten gleichverteilt: P(p < p*) = p* = 13/30."""
    assert Q.beta_cdf_fraction(1, 1, PS) == beta_cdf_poly(1, 1, PS) == PS
    assert Q.flip_share(1, 1, PS)[0] == pytest.approx(13 / 30, rel=1e-12)


def test_flip_share_below_the_threshold_counts_the_other_side():
    """Posterior-Mittel unter der Schwelle (k = 3 von 10 gibt Beta(4, 8), Mittel 1/3): gewählt K, Kippen = P(p > p*)."""
    share, chosen = Q.flip_share(4, 8, PS)
    assert chosen == "K"
    assert share == pytest.approx(1 - float(Q.beta_cdf_fraction(4, 8, PS)), rel=1e-12)


def test_equal_mean_and_threshold_is_a_tie():
    assert Q.flip_share(13, 17, PS)[1] == "Gleichstand"           # Mittel 13/30


def test_repeated_survey_n10_p_half_by_hand():
    """k <= 4 gibt (k + 1)/12 <= 5/12 < 13/30 und K, k = 5 gibt 6/12 > 13/30 und G. Falsch bei k <= 4: (1 + 10 + 45 + 120 + 210)/1024 = 193/512."""
    assert Fr(5, 12) < PS < Fr(6, 12)
    assert Q.rule_threshold_k(10, FLAT, PS) == 5
    assert sum(comb(10, j) for j in range(5)) == 386
    err = Q.repeat_error_fraction(Fr(1, 2), 10, FLAT, PS, 200, 100, 30)
    assert err == Fr(193, 512)
    e, loss, better = Q.repeat_error(Fr(1, 2), 10, FLAT, PS, 200, 100, 30)
    assert better == "G" and e == pytest.approx(193 / 512, rel=1e-12) and loss == pytest.approx(20 * 193 / 512, rel=1e-12)
    assert 20 * err == Fr(965, 128)                               # Verlust je Wiederholung in GE


def test_two_count_units_give_different_numbers():
    """Ziehungen (k = 5 von 10: 0,3248) und Wiederholungen (p = 1/2: 0,3770) sind verschiedene Größen."""
    flip = Q.flip_share(6, 6, PS)[0]
    err = Q.repeat_error(Fr(1, 2), 10, FLAT, PS, 200, 100, 30)[0]
    assert abs(flip - 0.3248) < 5e-5 and abs(err - 0.3770) < 5e-5 and flip != err


def test_rule_threshold_is_exact_at_a_tie():
    """(k + 1)/(n + 2) genau gleich p*: die Regel wählt K (strikt über der Schwelle verlangt). n = 28, k = 12: 13/30 = p*."""
    assert Fr(12 + 1, 28 + 2) == PS
    assert Q.rule_threshold_k(28, FLAT, PS) == 13


def test_known_hand_values_of_the_nonlinear_payoff():
    """G2 zahlt +220 bei p², sonst −100; K = 30. n = 6, k = 4: Posterior Beta(5, 3). Einsetzen 320 (5/8)² − 130 = −5, Mitteln 320 · 30/72 − 130 = 10/3."""
    plug, avg = Q.g2_values(5, 3, 30)
    assert plug == pytest.approx(-5.0, abs=1e-12) and avg == pytest.approx(10 / 3, abs=1e-12)
    assert 320 * Fr(5, 8) ** 2 - 130 == -5 and 320 * Fr(5 * 6, 8 * 9) - 130 == Fr(10, 3)
    n_dis, n_cases, first = Q.g2_disagreements(30, FLAT)
    assert (n_dis, n_cases) == (19, 5141) and first[:2] == (6, 4)


def test_counts_from_shares_sum_to_n():
    assert Q.counts_from_shares(30, (30, 40, 30)) == (9, 12, 9)
    assert Q.counts_from_shares(10, (30, 40, 30)) == (3, 4, 3)
    assert Q.counts_from_shares(0, (30, 40, 30)) == (0, 0, 0)
    for n in range(0, 60):
        assert sum(Q.counts_from_shares(n, (33, 33, 34))) == n and sum(Q.counts_from_shares(n, (1, 1, 1))) == n


def test_three_states_area_without_data_by_hand():
    """G − K je Zustand 160 / 10 / −150, Vorsprung 0,3·160 + 0,4·10 − 0,3·150 = 7. Kippgebiet 310 π1 + 160 π2 < 150, Ecken (0, 0), (15/31, 0), (0, 15/16):
    Fläche ½ · (15/31) · (15/16) = 225/992, Anteil am Dreieck (Fläche ½) 225/496 = 0,4536."""
    assert Q.pay3_diff() == (160, 10, -150)
    assert Fr(3, 10) * 160 + Fr(4, 10) * 10 - Fr(3, 10) * 150 == 7
    assert Fr(1, 2) * Fr(15, 31) * Fr(15, 16) == Fr(225, 992)
    assert Q.area_share_no_data() == Fr(225, 496)
    r = Q.dirichlet3_flip((1, 1, 1))
    assert r["share"] == pytest.approx(225 / 496, abs=1e-9) and r["chosen"] == "G"


def test_dirichlet_expected_advantage():
    """E[Δ] = Σ d_i α_i / Σα: Dirichlet(10, 13, 10): (1600 + 130 − 1500)/33 = 230/33."""
    assert Q.dirichlet3_expected_delta((10, 13, 10)) == pytest.approx(230 / 33, rel=1e-12)
    assert Q.dirichlet3_expected_delta((1, 1, 1)) == pytest.approx((160 + 10 - 150) / 3, rel=1e-12)


def test_normal_approximation_is_undefined_at_the_boundary():
    assert Q.normal_approx(0, 10, PS) is None and Q.normal_approx(10, 10, PS) is None and Q.normal_approx(0, 0, PS) is None
    assert Q.normal_approx(5, 10, PS) == pytest.approx(0.3366, abs=5e-5)       # Φ(−z), z = (0,5 − 13/30)/√(0,25/10)


def test_flip_curve_matches_the_single_calls():
    import numpy as np

    ns = np.arange(2, 200, 2)
    fast = Q.flip_curve_fast(ns, 0.5, FLAT, PS)
    slow = Q.flip_curve(ns, 0.5, FLAT, PS)
    assert np.allclose(fast, slow, rtol=1e-12, atol=1e-15)


def test_largest_even_n_above_five_and_one_percent():
    assert Q.last_n_above(0.05, 0.5, FLAT, PS) == 148        # ab n = 150 unter 5 % (gerade n, k = n/2, gleichverteilt)
    assert Q.last_n_above(0.01, 0.5, FLAT, PS) == 300
