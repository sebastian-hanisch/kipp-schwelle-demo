"""Zufallsstrom und Ziehungen: dieselbe Folge in Klasse und Feld, bekannte SplitMix64-Werte, Ordnungsstatistik gegen die exakte Verteilung (Band), Obergrenze der Ziehungen."""

from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import kps_constants as C
import kps_draws as D
import kps_posterior as Q
import kps_rng as R


def test_known_splitmix_values_for_seed_zero():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x1d5ccf3a0d3e2e3c."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF
    assert g.next_u64() == 0x6E789E6AA1B965F4


def test_array_stream_equals_class_stream():
    g = R.SplitMix64(12345)
    assert [g.next_u64() for _ in range(50)] == [int(v) for v in D.splitmix_array(12345, 50)]
    g = R.SplitMix64(7)
    assert [g.random() for _ in range(50)] == list(D.uniform_array(7, 50))


def test_offset_continues_the_stream():
    full = D.uniform_array(99, 40)
    assert np.array_equal(D.uniform_array(99, 15, offset=25), full[25:])


def test_uniforms_are_in_unit_interval_with_correct_moments():
    u = D.uniform_array(1301, 200000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 4.5 * (1 / 12 / len(u)) ** 0.5
    assert abs(u.var() - 1 / 12) < 2e-3


def test_budget_of_uniforms():
    assert D.affordable_draws(100000, 21) == 100000                      # 2,1 Millionen passen
    assert D.affordable_draws(200000, 1001) == C.MAX_UNIFORMS // 1001    # n = 1000: nur etwa 3996 Ziehungen
    assert D.affordable_draws(10, 10**9) == 1                            # mindestens eine


def test_beta_draws_order_statistic_against_the_exact_distribution_as_band():
    """Beta(11, 11) als 11-kleinste von 21 Gleichverteilten: Anteil unter 13/30 gegen die exakte Verteilung, innerhalb 4,5 Standardfehlern (100 000 Ziehungen)."""
    x, method = D.beta_draws(1301, 11, 11, 100000)
    assert method == "Ordnungsstatistik" and len(x) == 100000
    ex = Q.beta_cdf(11, 11, 13 / 30)
    se = (ex * (1 - ex) / len(x)) ** 0.5
    assert abs(float((x < 13 / 30).mean()) - ex) < 4.5 * se
    # Momente der Beta(11, 11): Mittel 1/2, Varianz 1/(4 · 23)
    assert abs(x.mean() - 0.5) < 4.5 * (1 / 92 / len(x)) ** 0.5
    assert abs(x.var() - 1 / 92) < 4e-4


def test_beta_draws_skewed_parameters_and_edge_case_a_b_equal_one():
    x, _ = D.beta_draws(5, 4, 17, 60000)                                 # Mittel 4/21
    assert abs(x.mean() - 4 / 21) < 4.5 * (stats.beta.var(4, 17) / len(x)) ** 0.5
    x1, _ = D.beta_draws(5, 1, 1, 60000)                                  # Gleichverteilung: eine Gleichverteilte
    assert abs(x1.mean() - 0.5) < 4.5 * (1 / 12 / len(x1)) ** 0.5


def test_beta_draws_with_half_integers_use_the_inverse_and_match_the_distribution():
    x, method = D.beta_draws(1301, 10.5, 10.5, 100000)
    assert method == "Umkehrfunktion"
    ex = Q.beta_cdf(10.5, 10.5, 13 / 30)
    se = (ex * (1 - ex) / len(x)) ** 0.5
    assert abs(float((x < 13 / 30).mean()) - ex) < 4.5 * se


def test_the_draw_count_is_capped_for_large_n():
    x, _ = D.beta_draws(3, 501, 501, 200000)                             # 1001 Gleichverteilte je Ziehung
    assert len(x) == C.MAX_UNIFORMS // 1001 and 0.35 < x.mean() < 0.65


def test_dirichlet_draws_against_the_exact_quadrature_as_band():
    """Dirichlet(10, 13, 10) (N = 30, Anteile 0,3 / 0,4 / 0,3): Anteil mit Vorsprung < 0 gegen die 1-dimensionale Quadratur."""
    alpha = Q.dirichlet3((9, 12, 9))
    p1, p2, p3 = D.dirichlet3_draws(1301, alpha, 100000)
    assert np.allclose(p1 + p2 + p3, 1.0, atol=1e-12) and (p1 >= 0).all() and (p2 >= 0).all() and (p3 >= -1e-12).all()
    ex = Q.dirichlet3_region(alpha)[0]
    share = float((160 * p1 + 10 * p2 - 150 * p3 < 0).mean())
    assert abs(share - ex) < 4.5 * (ex * (1 - ex) / len(p1)) ** 0.5
    s = sum(alpha)
    assert abs(p1.mean() - alpha[0] / s) < 4.5 * (stats.beta.var(alpha[0], s - alpha[0]) / len(p1)) ** 0.5


def test_dirichlet_without_data_is_uniform_on_the_triangle():
    p1, p2, p3 = D.dirichlet3_draws(8, (1, 1, 1), 60000)
    assert abs(p1.mean() - 1 / 3) < 4.5 * (stats.beta.var(1, 2) / len(p1)) ** 0.5
    share = float((310 * p1 + 160 * p2 < 150).mean())
    assert abs(share - 225 / 496) < 4.5 * (0.45 * 0.55 / len(p1)) ** 0.5


def test_same_seed_same_draws_other_seed_other_draws():
    a, _ = D.beta_draws(5, 11, 11, 3000)
    b, _ = D.beta_draws(5, 11, 11, 3000)
    c, _ = D.beta_draws(6, 11, 11, 3000)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
