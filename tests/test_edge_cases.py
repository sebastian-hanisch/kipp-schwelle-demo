"""Randfälle, an denen die Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): Gleichstand, k = 0 und k = n, n = 0, eine Alternative dominiert, Basiswert 0,
Spannweite über p = 1, Gleichstand in Gleitkomma, Dirichlet mit Eintrag 0, kleinste und größte Reglerwerte."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import kps_constants as C
import kps_draws as D
import kps_evaluation as ev
import kps_model as M
import kps_posterior as Q

FLAT = C.PRIORS[C.PRIOR_FLAT]


def test_tie_in_floating_point_is_a_tie_within_the_tolerance():
    """Bei p = 0,34, G = 200, L = 100, K = 2 ist der Vorsprung in Brüchen genau 0 und in Gleitkomma ein Rest von etwa 1,4e-14: die Toleranz 1e-9 macht daraus einen Gleichstand."""
    d_float = 0.34 * 200 - (1 - 0.34) * 100 - 2
    assert d_float != 0.0 and abs(d_float) < 1e-9
    assert M.choice(d_float) == "Gleichstand"
    assert M.delta(Fr(34, 100), Fr(200), Fr(100), Fr(2)) == 0
    # knapp außerhalb der Toleranz entscheidet das Vorzeichen
    assert M.choice(2e-9) == "G" and M.choice(-2e-9) == "K"


def test_threshold_exactly_at_the_mean_is_a_tie_for_the_posterior():
    a, b = Q.posterior(28, 12, FLAT)                                  # Posterior-Mittel 13/30 = p*
    assert Q.posterior_mean(a, b) == Fr(13, 30)
    assert Q.flip_share(a, b, Fr(13, 30))[1] == "Gleichstand"
    assert Q.rule_threshold_k(28, FLAT, Fr(13, 30)) == 13             # die Regel wählt bei Gleichheit K
    err, loss, better = Q.repeat_error(Fr(13, 30), 28, FLAT, Fr(13, 30), 200, 100, 30)
    assert (err, loss, better) == (0.0, 0.0, "Gleichstand")           # wahres p genau auf der Schwelle: keine schlechtere Alternative


def test_all_data_in_one_state_k_zero_and_k_n():
    for n, k in ((10, 0), (10, 10), (1, 0), (1, 1)):
        a, b = Q.posterior(n, k, FLAT)
        share, chosen = Q.flip_share(a, b, Fr(13, 30))
        assert 0 < share < 1 and chosen in ("G", "K")
        assert Q.info_value(a, b, Fr(13, 30), 300) >= 0
        assert Q.normal_approx(k, n, Fr(13, 30)) is None
    assert Q.flip_share(*Q.posterior(10, 0, FLAT), Fr(13, 30))[1] == "K"
    assert Q.flip_share(*Q.posterior(10, 10, FLAT), Fr(13, 30))[1] == "G"


def test_no_data_n_zero_is_the_prior():
    a, b = Q.posterior(0, 0, FLAT)
    share, chosen = Q.flip_share(a, b, Fr(13, 30))
    assert chosen == "G" and share == pytest.approx(13 / 30)           # Mittel 1/2 über der Schwelle; gleichverteilt: P(p < p*) = p*
    assert Q.k_from_share(0, 0.5) == 0


def test_one_alternative_dominates_threshold_outside_the_unit_interval():
    """K = 100 > G = 100 + ... : (L + K)/(G + L) über 1, wenn K > G: nie G; Kippanteil der Wahl K ist 0."""
    ps = Fr(60 + 150, 100 + 60)                                       # G = 100, L = 60, K = 150: p* = 21/16 > 1
    assert ps > 1
    a, b = Q.posterior(10, 5, FLAT)
    share, chosen = Q.flip_share(a, b, ps)
    assert chosen == "K" and share == 0.0
    assert Q.info_value(a, b, ps, 160) == pytest.approx(0.0, abs=1e-9)  # nichts zu gewinnen: K ist für jedes p besser
    rows = M.threshold_rows(Fr(1, 2), 100, 60, 150)
    assert not [r for r in rows if r["name"] == "p"][0]["in_range"]
    # untere Seite: p* < 0 (K negativ, hier −200): G für jedes p besser
    ps_neg = Fr(60 - 200, 160)
    share2, chosen2 = Q.flip_share(a, b, ps_neg)
    assert chosen2 == "G" and share2 == 0.0


def test_base_value_zero_has_no_relative_distance():
    rows = M.threshold_rows(Fr(1, 2), 200, 100, 0)                    # K = 0
    k = [r for r in rows if r["name"] == "K"][0]
    assert k["rel"] is None and k["star"] == 50
    assert M.nearest_input(rows) == "p"                               # die Eingabe ohne relative Entfernung wird ausgelassen
    a = ev.analyse_threshold(ev.Settings(kk=0))
    assert "relativ nicht definiert" in ev.describe_threshold([r for r in a.rows if r["name"] == "K"][0])


def test_tornado_range_above_p_equal_one_is_clipped():
    rows = M.tornado_rows(Fr(9, 10), Fr(200), Fr(100), Fr(30), {"p": Fr(1, 2), "G": Fr(0), "L": Fr(0), "K": Fr(0)})
    p = [r for r in rows if r["name"] == "p"][0]
    assert p["hi"] == 1 and p["lo"] == Fr(9, 20)                      # 0,9 · 1,5 = 1,35 wird auf 1 gekappt
    assert p["width"] == 300 * (1 - Fr(9, 20))


def test_tornado_around_a_tie_has_no_flip_and_zero_width_for_zero_range():
    rows = M.tornado_rows(Fr(13, 30), Fr(200), Fr(100), Fr(30), {n: Fr(1, 10) for n in M.NAMES})
    assert not any(r["flips"] or r["tie"] for r in rows)              # Basisvorsprung 0: „kippt“ ist nicht definiert
    zero = M.tornado_rows(Fr(1, 2), Fr(200), Fr(100), Fr(30), {n: Fr(0) for n in M.NAMES})
    assert all(r["width"] == 0 and not r["flips"] for r in zero)


def test_dirichlet_with_a_zero_count_and_shares_over_one_hundred():
    assert Q.dirichlet3((0, 0, 0)) == (1, 1, 1) and Q.dirichlet3((5, 0, 3)) == (6, 1, 4)
    r = Q.dirichlet3_flip(Q.dirichlet3((5, 0, 3)))
    assert 0 <= r["share"] <= 1 and r["chosen"] in ("G", "K")
    a = ev.analyse_three(ev.Settings(pi1=0, pi2=0, n3=10))             # Zustände 1 und 2 ohne Beobachtung: der dritte hat den Rest
    assert a.counts == (0, 0, 10) and a.alpha == (1, 1, 11)
    b = ev.analyse_three(ev.Settings(pi1=80, pi2=60, n3=10))           # Summe über 100: Zustand 2 wird gekappt
    assert b.shares == (80, 20, 0)


def test_three_states_at_the_largest_n_and_with_all_data_in_one_state():
    a = ev.analyse_three(ev.Settings(n3=1000, pi1=100, pi2=0))
    assert a.counts == (1000, 0, 0) and a.exact["chosen"] == "G" and a.exact["share"] < 1e-6


def test_draw_requests_are_at_least_one_and_at_most_the_cap():
    x, _ = D.beta_draws(1, 3, 3, 1)
    assert len(x) == 1
    x, _ = D.beta_draws(1, 1001, 1001, 200000)
    assert 1 <= len(x) <= C.MAX_UNIFORMS // 2001 + 1


def test_slider_extremes_run_in_every_analysis():
    for s in (
        ev.Settings(g=C.G_MIN, l=C.L_MAX, kk=C.K_MAX, p=C.P_MIN, n=C.N_MIN, anteil=C.ANTEIL_MIN, ptrue=C.PTRUE_MIN, n3=C.N3_MIN, draws=C.DRAWS_MIN, nprob=C.NPROB_MIN),
        ev.Settings(g=C.G_MAX, l=C.L_MIN, kk=C.K_MIN, p=C.P_MAX, n=C.N_MAX, anteil=C.ANTEIL_MAX, ptrue=C.PTRUE_MAX, n3=C.N3_MAX, draws=C.DRAWS_MAX, nprob=C.NPROB_MAX,
                    prior=C.PRIOR_TIGHT, payoff=C.PAYOFF_QUAD),
        ev.Settings(prior=C.PRIOR_JEFFREYS, payoff=C.PAYOFF_QUAD, h_p=C.H_MAX, h_g=C.H_MAX, h_l=C.H_MAX, h_k=C.H_MAX),
    ):
        ev.analyse_threshold(s)
        ev.analyse_tornado(s)
        ev.analyse_posterior(s)
        ev.analyse_repeat(s)
        ev.analyse_three(s)


def test_the_selection_fits():
    assert C.N_MIN >= 2 and C.N_MIN % 2 == 0 and C.G_MIN >= C.K_MAX                # G ≥ K in jedem Reglerstand (verpasste Chance nie negativ)
    assert C.L_MIN + C.K_MIN > 0                                                    # Nenner des Kostenverhältnisses
