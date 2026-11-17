"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg.

* Kippanteil: `scipy.special.betainc` gegen Binomialsumme in Brüchen gegen Polynomintegration in Brüchen gegen Quadratur der Dichte (`scipy.integrate.quad`);
* Verlust (Wert der Information über p): Bruchformel und Polynomintegration gegen Quadratur, beide Richtungen (G oder K gewählt), gegen E[max Δ] − max(E[Δ], 0);
* drei Zustände: eindimensionale Quadratur gegen `dblquad` über das Dreieck und gegen die exakte Fläche bei N = 0 (225/496);
* Fehlerquote über Wiederholungen: Brüche gegen `scipy.stats.binom` gegen Ziehen von k mit numpy (Band, 4,5 Standardfehler);
* Zählungen über Zufallsprobleme (vektorisiert) gegen die Einzelrechnung je Problem (Vorsprung an den Enden der Spannweite ausgewertet, Schwellen aus `stars`);
* Satz „p hat die kleinste relative Kipp-Entfernung“: Beweis in Brüchen an Zufallsproblemen und Gegenprobe ohne die Bedingung;
* Kalibrierung: in einer Welt mit p aus dem Vorab liegt p unter allen Welten mit dem beobachteten k so oft unter der Schwelle, wie der Posterior sagt (numpy-Simulation, Band);
* Einsetzen gegen Mitteln bei G2: Zählung über alle 5141 Datensätze gegen die Betafunktion;
* Regressionsanker (kein Literaturwert): die erste Rechnung an einer Investition mit Schwelle 13/32 (k = 9, n = 20) gab 0,330.
Plattform: scipy-Werte als Band (relativ 1e-9 bis 1e-12), Zufallszählungen als Band, Brüche exakt."""

import random
from fractions import Fraction as Fr
from math import comb

import numpy as np
import pytest
from conftest import beta_cdf_poly, beta_partial_mean_poly
from scipy import integrate, special, stats

import kps_constants as C
import kps_draws as D
import kps_model as M
import kps_posterior as Q

PS = Fr(13, 30)
PSF = float(PS)


# --- Kippanteil -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(0, 0), (1, 0), (1, 1), (4, 1), (10, 5), (10, 0), (10, 10), (20, 10), (40, 20), (60, 17)])
def test_flip_share_four_ways(n, k):
    a, b = Q.posterior(n, k, C.PRIORS[C.PRIOR_FLAT])
    a, b = int(a), int(b)
    by_sum = Q.beta_cdf_fraction(a, b, PS)
    by_poly = beta_cdf_poly(a, b, PS)
    assert by_sum == by_poly                                                   # Bruch gleich Bruch
    assert Q.beta_cdf(a, b, PSF) == pytest.approx(float(by_sum), rel=1e-11, abs=1e-15)
    quad = integrate.quad(lambda x: stats.beta.pdf(x, a, b), 0, PSF, epsabs=1e-13, epsrel=1e-12)[0]
    assert quad == pytest.approx(float(by_sum), rel=1e-8, abs=1e-12)
    assert float(stats.beta.cdf(PSF, a, b)) == pytest.approx(float(by_sum), rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("n", [10, 20, 40, 100, 200, 400])
def test_flip_share_against_scipy_for_the_table(n):
    a, b = n // 2 + 1, n // 2 + 1
    share, chosen = Q.flip_share(a, b, PS)
    assert chosen == "G" and share == pytest.approx(float(stats.beta.cdf(PSF, a, b)), rel=1e-11)


def test_flip_share_is_mirror_symmetric():
    """Hoch und niedrig vertauschen: p → 1 − p, Schwelle 13/30 → 17/30, Daten k → n − k. Der Kippanteil der Gegenseite bleibt gleich."""
    for n, k in ((10, 5), (20, 7), (30, 22), (50, 11)):
        a, b = Q.posterior(n, k, C.PRIORS[C.PRIOR_FLAT])
        a2, b2 = Q.posterior(n, n - k, C.PRIORS[C.PRIOR_FLAT])
        # P(p < p*) für Beta(a, b) ist 1 − P(q < 1 − p*) für das gespiegelte q = 1 − p ~ Beta(b, a)
        assert Q.beta_cdf(a, b, PSF) == pytest.approx(1 - Q.beta_cdf(a2, b2, float(1 - PS)), rel=1e-11, abs=1e-14)


# --- Verlust ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(2, 1), (10, 5), (20, 10), (40, 20), (100, 50), (10, 2), (20, 18), (30, 3)])
def test_info_value_against_quadrature_both_directions(n, k):
    """Wert der vollkommenen Information über p: E[max Δ] − max(E Δ, 0), Δ = 300 (p − 13/30), durch Quadratur der Dichte (G oder K gewählt)."""
    a, b = Q.posterior(n, k, C.PRIORS[C.PRIOR_FLAT])
    a, b = int(a), int(b)
    pdf = lambda x: stats.beta.pdf(x, a, b)
    e_delta = 300 * (a / (a + b) - PSF)
    e_pos = integrate.quad(lambda x: max(300 * (x - PSF), 0.0) * pdf(x), 0, 1, points=[PSF], epsabs=1e-12, epsrel=1e-11, limit=200)[0]
    expect = e_pos - max(e_delta, 0.0)
    assert Q.info_value(a, b, PSF, 300) == pytest.approx(expect, rel=1e-7, abs=1e-9)
    assert Q.info_value(a, b, PSF, 300) >= -1e-12                                      # nie negativ


@pytest.mark.parametrize("n,k", [(10, 5), (20, 10), (4, 2), (60, 30)])
def test_info_value_exact_fractions_agree(n, k):
    a, b = k + 1, n - k + 1
    poly = 300 * (PS * beta_cdf_poly(a, b, PS) - beta_partial_mean_poly(a, b, PS))
    assert Q.info_value(a, b, PS, 300) == pytest.approx(float(poly), rel=1e-10)


def test_information_value_is_the_regret_against_the_oracle_in_draws():
    """Ex-post-Mehrkosten gegen ein Orakel, gemittelt über Posterior-Ziehungen (SplitMix64-Ordnungsstatistik): Band 4,5 Standardfehler."""
    x, _ = D.beta_draws(77, 11, 11, 100000)
    d = 300 * (x - PSF)
    regret = np.maximum(-d, 0.0).mean()                                                 # G gewählt (Mittel 1/2 über der Schwelle)
    se = np.maximum(-d, 0.0).std() / len(x) ** 0.5
    assert abs(regret - Q.info_value(11, 11, PS, 300)) < 4.5 * se


# --- Drei Zustände ------------------------------------------------------------------------------------------------------------------------------

def _dblquad(alpha):
    """P(Δ < 0) und E[(−Δ)+] für Dirichlet(α) über das Dreieck mit dblquad (zweiter Rechenweg)."""
    from math import exp, lgamma, log

    c = lgamma(sum(alpha)) - sum(lgamma(v) for v in alpha)

    def dens(p2, p1):
        p3 = 1 - p1 - p2
        if p1 <= 0 or p2 <= 0 or p3 <= 0:
            return 0.0
        return exp(c + (alpha[0] - 1) * log(p1) + (alpha[1] - 1) * log(p2) + (alpha[2] - 1) * log(p3))

    lim = lambda p1: (0, (150 - 310 * p1) / 160)
    kw = dict(epsabs=1e-10, epsrel=1e-10)
    p = integrate.dblquad(dens, 0, 150 / 310, lambda x: 0, lambda x: lim(x)[1], **kw)[0]
    l = integrate.dblquad(lambda p2, p1: (150 - 310 * p1 - 160 * p2) * dens(p2, p1), 0, 150 / 310, lambda x: 0, lambda x: lim(x)[1], **kw)[0]
    return p, l


@pytest.mark.parametrize("counts", [(0, 0, 0), (3, 4, 3), (9, 12, 9), (30, 40, 30), (2, 1, 0), (0, 5, 5)])
def test_dirichlet_region_one_dimensional_against_dblquad(counts):
    alpha = Q.dirichlet3(counts)
    p, l = Q.dirichlet3_region(alpha)
    p2, l2 = _dblquad(alpha)
    assert p == pytest.approx(p2, rel=1e-6, abs=1e-9) and l == pytest.approx(l2, rel=1e-6, abs=1e-9)


def test_dirichlet_without_data_is_the_exact_area():
    p, l = Q.dirichlet3_region((1, 1, 1))
    assert p == pytest.approx(225 / 496, abs=1e-9)
    assert float(Q.area_share_no_data()) == pytest.approx(p, abs=1e-9)


def test_dirichlet_loss_without_data_in_fractions():
    """Gleichverteilt über das Dreieck: E[(−Δ)+] = (Fläche des Kippgebiets · Mittel von −Δ)/(Fläche des Dreiecks); −Δ ist linear, Mittel über das Dreieck mit den Ecken 150, 0, −... :
    −Δ = 150 − 310 π1 − 160 π2 hat auf dem Kippdreieck (Ecken (0, 0), (15/31, 0), (0, 15/16)) die Eckwerte 150, 0, 0: Mittel 50. Verlust = 50 · (225/496) = 11250/496."""
    assert Q.dirichlet3_region((1, 1, 1))[1] == pytest.approx(float(Fr(50) * Fr(225, 496)), rel=1e-9)


@pytest.mark.parametrize("counts", [(3, 4, 3), (9, 12, 9)])
def test_dirichlet_loss_against_draws_as_band(counts):
    alpha = Q.dirichlet3(counts)
    p1, p2, p3 = D.dirichlet3_draws(41, alpha, 100000)
    loss_draw = np.maximum(-(160 * p1 + 10 * p2 - 150 * p3), 0.0)
    se = loss_draw.std() / len(p1) ** 0.5
    assert abs(loss_draw.mean() - Q.dirichlet3_region(alpha)[1]) < 4.5 * se


def test_dirichlet_flip_when_k_is_chosen():
    """Daten, bei denen das Posterior-Mittel K wählt (viele Beobachtungen im dritten Zustand): Kippanteil = P(Δ > 0), Verlust = E[Δ+]."""
    alpha = Q.dirichlet3((1, 1, 20))
    r = Q.dirichlet3_flip(alpha)
    assert r["chosen"] == "K" and r["share"] == pytest.approx(1 - Q.dirichlet3_region(alpha)[0], rel=1e-12)
    p, l = _dblquad(alpha)
    assert r["loss"] == pytest.approx(l + r["e_delta"], rel=1e-6)                       # E[Δ+] = E[Δ] + E[(−Δ)+]
    assert r["loss"] >= 0


# --- Wiederholungen -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prior_key", [C.PRIOR_FLAT, C.PRIOR_TIGHT, C.PRIOR_JEFFREYS])
def test_repeat_error_fractions_against_scipy(prior_key):
    prior = C.PRIORS[prior_key]
    rnd = random.Random(5)
    for _ in range(25):
        n = rnd.randint(1, 60)
        p_true = Fr(rnd.randint(5, 95), 100)
        g, l, k = rnd.choice([(200, 100, 30), (150, 80, 20), (300, 60, 90)])
        ps = Fr(l + k, g + l)
        exact = Q.repeat_error_fraction(p_true, n, prior, ps, g, l, k)
        fast, loss, better = Q.repeat_error(p_true, n, prior, ps, g, l, k)
        assert fast == pytest.approx(float(exact), rel=1e-9, abs=1e-12)
        assert loss == pytest.approx(float(exact) * abs(float(M.delta(p_true, g, l, k))), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n,p_true", [(20, 0.5), (100, 0.45), (40, 0.4)])
def test_repeat_error_against_drawn_counts_as_band(n, p_true):
    """Anteil der Wiederholungen, in denen die Regel falsch liegt, durch Ziehen von k ~ Bin(n, p) (numpy, 2 Mio. Wiederholungen)."""
    rng = np.random.default_rng(1205)
    ks = rng.binomial(n, p_true, 2_000_000)
    pt = Fr(p_true).limit_denominator(100)
    better_g = M.delta(pt, 200, 100, 30) > 0
    chooses_g = (ks + 1) / (n + 2) > PSF + 1e-15           # Gleichheit gilt als K (Gleichstandsregel der Demo)
    wrong = float((chooses_g != better_g).mean())
    exact = Q.repeat_error(pt, n, C.PRIORS[C.PRIOR_FLAT], PS, 200, 100, 30)[0]
    se = (exact * (1 - exact) / len(ks)) ** 0.5
    assert abs(wrong - exact) < 4.5 * se


def test_repeat_error_is_not_monotone_in_n():
    """Gegenbeispiel zu „mehr Daten, weniger Fehler“: bei wahrem p = 0,45 steigt die Fehlerquote von n = 10 (0,504) bis n = 40 nicht durchgehend."""
    e = {n: Q.repeat_error(Fr(9, 20), n, C.PRIORS[C.PRIOR_FLAT], PS, 200, 100, 30)[0] for n in (10, 20, 40)}
    assert e[20] < e[10] and e[40] > e[20]


# --- Zählungen über Zufallsprobleme ----------------------------------------------------------------------------------------------------------------

def _problems(seed, n_total):
    """Dieselben Zufallsprobleme wie `M.tornado_study`, einzeln dekodiert (Einzelrechnung als Orakel)."""
    u = D.uniform_array(seed, 12 * n_total).reshape(n_total, 12)
    out = []
    for r in u:
        p = 0.2 + 0.6 * r[0]
        g = 100 + 200 * r[1]
        l = 50 + 100 * r[2]
        ew = p * g - (1 - p) * l
        if not ew > 5:
            continue
        k = r[3] * ew
        ha = [0.05 + 0.45 * v for v in r[4:8]]
        hb = [0.05 + 0.45 * v for v in r[8:12]]
        out.append((p, g, l, k, ha, hb))
    return out


def _widths_single(p, g, l, k, h):
    rows = M.tornado_rows(p, g, l, k, dict(zip(M.NAMES, h)))
    return {r["name"]: r["width"] for r in rows}, M.ranking(rows)


def test_tornado_study_against_single_problem_calculation():
    """Zählung der vektorisierten Studie gegen die Einzelrechnung je Problem (Vorsprung an den Enden, Schwellen aus `stars`)."""
    n_total = 2500
    study = M.tornado_study(1301, n_total)
    probs = _problems(1301, n_total)
    assert study["n"] == len(probs)
    conv1, conv2 = [x / 100 for x in C.CONV_ALL_EQUAL], [x / 100 for x in C.CONV_OTHER]
    c = dict(fixed=0, rand=0, order_rand=0, rand_kipp=0, order_rand_kipp=0, eq_kipp=0, p_first=0)
    for p, g, l, k, ha, hb in probs:
        _, o1 = _widths_single(p, g, l, k, conv1)
        _, o2 = _widths_single(p, g, l, k, conv2)
        _, oa = _widths_single(p, g, l, k, ha)
        _, ob = _widths_single(p, g, l, k, hb)
        d = M.delta(p, g, l, k)
        st = M.stars(p, g, l, k)
        rel = {"p": abs(st["p"] - p) / p, "G": abs(st["G"] - g) / g, "L": abs(st["L"] - l) / l, "K": abs(st["K"] - k) / k}
        ok_ = sorted(M.NAMES, key=lambda n: rel[n])
        c["fixed"] += o1[0] != o2[0]
        c["rand"] += oa[0] != ob[0]
        c["order_rand"] += oa != ob
        c["rand_kipp"] += oa[0] != ok_[0]
        c["order_rand_kipp"] += oa != ok_
        c["eq_kipp"] += o1 != ok_
        c["p_first"] += ok_[0] == "p"
    assert study["k_r1_fixed"] == c["fixed"] and study["k_r1_rand"] == c["rand"] and study["k_order_rand"] == c["order_rand"]
    assert study["k_r1_rand_vs_kipp"] == c["rand_kipp"] and study["k_order_rand_vs_kipp"] == c["order_rand_kipp"]
    assert study["k_order_equal_vs_kipp"] == c["eq_kipp"] == 0 and study["k_kipp_rank1_p"] == c["p_first"] == study["n"]


def test_rank_one_of_the_relative_distance_is_p_with_proof_in_fractions():
    """Beweis in drei Zeilen: G gewählt (Δ > 0), G > 0, L ≥ 0, K ≥ 0 ⇒ (G + L) p > G p, (1 − p) L und K, also hat p die größte Ableitung mal Wert und die kleinste relative Kipp-Entfernung."""
    rnd = random.Random(11)
    n_checked = 0
    for _ in range(3000):
        p = Fr(rnd.randint(1, 99), 100)
        g, l, k = (Fr(rnd.randint(1, 300)), Fr(rnd.randint(0, 150)), Fr(rnd.randint(0, 300)))
        d = M.delta(p, g, l, k)
        if d <= 0:
            continue
        n_checked += 1
        w = {"p": (g + l) * p, "G": p * g, "L": (1 - p) * l, "K": k}
        assert w["p"] > max(w["G"], w["L"], w["K"]) or (l == 0 and w["p"] >= w["G"])     # bei L = 0 ist (G + L) p = G p gleich
        rows = M.threshold_rows(p, g, l, k)
        rel = {r["name"]: r["rel"] for r in rows if r["rel"] is not None}
        assert min(rel.values()) == rel["p"]
    assert n_checked > 300


def test_without_the_condition_p_is_not_always_rank_one():
    """Gegenprobe: G, L, K beliebigen Vorzeichens, beide Vorzeichen des Vorsprungs: p ist nicht immer Rang 1 (feste Messung der Demo)."""
    c = M.condition_study(C.COND_SEED, 20000)
    assert c["k_p_first"] < c["n_ok"] and c["k_cond"] == c["n_cond"] > 0
    assert c["k_neg"] / c["n_neg"] < c["k_pos"] / c["n_pos"]


def test_condition_study_against_single_problem_calculation():
    c = M.condition_study(7, 3000)
    u = D.uniform_array(7, 12000).reshape(3000, 4)
    ok = k_p = n_c = k_c = 0
    for r in u:
        p, g, l, k = 0.05 + 0.9 * r[0], -100 + 400 * r[1], -100 + 250 * r[2], -100 + 400 * r[3]
        d = M.delta(p, g, l, k)
        if abs(d) <= 1e-9 or g + l == 0 or g == 0 or l == 0 or k == 0:
            continue
        dist = [abs(d / (g + l)) / p, abs(d) / (p * abs(g)), abs(d) / ((1 - p) * abs(l)), abs(d) / abs(k)]
        first = dist.index(min(dist)) == 0
        ok += 1
        k_p += first
        if d > 0 and g > 0 and l >= 0 and k >= 0:
            n_c += 1
            k_c += first
    assert (c["n_ok"], c["k_p_first"], c["n_cond"], c["k_cond"]) == (ok, k_p, n_c, k_c)


# --- Kalibrierung -----------------------------------------------------------------------------------------------------------------------------

def test_calibration_in_a_world_drawn_from_the_prior():
    """Welt: p gleichverteilt, k ~ Bin(20, p). Unter den Welten mit k = 10 liegt p in dem Anteil unter 13/30, den der Posterior Beta(11, 11) nennt (Band 4,5 Standardfehler)."""
    rng = np.random.default_rng(1204)
    p = rng.random(2_000_000)
    k = rng.binomial(20, p)
    sel = k == 10
    frac = float((p[sel] < PSF).mean())
    ex = Q.beta_cdf(11, 11, PSF)
    assert abs(frac - ex) < 4.5 * (ex * (1 - ex) / sel.sum()) ** 0.5


def test_a_wrong_prior_world_breaks_the_calibration():
    """Gilt in der Welt p ~ Beta(20, 20) und rechnet man mit dem gleichverteilten Vorab, ist die Zahl falsch; mit dem passenden Vorab stimmt sie."""
    rng = np.random.default_rng(1204)
    p = rng.beta(20, 20, 1_500_000)
    k = rng.binomial(20, p)
    sel = k == 10
    frac = float((p[sel] < PSF).mean())
    flat, tight = Q.beta_cdf(11, 11, PSF), Q.beta_cdf(30, 30, PSF)
    assert abs(frac - tight) < 4.5 * (tight * (1 - tight) / sel.sum()) ** 0.5 and frac < flat - 0.05


# --- Einsetzen gegen Mitteln -----------------------------------------------------------------------------------------------------------------------------

def test_plug_in_versus_averaging_count_against_the_beta_function():
    """E[p²] = B(a + 2, b)/B(a, b) über `betaln` gegen die Formel a(a + 1)/((a + b)(a + b + 1)): Zählung der verschieden entscheidenden Datensätze."""
    n_dis = 0
    n_cases = 0
    for n in range(4, 101):
        for k in range(n + 1):
            a, b = k + 1, n - k + 1
            e_p2 = float(np.exp(special.betaln(a + 2, b) - special.betaln(a, b)))
            plug = 320 * (a / (a + b)) ** 2 - 100 - 30
            avg = 320 * e_p2 - 100 - 30
            n_dis += (plug > 0) != (avg > 0)
            n_cases += 1
    assert (n_dis, n_cases) == Q.g2_disagreements(30, C.PRIORS[C.PRIOR_FLAT])[:2] == (19, 5141)


# --- Regressionsanker ---------------------------------------------------------------------------------------------------------------------------

def test_regression_anchor_investment_example():
    """Frühere Messung (Investition +100/−60/+5, Schwelle 13/32, k = 9 von n = 20, gleichverteilt): Kippanteil 0,330 (Regressionsanker, kein Literaturwert)."""
    assert Q.beta_cdf(10, 12, 13 / 32) == pytest.approx(0.3298, abs=6e-5)
    assert M.p_star(Fr(100), Fr(60), Fr(5)) == Fr(60 + 5, 100 + 60) == Fr(13, 32)
