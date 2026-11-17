"""Claims-Tests: jede Zahl der Befunde B1 bis B17 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: Kapazitätsfall (G +200 / −100 GE, K +30 GE, p = 1/2, Schwelle 13/30), gleichverteiltes Vorab, k = n/2, sofern nicht anders genannt.
Plattform: Brüche und kleine Zählungen exakt; scipy-Werte als Anzeigestellen (relative Toleranz 1e-9 gegen die Bruchrechnung); Zählungen über Zufallsprobleme als Band (± 2 Probleme);
Ziehungen als Band (4,5 Standardfehler). Jede README-Zahl wird hier gerechnet **und** im README-Text gesucht (`_readme_has`)."""

import re
from fractions import Fraction as Fr
from pathlib import Path

import pytest

import kps_constants as C
import kps_draws as D
import kps_evaluation as ev
import kps_model as M
import kps_posterior as Q
from kps_evaluation import de, ge


def pct(x, nd=2):
    """Anteil als Prozent mit Komma, hier standardmäßig mit zwei Stellen (die Tabellen der README); eine Stelle: pct1."""
    return ev.pct(x, nd)


def pct1(x):
    return ev.pct(x, 1)

ROOT = Path(__file__).resolve().parent.parent
README = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8").replace("\\|", "|").replace("\\*", "*"))   # Zeilenumbrüche der README zählen nicht
PS = Fr(13, 30)
FLAT = C.PRIORS[C.PRIOR_FLAT]


def _readme_has(*items):
    for n in items:
        assert re.sub(r"\s+", " ", n) in README, n


def _flip(n, k=None, prior=FLAT, ps=PS):
    k = n // 2 if k is None else k
    return Q.flip_share(*Q.posterior(n, k, prior), ps)[0]


# --- B1 -------------------------------------------------------------------------------------------------------------------------------------

def test_b1_thresholds_and_relative_distances():
    p, g, l, k = Fr(1, 2), Fr(200), Fr(100), Fr(30)
    assert M.p_star(g, l, k) == Fr(13, 30) and de(13 / 30, 3) == "0,433"
    assert M.stars(p, g, l, k) == {"p": Fr(13, 30), "G": 160, "L": 140, "K": 50}
    rel = {r["name"]: r["rel"] for r in M.threshold_rows(p, g, l, k)}
    assert rel == {"p": Fr(2, 15), "G": Fr(1, 5), "L": Fr(2, 5), "K": Fr(2, 3)}
    assert [de(100 * float(rel[n]), 1) for n in ("p", "G", "L", "K")] == ["13,3", "20,0", "40,0", "66,7"]
    assert M.cost_ratio(g, l, k) == Fr(17, 13)
    assert (Fr(1, 2) * 200 - Fr(1, 2) * 100, 30, M.delta(p, g, l, k)) == (50, 30, 20)
    _readme_has("p* = 13/30 = 0,433", "G* = 160", "L* = 140", "K* = 50", "13,3 %", "20 %", "40 %", "66,7 %", "r = 17/13", "G 50 GE, K 30 GE, Vorsprung von G 20 GE")


# --- B2, B3 ---------------------------------------------------------------------------------------------------------------------------------

def test_b2_tornado_widths_flags_and_rank_one():
    p, g, l, k = Fr(1, 2), Fr(200), Fr(100), Fr(30)
    k1 = {r["name"]: r for r in M.tornado_rows(p, g, l, k, {n: Fr(1, 5) for n in M.NAMES})}
    assert {n: r["width"] for n, r in k1.items()} == {"p": 60, "G": 40, "L": 20, "K": 12}
    assert k1["p"]["flips"] and k1["G"]["tie"] and not k1["G"]["flips"] and not k1["L"]["flips"] and not k1["K"]["flips"]
    assert M.ranking(list(k1.values()))[0] == "p"
    k2 = {r["name"]: r for r in M.tornado_rows(p, g, l, k, {"p": Fr(1, 5), "G": Fr(1, 2), "L": Fr(3, 10), "K": Fr(1, 20)})}
    assert {n: r["width"] for n, r in k2.items()} == {"p": 60, "G": 100, "L": 30, "K": 3}
    assert k2["p"]["flips"] and k2["G"]["flips"] and not k2["L"]["flips"] and not k2["K"]["flips"]
    assert M.ranking(list(k2.values()))[0] == "G"
    _readme_has("Breiten p 60, G 40, L 20, K 12 GE", "Breiten p 60, G 100, L 30, K 3 GE", "Rang 1 ist p", "Rang 1 ist G")


def test_b3_identity_and_same_order_for_every_equal_convention():
    p, g, l, k = Fr(1, 2), Fr(200), Fr(100), Fr(30)
    assert set(M.width_times_distance(p, g, l, k, Fr(1, 5)).values()) == {8}
    for x in (Fr(1, 20), Fr(1, 10), Fr(1, 5), Fr(1, 2)):
        assert M.ranking(M.tornado_rows(p, g, l, k, {n: x for n in M.NAMES})) == ["p", "G", "L", "K"]
        assert set(M.width_times_distance(p, g, l, k, x).values()) == {2 * x * 20}
    _readme_has("2x·Vorsprung (8 GE bei x = 20 %)", "x = 5, 10, 20, 50 %", "p > G > L > K")


# --- B4, B5: Zufallsproblem-Zählungen (Band ± 2 Probleme) ---------------------------------------------------------------------------------------

STUDY = {
    1301: dict(n=147366, k_r1_fixed=147366, k_r1_rand=68545, k_order_rand=110242, k_r1_rand_vs_kipp=49093, k_order_rand_vs_kipp=88635),
    1302: dict(n=147367, k_r1_fixed=147367, k_r1_rand=67983, k_order_rand=110273, k_r1_rand_vs_kipp=48996, k_order_rand_vs_kipp=89000),
}


@pytest.mark.parametrize("seed", [1301, 1302])
def test_b4_b5_random_problem_counts_as_band(seed):
    s = ev.tornado_study(seed, 200000)
    exp = STUDY[seed]
    assert s["n_total"] == 200000
    for key, value in exp.items():
        assert abs(s[key] - value) <= 2, (seed, key, s[key], value)
    assert s["k_r1_equal_vs_kipp"] == 0 and s["k_order_equal_vs_kipp"] == 0 and s["k_kipp_rank1_p"] == s["n"]


def test_b4_b5_readme_percentages_follow_from_the_counts():
    a, b = STUDY[1301], STUDY[1302]
    pc = lambda e, key: pct1(e[key] / e["n"])
    assert (pc(a, "k_r1_rand"), pc(b, "k_r1_rand")) == ("46,5 %", "46,1 %")
    assert (pct1(1 - a["k_r1_rand"] / a["n"]), pct1(1 - b["k_r1_rand"] / b["n"])) == ("53,5 %", "53,9 %")
    assert (pc(a, "k_order_rand"), pc(b, "k_order_rand")) == ("74,8 %", "74,8 %")
    assert (pc(a, "k_r1_rand_vs_kipp"), pc(b, "k_r1_rand_vs_kipp")) == ("33,3 %", "33,2 %")
    assert (pc(a, "k_order_rand_vs_kipp"), pc(b, "k_order_rand_vs_kipp")) == ("60,1 %", "60,4 %")
    assert pc(a, "k_r1_fixed") == pc(b, "k_r1_fixed") == "100,0 %"
    _readme_has("46,5 % und 46,1 %", "53,5 % und 53,9 %", "74,8 % und 74,8 %", "33,3 % und 33,2 %", "60,1 % und 60,4 %", "147 366 bei Zufalls-Seed 1301 und 147 367 bei 1302",
                "0 von 147 366 und 0 von 147 367", "147 366 von 147 366 und 147 367 von 147 367", "in allen Zufallsproblemen (100 %) statt in 99,0 %")


def test_b6_condition_counts():
    c = ev.condition_study()
    assert (c["n_total"], c["n_ok"]) == (400000, 400000)
    assert abs(c["n_cond"] - 23640) <= 2 and c["k_cond"] == c["n_cond"]                  # Beweis: immer p; Band für die Zahl der Probleme dieser Bedingung
    assert pct1(c["k_p_first"] / c["n_ok"]) == "22,2 %" and pct1(c["k_pos"] / c["n_pos"]) == "41,2 %" and pct1(c["k_neg"] / c["n_neg"]) == "12,2 %"
    assert abs(c["k_pos"] - 56726) <= 2 and abs(c["n_pos"] - 137785) <= 2 and abs(c["k_neg"] - 31903) <= 2 and abs(c["n_neg"] - 262215) <= 2
    _readme_has("23 640 von 23 640", "nur in 22,2 %", "41,2 % (56 726 von 137 785)", "12,2 % (31 903 von 262 215)", "Zufalls-Seed 1310", "400 000 Zufallsprobleme")
    assert C.COND_SEED == 1310 and C.COND_N == 400000


# --- B7, B8, B10, B11: Kippanteil, Verlust, Näherung --------------------------------------------------------------------------------------------

def test_b7_flip_share_table_and_exact_fraction():
    expect = {10: "32,48 %", 20: "26,73 %", 40: "19,41 %", 100: "8,86 %", 200: "2,86 %", 400: "0,36 %", 1000: "0,001 %"}
    for n, text in expect.items():
        assert pct(_flip(n), 3 if n == 1000 else 2) == text, n
    _readme_has(*expect.values())
    f = Q.beta_cdf_fraction(6, 6, PS)
    assert f == Fr(26634858184181, 82012500000000) and f == pytest.approx(0.3247658367, rel=1e-9)
    _readme_has("26 634 858 184 181 / 82 012 500 000 000")


def test_b8_largest_even_n_above_five_and_one_percent():
    assert Q.last_n_above(0.05, 0.5, FLAT, PS) == 148 and Q.last_n_above(0.01, 0.5, FLAT, PS) == 300
    assert _flip(150) < 0.05 and _flip(148) > 0.05 and _flip(302) < 0.01 and _flip(300) > 0.01
    _readme_has("über 5 %: 148 (unter 5 % ab n = 150)", "über 1 %: 300 (unter 1 % ab n = 302)", "n = 2, 4, …, 20 000")


def test_b9_prior_dependence():
    pri = {key: [_flip(n, prior=pr) for n in (20, 100)] for key, pr in C.PRIORS.items()}
    assert [pct(v) for v in pri[C.PRIOR_FLAT]] == ["26,73 %", "8,86 %"]
    assert [pct(v) for v in pri[C.PRIOR_JEFFREYS]] == ["27,21 %", "8,97 %"]
    assert [pct(v) for v in pri[C.PRIOR_TIGHT]] == ["15,08 %", "5,68 %"]
    assert Q.posterior(20, 10, C.PRIORS[C.PRIOR_TIGHT]) == (30, 30)
    _readme_has("gleichverteilt 26,73 % / 8,86 %", "Jeffreys 27,21 % / 8,97 %", "eng Beta(20, 20) 15,08 % / 5,68 %", "15,1 %", "26,7 %", "26,73 %, 27,21 % oder 15,08 %")
    assert pct1(pri[C.PRIOR_TIGHT][0]) == "15,1 %" and pct1(pri[C.PRIOR_FLAT][0]) == "26,7 %"


def test_b10_information_value_and_the_state_evpi():
    vals = {n: Q.info_value(*Q.posterior(n, n // 2, FLAT), PS, 300) for n in (10, 20, 40, 100, 200, 400)}
    assert [de(vals[n], 2) for n in (10, 20, 40, 100, 200)] == ["8,70", "5,01", "2,42", "0,60", "0,11"] and de(vals[400], 3) == "0,008"
    _readme_has("n = 10: 8,70 GE, 20: 5,01 GE, 40: 2,42 GE, 100: 0,60 GE, 200: 0,11 GE, 400: 0,008 GE", "8,70 GE bei n = 10, 0,60 GE bei n = 100", "der Vorsprung 20 GE")
    assert ev.evpi_state(Fr(1, 2), 200, 100, 30) == 65.0
    _readme_has("Zustand** (hohe oder niedrige Nachfrage): der ist bei p = 1/2 gleich 65 GE")


def test_b11_normal_approximation():
    approx = {n: Q.normal_approx(n // 2, n, PS) for n in (10, 100, 200, 400, 1000)}
    assert pct(approx[10]) == "33,66 %" and pct(_flip(10)) == "32,48 %"
    dev = max(abs(approx[n] - _flip(n)) for n in (100, 200, 400, 1000))
    assert de(100 * dev, 2) == "0,26"
    assert [de(100 * abs(approx[n] - _flip(n)), 2) for n in (200, 400, 1000)] == ["0,10", "0,02", "0,00"]
    _readme_has("33,66 % bei n = 10 gegen exakt 32,48 %", "Abweichung 0,26 Prozentpunkte bei n = 100, danach kleiner (0,10 / 0,02 / unter 0,01 bei n = 200 / 400 / 1 000)")


# --- B12: Linearität, G2 ----------------------------------------------------------------------------------------------------------------------------

def test_b12_linearity_and_the_nonlinear_counterexample():
    from conftest import beta_moment_poly

    for a, b in ((6, 6), (11, 11), (13, 9)):
        assert 300 * beta_moment_poly(a, b, 1) - 130 == 300 * (Fr(a, a + b) - PS)
    n_dis, n_cases, first = Q.g2_disagreements(30, FLAT)
    assert (n_dis, n_cases) == (19, 5141) and first[:2] == (6, 4)
    assert first[2] == pytest.approx(-5.0, abs=1e-9) and first[3] == pytest.approx(10 / 3, abs=1e-9)
    _readme_has("19 von 5 141 Datensätzen (n = 4 bis 100, alle k)", "n = 6, k = 4: Einsetzen −5 GE, Mitteln 10/3 GE", "+220 GE bei zwei hohen Perioden mit Wahrscheinlichkeit p², sonst −100 GE, K = 30 GE")
    assert (C.G2_WIN, C.G2_LOSE) == (220, -100)


# --- B13, B14: Wiederholungen --------------------------------------------------------------------------------------------------------------------------

def _err(pt, n):
    return Q.repeat_error(pt, n, FLAT, PS, 200, 100, 30)


def test_b13_repeat_error_table_and_hand_value():
    half = {n: _err(Fr(1, 2), n) for n in (10, 20, 40, 100, 200, 400)}
    assert [pct(half[n][0]) for n in half] == ["37,70 %", "25,17 %", "21,48 %", "9,67 %", "2,80 %", "0,40 %"]
    assert ge(half[10][1]) == "7,54 GE" and Q.repeat_error_fraction(Fr(1, 2), 10, FLAT, PS, 200, 100, 30) == Fr(193, 512)
    p45 = [_err(Fr(9, 20), n) for n in (10, 20, 40, 100, 200, 400, 1000)]
    assert [pct(e[0]) for e in p45] == ["50,44 %", "41,43 %", "43,91 %", "38,28 %", "31,01 %", "25,71 %", "14,71 %"] and ge(p45[-1][1]) == "0,74 GE"
    p40 = _err(Fr(2, 5), 1000)
    assert pct(p40[0]) == "1,56 %" and p40[2] == "K" and M.delta(Fr(2, 5), 200, 100, 30) == -10 and M.delta(Fr(9, 20), 200, 100, 30) == 5
    _readme_has("n = 10: 37,70 % (193/512, Verlust 7,54 GE je Wiederholung), 20: 25,17 %, 40: 21,48 %, 100: 9,67 %, 200: 2,80 %, 400: 0,40 %",
                "50,44 % / 41,43 % / 43,91 % / 38,28 % / 31,01 % / 25,71 % und bei n = 1 000 noch 14,71 % (Verlust 0,74 GE)", "bei n = 1 000: 1,56 %", "Vorsprung 20 GE", "Vorsprung 5 GE", "Vorsprung von K 10 GE")


def test_b14_the_error_rate_is_not_monotone():
    a = ev.analyse_repeat(ev.Settings())
    assert a.n_rises == 255 and len(C.REPEAT_N_RASTER) - 1 == 590
    e = [_err(Fr(9, 20), n)[0] for n in (10, 20, 40)]
    assert e[1] < e[0] and e[2] > e[1]
    _readme_has("255 von 590 Schritten", "50,44 % (n = 10), 41,43 % (n = 20), 43,91 % (n = 40)")


# --- B15: drei Zustände ---------------------------------------------------------------------------------------------------------------------------------

def test_b15_three_states_table():
    rows = {N: ev.dirichlet_row(Q.counts_from_shares(N, (30, 40, 30))) for N in C.N3_TABLE}
    assert Q.area_share_no_data() == Fr(225, 496) and pct(float(Q.area_share_no_data())) == "45,36 %"
    assert [pct(rows[N]["share"]) for N in C.N3_TABLE] == ["45,36 %", "41,51 %", "36,80 %", "27,68 %", "15,51 %", "3,25 %"]
    assert [de(rows[N]["loss"], 2) for N in (0, 10, 30, 100, 300)] == ["22,68", "9,88", "5,26", "2,02", "0,56"] and de(rows[1000]["loss"], 3) == "0,048"
    assert sum(p * g for p, g in zip(C.PROB3, C.PAY3_G)) == pytest.approx(54) and sum(p * k for p, k in zip(C.PROB3, C.PAY3_K)) == pytest.approx(47)
    _readme_has("225/496 = 45,36 %", "N = 10: 41,51 %, 30: 36,80 %, 100: 27,68 %, 300: 15,51 %, 1 000: 3,25 %", "22,68 / 9,88 / 5,26 / 2,02 / 0,56 / 0,048 GE", "Erwartungswert 54 gegen 47 GE, Vorsprung 7 GE")


def test_b15_the_draw_based_loss_stays_in_the_band_and_matches_the_readme():
    a = ev.analyse_three(ev.Settings(n3=10, draws=100000))
    exact = a.exact["loss"]
    assert abs(a.loss_drawn - exact) < 0.1                                           # Standardfehler der Ziehungen liegt bei etwa 0,03 GE
    assert de(exact, 2) == "9,88"
    _readme_has("9,88 GE bei N = 10 statt 9,92 GE aus Ziehungen", "9,87 GE bei 100 000 Ziehungen")
    assert de(a.loss_drawn, 2) == "9,87"


# --- B16: Ziehungen gegen exakt ---------------------------------------------------------------------------------------------------------------------------

def test_b16_draws_against_the_exact_values_as_band():
    x, method = D.beta_draws(1301, 11, 11, 100000)
    share = float((x < 13 / 30).mean())
    exact = Q.beta_cdf(11, 11, 13 / 30)
    assert method == "Ordnungsstatistik" and abs(share - exact) < 4.5 * (exact * (1 - exact) / len(x)) ** 0.5
    assert pct(share) == "26,68 %" and pct(exact) == "26,73 %"
    for n3 in (30, 10):
        a = ev.analyse_three(ev.Settings(n3=n3, draws=100000))
        assert abs(a.drawn - a.exact["share"]) < 4.5 * (a.exact["share"] * (1 - a.exact["share"]) / a.drawn_n) ** 0.5
    assert len(D.beta_draws(3, 501, 501, 200000)[0]) == 3996
    _readme_has("26,68 % gegen exakt 26,73 %", "Dirichlet bei N = 30 und N = 10", "3 996 Ziehungen", "höchstens 3 996 Ziehungen")


def test_b16_calibration_world_statements_in_the_readme_exist():
    _readme_has("Kalibrierungstest mit Zufallswelten", "Beta(20, 20) und gleichverteiltem Vorab überschätzt er den Kippanteil")


# --- B17: Regressionsanker ----------------------------------------------------------------------------------------------------------------------------

def test_b17_regression_anchor():
    assert M.p_star(Fr(100), Fr(60), Fr(5)) == Fr(13, 32)
    assert de(Q.beta_cdf(10, 12, 13 / 32), 4) == "0,3298" and de(Q.beta_cdf(10, 12, 13 / 32), 3) == "0,330"
    _readme_has("+100/−60/+5 (Schwelle 13/32)", "ergab die erste Rechnung 0,330; diese Demo rechnet 0,3298")


# --- weitere Zahlen der README --------------------------------------------------------------------------------------------------------------------------

def test_flip_share_jumps_when_the_choice_changes_example_in_the_readme():
    """Beta(20, 20), Anteil 0,6, Schwelle 160/300 = 0,533: bei n = 10 wählt das Mittel K, bei n = 20 G; der Kippanteil springt (steigt von n = 10 auf n = 20)."""
    ps = Fr(160, 300)
    tight = C.PRIORS[C.PRIOR_TIGHT]
    s10 = Q.flip_share(*Q.posterior(10, 6, tight), ps)
    s20 = Q.flip_share(*Q.posterior(20, 12, tight), ps)
    s30 = Q.flip_share(*Q.posterior(30, 18, tight), ps)
    assert (s10[1], s20[1], s30[1]) == ("K", "Gleichstand", "G")                     # n = 20: Mittel 32/60 = 8/15 = 160/300 genau auf der Schwelle
    assert s20[0] > s10[0] and s20[0] > s30[0] and float(ps) == pytest.approx(0.5333, abs=5e-5)
    _readme_has("Beta(20, 20), Anteil 0,6, Schwelle 0,533: bei n = 10 wählt das Mittel K, bei n = 20 liegt es genau auf der Schwelle, bei n = 30 wählt es G")


def test_sample_default_choices_and_the_threshold_of_the_app_defaults():
    s = ev.Settings()
    assert (s.g, s.l, s.kk, s.p, s.n, s.anteil, s.draws, s.seed, s.n3) == (200, 100, 30, 0.5, 20, 0.5, 100000, 1301, 100)
    assert s.pstar == PS and s.width == 300


def test_mean_decision_values_of_the_readme_intro():
    assert Fr(1, 2) * 200 - Fr(1, 2) * 100 == 50 and C.DEFAULT_K == 30
    _readme_has("(G 50 GE, K 30 GE, Vorsprung von G 20 GE)")


def test_draw_limits_and_budget_numbers_of_the_readme():
    assert C.MAX_UNIFORMS == 4_000_000 and D.affordable_draws(200000, 1001) == 3996
    _readme_has("Obergrenze von 4 Millionen")
