"""Kapazitätsfall in Brüchen: Vorsprung, Schwellen, Kipp-Entfernung, Tornado (Handrechnung, DEMO-PLAYBOOK §1b)."""

from fractions import Fraction as Fr

import pytest

import kps_model as M


def test_advantage_and_choice(base):
    d = M.delta(base["p"], base["g"], base["l"], base["k"])
    assert d == 20                                  # 0,5 · 200 − 0,5 · 100 − 30
    assert M.choice(float(d)) == "G"
    assert M.choice(-3.0) == "K"
    assert M.choice(0.0) == "Gleichstand" and M.choice(5e-10) == "Gleichstand"   # Toleranz 1e-9


def test_thresholds_are_exact_fractions(base):
    p, g, l, k = base.values()
    assert M.p_star(g, l, k) == Fr(13, 30)
    st = M.stars(p, g, l, k)
    assert st == {"p": Fr(13, 30), "G": 160, "L": 140, "K": 50}
    # Probe: am Schwellenwert ist der Vorsprung genau 0, dahinter negativ
    assert M.delta(Fr(13, 30), g, l, k) == 0 and M.delta(p, Fr(160), l, k) == 0
    assert M.delta(p, g, Fr(140), k) == 0 and M.delta(p, g, l, Fr(50)) == 0
    assert M.delta(Fr(13, 30) - Fr(1, 1000), g, l, k) < 0 and M.delta(p, Fr(159), l, k) < 0


def test_cost_ratio_gives_the_same_threshold(base):
    r = M.cost_ratio(base["g"], base["l"], base["k"])
    assert r == Fr(170, 130) == Fr(17, 13)          # verpasste Chance G − K = 170, unnötiger Bau L + K = 130
    assert 1 / (1 + r) == Fr(13, 30)


def test_relative_tipping_distances(base):
    rows = M.threshold_rows(base["p"], 200, 100, 30)       # ganze Zahlen werden zu Brüchen
    rel = {r["name"]: r["rel"] for r in rows}
    assert rel == {"p": Fr(2, 15), "G": Fr(1, 5), "L": Fr(2, 5), "K": Fr(2, 3)}
    assert [r["direction"] for r in rows] == ["fällt unter", "fällt unter", "steigt über", "steigt über"]
    assert M.nearest_input(rows) == "p"


def test_tornado_two_conventions_by_hand(base):
    p, g, l, k = base.values()
    k1 = M.tornado_rows(p, g, l, k, {n: Fr(1, 5) for n in M.NAMES})
    assert {r["name"]: r["width"] for r in k1} == {"p": 60, "G": 40, "L": 20, "K": 12}
    assert M.ranking(k1)[0] == "p"
    by = {r["name"]: r for r in k1}
    assert (by["p"]["d_lo"], by["p"]["d_hi"]) == (-10, 50) and by["p"]["flips"]
    assert by["G"]["d_lo"] == 0 and by["G"]["tie"] and not by["G"]["flips"]     # untere Grenze genau Gleichstand: kippt nicht
    k2 = M.tornado_rows(p, g, l, k, {"p": Fr(1, 5), "G": Fr(1, 2), "L": Fr(3, 10), "K": Fr(1, 20)})
    assert {r["name"]: r["width"] for r in k2} == {"p": 60, "G": 100, "L": 30, "K": 3}
    assert M.ranking(k2)[0] == "G"                   # dieselbe Entscheidung, andere Spannweiten, anderer Rang 1
    by2 = {r["name"]: r for r in k2}
    assert by2["G"]["flips"] and by2["p"]["flips"] and not by2["L"]["flips"] and not by2["K"]["flips"]
    assert (by2["G"]["d_lo"], by2["G"]["d_hi"]) == (-30, 70)


def test_width_times_distance_is_two_x_times_advantage(base):
    p, g, l, k = base.values()
    for x in (Fr(1, 20), Fr(1, 10), Fr(1, 5), Fr(2, 5)):
        prod = M.width_times_distance(p, g, l, k, x)
        assert set(prod.values()) == {2 * x * 20}    # 2x · Vorsprung für jede Eingabe


def test_all_equal_conventions_give_the_same_order(base):
    p, g, l, k = base.values()
    orders = {tuple(M.ranking(M.tornado_rows(p, g, l, k, {n: x for n in M.NAMES}))) for x in (Fr(1, 20), Fr(1, 10), Fr(1, 5), Fr(1, 2))}
    assert orders == {("p", "G", "L", "K")}


def test_derivative_times_value_by_hand(base):
    """Ableitung mal Wert: p 150, G 100, L 50, K 30; 150 > K + L = 130 (der Beweis in drei Zeilen)."""
    p, g, l, k = base.values()
    sl = M.slopes(p, g, l)
    assert sl["p"] * p == 150 and sl["G"] * g == 100 and abs(sl["L"]) * l == 50 and abs(sl["K"]) * k == 30
    assert (g + l) * p > k + l


def test_threshold_of_the_second_example_with_other_payoffs():
    """G = 300, L = 60, K = 90: Δ(1/2) = 150 − 30 − 90 = 30; p* = 150/360 = 5/12; relativ p: (1/2 − 5/12)/(1/2) = 1/6."""
    p = Fr(1, 2)
    assert M.delta(p, 300, 60, 90) == 30
    rows = {r["name"]: r for r in M.threshold_rows(p, 300, 60, 90)}
    assert rows["p"]["star"] == Fr(5, 12) and rows["p"]["rel"] == Fr(1, 6)


def test_choice_is_k_when_k_is_the_better_alternative():
    p = Fr(2, 5)                                     # unter der Schwelle 13/30
    assert M.delta(p, 200, 100, 30) == -10
    tr = M.tornado_rows(p, Fr(200), Fr(100), Fr(30), {n: Fr(1, 5) for n in M.NAMES})
    by = {r["name"]: r for r in tr}
    assert by["p"]["flips"]                          # p + 20 % = 0,48 liegt über der Schwelle: K wird von G überholt
    assert not by["K"]["flips"] and by["G"]["flips"]     # K ±20 % = 24 bis 36 hält den Vorsprung unter 0; G + 20 % = 240 kippt
