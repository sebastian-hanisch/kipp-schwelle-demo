"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen über das Verfahren, die ohne Kenntnis des richtigen Ergebnisses gelten müssen.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (Python `random.Random` nur zum Erzeugen der Testfälle). Jede Eigenschaft gilt unter den genannten Voraussetzungen:
* nur die Zahl k der Perioden mit hoher Nachfrage zählt (sequentielles Aktualisieren gleich dem Posterior);
* Skalieren aller Auszahlungen mit c > 0 lässt Schwelle, Kipp-Entfernungen und Kippanteil unverändert und skaliert Breiten und Verlust mit c;
* Addieren derselben Konstante zu allen Auszahlungen (G + c, L − c, K + c) ändert Vorsprung, Schwelle und Kippanteil nicht;
* Vertauschen von „hoch“ und „niedrig“ spiegelt (p → 1 − p, p* → 1 − p*);
* der Kippanteil fällt in n bei festem Anteil über der Schwelle (nur im Raster der Fälle, in denen k = Anteil · n ganzzahlig ist; kein allgemeiner Satz);
* E[Vorsprung] gleich Vorsprung beim Posterior-Mittel (Auszahlung linear in p);
* alle Konventionen „alle ± x“ geben dieselbe Reihenfolge; Breite × relative Kipp-Entfernung = 2x · |Vorsprung|;
* ohne Daten ist der Kippanteil die Schwelle (bzw. ihr Gegenstück) und 225/496 bei drei Zuständen."""

import random
from fractions import Fraction as Fr

import numpy as np
import pytest
from conftest import beta_moment_poly

import kps_constants as C
import kps_evaluation as ev
import kps_model as M
import kps_posterior as Q

FLAT = C.PRIORS[C.PRIOR_FLAT]


def _cases(seed, count, need_g=False):
    rnd = random.Random(seed)
    out = []
    while len(out) < count:
        p = Fr(rnd.randint(5, 95), 100)
        g, l, k = Fr(rnd.randint(100, 400)), Fr(rnd.randint(20, 200)), Fr(rnd.randint(0, 100))
        if need_g and M.delta(p, g, l, k) <= 0:
            continue
        out.append((p, g, l, k))
    return out


def test_only_the_count_of_high_periods_matters_sequential_update_equals_batch():
    rnd = random.Random(3)
    for _ in range(30):
        seq = [rnd.random() < 0.45 for _ in range(rnd.randint(0, 40))]
        a, b = FLAT
        for hoch in seq:                                      # Beobachtung für Beobachtung
            a, b = (a + 1, b) if hoch else (a, b + 1)
        shuffled = seq[:]
        rnd.shuffle(shuffled)
        a2, b2 = FLAT
        for hoch in shuffled:
            a2, b2 = (a2 + 1, b2) if hoch else (a2, b2 + 1)
        assert (a, b) == (a2, b2) == Q.posterior(len(seq), sum(seq), FLAT)


def test_scaling_all_payoffs_scales_the_values_and_keeps_threshold_and_flip_share():
    for (p, g, l, k) in _cases(1, 30):
        for c in (Fr(1, 2), Fr(3), Fr(10)):
            assert M.p_star(g * c, l * c, k * c) == M.p_star(g, l, k)
            assert M.delta(p, g * c, l * c, k * c) == c * M.delta(p, g, l, k)
            r1 = {r["name"]: r["rel"] for r in M.threshold_rows(p, g, l, k) if r["rel"] is not None}
            r2 = {r["name"]: r["rel"] for r in M.threshold_rows(p, g * c, l * c, k * c) if r["rel"] is not None}
            assert r1 == r2
            ps = M.p_star(g, l, k)
            a, b = Q.posterior(24, 11, FLAT)
            assert Q.flip_share(a, b, ps) == Q.flip_share(a, b, M.p_star(g * c, l * c, k * c))
            assert Q.info_value(a, b, ps, float(g + l) * float(c)) == pytest.approx(float(c) * Q.info_value(a, b, ps, float(g + l)), rel=1e-12)
            t1 = {r["name"]: r["width"] for r in M.tornado_rows(p, g, l, k, {n: Fr(1, 5) for n in M.NAMES})}
            t2 = {r["name"]: r["width"] for r in M.tornado_rows(p, g * c, l * c, k * c, {n: Fr(1, 5) for n in M.NAMES})}
            assert all(t2[n] == c * t1[n] for n in M.NAMES)


def test_adding_a_constant_to_all_payoffs_changes_nothing():
    """G zahlt G bei hoch und −L bei niedrig, K zahlt K: +c zu allen Auszahlungen gibt G + c, L − c, K + c."""
    for (p, g, l, k) in _cases(2, 30):
        for c in (Fr(5), Fr(-7), Fr(40)):
            assert M.delta(p, g + c, l - c, k + c) == M.delta(p, g, l, k)
            assert M.p_star(g + c, l - c, k + c) == M.p_star(g, l, k)


def test_swapping_high_and_low_mirrors_the_probability():
    """Hoch ↔ niedrig: G' = −L, L' = −G, p' = 1 − p: gleicher Vorsprung, p*' = 1 − p*."""
    for (p, g, l, k) in _cases(3, 30):
        assert M.delta(1 - p, -l, -g, k) == M.delta(p, g, l, k)
        assert M.p_star(-l, -g, k) == 1 - M.p_star(g, l, k)


def test_the_flip_share_falls_in_n_for_fixed_share_above_the_threshold_on_the_exact_raster():
    """Nur im Raster der geraden n bei Anteil 0,5 (k = n/2) bzw. n durch 10 bei Anteil 0,6 und 0,7; Anteil über der Schwelle (p* unter dem Anteil), nur dort, wo das Posterior-Mittel über der Schwelle liegt (sonst wechselt die Wahl und mit ihr die Seite, die der Kippanteil zählt); kein allgemeiner Satz."""
    for prior in C.PRIORS.values():
        for (g, l, k) in ((200, 100, 30), (200, 100, 60), (150, 120, 20), (300, 60, 90)):
            ps = Fr(l + k, g + l)
            for anteil, ns in ((0.5, range(2, 801, 2)), (0.6, range(10, 801, 10)), (0.7, range(10, 801, 10))):
                if ps >= Fr(anteil).limit_denominator(100) - Fr(1, 100):
                    continue
                nn = np.array(list(ns))
                curve = Q.flip_curve_fast(nn, anteil, prior, ps)
                mean = (nn * anteil + float(prior[0])) / (nn + float(prior[0] + prior[1]))
                g_chosen = mean > float(ps)                  # nur dort, wo G gewählt ist: wechselt die Wahl, wechselt die Seite, die der Kippanteil zählt
                assert g_chosen.any() and np.all(np.diff(curve[g_chosen]) <= 1e-12), (prior, g, l, k, anteil)


def test_the_information_value_falls_in_n_for_fixed_share_at_one_half():
    ps = Fr(13, 30)
    values = []
    for n in range(2, 402, 2):
        a, b = Q.posterior(n, n // 2, FLAT)
        values.append(Q.info_value(a, b, ps, 300))
    assert all(x > y for x, y in zip(values, values[1:]))


def test_expected_advantage_equals_advantage_at_the_posterior_mean():
    rnd = random.Random(4)
    for _ in range(25):
        a, b = rnd.randint(1, 30), rnd.randint(1, 30)
        g, l, k = Fr(rnd.randint(100, 400)), Fr(rnd.randint(20, 200)), Fr(rnd.randint(0, 100))
        e_delta = (g + l) * beta_moment_poly(a, b, 1) - (l + k)
        assert e_delta == M.delta(Fr(a, a + b), g, l, k)


def test_every_all_equal_convention_gives_the_same_order_and_width_times_distance_is_constant():
    for (p, g, l, k) in _cases(5, 40, need_g=True):
        if p > Fr(4, 5):
            continue                                          # Kappung von p bei 1 würde die Identität stören
        orders = set()
        for x in (Fr(1, 20), Fr(1, 10), Fr(1, 5), Fr(1, 2)):
            tr = M.tornado_rows(p, g, l, k, {n: x for n in M.NAMES})
            if k != 0:
                orders.add(tuple(M.ranking(tr)))
            if p * (1 + x) <= 1:
                prod = M.width_times_distance(p, g, l, k, x)
                assert set(prod.values()) == {2 * x * M.delta(p, g, l, k)}
        assert len(orders) <= 1


def test_without_data_the_flip_share_is_the_threshold_or_its_counterpart():
    for (p, g, l, k) in _cases(6, 40):
        ps = M.p_star(g, l, k)
        share, chosen = Q.flip_share(1, 1, ps)
        if ps < Fr(1, 2):
            assert chosen == "G" and share == pytest.approx(float(ps), rel=1e-12)
        elif ps > Fr(1, 2):
            assert chosen == "K" and share == pytest.approx(1 - float(ps), rel=1e-12)
        else:
            assert chosen == "Gleichstand"


def test_same_settings_same_output_other_seed_other_draws():
    a = ev.analyse_posterior(ev.Settings(seed=5, draws=20000))
    b = ev.analyse_posterior(ev.Settings(seed=5, draws=20000))
    c = ev.analyse_posterior(ev.Settings(seed=6, draws=20000))
    assert a.drawn == b.drawn and a.value_info_drawn == b.value_info_drawn
    assert (a.drawn, a.value_info_drawn) != (c.drawn, c.value_info_drawn) and a.exact == c.exact       # exakte Werte hängen nicht am Zufalls-Seed
    t1, t2 = ev.tornado_study(11, 5000), ev.tornado_study(11, 5000)
    assert t1 == t2 and ev.tornado_study(12, 5000) != t1


def test_a_better_alternative_never_has_a_negative_information_value():
    rnd = random.Random(8)
    for _ in range(40):
        n = rnd.randint(0, 120)
        k = rnd.randint(0, n)
        g, l, kc = rnd.randint(100, 400), rnd.randint(20, 200), rnd.randint(0, 100)
        a, b = Q.posterior(n, k, rnd.choice(list(C.PRIORS.values())))
        assert Q.info_value(a, b, Fr(l + kc, g + l), g + l) >= -1e-12


def test_repeat_error_is_zero_without_a_better_alternative_and_bounded_by_one():
    rnd = random.Random(9)
    for _ in range(30):
        n = rnd.randint(1, 80)
        p_true = Fr(rnd.randint(1, 99), 100)
        g, l, k = rnd.randint(100, 400), rnd.randint(20, 200), rnd.randint(0, 100)
        err, loss, better = Q.repeat_error(p_true, n, FLAT, Fr(l + k, g + l), g, l, k)
        assert 0.0 <= err <= 1.0 and loss >= 0.0 and better in ("G", "K", "Gleichstand")
