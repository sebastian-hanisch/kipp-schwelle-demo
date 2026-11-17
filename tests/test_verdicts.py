"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Zweigart steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Analysen. Die Tests setzen die Zahlen der Analyse mit `dataclasses.replace` auf glatte Werte und prüfen, dass genau
diese Zahlen, mit Dezimalkomma, im Text stehen; die Zweige (Gleichstand, Obergrenze der Ziehungen, Rasterrand) werden einzeln durchlaufen."""

from dataclasses import replace
from fractions import Fraction as Fr

import kps_constants as C
import kps_evaluation as ev


# --- Hilfen der Anzeige ---------------------------------------------------------------------------------------------------------------------

def test_german_number_formats():
    assert ev.de(0.4333, 3) == "0,433" and ev.de(-13.333, 1) == "−13,3" and ev.de(20.0, 1, sign=True) == "+20,0"
    assert ev.de_int(147366) == "147 366" and ev.de_int(1234.5, 1) == "1 234,5" and ev.de_int(-2000) == "−2 000"
    assert ev.pct(0.2673) == "26,7 %" and ev.pct(0.0001, 2) == "0,01 %"
    assert ev.pp(0.0010) == "0,1 Prozentpunkte"
    assert ev.ge(5.0123) == "5,01 GE" and ev.frac_str(Fr(13, 30)) == "13/30" and ev.frac_str(Fr(4, 2)) == "2"


# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------------

COND = dict(n_total=1000, n_ok=1000, k_p_first=222, n_pos=400, k_pos=410, n_neg=600, k_neg=70, n_cond=50, k_cond=50)


def _threshold(**kw):
    return replace(ev.analyse_threshold(ev.Settings(**kw)), cond=COND)


def test_verdict_threshold_names_every_threshold_and_the_condition():
    kind, text = ev.verdict_threshold(_threshold())
    assert kind == "info"
    assert "**G** wird gewählt, Vorsprung 20,00 GE bei p = 0,50" in text
    assert "p* = 13/30 = 0,433" in text and "r = 17/13" in text
    assert "p fällt unter 0,433 (−13,3 %)" in text and "G fällt unter 160,0 (−20,0 %)" in text
    assert "L steigt über 140,0 (+40,0 %)" in text and "K steigt über 50,0 (+66,7 %)" in text
    assert "Nächste Eingabe zur Schwelle: **p**" in text
    assert "(50 von 50 Zufallsproblemen dieser Bedingung)" in text and "nur in 22,2 % von 1 000 Zufallsproblemen" in text


def test_verdict_threshold_when_k_is_chosen_names_the_reverse_direction():
    a = _threshold(p=0.4)                                             # Vorsprung −10: K wird gewählt
    kind, text = ev.verdict_threshold(a)
    assert "**K** wird gewählt, Vorsprung 10,00 GE bei p = 0,40" in text
    assert "p steigt über 0,433" in text and "G steigt über 225,0" in text            # G* = (30 + 0,6 · 100)/0,4
    assert "gilt „p ist immer am nächsten“ nicht" in text and "nur in 11,7 % von 600 Zufallsproblemen" in text    # 70 von 600


def test_verdict_threshold_tie_is_a_warning_and_names_the_tolerance():
    kind, text = ev.verdict_threshold(_threshold(g=160))              # 0,5 · 160 − 0,5 · 100 − 30 = 0
    assert kind == "warning" and "**Gleichstand**" in text and "0 GE" in text and "1e-09" in text


def test_verdict_threshold_marks_a_threshold_outside_the_range():
    a = _threshold(g=100, l=60, kk=100)                               # p* = 160/160 = 1: K ist nie schlechter bei p < 1
    kind, text = ev.verdict_threshold(a)
    assert "p* = 1 = 1,000" in text


# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------------

STUDY = dict(n_total=2000, n=1000, k_r1_fixed=1000, k_r1_rand=465, k_order_rand=748, k_r1_rand_vs_kipp=333, k_order_rand_vs_kipp=601,
             k_r1_equal_vs_kipp=0, k_order_equal_vs_kipp=0, k_kipp_rank1_p=1000)


def _tornado(**kw):
    s = ev.Settings(**kw)
    return replace(ev.analyse_tornado(s), study=STUDY), s


def test_verdict_tornado_default_names_both_conventions_and_counts():
    a, s = _tornado()
    kind, text = ev.verdict_tornado(a, s)
    assert kind == "info"
    assert "**p** die breiteste Spanne (60,0 GE)" in text
    assert "bei „alle ±20 %“ steht p auf Rang 1" in text and "steht G auf Rang 1: dieselbe Entscheidung" in text
    assert "wechselt Rang 1 in 46,5 % von 1 000 Zufallsproblemen (Zufalls-Seed 1301" in text and "in 53,5 % bleibt er gleich" in text
    assert "die ganze Reihenfolge wechselt in 74,8 %" in text
    assert "gleich 8,00 GE (= 2 · 0,2 · Vorsprung)" in text and "in 0 von 1 000 Zufallsproblemen weicht" in text


def test_verdict_tornado_other_spans_make_g_the_widest_and_say_so():
    a, s = _tornado(h_p=20, h_g=50, h_l=30, h_k=5)
    text = ev.verdict_tornado(a, s)[1]
    assert "**G** die breiteste Spanne (100,0 GE)" in text and "p ±20 %, G ±50 %, L ±30 %, K ±5 %" in text


def test_verdict_tornado_same_rank_one_under_both_conventions_is_said():
    a, s = _tornado()
    a2 = replace(a, order_other=["p", "G", "L", "K"])
    assert "(bei diesen Zahlen dieselbe Eingabe)" in ev.verdict_tornado(a2, s)[1]


def test_verdict_tornado_tie_is_a_warning():
    a, s = _tornado(g=160)
    assert ev.verdict_tornado(a, s)[0] == "warning"


def test_tornado_flags_text():
    a, _ = _tornado()
    flags = {r["name"]: ev.tornado_flags(r) for r in a.rows}
    assert flags == {"p": "kippt", "G": "Gleichstand an der Grenze", "L": "kippt nicht", "K": "kippt nicht"}


# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------------

def _posterior(**kw):
    s = ev.Settings(draws=20000, **kw)
    return replace(ev.analyse_posterior(s), exact=0.123, drawn=0.125, se=0.002, value_info=7.5, advantage=12.0, n5=148, n1=300,
                   priors={C.PRIOR_FLAT: 0.123, C.PRIOR_JEFFREYS: 0.2, C.PRIOR_TIGHT: 0.05}), s


def test_verdict_posterior_numbers_and_priors():
    a, s = _posterior()
    kind, text = ev.verdict_posterior(a, s)
    assert kind == "info"
    assert "n = 20 gedachte frühere Perioden, davon 10 mit hoher Nachfrage, Vorab gleichverteilt Beta(1, 1)" in text
    assert "das Posterior-Mittel 0,500 wählt **G**" in text and "in 12,3 % der Posterior-Ziehungen läge K vorn" in text
    assert "aus 20 000 Ziehungen: 12,5 % ± 0,2 Prozentpunkte" in text
    assert "unter 5 % ab n = 150" in text and "unter 1 % ab n = 302" in text
    assert "gleichverteilt 12,3 %, Jeffreys 20,0 %, eng Beta(20, 20) 5,0 %" in text
    assert "7,50 GE gegen den Vorsprung 12,00 GE" in text


def test_verdict_posterior_raster_edges_and_draw_cap():
    a, s = _posterior()
    text = ev.verdict_posterior(replace(a, n5=None, n1=C.N_RASTER_MAX), s)[1]
    assert "schon ab n = 2 unter 5 %" in text and "bis n = 20 000 nicht unter 1 %" in text
    capped = replace(a, drawn_n=3996)
    assert "3 996 Ziehungen (begrenzt durch die Obergrenze der Gleichverteilten)" in ev.verdict_posterior(capped, s)[1]


def test_verdict_posterior_k_chosen_names_the_other_alternative():
    a, s = _posterior(anteil=0.30)                                   # Mittel 0,3 unter der Schwelle: K gewählt, G läge vorn
    kind, text = ev.verdict_posterior(a, s)
    assert "wählt **K**" in text and "läge G vorn" in text


def test_verdict_posterior_tie_is_a_warning():
    a, s = _posterior()
    kind, text = ev.verdict_posterior(replace(a, chosen="Gleichstand"), s)
    assert kind == "warning" and "Gleichstand" in text and "12,3 %" in text


# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------------

def _repeat(**kw):
    s = ev.Settings(**kw)
    return replace(ev.analyse_repeat(s), err=0.252, loss=5.03, flip=0.267, err_1000=0.147, loss_1000=0.74, n_rises=255), s


def test_verdict_repeat_linear_names_both_count_units_and_the_mean_rule():
    a, s = _repeat()
    kind, text = ev.verdict_repeat(a, s)
    assert kind == "info"
    assert "Wahres p = 0,50 (Vorsprung von G gegen K: 20,00 GE), n = 20 Beobachtungen" in text
    assert "**25,2 % der Wiederholungen**" in text and "kostet im Mittel 5,03 GE je Wiederholung" in text
    assert "Kippanteil des Datensatzes mit 10 von 20 (Ziehungen aus dem Posterior) ist 26,7 %: andere Zähleinheit, andere Zahl" in text
    assert "Bei n = 1 000 liegt die Regel in 14,7 % der Wiederholungen falsch bei 0,74 GE" in text
    assert "in 255 von 590 Schritten" in text and "nicht monoton" in text
    assert "Bei in p linearer Auszahlung entscheidet allein das Posterior-Mittel 0,500" in text and "G wird gewählt" in text


def test_verdict_repeat_quadratic_names_plug_in_and_averaging():
    a, s = _repeat(payoff=C.PAYOFF_QUAD, n=6, anteil=0.65)            # k = round(0,65 · 6) = 4
    assert a.k == 4
    kind, text = ev.verdict_repeat(a, s)
    assert "wählt das Einsetzen des Mittels K (Vorsprung −5,00 GE), das Mitteln über den Posterior G2 (Vorsprung 3,33 GE)" in text
    assert "in 19 von 5 141 Datensätzen (n = 4 bis 100, alle k) entscheiden beide Wege verschieden; kleinster Fall n = 6, k = 4" in text
    assert "Fehlerquote über Wiederholungen bei wahrem p = 0,50: Einsetzen" in text


def test_verdict_repeat_tie_in_the_true_value_is_a_warning():
    a, s = _repeat(ptrue=0.5, g=160)                                  # Schwelle 130/260 = 1/2: wahres p genau auf der Schwelle
    kind, text = ev.verdict_repeat(a, s)
    assert a.better == "Gleichstand" and kind == "warning" and "es gibt keine schlechtere Wahl" in text and "Verlust 0 GE" in text


# --- Experiment 5 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_three_numbers():
    s = ev.Settings(draws=20000)
    a = replace(ev.analyse_three(s), drawn=0.37, se=0.003)
    a = replace(a, exact=dict(a.exact, share=0.368, loss=5.25))
    kind, text = ev.verdict_three(a, s)
    assert kind == "info"
    assert "gewählt wird **G**" in text and "225/496 = 45,4 % der Fläche" in text
    assert "Bei N = 100 Beobachtungen mit den Anteilen 30 / 40 / 30 % (30 / 40 / 30 gezählt)" in text
    assert "läge in 36,8 % der Posterior-Ziehungen die andere Alternative vorn (Quadratur; aus 20 000 Ziehungen: 37,0 % ± 0,3 Prozentpunkte)" in text
    assert "N = 10: 41,5 %, N = 30: 36,8 %, N = 100: 27,7 %, N = 300: 15,5 %, N = 1000: 3,3 %" in text
    assert "5,25 GE" in text


def test_verdict_three_tie_is_a_warning_and_the_cap_is_named():
    s = ev.Settings(draws=20000)
    a = ev.analyse_three(s)
    assert ev.verdict_three(replace(a, chosen="Gleichstand"), s)[0] == "warning"
    assert "begrenzt durch die Obergrenze" in ev.verdict_three(replace(a, drawn_n=500), s)[1]
