"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Das Beispiel ist ein Kapazitätsfall mit ausgedachten Auszahlungen in Geldeinheiten (GE), ein Modell und kein Fall: G (groß bauen) zahlt G GE bei hoher und
verliert L GE bei niedriger Nachfrage, K (klein bauen oder mieten) zahlt sicher K GE. p ist die Wahrscheinlichkeit hoher Nachfrage."""

from fractions import Fraction

# --- Auszahlungen (Seitenleiste) -----------------------------------------------------------------------------------------------------------
G_MIN, G_MAX, DEFAULT_G = 100, 400, 200
L_MIN, L_MAX, DEFAULT_L = 20, 200, 100
K_MIN, K_MAX, DEFAULT_K = 0, 100, 30

# --- Experiment 1: Schwelle ----------------------------------------------------------------------------------------------------------------
P_MIN, P_MAX, DEFAULT_P = 0.05, 0.95, 0.50
TOL = 1e-9                                   # Gleichstand: |Vorsprung| <= TOL (in Brüchen exakt, in Gleitkomma bleibt ein Rest)

# --- Experiment 2: Tornado -----------------------------------------------------------------------------------------------------------------
H_MIN, H_MAX, DEFAULT_H = 5, 90, 20          # halbe Spannweite in Prozent des Basiswerts, je Eingabe
CONV_ALL_EQUAL = (20, 20, 20, 20)            # Konvention 1: alle Eingaben +/- 20 %
CONV_OTHER = (20, 50, 30, 5)                 # Konvention 2: p +/- 20 % (bei p = 0,5 sind das +/- 0,10), G +/- 50 %, L +/- 30 %, K +/- 5 %
NPROB_MIN, NPROB_MAX, DEFAULT_NPROB = 1000, 200000, 200000
RAND_H_LO, RAND_H_HI = 0.05, 0.50            # zufällige Konventionen: halbe Spannweite je Eingabe gleichverteilt zwischen 5 und 50 %
STUDY_P = (0.2, 0.8)                         # Zufallsprobleme: p, G, L gleichverteilt; K gleichverteilt zwischen 0 und dem Erwartungswert von G
STUDY_G = (100.0, 300.0)
STUDY_L = (50.0, 150.0)
STUDY_MIN_EW = 5.0                           # nur Probleme, in denen G mit mindestens 5 GE Erwartungswert gewählt wird
COND_SEED, COND_N = 1310, 400000             # feste Messung zum Rang der Kipp-Entfernung (Experiment 1)

# --- Experiment 3: geschätzte Wahrscheinlichkeit -------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 10, 1000, 20       # n gedachte frühere Perioden mit bekanntem Ausgang (Annahme des Modells, kein Datensatz)
ANTEIL_MIN, ANTEIL_MAX, DEFAULT_ANTEIL = 0.30, 0.70, 0.50
PRIOR_FLAT, PRIOR_JEFFREYS, PRIOR_TIGHT = "gleichverteilt", "Jeffreys", "eng"
PRIORS = {
    PRIOR_FLAT: (Fraction(1), Fraction(1)),
    PRIOR_JEFFREYS: (Fraction(1, 2), Fraction(1, 2)),
    PRIOR_TIGHT: (Fraction(20), Fraction(20)),
}
PRIOR_LABEL = {PRIOR_FLAT: "gleichverteilt Beta(1, 1)", PRIOR_JEFFREYS: "Jeffreys Beta(½, ½)", PRIOR_TIGHT: "eng Beta(20, 20)"}
DEFAULT_PRIOR = PRIOR_FLAT
DRAWS_MIN, DRAWS_MAX, DEFAULT_DRAWS = 1000, 200000, 100000
MAX_UNIFORMS = 4_000_000                     # Obergrenze der Gleichverteilten je Rechnung: bei großem n sinkt die Zahl der Ziehungen entsprechend
SEED_MAX, DEFAULT_SEED = 999999, 1301
N_RASTER_MAX = 20000                         # Raster der geraden n für „unter 5 % ab n“
N_TABLE = (10, 20, 40, 100, 200, 400, 1000)
THRESHOLD_MARKS = (0.05, 0.01)

# --- Experiment 4: Wiederholungen -----------------------------------------------------------------------------------------------------------
PTRUE_MIN, PTRUE_MAX, DEFAULT_PTRUE = 0.35, 0.65, 0.50
PAYOFF_LINEAR, PAYOFF_QUAD = "linear in p", "quadratisch (G2)"
DEFAULT_PAYOFF = PAYOFF_LINEAR
G2_WIN, G2_LOSE = 220, -100                  # G2 zahlt +220, wenn zwei unabhängige Perioden beide hoch sind (Wahrscheinlichkeit p²), sonst -100
G2_N_RANGE = (4, 100)                        # Datensätze n = 4 bis 100, alle k, für die Zählung „Einsetzen gegen Mitteln“
REPEAT_N_RASTER = tuple(range(10, 601))

# --- Experiment 5: drei Zustände ------------------------------------------------------------------------------------------------------------
N3_MIN, N3_MAX, DEFAULT_N3 = 0, 1000, 100
PI_MIN, PI_MAX, DEFAULT_PI1, DEFAULT_PI2 = 0, 100, 30, 40   # Anteile in Prozent; der dritte ist der Rest
PROB3 = (0.3, 0.4, 0.3)
PAY3_G = (250, 60, -150)                     # Auszahlungen von G in den drei Zuständen (ausgedacht)
PAY3_K = (90, 50, 0)
N3_TABLE = (0, 10, 30, 100, 300, 1000)
PLOT_POINTS = 2500                           # so viele Ziehungen werden im Dreieck gezeichnet

COLOR_G = "#2F6B65"
COLOR_K = "#b5651d"
COLOR_LINE = "#4c78a8"
COLOR_FLIP = "rgba(200,60,60,0.45)"
COLOR_GREY = "rgba(150,150,150,0.55)"

# --- Presets -------------------------------------------------------------------------------------------------------------------------------
DEFAULTS = {
    "g": DEFAULT_G, "l": DEFAULT_L, "kk": DEFAULT_K, "seed": DEFAULT_SEED, "draws": DEFAULT_DRAWS,
    "p": DEFAULT_P,
    "h_p": DEFAULT_H, "h_g": DEFAULT_H, "h_l": DEFAULT_H, "h_k": DEFAULT_H, "nprob": DEFAULT_NPROB,
    "n": DEFAULT_N, "anteil": DEFAULT_ANTEIL, "prior": DEFAULT_PRIOR,
    "ptrue": DEFAULT_PTRUE, "payoff": DEFAULT_PAYOFF,
    "n3": DEFAULT_N3, "pi1": DEFAULT_PI1, "pi2": DEFAULT_PI2,
}


def _preset(**changes):
    return {**DEFAULTS, **changes}


PRESETS = {
    "Voreinstellung (n = 20)": _preset(),
    "Von Hand (n = 10)": _preset(n=10),
    "n = 100": _preset(n=100),
    "Enges Vorab": _preset(prior=PRIOR_TIGHT),
    "Tornado: andere Spannweiten": _preset(h_p=CONV_OTHER[0], h_g=CONV_OTHER[1], h_l=CONV_OTHER[2], h_k=CONV_OTHER[3]),
    "Wahres p = 0,45": _preset(ptrue=0.45, n=100),
    "Drei Zustände, N = 30": _preset(n3=30),
}
PRESET_HELP = {
    "Voreinstellung (n = 20)": "Kapazitätsfall mit G +200/−100 GE und K +30 GE, p = 1/2; 20 gedachte frühere Perioden, 10 davon mit hoher Nachfrage, gleichverteiltes Vorab.",
    "Von Hand (n = 10)": "Zehn gedachte Perioden, fünf mit hoher Nachfrage: die kleinste Stichprobe, an der sich alle Zahlen von Hand nachrechnen lassen.",
    "n = 100": "Hundert gedachte Perioden: der Kippanteil sinkt, der Wert der Information über p ebenfalls.",
    "Enges Vorab": "Vorab Beta(20, 20) (so viel Gewicht wie 40 frühere Beobachtungen) statt gleichverteilt: dieselben Daten, ein anderer Kippanteil.",
    "Tornado: andere Spannweiten": "Spannweiten p ±20 %, G ±50 %, L ±30 %, K ±5 % statt alle ±20 %: dieselbe Entscheidung, eine andere Reihenfolge im Tornado-Diagramm.",
    "Wahres p = 0,45": "Das wahre p liegt nur knapp über der Schwelle: die Regel liegt auch bei großem n in vielen Wiederholungen falsch, kostet aber wenig.",
    "Drei Zustände, N = 30": "Drei Zustände mit Anteilen 0,3 / 0,4 / 0,3 und 30 Beobachtungen: Kippanteil über das Dreieck der Wahrscheinlichkeiten (Dirichlet-Verteilung).",
}
