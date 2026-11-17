"""Auswertung der fünf Experimente und die Verdikte (reine Funktionen: `verdict_*(…) -> (Art, Text)`).

Alle Zahlen im Text stammen aus den Analysen dieser Datei; Zähleinheit und Voraussetzung stehen im selben Satz. Das Beispiel ist ein Kapazitätsfall mit ausgedachten
Auszahlungen in GE, ein Modell und kein Fall."""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np

import kps_constants as C
import kps_draws as D
import kps_model as M
import kps_posterior as Q


# --------------------------------------------------------------------------------------------------------------------------------------------
# Anzeige
# --------------------------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def de_int(x: float, nd: int = 0) -> str:
    """Zahl mit Dezimalkomma und Tausenderabstand (147366 -> 147 366)."""
    whole, _, frac = f"{abs(x):.{nd}f}".partition(".")
    out = f"{int(whole):,}".replace(",", " ")
    out = out + ("," + frac if frac else "")
    return ("−" + out) if x < 0 else out


def pct(x: float, nd: int = 1, sign: bool = False) -> str:
    """Anteil 0–1 als Prozentangabe mit Komma."""
    return de(100 * x, nd, sign) + " %"


def pp(x: float, nd: int = 1) -> str:
    """Abstand in Prozentpunkten (Standardfehler eines Anteils)."""
    return de(100 * x, nd) + " Prozentpunkte"


def frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def ge(x: float, nd: int = 2) -> str:
    return de(x, nd) + " GE"


# --------------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    g: int = C.DEFAULT_G
    l: int = C.DEFAULT_L
    kk: int = C.DEFAULT_K
    seed: int = C.DEFAULT_SEED
    draws: int = C.DEFAULT_DRAWS
    p: float = C.DEFAULT_P
    h_p: int = C.DEFAULT_H
    h_g: int = C.DEFAULT_H
    h_l: int = C.DEFAULT_H
    h_k: int = C.DEFAULT_H
    nprob: int = C.DEFAULT_NPROB
    n: int = C.DEFAULT_N
    anteil: float = C.DEFAULT_ANTEIL
    prior: str = C.DEFAULT_PRIOR
    ptrue: float = C.DEFAULT_PTRUE
    payoff: str = C.DEFAULT_PAYOFF
    n3: int = C.DEFAULT_N3
    pi1: int = C.DEFAULT_PI1
    pi2: int = C.DEFAULT_PI2

    @property
    def pstar(self) -> Fraction:
        return Fraction(self.l + self.kk, self.g + self.l)

    @property
    def width(self) -> int:
        return self.g + self.l

    @property
    def prior_ab(self):
        return C.PRIORS[self.prior]

    @property
    def k_obs(self) -> int:
        return Q.k_from_share(self.n, self.anteil)

    def h(self) -> dict:
        return {"p": Fraction(self.h_p, 100), "G": Fraction(self.h_g, 100), "L": Fraction(self.h_l, 100), "K": Fraction(self.h_k, 100)}


# --------------------------------------------------------------------------------------------------------------------------------------------
# Gerechnete Studien (feste Zufalls-Seeds, gespeichert, damit ein Regler an anderer Stelle sie nicht neu rechnet)
# --------------------------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=16)
def tornado_study(seed: int, n_total: int) -> dict:
    return M.tornado_study(seed, n_total)


@lru_cache(maxsize=4)
def condition_study(seed: int = C.COND_SEED, n_total: int = C.COND_N) -> dict:
    return M.condition_study(seed, n_total)


@lru_cache(maxsize=32)
def dirichlet_row(counts: tuple) -> dict:
    return Q.dirichlet3_flip(Q.dirichlet3(counts))


# --------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Schwelle und Kipp-Entfernung
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class ThresholdAnalysis:
    p: Fraction
    g: int
    l: int
    k: int
    delta: float
    chosen: str
    pstar: Fraction
    ratio: Fraction
    rows: list
    nearest: str | None
    cond: dict
    evpi_state: float


def evpi_state(p, g, l, k) -> float:
    """Wert der vollkommenen Information über den Zustand (hohe oder niedrige Nachfrage) bei bekanntem p in GE: p·max(G, K) + (1 − p)·max(−L, K) − max(EW(G), K)."""
    ew_g = p * g - (1 - p) * l
    return float(p * max(g, k) + (1 - p) * max(-l, k) - max(ew_g, k))


def analyse_threshold(s: Settings) -> ThresholdAnalysis:
    p = M.as_fraction(s.p)
    d = M.delta(p, s.g, s.l, s.kk)
    rows = M.threshold_rows(p, s.g, s.l, s.kk)
    return ThresholdAnalysis(p, s.g, s.l, s.kk, float(d), M.choice(float(d)), s.pstar, M.cost_ratio(Fraction(s.g), Fraction(s.l), Fraction(s.kk)),
                             rows, M.nearest_input(rows), condition_study(), evpi_state(p, s.g, s.l, s.kk))


def _row(a, name):
    return next(r for r in a.rows if r["name"] == name)


def describe_threshold(r: dict, nd: int = 1) -> str:
    """Satz je Eingabe: „p fällt unter 0,433 (−13,3 % gegen den Basiswert)“."""
    name = r["name"]
    if r["star"] is None:
        return f"{name}: keine Schwelle (Basiswert 0)"
    star = float(r["star"])
    num = de(star, 3) if name == "p" else de(star, 1)
    if r["rel"] is None:
        rel = ", relativ nicht definiert (Basiswert 0)"
    else:
        rel = f" ({de(100 * float(r['dist']) / abs(float(r['base'])), nd, sign=True)} %)"
    note = "" if r["in_range"] else (" (außerhalb des Bereichs: kippt nie)" if name == "p" else " (negativ: kippt nie)")
    return f"{name} {r['direction']} {num}{rel}{note}"


def verdict_threshold(a: ThresholdAnalysis):
    c = a.cond
    if a.chosen == "K":
        cond_text = (f"Ist K gewählt (Vorsprung unter 0), gilt „p ist immer am nächsten“ nicht: in einer festen Messung mit gelockerter Verteilung der Eingaben ist p dann nur in "
                     f"{pct(c['k_neg'] / c['n_neg'])} von {de_int(c['n_neg'])} Zufallsproblemen die nächste Eingabe.")
    else:
        cond_text = (f"Relativ zum Basiswert ist p immer am nächsten an der Schwelle, solange G gewählt ist und G > 0, L ≥ 0, K ≥ 0 gilt "
                     f"({de_int(c['k_cond'])} von {de_int(c['n_cond'])} Zufallsproblemen dieser Bedingung); ohne diese Bedingung nur in {pct(c['k_p_first'] / c['n_ok'])} "
                     f"von {de_int(c['n_ok'])} Zufallsproblemen (gelockerte Verteilung der Eingaben, feste Messung).")
    if a.chosen == "Gleichstand":
        return "warning", (f"Kapazitätsfall (ausgedachtes Zahlenbeispiel): **Gleichstand**, der Vorsprung ist 0 GE (Toleranz {C.TOL:g}); p liegt genau auf der Schwelle "
                           f"p* = {frac_str(a.pstar)}, jede Eingabe auf ihrer: die Kipp-Entfernung ist 0 und die Wahl eine Frage der Konvention.")
    lines = "; ".join(describe_threshold(r) for r in a.rows)
    return "info", (f"Kapazitätsfall (ausgedachtes Zahlenbeispiel, Modell und kein Fall): **{a.chosen}** wird gewählt, Vorsprung {ge(abs(a.delta))} bei p = {de(float(a.p), 2)}. "
                    f"Schwelle p* = {frac_str(a.pstar)} = {de(float(a.pstar), 3)} (Kostenverhältnis verpasste Chance zu unnötigem Bau r = {frac_str(a.ratio)}, p* = 1/(1 + r)). "
                    f"Die Entscheidung kippt: {lines}. Nächste Eingabe zur Schwelle: **{a.nearest}**. {cond_text}")


# --------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Tornado
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class TornadoAnalysis:
    rows: list
    order: list
    rows_equal: list
    order_equal: list
    rows_other: list
    order_other: list
    kipp_order: list
    products: dict
    study: dict
    chosen: str
    delta: float


def analyse_tornado(s: Settings) -> TornadoAnalysis:
    p = M.as_fraction(s.p)
    d = float(M.delta(p, s.g, s.l, s.kk))
    h = s.h()
    rows = M.tornado_rows(p, s.g, s.l, s.kk, h)
    rows_equal = M.tornado_rows(p, s.g, s.l, s.kk, {n: Fraction(C.CONV_ALL_EQUAL[i], 100) for i, n in enumerate(M.NAMES)})
    rows_other = M.tornado_rows(p, s.g, s.l, s.kk, {n: Fraction(C.CONV_OTHER[i], 100) for i, n in enumerate(M.NAMES)})
    th = M.threshold_rows(p, s.g, s.l, s.kk)
    cand = [r for r in th if r["rel"] is not None]
    kipp_order = [r["name"] for r in sorted(cand, key=lambda r: (r["rel"], M.NAMES.index(r["name"])))]
    products = M.width_times_distance(p, s.g, s.l, s.kk, Fraction(1, 5))
    return TornadoAnalysis(rows, M.ranking(rows), rows_equal, M.ranking(rows_equal), rows_other, M.ranking(rows_other), kipp_order, products,
                           tornado_study(s.seed, s.nprob), M.choice(d), d)


def tornado_flags(r: dict) -> str:
    return "kippt" if r["flips"] else ("Gleichstand an der Grenze" if r["tie"] else "kippt nicht")


def verdict_tornado(a: TornadoAnalysis, s: Settings):
    st_ = a.study
    m = st_["n"]
    if a.chosen == "Gleichstand":
        return "warning", "Gleichstand: der Vorsprung ist 0 GE; ein Tornado-Diagramm um den Gleichstand sagt nichts darüber, was „kippt“ heißt (jede Änderung wählt die andere Alternative)."
    top, top_eq, top_other = a.order[0], a.order_equal[0], a.order_other[0]
    wid = {r["name"]: r["width"] for r in a.rows}
    cur = f"Mit Ihren Spannweiten (p ±{s.h_p} %, G ±{s.h_g} %, L ±{s.h_l} %, K ±{s.h_k} %) hat **{top}** die breiteste Spanne ({ge(float(wid[top]), 1)})"
    conv = (f"bei „alle ±{C.CONV_ALL_EQUAL[0]} %“ steht {top_eq} auf Rang 1, bei „p ±{C.CONV_OTHER[0]} %, G ±{C.CONV_OTHER[1]} %, L ±{C.CONV_OTHER[2]} %, K ±{C.CONV_OTHER[3]} %“ steht {top_other} "
            f"auf Rang 1" + (": dieselbe Entscheidung, zwei plausible Spannweiten, andere Reihenfolge." if top_eq != top_other else " (bei diesen Zahlen dieselbe Eingabe)."))
    rand = (f"Bei zufällig gezogenen relativen Spannweiten (je Eingabe gleichverteilt zwischen {int(C.RAND_H_LO * 100)} und {int(C.RAND_H_HI * 100)} %) wechselt Rang 1 in "
            f"{pct(st_['k_r1_rand'] / m)} von {de_int(m)} Zufallsproblemen (Zufalls-Seed {s.seed}; Verteilung der Probleme in der Beschreibung des Experiments), in {pct(1 - st_['k_r1_rand'] / m)} bleibt er gleich; "
            f"die ganze Reihenfolge wechselt in {pct(st_['k_order_rand'] / m)}.")
    ident = (f"Bei gleichen relativen Spannweiten ordnet die Kipp-Entfernung umgekehrt zur Breite: bei „alle ±20 %“ ist Breite × relative Kipp-Entfernung für jede Eingabe gleich "
             f"{de(float(next(iter(a.products.values()))), 2)} GE (= 2 · 0,2 · Vorsprung); in {de_int(st_['k_order_equal_vs_kipp'])} von {de_int(m)} Zufallsproblemen weicht die Reihenfolge nach Breite bei „alle ±20 %“ von der nach Kipp-Entfernung ab. "
             f"Ein Balken sagt nichts über gemeinsames Kippen mehrerer Eingaben.")
    return "info", f"{cur}; {conv} {rand} {ident}"


# --------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: geschätzte Wahrscheinlichkeit
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class PosteriorAnalysis:
    n: int
    k: int
    prior: str
    a: Fraction
    b: Fraction
    pstar: float
    mean: float
    chosen: str
    exact: float
    exact_fraction: Fraction | None
    drawn: float
    drawn_n: int
    drawn_method: str
    se: float
    approx: float | None
    value_info: float
    value_info_drawn: float
    evpi_state: float
    advantage: float
    other: str
    n5: int | None
    n1: int | None
    table: list
    priors: dict
    curve_n: np.ndarray
    curve: np.ndarray
    sample: np.ndarray


def _other(chosen: str) -> str:
    return "K" if chosen == "G" else "G"


def analyse_posterior(s: Settings) -> PosteriorAnalysis:
    n, k, prior = s.n, s.k_obs, s.prior_ab
    a, b = Q.posterior(n, k, prior)
    ps = s.pstar
    mean = float(Q.posterior_mean(a, b))
    exact, chosen = Q.flip_share(a, b, ps)
    exact_fraction = None
    if a.denominator == 1 and b.denominator == 1 and n <= 60:
        f = Q.beta_cdf_fraction(int(a), int(b), ps)
        exact_fraction = f if chosen != "K" else 1 - f
    x, method = D.beta_draws(s.seed, a, b, s.draws)
    m = len(x)
    flips = (x > float(ps)) if chosen == "K" else (x < float(ps))
    drawn = float(flips.mean())
    se = float(np.sqrt(max(drawn * (1 - drawn), 0.0) / m))
    width = s.width
    ed_exact = width * (mean - float(ps))
    dl = width * (x - float(ps))
    value_drawn = float(np.maximum(dl, 0).mean() - max(ed_exact, 0.0))
    value = Q.info_value(a, b, ps, width)
    approx = Q.normal_approx(k, n, ps)
    if approx is not None and chosen == "K":
        approx = float(1 - approx)             # Φ(−z) mit z < 0 meint P(p < p*) für p̂ < p*; der Kippanteil ist dann das Gegenstück
    n5 = Q.last_n_above(0.05, s.anteil, prior, ps)
    n1 = Q.last_n_above(0.01, s.anteil, prior, ps)
    table = []
    for nn in C.N_TABLE:
        kk = Q.k_from_share(nn, s.anteil)
        aa, bb = Q.posterior(nn, kk, prior)
        table.append(dict(n=nn, k=kk, share=Q.flip_share(aa, bb, ps)[0], value=Q.info_value(aa, bb, ps, width), approx=Q.normal_approx(kk, nn, ps)))
    priors = {}
    for key, pr in C.PRIORS.items():
        aa, bb = Q.posterior(n, k, pr)
        priors[key] = Q.flip_share(aa, bb, ps)[0]
    ns = np.arange(2, 1001, 2)
    curve = Q.flip_curve_fast(ns, s.anteil, prior, ps)
    return PosteriorAnalysis(n, k, s.prior, a, b, float(ps), mean, chosen, exact, exact_fraction, drawn, m, method, se, approx, value, value_drawn,
                             evpi_state(mean, s.g, s.l, s.kk), abs(ed_exact), _other(chosen), n5, n1, table, priors, ns, curve, x[:20000])


def from_n_text(level_pct: int, n_last: int | None) -> str:
    if n_last is None:
        return f"schon ab n = 2 unter {level_pct} %"
    if n_last >= C.N_RASTER_MAX:
        return f"bis n = {de_int(C.N_RASTER_MAX)} nicht unter {level_pct} %"
    return f"unter {level_pct} % ab n = {de_int(n_last + 2)}"


def verdict_posterior(a: PosteriorAnalysis, s: Settings):
    if a.chosen == "Gleichstand":
        return "warning", ("Gleichstand: das Posterior-Mittel liegt genau auf der Schwelle; die Wahl ist eine Frage der Konvention, der Kippanteil zählt hier P(p < p*) "
                           f"= {pct(a.exact)}.")
    nenner = f"{de_int(a.drawn_n)} Ziehungen" + (" (begrenzt durch die Obergrenze der Gleichverteilten)" if a.drawn_n < s.draws else "")
    pr = a.priors
    prior_text = (f"Nach Vorab (n und k wie oben): gleichverteilt {pct(pr[C.PRIOR_FLAT])}, Jeffreys {pct(pr[C.PRIOR_JEFFREYS])}, eng Beta(20, 20) {pct(pr[C.PRIOR_TIGHT])} "
                  f"(dieselben Daten, die Zahl hängt am Vorab)")
    return "info", (f"n = {a.n} gedachte frühere Perioden, davon {a.k} mit hoher Nachfrage, Vorab {C.PRIOR_LABEL[a.prior]}: das Posterior-Mittel {de(a.mean, 3)} wählt **{a.chosen}**; "
                    f"in {pct(a.exact)} der Posterior-Ziehungen läge {a.other} vorn (exakt; aus {nenner}: {pct(a.drawn)} ± {pp(a.se)}). "
                    f"Im Raster der geraden n (k = {de(100 * s.anteil, 0)} % von n, dieses Vorab) liegt der Kippanteil {from_n_text(5, a.n5)} und {from_n_text(1, a.n1)}. {prior_text}. "
                    f"Der Verlust, den die Unsicherheit über p kostet (Wert der vollkommenen Information über p, nicht über den Zustand): {ge(a.value_info)} "
                    f"gegen den Vorsprung {ge(a.advantage)} beim Posterior-Mittel.")


# --------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Ziehungen, Wiederholungen, Mittelwert
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class RepeatAnalysis:
    n: int
    k: int
    ptrue: Fraction
    delta_true: float
    better: str
    mean: float
    chosen: str
    flip: float
    err: float
    loss: float
    err_1000: float
    loss_1000: float
    table: list
    curve_n: np.ndarray
    curve_err: np.ndarray
    n_rises: int
    quad: dict | None
    kmin: int
    delta_mean: float


def _g2_error(p_true: Fraction, n: int, prior, k_const: float):
    """Fehlerquote über Wiederholungen für G2 mit den beiden Regeln „Einsetzen“ und „Mitteln“ (jeweils G2 gewählt, wenn der Vorsprung strikt über 0 liegt)."""
    from scipy import stats

    ks = np.arange(n + 1, dtype=float)
    plug, avg = Q.g2_values(ks + float(prior[0]), n - ks + float(prior[1]), k_const)
    pt = float(p_true)
    d_true = (C.G2_WIN - C.G2_LOSE) * pt * pt + C.G2_LOSE - k_const
    pmf = stats.binom.pmf(np.arange(n + 1), n, pt)
    better_g2 = d_true > C.TOL
    out = {}
    for name, v in (("Einsetzen", plug), ("Mitteln", avg)):
        wrong = (v > 0) != better_g2
        out[name] = float(pmf[wrong].sum()) if abs(d_true) > C.TOL else 0.0
    return out, float(d_true)


def analyse_repeat(s: Settings) -> RepeatAnalysis:
    n, k, prior = s.n, s.k_obs, s.prior_ab
    ps = s.pstar
    pt = M.as_fraction(s.ptrue)
    a, b = Q.posterior(n, k, prior)
    mean = float(Q.posterior_mean(a, b))
    chosen = M.choice((mean - float(ps)) * s.width, tol=1e-12)
    flip = Q.flip_share(a, b, ps)[0]
    err, loss, better = Q.repeat_error(pt, n, prior, ps, s.g, s.l, s.kk)
    err1k, loss1k, _ = Q.repeat_error(pt, 1000, prior, ps, s.g, s.l, s.kk)
    table = []
    for nn in C.N_TABLE:
        kk = Q.k_from_share(nn, s.anteil)
        aa, bb = Q.posterior(nn, kk, prior)
        e, ls, _ = Q.repeat_error(pt, nn, prior, ps, s.g, s.l, s.kk)
        table.append(dict(n=nn, flip=Q.flip_share(aa, bb, ps)[0], err=e, loss=ls))
    ns = np.array(C.REPEAT_N_RASTER)
    errs = np.array([Q.repeat_error(pt, int(nn), prior, ps, s.g, s.l, s.kk)[0] for nn in ns])
    rises = int(np.count_nonzero(np.diff(errs) > 1e-12))
    quad = None
    if s.payoff == C.PAYOFF_QUAD:
        plug, avg = Q.g2_values(a, b, s.kk)
        errs_g2, d_true = _g2_error(pt, n, prior, s.kk)
        n_dis, n_cases, first = Q.g2_disagreements(s.kk, prior)
        quad = dict(plug=float(plug), avg=float(avg), err=errs_g2, d_true=d_true, n_dis=n_dis, n_cases=n_cases, first=first)
    return RepeatAnalysis(n, k, pt, float(M.delta(pt, s.g, s.l, s.kk)), better, mean, chosen, flip, err, loss, err1k, loss1k, table, ns, errs, rises, quad,
                          Q.rule_threshold_k(n, prior, ps), float((mean - float(ps)) * s.width))


def verdict_repeat(a: RepeatAnalysis, s: Settings):
    pt = de(float(a.ptrue), 2)
    if a.better == "Gleichstand":
        return "warning", (f"Gleichstand: bei wahrem p = {pt} liegt der Vorsprung bei 0 GE, beide Alternativen sind gleich gut; es gibt keine schlechtere Wahl, die Fehlerquote ist 0 und "
                           f"der Verlust 0 GE. Der Kippanteil des Datensatzes (Ziehungen) ist {pct(a.flip)}.")
    base = (f"Wahres p = {pt} (Vorsprung von {a.better} gegen {'K' if a.better == 'G' else 'G'}: {ge(abs(a.delta_true))}), n = {a.n} Beobachtungen je Erhebung: Die Regel „G, wenn das Posterior-Mittel "
            f"über der Schwelle liegt“ wählt in **{pct(a.err)} der Wiederholungen** der Erhebung die schlechtere Alternative (exakt binomial) und kostet im Mittel {ge(a.loss)} je Wiederholung. "
            f"Der Kippanteil des Datensatzes mit {a.k} von {a.n} (Ziehungen aus dem Posterior) ist {pct(a.flip)}: andere Zähleinheit, andere Zahl. "
            f"Bei n = 1 000 liegt die Regel in {pct(a.err_1000)} der Wiederholungen falsch bei {ge(a.loss_1000)} je Wiederholung. "
            f"Die Fehlerquote steigt zwischen n = {C.REPEAT_N_RASTER[0]} und n = {C.REPEAT_N_RASTER[-1]} in {a.n_rises} von {len(C.REPEAT_N_RASTER) - 1} Schritten von n auf n + 1 "
            f"(also nicht monoton; Sägezahn, vermutlich durch die Ganzzahligkeit von k, nicht weiter geprüft).")
    if a.quad is None:
        tail = (f" Bei in p linearer Auszahlung entscheidet allein das Posterior-Mittel {de(a.mean, 3)} (Erwartungswert des Vorsprungs = Vorsprung beim Mittel = {ge(a.delta_mean)}): "
                f"{a.chosen} wird gewählt.")
    else:
        q = a.quad
        pl = "G2" if q["plug"] > 0 else "K"
        av = "G2" if q["avg"] > 0 else "K"
        first = f"; kleinster Fall n = {q['first'][0]}, k = {q['first'][1]}" if q["first"] else ""
        tail = (f" Bei der nicht in p linearen Auszahlung G2 (+{C.G2_WIN} GE, wenn zwei unabhängige Perioden beide hoch sind, sonst −{abs(C.G2_LOSE)} GE) wählt das Einsetzen des Mittels {pl} "
                f"(Vorsprung {ge(q['plug'])}), das Mitteln über den Posterior {av} (Vorsprung {ge(q['avg'])}); in {q['n_dis']} von {de_int(q['n_cases'])} Datensätzen "
                f"(n = {C.G2_N_RANGE[0]} bis {C.G2_N_RANGE[1]}, alle k) entscheiden beide Wege verschieden{first}. "
                f"Fehlerquote über Wiederholungen bei wahrem p = {pt}: Einsetzen {pct(q['err']['Einsetzen'])}, Mitteln {pct(q['err']['Mitteln'])}.")
    return "info", base + tail


# --------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: drei Zustände
# --------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class ThreeAnalysis:
    n: int
    shares: tuple
    counts: tuple
    alpha: tuple
    exact: dict
    drawn: float
    drawn_n: int
    se: float
    loss_drawn: float
    table: list
    area_share: Fraction
    pts: tuple
    chosen: str


def analyse_three(s: Settings) -> ThreeAnalysis:
    pi1 = min(s.pi1, 100)
    pi2 = min(s.pi2, 100 - pi1)
    shares = (pi1, pi2, 100 - pi1 - pi2)
    counts = Q.counts_from_shares(s.n3, shares)
    alpha = Q.dirichlet3(counts)
    exact = dirichlet_row(counts)
    p1, p2, p3 = D.dirichlet3_draws(s.seed + 1, alpha, s.draws)
    dl = 160 * p1 + 10 * p2 - 150 * p3
    m = len(p1)
    flips = (dl < 0) if exact["chosen"] != "K" else (dl > 0)
    drawn = float(flips.mean())
    se = float(np.sqrt(max(drawn * (1 - drawn), 0.0) / m))
    ed = exact["e_delta"]
    loss_drawn = float(np.maximum(dl, 0).mean() - max(ed, 0.0))
    table = []
    for nn in C.N3_TABLE:
        c = Q.counts_from_shares(nn, shares)
        r = dirichlet_row(c)
        table.append(dict(n=nn, share=r["share"], loss=r["loss"], when_flip=r["when_flip"]))
    return ThreeAnalysis(s.n3, shares, counts, alpha, exact, drawn, m, se, loss_drawn, table, Q.area_share_no_data(), (p1[: C.PLOT_POINTS], p2[: C.PLOT_POINTS]), exact["chosen"])


def verdict_three(a: ThreeAnalysis, s: Settings):
    e = a.exact
    nenner = f"{de_int(a.drawn_n)} Ziehungen" + (" (begrenzt durch die Obergrenze der Gleichverteilten)" if a.drawn_n < s.draws else "")
    if a.chosen == "Gleichstand":
        return "warning", "Gleichstand: der erwartete Vorsprung ist 0 GE."
    t = {r["n"]: r for r in a.table}
    series = ", ".join(f"N = {nn}: {pct(t[nn]['share'])}" for nn in C.N3_TABLE if nn > 0)
    return "info", (f"Drei Zustände (ausgedachtes Zahlenbeispiel, erwarteter Vorsprung {ge(abs(e['e_delta']))} beim Posterior-Mittel, gewählt wird **{a.chosen}**): ohne Daten liegen "
                    f"{frac_str(a.area_share)} = {pct(float(a.area_share))} der Fläche des Dreiecks im Kippgebiet (exakt). Bei N = {a.n} Beobachtungen mit den Anteilen "
                    f"{a.shares[0]} / {a.shares[1]} / {a.shares[2]} % ({a.counts[0]} / {a.counts[1]} / {a.counts[2]} gezählt) läge in {pct(e['share'])} der Posterior-Ziehungen die andere Alternative vorn "
                    f"(Quadratur; aus {nenner}: {pct(a.drawn)} ± {pp(a.se)}). Bei gleichen Anteilen: {series}. Wert der Information über die Zustandswahrscheinlichkeiten: {ge(e['loss'])}.")
