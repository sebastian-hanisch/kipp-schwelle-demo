"""Kapazitätsfall (ausgedachtes Zahlenbeispiel, ein Modell und kein Fall): Vorsprung, Schwellen, Kipp-Entfernung, Tornado, Zählungen über Zufallsprobleme.

G zahlt +G GE bei hoher und −L GE bei niedriger Nachfrage, K zahlt sicher +K GE, p = P(hoch).
Vorsprung Δ(p) = p·G − (1 − p)·L − K = (G + L)·(p − p*) mit der Schwelle p* = (L + K)/(G + L).
Alle Funktionen rechnen mit Zahlen jeder Art (int, float, Fraction); die Zählungen über Zufallsprobleme sind vektorisiert (numpy)."""

from fractions import Fraction

import numpy as np

import kps_constants as C
from kps_draws import uniform_array

NAMES = ("p", "G", "L", "K")


def delta(p, g, l, k):
    """Erwartungswertvorsprung von G gegen K in GE."""
    return p * g - (1 - p) * l - k


def p_star(g, l, k):
    """Schwelle der Wahrscheinlichkeit: G ist genau für p > p* besser (G + L > 0)."""
    return (l + k) / (g + l)


def cost_ratio(g, l, k):
    """Verpasste Chance (K gewählt, Nachfrage hoch: G − K) geteilt durch unnötigen Bau (G gewählt, Nachfrage niedrig: L + K); p* = 1/(1 + r)."""
    return (g - k) / (l + k)


def choice(d, tol=C.TOL):
    """'G', 'K' oder 'Gleichstand' nach dem Vorzeichen des Vorsprungs."""
    if abs(d) <= tol:
        return "Gleichstand"
    return "G" if d > 0 else "K"


def stars(p, g, l, k):
    """Wert jeder Eingabe, bei dem der Vorsprung 0 wird (die anderen drei auf dem Basiswert)."""
    out = {"p": p_star(g, l, k)}
    out["G"] = (k + (1 - p) * l) / p if p != 0 else None
    out["L"] = (p * g - k) / (1 - p) if p != 1 else None
    out["K"] = p * g - (1 - p) * l
    return out


def slopes(p, g, l):
    """Ableitung des Vorsprungs nach jeder Eingabe (der Vorsprung ist in jeder Eingabe linear)."""
    return {"p": g + l, "G": p, "L": -(1 - p), "K": -1}


def exact(x):
    """Ganze Zahlen als Brüche (die Rechnung mit int würde bei der Division zu float); alles andere bleibt."""
    return Fraction(x) if isinstance(x, int) else x


def threshold_rows(p, g, l, k):
    """Je Eingabe: Basis, Schwelle, Abstand absolut und relativ zum Basiswert (None bei Basiswert 0), Richtung, ob die Schwelle im sinnvollen Bereich liegt."""
    p, g, l, k = exact(p), exact(g), exact(l), exact(k)
    base = {"p": p, "G": g, "L": l, "K": k}
    star = stars(p, g, l, k)
    rows = []
    for name in NAMES:
        s, b = star[name], base[name]
        if s is None:
            rows.append(dict(name=name, base=b, star=None, dist=None, rel=None, direction="", in_range=False))
            continue
        dist = s - b
        rel = None if b == 0 else abs(dist) / abs(b)
        in_range = (0 <= s <= 1) if name == "p" else s >= 0
        rows.append(dict(name=name, base=b, star=s, dist=dist, rel=rel, direction="fällt unter" if dist < 0 else "steigt über", in_range=in_range))
    return rows


def nearest_input(rows):
    """Name der Eingabe mit der kleinsten relativen Kipp-Entfernung (Eingaben mit Basiswert 0 oder ohne Schwelle ausgelassen)."""
    cand = [r for r in rows if r["rel"] is not None]
    return min(cand, key=lambda r: r["rel"])["name"] if cand else None


def tornado_rows(p, g, l, k, h):
    """Tornado: Spannweite je Eingabe um den Basiswert (halbe Spannweite h[name] als Bruchteil des Basiswerts, p nach oben bei 1 gekappt).

    Je Eingabe: Vorsprung am unteren und oberen Ende, Breite = |Δ(oben) − Δ(unten)|, kippt (Vorzeichen wechselt strikt), Gleichstand (erreicht 0, kippt nicht)."""
    base = {"p": p, "G": g, "L": l, "K": k}
    d0 = delta(p, g, l, k)
    sgn = 0 if abs(d0) <= C.TOL else (1 if d0 > 0 else -1)
    rows = []
    for name in NAMES:
        b = base[name]
        lo = b * (1 - h[name])
        hi = b * (1 + h[name])
        if name == "p":
            hi = min(hi, 1)
        vals = []
        for v in (lo, hi):
            args = dict(base)
            args[name] = v
            vals.append(delta(args["p"], args["G"], args["L"], args["K"]))
        worst = min(sgn * x for x in vals) if sgn else 0.0     # kleinster Vorsprung der gewählten Alternative im Bereich
        flips = bool(sgn) and worst < -C.TOL
        tie = bool(sgn) and not flips and worst <= C.TOL
        rows.append(dict(name=name, lo=lo, hi=hi, d_lo=vals[0], d_hi=vals[1], width=abs(vals[1] - vals[0]), flips=flips, tie=tie))
    return rows


def ranking(rows, key="width"):
    """Namen nach dem Schlüssel absteigend (Breite); bei gleicher Breite die Reihenfolge p, G, L, K."""
    order = sorted(range(len(rows)), key=lambda i: (-rows[i][key], i))
    return [rows[i]["name"] for i in order]


def width_times_distance(p, g, l, k, x):
    """Breite unter „alle ± x“ mal relative Kipp-Entfernung je Eingabe (Beweis der Identität: beides gleich 2x·|Vorsprung|, solange p nicht bei 1 gekappt wird)."""
    h = {n: x for n in NAMES}
    tr = {r["name"]: r for r in tornado_rows(p, g, l, k, h)}
    rel = {r["name"]: r["rel"] for r in threshold_rows(p, g, l, k)}
    return {n: tr[n]["width"] * rel[n] for n in NAMES if rel[n] is not None}


# --------------------------------------------------------------------------------------------------------------------------------------------
# Zufallsprobleme (Zähleinheit: Zufallsproblem). Alle Zahlen sind Zählungen dieser Verteilung der Probleme, keine allgemeinen Häufigkeiten.
# --------------------------------------------------------------------------------------------------------------------------------------------

def _widths(dp, dg, dl, dk, p, g, l, k, h):
    """Breiten des Tornado je Eingabe für Zufallsprobleme (Spalten p, G, L, K); h: Feld (Probleme, 4) der halben Spannweiten (Bruchteile)."""
    p_hi = np.minimum(p * (1 + h[:, 0]), 1.0)
    w_p = dp * (p_hi - p * (1 - h[:, 0]))
    return np.stack([w_p, dg * 2 * h[:, 1] * g, dl * 2 * h[:, 2] * l, dk * 2 * h[:, 3] * k], axis=1)


def tornado_study(seed: int, n_total: int) -> dict:
    """Zufallsprobleme: p ~ U(0,2; 0,8), G ~ U(100; 300), L ~ U(50; 150), K ~ U(0; EW(G)); nur Probleme mit EW(G) über 5 GE (G gewählt).

    Gezählt wird (Zufallsproblem als Einheit): Rang 1 des Tornado bei zwei festen Konventionen verschieden; bei zwei unabhängig gezogenen Konventionen
    (halbe Spannweite je Eingabe gleichverteilt zwischen 5 und 50 %) verschieden; Rang 1 und ganze Reihenfolge gegen die Kipp-Entfernung (= alle Spannweiten gleich);
    Rang 1 der Kipp-Entfernung ist p."""
    u = uniform_array(seed, 12 * n_total).reshape(n_total, 12)
    p = C.STUDY_P[0] + (C.STUDY_P[1] - C.STUDY_P[0]) * u[:, 0]
    g = C.STUDY_G[0] + (C.STUDY_G[1] - C.STUDY_G[0]) * u[:, 1]
    l = C.STUDY_L[0] + (C.STUDY_L[1] - C.STUDY_L[0]) * u[:, 2]
    ewg = p * g - (1 - p) * l
    ok = ewg > C.STUDY_MIN_EW
    k = u[:, 3] * ewg
    sel = np.flatnonzero(ok)
    p, g, l, k, ewg, u = p[sel], g[sel], l[sel], k[sel], ewg[sel], u[sel]
    m = len(p)
    d = ewg - k                                    # Vorsprung > 0
    dp, dg, dl, dk = g + l, p, 1 - p, np.ones_like(p)
    conv1 = np.tile(np.array(C.CONV_ALL_EQUAL, float) / 100, (m, 1))
    conv2 = np.tile(np.array(C.CONV_OTHER, float) / 100, (m, 1))
    ha = C.RAND_H_LO + (C.RAND_H_HI - C.RAND_H_LO) * u[:, 4:8]
    hb = C.RAND_H_LO + (C.RAND_H_HI - C.RAND_H_LO) * u[:, 8:12]
    w1, w2 = _widths(dp, dg, dl, dk, p, g, l, k, conv1), _widths(dp, dg, dl, dk, p, g, l, k, conv2)
    wa, wb = _widths(dp, dg, dl, dk, p, g, l, k, ha), _widths(dp, dg, dl, dk, p, g, l, k, hb)
    with np.errstate(divide="ignore"):
        dist = np.stack([d / dp / p, d / dg / g, d / dl / l, d / dk / np.maximum(k, 1e-300)], axis=1)   # relative Kipp-Entfernung
    order = lambda w: np.argsort(-w, axis=1, kind="stable")
    o1, o2, oa, ob, od = order(w1), order(w2), order(wa), order(wb), np.argsort(dist, axis=1, kind="stable")
    count = lambda mask: int(np.count_nonzero(mask))
    return dict(
        n_total=n_total, n=m,
        k_r1_fixed=count(w1.argmax(1) != w2.argmax(1)),                 # Konvention 1 gegen Konvention 2
        k_r1_rand=count(wa.argmax(1) != wb.argmax(1)),                  # zwei zufällige Konventionen A und B
        k_order_rand=count(np.any(oa != ob, axis=1)),
        k_r1_rand_vs_kipp=count(wa.argmax(1) != dist.argmin(1)),        # zufällige Konvention gegen Kipp-Entfernung
        k_order_rand_vs_kipp=count(np.any(oa != od, axis=1)),
        k_r1_equal_vs_kipp=count(w1.argmax(1) != dist.argmin(1)),       # alle gleich gegen Kipp-Entfernung (Identität: 0)
        k_order_equal_vs_kipp=count(np.any(o1 != od, axis=1)),
        k_kipp_rank1_p=count(dist.argmin(1) == 0),
    )


def condition_study(seed: int, n_total: int) -> dict:
    """Gegenprobe zu „p hat die kleinste relative Kipp-Entfernung“: Zufallsprobleme mit gelockerten Annahmen (G, L, K beliebigen Vorzeichens, beide Vorzeichen des Vorsprungs).

    Verteilung: p ~ U(0,05; 0,95), G ~ U(−100; 300), L ~ U(−100; 150), K ~ U(−100; 300). Rang 1 der relativen Kipp-Entfernung ist die Eingabe mit dem kleinsten Abstand
    |Vorsprung| / |Ableitung · Basiswert|. Gezählt wird mit und ohne die Bedingung G > 0, L ≥ 0, K ≥ 0, Vorsprung > 0."""
    u = uniform_array(seed, 4 * n_total).reshape(n_total, 4)
    p = 0.05 + 0.9 * u[:, 0]
    g = -100 + 400 * u[:, 1]
    l = -100 + 250 * u[:, 2]
    k = -100 + 400 * u[:, 3]
    d = p * g - (1 - p) * l - k
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.stack([np.abs(d) / np.abs(g + l) / p, np.abs(d) / (p * np.abs(g)), np.abs(d) / ((1 - p) * np.abs(l)), np.abs(d) / np.abs(k)], axis=1)
    ok = np.isfinite(dist).all(axis=1) & (np.abs(d) > C.TOL)
    p_first = dist.argmin(axis=1) == 0
    cond = ok & (d > 0) & (g > 0) & (l >= 0) & (k >= 0)
    count = lambda mask: int(np.count_nonzero(mask))
    return dict(
        n_total=n_total, n_ok=count(ok), k_p_first=count(ok & p_first),
        n_pos=count(ok & (d > 0)), k_pos=count(ok & (d > 0) & p_first),
        n_neg=count(ok & (d < 0)), k_neg=count(ok & (d < 0) & p_first),
        n_cond=count(cond), k_cond=count(cond & p_first),
    )


def as_fraction(x) -> Fraction:
    """Wert als Bruch (Regler mit Schritt 0,01 geben Gleitkommazahlen, die hier auf den gedachten Bruch zurückgeführt werden)."""
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10000)
