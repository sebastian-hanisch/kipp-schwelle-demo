"""Beta- und Dirichlet-Posterior, Kippanteil, Wert der Information über p, Fehlerquote über Wiederholungen.

Modell: p = P(hohe Nachfrage) ist unbekannt; aus n gedachten früheren Perioden mit bekanntem Ausgang (k mal hoch) und einem Vorab Beta(a0, b0) folgt der Posterior
Beta(k + a0, n − k + b0). Der Vorsprung ist in p linear: Δ(p) = (G + L)·(p − p*). Zwei Zähleinheiten werden getrennt gehalten:

* **Kippanteil** (Zähleinheit Ziehung): Anteil der Posterior-Ziehungen, in denen die vom Posterior-Mittel gewählte Alternative nicht die bessere ist.
* **Fehlerquote** (Zähleinheit Wiederholung): Anteil der Wiederholungen der Erhebung bei festem wahrem p, in denen die Regel „G, wenn das Posterior-Mittel über p* liegt“
  die schlechtere Alternative wählt.

Exakte Wege: Betaverteilung über `scipy.special.betainc` (regularisierte unvollständige Betafunktion), bei ganzzahligen Parametern zusätzlich die Binomialsumme in Brüchen."""

from fractions import Fraction
from math import comb, exp, log

import numpy as np
from scipy import integrate, special, stats

import kps_constants as C
from kps_model import as_fraction, delta


def beta_cdf(a, b, x) -> float:
    """P(p < x) für p ~ Beta(a, b); 0 für x <= 0, 1 für x >= 1."""
    x = float(x)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return float(special.betainc(float(a), float(b), x))


def beta_cdf_fraction(a: int, b: int, x: Fraction) -> Fraction:
    """Dieselbe Verteilungsfunktion in Brüchen für ganzzahlige a, b: P(Bin(a + b − 1, x) >= a) (Binomialsumme)."""
    m = a + b - 1
    return sum(comb(m, j) * x**j * (1 - x) ** (m - j) for j in range(a, m + 1))


def posterior(n: int, k: int, prior) -> tuple:
    """Parameter (a, b) des Posterior Beta(k + a0, n − k + b0); prior = (a0, b0) als Brüche."""
    a0, b0 = prior
    return k + a0, n - k + b0


def posterior_mean(a, b):
    return a / (a + b)


def shortfall(a, b, ps) -> float:
    """E[(p* − p)+] für p ~ Beta(a, b), p* beliebig (Schwellen außerhalb von [0, 1] sind abgedeckt)."""
    a, b, ps = float(a), float(b), float(ps)
    return float(ps * beta_cdf(a, b, ps) - a / (a + b) * beta_cdf(a + 1, b, ps))


def flip_share(a, b, ps, tol=1e-12):
    """Kippanteil: Anteil der Ziehungen, in denen die vom Posterior-Mittel gewählte Alternative nicht die bessere ist, und die gewählte Alternative.

    Mittel über der Schwelle: G gewählt, Kippen = p < p*. Mittel unter der Schwelle: K gewählt, Kippen = p > p*. Mittel genau auf der Schwelle: Gleichstand (P(p < p*))."""
    f = beta_cdf(float(a), float(b), float(ps))
    mean = float(a / (a + b))
    if abs(mean - float(ps)) <= tol:
        return f, "Gleichstand"
    return (f, "G") if mean > float(ps) else (1.0 - f, "K")


def info_value(a, b, ps, width) -> float:
    """Wert der vollkommenen Information über p in GE (Verlust): E[max Δ] − max(E[Δ], 0) mit Δ = width·(p − p*), width = G + L.

    Gleich den ex-post-Mehrkosten gegenüber der besseren Alternative, gemittelt über die Posterior-Ziehungen (bei gewähltem G: E[(−Δ)+])."""
    a, b, ps = float(a), float(b), float(ps)
    mean = a / (a + b)
    sf = shortfall(a, b, ps)                      # E[(p* − p)+]
    return float(width * (sf if mean >= ps else sf + mean - ps))


def normal_approx(k: int, n: int, ps) -> float | None:
    """Näherung Φ(−z) mit z = (p̂ − p*)/√(p̂(1 − p̂)/n) und p̂ = k/n. Nur eine Näherung, kein Orakel; undefiniert bei p̂ = 0 oder 1 und n = 0."""
    if n <= 0 or k <= 0 or k >= n:
        return None
    ph = k / n
    z = (ph - float(ps)) / np.sqrt(ph * (1 - ph) / n)
    return float(stats.norm.cdf(-z))


def k_from_share(n: int, anteil: float) -> int:
    """Zahl der Perioden mit hoher Nachfrage aus dem Anteil (auf eine ganze Zahl gerundet, 0 bis n)."""
    return int(min(max(round(anteil * n), 0), n))


def flip_curve(ns, anteil, prior, ps):
    """Kippanteil für jedes n der Folge (k = Anteil·n gerundet)."""
    out = []
    for n in ns:
        k = k_from_share(int(n), anteil)
        a, b = posterior(int(n), k, prior)
        out.append(flip_share(a, b, ps)[0])
    return np.array(out)


def flip_curve_fast(ns: np.ndarray, anteil: float, prior, ps) -> np.ndarray:
    """Dasselbe wie `flip_curve`, vektorisiert (Raster der geraden n bis 20 000)."""
    ns = np.asarray(ns, dtype=np.int64)
    k = np.clip(np.rint(anteil * ns), 0, ns).astype(np.int64)
    a = k + float(prior[0])
    b = ns - k + float(prior[1])
    f = np.where(ps <= 0, 0.0, np.where(ps >= 1, 1.0, special.betainc(a, b, min(max(float(ps), 0.0), 1.0))))
    mean = a / (a + b)
    return np.where(mean > float(ps), f, 1.0 - f)


def last_n_above(level: float, anteil: float, prior, ps, n_max=C.N_RASTER_MAX):
    """Größtes gerades n im Raster 2, 4, …, n_max mit Kippanteil über `level` (None, wenn keines; n_max, wenn es bis zum Rand über der Marke bleibt)."""
    ns = np.arange(2, n_max + 1, 2)
    curve = flip_curve_fast(ns, anteil, prior, ps)
    above = np.flatnonzero(curve > level)
    return int(ns[above.max()]) if above.size else None


# --------------------------------------------------------------------------------------------------------------------------------------------
# Wiederholungen: Regel „G, wenn das Posterior-Mittel strikt über p* liegt“ bei festem wahrem p
# --------------------------------------------------------------------------------------------------------------------------------------------

def rule_threshold_k(n: int, prior, ps: Fraction) -> int:
    """Kleinstes k, bei dem die Regel G wählt: (k + a0)/(n + a0 + b0) > p*, exakt in Brüchen (bei Gleichheit gilt K)."""
    a0, b0 = prior
    x = ps * (n + a0 + b0) - a0                     # k > x
    kmin = int(x // 1) + 1                          # kleinste ganze Zahl strikt über x
    return max(kmin, 0)


def repeat_error(p_true, n: int, prior, ps: Fraction, g, l, k_const):
    """Fehlerquote über Wiederholungen: Anteil der Wiederholungen (je n Beobachtungen bei wahrem p), in denen die Regel die schlechtere Alternative wählt.

    Rückgabe (Fehlerquote, Verlust je Wiederholung in GE, bessere Alternative). Bei Gleichstand im wahren p (Vorsprung 0) gibt es keine schlechtere: (0, 0, 'Gleichstand')."""
    pf = as_fraction(p_true)
    d = delta(pf, g, l, k_const)
    if abs(d) <= C.TOL:
        return 0.0, 0.0, "Gleichstand"
    kmin = rule_threshold_k(n, prior, ps)
    pt = float(pf)
    if d > 0:                                        # G ist besser: Fehler, wenn die Regel K wählt (k < kmin)
        err, better = float(stats.binom.cdf(kmin - 1, n, pt)), "G"
    else:                                            # K ist besser: Fehler, wenn die Regel G wählt (k >= kmin)
        err, better = float(stats.binom.sf(kmin - 1, n, pt)), "K"
    return err, err * abs(float(d)), better


def repeat_error_fraction(p_true: Fraction, n: int, prior, ps: Fraction, g, l, k_const) -> Fraction:
    """Dieselbe Fehlerquote exakt in Brüchen (Binomialsumme über alle k), für kleine n; Orakel und Handbeispiel."""
    d = delta(p_true, g, l, k_const)
    if d == 0:
        return Fraction(0)
    kmin = rule_threshold_k(n, prior, ps)
    better_g = d > 0
    total = Fraction(0)
    for k in range(n + 1):
        chooses_g = k >= kmin
        if chooses_g != better_g:
            total += comb(n, k) * p_true**k * (1 - p_true) ** (n - k)
    return total


# --------------------------------------------------------------------------------------------------------------------------------------------
# Nicht lineare Auszahlung G2: Einsetzen des Posterior-Mittels gegen Mitteln über den Posterior
# --------------------------------------------------------------------------------------------------------------------------------------------

def g2_values(a, b, k_const, win=C.G2_WIN, lose=C.G2_LOSE):
    """Vorsprung von G2 gegen K beim Einsetzen des Posterior-Mittels und beim Mitteln über den Posterior.

    G2 zahlt `win`, wenn zwei unabhängige Perioden beide hoch sind (Wahrscheinlichkeit p²), sonst `lose`: Erwartungswert (win − lose)·p² + lose.
    Mitteln: E[p²] = a(a + 1)/((a + b)(a + b + 1)); Einsetzen: (a/(a + b))²."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)   # Feld oder Zahl
    m = a / (a + b)
    e_p2 = a * (a + 1) / ((a + b) * (a + b + 1))
    return (win - lose) * m * m + lose - k_const, (win - lose) * e_p2 + lose - k_const


def g2_disagreements(k_const, prior, n_range=C.G2_N_RANGE):
    """Zahl der Datensätze (n = n_lo bis n_hi, alle k = 0 bis n), bei denen Einsetzen und Mitteln verschieden entscheiden; dazu die Zahl der Datensätze und der kleinste Fall."""
    cases = [(n, k) for n in range(n_range[0], n_range[1] + 1) for k in range(n + 1)]
    ns = np.array([c[0] for c in cases], float)
    ks = np.array([c[1] for c in cases], float)
    a, b = ks + float(prior[0]), ns - ks + float(prior[1])
    plug, avg = g2_values(a, b, k_const)
    diff = (plug > 0) != (avg > 0)
    first = None
    idx = np.flatnonzero(diff)
    if idx.size:
        i = int(idx[0])
        first = (int(ns[i]), int(ks[i]), float(plug[i]), float(avg[i]))
    return int(diff.sum()), len(cases), first


# --------------------------------------------------------------------------------------------------------------------------------------------
# Drei Zustände: Dirichlet-Posterior, Kippgebiet 310 π1 + 160 π2 < 150 (Auszahlungen G = (250, 60, −150), K = (90, 50, 0))
# --------------------------------------------------------------------------------------------------------------------------------------------

def pay3_diff():
    """G minus K je Zustand: (160, 10, −150)."""
    return tuple(g - k for g, k in zip(C.PAY3_G, C.PAY3_K))


def counts_from_shares(n: int, shares_percent) -> tuple:
    """Zahl der Beobachtungen je Zustand: n · Anteil, auf ganze Zahlen mit Summe n gerundet (größter Rest, bei Gleichstand der kleinere Index)."""
    tot = sum(shares_percent)
    raw = [Fraction(n * s, tot) for s in shares_percent]
    base = [int(r // 1) for r in raw]
    rest = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return tuple(base)


def dirichlet3(counts) -> tuple:
    """Posterior-Parameter Dirichlet(Zahl + 1) bei gleichverteiltem Vorab."""
    return tuple(int(c) + 1 for c in counts)


def dirichlet3_expected_delta(alpha) -> float:
    """E[Δ] = Σ d_i α_i / Σα mit d = G − K je Zustand."""
    d = pay3_diff()
    s = sum(alpha)
    return float(sum(di * ai for di, ai in zip(d, alpha)) / s)


def dirichlet3_region(alpha) -> tuple:
    """P(Δ < 0) und E[(−Δ)+] für π ~ Dirichlet(α), Δ = 310 π1 + 160 π2 − 150 (Quadratur in einer Dimension).

    π1 = x ~ Beta(α1, α2 + α3), π2 = (1 − x)·t mit t ~ Beta(α2, α3). Δ < 0 genau für t < t*(x) = (150 − 310 x)/(160 (1 − x)) und x < 15/31; innen geschlossene Form über die Betaverteilung."""
    d = pay3_diff()
    c = -d[2]                                                     # 150
    a1c, a2c = d[0] - d[2], d[1] - d[2]                           # 310, 160
    al1, al2, al3 = (float(x) for x in alpha)
    x0 = c / a1c

    def tstar(x):
        return (c - a1c * x) / (a2c * (1 - x))

    def f_x(x):
        """Dichte von Beta(α1, α2 + α3) an x."""
        if not 0 < x < 1:
            return 0.0
        return exp(-float(special.betaln(al1, al2 + al3)) + (al1 - 1) * log(x) + (al2 + al3 - 1) * log(1 - x))

    def inner_p(x):
        return beta_cdf(al2, al3, tstar(x))

    def inner_l(x):
        ts = tstar(x)
        return (c - a1c * x) * beta_cdf(al2, al3, ts) - a2c * (1 - x) * al2 / (al2 + al3) * beta_cdf(al2 + 1, al3, ts)

    # Dichte und Innenfunktion sind glatt; der Integrand fällt am Rand x0 gegen 0, bei großen Parametern konzentriert sich die Masse: Stützstellen an den Gipfeln
    mean_x = al1 / (al1 + al2 + al3)
    pts = [mean_x] if 0 < mean_x < x0 else None
    p = integrate.quad(lambda x: f_x(x) * inner_p(x), 0.0, x0, points=pts, epsabs=1e-12, epsrel=1e-11, limit=200)[0]
    ls = integrate.quad(lambda x: f_x(x) * inner_l(x), 0.0, x0, points=pts, epsabs=1e-12, epsrel=1e-11, limit=200)[0]
    return float(p), float(ls)


def dirichlet3_flip(alpha) -> dict:
    """Kippanteil, Verlust (Wert der Information über π), Verlust beim Kippen für Dirichlet(α); gewählt wird nach dem Posterior-Mittel."""
    p_neg, loss_neg = dirichlet3_region(alpha)
    ed = dirichlet3_expected_delta(alpha)
    chosen = "G" if ed > 0 else ("K" if ed < 0 else "Gleichstand")
    share = p_neg if ed >= 0 else 1.0 - p_neg
    loss = loss_neg if ed >= 0 else loss_neg + ed          # E[Δ+] − max(E Δ, 0) = E[(−Δ)+] + EΔ bei EΔ < 0
    when_flip = (loss_neg / p_neg) if (ed >= 0 and p_neg > 0) else None
    return dict(share=share, loss=loss, when_flip=when_flip, chosen=chosen, e_delta=ed, p_neg=p_neg)


def area_share_no_data() -> Fraction:
    """Ohne Daten (Dirichlet(1, 1, 1), gleichverteilt über das Dreieck): Anteil der Fläche im Kippgebiet, exakt: Dreieck mit den Ecken (0, 0), (150/310, 0), (0, 150/160)."""
    d = pay3_diff()
    c = -d[2]
    x_int = Fraction(c, d[0] - d[2])
    y_int = Fraction(c, d[1] - d[2])
    return (x_int * y_int / 2) / Fraction(1, 2)
