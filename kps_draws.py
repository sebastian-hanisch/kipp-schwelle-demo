"""Ziehungen aus Beta- und Dirichlet-Verteilungen allein aus dem ganzzahligen SplitMix64 (`kps_rng.py`), ohne Logarithmus und Kosinus.

Herkunft: der Generator ist die geprüfte Vorlage `kps_rng.py` (Kopie, nicht geändert); hier steht nur die vektorisierte Fassung derselben Folge (uint64-Felder rechnen
modulo 2^64) und das Ziehverfahren über Ordnungsstatistiken:

* Beta(a, b) mit ganzzahligen a, b ist die a-kleinste von a + b − 1 Gleichverteilten.
* Dirichlet(α1, α2, α3) mit ganzzahligen α sind die Abstände von Σα − 1 sortierten Gleichverteilten, zu α_i Abständen zusammengefasst:
  π1 = u_(α1), π1 + π2 = u_(α1 + α2), π3 = 1 − u_(α1 + α2).

Das braucht keine Fließkomma-Funktion der C-Bibliothek und ist damit auf jeder Plattform bis auf die Summationsreihenfolge gleich. Der Preis: eine Ziehung kostet a + b − 1
Gleichverteilte; `MAX_UNIFORMS` begrenzt die Rechnung, bei großem n sinkt die Zahl der Ziehungen (die App zeigt die tatsächliche Zahl). Nicht ganzzahlige Parameter
(Jeffreys-Vorab Beta(k + ½, n − k + ½)) ziehen über die Umkehrfunktion `scipy.special.betaincinv` einer Gleichverteilten aus demselben Strom."""

import numpy as np
from scipy import special

import kps_constants as C
from kps_rng import MASK

GAMMA = 0x9E3779B97F4A7C15
CHUNK = 1_000_000                            # so viele Gleichverteilte werden höchstens auf einmal erzeugt (Speicher)


def splitmix_array(seed: int, n: int, offset: int = 0) -> np.ndarray:
    """Die Ausgaben offset + 1 bis offset + n von SplitMix64(seed) als uint64-Feld (vektorisiert, gleiche Folge wie die Klasse)."""
    k = np.arange(offset + 1, offset + n + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + k * np.uint64(GAMMA)          # uint64-Felder rechnen modulo 2^64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, n: int, offset: int = 0) -> np.ndarray:
    """n Gleichverteilte in [0, 1) mit 53 Bit (wie `SplitMix64.random`)."""
    return (splitmix_array(seed, n, offset) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def affordable_draws(requested: int, per_draw: int) -> int:
    """Zahl der Ziehungen, die in die Obergrenze der Gleichverteilten passt (mindestens 1)."""
    return int(max(1, min(requested, C.MAX_UNIFORMS // max(per_draw, 1))))


def _is_int(x) -> bool:
    return float(x) == int(x)


def beta_draws(seed: int, a, b, requested: int):
    """Ziehungen aus Beta(a, b). Rückgabe (Feld der Ziehungen, Verfahren). Ganzzahlige a, b: Ordnungsstatistik; sonst Umkehrfunktion."""
    if not (_is_int(a) and _is_int(b)):
        n = int(min(requested, C.DRAWS_MAX))
        return special.betaincinv(float(a), float(b), uniform_array(seed, n)), "Umkehrfunktion"
    a, b = int(a), int(b)
    m = a + b - 1
    total = affordable_draws(requested, m)
    rows_per_chunk = max(1, CHUNK // m)
    out, done = [], 0
    while done < total:
        r = min(rows_per_chunk, total - done)
        u = uniform_array(seed, r * m, offset=done * m).reshape(r, m)
        out.append(np.partition(u, a - 1, axis=1)[:, a - 1])
        done += r
    return np.concatenate(out), "Ordnungsstatistik"


def dirichlet3_draws(seed: int, alpha, requested: int):
    """Ziehungen (π1, π2, π3) aus Dirichlet(α) mit ganzzahligen α (drei Einträge); Rückgabe drei Felder gleicher Länge."""
    a1, a2, a3 = (int(x) for x in alpha)
    m = a1 + a2 + a3 - 1
    total = affordable_draws(requested, m)
    rows_per_chunk = max(1, CHUNK // m)
    i1, i2 = a1 - 1, a1 + a2 - 1
    p1, p2 = [], []
    done = 0
    while done < total:
        r = min(rows_per_chunk, total - done)
        u = uniform_array(seed, r * m, offset=done * m).reshape(r, m)
        part = np.partition(u, [i1, i2], axis=1)
        x1, x2 = part[:, i1], part[:, i2]
        p1.append(x1)
        p2.append(x2 - x1)
        done += r
    pi1, pi2 = np.concatenate(p1), np.concatenate(p2)
    return pi1, pi2, 1.0 - pi1 - pi2
