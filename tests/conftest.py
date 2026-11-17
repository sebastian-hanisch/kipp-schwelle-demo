"""Gemeinsame Fixtures und Hilfen: der Kapazitätsfall (ausgedachtes Zahlenbeispiel), an dem jede Einheit von Hand geprüft wird.

Handrechnung (alles in Brüchen): G +200 GE bei hoher, −100 GE bei niedriger Nachfrage (L = 100), K sicher +30 GE, p = 1/2:
Erwartungswert von G = 100 − 50 = 50, von K = 30, Vorsprung 20 GE; Schwelle p* = (100 + 30)/(200 + 100) = 13/30."""

import sys
from fractions import Fraction as Fr
from math import comb, factorial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def base():
    return dict(p=Fr(1, 2), g=Fr(200), l=Fr(100), k=Fr(30))


def beta_cdf_poly(a, b, x):
    """Verteilungsfunktion Beta(a, b) für ganze a, b durch Integration des Polynoms x^(a-1) (1-x)^(b-1) / B(a, b) in Brüchen (zweiter Rechenweg neben der Binomialsumme)."""
    norm = Fr(factorial(a + b - 1), factorial(a - 1) * factorial(b - 1))
    return norm * sum(Fr(comb(b - 1, i) * (-1) ** i) * x ** (a + i) / (a + i) for i in range(b))


def beta_partial_mean_poly(a, b, x):
    """E[p · 1{p < x}] für Beta(a, b) durch Polynomintegration in Brüchen."""
    norm = Fr(factorial(a + b - 1), factorial(a - 1) * factorial(b - 1))
    return norm * sum(Fr(comb(b - 1, i) * (-1) ** i) * x ** (a + 1 + i) / (a + 1 + i) for i in range(b))


def beta_moment_poly(a, b, power):
    """E[p^power] für Beta(a, b) durch Polynomintegration über [0, 1] in Brüchen."""
    norm = Fr(factorial(a + b - 1), factorial(a - 1) * factorial(b - 1))
    return norm * sum(Fr(comb(b - 1, i) * (-1) ** i) / (a + power + i) for i in range(b))
