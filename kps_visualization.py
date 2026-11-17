"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from scipy import stats

import kps_constants as C
import kps_posterior as Q


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _note(fig, x, y, text, **kw):
    """Beschriftung als Annotation (kein Text-Trace: der hat keinen Hintergrund und überlappt)."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, bgcolor="rgba(255,255,255,0.85)", font=dict(color="#222", size=11), **kw)


# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------

def build_delta_line(a):
    """Vorsprung Δ(p) von G gegen K über p; Schwelle und Basiswert markiert."""
    g, l, k = a.g, a.l, a.k
    ps = np.linspace(0.0, 1.0, 101)
    dl = ps * g - (1 - ps) * l - k
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps, y=dl, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Vorsprung von G (GE)", hovertemplate="p = %{x:.2f}: %{y:.1f} GE<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="grey", width=1))
    pstar = float(a.pstar)
    if 0 <= pstar <= 1:
        fig.add_vline(x=pstar, line=dict(color=C.COLOR_K, width=2, dash="dash"))
        _note(fig, pstar, float(np.max(dl)) * 0.9, f"Schwelle p* = {pstar:.3f}".replace(".", ","), xanchor="left")
    fig.add_trace(go.Scatter(x=[float(a.p)], y=[a.delta], mode="markers", marker=dict(size=12, color=C.COLOR_G), name="Basiswert", hovertemplate="p = %{x:.2f}: %{y:.1f} GE<extra></extra>"))
    fig.update_xaxes(title_text="Wahrscheinlichkeit hoher Nachfrage p", range=[0, 1])
    fig.update_yaxes(title_text="Vorsprung von G gegen K (GE)")
    return _base(fig, 320)


def build_distance_bars(a):
    """Relative Kipp-Entfernung je Eingabe (Abstand zur Schwelle geteilt durch den Basiswert, mit Vorzeichen: Richtung der nötigen Änderung)."""
    names, vals, cols = [], [], []
    for r in a.rows:
        if r["rel"] is None:
            continue
        v = 100 * float(r["dist"]) / abs(float(r["base"]))
        names.append(r["name"])
        vals.append(v)
        cols.append(C.COLOR_G if r["in_range"] else C.COLOR_GREY)
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=cols, hovertemplate="%{y}: %{x:+.1f} % gegen den Basiswert<extra></extra>"))
    fig.update_xaxes(title_text="Änderung des Basiswerts bis zur Schwelle (%)", zeroline=True)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 280)


# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------

def build_tornado(a, rows=None, order=None, title=""):
    """Tornado-Balken: Vorsprung am unteren und oberen Ende der Spannweite je Eingabe, nach Breite sortiert; Nulllinie und Basisvorsprung."""
    rows = rows if rows is not None else a.rows
    order = order if order is not None else a.order
    by = {r["name"]: r for r in rows}
    fig = go.Figure()
    for name in order:
        r = by[name]
        lo, hi = float(min(r["d_lo"], r["d_hi"])), float(max(r["d_lo"], r["d_hi"]))
        color = "rgba(200,60,60,0.8)" if r["flips"] else C.COLOR_G
        fig.add_trace(go.Bar(x=[hi - lo], base=[lo], y=[name], orientation="h", marker_color=color, showlegend=False,
                             hovertemplate=f"{name}: {lo:.1f} bis {hi:.1f} GE (Breite {hi - lo:.1f})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="black", width=1.5))
    fig.add_vline(x=a.delta, line=dict(color="grey", width=1, dash="dot"))
    for name in order:
        r = by[name]
        flag = "kippt" if r["flips"] else ("Gleichstand" if r["tie"] else "")
        if flag:
            _note(fig, float(max(r["d_lo"], r["d_hi"])), name, flag, xanchor="left")
    fig.update_xaxes(title_text="Vorsprung von G gegen K (GE); rot = Vorzeichen wechselt")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(order))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(fig, 260)


# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------

def build_posterior(a):
    """Dichte des Posterior mit Schwelle; Fläche auf der anderen Seite der Schwelle gefüllt; Histogramm der Ziehungen darunter."""
    af, bf = float(a.a), float(a.b)
    grid = np.linspace(0.0005, 0.9995, 800)
    dens = stats.beta.pdf(grid, af, bf)
    ps = a.pstar
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a.sample, xbins=dict(start=0, end=1, size=0.01), histnorm="probability density", marker_color="rgba(76,120,168,0.28)", name="Ziehungen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid, y=dens, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Posterior-Dichte", hovertemplate="p = %{x:.3f}<extra></extra>"))
    mask = grid < ps if a.chosen != "K" else grid > ps
    fig.add_trace(go.Scatter(x=grid[mask], y=dens[mask], mode="lines", fill="tozeroy", fillcolor=C.COLOR_FLIP, line=dict(width=0), name=f"andere Alternative ({a.other}) besser", hoverinfo="skip"))
    fig.add_vline(x=ps, line=dict(color=C.COLOR_K, width=2, dash="dash"))
    fig.add_vline(x=a.mean, line=dict(color=C.COLOR_G, width=2))
    _note(fig, ps, float(dens.max()) * 0.55, "Schwelle p*", xanchor="right")
    _note(fig, a.mean, float(dens.max()) * 0.95, "Posterior-Mittel", xanchor="left")
    fig.update_xaxes(title_text="Wahrscheinlichkeit hoher Nachfrage p", range=[0, 1])
    fig.update_yaxes(title_text="Dichte")
    fig.update_layout(barmode="overlay")
    return _base(fig, 330)


def build_flip_curve(a, s):
    """Kippanteil gegen n (gerade n, k = Anteil · n gerundet) mit den Marken 5 % und 1 %; Raster n = 2 bis 1000."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.curve_n, y=100 * a.curve, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Kippanteil (exakt)", hovertemplate="n = %{x}: %{y:.2f} %<extra></extra>"))
    for lv in C.THRESHOLD_MARKS:
        fig.add_hline(y=100 * lv, line=dict(color="grey", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=[a.n], y=[100 * a.exact], mode="markers", marker=dict(size=12, color=C.COLOR_K), name="gewählte Einstellung", hovertemplate="n = %{x}: %{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Beobachtungen n (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Kippanteil (% der Ziehungen)", range=[0, max(60.0, float(100 * a.curve[:5].max()) + 5)])
    return _base(fig, 320)


# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------

def build_repeat_curve(r, s):
    """Fehlerquote über Wiederholungen gegen n (Sägezahn) und der Kippanteil des Datensatzes (k = Anteil · n) im selben Bild."""
    ns = r.curve_n
    flip = Q.flip_curve(ns[::5], s.anteil, s.prior_ab, s.pstar)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=100 * r.curve_err, mode="lines", line=dict(color=C.COLOR_K, width=2), name=f"Fehlerquote über Wiederholungen (wahres p = {float(r.ptrue):.2f})".replace(".", ","),
                             hovertemplate="n = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns[::5], y=100 * flip, mode="lines", line=dict(color=C.COLOR_LINE, width=3, dash="dash"), name="Kippanteil des Datensatzes (Ziehungen)",
                             hovertemplate="n = %{x}: %{y:.1f} %<extra></extra>"))
    top = max(float(np.max(100 * r.curve_err)), float(np.max(100 * flip)), 1.0)
    fig.update_xaxes(title_text="Zahl der Beobachtungen n (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Anteil (%)", range=[0, min(100.0, top * 1.1)])
    return _base(fig, 330)


# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------------

def build_triangle(a):
    """Ziehungen (π1, π2) im Dreieck der Zustandswahrscheinlichkeiten mit der Geraden Vorsprung = 0 (310 π1 + 160 π2 = 150)."""
    p1, p2 = a.pts
    flips = (160 * p1 + 10 * p2 - 150 * (1 - p1 - p2)) < 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, 0, 0], y=[0, 0, 1, 0], mode="lines", line=dict(color="grey", width=1), name="Dreieck π1 + π2 ≤ 1", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p1[~flips], y=p2[~flips], mode="markers", marker=dict(size=3, color="rgba(47,107,101,0.5)"), name="G vorn", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p1[flips], y=p2[flips], mode="markers", marker=dict(size=3, color="rgba(200,60,60,0.55)"), name="K vorn", hoverinfo="skip"))
    d = (160 + 150, 10 + 150, 150)
    fig.add_trace(go.Scatter(x=[0, d[2] / d[0]], y=[d[2] / d[1], 0], mode="lines", line=dict(color="black", width=2, dash="dash"), name="Vorsprung = 0", hoverinfo="skip"))
    pm = a.alpha
    tot = sum(pm)
    fig.add_trace(go.Scatter(x=[pm[0] / tot], y=[pm[1] / tot], mode="markers", marker=dict(size=11, color=C.COLOR_K, symbol="diamond"), name="Posterior-Mittel", hoverinfo="skip"))
    fig.update_xaxes(title_text="π1 (Zustand 1)", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="π2 (Zustand 2)", range=[-0.02, 1.02])
    return _base(fig, 380)


def build_three_table_curve(a):
    """Kippanteil gegen N für die gewählten Anteile (Quadratur)."""
    ns = [r["n"] for r in a.table]
    fig = go.Figure(go.Bar(x=[str(n) for n in ns], y=[100 * r["share"] for r in a.table], marker_color=C.COLOR_LINE, hovertemplate="N = %{x}: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Beobachtungen N", type="category")
    fig.update_yaxes(title_text="Kippanteil (% der Ziehungen)")
    return _base(fig, 260)
