"""Kippt die Entscheidung? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein ausgedachtes Zahlenbeispiel (Kapazitätsfall, ein Modell und kein Fall): wo kippt die Wahl zwischen großem und kleinem Bau, was zeigt ein Tornado-Diagramm,
und wie oft läge die andere Alternative vorn, wenn die Wahrscheinlichkeit nur aus n Beobachtungen geschätzt ist.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kps_constants as C
import kps_evaluation as ev
import kps_posterior as Q
from kps_evaluation import Settings, de, de_int, frac_str, ge, pct, pp
from kps_layout import dunkles_design, kopf_css
from kps_presets import (
    PRESET_KEYS,
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kps_visualization import (
    build_delta_line,
    build_distance_bars,
    build_flip_curve,
    build_posterior,
    build_repeat_curve,
    build_three_table_curve,
    build_tornado,
    build_triangle,
)

st.set_page_config(page_title="Kippt die Entscheidung? – Sebastian Hanisch", layout="wide")


def _only(s: Settings, *fields) -> Settings:
    """Einstellungen, in denen nur die genannten Felder von den Voreinstellungen abweichen dürfen: so rechnet ein Regler an anderer Stelle nichts neu."""
    return Settings(**{f: getattr(s, f) for f in fields})


@st.cache_data(show_spinner=False)
def _threshold(s):
    return ev.analyse_threshold(s)


@st.cache_data(show_spinner="Zufallsprobleme werden gezählt (bei 200 000 Problemen meist unter einer Sekunde) …")
def _tornado(s):
    return ev.analyse_tornado(s)


@st.cache_data(show_spinner="Ziehungen werden gerechnet (meist unter zwei Sekunden) …")
def _posterior(s):
    return ev.analyse_posterior(s)


@st.cache_data(show_spinner=False)
def _repeat(s):
    return ev.analyse_repeat(s)


@st.cache_data(show_spinner="Ziehungen werden gerechnet (meist unter zwei Sekunden) …")
def _three(s):
    return ev.analyse_three(s)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Textzellen (Dezimalkomma und Tausenderabstand kommen aus den Formatfunktionen, nicht aus der Locale des Browsers)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kps_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Kippt die Entscheidung?")
st.markdown(
    """
Ein **ausgedachtes Zahlenbeispiel** (Kapazitätsfall, ein Modell und kein Fall; Auszahlungen in Geldeinheiten GE): **G** (groß bauen) zahlt 200 GE bei hoher und −100 GE bei niedriger Nachfrage,
**K** (klein bauen oder mieten) sicher 30 GE; **p** ist die Wahrscheinlichkeit hoher Nachfrage. Bei p = 1/2 gewinnt G mit 20 GE Vorsprung. Gefragt wird nicht, welche Alternative besser ist,
sondern **wo die Wahl kippt**: an welcher **Schwelle**, was ein **Tornado-Diagramm** davon zeigt (und was nur Konvention ist) und wie oft die andere Alternative vorn läge, wenn p nur aus
**n Beobachtungen geschätzt** ist.

Weiter unten: fünf Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Entscheidung an der Schwelle", expanded=True):
    st.markdown(
        """
1. **Vorsprung:** G gegen K ist ein Erwartungswertvergleich, der Vorsprung von G ist Δ(p) = p·G − (1 − p)·L − K. Er ist in p und in jeder anderen Eingabe **linear**.
2. **Schwelle:** Δ ist genau bei p* = (L + K)/(G + L) gleich 0. Liegt p darüber, gewinnt G, darunter K. Für G, L und K gibt es entsprechende Schwellen.
3. **Geschätztes p:** Aus n gedachten früheren Perioden mit bekanntem Ausgang (k mal hohe Nachfrage) und einem Vorab folgt eine Beta-Verteilung für p (der Posterior). Der **Kippanteil** zählt, in welchem Anteil der Ziehungen aus diesem Posterior die andere Alternative besser wäre.
4. **Zwei Zähleinheiten:** Der Kippanteil zählt Ziehungen aus dem Posterior *eines* Datensatzes. Die **Fehlerquote** zählt Wiederholungen der Erhebung bei festem wahrem p. Beides sind verschiedene Größen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kapazitätsfall (ausgedachte Auszahlungen in GE)**")
    st.slider("G: Auszahlung bei hoher Nachfrage", *bounds("g_slider"), key="g_slider", help="Groß bauen zahlt G GE, wenn die Nachfrage hoch ist.")
    st.slider("L: Verlust bei niedriger Nachfrage", *bounds("l_slider"), key="l_slider", help="Groß bauen verliert L GE, wenn die Nachfrage niedrig ist.")
    st.slider("K: sichere Auszahlung (klein bauen oder mieten)", *bounds("k_slider"), key="k_slider", help="Klein bauen oder mieten zahlt sicher K GE.")
    st.markdown("**Zufall**")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt die Ziehungen und die Zufallsprobleme (ganzzahliger SplitMix64).")
    st.slider("Zahl der Ziehungen (Experimente 3 und 5)", *bounds("draws_slider"), key="draws_slider", step=1000,
              help="Ziehungen aus dem Posterior. Bei großem n sinkt die Zahl, weil jede Ziehung n + 1 Gleichverteilte kostet (Obergrenze 4 Millionen).")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

s = Settings(**{key: st.session_state[state_key] for key, state_key in PRESET_KEYS.items()})
sync_query_params({state_key: st.session_state[state_key] for state_key in SETTING_SPECS})

a1 = _threshold(_only(s, "g", "l", "kk", "p"))
a2 = _tornado(_only(s, "g", "l", "kk", "p", "h_p", "h_g", "h_l", "h_k", "nprob", "seed"))
a3 = _posterior(_only(s, "g", "l", "kk", "seed", "draws", "n", "anteil", "prior"))
a4 = _repeat(_only(s, "g", "l", "kk", "n", "anteil", "prior", "ptrue", "payoff"))
a5 = _three(_only(s, "seed", "draws", "n3", "pi1", "pi2"))

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1. Wo kippt die Entscheidung?**")
    st.metric("Schwelle p* der Wahrscheinlichkeit", f"{frac_str(a1.pstar)} = {de(float(a1.pstar), 3)}")
    st.caption(f"Basis p = {de(float(a1.p), 2)}, Wahl {a1.chosen}")
    st.caption(f"Nächste Eingabe zur Schwelle (relativ zum Basiswert): **{a1.nearest}**. Experiment 1.")
with c2:
    st.markdown("**2. Was sagt der Tornado, was ist Konvention?**")
    st.metric("Rang 1 im Tornado bei Ihren Spannweiten", a2.order[0] if a2.chosen != "Gleichstand" else "–")
    st.caption(f"bei „alle ±20 %“: {a2.order_equal[0]}; bei anderen Spannweiten: {a2.order_other[0]}")
    st.caption("Dieselbe Entscheidung, andere Spannweiten, andere Reihenfolge. Experiment 2.")
with c3:
    st.markdown("**3. Wie oft läge die andere Alternative vorn, wenn p geschätzt ist?**")
    st.metric("Kippanteil (Zähleinheit: Ziehung)", pct(a3.exact) if a3.chosen != "Gleichstand" else "Gleichstand")
    st.caption(f"n = {a3.n}, k = {a3.k}, Vorab {C.PRIOR_LABEL[a3.prior]}")
    st.caption("Exakt aus der Betaverteilung, daneben aus Ziehungen. Experiment 3.")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wo kippt die Entscheidung? (Schwelle und Kipp-Entfernung)")
st.caption("Schwellen für p, G, L und K: der Wert, bei dem der Vorsprung 0 wird, wenn die drei anderen Eingaben auf dem Basiswert bleiben. Die Kipp-Entfernung ist der Abstand dorthin, "
           "geteilt durch den Basiswert (relativ). Gleichstand heißt: Vorsprung 0 GE (Toleranz 1e-9), dann gilt G nicht als gekippt.")
st.slider("p: Wahrscheinlichkeit hoher Nachfrage", *bounds("p_slider"), key="p_slider", step=0.01, format="%.2f",
          help="Basiswert von p für Experimente 1 und 2. Die Experimente 3 bis 5 schätzen p aus Beobachtungen.")
e1, e2 = st.columns([3, 2])
rows = []
for r in a1.rows:
    name = r["name"]
    fmt = (lambda v: de(float(v), 3)) if name == "p" else (lambda v: de(float(v), 1))
    rows.append([name, fmt(r["base"]), "–" if r["star"] is None else fmt(r["star"]),
                 "–" if r["dist"] is None else de(float(r["dist"]), 3 if name == "p" else 1, sign=True),
                 "–" if r["rel"] is None else de(100 * float(r["dist"]) / abs(float(r["base"])), 1, sign=True) + " %",
                 r["direction"] + ("" if r["in_range"] else " (kippt nie)")])
with e1:
    st.markdown(md_table(["Eingabe", "Basis", "Schwelle", "Abstand", "relativ", "kippt, wenn die Eingabe …"], rows))
    st.caption(f"Schwelle von p als Bruch: p* = (L + K)/(G + L) = {frac_str(a1.pstar)}; als Kostenverhältnis r = (G − K)/(L + K) = {frac_str(a1.ratio)} "
               f"(verpasste Chance, wenn K gewählt wird und die Nachfrage hoch ist, zu unnötigem Bau, wenn G gewählt wird und die Nachfrage niedrig ist), p* = 1/(1 + r). "
               f"Zum Vergleich: der Wert der vollkommenen Information über den Zustand (Nachfrage hoch oder niedrig) bei diesem p ist {ge(a1.evpi_state)}.")
with e2:
    st.plotly_chart(build_distance_bars(a1), width="stretch", key="chart_distance")
st.plotly_chart(build_delta_line(a1), width="stretch", key="chart_delta")
kind, text = ev.verdict_threshold(a1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Tornado – die Reihenfolge ist eine Wahl")
st.caption("Je Eingabe wird nur diese um ihre Spannweite (halbe Spannweite in Prozent des Basiswerts) verändert, die anderen bleiben auf dem Basiswert; die Breite ist der Unterschied der Vorsprünge. "
           "p ±20 % sind bei p = 0,5 gleich ±0,10. „Kippt“ heißt, dass das Vorzeichen des Vorsprungs strikt wechselt. "
           "Zufallsprobleme für die Zählung: p, G und L gleichverteilt in [0,2; 0,8], [100; 300] und [50; 150], K gleichverteilt zwischen 0 und dem Erwartungswert von G, nur Probleme mit mehr als 5 GE Erwartungswert von G "
           "(Annahme dieser Messung); zufällige Konventionen: halbe Spannweite je Eingabe gleichverteilt zwischen 5 und 50 %.")
h1, h2, h3, h4 = st.columns(4)
h1.slider("p ± (%)", *bounds("hp_slider"), key="hp_slider", step=5, help="Halbe Spannweite von p in Prozent des Basiswerts (nach oben bei p = 1 gekappt).")
h2.slider("G ± (%)", *bounds("hg_slider"), key="hg_slider", step=5, help="Halbe Spannweite von G in Prozent des Basiswerts.")
h3.slider("L ± (%)", *bounds("hl_slider"), key="hl_slider", step=5, help="Halbe Spannweite von L in Prozent des Basiswerts.")
h4.slider("K ± (%)", *bounds("hk_slider"), key="hk_slider", step=5, help="Halbe Spannweite von K in Prozent des Basiswerts.")
st.slider("Zahl der Zufallsprobleme", *bounds("nprob_slider"), key="nprob_slider", step=1000,
          help="Zufallsprobleme für die Zählung unten (nur Probleme, in denen G mit mehr als 5 GE Erwartungswert gewählt wird; Zufalls-Seed aus der Seitenleiste).")
st.plotly_chart(build_tornado(a2, title="Ihre Spannweiten"), width="stretch", key="chart_tornado")
conv_rows = [
    ["Ihre Spannweiten", " > ".join(a2.order), a2.order[0], ", ".join(f"{r['name']}: {de(float(r['width']), 1)}" for r in a2.rows)],
    [f"alle ±{C.CONV_ALL_EQUAL[0]} %", " > ".join(a2.order_equal), a2.order_equal[0], ", ".join(f"{r['name']}: {de(float(r['width']), 1)}" for r in a2.rows_equal)],
    [f"p ±{C.CONV_OTHER[0]} %, G ±{C.CONV_OTHER[1]} %, L ±{C.CONV_OTHER[2]} %, K ±{C.CONV_OTHER[3]} %", " > ".join(a2.order_other), a2.order_other[0],
     ", ".join(f"{r['name']}: {de(float(r['width']), 1)}" for r in a2.rows_other)],
    ["relative Kipp-Entfernung (kleinste zuerst)", " > ".join(a2.kipp_order), a2.kipp_order[0], "Breite × Entfernung = 2x · Vorsprung bei gleichem x"],
]
st.markdown(md_table(["Konvention", "Reihenfolge nach Breite", "Rang 1", "Breiten (GE)"], conv_rows))
flag_rows = [[r["name"], f"{de(float(r['lo']), 2)} bis {de(float(r['hi']), 2)}", f"{de(float(r['d_lo']), 1)} bis {de(float(r['d_hi']), 1)}", de(float(r["width"]), 1), ev.tornado_flags(r)] for r in a2.rows]
st.markdown(md_table(["Eingabe", "Spannweite", "Vorsprung (GE)", "Breite (GE)", "Ihre Spannweiten"], flag_rows))
kind, text = ev.verdict_tornado(a2, s)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie oft läge die andere Alternative vorn, wenn p nur geschätzt ist?")
st.caption("Annahme des Modells: n gedachte frühere Perioden mit bekanntem Ausgang (kein Datensatz), je Periode hohe Nachfrage mit derselben Wahrscheinlichkeit p, unabhängig. "
           "Der Kippanteil zählt Ziehungen aus dem Posterior (Zähleinheit: Ziehung). Der Verlust ist der Wert der vollkommenen Information über p: die ex-post-Mehrkosten gegenüber der besseren Alternative, "
           "gemittelt über die Ziehungen (im Bestand der Demos heißt das „Regret“; gemeint sind überall ex-post-Mehrkosten gegen ein Orakel, nicht das Savage-Kriterium).")
x1, x2, x3 = st.columns(3)
x1.slider("n: Zahl der Beobachtungen", *bounds("n_slider"), key="n_slider", step=2, help="Zahl der gedachten früheren Perioden (gerade Zahlen).")
x2.slider("Anteil der Perioden mit hoher Nachfrage", *bounds("anteil_slider"), key="anteil_slider", step=0.05, format="%.2f", help="k = Anteil · n, auf eine ganze Zahl gerundet.")
x3.selectbox("Vorab für p", list(C.PRIORS), key="prior_select", format_func=lambda k: C.PRIOR_LABEL[k],
             help="Gleichverteilt, Jeffreys oder eng um 1/2 (so viel Gewicht wie 40 frühere Beobachtungen).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Kippanteil exakt", pct(a3.exact, 2))
m1.caption(f"k = {a3.k} von n = {a3.n}")
m2.metric("Kippanteil aus Ziehungen", pct(a3.drawn, 2))
m2.caption(f"± {pp(a3.se, 2)} bei {de_int(a3.drawn_n)} Ziehungen")
m3.metric("Näherung Φ(−z)", "nicht definiert" if a3.approx is None else pct(a3.approx, 2))
m3.caption("Näherung, kein Orakel")
m4.metric("Verlust (Wert der Information über p)", ge(a3.value_info))
m4.caption(f"aus Ziehungen {ge(a3.value_info_drawn)}")
if a3.drawn_n < s.draws:
    st.caption(f"Bei diesem n sind nur {de_int(a3.drawn_n)} Ziehungen möglich (Obergrenze von 4 Millionen Gleichverteilten, eine Ziehung kostet {int(a3.a + a3.b - 1)} davon).")
if a3.drawn_method == "Umkehrfunktion":
    st.caption("Beim Jeffreys-Vorab sind die Beta-Parameter nicht ganzzahlig: die Ziehungen kommen aus der Umkehrfunktion der Betaverteilung, angewandt auf Gleichverteilte desselben Stroms.")
v1, v2 = st.columns(2)
v1.plotly_chart(build_posterior(a3), width="stretch", key="chart_posterior")
v2.plotly_chart(build_flip_curve(a3, s), width="stretch", key="chart_flip")
t_rows = [[r["n"], r["k"], pct(r["share"], 2), "–" if r["approx"] is None else pct(r["approx"], 2), ge(r["value"], 3)] for r in a3.table]
st.markdown(md_table(["n", "k = Anteil · n", "Kippanteil exakt", "Näherung Φ(−z)", "Verlust (GE)"], t_rows))
kind, text = ev.verdict_posterior(a3, s)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Was „Kippen“ bedeutet – Ziehungen, Wiederholungen, Mittelwert")
st.caption("Drei Zeilen für dieselben n, k und denselben Vorab wie in Experiment 3 (Regler dort): (a) das Posterior-Mittel und der Entschluss, den es allein bestimmt; (b) der Kippanteil des Datensatzes (Ziehungen); "
           "(c) die Fehlerquote der Regel über Wiederholungen der Erhebung bei festem wahrem p (exakt binomial, Zähleinheit: Wiederholung).")
y1, y2 = st.columns(2)
y1.slider("Wahres p (nur für die Wiederholungen)", *bounds("ptrue_slider"), key="ptrue_slider", step=0.01, format="%.2f",
          help="Das wahre p, bei dem die Erhebung wiederholt wird. Es kennt die Regel nicht.")
y2.radio("Auszahlung von G", [C.PAYOFF_LINEAR, C.PAYOFF_QUAD], key="payoff_radio", horizontal=True,
         help="Linear: der Kapazitätsfall der Seitenleiste. Quadratisch (G2): zahlt +220 GE, wenn zwei unabhängige Perioden beide hoch sind (Wahrscheinlichkeit p²), sonst −100 GE; K bleibt wie in der Seitenleiste.")
rows4 = [
    ["(a) Posterior-Mittel und Entschluss", f"{de(a4.mean, 3)}: {a4.chosen}", "kein Zufall; der Entschluss hängt nur an k, n und dem Vorab"],
    ["(b) Kippanteil des Datensatzes", pct(a4.flip, 2), "Ziehungen aus dem Posterior"],
    ["(c) Fehlerquote über Wiederholungen", pct(a4.err, 2), f"wahres p = {de(float(a4.ptrue), 2)}, Verlust je Wiederholung {ge(a4.loss)}"],
]
st.markdown(md_table(["Zeile", "Wert", "Zähleinheit und Voraussetzung"], rows4))
st.plotly_chart(build_repeat_curve(a4, s), width="stretch", key="chart_repeat")
t4 = [[r["n"], pct(r["flip"], 2), pct(r["err"], 2), ge(r["loss"], 2)] for r in a4.table]
st.markdown(md_table(["n", "Kippanteil (Ziehungen)", f"Fehlerquote bei wahrem p = {de(float(a4.ptrue), 2)} (Wiederholungen)", "Verlust je Wiederholung"], t4))
kind, text = ev.verdict_repeat(a4, s)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Drei Zustände – Dirichlet-Verteilung")
st.caption("Ein zweites ausgedachtes Beispiel mit drei Nachfragezuständen: G zahlt 250 / 60 / −150 GE, K zahlt 90 / 50 / 0 GE (unabhängig von G, L, K der Seitenleiste). "
           "Die Wahrscheinlichkeiten der drei Zustände sind aus N Beobachtungen geschätzt (gleichverteiltes Vorab, Dirichlet-Posterior). Der Anteil des dritten Zustands ist der Rest auf 100 %. "
           "Nach der Modellform ist das dasselbe Ziehverfahren wie bei der Rangakzeptanz von SMAA (Ziehen aus dem Simplex, linearer Wert, zählen; dort Kriteriengewichte, hier Wahrscheinlichkeiten); die dortige Demo ist nicht gelesen und die Gleichheit nicht nachgerechnet.")
z1, z2, z3 = st.columns(3)
z1.slider("N: Zahl der Beobachtungen", *bounds("n3_slider"), key="n3_slider", step=10, help="0 heißt: keine Daten, Dirichlet(1, 1, 1).")
z2.slider("Anteil Zustand 1 (%)", *bounds("pi1_slider"), key="pi1_slider", step=5, help="Anteil der Beobachtungen im ersten Zustand.")
z3.slider("Anteil Zustand 2 (%)", *bounds("pi2_slider"), key="pi2_slider", step=5, help="Anteil im zweiten Zustand; liegt die Summe über 100 %, wird Zustand 2 auf den Rest gekappt.")
q1, q2, q3 = st.columns(3)
q1.metric("Kippanteil (Quadratur)", pct(a5.exact["share"], 2))
q1.caption(f"gewählt: {a5.chosen}")
q2.metric("Kippanteil aus Ziehungen", pct(a5.drawn, 2))
q2.caption(f"± {pp(a5.se, 2)} bei {de_int(a5.drawn_n)} Ziehungen")
q3.metric("Verlust (Wert der Information über die Zustände)", ge(a5.exact["loss"]))
q3.caption(f"aus Ziehungen {ge(a5.loss_drawn)}")
if a5.drawn_n < s.draws:
    st.caption(f"Bei diesem N sind nur {de_int(a5.drawn_n)} Ziehungen möglich (Obergrenze von 4 Millionen Gleichverteilten).")
w1, w2 = st.columns([3, 2])
w1.plotly_chart(build_triangle(a5), width="stretch", key="chart_triangle")
w2.plotly_chart(build_three_table_curve(a5), width="stretch", key="chart_three")
t5 = [[r["n"], pct(r["share"], 2), ge(r["loss"], 3), "–" if r["when_flip"] is None else ge(r["when_flip"], 1)] for r in a5.table]
st.markdown(md_table(["N", "Kippanteil (Quadratur)", "Verlust (Wert der Information)", "Verlust, wenn gekippt"], t5))
kind, text = ev.verdict_three(a5, s)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
n_dis, n_cases, first = Q.g2_disagreements(s.kk, s.prior_ab)
pr = a3.priors
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Gemessen in dieser App |
|---|---|---|
| **Das Vorab ist harmlos** | Der Kippanteil hängt am Vorab: dieselben Daten (n = {a3.n}, k = {a3.k}) geben {pct(pr[C.PRIOR_FLAT])} (gleichverteilt), {pct(pr[C.PRIOR_JEFFREYS])} (Jeffreys), {pct(pr[C.PRIOR_TIGHT])} (eng Beta(20, 20)). | Experiment 3, exakt |
| **Die Auszahlung ist linear in p** | Nur dann entscheidet das Posterior-Mittel allein. Bei G2 (quadratisch in p) entscheiden Einsetzen und Mitteln in {n_dis} von {de_int(n_cases)} Datensätzen verschieden. | Experiment 4, Zählung über alle Datensätze mit n = {C.G2_N_RANGE[0]} bis {C.G2_N_RANGE[1]} |
| **Der Kippanteil sagt, wie oft man falsch liegt** | Nein: er ist die Posterior-Wahrscheinlichkeit der anderen Alternative bei *einem* Datensatz, nicht die Fehlerquote über Wiederholungen und kein Grund zu wechseln. | Experiment 4: beide Zahlen nebeneinander |
| **Die Spannweiten sind vorgegeben** | Die Reihenfolge im Tornado ist Konvention: Rang 1 wechselt bei zufälligen Spannweiten in {pct(a2.study['k_r1_rand'] / a2.study['n'])} von {de_int(a2.study['n'])} Zufallsproblemen (Experiment 2, nur diese Verteilung der Probleme). | Experiment 2 |
| **Eine Eingabe nach der anderen** | Ein Tornado-Balken variiert eine Eingabe allein; gemeinsames Kippen mehrerer Eingaben und korrelierte Eingaben sind **nicht gemessen**. | nicht gemessen |
| **Geld ist Nutzen (Risikoneutralität)** | Hier zählt nur der Erwartungswert in GE; ob das Posterior-Mittel auch für Nutzenfunktionen genügt, ist nicht gemessen (der Erwartungsnutzen ist in p zwar linear, andere Eingaben laufen aber durch die Nutzenfunktion). | nicht gemessen |
| **Die Beobachtungen sind unabhängig mit gleichem p** | Gedachte Perioden, kein Datensatz; abhängige Perioden oder wechselndes p ändern den Posterior. | nicht gemessen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Auszahlungen $G$, $L$, $K$ (ausgedacht, in GE) und die Wahrscheinlichkeit $p$ hoher Nachfrage; gesucht sind die Schwelle, bei der die Wahl kippt, die Reihenfolge der Eingaben nach ihrem
Einfluss, und der Anteil der Ziehungen, in denen die andere Alternative besser wäre, wenn $p$ aus $n$ Beobachtungen geschätzt ist.

| Symbol | Bedeutung |
|---|---|
| $G$, $L$, $K$ | Auszahlung von G bei hoher Nachfrage, Verlust von G bei niedriger Nachfrage, sichere Auszahlung von K (GE) |
| $p$, $p^*$ | Wahrscheinlichkeit hoher Nachfrage, Schwelle von $p$ |
| $\Delta$ | Vorsprung von G gegen K (GE) |
| $r$ | Kostenverhältnis verpasste Chance zu unnötigem Bau |
| $d_x$ | relative Kipp-Entfernung der Eingabe $x$ |
| $h_x$, $B_x$ | halbe Spannweite (Anteil des Basiswerts) und Breite im Tornado |
| $n$, $k$ | Zahl der gedachten Beobachtungen, davon mit hoher Nachfrage |
| $a_0$, $b_0$ | Parameter des Vorab Beta$(a_0, b_0)$ |
| $a$, $b$ | Parameter des Posterior: $a = k + a_0$, $b = n - k + b_0$ |
| $I_x(a, b)$ | regularisierte unvollständige Betafunktion (Verteilungsfunktion der Beta-Verteilung) |
| $V$ | Verlust: Wert der vollkommenen Information über $p$ (GE) |
| $\hat p$ | Anteil $k/n$ |
| $c_i$, $\pi_i$ | Zahl der Beobachtungen und Wahrscheinlichkeit im Zustand $i$ (drei Zustände) |

**Vorsprung und Schwelle.**

$$\Delta(p) = pG - (1-p)L - K = (G+L)\,(p - p^*)$$

$$p^* = \frac{L+K}{G+L} = \frac{1}{1+r}, \qquad r = \frac{G-K}{L+K}$$

G ist genau für $p > p^*$ besser. Die Schwellen der anderen Eingaben (je eine Eingabe, die übrigen auf dem Basiswert):

$$G^* = \frac{K + (1-p)L}{p}, \qquad L^* = \frac{pG - K}{1-p}, \qquad K^* = pG - (1-p)L$$

**Kipp-Entfernung und Tornado.** Die relative Kipp-Entfernung ist der Abstand zur Schwelle geteilt durch den Basiswert:

$$d_x = \frac{|x^* - x|}{|x|}$$

Der Vorsprung ist in jeder Eingabe linear. Die Breite im Tornado ist daher

$$B_x = \left|\frac{\partial \Delta}{\partial x}\right| \cdot x\,(2h_x), \qquad B_x \, d_x = 2 h_x \,|\Delta|$$

Bei gleichen $h_x$ (und ohne Kappung von $p$ bei 1) ordnet die Kipp-Entfernung also genau umgekehrt zur Breite. Ist G gewählt ($\Delta > 0$) und gilt $G > 0$, $L \ge 0$, $K \ge 0$, so ist $(G+L)\,p > G\,p$, $(1-p)L$ und $K$, und damit hat $p$ die kleinste relative Kipp-Entfernung.

**Geschätztes $p$.** Beobachtungen unabhängig, gleiches $p$; Vorab Beta$(a_0, b_0)$, Posterior Beta$(a, b)$ mit $a = k + a_0$, $b = n - k + b_0$. Der Erwartungswert des Vorsprungs ist der Vorsprung beim Posterior-Mittel:

$$\mathbb{E}[\Delta] = (G+L)\left(\frac{a}{a+b} - p^*\right)$$

Kippanteil (bei gewähltem G, Zähleinheit Ziehung) und Verlust:

$$F = I_{p^*}(a, b), \qquad V = (G+L)\left(p^* \, I_{p^*}(a, b) - \frac{a}{a+b}\, I_{p^*}(a+1, b)\right)$$

Näherung mit der Normalverteilung (nur eine Näherung):

$$F \approx \Phi(-z), \qquad z = \frac{\hat p - p^*}{\sqrt{\hat p\,(1-\hat p)/n}}$$

**Fehlerquote über Wiederholungen** (feste wahre Wahrscheinlichkeit $p$, Zähleinheit Wiederholung). Die Regel wählt G genau dann, wenn $(k + a_0)/(n + a_0 + b_0) > p^*$, also für $k \ge k_{\min}$; ist G besser:

$$P(\text{falsch}) = \sum_{k < k_{\min}} \binom{n}{k}\, p^k (1-p)^{n-k}$$

**Drei Zustände.** Zähler $c_i$, Vorab gleichverteilt, Posterior Dirichlet$(c_1 + 1, c_2 + 1, c_3 + 1)$. Mit den Auszahlungen G = (250, 60, −150) und K = (90, 50, 0):

$$\Delta = 160\,\pi_1 + 10\,\pi_2 - 150\,\pi_3 = 310\,\pi_1 + 160\,\pi_2 - 150$$

Das Kippgebiet $\Delta < 0$ ist das Dreieck mit den Ecken (0, 0), (15/31, 0) und (0, 15/16) im Dreieck der Wahrscheinlichkeiten; ohne Daten (Dirichlet(1, 1, 1), gleichverteilt) ist sein Flächenanteil $225/496$.

**Ziehen ohne Logarithmus.** Beta$(a, b)$ mit ganzzahligen $a, b$ ist die $a$-kleinste von $a + b - 1$ Gleichverteilten, Dirichlet mit ganzzahligen Parametern sind die zusammengefassten Abstände sortierter Gleichverteilter. Die Gleichverteilten kommen aus dem ganzzahligen SplitMix64; bei nicht ganzzahligen Parametern (Jeffreys-Vorab) wird die Umkehrfunktion der Betaverteilung auf Gleichverteilte angewandt.

**Hinweise zur Literatur (nicht im Wortlaut gelesen).** Schwellenwahrscheinlichkeit, Beta-Posterior und Tornado-Diagramm sind Lehrbuchstoff (etwa Clemen/Reilly, *Making Hard Decisions*, Kapitel zur Sensitivitätsanalyse, und Morgan/Henrion, *Uncertainty*, Kapitel 8; gesehen wurden nur Inhaltsverzeichnis bzw. Katalogeintrag). Briggs u. a. 2012 (Value in Health 15(5), 835–842; nur das Abstract gelesen) stellen die probabilistische Sensitivitätsanalyse mit dem Wert der vollkommenen Information dar; Felli/Hazen 1998 (Medical Decision Making 18(1), 95–109; nur eine Zusammenfassung gesehen) unterscheiden die Wahrscheinlichkeit eines Entscheidungswechsels von der Auszahlungsänderung. Alle Formeln und Zahlen dieser Demo sind eigene Rechnung in diesem Modell, bei diesen Parametern.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html)."
)
