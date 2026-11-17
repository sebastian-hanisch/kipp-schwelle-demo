"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_choice_of_prior_and_payoff_runs():
    import kps_constants as C

    for prior in C.PRIORS:
        for payoff in (C.PAYOFF_LINEAR, C.PAYOFF_QUAD):
            at = _fresh_app()
            at.selectbox(key="prior_select").set_value(prior)
            at.radio(key="payoff_radio").set_value(payoff).run(timeout=TIMEOUT)
            assert not at.exception, (prior, payoff, [str(e) for e in at.exception])


def test_all_five_verdicts_are_shown_with_the_default_numbers():
    at = _fresh_app()
    texts = [b.value for b in list(at.info) + list(at.warning) + list(at.success)]
    assert len(texts) == 5
    joined = " ".join(texts)
    assert "13/30" in joined and "26,7 %" in joined and "25,2 %" in joined and "225/496" in joined


def test_permalink_sets_the_settings_and_clamps_them():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "100"
    at.query_params["prior"] = "eng"
    at.query_params["g"] = "9999"                          # über der Reglergrenze: wird auf 400 begrenzt
    at.query_params["po"] = "unzulässig"                   # kein zulässiger Wert: Voreinstellung bleibt
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 100 and at.selectbox(key="prior_select").value == "eng"
    assert at.slider(key="g_slider").value == 400 and at.radio(key="payoff_radio").value == "linear in p"


def test_worst_case_all_heavy_sliders_at_their_maximum_at_once_ends_without_error():
    """Worst-Case-Test am Reglermaximum (DEMO-PLAYBOOK §1c): Timeout statt Zeit-Assertion. Alle aufwendigen Regler zugleich am Maximum, enges Vorab, G2."""
    import kps_constants as C

    at = _fresh_app()
    at.slider(key="n_slider").set_value(C.N_MAX)
    at.slider(key="draws_slider").set_value(C.DRAWS_MAX)
    at.slider(key="nprob_slider").set_value(C.NPROB_MAX)
    at.slider(key="n3_slider").set_value(C.N3_MAX)
    at.selectbox(key="prior_select").set_value(C.PRIOR_TIGHT)
    at.radio(key="payoff_radio").set_value(C.PAYOFF_QUAD)
    at.run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
