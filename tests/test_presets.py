"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und die erwartete Wirkung je Preset (ein Preset ohne Wirkung meldet keine Wirkung)."""

import pytest

import kps_constants as C
import kps_evaluation as ev
import kps_presets as P

DEFAULT_PRESET = next(iter(C.PRESETS))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = preset[key]
            if spec.options is not None:
                assert v in spec.options, (name, key)
            else:
                assert spec.lo <= v <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT_PRESET]) == ev.Settings()
    assert all(spec.default == C.PRESETS[DEFAULT_PRESET][key] for key, spec in ((k, P.SETTING_SPECS[sk]) for k, sk in P.PRESET_KEYS.items()))


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)      # kein Adress-Parameter doppelt


def test_every_non_default_preset_differs_from_the_default():
    for name, preset in C.PRESETS.items():
        if name != DEFAULT_PRESET:
            assert preset != C.PRESETS[DEFAULT_PRESET], name


# --- Wirkung je Preset (Verdikt-Zeile) ------------------------------------------------------------------------------------------------------------

def _flip(name):
    return ev.analyse_posterior(ev.Settings(**{**C.PRESETS[name], "draws": 20000}))


def test_preset_by_hand_reports_the_n_10_numbers():
    a = _flip("Von Hand (n = 10)")
    assert (a.n, a.k) == (10, 5) and ev.pct(a.exact) == "32,5 %" and ev.ge(a.value_info) == "8,70 GE"


def test_preset_n_100_lowers_the_flip_share():
    a = _flip("n = 100")
    assert a.n == 100 and ev.pct(a.exact) == "8,9 %" and ev.ge(a.value_info) == "0,60 GE"
    assert a.exact < _flip(DEFAULT_PRESET).exact


def test_preset_tight_prior_reports_the_lower_number():
    a, d = _flip("Enges Vorab"), _flip(DEFAULT_PRESET)
    assert ev.pct(a.exact) == "15,1 %" and ev.pct(d.exact) == "26,7 %" and a.exact < d.exact


def test_preset_other_spans_changes_rank_one_to_g():
    s = ev.Settings(**C.PRESETS["Tornado: andere Spannweiten"])
    a = ev.analyse_tornado(s)
    assert a.order[0] == "G" and ev.analyse_tornado(ev.Settings()).order[0] == "p"


def test_preset_true_p_045_reports_the_high_error_rate():
    s = ev.Settings(**C.PRESETS["Wahres p = 0,45"])
    a = ev.analyse_repeat(s)
    assert a.n == 100 and ev.pct(a.err) == "38,3 %" and ev.ge(a.loss) == "1,91 GE"
    assert ev.pct(a.err_1000) == "14,7 %" and ev.ge(a.loss_1000) == "0,74 GE"            # auch bei n = 1 000 noch eine hohe Fehlerquote bei kleinem Verlust


def test_preset_three_states_reports_the_n_30_share():
    a = ev.analyse_three(ev.Settings(**{**C.PRESETS["Drei Zustände, N = 30"], "draws": 20000}))
    assert a.n == 30 and ev.pct(a.exact["share"]) == "36,8 %"
    assert ev.pct(ev.analyse_three(ev.Settings(draws=20000)).exact["share"]) == "27,7 %"       # die Voreinstellung N = 100 gibt eine andere Zahl


def test_the_threshold_is_not_changed_by_any_preset():
    """Alle Presets lassen den Kapazitätsfall (G, L, K) unverändert: die Schwelle bleibt 13/30."""
    for name, preset in C.PRESETS.items():
        assert ev.Settings(**preset).pstar == ev.Settings().pstar, name
