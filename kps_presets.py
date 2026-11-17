"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in `SETTING_SPECS`: Die Hervorhebung des aktiven Presets vergleicht jeden von ihnen mit dem Preset (Regler im Hauptbereich beim Experiment
gehören dazu). Jeder Regler ist immer sichtbar und immer wirksam (keine ausgeblendeten Regler)."""

import random
from dataclasses import dataclass

import streamlit as st

import kps_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "g_slider": SettingSpec("g", int, C.DEFAULT_G, C.G_MIN, C.G_MAX),
    "l_slider": SettingSpec("l", int, C.DEFAULT_L, C.L_MIN, C.L_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "draws_slider": SettingSpec("draws", int, C.DEFAULT_DRAWS, C.DRAWS_MIN, C.DRAWS_MAX),
    "p_slider": SettingSpec("p", float, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "hp_slider": SettingSpec("hp", int, C.DEFAULT_H, C.H_MIN, C.H_MAX),
    "hg_slider": SettingSpec("hg", int, C.DEFAULT_H, C.H_MIN, C.H_MAX),
    "hl_slider": SettingSpec("hl", int, C.DEFAULT_H, C.H_MIN, C.H_MAX),
    "hk_slider": SettingSpec("hk", int, C.DEFAULT_H, C.H_MIN, C.H_MAX),
    "nprob_slider": SettingSpec("np", int, C.DEFAULT_NPROB, C.NPROB_MIN, C.NPROB_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "anteil_slider": SettingSpec("a", float, C.DEFAULT_ANTEIL, C.ANTEIL_MIN, C.ANTEIL_MAX),
    "prior_select": SettingSpec("prior", str, C.DEFAULT_PRIOR, options=tuple(C.PRIORS)),
    "ptrue_slider": SettingSpec("pt", float, C.DEFAULT_PTRUE, C.PTRUE_MIN, C.PTRUE_MAX),
    "payoff_radio": SettingSpec("po", str, C.DEFAULT_PAYOFF, options=(C.PAYOFF_LINEAR, C.PAYOFF_QUAD)),
    "n3_slider": SettingSpec("n3", int, C.DEFAULT_N3, C.N3_MIN, C.N3_MAX),
    "pi1_slider": SettingSpec("pi1", int, C.DEFAULT_PI1, C.PI_MIN, C.PI_MAX),
    "pi2_slider": SettingSpec("pi2", int, C.DEFAULT_PI2, C.PI_MIN, C.PI_MAX),
}
PRESET_KEYS = {
    "g": "g_slider", "l": "l_slider", "kk": "k_slider", "seed": "seed_input", "draws": "draws_slider", "p": "p_slider",
    "h_p": "hp_slider", "h_g": "hg_slider", "h_l": "hl_slider", "h_k": "hk_slider", "nprob": "nprob_slider",
    "n": "n_slider", "anteil": "anteil_slider", "prior": "prior_select", "ptrue": "ptrue_slider", "payoff": "payoff_radio",
    "n3": "n3_slider", "pi1": "pi1_slider", "pi2": "pi2_slider",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Auswahlwerte nur, wenn sie zulässig sind)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None and value not in spec.options:
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    tests/test_layout.py prüft, dass jeder Regler (Seitenleiste und Hauptbereich) dort steht und seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
