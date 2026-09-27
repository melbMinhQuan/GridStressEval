"""Run with: python3 -m streamlit run apps/streamlit_app.py"""

import json
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gridstress.models.one_interval.exact import evaluate, exact_settings, exact_inputs
from gridstress.results import (
    load_results, overall_probability,
    type_settings, split_viewer_settings, composition_weights,
)


st.set_page_config(page_title="Load violation probability", layout="wide")
st.markdown(
    '<style>[data-testid="stForm"] [data-testid="InputInstructions"] '
    '{display: none;}</style>',
    unsafe_allow_html=True,
)


def saved_results():
    return load_results()


@st.cache_data(max_entries=16, show_spinner=False)
def calculate(settings):
    return evaluate(settings)


saved = saved_results()
inputs = saved["inputs"]
defaults = exact_settings(saved)
mix_defaults = type_settings(saved)
# Match the stored input precision to the one-decimal display. Preserve a
# total of 100% when rounding the saved mix (for example, equal thirds).
rounded_mix = {key: round(value, 1) for key, value in mix_defaults.items()}
remainder_key = max(mix_defaults, key=mix_defaults.get)
rounded_mix[remainder_key] = round(
    rounded_mix[remainder_key] + 100 - sum(rounded_mix.values()), 1
)
mix_defaults = rounded_mix
for key, value in {**defaults, **mix_defaults}.items():
    if key not in st.session_state:
        st.session_state[key] = value
    if key in mix_defaults or key.startswith("probability_"):
        st.session_state[key] = round(float(st.session_state[key]), 1)


errors = []
with st.sidebar.form("model_inputs"):
    st.subheader("Model inputs")
    settings = {}
    for key, label, low, high, step in (
        ("small", "Small customer current (A)", 0, 500, 1),
        ("medium", "Medium customer current (A)", 0, 500, 1),
        ("large", "Large customer current (A)", 0, 500, 1),
        ("probability_small", "Small On probability (%)", 0, 100, 1),
        ("probability_medium", "Medium On probability (%)", 0, 100, 1),
        ("probability_large", "Large On probability (%)", 0, 100, 1),
        ("customers", "Total customers", 0, 100, 1),
        ("threshold", "Violation threshold", 0, 50000, 1),
    ):
        is_probability = key.startswith("probability_")
        if is_probability:
            st.session_state[key] = float(st.session_state[key])
        settings[key] = st.number_input(label, step=0.1 if is_probability else step,
                                        format="%.1f" if is_probability else "%d", key=key)
        if not low <= settings[key] <= high:
            message = f"{label} must be between {low:,} and {high:,}."
            st.error(message)
            errors.append(message)
    st.subheader("Customer type probabilities")
    mix_settings = {}
    for key, label in (("type_small", "Small type (%)"),
                       ("type_medium", "Medium type (%)"),
                       ("type_large", "Large type (%)")):
        mix_settings[key] = st.number_input(
            label, step=0.1, format="%.1f", key=key,
        )
        if not 0 <= mix_settings[key] <= 100:
            message = f"{label} must be between 0 and 100."
            st.error(message)
            errors.append(message)
    st.form_submit_button("Update visualization")

if errors:
    st.stop()

try:
    _, type_probabilities = split_viewer_settings(mix_settings)
except ValueError as error:
    st.error(str(error))
    st.stop()

is_saved = settings == defaults and inputs.get("method") == "exact_binomial"
with st.spinner("Calculating exact probabilities…"):
    rows = saved["results"] if is_saved else calculate(settings)
weights = composition_weights(rows, type_probabilities)
weight_labels = [
    "0%" if weight == 0 else
    f"{weight * 100:.3e}%" if weight < 0.000001 else
    f"{weight * 100:.4f}%"
    for weight in weights
]

figure = go.Figure(go.Scatter3d(
    x=[r["S"] for r in rows], y=[r["M"] for r in rows], z=[r["L"] for r in rows],
    mode="markers",
    marker=dict(size=4, color=[r["violation_probability"] for r in rows],
                colorscale=[[0, "blue"], [1, "red"]], cmin=0, cmax=1,
                colorbar=dict(title="Conditional probability", tickformat=".0%")),
    customdata=[[r["violation_probability"], weight] for r, weight in zip(rows, weights)],
    text=weight_labels,
    hovertemplate="S=%{x}, M=%{y}, L=%{z}<br>Violation given this mix: %{customdata[0]:.2%}"
                  "<br>Probability of this mix: %{text}<extra></extra>",
))
figure.update_layout(
    height=650, margin=dict(l=0, r=0, t=10, b=0), uirevision="camera",
    scene=dict(xaxis_title="Small customers (S)", yaxis_title="Medium customers (M)",
               zaxis_title="Large customers (L)", aspectmode="cube",
               camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
)
st.plotly_chart(figure, use_container_width=True)
overall = overall_probability(rows, type_probabilities)
st.markdown(f"**Overall Probability: {overall['violation_probability']:.2%}**")

export = dict(inputs=dict(inputs if is_saved else exact_inputs(settings), p_type=type_probabilities,
                          p_type_source="viewer_selection"),
              source="exact_binomial",
              results=rows, overall=overall,
              overall_probability=overall['violation_probability'])
st.download_button("Download results JSON", json.dumps(export, indent=2),
                   file_name="streamlit_results.json", mime="application/json")
