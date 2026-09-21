"""Run with: python3 -m streamlit run apps/streamlit_app.py"""

import json
from pathlib import Path
import sys
import threading
from time import perf_counter

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gridstress.models.interactive import evaluate
from gridstress.results import load_results, viewer_settings, overall_probability


st.set_page_config(page_title="Load violation probability", layout="wide")


@st.cache_resource
def simulation_lock():
    # The existing engine writes a shared active-count disk cache.
    return threading.Lock()


@st.cache_data(max_entries=1)
def saved_results():
    return load_results()


@st.cache_data(max_entries=16, show_spinner=False)
def simulate(settings):
    with simulation_lock():
        return evaluate(settings)


saved = saved_results()
inputs = saved["inputs"]
defaults = viewer_settings(saved)

with st.sidebar:
    st.subheader("Model inputs")
    settings = {}
    for key, label, low, high, step in (
        ("small", "Small customer load", 0, 500, 1),
        ("medium", "Medium customer load", 0, 500, 1),
        ("large", "Large customer load", 0, 500, 1),
        ("probability", "On probability (%)", 0, 100, 1),
        ("customers", "Total customers", 0, 100, 1),
        ("threshold", "Violation threshold", 0, 50000, 1),
        ("samples", "Samples per combination per seed", 1000, 100000, 1000),
    ):
        settings[key] = st.slider(label, low, high, defaults[key], step, key=key)

started = perf_counter()
is_saved = settings == defaults
with st.spinner("Updating simulation…"):
    rows = saved["results"] if is_saved else simulate(settings)
calculation_seconds = perf_counter() - started

chart_started = perf_counter()
figure = go.Figure(go.Scatter3d(
    x=[r["S"] for r in rows], y=[r["M"] for r in rows], z=[r["L"] for r in rows],
    mode="markers",
    marker=dict(size=4, color=[r["violation_probability"] for r in rows],
                colorscale=[[0, "blue"], [1, "red"]], cmin=0, cmax=1,
                colorbar=dict(title="Probability", tickformat=".0%")),
    customdata=[r["violation_probability"] for r in rows],
    hovertemplate="S=%{x}, M=%{y}, L=%{z}<br>Probability: %{customdata:.2%}<extra></extra>",
))
figure.update_layout(
    height=650, margin=dict(l=0, r=0, t=10, b=0), uirevision="camera",
    scene=dict(xaxis_title="Small customers (S)", yaxis_title="Medium customers (M)",
               zaxis_title="Large customers (L)", aspectmode="cube",
               camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
)
st.plotly_chart(figure, use_container_width=True)
chart_seconds = perf_counter() - chart_started
overall = overall_probability(rows)
st.markdown(f"**Overall Probability: {overall['violation_probability']:.2%}**")

with st.expander("Run details and timing"):
    st.write("Saved five-seed batch" if is_saved and inputs.get("random_seeds")
             else "Saved batch" if is_saved else "Interactive simulation · seed 42")
    st.write(f"Results retrieval / calculation: {calculation_seconds:.3f} s")
    st.write(f"Chart preparation and serialization: {chart_seconds:.3f} s")
    st.caption("Server-side timings exclude browser rendering, network latency and app startup. "
               "Repeated settings may use cached results. Compositions are equally weighted.")
    st.write(f"{overall['violation_count']:,} violating trials / {overall['iterations']:,} total trials")

export = dict(inputs=inputs if is_saved else dict(settings, random_seed=42),
              source="saved_batch" if is_saved else "interactive_simulation",
              results=rows, overall_probability=overall['violation_probability'])
st.download_button("Download results JSON", json.dumps(export, indent=2),
                   file_name="streamlit_results.json", mime="application/json")
