"""Run: python3 -m streamlit run apps/multi_interval_app.py --server.port 8502"""

import json
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gridstress.config import LOAD_CURRENT, NUM_CUSTOMERS, THRESHOLD, TYPE_PROBABILITIES
from gridstress.models.multi_interval.all_on import evaluate, summarize
from gridstress.results import composition_weights, split_viewer_settings


st.set_page_config(page_title="Multi-interval All-On allocation", layout="wide")
st.markdown('<style>[data-testid="stForm"] [data-testid="InputInstructions"] '
            '{display:none;}</style>', unsafe_allow_html=True)


@st.cache_data(max_entries=16, show_spinner=False)
def calculate(customers, currents, threshold, intervals, random_seed):
    return evaluate(total_customers=customers, currents=currents, threshold=threshold,
                    intervals=intervals, random_seed=random_seed)


def redraw_random():
    st.session_state['allocation_seed'] += 1


def percent(value):
    if value == 0:
        return '0%'
    return f'{100*value:.3e}%' if value < 1e-5 else f'{100*value:.4f}%'


defaults = dict(small=LOAD_CURRENT['S'], medium=LOAD_CURRENT['M'], large=LOAD_CURRENT['L'],
                customers=NUM_CUSTOMERS, threshold=THRESHOLD)
mix_defaults = {f'type_{name}': round(TYPE_PROBABILITIES[j]*100, 1)
                for name, j in zip(('small', 'medium', 'large'), 'SML')}
remainder_key = max(mix_defaults, key=mix_defaults.get)
mix_defaults[remainder_key] = round(mix_defaults[remainder_key]+100-sum(mix_defaults.values()), 1)
for key, value in {**defaults, **mix_defaults}.items():
    if key not in st.session_state:
        st.session_state[key] = value
    if key in mix_defaults:
        st.session_state[key] = round(float(st.session_state[key]), 1)

errors = []
if 'allocation_seed' not in st.session_state:
    st.session_state['allocation_seed'] = 42
intervals = st.radio('Number of intervals', [2, 3], index=0, horizontal=True)
with st.sidebar.form('allocation_inputs'):
    st.subheader('Model inputs')
    settings = {}
    for key, label, low, high in (
        ('small', 'Small customer current (A)', 0, 500),
        ('medium', 'Medium customer current (A)', 0, 500),
        ('large', 'Large customer current (A)', 0, 500),
        ('customers', 'Total customers', 0, 100),
        ('threshold', 'Violation threshold (A)', 0, 50000),
    ):
        settings[key] = st.number_input(label, step=1, format='%d', key=key)
        if not low <= settings[key] <= high:
            st.error(f'{label} must be between {low:,} and {high:,}.')
            errors.append(key)
    st.subheader('Customer type probabilities')
    mix = {}
    for key, label in (('type_small', 'Small type (%)'), ('type_medium', 'Medium type (%)'),
                       ('type_large', 'Large type (%)')):
        mix[key] = st.number_input(label, step=.1, format='%.1f', key=key)
    st.form_submit_button('Update visualization')

if errors:
    st.stop()
currents = dict(zip('SML', (settings['small'], settings['medium'], settings['large'])))
try:
    _, type_probabilities = split_viewer_settings(mix)
    with st.spinner('Finding optimal allocations…'):
        rows = calculate(settings['customers'], currents, settings['threshold'], intervals,
                         st.session_state['allocation_seed'])
except ValueError as error:
    st.error(str(error))
    st.stop()
weights = composition_weights(rows, type_probabilities)

st.subheader(f'{intervals}-interval allocation · All-On')

counts = {category: sum(r['category'] == category for r in rows)
          for category in ('already_safe', 'rescued', 'still_violating')}


def interval_figure(current_field, title):
    figure = go.Figure()
    for violates, label, colour in ((False, 'Within limit', '#3465a4'),
                                    (True, 'Violating', '#cf4446')):
        selected = [(r, w) for r, w in zip(rows, weights)
                    if (r[current_field] > settings['threshold']) == violates]
        figure.add_trace(go.Scatter3d(
            x=[r['S'] for r, _ in selected], y=[r['M'] for r, _ in selected],
            z=[r['L'] for r, _ in selected], mode='markers',
            name=f'{label} ({len(selected)})', marker=dict(size=4, color=colour),
            customdata=[[r[current_field], percent(w)] for r, w in selected],
            text=[allocation_text(r) for r, _ in selected],
            hovertemplate='Composition: S=%{x}, M=%{y}, L=%{z}'
                          '<br>Current in this interval: %{customdata[0]} A'
                          + ('' if current_field == 'shared_current' else
                             '<br>%{text}')
                          + '<br>Composition probability: %{customdata[1]}<extra>%{fullData.name}</extra>',
        ))
    n = max(1, settings['customers'])
    axis_range = [-.03*n, 1.03*n]
    figure.update_layout(
        title=dict(text=title, x=.02), height=560, margin=dict(l=0, r=0, t=70, b=0),
        uirevision=current_field,
        legend=dict(orientation='h', x=0, y=1.04),
        scene=dict(xaxis=dict(title='Small customers (S)', range=axis_range),
                   yaxis=dict(title='Medium customers (M)', range=axis_range),
                   zaxis=dict(title='Large customers (L)', range=axis_range),
                   aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
    )
    return figure


def allocation_text(row):
    return '<br>'.join(f'Interval {t}: {a:g} A · M={m}, L={l}'
                      for t, (m, l, a) in enumerate(zip(row['medium_allocation'],
                                                       row['large_allocation'], row['interval_currents']), 1))


comparison_tab, intervals_tab, random_tab = st.tabs(['Allocation comparison', 'Interval graphs', 'Optimal vs random'])
with comparison_tab:
    figure = go.Figure()
    for category, label, colour in (
        ('already_safe', 'Already safe', '#3465a4'),
        ('rescued', 'Made safe by allocation', '#26966f'),
        ('still_violating', 'Still violating', '#cf4446'),
    ):
        selected = [(r, w) for r, w in zip(rows, weights) if r['category'] == category]
        figure.add_trace(go.Scatter3d(
            x=[r['S'] for r, _ in selected], y=[r['M'] for r, _ in selected],
            z=[r['L'] for r, _ in selected], mode='markers',
            name=f'{label} ({len(selected)})', marker=dict(size=4, color=colour),
            customdata=[[r['shared_current'], percent(w)] for r, w in selected],
            text=[allocation_text(r) for r, _ in selected],
            hovertemplate='S=%{x}, M=%{y}, L=%{z}'
                          '<br>Shared interval: %{customdata[0]} A'
                          '<br>%{text}'
                          '<br>Composition probability: %{customdata[1]}<extra>%{fullData.name}</extra>',
        ))
    figure.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0),
                         uirevision='allocation-camera',
                         legend=dict(orientation='h', x=0, y=1.08),
                         scene=dict(xaxis_title='Small customers (S)', yaxis_title='Medium customers (M)',
                                    zaxis_title='Large customers (L)', aspectmode='cube',
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))))
    st.plotly_chart(figure, use_container_width=True, key='comparison_graph')

with intervals_tab:
    st.plotly_chart(interval_figure('shared_current', 'One shared interval'),
                    use_container_width=True, key='shared_graph')
    if intervals > 1:
        for t, column in enumerate(st.columns(intervals), 1):
            with column:
                st.plotly_chart(interval_figure(f'current_interval_{t}', f'Allocated interval {t}'),
                                use_container_width=True, key=f'interval_{t}_graph')

summary = summarize(rows, type_probabilities)
with comparison_tab:
    st.metric('Total points reversed', counts['rescued'])
    st.dataframe([
        {'Outcome': label, 'Compositions': counts[key],
         'Population probability': percent(summary['probability_mass'][key])}
        for key, label in (('already_safe', 'Already safe'), ('rescued', 'Made safe by allocation'),
                           ('still_violating', 'Still violating'))
    ], hide_index=True, width='stretch')

    st.subheader('Inspect one composition')
    composition_labels = [f"S={r['S']}, M={r['M']}, L={r['L']}" for r in rows]
    selected_label = st.selectbox('Customer counts (S, M, L)', composition_labels,
                                  index=max(range(len(weights)), key=weights.__getitem__),
                                  label_visibility='collapsed')
    selected_index = composition_labels.index(selected_label)
    row = rows[selected_index]
    comparison = go.Figure(go.Bar(
        x=['Shared interval']+[f'Allocated interval {t}' for t in range(1, intervals+1)],
        y=[row['shared_current']]+row['interval_currents'],
        text=[f'{value:,} A' for value in [row['shared_current']]+row['interval_currents']],
        textposition='auto', marker_color=['#3465a4']+['#26966f']*intervals))
    comparison.add_hline(y=settings['threshold'], line_dash='dash', line_color='#cf4446',
                         annotation_text=f"Limit: {settings['threshold']:,} A")
    comparison.update_layout(height=330, yaxis_title='Current (A)', margin=dict(t=30, b=20))
    st.plotly_chart(comparison, use_container_width=True, key='composition_bars')


with random_tab:
    st.button('New random assignment', on_click=redraw_random)
    for method, column in zip(('optimal', 'random'), st.columns(2)):
        with column:
            outcomes = [int(r['optimal_violation']) if method == 'optimal'
                        else r['random_violation'] for r in rows]
            figure = go.Figure(go.Scatter3d(
                x=[r['S'] for r in rows], y=[r['M'] for r in rows], z=[r['L'] for r in rows],
                mode='markers', marker=dict(size=4, color=outcomes, cmin=0, cmax=1,
                    colorscale=[[0, '#3465a4'], [1, '#cf4446']],
                    showscale=False),
                customdata=outcomes,
                text=[('<br>'.join(f'Interval {t}: {a:g} A · M={m}, L={l}'
                       for t, (m, l, a) in enumerate(zip(r['random_medium_allocation'],
                        r['random_large_allocation'], r['random_interval_currents']), 1))
                       if method == 'random' else allocation_text(r)) for r in rows],
                hovertemplate='S=%{x}, M=%{y}, L=%{z}<br>Violation: %{customdata}<br>%{text}<extra></extra>'))
            figure.update_layout(title=f'{method.title()} allocation', height=560,
                margin=dict(l=0, r=0, t=45, b=0), uirevision=f'{method}-camera',
                scene=dict(xaxis_title='Small customers (S)', yaxis_title='Medium customers (M)',
                           zaxis_title='Large customers (L)', aspectmode='cube',
                           camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))))
            st.plotly_chart(figure, use_container_width=True, key=f'{method}_risk_graph')
    st.dataframe([
        {'Allocation': method.title(),
         'Violating compositions': (counts['still_violating'] if method == 'optimal'
                                    else summary['random_violating_compositions']),
         'Total points reversed': (counts['rescued'] if method == 'optimal'
                                   else summary['random_points_reversed'])}
        for method in ('optimal', 'random')
    ], hide_index=True, width='stretch')

export = dict(inputs=dict(method='exact_all_on_allocation', intervals=intervals,
                          random_allocation='single_independent_uniform_assignment',
                          random_seed=st.session_state['allocation_seed'], num_customers=settings['customers'],
                          load_current=currents, threshold=settings['threshold'], p_type=type_probabilities),
              summary=dict(summary, counts=counts), results=rows)
st.download_button('Download results JSON', json.dumps(export, indent=2),
                   file_name='multi_interval_results.json', mime='application/json')
