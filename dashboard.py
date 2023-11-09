from operator import itemgetter
import os
import re
from math import ceil

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from FAMS.utils import dist_median, TOPSIS
from FAMS.model_rankings import Technology, Ranking, Order

load_figure_template('SOLAR')

""" Process Files """
technologies, rankings, metrics = list(), dict(), list()
for file in os.listdir(os.environ['USERPROFILE']):
    if re.match(f'Metric \d+.json', file):
        ranking = Ranking.read_json(os.path.join(os.environ['USERPROFILE'],
                                                 file))
        name, _ = os.path.splitext(file)
        metrics.append(name)
        if not technologies:
            technologies = list(ranking.items)
            tech_id_dict = {_.id: _ for _ in technologies}
            tech_name_dict = {_.name: _ for _ in technologies}
        rankings[name] = ranking

""" Process Score Data """
data = dict()
data['ID'] = [_.id for _ in technologies]
data['Name'] = [_.name for _ in technologies]
data = pd.DataFrame(data)
for metric in metrics:
    data[metric] = rankings[metric].prob_level().values()
    data = data.sort_values(metric)
    data[f'{metric} Cumulative'] = data.loc[::-1, metric].cumsum()[::-1]
    data = data.sort_values('ID')
maximum = data[metrics].max().max()

""" Initialize Parallel Plot """
ticks = [_ / 10 for _ in range(ceil(maximum * 10))]
if maximum > ticks[-1]:
    ticks.append(ticks[-1] + 0.05)
dimensions = [
    dict(range=[0, maximum], tickvals=ticks, label=metric, values=data[metric])
    for metric in metrics
]
parallel_fig = go.Figure(
    data=go.Parcoords(
        line=dict(color=data['ID']),
        dimensions=dimensions,
        # unselected=dict(line=dict(opacity=0.3))
    )
)
parallel_fig.update_layout(font=dict(color='yellow', size=20))
parallel_fig.update_layout(paper_bgcolor='gray')


def percentile_slider(id_):
    default = 50
    return dbc.Row([
        dbc.Label(id_, style={'font-weight': 'bold'}, width=2),
        dbc.Col(dcc.Slider(
            id=f'{id_}-slider', min=0, max=100, step=5, value=default,
            marks={0: '0', 100: '100'}), width=4),
        dbc.Col(html.P(id=f'{id_}-value'))
    ])


def polling_results():
    techs = [_.name for _ in technologies]
    return [
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H2('Show scores by metric'),
                        dcc.Dropdown(id='poll-metric-select', value=metrics[0],
                                     options=metrics, multi=False)
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3('Technology Scores and Cumulative Score'),
                        dcc.Graph(id='metric-score-bars')
                    ], width=4),  # metric selector and score plot by metric
                    dbc.Col([
                        html.H2('Technology Score Distributions by Metric'),
                        dcc.Graph(id='metric-ridgeline')
                    ], width=8)  # ridgeline
                ])
            ], width=9),
            dbc.Col([
                html.H2('Show statistics by technology'),
                dcc.Dropdown(id='poll-tech-select', value=techs[0],
                             options=techs, multi=False),
                dcc.Graph(id='tech-score-bars'),
                dcc.Graph(id='tech-order-freq')
            ], width=3)  # tech selector, hist, statistics
        ]),  # Select metric and show data
        dbc.Row([  # TODO: add table of selected technologies?
            dbc.Col([
                dcc.Graph(id='parallel-plot', figure=parallel_fig)
            ], width=12)
        ])  # Parallel Plot
    ]


def decision_making():
    sliders = [percentile_slider(metric) for metric in metrics]
    return [
        dbc.Row([
            dbc.Col([html.H3('Set Weightings')] + sliders,
                    style={'overflowY': 'auto'}),  # sliders
            dbc.Col([
                html.H3('TOPSIS Sorted Technologies'),
                html.Button('Set Default', id='set-default'),
                html.Button('Sort Technologies', id='run-topsis')
            ]),  # buttons
            dbc.Col([
                dash_table.DataTable(
                    id='topsis-results',
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell={'textAlign': 'center'},
                    style_header={'text-align': 'center'})
            ]),  # ranked table
            dbc.Col([
                html.H3('Weighting Independent Likelihood of Leading'),
                dcc.Input(1_000, type='number', id='num-sim', name='num-sim'),
                html.Button('Simulate', id='run-sim'),
            ]),  # simulate buttons
            dbc.Col([
                dash_table.DataTable(
                    id='sim-results',
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell={'textAlign': 'center'},
                    style_header={'text-align': 'center'})
            ])  # simulate table
        ]),  # TOPSIS sliders and table, frequency analysis
        dbc.Row([
            dbc.Col(),  # selectable pareto plot
            dbc.Col(dcc.Graph(id='sim-heatmap'))  # simulate likelihood plot
        ])  # Plots (pareto and frequency)
    ]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

app.layout = html.Div([
    html.H1('Technology Evaluation Dashboard'),
    dcc.Tabs([
        dcc.Tab(polling_results(), label='Polling Results'),
        dcc.Tab(decision_making(), label='Decision-Making')
    ])
])


@app.callback(
    Output('metric-score-bars', 'figure'),
    Output('metric-ridgeline', 'figure'),
    Input('poll-metric-select', 'value')
)
def polling_metric_figures(selected_metric):
    if not selected_metric:
        return {}, {}
    """ Score bar chart with cumulative line """
    df = data.sort_values(selected_metric)
    score_fig = make_subplots(shared_yaxes=True)
    score_fig.add_bar(x=df[selected_metric], y=df['Name'], orientation='h',
                      name=f'{selected_metric} Score')
    score_fig.add_scatter(x=df[f'{selected_metric} Cumulative'], y=df['Name'],
                          name=f'{selected_metric} Cumulative')
    score_fig.update_layout(height=800)
    """ Ridgeline plot of technology Kernel Densities """
    ranking = rankings[selected_metric]
    ridge_data = list()
    for tech in technologies:
        sample = ranking.score_dists[tech.id]
        sample = sample[sample > 1e-3]
        xs = np.array([_ for _ in sample.index])
        ys = np.array([_ for _ in sample])
        median, _ = dist_median(xs, ys)
        ridge_data.append((median, tech, xs, ys))
    ridge_data.sort(key=itemgetter(0), reverse=True)
    ridge_fig = go.Figure()
    for i, (median, tech, xs, ys) in enumerate(ridge_data):
        min_x, max_x = min(xs), max(xs)
        ridge_fig.add_trace(go.Scatter(x=[min_x, max_x],
                                 y=np.full(2, len(ranking.items) - i),
                                 mode='lines', line_color='white'))
        text = f'{tech.name}<br>Median: {median:.5f}<br>Stdev: {np.std(ys):.5f}'
        text += f'<br>CV: {np.std(ys) / np.mean(ys):.5f}'
        ridge_fig.add_trace(go.Scatter(x=xs, y=ys + (len(ranking.items) - i) + 0.1,
                                 fill='tonexty', name=f'{tech.id}',
                                 hovertext=text))
        ridge_fig.add_annotation(x=min_x, y=len(ranking.items) - i,
                           text=f'{tech.id}',
                           showarrow=False, yshift=10,
                           hovertext=f'{tech.name}')
    ridge_fig.update_layout(height=800)
    vals = [len(ridge_data) - i for i in range(len(ridge_data))]
    ids = [tech.id for _, tech, _, _ in ridge_data]
    ridge_fig.update_layout(yaxis=dict(tickvals=vals, ticktext=ids))
    return score_fig, ridge_fig


@app.callback(
    Output('tech-score-bars', 'figure'),
    Output('tech-order-freq', 'figure'),
    Output('parallel-plot', 'figure'),
    Input('poll-tech-select', 'value'),
    [State('parallel-plot', 'figure')]
)
def polling_tech_figures(selected_tech, pfig):
    if not selected_tech:
        return {}, {}, pfig
    scores = data.loc[data['Name'] == selected_tech][metrics].iloc[0]
    scores.name = 'Score'
    score_fig = px.bar(scores)

    frequencies = {tech.name: [0 for _ in technologies] for tech in
                   technologies}
    for metric in metrics:
        for order in rankings[metric].orders:
            for i, slot in enumerate(order.order):
                name = slot[0].name
                frequencies[name][i] += 1
    order_fig = px.bar(pd.DataFrame(frequencies), y=selected_tech)

    row = data[metrics].loc[tech_name_dict[selected_tech].id].to_list()
    for i, v in enumerate(pfig.get('data')[0].get('dimensions')):
        v.update({'constraintrange': [row[i] - row[i] / 100000, row[i]]})
    return score_fig, order_fig, pfig


@app.callback(
    [Output(f'{_}-value', 'children') for _ in metrics],
    [Input(f'{_}-slider', 'value') for _ in metrics]
)
def slider_values(*sliders):
    return sliders


@app.callback(
    Output('topsis-results', 'data'),
    Input('run-topsis', 'n_clicks'),
    [State(f'{_}-slider', 'value') for _ in metrics]
)
def run_topsis(_, *args):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'run-topsis' in changed:
        args = iter(args)
        sliders = list()
        for _ in metrics:
            sliders.append(next(args))
        sum_ = sum(sliders)
        weights = np.array([_ / sum_ for _ in sliders])
        sorted_data = TOPSIS(data[metrics]).sort(weights)
        names = [tech_id_dict[_].name for _ in sorted_data.index]
        table = pd.DataFrame({'Rank': range(1, len(names) + 1),
                              'ID': sorted_data.index,
                              'Name': names})
        return table.to_dict('records')


@app.callback(
    Output('sim-results', 'data'),
    Output('sim-heatmap', 'figure'),
    Input('run-sim', 'n_clicks'),
    State('num-sim', 'value')
)
def run_sim(_, num):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    freq = np.zeros((len(technologies), len(technologies)))
    if 'run-sim' in changed:
        topsis = TOPSIS(data[metrics])
        for _ in range(num):
            weights = np.random.rand(len(technologies))
            for i, j in zip(data.index, np.argsort(topsis(weights))[::-1]):
                freq[i, j] += 1
        frequency = freq[0, :] / num * 100
        df = pd.DataFrame({'Name': [_.name for _ in technologies],
                           'Frequency': frequency})
        df = df.sort_values('Frequency', ascending=False)
        sim_fig = px.imshow(freq)
        return df.to_dict('records'), sim_fig
    else:
        return None, px.imshow(freq)


if __name__ == '__main__':
    app.run_server(debug=True)
