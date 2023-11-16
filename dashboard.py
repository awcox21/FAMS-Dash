"""
TODO
----
- Add more documentation/descriptions/headers
- Fix parallel plot callback
- Heatmap might be hard to read for high dimensionality
- Make tables and weightings scrollable instead of all of it
- Prettify tables
- Make all fonts bigger and the same
- Heatmap color(?), hover, and maybe alternative/addition?
"""
from operator import itemgetter, attrgetter
import os
import re
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objs
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from FAMS.utils import dist_median, TOPSIS, pareto_front
from FAMS.model_rankings import Technology, Ranking, Order

load_figure_template('SOLAR')

path = os.path.join(os.path.dirname(__file__), 'data')
h1 = 700

""" Process Files """
technologies, rankings, metrics = list(), dict(), list()
for file in os.listdir(path):
    if re.match(r'Metric \d+.json', file):
        ranking = Ranking.read_json(os.path.join(path, file))
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
                                     options=metrics, multi=False,
                                     clearable=False)
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
                    ], width=8),  # ridgeline
                    # dbc.Col(dash_table.DataTable(
                    #     id='parcoord-select',
                    #     style_data={'whiteSpace': 'normal',
                    #                 'height': 'auto'},
                    #     style_cell={'textAlign': 'center'},
                    #     style_header={'text-align': 'center'},
                    #     page_size=15
                    # ), width=3)
                ])
            ], width=9),
            dbc.Col([
                html.H2('Show statistics by technology'),
                dcc.Dropdown(id='poll-tech-select', value=techs[0],
                             options=techs, multi=False, clearable=False),
                dcc.Graph(id='tech-score-bars'),
                dcc.Graph(id='tech-order-freq')
            ], width=3)  # tech selector, hist, statistics
        ]),  # Select metric and show data
        dbc.Row(dbc.Col(html.Div(id='par-selected'))),
        dbc.Row([
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
                    style={'overflowY': 'auto', 'height': '50%'},
                    width=2),  # sliders
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H3('TOPSIS Sorted Technologies'),
                        html.Button('Set Default', id='set-default'),
                        html.Button('Sort Technologies', id='run-topsis')
                    ], width=1),  # buttons
                    dbc.Col([
                        dash_table.DataTable(
                            id='topsis-results',
                            style_data={'whiteSpace': 'normal',
                                        'height': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={'text-align': 'center'},
                            page_size=15)
                    ], width=5),  # ranked table
                    dbc.Col([
                        html.H3('Weighting Independent Likelihood of Leading'),
                        dcc.Input(1_000, type='number', id='num-sim',
                                  name='num-sim'),
                        html.Button('Simulate', id='run-sim'),
                    ], width=1),  # simulate buttons
                    dbc.Col([
                        dash_table.DataTable(
                            id='sim-results',
                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={'text-align': 'center'},
                            page_size=15)  # TODO: only show non-zero numbers
                    ], width=5)  # simulate table
                ]),  # TOPSIS sliders and table, frequency analysis
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='pareto-metric-1', value=metrics[0],
                                     options=metrics, multi=False,
                                     clearable=False),
                        dcc.Dropdown(id='pareto-metric-2', value=metrics[1],
                                     options=metrics, multi=False,
                                     clearable=False)],
                        width=1),
                    dbc.Col(dcc.Graph(id='pareto-plot'),
                            width=5),  # selectable pareto plot
                    dbc.Col(dcc.Graph(id='sim-heatmap'),
                            width=6)  # simulate likelihood plot
                ])  # Plots (pareto and frequency)
            ])
        ])
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
    """ Score bar chart with cumulative line """
    df = data.sort_values(selected_metric)
    score_fig = make_subplots(shared_yaxes=True)
    score_fig.add_bar(x=df[selected_metric], y=df['Name'], orientation='h',
                      name=f'{selected_metric} Score')
    score_fig.add_scatter(x=df[f'{selected_metric} Cumulative'], y=df['Name'],
                          name=f'{selected_metric} Cumulative')
    score_fig.update_layout(height=h1)
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
        std = np.std(ys)
        text = f'{tech.name}<br>Median: {median:.5f}<br>'
        text += f'Stdev: {std:.5f}<br>CV: {std / np.mean(ys):.5f}'
        ridge_fig.add_trace(go.Scatter(
            x=xs, y=ys + (len(ranking.items) - i) + 0.1, fill='tonexty',
            name=f'{tech.id}', hovertext=text))
        ridge_fig.add_annotation(
            x=min_x, y=len(ranking.items) - i, text=f'{tech.id}',
            showarrow=False, yshift=10, hovertext=f'{tech.name}')
    ridge_fig.update_layout(height=h1)
    vals = [len(ridge_data) - i for i in range(len(ridge_data))]
    ids = [tech.id for _, tech, _, _ in ridge_data]
    ridge_fig.update_layout(yaxis=dict(tickvals=vals, ticktext=ids))
    return score_fig, ridge_fig


@app.callback(
    Output('par-selected', 'children'),
    Input('parallel-plot', 'restyleData'),
    State('parallel-plot', 'figure')
)
def parallel_select(par_restyle, par_plot_data):
    par_data = par_plot_data['data'][0]['dimensions']
    selected = list()
    for column in par_data:
        try:
            low, high = column['constraintrange']
        except KeyError:
            continue
        selected_ = list()
        for tech, value in zip(technologies, column['values']):
            if low <= value <= high:
                selected_.append(tech)
        selected.append(selected_)
    if selected:
        selected = list(set.intersection(*map(set, selected)))
        selected.sort(key=lambda x: x.id)
        message = f"Selected below: {', '.join(_.name for _ in selected)}"
    else:
        message = 'None selected below'
    return html.P(message)


@app.callback(
    Output('tech-score-bars', 'figure'),
    Output('tech-order-freq', 'figure'),
    Input('poll-tech-select', 'value')
)
def polling_tech_figures(selected_tech):
    scores = data.loc[data['Name'] == selected_tech][metrics].iloc[0]
    scores.name = 'Score'
    score_fig = px.bar(scores, height=h1 / 2)

    frequencies = {tech.name: [0 for _ in technologies] for tech in
                   technologies}
    for metric in metrics:
        for order in rankings[metric].orders:
            for i, slot in enumerate(order.order):
                name = slot[0].name
                frequencies[name][i] += 1
    order_fig = px.bar(pd.DataFrame(frequencies), y=selected_tech,
                       height=h1 / 2)
    return score_fig, order_fig


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
    Output('pareto-plot', 'figure'),
    Input('pareto-metric-1', 'value'),
    Input('pareto-metric-2', 'value')
)
def pareto_plot(metric1, metric2,) -> plotly.graph_objs.Figure:
    arr = data[metrics].to_numpy()
    data['Pareto Optimal'] = [_ in pareto_front(arr) for _ in data.index]
    return px.scatter(data, x=metric1, y=metric2, range_x=[0, 1.1 * maximum],
                      range_y=[0, 1.1 * maximum], color='Pareto Optimal',
                      hover_name='Name', height=h1 + 100)


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
        df = df[df['Frequency'] != 0]
        sim_fig = px.imshow(freq, height=h1 + 100)
        return df.to_dict('records'), sim_fig
    else:
        return None, px.imshow(freq)


if __name__ == '__main__':
    app.run_server(debug=True)
