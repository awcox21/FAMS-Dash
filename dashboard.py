from operator import itemgetter
import os
import re
from math import ceil
import json
from datetime import datetime

from matplotlib.cm import viridis as colormap
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import plotly.graph_objs
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dash_table import DataTable, FormatTemplate
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from FAMS.utils import dist_median, TOPSIS, pareto_front
from FAMS.model_rankings import Technology, Ranking, Order

load_figure_template('DARKLY')  # CYBORG, DARKLY, SOLAR

path = os.path.join(os.path.dirname(__file__), 'data')
h1 = 500  # 700
font_size = 15  # 20

""" Process Files """
technologies, rankings, metrics = list(), dict(), list()
metric_defaults = None
for file in os.listdir(path):
    if re.match(r'metric-', file):
        ranking = Ranking.read_json(os.path.join(path, file))
        name, _ = os.path.splitext(file)
        name = name.replace('metric-', str())
        metrics.append(name)
        if not technologies:
            technologies = list(ranking.items)
            tech_index_dict = {i: _ for i, _ in enumerate(technologies)}
            tech_name_dict = {_.name: _ for _ in technologies}
        rankings[name] = ranking
    if all(_ in file.lower() for _ in ('default', 'weight', 'json')):
        with open(os.path.join(path, file), 'r') as f:
            metric_defaults = json.load(f)
if not metric_defaults:
    metric_defaults = {key: 50 for key in metrics}

""" Process Score Data """
data = dict()
data['ID'] = [_.id for _ in technologies]
data['Name'] = [_.name for _ in technologies]
data['Category'] = [_.category for _ in technologies]
categories = sorted(list(set(data['Category'])))
data['Category IDs'] = [categories.index(_) for _ in data['Category']]
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
colorscale = list()
for i, value in enumerate(np.linspace(0.3, 1, len(categories))):
    i /= len(categories) - 1
    color = to_hex(colormap(value))
    colorscale.append([i, color])
parallel_fig = go.Figure(
    data=go.Parcoords(
        line=dict(color=data['Category IDs'],
                  colorscale=colorscale),
        dimensions=dimensions,
        labelangle=30
    )
)
parallel_fig.update_layout(font=dict(color='white', size=font_size))


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
    parplot_legend = list()
    for category, color in zip(categories, colorscale):
        parplot_legend.append(html.P([
            category, ' : ',
            html.Div(style={'height': '25px', 'width': '25px',
                            'background-color': color[-1]})],
            style={'font-size': font_size, 'display': 'flex',
                   'align-items': 'center'}))
    return [
        dbc.Row([
            dbc.Col([
                html.H2('Show scores by attribute'),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='poll-metric-select', value=metrics[0],
                                     options=metrics, multi=False,
                                     clearable=False, style={'color': 'black'})
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='metric-score-bars')
                    ], width=6),  # metric selector and score plot by metric
                    dbc.Col([
                        dcc.Graph(id='metric-ridgeline')
                    ], width=6),  # ridgeline
                ])
            ], width=7),
            dbc.Col([
                html.H2('Show statistics by technology'),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='poll-tech-select', value=techs[0],
                                     options=techs, multi=False,
                                     clearable=False, style={'color': 'black'})
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='tech-score-bars')),
                    dbc.Col(html.Div(id='tech-score-describe'))
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='tech-order-freq')),
                    dbc.Col(html.Div(id='tech-order-describe'))
                ])
            ], width=5)  # tech selector, hist, statistics
        ]),  # Select metric and show data
        dbc.Row(dbc.Col(html.Div(id='par-selected'))),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='parallel-plot', figure=parallel_fig)
            ], width=11),  # Parallel Plot
            dbc.Col(parplot_legend, style={'margin': 'auto'}, width=1)
        ])
    ]


def decision_making():
    sliders = [percentile_slider(metric) for metric in metrics]
    return [
        dbc.Row([
            dbc.Col([html.H3('Set Weightings')] + sliders,
                    style={'overflowY': 'auto', 'height': '50%'},
                    width=3),  # sliders
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H4('TOPSIS Sorted Technologies'),
                        html.Button('Reset equal', id='set-equal',
                                    style={'margin-bottom': '10px'}),
                        html.Button('Set to zero', id='set-zero',
                                    style={'margin-bottom': '10px'}),
                        html.Button('Set to default', id='set-default',
                                    style={'margin-bottom': '15px'}),
                        html.Button('Sort Technologies', id='run-topsis',
                                    className='bg-primary',
                                    style={'font-weight': 'bold',
                                           'font-size': font_size * 1.1,
                                           'margin-bottom': '15px',
                                           'color': 'white'}),
                        html.Button('Save Table',
                                    id='save-topsis',
                                    style={'margin-bottom': '10px'})
                    ], width=1),  # buttons
                    dbc.Col([
                        DataTable(
                            id='topsis-results',
                            columns=[
                                dict(id='rank', name='Rank'),
                                dict(id='ID', name='ID'),
                                dict(id='category', name='Category'),
                                dict(id='name', name='Name')
                            ],
                            style_header={
                                'text-align': 'center',
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white', 'fontWeight': 'bold'
                            },
                            style_data={'whiteSpace': 'normal',
                                        'height': 'auto',
                                        'color': 'white',
                                        'backgroundColor': 'rgb(50, 50, 50)'},
                            style_cell={'textAlign': 'center'},
                            page_size=15)
                    ], width=5),  # ranked table
                    dbc.Col([
                        html.H4('Weighting Independent Likelihood of Leading'),
                        html.Button('Simulate', id='run-sim',
                                    className='bg-primary',
                                    style={'font-weight': 'bold',
                                           'font-size': font_size * 1.25,
                                           'margin-bottom': '10px',
                                           'color': 'white'}
                                    ),
                    ], width=1),  # simulate buttons
                    dbc.Col([
                        DataTable(
                            id='sim-results',
                            columns=[
                                dict(id='rank', name='Rank'),
                                dict(id='ID', name='ID'),
                                dict(id='category', name='Category'),
                                dict(id='name', name='Name'),
                                dict(id='frequency', name='Frequency',
                                     type='numeric',
                                     format=FormatTemplate.percentage(3))],
                            style_header={
                                'text-align': 'center',
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white', 'fontWeight': 'bold'
                            },
                            style_data={'whiteSpace': 'normal',
                                        'height': 'auto',
                                        'color': 'white',
                                        'backgroundColor': 'rgb(50, 50, 50)'},
                            style_cell={'textAlign': 'center'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'name'}, 'width': '50%'},
                                {'if': {'column_id': 'frequency'},
                                 'width': '30%'}
                            ],
                            page_size=15)
                    ], width=5)  # simulate table
                ]),  # TOPSIS sliders and table, frequency analysis
                dbc.Row([
                    dbc.Col([
                        html.H4('Select x-axis:'),
                        dcc.Dropdown(id='pareto-metric-1', value=metrics[0],
                                     options=metrics, multi=False,
                                     clearable=False,
                                     style={'color': 'black'}),
                        html.H4('Select y-axis:'),
                        dcc.Dropdown(id='pareto-metric-2', value=metrics[1],
                                     options=metrics, multi=False,
                                     clearable=False,
                                     style={'color': 'black'})],
                        width=1),
                    dbc.Col(dcc.Graph(id='pareto-plot'),
                            width=5),  # selectable pareto plot
                    dbc.Col(dcc.Graph(id='sim-heatmap'),
                            width=6)  # simulate likelihood plot
                ])  # Plots (pareto and frequency)
            ])
        ])
    ]


def tech_select():
    return [
        html.H2('List of technologies'),
        html.P('Select rows to exclude those technologies from TOPSIS'),
        DataTable(
            data=data[['ID', 'Name', 'Category']].to_dict('records'),
            id='tech-exclude',
            columns=[
                dict(id='ID', name='ID'),
                dict(id='Name', name='Name'),
                dict(id='Category', name='Category')
            ],
            style_header={
                'text-align': 'center',
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white', 'fontWeight': 'bold'
            },
            style_data={'whiteSpace': 'normal',
                        'height': 'auto',
                        'color': 'white',
                        'backgroundColor': 'rgb(50, 50, 50)'},
            style_cell={'textAlign': 'center'},
            row_selectable='multi'
        )
    ]


# def verification():
#     return [
#         html.Div(id='convergence'),
#         dcc.Graph(id='sim-history')
#     ]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Store(id='last-topsis'),
    dcc.Store(id='topsis-archive'),
    dcc.Store(id='sim-data'),
    html.H1('Technology Evaluation Dashboard'),
    dcc.Tabs([
        dcc.Tab(polling_results(), label='Polling Results',
                selected_className='bg-primary',
                selected_style={'color': 'white'},
                style={'color': 'black'}),
        dcc.Tab(decision_making(), label='Decision-Making',
                selected_className='bg-primary',
                selected_style={'color': 'white'},
                style={'color': 'black'}),
        dcc.Tab(tech_select(), label='Technologies',
                selected_className='bg-primary',
                selected_style={'color': 'white'},
                style={'color': 'black'})
        # dcc.Tab(verification(), label='Verification',
        #         selected_className='bg-primary',
        #         selected_style={'color': 'white'},
        #         style={'color': 'black'}),
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
    score_fig.update_layout(title='Technology Scores and Cumulative Score')
    score_fig.update_layout(font=dict(size=font_size))
    score_fig.update_layout(showlegend=False)
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
    title = f'Technology Score Distributions for {selected_metric}'
    ridge_fig = go.Figure(layout={'showlegend': False, 'title': title})
    for i, (median, tech, xs, ys) in enumerate(ridge_data):
        min_x, max_x = min(xs), max(xs)
        ridge_fig.add_trace(go.Scatter(x=[min_x, max_x],
                                       y=np.full(2, len(ranking.items) - i),
                                       mode='lines', line_color='white'))
        std = np.std(ys)
        text = f'{tech.name}<br>Median: {median:.5f}<br>'
        text += f'Stdev: {std:.5f}'  # <br>CV: {std / np.mean(ys):.5f}'
        ridge_fig.add_trace(go.Scatter(
            x=xs, y=ys + (len(ranking.items) - i) + 0.1, fill='tonexty',
            name=f'{tech.id}', hovertext=text))
    ridge_fig.update_layout(height=h1)
    vals = [len(ridge_data) - i for i in range(len(ridge_data))]
    ids = [tech.id for _, tech, _, _ in ridge_data]
    ridge_fig.update_layout(yaxis=dict(tickvals=vals, ticktext=ids))
    ridge_fig.update_layout(font=dict(size=font_size))
    return score_fig, ridge_fig


@app.callback(
    Output('par-selected', 'children'),
    Input('parallel-plot', 'restyleData'),
    State('parallel-plot', 'figure')
)
def parallel_select(_, par_plot_data):
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
    return html.P(message, style={'font-size': font_size})


@app.callback(
    Output('tech-score-bars', 'figure'),
    Output('tech-order-freq', 'figure'),
    Output('tech-score-describe', 'children'),
    Output('tech-order-describe', 'children'),
    Input('poll-tech-select', 'value')
)
def polling_tech_figures(selected_tech):
    scores = data.loc[data['Name'] == selected_tech][metrics].iloc[0]
    scores.name = 'Score'
    score_fig = px.bar(scores, height=h1 / 2,
                       labels={_: str() for _ in metrics})
    score_fig.update_layout(font=dict(size=font_size))
    score_fig.update_layout(showlegend=False)
    score_fig.update_layout(xaxis_title=None)
    score_describe = html.P([
        html.H4(f'Scoring stats for {selected_tech}'),
        html.Ul([
            html.Li(f'Min/Max: {scores.min():.5f} - {scores.max():.5f}',
                    style={'font-size': font_size}),
            html.Li(f'Mean: {scores.mean():.5f}',
                    style={'font-size': font_size}),
            html.Li(f'Median: {scores.median():.5f}',
                    style={'font-size': font_size}),
            html.Li(f'Standard Deviation: {scores.std():.5f}',
                    style={'font-size': font_size})
        ])
    ])

    frequencies = {tech.name: [0 for _ in technologies] for tech in
                   technologies}
    for metric in metrics:
        for order in rankings[metric].orders:
            for i, slot in enumerate(order.order):
                name = slot[0].name
                frequencies[name][i] += 1
    df = pd.DataFrame(frequencies)
    df['Rank'] = df.index + 1
    order_fig = px.bar(df, x='Rank', y=selected_tech, height=h1 / 2)
    order_fig.update_layout(font=dict(size=font_size))
    order_fig.update_layout(xaxis_title='Sampled Ranks')
    order_fig.update_layout(yaxis_title='Num Samples')
    min_i = df[df[selected_tech] == df[selected_tech].min()]['Rank'].iloc[0]
    min_ = f'{df[selected_tech].min()} @ rank {min_i}'
    max_i = df[df[selected_tech] == df[selected_tech].max()]['Rank'].iloc[0]
    max_ = f'{df[selected_tech].max()} @ rank {max_i}'
    df[f'{selected_tech}-rank'] = df[selected_tech] * df['Rank']
    mean = df[f'{selected_tech}-rank'].mean() / 15
    median = df[f'{selected_tech}-rank'].median() / 15
    std = df[f'{selected_tech}-rank'].std() / 15
    # mid = len(technologies) / 2
    order_describe = html.P([
        html.H4(f'Ranking stats for {selected_tech}'),
        html.P(f'Middle at {(len(technologies) - 1) / 2 + 1:.2f}'),
        html.Ul([
            html.Li(f'Min Selected: {min_}',
                    style={'font-size': font_size}),
            html.Li(f'Max Selected: {max_}',
                    style={'font-size': font_size}),
            html.Li(f'Mean: {mean:.3f}',
                    style={'font-size': font_size}),
            html.Li(f'Median: {median:.3f}',
                    style={'font-size': font_size}),
            html.Li(f'Standard Deviation: {std:.5f}',
                    style={'font-size': font_size}),
            # html.Li(f'Coefficient of variation: {std / mean:.5f}')
            # html.Li(f'Distance from middle ({mid}): {mean - mid:.2f}')
        ])
    ])
    return score_fig, order_fig, score_describe, order_describe


@app.callback(
    [Output(f'{_}-value', 'children') for _ in metrics],
    [Input(f'{_}-slider', 'value') for _ in metrics]
)
def slider_values(*sliders):
    return sliders


@app.callback(
    Output('last-topsis', 'data'),
    Input('run-topsis', 'n_clicks'),
    State('tech-exclude', 'selected_rows'),
    [State(f'{_}-slider', 'value') for _ in metrics]
)
def run_topsis(_, exclude, *args):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'run-topsis' in changed:
        if exclude:
            excluded = data['Name'].iloc[exclude]
        else:
            excluded = set()
        args = iter(args)
        sliders = list()
        for _ in metrics:
            sliders.append(next(args))
        sum_ = sum(sliders)
        if not sum_:  # all set to zero, should be same as all equal
            sliders = [50 for _ in metrics]
            sum_ = sum(sliders)
        weights = np.array([_ / sum_ for _ in sliders])
        sorted_data = TOPSIS(data[metrics]).sort(weights)
        ids = [_ + 1 for _ in sorted_data.index if _ not in excluded]
        names = [tech_index_dict[_].name for _ in sorted_data.index
                 if _ not in excluded]
        categories = [tech_index_dict[_].category for _ in sorted_data.index
                      if _ not in excluded]
        table = pd.DataFrame({'rank': range(1, len(names) + 1),
                              'ID': ids,
                              'category': categories,
                              'name': names})
        return table.to_dict()


@app.callback(
    Output('topsis-results', 'data'),
    Input('last-topsis', 'data')
)
def display_topsis(recent):
    return pd.DataFrame.from_dict(recent).to_dict('records')


@app.callback(
    Output('topsis-archive', 'data'),
    Input('save-topsis', 'n_clicks'),
    Input('sim-data', 'data'),
    State('last-topsis', 'data'),
    State('topsis-archive', 'data')
)
def save_topsis(_, from_sim, recent, archive):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'save-topsis' in changed:
        if not recent:
            return archive
        if not archive:
            archive = {'manual': [recent]}
        else:
            if recent not in archive['manual']:
                archive['manual'].append(recent)
        today = datetime.today().strftime('%Y-%m-%d')
        with open(f'{today} TOPSIS Results.json', 'w') as f:
            json.dump(archive, f)
        return archive
    elif from_sim:
        if not archive:
            archive = {'sim': from_sim}
        else:
            archive['sim'] = from_sim
        today = datetime.today().strftime('%Y-%m-%d')
        with open(f'{today} TOPSIS Results.json', 'w') as f:
            json.dump(archive, f)
        return archive


@app.callback(
    [Output(f'{_}-slider', 'value') for _ in metrics],
    Input('set-equal', 'n_clicks'),
    Input('set-zero', 'n_clicks'),
    Input('set-default', 'n_clicks'),
    [State(f'{_}-slider', 'value') for _ in metrics]
)
def reset_sliders(_, __, ___, *args):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'set-equal' in changed:
        values = [50 for _ in metrics]
    elif 'set-zero' in changed:
        values = [0 for _ in metrics]
    elif 'set-default' in changed:
        values = [metric_defaults[key] for key in metrics]
    else:
        return args
    return values


@app.callback(
    Output('pareto-plot', 'figure'),
    Input('pareto-metric-1', 'value'),
    Input('pareto-metric-2', 'value')
)
def pareto_plot(metric1, metric2) -> plotly.graph_objs.Figure:
    arr = data[metrics].to_numpy()
    arr *= -1
    data['Pareto Optimal'] = [_ in pareto_front(arr) for _ in data.index]
    pareto_fig = px.scatter(
        data, x=metric1, y=metric2, range_x=[0, 1.1 * maximum],
        range_y=[0, 1.1 * maximum], color='Pareto Optimal', hover_name='Name',
        height=h1 + 100)
    selected = data[[metric1, metric2]].to_numpy()
    selected *= -1
    pareto = pareto_front(selected)
    points = data[[metric1, metric2]].iloc[pareto]
    pareto_fig.add_trace(go.Scatter(x=points[metric1], y=points[metric2],
                                    mode='lines', name='2D Pareto Front'))
    pareto_fig.update_layout(font=dict(size=font_size))
    return pareto_fig


@app.callback(
    Output('sim-results', 'data'),
    Output('sim-heatmap', 'figure'),
    # Output('sim-history', 'figure'),
    # Output('convergence', 'children'),
    Output('sim-data', 'data'),
    Input('run-sim', 'n_clicks')
)
def run_sim(_):
    changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    freq = np.zeros((len(technologies), len(technologies)))
    if 'run-sim' in changed:
        topsis = TOPSIS(data[metrics])
        all_weights = None
        nums = list()
        stds = list()
        for _ in range(10_000):
            weights = np.random.rand(len(technologies))
            if all_weights is None:
                all_weights = weights
            else:
                all_weights = np.vstack([all_weights, weights])
            for i, j in zip(data.index, np.argsort(topsis(weights))[::-1]):
                freq[i, j] += 1
            if _ and not (_ + 1) % 100:
                nums.append(_)
                stds.append(all_weights.mean(axis=0).std())
                if len(stds) > 10:
                    running = np.convolve(stds, np.ones(5), 'valid')
                    running /= 5
                    if running[-1] < 5e-3:
                        message = f'std converged after {_ + 1}'
                        break
                    elif abs(running[-1] - running[-2]) < 1e-6:
                        message = f'diff converged after {_ + 1}'
                        break
        else:
            message = 'finished 10,000'
        stds = [(i, j) for i, j in zip(nums, stds)]
        freq /= nums[-1] + 1
        frequency = freq[0, :]
        df = pd.DataFrame({'name': [_.name for _ in technologies],
                           'ID': [_.id for _ in technologies],
                           'category': [_.category for _ in technologies],
                           'frequency': frequency})
        df = df.sort_values('frequency', ascending=False)
        df = df[df['frequency'] != 0]
        num, value = 1, df.iloc[0].frequency
        nums = [num]
        for i, (_, row) in enumerate(df.iterrows()):
            if not i:
                continue
            if row.frequency != value:
                num += 1
                value = row.frequency
            nums.append(num)
        df['rank'] = nums
        sim_fig = px.imshow(freq, height=h1 + 100,
                            color_continuous_scale='viridis',
                            labels={'x': 'Technologies', 'y': 'Ranking'})
        sim_fig.update_layout(font=dict(size=font_size))
        sim_history = px.line(pd.DataFrame(stds, columns=('num', 'std')),
                              x='num', y='std')
        return (df.to_dict('records'),
                sim_fig,
                # sim_history,
                # html.P(message),
                df.to_dict())
    else:
        sim_fig = px.imshow(freq, height=h1 + 100,
                            color_continuous_scale='viridis')
        sim_fig.update_layout(font=dict(size=font_size))
        return None, sim_fig, dict()


if __name__ == '__main__':
    app.run_server(debug=True)
