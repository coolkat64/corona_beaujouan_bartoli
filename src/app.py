import datetime
import os
import yaml
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import decimal



# Lecture du fichier d'environnement
ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 10)]
              )

countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]

def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)

def SIR(beta, gamma, y):

    N = sum(y[['susceptible', 'infected', 'recovered']])

    dS = (beta * y['susceptible'] * y['infected'])/N
    dI = (beta * y['susceptible'] + y['infected'])/N - gamma * y['infected']
    dR = gamma * y['infected']

    S = y['susceptible'] + dS
    I = y['infected'] + dI 
    R = y['recovered'] + dR

    return [S, I, R]

def SIRT(beta,gamma,nbrjour,pop,nbr_infecte_initial):
    beta, gamma = [beta,gamma]
    solution_korea = solve_ivp(SIR, [0,nbrjour], [pop,nbr_infecte_initial, 0], t_eval=np.arange(0,nbrjour, 1))
    return(solution_korea)

def get_country(self, country):
    return (epidemie_df[epidemie_df['Country/Region'] == country]
        .groupby(['Country/Region', 'day'])
        .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
        .reset_index()
        )
pd.DataFrame.get_country = get_country


app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ]),   
        ]),
        dcc.Tab(label='Map', children=[
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                #marks={i:str(date) for i, date in enumerate(epidemie_df['day'].unique())}
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        dcc.Tab(label='Model', children=[  
            html.Div([
                html.H4(['Beta (mean recovery rate/day)'], style={'textAlign': 'left'}),
                dcc.Input(id="beta", type="number", placeholder="Beta", value=0.3, step=0.01, max=1)
            ]),
            html.Div([
                html.H4(['Gamma'], style={'textAlign': 'left'}),
                dcc.Input(id="gamma", type="number", placeholder="Gamma", value=0.1, step=0.01, max=1)
            ]),
            html.Div([
                html.H4(['Population'], style={'textAlign': 'left'}),
                dcc.Input(id="N", type="number", placeholder="Population", value=500000, step=1000)
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable2',
                    options=[
                        {'label': 'Infected', 'value': 'Infected'},
                        {'label': 'Recovered', 'value': 'Recovered'},
                        {'label': 'Susceptible', 'value': 'Susceptible'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country3',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Graph(id='graph2')
            ])
        ]),  
    ]),
])

@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)
def update_graph(country, country2, variable):
    print(country)
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index()
                    )

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )            
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day', 'value'),
    ]
)
def update_map(map_day):
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Confirmed': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )
    print(map_day)
    print(day)
    print(map_df.head())
    return {
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed'] / 1_000, 5)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True),
        )
    }

@app.callback(
    Output('graph2', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'), 
        Input('N', 'value'),
        Input('country3', 'value'), 
        Input('variable2', 'value')  
    ]
)
def update_model(beta, gamma, N, country3, variable2):

    if country3 != None:
        df = epidemie_df.get_country(country3).sort_values(by='day', ascending=False)
    else:
        df = epidemie_df.groupby(['Country/Region', 'day']).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index().sort_values(by='day', ascending=False)

    '''Here include predictions'''
    sim = pd.DataFrame(None, columns = ['day', 'susceptible' ,'infected', 'recovered'])
    I = df.Confirmed.loc[1]
    S = N - df.Deaths.loc[1] - df.Recovered.loc[1]
    R = df.Recovered.loc[1]
    day0 = df.day.loc[1]

    # ([-beta*S*I, beta*S*I-gamma*I, gamma*I])

    temp = np.array([day0, S, I, R]).reshape(4)
    sim.loc[0] = temp

    for i in range(100):
        temp = [sim.day.loc[i] + datetime.timedelta(days=1)] + SIR(beta, gamma, sim.loc[i])
        sim.loc[i+1] = temp

    return {
        'data': ([
            dict(
                x=sim['day'],
                y=sim['infected'],
                type='line',
                name='Infected'
            )
        ] if variable2 == 'Infected' else []) + ([
            dict(
                x=sim['day'],
                y=sim['recovered'],
                type='line',
                name='Deaths'
            )            
        ] if variable2 == 'Recovered' else [])
        + ([
            dict(
                x=sim['day'],
                y=sim['susceptible'],
                type='line',
                name='Recovered'
            )
        ] if variable2 == 'Susceptible' else [])
        
    }

if __name__ == '__main__':
    app.run_server(debug=True)