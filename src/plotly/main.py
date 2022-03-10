import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import pandas as pd
from src.project_config import DATA_DIR
import os

TITLE = "Recommended Frequencies: a recommendation system for playlists"
SUBTITLE = "This app allows users to identify songs in their library that may fit well into a selected playlist"


def generate_table(dataframe):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) if col != "PreviewURL"
            else html.Audio(id=f"audio_{i}", src=dataframe.iloc[i][col], controls=True, autoPlay=True)
            for col in dataframe.columns
        ]) for i in range(dataframe.shape[0])]
    )


df = pd.read_pickle(os.path.join(DATA_DIR, "test.pickle"))

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


similarity_controls = dbc.Card(
    [
        html.H4("Settings"),
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": "col", "value": "col"}
                    ],
                    value="sepal length (cm)",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": "col", "value": "col"}
                    ],
                    value="sepal width (cm)",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
)

selection_part = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("Examples of songs in playlist «80's Retro»"), width=6),
        dbc.Col(html.H4("Song attributes for playlist «80's Retro»"), width=6)
    ]),
    dbc.Row([
        dbc.Col(generate_table(df[["PreviewURL"]]), width=2),
        dbc.Col(generate_table(df[["SongName", "Artist"]]), width=4),
        dbc.Col(html.H4("Image placeholder"), width=6)
    ])

])

suggestion_part = []

app.layout = dbc.Container(
    [
        html.H1(TITLE),
        html.Hr(),
        html.H4(SUBTITLE),
        dbc.Row(
            selection_part,
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(similarity_controls, width=4)
            ]
        )
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True, port=9000)
