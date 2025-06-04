from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from recommended_frequencies.plotting.config import RADIAL_COLS, RADIAL_COLS_PRETTY


def plot_radial_plot(
    song_features: pd.DataFrame,
    title: str,
    inline: bool = False,
    only_return_trace: bool = False,
) -> Union[go.Figure, go.Scatterpolar, None]:
    """
    Plot radial plot of a specified song.

    Parameters
    ----------
    song_features : pd.DataFrame
        Data-frame containing song features.
    title : str
        Title of plot.
    inline : bool
        Whether to show plot inline.
    only_return_trace : bool
        If set to true, don't return the figure but only the trace object.
        Useful if combining multiple plots, e.g. song radial plot + playlist mood board

    Returns
    -------
    fig : plt.Figure :
        The figure object.
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "polar"}]], column_widths=[1])

    trace = go.Scatterpolar(
        r=song_features[RADIAL_COLS].values,
        theta=RADIAL_COLS_PRETTY,
        name="Song",
        line=dict(color="darkviolet"),
        showlegend=True,
    )

    if only_return_trace:
        return trace

    fig.add_trace(trace, row=1, col=1)

    fig.update_layout(
        template="plotly_white",
        title=title,
        plot_bgcolor="rgb(255,255,255)",
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=""),
        ),
    )

    if inline:
        fig.show()
    else:
        return fig


def plot_mood_board(
    playlist_features: pd.DataFrame,
    title: str,
    inline: bool = False,
    metrics_version: int = 1,
) -> go.Figure | None:
    """
    Plot a radial plot that summarises the characteristics of a playlist.

    Parameters
    ----------
    playlist_features : pd.DataFrame
        Song features of every song in a chosen playlist.
    title : str
        Title of plot.
    inline : bool
        Whether to plot the figure inline.
    metrics_version : int
        What sort of metrics to use for the plot.
        1 = Mean for the line and Std for the range
        2 = Median for the line and 10th and 90th percentile for the range

    Returns
    -------
    fig : plotly.graph_objects.Figure :
        The figure object.
    """
    # TODO: Add custom trace for +/- std around mean (e.g. area symbol)
    if metrics_version == 1:
        # Mean & std as variation
        playlist_features_summary = playlist_features.agg(["mean", "std"])
        main_line = playlist_features_summary.loc["mean", RADIAL_COLS].values
        upper_bound = (
            main_line + playlist_features_summary.loc["std", RADIAL_COLS].values
        )
        lower_bound = (
            main_line - playlist_features_summary.loc["std", RADIAL_COLS].values
        )
    elif metrics_version == 2:
        # Median and bottom / top 10th percentile as variation
        main_line = np.array(
            [np.percentile(playlist_features[c], 50) for c in RADIAL_COLS]
        )
        upper_bound = np.array(
            [np.percentile(playlist_features[c], 90) for c in RADIAL_COLS]
        )
        lower_bound = np.array(
            [np.percentile(playlist_features[c], 10) for c in RADIAL_COLS]
        )
    else:
        raise Exception(f"metrics_version = {metrics_version} is not supported")

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "polar"}]], column_widths=[1])

    fig.add_trace(
        go.Scatterpolar(
            r=main_line, theta=RADIAL_COLS_PRETTY, name="Average of Playlist"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatterpolar(
            r=lower_bound,
            theta=RADIAL_COLS_PRETTY,
            opacity=0.3,
            name="Lower Bound",
            line=dict(color="royalblue", width=0.1, dash="dash"),
            fillcolor="rgba(80,105,221,0.0)",
            fill="tonext",
            marker=dict(symbol="circle", size=0.1, opacity=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatterpolar(
            r=upper_bound,
            theta=RADIAL_COLS_PRETTY,
            opacity=0.3,
            name="Upper Bound",
            line=dict(color="royalblue", width=0.1, dash="dash"),
            fillcolor="rgba(80,105,221,0.5)",
            fill="tonext",
            marker=dict(symbol="circle", size=0.1, opacity=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        plot_bgcolor="rgb(255,255,255)",
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=""),
        ),
        showlegend=True,
    )

    if inline:
        fig.show()
    else:
        return fig
