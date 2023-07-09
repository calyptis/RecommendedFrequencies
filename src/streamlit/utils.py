import os

import pandas as pd
import streamlit as st

from src.project_config import DATA_DIR


FILE_ADDITIONAL_TRAINING_EXAMPLES = os.path.join(DATA_DIR, "collected_training_examples.csv")


def make_clickable_df(val: str) -> str:
    """
    Wrap value into an HTML tag in order to allow the value to be clickable if rendered as HTML.

    Parameters
    ----------
    val

    Returns
    -------

    """
    return '<a target="_blank" href="{}">{}</a>'.format(val, "preview")


def make_clickable_html(val: str, version: int = 2) -> str:
    """
    Wrap value into an HTML tag that allows audio to be played.

    Parameters
    ----------
    val
    version

    Returns
    -------
    """
    v1 = """
    <audio controls="controls">
        <source src="{}" type="audio/mpeg"/>
    </audio>
    """
    v2 = """
    <iframe 
        src="{}" width="80" height="30" frameborder="0" allowtransparency="true" allow="encrypted-media"> 
    </iframe>
    """
    versions = {1: v1, 2: v2}
    return versions[version].format(val)


def dataframe_with_selections(st_element: st.elements, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column to a dataframe when rendering it via streamlit.

    See: https://github.com/streamlit/streamlit/issues/688

    Parameters
    ----------
    st_element: Streamlit element to which the dataframe should be written to (e.g., column)
    df: pd.DataFrame
        Pandas dataframe to render

    Returns
    -------
    selected_rows: Dict[str, pd.DataFrame]
        Selected rows
    """
    col_name = "Save as negative training example"
    df_with_selections = df.copy()
    df_with_selections.insert(0, col_name, False)
    edited_df = st_element.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={col_name: st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )
    selected_rows = df[edited_df[col_name]]
    return selected_rows
