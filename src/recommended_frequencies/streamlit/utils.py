import pandas as pd
import streamlit as st


def make_clickable_df(val: str) -> str:
    """
    Wrap value into an HTML tag in order to allow the value to be clickable if dataframe is rendered as HTML.

    Parameters
    ----------
    val: str :
        String to be wrapped into an HTML tag.

    Returns
    -------
    html_tag : str :
        HTML tag that allows val to be clickable.
    """
    html_tag = '<a target="_blank" href="{}">{}</a>'.format(val, "preview")
    return html_tag


def make_clickable_html(val: str, version: int = 2) -> str:
    """
    Wrap value into an HTML tag that allows audio to be played.

    Parameters
    ----------
    val: str :
        The string to be wrapped into an HTML tag.
    version : int :
        The version of the audio HTML tag. Either 1 or 2.

    Returns
    -------
    html_tag : str :
        The HTML tag that allows audio to be played.
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
    html_tag = versions[version].format(val)
    return html_tag


def dataframe_with_selections(st_element: st.elements, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column to a dataframe when rendering it via streamlit.

    See: https://github.com/streamlit/streamlit/issues/688

    Parameters
    ----------
    st_element: st.elements :
        Streamlit element to which the dataframe should be written to (e.g., column)
    df: pd.DataFrame :
        Pandas dataframe to render

    Returns
    -------
    selected_rows: Dict[str, pd.DataFrame] :
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
