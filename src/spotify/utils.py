import pickle
from difflib import SequenceMatcher

import pandas as pd

from src.spotify.config import MAIN_DATA_FILE


def clean_string(x: str):
    """
    Clean a string.

    This function wraps several string operations into one.

    Parameters
    ----------
    x : str
        String to be cleaned

    Returns
    -------
    """
    return x.lower().strip()


def string_similarity(string_a: str, string_b: str):
    """
    Perform string similarity.

    Parameters
    ----------
    string_a : str
    string_b : str

    Returns
    -------
    """
    return SequenceMatcher(None, string_a, string_b).ratio()


def get_chunks(original_list: list, n: int):
    """
    Yield successive n-sized chunks from list.

    Parameters
    ----------
    original_list : list
        Original list to be split into chunks
    n : int
        Number of items in chunk

    Returns
    -------
    """
    for i in range(0, len(original_list), n):
        yield original_list[i : i + n]


def read_pickle(path_to_file):
    """
    TODO: add docstring.

    Parameters
    ----------
    path_to_file

    Returns
    -------

    """
    output = []
    with open(path_to_file, "rb") as f:
        try:
            while True:
                output += pickle.load(f)
        except EOFError:
            pass
    return output


def concat_lists(list_of_lists: list):
    """
    Transform a list of lists into a flat list.

    Parameters
    ----------
    list_of_lists : list[list]

    Returns
    -------
    List of elements that in turn are not lists anymore
    """
    return [val for sublist in list_of_lists for val in sublist]


def get_frequent_genres(
    out_path: str = MAIN_DATA_FILE, thr: int = 50, return_type="list"
):
    """
    Find the most frequent genres in a user's library.

    This is useful because rarely occurring genres can be disregarded when measuring genre similarities.

    Parameters
    ----------
    out_path : str
    thr : int
        How many songs need to be of a given genre in order for the genre to be considered frequent.
    return_type : str
        How to return the most frequent genres:
            - "list" => just the names of the genres
            - "df" => pd.DataFrame with genre name as index and occurrence as value

    Returns
    -------

    """
    data = pd.read_pickle(out_path)
    vals = pd.Series(concat_lists(data.GenreList)).value_counts()
    mask = vals >= thr
    if return_type == "list":
        return vals[mask].index.tolist()
    elif return_type == "df":
        return vals[mask]
    else:
        raise Exception(f"return_type == '{return_type}' is not supported")
