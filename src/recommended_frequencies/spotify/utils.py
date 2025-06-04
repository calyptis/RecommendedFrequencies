import pickle
from difflib import SequenceMatcher
from typing import Any, Iterator

import pandas as pd

from recommended_frequencies.spotify.config import MAIN_DATA_FILE


def clean_string(x: str):
    """
    Clean a string.

    This function wraps several string operations into one.

    Parameters
    ----------
    x : str :
        String to be cleaned

    Returns
    -------
    x : str :
        Cleaned string.
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
    string_sim: float :
        Similarity between two strings [0, 1].
    """
    string_sim = SequenceMatcher(None, string_a, string_b).ratio()
    return string_sim


def get_chunks(original_list: list[Any], n: int) -> Iterator[Any]:
    """
    Yield successive n-sized chunks from list.

    Parameters
    ----------
    original_list : list
        Original list to be split into chunks
    n : int
        Number of items in chunk
    """
    for i in range(0, len(original_list), n):
        yield original_list[i : i + n]


def read_pickle(path_to_file: str) -> list[Any]:
    """
    Sequentially read a pickle file.

    Parameters
    ----------
    path_to_file: str :
        Path to the pickle file.

    Returns
    -------
    output : list[Any] :
        Content of the pickle file.
    """
    output = []
    with open(path_to_file, "rb") as f:
        try:
            while True:
                output += pickle.load(f)
        except EOFError:
            pass
    return output


def concat_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    """
    Transform a list of lists into a flat list.

    Parameters
    ----------
    list_of_lists : list[list[Any]]
        List of lists.

    Returns
    -------
    flattened_list : list[Any] :
        List of elements that in turn are not lists anymore
    """
    flattened_list = [val for sublist in list_of_lists for val in sublist]
    return flattened_list


def get_frequent_genres(
    out_path: str = MAIN_DATA_FILE, thr: int = 50, return_type: str = "list"
) -> pd.DataFrame | list[str]:
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
    vals : pd.DataFrame | list[str] :
        List of genres ordered by frequency.
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
