import pandas as pd
import itertools


def set_similarity(a: set, b: set):
    """
    Set similarity, which equals |A u B| / |A v B|

    Parameters
    ----------
    a : set
    b : set

    Returns
    -------

    """
    return len(a.intersection(b)) / len(a.union(b))


def weighted_set_similarity(a: set, b: set, df_genre_occurrence: pd.DataFrame):
    """
    TODO: Re-think approach
    Weighted similarity between two sets, where weight is the occurrence of the respective genre.
    https://mathoverflow.net/questions/123339/weighted-jaccard-similarity

    Parameters
    ----------
    a: set
    b: set
    df_genre_occurrence: pd.DataFrame

    Returns
    -------

    """
    genre_intersection = list(a.intersection(b))
    genre_union = list(a.union(b))
    return df_genre_occurrence.loc[genre_intersection].sum() / df_genre_occurrence.loc[genre_union].sum()


def co_occurrence_similarity(co_occurrence_lookup_table: pd.DataFrame, playlist_genres: list, tuple_b: tuple):
    """
    Measures co-occurrence similarity of genres between songs in a selected playlist and given song.

    Parameters
    ----------
    co_occurrence_lookup_table : pd.Dataframe
        Contains all co-occurrence values of genres, in the form of:
        | genre_a | genre_b | co-occurrence value
    playlist_genres : list
        Genre list of all songs in a given playlist A.
    tuple_b : tuple
        Song ID and genre list of song B.

    Returns
    -------
    Tuple of Song ID of song B and similarity between songs in playlist A and song B.
    """
    song_id, song_genres = tuple_b
    genre_overlap_pairs = [tuple(sorted(i)) for i in itertools.product(playlist_genres, song_genres)]
    genre_playlist_pairs = [tuple(sorted(i)) for i in itertools.combinations(playlist_genres, 2)]
    co_occurrence_overlap = co_occurrence_lookup_table.loc[genre_overlap_pairs, "CoOccurrenceValue"].sum()
    co_occurrence_playlist = co_occurrence_lookup_table.loc[genre_playlist_pairs, "CoOccurrenceValue"].sum()
    similarity = co_occurrence_overlap / co_occurrence_playlist
    return song_id, similarity
