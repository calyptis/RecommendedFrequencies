import pandas as pd


def get_top_results(
    df_results: pd.DataFrame, genre_weight: int = 0, n: int = 10
) -> pd.DataFrame:
    """
    Limit the results to a selected number of most similar songs.

    Parameters
    ----------
    df_results : pd.DataFrame
        Songs and their similarity scores
    genre_weight : float
        How much genre similarity matters for selecting the most similar songs.
    n : int
        Number of songs to show.
    Returns
    -------
    df_top_n : pd.DataFrame
    """
    euclidean_weight = 1 - genre_weight
    df_top_n = (
        df_results.assign(
            Similarity=lambda row: 1
            - (
                (euclidean_weight * row.EuclideanDistance)
                + (genre_weight * row.GenreDistance)
            )
        )
        .sort_values(by="Similarity", ascending=False)
        .head(n)
    )
    return df_top_n
