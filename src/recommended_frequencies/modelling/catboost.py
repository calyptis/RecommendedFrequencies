from typing import Tuple

import catboost
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from recommended_frequencies.modelling.config import CATBOOST_FEATURES, TRAIN_FRAC


def create_train_test_split(
    df_for_model: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split model data into training and test sets.

    Randomly samples a fraction of examples for training, with the remainder
    used for testing. Note: does not set a random seed, so results are not
    reproducible across runs.

    Parameters
    ----------
    df_for_model : pd.DataFrame
        DataFrame with features and target column, indexed by example ID.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test) - Training and test DataFrames.
    """
    # TODO: Use train_test_split
    train_examples = np.random.choice(
        a=df_for_model.index, size=int(len(df_for_model) * TRAIN_FRAC), replace=False
    )
    test_examples = list(set(df_for_model.index) - set(train_examples))
    df_train = df_for_model.loc[train_examples]
    df_test = df_for_model.loc[test_examples]

    return df_train, df_test


def train_catboost(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> catboost.core.CatBoostClassifier:
    """
    Train a CatBoost classifier on song pair similarity data.

    Fits a CatBoostClassifier using 'playlist' as a categorical feature.
    Logs train and test accuracy after fitting.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with feature columns and 'target' column.
    df_test : pd.DataFrame
        Test data for evaluation, same structure as df_train.

    Returns
    -------
    catboost.core.CatBoostClassifier
        The trained CatBoost model.
    """
    target_col = "target"
    feature_cols = df_train.columns.difference([target_col, "ID"])
    categorical_feature_cols = ["playlist"]
    model = catboost.CatBoostClassifier(cat_features=categorical_feature_cols)
    logger.info("Train catboost model")
    model.fit(X=df_train[feature_cols], y=df_train[target_col], verbose=0)

    # Assess performance
    yhat_train = model.predict(df_train[feature_cols])
    yhat_test = model.predict(df_test[feature_cols])
    train_acc = accuracy_score(df_train[target_col], yhat_train)
    test_acc = accuracy_score(df_test[target_col], yhat_test)
    logger.info("Accuracy:\n")
    logger.info(f"Train:\t{train_acc*100:.1f}%")
    logger.info(f"Test:\t{test_acc*100:.1f}%")

    return model


def get_catboost_predictions(
    model_catboost: catboost.core.CatBoostClassifier,
    df_playlist_features: pd.DataFrame,
    df_songs_available_for_suggestion_features: pd.DataFrame,
    playlist_name: str,
) -> pd.DataFrame:
    """
    Generate song recommendations using the trained CatBoost model.

    Creates pairwise combinations of playlist songs and candidate songs,
    predicts similarity scores, and returns averaged scores per candidate
    song sorted by descending similarity.

    Parameters
    ----------
    model_catboost : catboost.core.CatBoostClassifier
        Trained CatBoost model for similarity prediction.
    df_playlist_features : pd.DataFrame
        Features for songs currently in the target playlist.
    df_songs_available_for_suggestion_features : pd.DataFrame
        Features for candidate songs to recommend.
    playlist_name : str
        Name of the playlist (used as categorical feature).

    Returns
    -------
    pd.Series
        Similarity scores indexed by song ID, sorted descending.
    """
    df_inference = create_inference_data(
        df_playlist_features[CATBOOST_FEATURES],
        df_songs_available_for_suggestion_features[CATBOOST_FEATURES],
        playlist_name,
    )
    df_inference["Similarity"] = model_catboost.predict_proba(
        df_inference[model_catboost.feature_names_]
    )[:, 1]
    df_inference_results = (
        df_inference.rename(columns={"ID_b": "ID"})
        .groupby("ID")
        .Similarity.mean()
        .sort_values(ascending=False)
    )

    return df_inference_results


def create_inference_data(
    df_playlist_features: pd.DataFrame,
    df_songs_available_for_suggestion_features: pd.DataFrame,
    playlist_name: str,
) -> pd.DataFrame:
    """
    Create cross-product of playlist and candidate songs for inference.

    Generates all pairwise combinations between songs in the playlist and
    candidate songs, with features suffixed '_a' (playlist) and '_b' (candidate).

    Note: Modifies input DataFrames in-place by resetting index if 'ID' column
    is not present.

    Parameters
    ----------
    df_playlist_features : pd.DataFrame
        Features for songs in the target playlist.
    df_songs_available_for_suggestion_features : pd.DataFrame
        Features for candidate songs.
    playlist_name : str
        Playlist name to assign as categorical feature.

    Returns
    -------
    pd.DataFrame
        Cross-product DataFrame with suffixed feature columns and 'playlist' column.
    """
    if "ID" not in df_playlist_features.columns:
        df_playlist_features.reset_index(inplace=True)
    if "ID" not in df_songs_available_for_suggestion_features.columns:
        df_songs_available_for_suggestion_features.reset_index(inplace=True)

    df_inference = df_playlist_features.merge(
        df_songs_available_for_suggestion_features, how="cross", suffixes=("_a", "_b")
    ).assign(playlist=playlist_name)

    return df_inference
