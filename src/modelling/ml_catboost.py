from typing import Tuple

import catboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.modelling.config import CATBOOST_FEATURES


def create_train_test_split(
    df_for_model: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO: add docstring.

    Parameters
    ----------
    df_for_model: pd.DataFrame

    Returns
    -------

    """
    # TODO: Use train_test_split
    train_examples = np.random.choice(
        a=df_for_model.index, size=int(len(df_for_model) * 0.8), replace=False
    )
    test_examples = list(set(df_for_model.index) - set(train_examples))
    df_train = df_for_model.loc[train_examples]
    df_test = df_for_model.loc[test_examples]

    return df_train, df_test


def train_catboost(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> catboost.core.CatBoostClassifier:
    """
    TODO: add docstring.

    Parameters
    ----------
    df_train
    df_test

    Returns
    -------

    """
    target_col = "target"
    feature_cols = df_train.columns.difference([target_col, "ID"])
    categorical_feature_cols = ["playlist"]
    model = catboost.CatBoostClassifier(cat_features=categorical_feature_cols)
    # TODO: print => logging
    print("Train catboost model")
    model.fit(X=df_train[feature_cols], y=df_train[target_col], verbose=0)

    # Assess performance
    yhat_train = model.predict(df_train[feature_cols])
    yhat_test = model.predict(df_test[feature_cols])
    train_acc = accuracy_score(df_train[target_col], yhat_train)
    test_acc = accuracy_score(df_test[target_col], yhat_test)
    print("Accuracy:\n")
    print(f"Train:\t{train_acc*100:.1f}%")
    print(f"Test:\t{test_acc*100:.1f}%")

    return model


def get_catboost_predictions(
    model_catboost: catboost.core.CatBoostClassifier,
    df_playlist_features: pd.DataFrame,
    df_songs_available_for_suggestion_features: pd.DataFrame,
    playlist_name: str,
    **kwargs,
) -> pd.DataFrame:
    """
    TODO: add docstring.

    Parameters
    ----------
    model_catboost
    df_playlist_features
    df_songs_available_for_suggestion_features
    playlist_name
    kwargs

    Returns
    -------

    """
    # TODO: add genre_{x,y} already in MAIN_DATA_FILE
    for df in [df_playlist_features, df_songs_available_for_suggestion_features]:
        df["genre_x"] = df["GenreEveryNoiseEmbedding"].apply(lambda x: x[0])
        df["genre_y"] = df["GenreEveryNoiseEmbedding"].apply(lambda x: x[1])
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
    TODO: add docstring.

    Parameters
    ----------
    df_playlist_features
    df_songs_available_for_suggestion_features
    playlist_name

    Returns
    -------

    """
    if "ID" not in df_playlist_features.columns:
        df_playlist_features.reset_index(inplace=True)
    if "ID" not in df_songs_available_for_suggestion_features.columns:
        df_songs_available_for_suggestion_features.reset_index(inplace=True)

    df_inference = df_playlist_features.merge(
        df_songs_available_for_suggestion_features, how="cross", suffixes=("_a", "_b")
    ).assign(playlist=playlist_name)

    return df_inference
