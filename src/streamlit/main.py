import streamlit as st
import os
import pandas as pd
from io import BytesIO
from src.project_config import DATA_DIR
from src.plotting.mood_board import plot_mood_board, plot_radial_plot
from src.plotting.album_cover_collage import plot_album_covers
from src.modelling.baseline import get_baseline, get_top_results
from src.streamlit.utils import make_clickable_html
from src.spotify.config import PLAYLIST_FILE, ALBUM_COVER_FILE
from src.spotify.utils import read_pickle


COL_ORDER = [
    "PreviewURL",
    "SongName",
    "Artist",
    "ID"
]

SUGGESTION_ENGINES = {
    "Numeric Song Attributes": get_baseline,
    "Numeric Song Attributes + Genre": get_baseline
}

GENRE_SIMILARITY = [
    "word2vec",
    "everynoise",
    "co-occurrence",
    "set-similarity",
    # "weighted set-similarity"
]

# If certain playlists should be excluded from the dashboard
# they need to be listed in the file below
exclude_playlist_path = os.path.join(DATA_DIR, "exclude_playlists.txt")
if os.path.exists(exclude_playlist_path):
    with open(exclude_playlist_path, "r") as f:
        exclude_playlists = f.read().split("\n")
else:
    exclude_playlists = []

# In case it is easier to just list the playlists that a user should
# be able to choose from
show_playlist_path = os.path.join(DATA_DIR, "show_playlists.txt")
if os.path.exists(show_playlist_path):
    with open(show_playlist_path, "r") as f:
        show_playlists = f.read().split("\n")
else:
    show_playlists = []


app_title = 'Recommended Frequencies'
st.set_page_config(page_title=app_title, page_icon="🎵", layout="wide")

# --------- Data


@st.cache
def get_playlists():
    list_playlists = read_pickle(PLAYLIST_FILE)
    dict_playlists = {playlist["name"][0]: playlist for playlist in list_playlists}
    return dict_playlists


@st.cache
def get_data():
    data = pd.read_pickle(os.path.join(DATA_DIR, "data.pickle"))
    data.index.name = "ID"
    data.Artist = data.Artist.apply(lambda x: x.split(" | ")[0])
    return data


@st.cache
def get_album_covers():
    album_covers = read_pickle(ALBUM_COVER_FILE)
    return {i[0]: i[-1] for i in album_covers}


playlists = get_playlists()
features = get_data()
playlist_album_covers = get_album_covers()
track_info = features[[i for i in COL_ORDER if i != "ID"]].copy()
all_songs_with_features = set(features.index)

if show_playlists:
    playlist_options = show_playlists
else:
    playlist_options = sorted(set(playlists.keys()) - set(exclude_playlists))

# --------- First row of page
st.title("Recommended Frequencies: a recommendation system for playlists")
st.markdown("**This app allows users to identify songs in their library that may fit well into a selected playlist**")
controls = st.expander("Page Settings")
include_audio_preview = controls.checkbox("Include Audio Preview", value=False)
st.markdown("---")
st.markdown("### 1. Choose a playlist")
select_playlist = st.selectbox("", playlist_options)

# --------- Get playlist related info
playlist_tracks = sorted(list(set(playlists[select_playlist]["tracks"]).intersection(set(all_songs_with_features))))
playlist_features = features.loc[playlist_tracks].copy()
playlist_info = track_info.loc[playlist_tracks]
mood_board = plot_mood_board(playlist_features, title="", inline=False, metrics_version=1)

# --------- Second row of page: playlist info
col1, col2 = st.columns(2)
col1.markdown(f"#### Examples of songs in playlist «{select_playlist}»")
if include_audio_preview:
    col1.write(
        (
            playlist_info
            .reset_index()
            [COL_ORDER]
            .head(10)
            .style.format({'PreviewURL': make_clickable_html})
            .to_html()
        ),
        unsafe_allow_html=True
    )
else:
    col1.dataframe(
        (
            playlist_info
            .reset_index()
            [COL_ORDER]
            .head(10)
            .drop("PreviewURL", axis=1)
        ),
        height=600
    )
col2.markdown(f"#### Song attributes for playlist «{select_playlist}»")
col2.plotly_chart(mood_board)

try:
    # Collage of album covers
    # collage = plot_album_covers(list(playlist_features.AlbumCover.values[:10]))
    collage = plot_album_covers(playlist_album_covers[select_playlist])
    # Use st.image with image in buffer to allow transparent background, instead of st.pyplot(collage)
    buf = BytesIO()
    collage.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=3*collage.dpi)
    st.image(buf, use_column_width=True)
except KeyError:
    pass

# --------- Third row of page: similarity metric controls
st.markdown("---")
st.markdown("### 2. Find similar songs")
controls = st.expander("Similarity Settings")
select_suggestion_engine = controls.selectbox("Similarity Features", SUGGESTION_ENGINES.keys())
suggestion_engine = SUGGESTION_ENGINES[select_suggestion_engine]
if select_suggestion_engine == "Numeric Song Attributes + Genre":
    genre_similarity = controls.selectbox("Genre Similarity Metric", GENRE_SIMILARITY, index=1)
    genre_weight = controls.slider("Genre Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
else:
    # Default values
    genre_weight = 0
    genre_similarity = None
top_n = controls.slider("Nr of Suggestions", min_value=5, max_value=20, value=10, step=1)

# --------- Find most similar songs


@st.cache
def get_results_wrapper(_suggestion_engine, _playlist_features, _songs_available_for_suggestion_features,
                        _genre_similarity):
    """
    Wrap suggestion_engine into a function in order to use st.cache()
    """
    if _genre_similarity == "everynoise":
        _songs_available_for_suggestion_features = (
            _songs_available_for_suggestion_features
            .query("missing_everynoise_genre == False")
        )
    return _suggestion_engine(
        playlist_features=_playlist_features,
        songs_available_for_suggestion_features=_songs_available_for_suggestion_features,
        genre_similarity=_genre_similarity,
    )


@st.cache
def get_top_results_wrapper(_song_similarity, _genre_weight, _top_n):
    """
    Wrap get_top_results into a function in order to use st.cache()
    """
    return get_top_results(results=_song_similarity, genre_weight=genre_weight, n=_top_n)


songs_available_for_suggestion = list(all_songs_with_features - set(playlist_tracks))
songs_available_for_suggestion_features = features.loc[songs_available_for_suggestion].copy()
song_similarity = get_results_wrapper(
    _suggestion_engine=suggestion_engine,
    _playlist_features=playlist_features,
    _songs_available_for_suggestion_features=songs_available_for_suggestion_features,
    _genre_similarity=genre_similarity,
)
suggested_songs = get_top_results_wrapper(_song_similarity=song_similarity, _genre_weight=genre_weight, _top_n=top_n)
suggested_songs_info = track_info.join(suggested_songs, how="inner").sort_values(by="Distance")
for c in ["EuclideanDistance", "GenreDistance", "Distance"]:
    suggested_songs_info[c] = suggested_songs_info[c].map(lambda x: '{0:.2f}'.format(x))

# --------- Fourth row of page: display results
col1, col2 = st.columns(2)
col1.markdown(f"#### Most similar songs to playlist «{select_playlist}»")
if include_audio_preview:
    col1.write(
        (
            suggested_songs_info
            .reset_index()
            [COL_ORDER + ["Distance"]]
            .style.format({'PreviewURL': make_clickable_html})
            .to_html()
        ),
        height=600,
        unsafe_allow_html=True
    )
else:
    col1.dataframe(
        (
            suggested_songs_info
            .reset_index()
            [COL_ORDER]
            .drop("PreviewURL", axis=1)
        ),
        height=600
    )
col2.markdown(
    "#### Visualise similarity of proposed song"
)
select_song_suggested = col2.text_input(
    "Spotify song ID to visualise",
    suggested_songs.index[0]
)
selected_song_name, selected_song_artist = track_info.loc[select_song_suggested, ["SongName", "Artist"]].values
song_radial_plot_trace = plot_radial_plot(
    features.loc[select_song_suggested].copy(),
    title=f"{selected_song_name} by {selected_song_artist}",
    only_return_trace=True
)
mood_board.add_trace(song_radial_plot_trace)
mood_board.update_layout(
    title=f"Song: {selected_song_name} by {selected_song_artist}" + "<br>" +
    f"Playlist: {select_playlist}"

)
col2.plotly_chart(mood_board)
if not include_audio_preview:
    col2.markdown("##### Listen to proposed song")
    col2.audio(track_info.loc[select_song_suggested, "PreviewURL"])
