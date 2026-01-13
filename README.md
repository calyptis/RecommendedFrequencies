# Recommended Frequencies

[**Slide deck**](resources/presentation/Presentation.pdf) | [**Blog post**](https://calyptis.github.io/RecommendedFrequencies/)

# Announcement
As of May 2025, Spotify reserves access to ["audio features to apps with established, scalable, and impactful use cases".](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)
Unfortunately, this means that this project will not work anymore unless Spotify grants access on an individual basis to projects such as these.
Unless your developer account has been granted extended access by Spotify, this use case will not work.

# Introduction
Recommended Frequencies is a playlist recommendation engine designed for Spotify. It analyzes a selected playlist from a user's library and suggests additional tracks from their liked songs that would complement it well.

This tool exclusively uses information available in a user’s Spotify library — no collaborative filtering or social data is involved. Recommendations are powered by a CatBoost model trained on audio features, song attributes, and genre embeddings provided by [Every Noise at Once](https://everynoise.com/).
For more details, see the [the model section](#model).
The audio features are sourced from [Spotify's API](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features) along with the year of album release. A full list of features can be found [here](#audio_features).

The app is developed using `streamlit`.

# Dashboard

## Step 1: Select a Playlist
Choose any playlist from your Spotify library. The app displays:

A song selection
Audio-based profile of the playlist
Album covers for visual reference

<img src="resources/images/page1.png" width="800" />

## Step 2: Navigate Recommendations

- Under the `Similarity Settings` drop-down menu, one can select the method and the number of results to show.
- The radial plot visualises a selected recommendation.

<img src="resources/images/page2.png" width="800" />

# Installation guide

## 1. Set up Spotify developer account & register app

These instructions were tested with Spotify’s website version from 2021-01-17.

1. Go to Spotify Developer Dashboard and log in or create an account.
2. Click CREATE AN APP.
3. Fill in the app name and description. Agree to the terms and click CREATE.
4. Click EDIT SETTINGS and add the following Redirect URIs:
  - http://localhost:9000/callback/
  - http://localhost:8090
5. Note your Client ID and Client Secret — you’ll need them in Step 2.2.

## 2. Set up the environment on your machine

### 2.1 Get the code & set environmental variables

The below instructions are for Linux or MacOS.

```commandline
git clone git@github.com:calyptis/RecommendedFrequencies.git
cd RecommendedFrequencies
pip install -r requirements.txt
pip install -e .
```

### 2.2 Specify your credentials

Create a `credentials.json` file inside the `credentials` folder:

```python
{
	"client_id": "{your_client_id}",
	"client_secret": "{you_client_secret}",
	"redirect_uri": "http://localhost:9000/callback/",
	"redirect_flask_uri": "http://localhost:8090"
}
```

Replace `{your_client_id}` and `{your_client_secret}` with the values from Step 1.

### 2.3 Collect Spotify Data

To fetch the necessary data and cache it locally, run:

```commandline
python src/recommended_frequencies/spotify/main.py
```

Note that this will open a Spotify login page in the browser were one must
authenticate themselves.

Once logged in, a `.spotify_caches` will be created in the root folder.

Subsequent API calls will not trigger a log-in page anymore unless this file is deleted.

### 2.4 Run the dashboard

Start the app:

```commandline
streamlit run src/recommended_frequencies/streamlit/main.py
```

# User guide

## 1. Audio features<a name="audio_features"></a>

Documentation of all the available audio features can be found on
[Spotify's official documentation page](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features).

In this project, a subset of those are used and are listed below with their official description for convenience:

1. **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
2. **Danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
3. **Energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
4. **Instrumentalness**: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
5. **Liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
6. **Loudness**: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
7. **Speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words.
8. **Tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
9. **Valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
10. **Year of Album Release**: Release year of the album the track appears on.

All these features are scaled to [0, 1] – if they are not already.

## 2. [Model](#model)

The recommendation model is based on CatBoost, trained to learn song similarity.

- Positive pairs: Songs appearing together in playlists.
- Negative pairs: Songs from deliberately dissimilar playlists.

Each pair is represented by:
- The difference in audio features
- Their positions in a 2D genre space based on [Every Noise at Once](https://everynoise.com)

A song’s genre vector is computed as the average position of its associated genres in this space.

# Known Bugs:
- All audio previews are sometimes played at the same time when updating the dashboard.

# TODO:
- [ ] Get the oldest year of release for a given song.
  - For example, if a song appeared in a recent remastered album, track down the first album the song appears in and use that as a release date.
- [ ] Simplify `spotify/` submodule => one module per data type (tracks, playlists, genres, embeddings, etc.)

# Related projects
- https://dubolt.com
- https://newsroom.spotify.com/2021-09-09/get-perfect-song-recommendations-in-the-playlists-you-create-with-enhance/
- https://spotibot.com
- https://www.chosic.com/

# Resources:
## Spotify
- Sample of data: https://www.kaggle.com/nadintamer/top-tracks-of-2017
- Description of features: https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features
## Music Genre Analysis
  - Genre similarity: https://everynoise.com/
## Music Recommendation Research
- [ACM model](https://dl.acm.org/doi/10.1145/3383313.3412248)
- [PopSci Article](https://www.popsci.com/technology/spotify-audio-recommendation-research/)
- [Pre-trained Essentia models](https://essentia.upf.edu/machine_learning.html)
