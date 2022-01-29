import os
import pathlib
import json


MAIN_DIR = os.path.split(pathlib.Path(__file__).parent.resolve())[0]

DATA_DIR = os.path.join(MAIN_DIR, "data")

MODEL_DIR = os.path.join(MAIN_DIR, "model")

CREDENTIALS_DIR = os.path.join(MAIN_DIR, "credentials")

ALLOWED_EXTENSIONS = {'xml'}

SCOPES_LIST = [
    "user-library-modify",
    "user-library-read",
    "playlist-modify-private",
    "playlist-read-private",
    "playlist-modify-public"
]

SCOPES = " ".join(SCOPES_LIST)

CREDENTIALS = json.load(open(os.path.join(CREDENTIALS_DIR, "credentials.json"), "rb"))

DIRS = [MAIN_DIR, DATA_DIR]
