import json
import os
import pathlib

# Directories
MAIN_DIR = os.path.split(pathlib.Path(__file__).parent.parent.resolve())[0]
DATA_DIR = os.path.join(MAIN_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared")
CREATED_DATA_DIR = os.path.join(DATA_DIR, "created")
MODEL_DIR = os.path.join(MAIN_DIR, "model")
DIRS = [
    MAIN_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PREPARED_DATA_DIR,
    MODEL_DIR,
    CREATED_DATA_DIR,
]
for directory in DIRS:
    if not os.path.exists(directory):
        os.makedirs(directory)

CREDENTIALS_DIR = os.path.join(MAIN_DIR, "credentials")

ALLOWED_EXTENSIONS = {"xml"}

SCOPES_LIST = [
    "user-library-modify",
    "user-library-read",
    "playlist-modify-private",
    "playlist-read-private",
    "playlist-modify-public",
]

SCOPES = " ".join(SCOPES_LIST)

CREDENTIALS = json.load(open(os.path.join(CREDENTIALS_DIR, "credentials.json"), "rb"))
