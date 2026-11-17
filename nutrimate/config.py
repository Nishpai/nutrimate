"""Paths and default settings shared by the scripts and the app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
APP_DB = ROOT / "data" / "nutrimate_app.db"

RECIPES_FILE = DATA_PROCESSED / "recipes.csv.gz"
INTERACTIONS_FILE = DATA_PROCESSED / "interactions.csv.gz"
FOODS_FILE = DATA_PROCESSED / "foods.csv.gz"
SPLIT_FILE = DATA_PROCESSED / "split.csv.gz"

# Hybrid weights. w_cf = 0.75 was the best of {0, 0.25, 0.5, 0.75, 1} on the validation
# split in evaluate.py. It is a tuned setting for this data, not a general constant.
DEFAULT_W_CF = 0.75
DEFAULT_W_GOAL = 0.15
MMR_LAMBDA = 0.7
TOP_K = 10

# A rating of 4 or 5 counts as "liked" for the ranking metrics.
RELEVANT_RATING = 4

# With fewer liked recipes than this, CF scores are averaged with popularity, because in
# the cold-start experiment (3 known ratings) popularity beat CF alone.
CF_MIN_HISTORY = 5

# Weight of popularity for a new user who typed preferences but has no likes yet.
# This one is a design choice, not tuned: we have no logged typed preferences to tune it on.
COLD_START_POP_WEIGHT = 0.4

# Number of meals a day, used only to turn a daily target into a rough per-meal target.
MEALS_PER_DAY = 3
