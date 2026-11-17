"""Loading the processed data. Nutrient columns keep NaN for unknown values."""

from __future__ import annotations

import pandas as pd

from . import constraints as cons
from .config import FOODS_FILE, INTERACTIONS_FILE, RECIPES_FILE

NUTRIENT_COLS = ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]


def load_recipes(path=RECIPES_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["keywords", "ingredients", "instructions", "description", "allergens_found"]:
        df[col] = df[col].fillna("")
    df["search_text"] = [cons.recipe_text(n, i, s) for n, i, s in
                         zip(df.name, df.ingredients, df.instructions)]
    return df.reset_index(drop=True)


def load_interactions(path=INTERACTIONS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df.ts)
    return df


def load_foods(path=FOODS_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def nutrient(value):
    """Turn a pandas value into a float or None. None means unknown, never zero."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def recipe_nutrition(row) -> dict:
    """Nutrition of one recipe with its basis and source, unknown values as None."""
    return {
        "basis": row["nutrition_basis"],
        "source": row["source"],
        "status": row["nutrition_status"],
        **{c: nutrient(row[c]) for c in NUTRIENT_COLS},
    }
