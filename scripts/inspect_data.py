"""Milestone 0: look at the raw files before building anything.

Writes results/data_inspection.md with row counts, columns, missing values,
rating distribution, duplicate checks and unit notes. Run after download_data.py.
"""

import ast

import _path  # noqa: F401
import duckdb
import pandas as pd

from nutrimate.config import DATA_RAW, RESULTS_DIR

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def table(df):
    out(df.to_markdown() if hasattr(df, "to_markdown") and _has_tabulate() else "```\n" + df.to_string() + "\n```")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def inspect_foodcom():
    path = DATA_RAW / "foodrs.duckdb"
    con = duckdb.connect(str(path), read_only=True)
    out("## 1. Food.com Recipes and Reviews (recipes_raw, reviews_raw)")
    rec = con.sql("SELECT * EXCLUDE (AuthorName, Images) FROM recipes_raw").df()
    rev = con.sql("SELECT ReviewId, RecipeId, AuthorId, Rating, DateSubmitted FROM reviews_raw").df()
    out(f"recipes: {len(rec):,} rows, {rec.shape[1]} columns (author names and image URLs not loaded)")
    out(f"reviews: {len(rev):,} rows")
    out()
    out("Missing values in recipes (only columns with any):")
    na = rec.isna().sum()
    table(na[na > 0].to_frame("missing"))
    out()
    out(f"Duplicate RecipeId: {rec.RecipeId.duplicated().sum()}  |  duplicate (user, recipe) reviews: "
        f"{rev.duplicated(['AuthorId', 'RecipeId']).sum()}")
    out(f"Reviews pointing to a recipe that does not exist: {(~rev.RecipeId.isin(rec.RecipeId)).sum()}")
    out()
    out("Rating distribution (0 means the review had no star rating):")
    table(rev.Rating.value_counts().sort_index().to_frame("reviews"))
    out()
    nut = ["Calories", "ProteinContent", "CarbohydrateContent", "FatContent", "FiberContent",
           "SugarContent", "SodiumContent"]
    out("Nutrition columns (Food.com shows these per serving; kcal, grams, sodium in mg):")
    table(rec[nut].describe(percentiles=[0.01, 0.5, 0.99]).T.round(1))
    m = rec[rec.Calories > 50]
    est = 4 * m.ProteinContent + 4 * m.CarbohydrateContent + 9 * m.FatContent
    out()
    out(f"Check: median of (4*protein + 4*carbs + 9*fat) / calories = {(est / m.Calories).median():.3f}. "
        "Close to 1 means the macros and calories describe the same serving.")
    lengths = rec.RecipeIngredientParts.map(len) - rec.RecipeIngredientQuantities.map(len)
    out(f"Recipes where the ingredient list and quantity list have different lengths: {(lengths != 0).sum():,}")
    out()
    out("Categories that are not food (dropped in preprocessing):")
    table(rec[rec.RecipeCategory.isin(["Household Cleaner", "Bath/Beauty", "Homeopathy/Remedies"])]
          .RecipeCategory.value_counts().to_frame("recipes"))
    out()


def inspect_majumder():
    path = DATA_RAW / "RAW_recipes.csv"
    if not path.exists():
        out("## 2. Food.com RAW_recipes.csv (Majumder et al.): not present, skipped")
        return
    out("## 2. Food.com RAW_recipes.csv / RAW_interactions.csv (Majumder et al. 2019)")
    rec = pd.read_csv(path, usecols=["id", "nutrition"])
    inter = pd.read_csv(DATA_RAW / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "rating"])
    out(f"recipes: {len(rec):,}  |  interactions: {len(inter):,}  |  users: {inter.user_id.nunique():,}")
    nut = pd.DataFrame(rec.nutrition.map(ast.literal_eval).tolist(),
                       columns=["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv",
                                "sat_fat_pdv", "carbs_pdv"])
    out("The `nutrition` column is a 7-item list: calories (kcal) and then six percent daily value "
        "numbers. Converting %DV to grams needs reference daily values that the dataset does not state, "
        "so we did not use this copy for macro tracking.")
    table(nut.describe(percentiles=[0.5]).T.round(1))
    out()


def inspect_indb():
    path = DATA_RAW / "Anuvaad_INDB_2024.11.xlsx"
    out("## 3. Indian Nutrient Databank (INDB 2024.11)")
    d = pd.read_excel(path)
    out(f"{len(d):,} recipes, {d.shape[1]} columns. Plain nutrient columns are per 100 g; "
        "`unit_serving_*` columns are per one serving of `servings_unit` (for example 1 bowl).")
    cols = ["energy_kcal", "protein_g", "carb_g", "fat_g", "fibre_g", "unit_serving_energy_kcal",
            "unit_serving_protein_g"]
    table(d[cols].isna().sum().to_frame("missing"))
    out(f"Distinct serving units: {d.servings_unit.nunique()} (top: "
        f"{', '.join(d.servings_unit.value_counts().head(5).index)})")
    out("INDB has no ingredient list in this file, so it is used for nutrition lookup and logging, "
        "not for allergen filtering.")
    out()


def inspect_usda():
    files = list(DATA_RAW.glob("usda_sr_legacy/**/food.csv"))
    out("## 4. USDA FoodData Central SR Legacy (2018-04)")
    if not files:
        out("not present, skipped")
        return
    base = files[0].parent
    food = pd.read_csv(base / "food.csv")
    fn = pd.read_csv(base / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    out(f"{len(food):,} foods, {len(fn):,} food-nutrient rows. All amounts are per 100 g.")
    for nid, name in [(1008, "Energy (kcal)"), (1003, "Protein"), (1005, "Carbohydrate, by difference"),
                      (1004, "Total lipid (fat)")]:
        have = fn[fn.nutrient_id == nid].fdc_id.nunique()
        out(f"- {name}: present for {have:,} of {len(food):,} foods")
    out()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    out("# Data inspection report")
    out()
    out("Generated by scripts/inspect_data.py from the raw files in data/raw/.")
    out()
    inspect_foodcom()
    inspect_majumder()
    inspect_indb()
    inspect_usda()
    (RESULTS_DIR / "data_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote results/data_inspection.md")


if __name__ == "__main__":
    main()
