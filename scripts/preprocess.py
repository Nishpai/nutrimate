"""Build the processed data used by the models and the app.

Input  (data/raw/):  foodrs.duckdb, Anuvaad_INDB_2024.11.xlsx, usda_sr_legacy/
Output (data/processed/):
  recipes.csv.gz         recipe catalogue with cleaned per-serving nutrition and derived fields
  interactions.csv.gz    user-recipe ratings (1 to 5) for collaborative filtering
  foods.csv.gz           per-100 g (and INDB per-serving) foods for the macro tracker
  nutrition_flags.csv    recipes whose listed nutrition we did not trust, with the original values

Main decisions (see METHODOLOGY.md for the reasons):
  * ratings of 0 are reviews without stars, so they are dropped
  * the interaction data is reduced to a 20-user / 15-recipe core so CF has enough signal
  * extra Indian recipes outside the core are added to the catalogue as cold-start items
  * implausible nutrition is set to missing (NaN), never to zero
"""

import html
import json
import re

import _path  # noqa: F401
import duckdb
import numpy as np
import pandas as pd

from nutrimate import constraints as cons
from nutrimate.config import DATA_PROCESSED, DATA_RAW

MIN_USER_RATINGS = 20
MIN_RECIPE_RATINGS = 15
EXTRA_INDIAN = 1500
NON_FOOD = {"Household Cleaner", "Bath/Beauty", "Homeopathy/Remedies"}
# A few non-food "recipes" are filed under food categories, so we also check names.
NON_FOOD_NAME = re.compile(
    r"play ?dough|dog (?:treat|biscuit|cookie)s?|(?:for|doggie) dogs?|dog treats|pumpkin dog|"
    r"frosty paws|cleaner|tub and shower|chews for dogs|good doggie", re.I)


def is_food(name, category, keywords):
    if category in NON_FOOD or NON_FOOD & set(k for k in keywords if k):
        return False
    return not NON_FOOD_NAME.search(name or "")

CUISINES = [  # specific names first, broad regions last
    ("Indian", ["Indian", "Pakistani"]),
    ("Thai", ["Thai"]),
    ("Chinese", ["Chinese", "Szechuan", "Hunan"]),
    ("Japanese", ["Japanese"]),
    ("Korean", ["Korean"]),
    ("Vietnamese", ["Vietnamese"]),
    ("Mexican", ["Mexican", "Tex Mex"]),
    ("Middle Eastern", ["Southwest Asia (middle East)", "Lebanese", "Turkish", "Moroccan", "Egyptian"]),
    ("Greek", ["Greek"]),
    ("Caribbean", ["Caribbean", "Cuban", "Puerto Rican"]),
    ("Cajun/Creole", ["Cajun", "Creole"]),
    ("African", ["African", "South African", "Ethiopian"]),
    ("European", ["European", "German", "Spanish", "Polish", "Swedish", "Scandinavian", "Russian",
                  "Swiss", "Hungarian", "Portuguese", "Danish", "Norwegian", "Scottish", "Belgian",
                  "Dutch", "Czech", "Welsh", "Finnish", "Icelandic"]),
    ("Asian", ["Asian", "Indonesian", "Malaysian", "Filipino"]),
    ("South American", ["South American", "Brazilian", "Peruvian", "Chilean", "Colombian", "Venezuelan"]),
    ("American", ["Southwestern U.S.", "Canadian", "Hawaiian", "Pennsylvania Dutch", "Native American"]),
]

MEAL_TYPES = {
    "breakfast": {"Breakfast", "Brunch", "Scones"},
    "dessert": {"Dessert", "Pie", "Bar Cookie", "Drop Cookies", "Candy", "Frozen Desserts", "Cheesecake",
                "Tarts", "Jellies", "Chocolate Chip Cookies", "Gelatin"},
    "drink": {"Beverages", "Smoothies", "Shakes", "Punch Beverage"},
    "main": {"One Dish Meal", "Chicken", "Chicken Breast", "Chicken Thigh & Leg", "Whole Chicken", "Meat",
             "Pork", "Steak", "Roast Beef", "Stew", "Curries", "Poultry", "Lamb/Sheep", "Ham", "Meatloaf",
             "Tilapia", "Tuna", "Crab", "Whole Turkey", "Turkey Breasts", "Savory Pies", "Spaghetti",
             "Penne", "Pasta Shells", "Gumbo", "Chowders", "Lentil", "Black Beans", "Soy/Tofu", "Manicotti",
             "Macaroni And Cheese", "Halibut", "Catfish", "Trout", "Lobster", "Squid", "Mahi Mahi",
             "Whole Duck", "Duck", "Duck Breasts", "Deer", "Wild Game", "Perch", "Whitefish",
             "Orange Roughy", "Veal", "Moose", "Elk", "Chicken Livers", "Crawfish", "Clear Soup"},
    "snack": {"Lunch/Snacks", "Spreads", "< 15 Mins"},
    "side": {"Vegetable", "Potato", "Rice", "Long Grain Rice", "White Rice", "Brown Rice", "Beans", "Corn",
             "Yam/Sweet Potato", "Cauliflower", "Grains", "Onions", "Spinach", "Greens", "Peppers",
             "Collard Greens", "Chard", "Short Grain Rice", "Medium Grain Rice", "Fruit"},
    "baking": {"Quick Breads", "Breads", "Yeast Breads", "Sourdough Breads"},
    "sauce": {"Sauces", "Salad Dressings", "Stocks"},
}

NUTRIENTS = {  # source column -> our column (all per serving)
    "Calories": "calories_kcal", "ProteinContent": "protein_g", "CarbohydrateContent": "carbs_g",
    "FatContent": "fat_g", "FiberContent": "fiber_g", "SugarContent": "sugar_g",
    "SodiumContent": "sodium_mg", "SaturatedFatContent": "sat_fat_g",
}


def iso_minutes(s):
    """'PT1H30M' -> 90. Returns NaN when missing or unparseable."""
    if not isinstance(s, str):
        return np.nan
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m or not any(m.groups()):
        return np.nan
    h, mi, _ = (int(g) if g else 0 for g in m.groups())
    return h * 60 + mi


def join_list(items, sep="|"):
    """Join a list from the source, skipping empty entries."""
    return sep.join(str(x).strip() for x in (items if items is not None else []) if x)


def pick_cuisine(keywords):
    kw = set(keywords)
    for name, tags in CUISINES:
        if kw & set(tags):
            return name
    return "Unspecified"


def pick_meal_type(category, keywords):
    for mt, cats in MEAL_TYPES.items():
        if category in cats:
            return mt
    kw = set(keywords)
    for mt in ["breakfast", "dessert", "drink", "main"]:
        if kw & MEAL_TYPES[mt]:
            return mt
    return "other"


def clean_nutrition(rec):
    """Copy per-serving nutrition and blank out values that cannot be right.

    Returns the cleaned frame plus a frame of flagged recipes with the original values.
    """
    reasons = pd.Series("", index=rec.index)
    kcal, p, c, f = rec.Calories, rec.ProteinContent, rec.CarbohydrateContent, rec.FatContent
    all_zero = (kcal == 0) & (p == 0) & (c == 0) & (f == 0)
    reasons[all_zero] = "all values zero (calculator probably failed)"
    too_big = (kcal > 2500) | (p > 250) | (c > 400) | (f > 250)
    reasons[too_big & (reasons == "")] = "implausibly large for one serving"
    atwater = 4 * p + 4 * c + 9 * f
    mismatch = (kcal > 50) & ((atwater - kcal).abs() / kcal > 0.5)
    reasons[mismatch & (reasons == "")] = "calories do not match macros (>50% off)"
    flagged = reasons != ""

    out = pd.DataFrame(index=rec.index)
    for src, dst in NUTRIENTS.items():
        out[dst] = rec[src].astype(float).where(~flagged)  # NaN, not zero
    out["nutrition_status"] = np.where(flagged, "flagged_missing", "listed")
    flags = rec.loc[flagged, ["RecipeId", "Name", *NUTRIENTS.keys()]].copy()
    flags["reason"] = reasons[flagged]
    return out, flags


def build_interactions(con, recipe_ids_food):
    rev = con.sql("""SELECT AuthorId AS user_src, RecipeId AS recipe_id, Rating AS rating,
                            DateSubmitted AS ts FROM reviews_raw WHERE Rating > 0""").df()
    rev = rev[rev.recipe_id.isin(recipe_ids_food)]
    rev = rev.sort_values("ts").drop_duplicates(["user_src", "recipe_id"], keep="last")
    d = rev
    for _ in range(30):  # iterate until both thresholds hold at once (k-core)
        before = len(d)
        d = d[d.groupby("user_src").user_src.transform("size") >= MIN_USER_RATINGS]
        d = d[d.groupby("recipe_id").recipe_id.transform("size") >= MIN_RECIPE_RATINGS]
        if len(d) == before:
            break
    # anonymise user ids: u00001, u00002, ... in order of first activity
    order = d.groupby("user_src").ts.min().sort_values().index
    mapping = {u: f"u{i + 1:05d}" for i, u in enumerate(order)}
    d = d.assign(user_id=d.user_src.map(mapping))
    d["ts"] = pd.to_datetime(d.ts, utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return d[["user_id", "recipe_id", "rating", "ts"]].reset_index(drop=True)


def build_recipes(con, core_ids, rating_stats):
    rec = con.sql("SELECT * EXCLUDE (AuthorName, Images, AuthorId) FROM recipes_raw").df()
    rec["RecipeId"] = rec.RecipeId.astype(int)
    rec = rec[[is_food(n, c, k) for n, c, k in zip(rec.Name, rec.RecipeCategory, rec.Keywords)]]

    core = rec[rec.RecipeId.isin(core_ids)]
    is_indian = rec.Keywords.map(lambda k: "Indian" in set(k) or "Pakistani" in set(k))
    extra = rec[is_indian & ~rec.RecipeId.isin(core_ids) & (rec.ReviewCount.fillna(0) >= 3)]
    extra = extra.sort_values("ReviewCount", ascending=False).head(EXTRA_INDIAN)
    rec = pd.concat([core, extra]).reset_index(drop=True)

    nut, flags = clean_nutrition(rec)
    df = pd.DataFrame({
        "recipe_id": rec.RecipeId,
        "name": rec.Name.map(lambda x: html.unescape(x).strip()),
        "category": rec.RecipeCategory.fillna("Unknown"),
        "keywords": rec.Keywords.map(join_list),
        "ingredients": rec.RecipeIngredientParts.map(join_list),
        "instructions": rec.RecipeInstructions.map(lambda k: html.unescape(join_list(k, "\n"))),
        "description": rec.Description.fillna("").map(html.unescape).str.slice(0, 400),
        "total_minutes": rec.TotalTime.map(iso_minutes),
        "servings": rec.RecipeServings.astype("Int64"),
        "year": pd.to_datetime(rec.DatePublished).dt.year,
    })
    df = pd.concat([df, nut], axis=1)
    df["nutrition_basis"] = "per_serving"
    df["cuisine"] = rec.Keywords.map(pick_cuisine)
    df["meal_type"] = [pick_meal_type(c, k) for c, k in zip(rec.RecipeCategory, rec.Keywords)]

    texts = [cons.recipe_text(n, i, s) for n, i, s in zip(df.name, df.ingredients, df.instructions)]
    df["is_eggetarian"] = [not cons.has_meat_or_fish(t) for t in texts]        # no meat or fish
    df["is_vegetarian"] = [v and not cons.has_egg(t) for v, t in zip(df.is_eggetarian, texts)]  # also no egg
    df["is_vegan"] = [v and not cons.has_animal_product(t) for v, t in zip(df.is_vegetarian, texts)]
    df["allergens_found"] = ["|".join(cons.allergens_in(t)) for t in texts]

    df["in_cf_data"] = df.recipe_id.isin(core_ids)
    df = df.merge(rating_stats, on="recipe_id", how="left")
    df["source"] = "Food.com Recipes and Reviews (CC0)"
    df["source_id"] = df.recipe_id.astype(str)
    return df, flags


def build_foods():
    rows = []
    indb = pd.read_excel(DATA_RAW / "Anuvaad_INDB_2024.11.xlsx")
    for _, r in indb.iterrows():
        rows.append({
            "food_id": f"INDB:{r.food_code}", "name": str(r.food_name).strip(),
            "source": "Indian Nutrient Databank (INDB 2024.11)", "source_id": r.food_code,
            "kcal_100g": r.energy_kcal, "protein_100g": r.protein_g, "carbs_100g": r.carb_g,
            "fat_100g": r.fat_g, "fiber_100g": r.fibre_g,
            "serving_unit": r.servings_unit if isinstance(r.servings_unit, str) else None,
            "kcal_serving": r.unit_serving_energy_kcal, "protein_serving": r.unit_serving_protein_g,
            "carbs_serving": r.unit_serving_carb_g, "fat_serving": r.unit_serving_fat_g,
        })
    base = next(DATA_RAW.glob("usda_sr_legacy/**/food.csv")).parent
    food = pd.read_csv(base / "food.csv")
    fn = pd.read_csv(base / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    ids = {1008: "kcal_100g", 1003: "protein_100g", 1005: "carbs_100g", 1004: "fat_100g", 1079: "fiber_100g"}
    wide = (fn[fn.nutrient_id.isin(ids)].pivot_table(index="fdc_id", columns="nutrient_id", values="amount")
            .rename(columns=ids).reset_index())
    usda = food.merge(wide, on="fdc_id", how="left")
    for _, r in usda.iterrows():
        rows.append({
            "food_id": f"USDA:{r.fdc_id}", "name": r.description,
            "source": "USDA FoodData Central SR Legacy (2018)", "source_id": str(r.fdc_id),
            **{c: r.get(c) for c in ids.values()},
            "serving_unit": None, "kcal_serving": np.nan, "protein_serving": np.nan,
            "carbs_serving": np.nan, "fat_serving": np.nan,
        })
    return pd.DataFrame(rows)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DATA_RAW / "foodrs.duckdb"), read_only=True)

    meta = con.sql("SELECT CAST(RecipeId AS INTEGER) AS rid, Name, RecipeCategory, Keywords "
                   "FROM recipes_raw").df()
    keep = [is_food(n, c, k) for n, c, k in zip(meta.Name, meta.RecipeCategory, meta.Keywords)]
    food_ids = set(meta.rid[keep])
    print(f"dropped {len(meta) - len(food_ids):,} non-food entries (cleaners, play dough, pet treats)")
    inter = build_interactions(con, food_ids)
    stats = inter.groupby("recipe_id").rating.agg(n_ratings="size", mean_rating="mean").reset_index()
    stats["mean_rating"] = stats.mean_rating.round(3)

    recipes, flags = build_recipes(con, set(inter.recipe_id), stats)
    recipes["n_ratings"] = recipes.n_ratings.fillna(0).astype(int)
    foods = build_foods()

    recipes.to_csv(DATA_PROCESSED / "recipes.csv.gz", index=False)
    inter.to_csv(DATA_PROCESSED / "interactions.csv.gz", index=False)
    foods.to_csv(DATA_PROCESSED / "foods.csv.gz", index=False)
    flags.to_csv(DATA_PROCESSED / "nutrition_flags.csv", index=False)

    summary = {
        "recipes_in_catalogue": int(len(recipes)),
        "recipes_with_ratings_in_cf_data": int(recipes.in_cf_data.sum()),
        "extra_indian_cold_start_recipes": int((~recipes.in_cf_data).sum()),
        "interactions": int(len(inter)),
        "users": int(inter.user_id.nunique()),
        "density": round(len(inter) / (inter.user_id.nunique() * inter.recipe_id.nunique()), 5),
        "rating_share": inter.rating.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "nutrition_flagged_missing": int((recipes.nutrition_status != "listed").sum()),
        "eggetarian_recipes": int(recipes.is_eggetarian.sum()),
        "vegetarian_recipes": int(recipes.is_vegetarian.sum()),
        "vegan_recipes": int(recipes.is_vegan.sum()),
        "foods_indb": int(foods.food_id.str.startswith("INDB").sum()),
        "foods_usda": int(foods.food_id.str.startswith("USDA").sum()),
        "cuisine_counts": recipes.cuisine.value_counts().to_dict(),
        "meal_type_counts": recipes.meal_type.value_counts().to_dict(),
    }
    (DATA_PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
