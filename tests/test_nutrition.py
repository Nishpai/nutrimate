"""Unknown nutrients must stay unknown (None/NaN), never become zero."""

import math
from datetime import date

import pandas as pd

from nutrimate import tracker
from nutrimate.data import nutrient, recipe_nutrition
from nutrimate.explain import nutrition_line


def test_flagged_recipes_have_no_numbers(recipes):
    flagged = recipes[recipes.nutrition_status != "listed"]
    assert len(flagged) > 0
    assert flagged.calories_kcal.isna().all()
    assert flagged.protein_g.isna().all()
    row = flagged.iloc[0]
    n = recipe_nutrition(row)
    assert n["calories_kcal"] is None and n["protein_g"] is None
    assert "not available" in nutrition_line(row)


def test_nutrient_helper():
    assert nutrient(float("nan")) is None
    assert nutrient(None) is None
    assert nutrient(0) == 0.0  # a real zero stays a zero


def test_tracker_math_and_missing(tmp_path):
    con = tracker.connect(tmp_path / "t.db")
    d = date(2026, 1, 1)
    # per 100 g: 150 g of something with 20 g protein / 100 g -> 30 g
    tracker.add_log(con, item_type="food", item_id="X", item_name="test food", source="test",
                    basis="per_100g", quantity=150, unit_values={"kcal": 100, "protein_g": 20, "carbs_g": 0, "fat_g": 1},
                    log_date=d)
    # per serving: 1.5 servings with unknown protein
    tracker.add_log(con, item_type="recipe", item_id="Y", item_name="test recipe", source="test",
                    basis="per_serving", quantity=1.5, unit_values={"kcal": 400, "protein_g": None,
                                                                     "carbs_g": 50, "fat_g": 10}, log_date=d)
    s = tracker.daily_summary(con, d, {"kcal": 2000, "protein_g": 100, "carbs_g": None, "fat_g": None})
    assert math.isclose(s["totals"]["kcal"]["consumed"], 150 + 600)
    assert math.isclose(s["totals"]["protein_g"]["consumed"], 30)
    assert s["totals"]["protein_g"]["missing_items"] == 1
    assert s["logs"][1]["protein_g"] is None
    assert math.isclose(s["totals"]["kcal"]["remaining"], 1250)
    assert s["totals"]["carbs_g"]["remaining"] is None  # no target, no remaining
    log_id = s["logs"][0]["id"]
    tracker.update_quantity(con, log_id, 50)
    s = tracker.daily_summary(con, d, {})
    assert math.isclose(s["logs"][0]["protein_g"], 10)
    tracker.delete_log(con, log_id)
    assert len(tracker.logs_for_day(con, d)) == 1


def test_foods_keep_basis_and_source():
    from nutrimate.data import load_foods
    foods = load_foods()
    assert set(foods.source.str.split(" ").str[0]) == {"Indian", "USDA"}
    indb = foods[foods.food_id.str.startswith("INDB")]
    assert indb.kcal_100g.notna().all()
    # INDB rows without serving data keep NaN, they are not filled with zero
    assert indb.kcal_serving.isna().sum() > 0
    assert not (indb.kcal_serving == 0).any()
