"""The hard filter must never let an excluded thing through."""

import pandas as pd
import pytest

from nutrimate import constraints as cons
from nutrimate.constraints import Constraints, check_recipe, filter_recipes
from nutrimate.hybrid import UserState


def test_keyword_edge_cases():
    assert cons.has_egg("2 eggs, flour")
    assert not cons.has_egg("roasted eggplant with garlic")
    assert not cons.allergens_in("peanut butter cookies").count("dairy")
    assert "peanut" in cons.allergens_in("peanut butter cookies")
    assert not cons.term_matches("a pinch of nutmeg", "nuts")
    assert cons.term_matches("chopped walnuts", "nuts")
    assert cons.term_matches("1 cup cashews", "tree_nut")
    assert cons.has_meat_or_fish("chicken broth, rice")
    assert not cons.has_meat_or_fish("vegetable broth, rice")
    assert cons.term_matches("2 cups all-purpose flour", "gluten")
    assert not cons.term_matches("1 cup rice flour", "gluten")


@pytest.mark.parametrize("term", ["peanut", "mushroom", "dairy", "egg", "sesame", "nuts"])
def test_filter_never_leaks_excluded_ingredient(recipes, term):
    kept, removed = filter_recipes(recipes, Constraints(exclude=[term]))
    assert len(kept) > 0
    assert removed  # something was actually removed
    leaks = kept.search_text.map(lambda t: cons.term_matches(t, term))
    assert not leaks.any()
    # simple substring check as a second, independent look (only for plain words)
    if term in ("peanut", "mushroom", "sesame"):
        assert not kept.search_text.str.contains(term, regex=False).any()


@pytest.mark.parametrize("diet", ["eggetarian", "vegetarian", "vegan"])
def test_diet_filter(recipes, diet):
    kept, _ = filter_recipes(recipes, Constraints(diet=diet))
    for word in ["chicken", "beef", "pork", "shrimp", "bacon", "salmon"]:
        assert not kept.search_text.str.contains(rf"\b{word}\b", regex=True).any(), word
    if diet in ("vegetarian", "vegan"):
        assert not kept.search_text.map(cons.has_egg).any()
    if diet == "vegan":
        assert not kept.search_text.map(cons.has_animal_product).any()


def test_numeric_limits_and_unknown_values(recipes):
    c = Constraints(max_calories=500, min_protein_g=20)
    kept, removed = filter_recipes(recipes, c)
    assert (kept.calories_kcal <= 500).all()
    assert (kept.protein_g >= 20).all()
    assert kept.calories_kcal.notna().all()  # unknown calories cannot pass a calorie limit
    assert removed.get("calories unknown", 0) > 0


def test_filter_and_row_check_agree(recipes):
    c = Constraints(diet="vegetarian", exclude=["peanut", "gluten"], max_calories=600)
    kept, _ = filter_recipes(recipes, c)
    sample = kept.sample(min(300, len(kept)), random_state=0)
    assert all(check_recipe(r, c) == [] for _, r in sample.iterrows())


def test_recommender_output_respects_constraints(recommender):
    c = Constraints(diet="vegan", exclude=["soy", "nuts"], max_calories=700)
    out, info = recommender.recommend(UserState(preference_text="Indian curry chickpea"), c, k=10)
    assert len(out) == 10
    assert all(check_recipe(r, c) == [] for _, r in out.iterrows())
