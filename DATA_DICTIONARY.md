# Data dictionary

All processed files are in `data/processed/` and are rebuilt by `scripts/preprocess.py`.
Empty cells mean the value is unknown. We never fill an unknown nutrient with zero.

## recipes.csv.gz (7,903 rows, one per recipe)

| Column | Type | Meaning |
|---|---|---|
| recipe_id | int | Food.com RecipeId (kept from the source so joins stay traceable) |
| name | text | Recipe name (HTML entities decoded) |
| category | text | Food.com RecipeCategory |
| keywords | text, `\|` separated | Food.com keyword tags (for example `Indian\|Vegetable\|< 60 Mins`) |
| ingredients | text, `\|` separated | Ingredient names as listed (RecipeIngredientParts). Quantities are not used |
| instructions | text, newline separated | Recipe steps |
| description | text | First 400 characters of the description |
| total_minutes | float | Parsed from the ISO 8601 TotalTime field (PT1H30M becomes 90) |
| servings | int | RecipeServings, missing for about a third of recipes |
| year | int | Year published |
| calories_kcal | float | kcal per serving |
| protein_g, carbs_g, fat_g, fiber_g, sugar_g, sat_fat_g | float | grams per serving |
| sodium_mg | float | mg per serving |
| nutrition_status | text | `listed` (value from Food.com) or `flagged_missing` (failed our checks, set to empty) |
| nutrition_basis | text | Always `per_serving` for recipes |
| cuisine | text | Derived from keywords (Indian, Thai, Mexican, European, ... or Unspecified) |
| meal_type | text | Derived from category: breakfast, main, side, snack, dessert, drink, baking, sauce, other |
| is_eggetarian | bool | No meat or fish word found in name, ingredients or steps |
| is_vegetarian | bool | Also no egg (Indian convention for "vegetarian") |
| is_vegan | bool | Also no dairy, egg or honey |
| allergens_found | text, `\|` separated | Allergen groups found by keyword matching (peanut, tree_nut, dairy, egg, gluten, soy, fish, shellfish, sesame) |
| in_cf_data | bool | True if the recipe has ratings in interactions.csv.gz; False for the extra Indian cold-start recipes |
| n_ratings, mean_rating | int, float | Ratings inside our subset (not the full Food.com count) |
| source, source_id | text | Provenance of the record |

## interactions.csv.gz (225,842 rows)

| Column | Type | Meaning |
|---|---|---|
| user_id | text | Anonymised id (u00001 ...). Original Food.com author ids and names are not stored |
| recipe_id | int | Joins to recipes.recipe_id |
| rating | int | 1 to 5 stars (reviews with 0 stars were dropped, they had no rating) |
| ts | datetime (UTC) | When the review was submitted, used for the temporal split |

## foods.csv.gz (8,807 rows) for the macro tracker

| Column | Type | Meaning |
|---|---|---|
| food_id | text | `INDB:<food_code>` or `USDA:<fdc_id>` |
| name | text | Food name from the source |
| source, source_id | text | INDB 2024.11 or USDA FoodData Central SR Legacy (2018) |
| kcal_100g, protein_100g, carbs_100g, fat_100g, fiber_100g | float | Per 100 g |
| serving_unit | text | INDB only, for example `bowl`, `piece`, `idli` |
| kcal_serving, protein_serving, carbs_serving, fat_serving | float | INDB only, per one `serving_unit`. Empty for 82 INDB rows and all USDA rows |

## nutrition_flags.csv

Recipes whose listed nutrition we did not trust, with the original source values and the reason
(implausibly large for one serving, calories not matching 4/4/9 times the macros by more than 50%,
or every value zero).

## split.csv.gz

Written by `scripts/evaluate.py`: the train / val / test label of every interaction.
