"""Template explanations built only from fields the backend returned.

Nothing here makes up a number. If a value is missing we say it is missing.
"""

from __future__ import annotations

import pandas as pd

from .constraints import Constraints
from .data import nutrient


def fmt(value, unit, digits=0):
    return "not available" if value is None else f"{value:.{digits}f} {unit}"


def nutrition_line(row) -> str:
    kcal, p = nutrient(row["calories_kcal"]), nutrient(row["protein_g"])
    c, f = nutrient(row["carbs_g"]), nutrient(row["fat_g"])
    if kcal is None:
        return "Nutrition: not available (the listed values failed our plausibility check)."
    return (f"Per serving: {fmt(kcal, 'kcal')}, protein {fmt(p, 'g', 1)}, carbs {fmt(c, 'g', 1)}, "
            f"fat {fmt(f, 'g', 1)} (Food.com listed values).")


def reasons(row, c: Constraints, info: dict, liked_ingredients: set | None = None) -> list[str]:
    out = []
    if c.diet == "vegan":
        out.append("No meat, fish, dairy, egg or honey found in the listed ingredients and steps.")
    elif c.diet == "vegetarian":
        out.append("No meat, fish or egg found in the listed ingredients and steps.")
    elif c.diet == "eggetarian":
        out.append("No meat or fish found in the listed ingredients and steps.")
    if c.exclude:
        out.append(f"Does not mention {', '.join(c.exclude)} in the listed ingredients or steps "
                   "(text check only, verify if you have an allergy).")
    kcal, prot = nutrient(row["calories_kcal"]), nutrient(row["protein_g"])
    if c.max_calories is not None and kcal is not None:
        out.append(f"{kcal:.0f} kcal per serving, within your {c.max_calories:g} kcal limit.")
    if c.min_protein_g is not None and prot is not None:
        out.append(f"{prot:.1f} g protein per serving, meets your {c.min_protein_g:g} g minimum.")
    if c.max_minutes is not None and not pd.isna(row["total_minutes"]):
        out.append(f"Listed total time {row['total_minutes']:.0f} min.")
    if c.cuisine and row["cuisine"] == c.cuisine:
        out.append(f"Tagged {c.cuisine} on Food.com.")
    if "High Protein" in row["keywords"].split("|"):
        out.append("Tagged High Protein on Food.com.")
    if liked_ingredients:
        common = sorted(liked_ingredients & set(row["ingredients"].lower().split("|")))
        if common:
            out.append("Shares ingredients with recipes you liked: " + ", ".join(common[:4]) + ".")
    mode = info.get("cf_mode", "")
    if mode.startswith("collaborative") and row.get("score_cf", 0) >= 0.7:
        out.append("Users with a rating history similar to yours rated it highly.")
    if row["n_ratings"] > 0:
        out.append(f"Rated by {row['n_ratings']} users in our data subset, average "
                   f"{row['mean_rating']:.2f} / 5.")
    elif not row["in_cf_data"]:
        out.append("New to the model (no ratings in our subset), so it was ranked on content only.")
    return out


def summary_text(results: pd.DataFrame, c: Constraints, info: dict) -> str:
    """Short reply for the chat tab. Uses only the returned rows."""
    if results.empty:
        removed = ", ".join(f"{k}: {v}" for k, v in info.get("removed", {}).items())
        return ("I could not find a recipe that meets all of those rules in the catalogue. "
                f"Recipes removed by each rule: {removed or 'none'}. Try loosening one limit.")
    rules = "; ".join(c.describe()) or "no hard rules"
    top = results.iloc[0]
    kcal = nutrient(top["calories_kcal"])
    prot = nutrient(top["protein_g"])
    lines = [f"I searched {info['n_candidates']:,} recipes that pass your rules ({rules}) and ranked them "
             f"with the hybrid model ({info['cf_mode']}).",
             f"Top pick: **{top['name']}**, {fmt(kcal, 'kcal')} and {fmt(prot, 'g', 1)} protein per serving "
             f"according to {top['source']}."]
    if info.get("cuisine_relaxed"):
        lines.append(f"Only a few {c.cuisine} recipes passed the other rules, so I also included other cuisines.")
    return " ".join(lines)
