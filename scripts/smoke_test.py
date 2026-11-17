"""Quick end-to-end check of the main functions without the UI.

python scripts/smoke_test.py
"""

import tempfile
from datetime import date
from pathlib import Path

import _path  # noqa: F401

from nutrimate import tracker
from nutrimate.constraints import Constraints, check_recipe
from nutrimate.data import load_interactions, nutrient
from nutrimate.explain import reasons, summary_text
from nutrimate.hybrid import Recommender, UserState
from nutrimate.parser import parse, request_preference_text, to_constraints


def show(title, out, info, c):
    print(f"\n== {title}")
    print(f"   candidates after rules: {info['n_candidates']}, content: {info.get('content_mode')}, "
          f"cf: {info.get('cf_mode')}")
    for _, r in out.head(5).iterrows():
        kcal = nutrient(r.calories_kcal)
        print(f"   {r['name'][:45]:45s} kcal={'n/a' if kcal is None else round(kcal)} "
              f"content={r.score_content:.2f} cf={r.score_cf:.2f} final={r.score_final:.2f}")
        assert check_recipe(r, c) == [], "constraint violated"


def main():
    rec = Recommender.load()

    c = Constraints()
    out, info = rec.recommend(UserState(), c, k=5)
    show("new user, no preferences (popularity fallback)", out, info, c)

    c = Constraints(diet="vegetarian", exclude=["peanut"], max_calories=600)
    u = UserState(preference_text="cuisine_indian Indian paneer chickpeas spinach")
    out, info = rec.recommend(u, c, k=5)
    show("new user with typed preferences, vegetarian, no peanut, <= 600 kcal", out, info, c)

    inter = load_interactions()
    uid = inter.user_id.value_counts().index[0]
    h = inter[inter.user_id == uid]
    u = UserState(liked_ids=list(h[h.rating >= 4].recipe_id), disliked_ids=list(h[h.rating <= 2].recipe_id))
    c = Constraints(meal_type="main")
    out, info = rec.recommend(u, c, k=5)
    show(f"dataset user {uid} with {len(h)} ratings (collaborative)", out, info, c)

    msg = "I want a high-protein vegetarian dinner under 600 calories. Avoid peanuts and suggest something Indian."
    p = parse(msg)
    print("\n== chat parse:", p["parser"], p["request"])
    c = to_constraints(p["request"])
    u = UserState(prefer_high_protein=True, preference_text=request_preference_text(p["request"]))
    out, info = rec.recommend(u, c, k=5, w_goal=0.5)
    show("chat request", out, info, c)
    print("   reply:", summary_text(out, c, info))
    print("   why #1:", reasons(out.iloc[0], c, info))

    with tempfile.TemporaryDirectory() as d:
        con = tracker.connect(Path(d) / "smoke.db")
        r = out.iloc[0]
        tracker.add_log(con, item_type="recipe", item_id=r.recipe_id, item_name=r["name"], source=r.source,
                        basis="per_serving", quantity=1.5,
                        unit_values={"kcal": nutrient(r.calories_kcal), "protein_g": nutrient(r.protein_g),
                                     "carbs_g": nutrient(r.carbs_g), "fat_g": nutrient(r.fat_g)})
        s = tracker.daily_summary(con, date.today(), {"kcal": 2000, "protein_g": 90})
        print("\n== tracker after logging 1.5 servings:",
              {k: round(v["consumed"], 1) for k, v in s["totals"].items()},
              "remaining kcal:", round(s["totals"]["kcal"]["remaining"], 1))
        con.close()
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
