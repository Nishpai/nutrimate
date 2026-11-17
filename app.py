"""NutriMate demo app (Streamlit).

Run from the project folder:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "nutrimate" imports from any working dir

from nutrimate import config, tracker
from nutrimate.constraints import GROUP_LABELS, Constraints, normalize_exclusion
from nutrimate.data import load_foods, load_interactions, nutrient
from nutrimate.explain import nutrition_line, reasons, summary_text
from nutrimate.hybrid import Recommender, UserState
from nutrimate.parser import CUISINES, MEAL_TYPES, llm_available, parse, request_preference_text, to_constraints

st.set_page_config(page_title="NutriMate", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .nm-badge {display:inline-block; padding:1px 8px; margin:0 4px 4px 0; border-radius:10px;
               font-size:0.78rem; background:#E8F0EB; color:#1E4D36; border:1px solid #CFE0D6;}
    .nm-badge.warn {background:#FFF4E5; color:#7A4B00; border-color:#F3D7A6;}
    .nm-badge.src {background:#EEF2F8; color:#253B5B; border-color:#D3DCEA;}
    .nm-muted {color:#5C6B63; font-size:0.85rem;}
    .nm-score {font-size:0.8rem; color:#3D4A43;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading recipes and models ...")
def get_recommender():
    return Recommender.load()


@st.cache_resource
def get_db():
    return tracker.connect()


@st.cache_data
def get_foods():
    return load_foods()


@st.cache_data
def get_dataset_users():
    inter = load_interactions()
    counts = inter.groupby("user_id").size().sort_values(ascending=False)
    return inter, counts


rec = get_recommender()
db = get_db()
recipes = rec.recipes

DEFAULT_PROFILE = {
    "diet": "any", "allergens": [], "dislikes": "", "cuisines": [], "liked_ingredients": "",
    "goal": "General balanced eating", "daily_kcal": None, "daily_protein_g": None,
    "daily_carbs_g": None, "daily_fat_g": None, "max_meal_kcal": None, "min_meal_protein_g": None,
}
if "profile" not in st.session_state:
    st.session_state.profile = {**DEFAULT_PROFILE, **(tracker.load_profile(db) or {})}
if "chat" not in st.session_state:
    st.session_state.chat = []
profile = st.session_state.profile


def badge(text, kind=""):
    return f'<span class="nm-badge {kind}">{text}</span>'


def label_exclusion(term):
    key = normalize_exclusion(term)
    return GROUP_LABELS.get(key, key)


def profile_constraints(p) -> Constraints:
    dislikes = [d.strip().lower() for d in p["dislikes"].split(",") if d.strip()]
    return Constraints(diet=p["diet"], exclude=list(dict.fromkeys(p["allergens"] + dislikes)),
                       max_calories=p["max_meal_kcal"], min_protein_g=p["min_meal_protein_g"])


def current_user_state(p) -> UserState:
    liked, disliked = tracker.load_feedback(db)
    demo = st.session_state.get("demo_user")
    if demo:
        inter, _ = get_dataset_users()
        h = inter[inter.user_id == demo]
        liked = list(h[h.rating >= config.RELEVANT_RATING].recipe_id) + liked
        disliked = list(h[h.rating <= 2].recipe_id) + disliked
    text = " ".join([f"cuisine_{c.lower().replace(' ', '_').replace('/', '_')} {c}" for c in p["cuisines"]]
                    + [p["liked_ingredients"].replace(",", " ")])
    return UserState(liked_ids=liked, disliked_ids=disliked, preference_text=text,
                     daily_kcal=p["daily_kcal"], daily_protein_g=p["daily_protein_g"])


def liked_ingredient_set(user: UserState) -> set:
    rows = recipes[recipes.recipe_id.isin(user.liked_ids[-30:])]
    common = {}
    for ing in rows.ingredients:
        for i in ing.lower().split("|"):
            common[i] = common.get(i, 0) + 1
    # ingredients that show up in at least two liked recipes, minus pantry basics
    basics = {"salt", "pepper", "water", "sugar", "butter", "olive oil", "oil", "flour", "eggs", "garlic"}
    return {k for k, v in common.items() if v >= 2 and k not in basics}


def log_recipe(row, servings=1.0):
    tracker.add_log(
        db, item_type="recipe", item_id=row["recipe_id"], item_name=row["name"], source=row["source"],
        basis="per_serving", quantity=servings,
        unit_values={"kcal": nutrient(row["calories_kcal"]), "protein_g": nutrient(row["protein_g"]),
                     "carbs_g": nutrient(row["carbs_g"]), "fat_g": nutrient(row["fat_g"])},
    )


def show_recipe_card(row, c, info, liked_set, key_prefix, compact=False):
    with st.container(border=True):
        left, right = st.columns([3, 1.25])
        with left:
            st.markdown(f"**{row['name']}**")
            tags = [badge(row["cuisine"]) if row["cuisine"] != "Unspecified" else "",
                    badge(row["meal_type"]),
                    badge("vegan") if row["is_vegan"] else (badge("vegetarian") if row["is_vegetarian"] else
                                                            (badge("eggetarian") if row["is_eggetarian"] else "")),
                    badge(f"{row['total_minutes']:.0f} min") if not pd.isna(row["total_minutes"]) else "",
                    badge("Food.com, per serving", "src")]
            if row["nutrition_status"] != "listed":
                tags.append(badge("nutrition missing", "warn"))
            if not row["in_cf_data"]:
                tags.append(badge("cold-start item", "warn"))
            st.markdown("".join(tags), unsafe_allow_html=True)
            m = st.columns(4)
            for col, (lab, field, unit) in zip(m, [("Calories", "calories_kcal", "kcal"), ("Protein", "protein_g", "g"),
                                                   ("Carbs", "carbs_g", "g"), ("Fat", "fat_g", "g")]):
                v = nutrient(row[field])
                col.metric(lab, "n/a" if v is None else f"{v:.0f} {unit}")
        with right:
            cf_label = "Popularity" if info["cf_mode"].startswith("popularity") else "CF"
            goal = row["score_goal"]
            st.markdown(
                f"<div class='nm-score'>Content {row['score_content']:.2f}<br>{cf_label} {row['score_cf']:.2f}<br>"
                f"Hybrid {row['score_hybrid']:.2f}<br>"
                + (f"Goal fit {goal:.2f}<br>" if not pd.isna(goal) else "")
                + f"<b>Final {row['score_final']:.2f}</b></div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            if b1.button("Like", key=f"{key_prefix}_like_{row['recipe_id']}"):
                tracker.add_feedback(db, row["recipe_id"], 1)
                st.toast(f"Liked {row['name']}. Your profile was updated.")
                st.rerun()
            if b2.button("Nope", key=f"{key_prefix}_dis_{row['recipe_id']}"):
                tracker.add_feedback(db, row["recipe_id"], -1)
                st.toast("Noted. Similar recipes will rank lower.")
                st.rerun()
            if b3.button("Log", key=f"{key_prefix}_log_{row['recipe_id']}", help="Log 1 serving for today"):
                log_recipe(row)
                st.toast("Logged 1 serving in today's tracker.")
        with st.expander("Why this, ingredients and steps", expanded=False):
            for r in reasons(row, c, info, liked_set):
                st.markdown(f"- {r}")
            st.caption(nutrition_line(row))
            if row["allergens_found"]:
                st.caption("Allergen groups found by text matching: "
                           + ", ".join(GROUP_LABELS.get(a, a) for a in row["allergens_found"].split("|")))
            if not compact:
                st.markdown("**Ingredients:** " + ", ".join(row["ingredients"].split("|")))
                st.markdown("**Steps:**")
                st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(row["instructions"].split("\n")[:12])))
            st.caption(f"Source: {row['source']}, recipe id {row['source_id']}. Servings listed: "
                       f"{'not given' if pd.isna(row['servings']) else int(row['servings'])}.")


def run_recommend(c: Constraints, user: UserState, k=8, w_goal=None):
    w_cf = st.session_state.get("w_cf", config.DEFAULT_W_CF)
    if w_goal is None:
        w_goal = st.session_state.get("w_goal", config.DEFAULT_W_GOAL)
    use_mmr = st.session_state.get("use_mmr", True)
    out, info = rec.recommend(user, c, k=k, w_cf=w_cf, w_goal=w_goal, use_mmr=use_mmr,
                              mmr_lambda=st.session_state.get("mmr_lambda", config.MMR_LAMBDA))
    if c.cuisine and c.hard_cuisine and len(out) < 3:
        soft = Constraints(**{**c.__dict__, "hard_cuisine": False})
        out, info = rec.recommend(user, soft, k=k, w_cf=w_cf, w_goal=w_goal, use_mmr=use_mmr)
        info["cuisine_relaxed"] = True
    return out, info


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### NutriMate")
    st.caption("Hybrid food recommender with dietary constraints. Student project demo.")
    summary = json.loads((config.DATA_PROCESSED / "summary.json").read_text())
    st.markdown(f"Recipes: **{summary['recipes_in_catalogue']:,}**  \nUsers: **{summary['users']:,}**  \n"
                f"Ratings: **{summary['interactions']:,}**")
    st.divider()
    st.markdown("**Model settings**")
    st.slider("CF weight (w_cf)", 0.0, 1.0, config.DEFAULT_W_CF, 0.05, key="w_cf",
              help="0 = content only, 1 = collaborative only. 0.75 was best on the validation split.")
    st.slider("Goal-fit weight", 0.0, 0.5, config.DEFAULT_W_GOAL, 0.05, key="w_goal",
              help="Only used when daily targets are set in the profile.")
    st.toggle("Diversity re-ranking (MMR)", value=True, key="use_mmr")
    st.slider("MMR lambda", 0.3, 1.0, config.MMR_LAMBDA, 0.05, key="mmr_lambda",
              help="1.0 = pure relevance, lower = more varied results.")
    st.divider()
    st.markdown("**Who is using the app**")
    _, counts = get_dataset_users()
    options = ["New user (cold start)"] + [f"{u} ({n} ratings in dataset)" for u, n in counts.head(25).items()]
    choice = st.selectbox("User", options, help="Pick a dataset user to see collaborative filtering with a real "
                                                "rating history. Their ratings are not shown as likes in the UI.")
    st.session_state.demo_user = None if choice.startswith("New") else choice.split()[0]
    liked, disliked = tracker.load_feedback(db)
    st.caption(f"Your feedback in this app: {len(liked)} likes, {len(disliked)} dislikes")
    if st.button("Clear my feedback"):
        tracker.clear_feedback(db)
        st.rerun()
    st.caption("Chat parser: " + ("LLM (key found) with schema check" if llm_available()
                                  else "rule-based (no API key needed)"))


st.title("NutriMate")
st.markdown("<span class='nm-muted'>Personalized food recommendations using hybrid filtering and dietary "
            "constraints. Nutrition values come from the datasets, never from the chat model.</span>",
            unsafe_allow_html=True)

tabs = st.tabs(["Profile", "Recommendations", "Chat assistant", "Macro tracker", "Evaluation", "Data and sources"])

# ----------------------------------------------------------------------------- profile
with tabs[0]:
    st.subheader("Your preferences")
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            diets = ["any", "eggetarian", "vegetarian", "vegan"]
            diet = st.radio("Diet", diets, index=diets.index(profile["diet"]), horizontal=True,
                            help="vegetarian = no meat, fish or egg. eggetarian = vegetarian plus egg.")
            allergen_keys = list(GROUP_LABELS)
            allergens = st.multiselect("Allergens to avoid (hard rule)", allergen_keys, default=profile["allergens"],
                                       format_func=lambda k: GROUP_LABELS[k])
            dislikes = st.text_input("Disliked ingredients (comma separated, hard rule)", profile["dislikes"],
                                     placeholder="mushroom, olives")
            cuisines = st.multiselect("Preferred cuisines (soft preference)", CUISINES, default=profile["cuisines"])
            liked_ing = st.text_input("Ingredients you enjoy (soft preference)", profile["liked_ingredients"],
                                      placeholder="paneer, chickpeas, spinach")
        with c2:
            goal = st.selectbox("Goal label (only a label, not medical advice)",
                                ["General balanced eating", "Maintain weight", "Muscle gain", "Lighter meals"],
                                index=["General balanced eating", "Maintain weight", "Muscle gain", "Lighter meals"]
                                .index(profile["goal"]))
            st.markdown("Daily targets (leave empty if you do not want one)")
            t1, t2 = st.columns(2)
            daily_kcal = t1.number_input("Calories (kcal)", min_value=800, max_value=6000, value=profile["daily_kcal"], step=50)
            daily_p = t2.number_input("Protein (g)", min_value=10, max_value=400, value=profile["daily_protein_g"], step=5)
            daily_c = t1.number_input("Carbs (g)", min_value=10, max_value=800, value=profile["daily_carbs_g"], step=5)
            daily_f = t2.number_input("Fat (g)", min_value=10, max_value=300, value=profile["daily_fat_g"], step=5)
            st.markdown("Per-meal hard limits (optional)")
            h1, h2 = st.columns(2)
            max_meal = h1.number_input("Max kcal per serving", min_value=50, max_value=3000, value=profile["max_meal_kcal"], step=25)
            min_prot = h2.number_input("Min protein per serving (g)", min_value=0, max_value=200,
                                       value=profile["min_meal_protein_g"], step=5)
        if st.form_submit_button("Save profile", type="primary"):
            profile.update(diet=diet, allergens=allergens, dislikes=dislikes, cuisines=cuisines,
                           liked_ingredients=liked_ing, goal=goal, daily_kcal=daily_kcal, daily_protein_g=daily_p,
                           daily_carbs_g=daily_c, daily_fat_g=daily_f, max_meal_kcal=max_meal, min_meal_protein_g=min_prot)
            tracker.save_profile(db, profile)
            st.success("Profile saved.")
    c = profile_constraints(profile)
    st.caption("Hard rules from this profile: " + ("; ".join(
        [x if not x.startswith("excluding") else "excluding " + ", ".join(label_exclusion(t) for t in c.exclude)
         for x in c.describe()]) or "none") + ". These are applied before ranking and cannot be overridden from chat.")

# ----------------------------------------------------------------------------- recommendations
with tabs[1]:
    st.subheader("Recommended for you")
    f1, f2, f3, f4 = st.columns(4)
    meal = f1.selectbox("Meal type", ["any"] + MEAL_TYPES, index=1 + MEAL_TYPES.index("main"))
    cuisine = f2.selectbox("Cuisine", ["any"] + CUISINES)
    hard_c = f3.checkbox("Only this cuisine", value=False, help="Otherwise the cuisine is a soft boost.")
    max_min = f4.number_input("Max total time (min)", min_value=5, max_value=600, value=None, step=5)
    user = current_user_state(profile)
    c = profile_constraints(profile)
    c.meal_type = None if meal == "any" else meal
    c.cuisine = None if cuisine == "any" else cuisine
    c.hard_cuisine = hard_c and c.cuisine is not None
    c.max_minutes = max_min
    out, info = run_recommend(c, user)
    removed = ", ".join(f"{k} ({v:,})" for k, v in info["removed"].items()) or "nothing"
    st.markdown(
        f"<span class='nm-muted'>{info['n_candidates']:,} recipes passed the hard rules. Removed: {removed}. "
        f"Content signal: {info.get('content_mode')}; CF signal: {info.get('cf_mode')}.</span>",
        unsafe_allow_html=True)
    if info.get("content_mode") == "none":
        st.info("New user with no preferences yet, so this list is by popularity. Add cuisines or liked "
                "ingredients in Profile, or like a few recipes, and the content and CF parts kick in.")
    if out.empty:
        st.warning("No recipe meets all the rules. Try relaxing a limit.")
    liked_set = liked_ingredient_set(user)
    for _, row in out.iterrows():
        show_recipe_card(row, c, info, liked_set, "rec")

# ----------------------------------------------------------------------------- chat
with tabs[2]:
    st.subheader("Ask in plain language")
    st.caption("Example: I want a high-protein vegetarian dinner under 600 calories. Avoid peanuts and suggest "
               "something Indian.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if "parsed" in msg:
                with st.expander("Parsed request (validated JSON)"):
                    st.json(msg["parsed"])
            for r in msg.get("results", []):
                st.markdown(r, unsafe_allow_html=True)
    prompt = st.chat_input("What would you like to eat?")
    if prompt:
        st.session_state.chat.append({"role": "user", "text": prompt})
        parsed = parse(prompt)
        req = parsed["request"]
        reply = {"role": "assistant", "parsed": {**req, "_parser": parsed["parser"], "_notes": parsed["notes"]}}
        if parsed["follow_up"]:
            reply["text"] = parsed["follow_up"]
        else:
            user = current_user_state(profile)
            cc = to_constraints(req, profile_constraints(profile))
            user.prefer_high_protein = req["prefer_high_protein"]
            user.prefer_low_calorie = req["prefer_low_calorie"]
            user.preference_text = (user.preference_text + " " + request_preference_text(req)).strip()
            # an explicit "high protein" or "low calorie" in the message gets a larger goal weight
            explicit = req["prefer_high_protein"] or req["prefer_low_calorie"]
            res, info = run_recommend(cc, user, k=5, w_goal=0.5 if explicit else None)
            text = summary_text(res, cc, info)
            if parsed["notes"]:
                text += "\n\n_" + " ".join(parsed["notes"]) + "_"
            reply["text"] = text
            items = []
            liked_set = liked_ingredient_set(user)
            for i, (_, row) in enumerate(res.iterrows(), 1):
                why = " ".join(reasons(row, cc, info, liked_set)[:4])
                items.append(f"**{i}. {row['name']}**  \n{nutrition_line(row)}  \n<span class='nm-muted'>{why}</span>")
            reply["results"] = items
        st.session_state.chat.append(reply)
        st.rerun()
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ----------------------------------------------------------------------------- tracker
with tabs[3]:
    st.subheader("Daily macro tracker")
    day = st.date_input("Day", value=date.today())
    with st.expander("Add something you ate", expanded=True):
        kind = st.radio("What", ["Recipe from the catalogue (per serving)", "Food from INDB or USDA"], horizontal=True)
        if kind.startswith("Recipe"):
            q = st.text_input("Search recipes", placeholder="dal, paneer, oatmeal ...", key="trk_recipe_q")
            matches = recipes[recipes.name.str.contains(q, case=False, regex=False)] if q else recipes.head(0)
            if q and matches.empty:
                st.caption("No recipe matched.")
            if not matches.empty:
                pick = st.selectbox("Recipe", matches.head(50).index,
                                    format_func=lambda i: recipes.loc[i, "name"])
                row = recipes.loc[pick]
                st.caption(nutrition_line(row))
                servings = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
                if st.button("Add to log", type="primary", key="add_recipe"):
                    tracker.add_log(db, item_type="recipe", item_id=row["recipe_id"], item_name=row["name"],
                                    source=row["source"], basis="per_serving", quantity=servings,
                                    unit_values={"kcal": nutrient(row["calories_kcal"]), "protein_g": nutrient(row["protein_g"]),
                                                 "carbs_g": nutrient(row["carbs_g"]), "fat_g": nutrient(row["fat_g"])},
                                    log_date=day)
                    st.success("Added.")
        else:
            foods = get_foods()
            q = st.text_input("Search foods", placeholder="masoor dal, idli, banana, rice cooked ...", key="trk_food_q")
            matches = foods[foods.name.str.contains(q, case=False, regex=False)] if q else foods.head(0)
            if q and matches.empty:
                st.caption("No food matched.")
            if not matches.empty:
                matches = matches.assign(_indb=matches.food_id.str.startswith("INDB")).sort_values("_indb", ascending=False)
                pick = st.selectbox("Food", matches.head(60).index,
                                    format_func=lambda i: f"{foods.loc[i, 'name']}  [{foods.loc[i, 'source'].split(' (')[0]}]")
                f = foods.loc[pick]
                bases = ["per 100 g (enter grams)"]
                if not pd.isna(f["kcal_serving"]) and isinstance(f["serving_unit"], str):
                    bases.append(f"per serving (1 {f['serving_unit']})")
                basis_label = st.radio("Amount as", bases, horizontal=True)
                if basis_label.startswith("per 100"):
                    qty = st.number_input("Grams", min_value=1.0, max_value=2000.0, value=100.0, step=10.0)
                    basis, units = "per_100g", {"kcal": f["kcal_100g"], "protein_g": f["protein_100g"],
                                                 "carbs_g": f["carbs_100g"], "fat_g": f["fat_100g"]}
                else:
                    qty = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
                    basis, units = "per_serving", {"kcal": f["kcal_serving"], "protein_g": f["protein_serving"],
                                                    "carbs_g": f["carbs_serving"], "fat_g": f["fat_serving"]}
                units = {k: nutrient(v) for k, v in units.items()}
                st.caption(f"Source: {f['source']} (id {f['source_id']}). {basis.replace('_', ' ')}: "
                           + ", ".join(f"{k} {('n/a' if v is None else round(v, 1))}" for k, v in units.items()))
                if st.button("Add to log", type="primary", key="add_food"):
                    tracker.add_log(db, item_type="food", item_id=f["food_id"], item_name=f["name"], source=f["source"],
                                    basis=basis, quantity=qty, unit_values=units, log_date=day)
                    st.success("Added.")

    targets = {"kcal": profile["daily_kcal"], "protein_g": profile["daily_protein_g"],
               "carbs_g": profile["daily_carbs_g"], "fat_g": profile["daily_fat_g"]}
    s = tracker.daily_summary(db, day, targets)
    cols = st.columns(4)
    names = {"kcal": ("Calories", "kcal"), "protein_g": ("Protein", "g"), "carbs_g": ("Carbs", "g"), "fat_g": ("Fat", "g")}
    for col, (m, (lab, unit)) in zip(cols, names.items()):
        t = s["totals"][m]
        col.metric(lab, f"{t['consumed']:.0f} {unit}")
        if t["target"]:
            left = t["remaining"]
            sub = (f"{left:.0f} {unit} left of {t['target']:.0f}" if left >= 0
                   else f"{-left:.0f} {unit} over the {t['target']:.0f} target")
        else:
            sub = "no target set"
        if t["missing_items"]:
            sub += f"; {t['missing_items']} item(s) with unknown value not counted"
        col.caption(sub)
    if s["logs"]:
        chart_rows = []
        for m, (lab, unit) in names.items():
            if m == "kcal":
                continue
            t = s["totals"][m]
            chart_rows.append({"macro": lab, "what": "consumed", "grams": t["consumed"]})
            if t["target"]:
                chart_rows.append({"macro": lab, "what": "target", "grams": t["target"]})
        chart = alt.Chart(pd.DataFrame(chart_rows)).mark_bar().encode(
            x=alt.X("grams:Q", title="grams"), y=alt.Y("what:N", title=None),
            color=alt.Color("what:N", scale=alt.Scale(range=["#2E7D5B", "#C9D6CE"]), legend=None),
            row=alt.Row("macro:N", title=None),
        ).properties(height=45, width=520)
        st.altair_chart(chart)
        st.markdown("**Logged items**")
        for lg in s["logs"]:
            with st.container(border=True):
                a, b, cq, d, e = st.columns([3.2, 2.4, 1.2, 0.8, 0.8])
                unit = "serving(s)" if lg["basis"] == "per_serving" else "g"
                a.markdown(f"{lg['item_name']}  \n<span class='nm-muted'>{lg['source']}, {lg['basis'].replace('_', ' ')}"
                           f"</span>", unsafe_allow_html=True)
                vals = [("kcal", "kcal"), ("protein_g", "P"), ("carbs_g", "C"), ("fat_g", "F")]
                b.markdown(" | ".join(f"{lab} {'n/a' if lg[m] is None else f'{lg[m]:.0f}'}" for m, lab in vals))
                new_q = cq.number_input(unit, value=float(lg["quantity"]), min_value=0.25, step=0.25 if unit != "g" else 10.0,
                                        key=f"q_{lg['id']}", label_visibility="collapsed")
                if d.button("Save", key=f"upd_{lg['id']}"):
                    tracker.update_quantity(db, lg["id"], new_q)
                    st.rerun()
                if e.button("Delete", key=f"del_{lg['id']}"):
                    tracker.delete_log(db, lg["id"])
                    st.rerun()
    else:
        st.caption("Nothing logged for this day yet.")

# ----------------------------------------------------------------------------- evaluation
with tabs[4]:
    st.subheader("Offline evaluation")
    res = config.RESULTS_DIR
    if not (res / "ranking_metrics_test.csv").exists():
        st.warning("Run python scripts/evaluate.py first.")
    else:
        ev = json.loads((res / "summary.json").read_text())
        st.markdown(
            f"Per-user temporal split (train 70%, validation 10%, test 20% of each user's ratings, by date). "
            f"{ev['test_users_evaluated']:,} test users, top-{ev['k']} lists, a rating of {ev['relevant_rating_threshold']}+ "
            f"counts as relevant. Settings chosen on validation: {ev['chosen_svd_factors']} SVD factors, "
            f"w_cf = {ev['chosen_w_cf_on_validation']}.")
        st.markdown("**Top-10 ranking (test split)**")
        st.dataframe(pd.read_csv(res / "ranking_metrics_test.csv").round(4), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.image(str(res / "ranking_comparison.png"))
        c2.image(str(res / "hybrid_weight_sweep.png"))
        st.markdown("**Rating prediction (test split)**")
        st.dataframe(pd.read_csv(res / "rating_metrics_test.csv").round(4), hide_index=True)
        st.markdown("**Cold start: only the first 3 ratings of each user are known**")
        c1, c2 = st.columns([1.4, 1])
        c1.dataframe(pd.read_csv(res / "cold_start_3_ratings.csv").round(4), hide_index=True)
        c2.image(str(res / "cold_start.png"))
        st.markdown("**Constraint check on hybrid top-10 lists (300 sampled test users)**")
        st.dataframe(pd.read_csv(res / "constraint_check.csv").round(4), hide_index=True, use_container_width=True)
        st.caption("Violation = a returned recipe breaks a machine-checkable rule (keyword match, kcal, protein, "
                   "time). Zero with the filter is expected by construction; the useful number is how often an "
                   "unfiltered ranker would break the rule.")
        with st.expander("Validation runs (used for choosing settings)"):
            st.dataframe(pd.read_csv(res / "svd_factor_tuning.csv").round(4), hide_index=True)
            st.dataframe(pd.read_csv(res / "validation_weight_sweep.csv").round(4), hide_index=True)

# ----------------------------------------------------------------------------- data
with tabs[5]:
    st.subheader("Data and sources")
    st.markdown(
        """
| Source | Used for | Basis | Licence / terms |
|---|---|---|---|
| Food.com Recipes and Reviews (Kaggle, A. Irkaal), via the jiava/foodrs-data mirror on Hugging Face | recipe catalogue, ingredients, tags, ratings for CF | nutrition per serving as listed by Food.com | CC0 per the Kaggle page |
| Indian Nutrient Databank (INDB 2024.11, Anuvaad) | macro tracker lookup for Indian dishes | per 100 g and per serving | open access, cite Vijayakumar et al. 2024 |
| USDA FoodData Central, SR Legacy (2018) | macro tracker lookup for basic foods | per 100 g | public domain (CC0) |
        """)
    s1, s2, s3 = st.columns(3)
    s1.metric("Recipes in catalogue", f"{summary['recipes_in_catalogue']:,}")
    s2.metric("Ratings (users x recipes)", f"{summary['interactions']:,}")
    s3.metric("Matrix density", f"{summary['density'] * 100:.2f}%")
    s1.metric("Cold-start Indian recipes", f"{summary['extra_indian_cold_start_recipes']:,}")
    s2.metric("Recipes with nutrition set to missing", f"{summary['nutrition_flagged_missing']:,}")
    s3.metric("Foods for tracking (INDB + USDA)", f"{summary['foods_indb'] + summary['foods_usda']:,}")
    st.markdown("**Rating distribution in the subset**")
    rs = pd.DataFrame({"rating": list(summary["rating_share"].keys()), "share": list(summary["rating_share"].values())})
    st.altair_chart(alt.Chart(rs).mark_bar(color="#2E7D5B").encode(x="rating:N", y=alt.Y("share:Q", axis=alt.Axis(format="%")))
                    .properties(height=200, width=380))
    insp = config.RESULTS_DIR / "data_inspection.md"
    if insp.exists():
        with st.expander("Raw data inspection report (Milestone 0)"):
            st.markdown(insp.read_text(encoding="utf-8"))
