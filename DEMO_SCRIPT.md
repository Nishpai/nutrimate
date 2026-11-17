# Demo script (about 5 minutes)

## Before you start

1. Open PowerShell in the project folder, activate the venv and run `streamlit run app.py`.
2. In the sidebar: User = "New user (cold start)", click "Clear my feedback". Leave the sliders at their defaults
   (w_cf 0.75, goal 0.15, MMR on, lambda 0.7).
3. Keep METHODOLOGY.md open in another window in case the teacher asks about formulas.

## 1. The idea (30 seconds, no clicking)

"NutriMate recommends recipes that fit a person's taste, diet and nutrition targets, and tracks macros.
We split the work so each part does what it can be trusted with: a hybrid ML model ranks recipes, the
datasets give every nutrition number, a rule layer enforces allergies and diet, and the chat part only
understands the request and explains the result. The language part never makes up calories."

## 2. Data and sources tab (40 seconds)

Open **Data and sources**.

- "We use Food.com recipes and about 226,000 real ratings from 3,666 users, plus INDB for Indian dishes
  and USDA for basic foods."
- Point at the matrix density (0.92%) and the rating chart: "80% of ratings are 5 stars, so the data is
  very sparse and skewed. That shaped our evaluation."
- Open the inspection report expander: "Before training we inspected the raw files. The other Food.com
  version stores nutrition as percent daily value, so we picked the version with grams per serving. We set
  176 implausible nutrition values to missing instead of zero."

## 3. Profile (40 seconds)

Open **Profile**. Set: Diet = vegetarian, Allergens = peanut, Disliked = mushroom, Cuisine = Indian,
Ingredients you enjoy = paneer, chickpeas, spinach, lentils. Daily targets 2200 kcal and 100 g protein.
Click **Save profile**.

"Allergens, diet and dislikes are hard rules. Cuisine and liked ingredients are soft preferences that
only affect ranking. Vegetarian here follows the Indian meaning, so no egg; there is a separate
eggetarian option."

## 4. Recommendations (70 seconds)

Open **Recommendations**.

- Read the grey line: "about this many recipes passed the rules, and here is how many each rule removed."
- On the first card point at the badges (source and per-serving basis), the four nutrition numbers and the
  score breakdown on the right: "content score from TF-IDF similarity, popularity because I am a new user
  with no likes yet, goal fit from my targets, and the final score."
- Open **Why this**: "every line comes from the recipe record. For allergens it says not found in the
  listed text, not safe, because it is keyword matching."
- Click **Like** on two or three Indian dishes. The grey line changes to "CF signal: collaborative +
  popularity". "My likes are folded into the SVD model immediately, no retraining."
- Optional: change the sidebar user to a dataset user to show a pure collaborative list.

## 5. Chat assistant (60 seconds)

Open **Chat assistant** and type:

`I want a high-protein vegetarian dinner under 600 calories. Avoid peanuts and suggest something Indian.`

- Open **Parsed request (validated JSON)**: "the message becomes this JSON, which is checked against a
  schema before the backend uses it. It works without any API key; an LLM can be switched on with a key,
  and its output has to pass the same schema."
- Point at the note about dinner mapping to main dishes: "when it assumes something, it says so."
- Then type `something under 700`: "no unit, so it asks instead of guessing."

## 6. Macro tracker (40 seconds)

Open **Macro tracker**. Choose "Food from INDB or USDA", search `idli`, pick per serving, 3 servings, Add.
Then search a recipe like `dal`, 1 serving, Add.

"Per-100 g and per-serving values are never mixed. Each log keeps its basis and source, you can edit or
delete it, and if a value is unknown it is shown as not counted rather than zero."

## 7. Evaluation (50 seconds)

Open **Evaluation**.

- "Temporal split per user: older 70% train, 10% validation, newest 20% test."
- "CF beats popularity slightly with six times the coverage. Content alone is weak on this data. The hybrid
  at 0.75 was best on validation; on test it matches CF on NDCG and covers about 48% more of the catalogue,
  and MMR makes lists more varied without losing accuracy."
- "Without the constraint layer, 70 to 96 percent of top-10 items would break the user's rules. With it, zero."
- "Matrix factorisation did not beat the bias baseline on RMSE because almost every rating is 5. We report
  that as it is."

## Likely questions

- **Is it accurate?** Precision@10 is about 0.024, roughly 16 times a random ranker, over about 7,900 recipes.
  It is an offline measure on old reviews, not a user study.
- **Why not train an LLM on nutrition data?** Language models can invent numbers. Here every number comes
  from a dataset with its source and basis; the LLM part is optional and only fills request fields.
- **Is it allergy safe?** No system built on user-written recipe text can promise that. It removes recipes
  that mention the allergen in the name, ingredients or steps, and tells the user to verify.
- **Why these weights?** w_cf and the SVD size were picked on a validation split. The goal weight and the
  cold-start popularity weight are design choices we have not tuned yet; that is listed as future work.
- **Why Streamlit, not React?** For the demo it keeps everything in one Python process. All logic is in the
  `nutrimate` package, so a FastAPI + React version can reuse it unchanged.
