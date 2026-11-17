# NutriMate: Methodology

Ninad N Hebbar, Nishant Satish Pai
Department of Information Science and Engineering, M S Ramaiah Institute of Technology, Bengaluru

Every number in this document was produced by the scripts in this repository (`scripts/inspect_data.py`,
`scripts/preprocess.py`, `scripts/evaluate.py`). Re-running them gives the same values; the random seed is fixed at 42.

---

## 1. Problem statement

People who want to eat according to a diet or a nutrition goal have two problems at once. Generic recipe
sites do not know their taste, and generic recommenders do not know their restrictions. An allergy or a
vegetarian diet is not a preference that can be traded off against a high score; it is a rule. Calorie and
protein targets are numbers that must come from real data, not from a language model's guess.

NutriMate addresses this with four separate parts:

1. a structured data layer that holds recipe and food nutrition with its unit basis and source,
2. a hybrid recommender (content-based plus collaborative filtering) that ranks recipes,
3. a deterministic constraint layer that removes recipes breaking the user's rules before ranking,
4. a conversational layer that converts natural language into a validated request and explains results
   using only the fields the backend returned.

### Scope of this version

In scope: preference profile, hybrid recommendation, hard constraints, cold-start handling, diversity,
chat request parsing with explanations, daily macro tracking, offline evaluation.

Out of scope (on purpose): medical or disease-specific advice, automatic calorie target calculation,
image-based food recognition, wearables, training or fine-tuning a language model.

---

## 2. Data sources

We kept nutrient data and interaction data apart and used each source only for what it can support.

| Source | Size (raw) | What it gives | How we use it |
|---|---|---|---|
| Food.com Recipes and Reviews (Kaggle, A. Irkaal; CC0). Downloaded without login from the `jiava/foodrs-data` Hugging Face dataset, which stores the untouched `recipes_raw` and `reviews_raw` tables in a DuckDB file | 522,517 recipes, 1,401,982 reviews | ingredients, steps, tags, category, time, **kcal and grams per serving**, user ratings with timestamps | recipe catalogue, content features, ratings for CF, per-serving nutrition |
| Food.com RAW_recipes / RAW_interactions (Majumder et al., EMNLP 2019), course mirror on Google Drive | 83,782 recipes, 731,927 interactions | same kind of data | inspected only, see below |
| Indian Nutrient Databank, INDB 2024.11 (Anuvaad; Vijayakumar et al., 2024) | 1,014 recipes, 82 columns | nutrients **per 100 g and per serving** of Indian dishes | macro tracker lookup |
| USDA FoodData Central, SR Legacy (April 2018); public domain | 7,793 foods | nutrients **per 100 g** | macro tracker lookup for basic foods |

**Why not the Majumder version for nutrition.** Its `nutrition` column is a list of seven numbers: calories
in kcal followed by six percent daily value (%DV) figures (fat, sugar, sodium, protein, saturated fat,
carbohydrates). Turning %DV into grams needs the reference daily values the site used, which the dataset does
not state. The Irkaal version stores the same site's nutrition directly in kcal and grams per serving. As a
consistency check, the median of (4 x protein + 4 x carbs + 9 x fat) / calories over recipes above 50 kcal is
**1.014**, so the calories and macros describe the same serving.

**Why INDB and USDA are not in the recommender.** INDB has no ingredient list in the released file, so
allergen and diet rules cannot be checked for its dishes, and neither source has user ratings. They are used
where they are strong: reliable per-100 g and per-serving values for logging what was eaten.

---

## 3. Preprocessing (`scripts/preprocess.py`)

1. **Non-food entries removed.** Food.com contains cleaning products, bath and beauty items, home remedies,
   play dough and pet treats. We dropped 1,481 entries by category, keyword and a short name pattern.
2. **Ratings.** 76,248 reviews have rating 0, which on Food.com means "review without stars". They are
   not a rating of zero, so they were dropped. Duplicate (user, recipe) pairs keep the latest review.
3. **Dense core for CF.** Collaborative filtering needs enough ratings per user and per item. We kept a
   20-user / 15-recipe core: users with at least 20 ratings and recipes with at least 15, iterated until both
   hold together. Result: **225,842 ratings, 3,666 users, 6,721 recipes, density 0.92%**.
4. **Indian coverage.** Food.com is US-centred. We added the 1,182 most reviewed Indian or Pakistani tagged
   recipes (at least 3 reviews on the site) that are not in the core. They have no ratings in our subset, so
   they are genuine item cold-start cases and are ranked by content and a neutral popularity prior.
   Final catalogue: **7,903 recipes**.
5. **Nutrition cleaning.** Values were copied per serving. A recipe's nutrition was set to missing (NaN),
   never zero, when it failed a check: implausibly large for one serving (kcal above 2,500 or protein above
   250 g, carbs above 400 g, fat above 250 g; 111 recipes), calories more than 50% away from the 4/4/9 estimate
   (33 recipes), or every value exactly zero (32 recipes). The original values are kept in
   `nutrition_flags.csv`. In total 176 recipes show "nutrition not available".
6. **Derived fields.** Total time parsed from ISO 8601 (`PT1H30M` to 90 min). Cuisine from keyword tags
   (16 groups plus Unspecified; Food.com has no Italian tag in this data). Meal type from the category.
   Diet flags and allergen groups from keyword matching (section 7). HTML entities in names decoded.
7. **Privacy.** Food.com author names and ids are not stored. Users are renumbered `u00001`, `u00002`, ...
8. **Foods table.** INDB rows keep both bases (per 100 g and per one serving unit such as "bowl"); 82 INDB
   rows have no per-serving values and stay empty. USDA rows are per 100 g (energy nutrient 1008, protein
   1003, carbohydrate by difference 1005, total fat 1004, fibre 1079).

Subset facts: 80.1% of ratings are 5 stars, 15.4% are 4, and only 4.4% are 3 or lower. This skew matters
for how the evaluation is set up (section 9).

---

## 4. Content-based model (`nutrimate/content.py`)

Each recipe $i$ becomes a document: its name, ingredient names and tags written as tokens
(`tag_indian`, `cat_one_dish_meal`, `cuisine_indian`). A TF-IDF vectorizer with unigrams and bigrams,
`min_df = 3`, sublinear term frequency and English stop words gives a vocabulary of 11,174 terms
(each row is then L2-normalised):

$$\text{tfidf}(t, i) = (1 + \ln tf_{t,i}) \cdot \Big(\ln\frac{1 + N}{1 + df_t} + 1\Big)$$

Seven nutrients per serving (kcal, protein, carbs, fat, fibre, sugar, sodium) are log-transformed and
standardised, $z = (\log(1 + x) - \mu) / \sigma$, then scaled by $a / \sqrt{7}$ with $a = 0.5$ so the
nutrient block cannot outweigh the text block. Recipes with unknown nutrition get the neutral vector 0 in
this block (only for similarity; the value is never displayed). The item vector is

$$\mathbf{x}_i = \text{normalize}\big([\,\text{tfidf}_i \;|\; \tfrac{a}{\sqrt 7} z_i\,]\big), \qquad \dim = 11{,}181$$

A user profile is a weighted mean of item vectors: weight +1 for liked recipes, -0.5 for disliked ones,
plus the TF-IDF vector of typed preferences (cuisines, liked ingredients) with weight 0.7. The content
score is cosine similarity, $s^{cb}_{u,i} = \mathbf{p}_u \cdot \mathbf{x}_i$.

---

## 5. Collaborative filtering (`nutrimate/collab.py`)

**Ranking model: implicit SVD ("PureSVD").** Let $R$ be the binary user by recipe matrix (1 if the user
rated the recipe). A truncated SVD $R \approx U \Sigma V^T$ with $k$ factors gives item factors $V$. A user's
scores are obtained by folding in their history vector $\mathbf{r}_u$:

$$\mathbf{s}^{cf}_u = \mathbf{r}_u V V^T$$

Because $\mathbf{r}_u$ is just the list of liked recipes, the same formula works for a new app user after a
few likes, without retraining. $k$ was chosen on the validation split from {16, 32, 64, 128}; $k = 16$ was best
(validation NDCG@10 0.0203, 0.0189, 0.0152, 0.0135).

**Rating model: biased matrix factorisation.** For RMSE we also predict explicit stars:

$$\hat r_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

trained by stochastic gradient descent on squared error with L2 regularisation (32 factors, learning rate
0.01, $\lambda = 0.05$, 25 epochs):

$$e = r_{ui} - \hat r_{ui},\quad b_u \mathrel{+}= \eta(e - \lambda b_u),\quad
\mathbf{p}_u \mathrel{+}= \eta(e\,\mathbf{q}_i - \lambda \mathbf{p}_u),\quad
\mathbf{q}_i \mathrel{+}= \eta(e\,\mathbf{p}_u - \lambda \mathbf{q}_i)$$

Both models are written with numpy, scipy and scikit-learn only, so they install on Windows without a compiler.

---

## 6. Hybrid scoring, goal fit and cold start (`nutrimate/hybrid.py`)

Scoring happens only on recipes that already passed the hard constraints. Each component is min-max
normalised over that candidate set, written $\tilde s$:

$$h_{u,i} = (1 - w_{cf})\,\tilde s^{cb}_{u,i} + w_{cf}\,\tilde s^{cf}_{u,i}$$

$$\text{final}_{u,i} = (1 - w_g)\,h_{u,i} + w_g\, g_{u,i} \;+\; 0.1 \cdot [\text{cuisine matches a soft preference}]$$

- $w_{cf} = 0.75$ was the best of {0, 0.25, 0.5, 0.75, 1} on validation NDCG@10 (section 9). It is a
  setting tuned on this data, not a general constant. The sidebar slider lets you change it live.
- **Goal fit** $g$ is used only if the user set targets or asked for "high protein" / "low calorie". With
  daily targets $T$ and 3 meals a day, per-meal target $t = T/3$:
  calorie fit $= \max(0, 1 - |kcal - t| / t)$, protein fit $= \min(1, p / t)$; "high protein" uses
  $0.5\min(1, \frac{p}{kcal} \cdot 100 / 8) + 0.5\min(1, p / 25)$. Unknown values score 0: we do not reward what
  we cannot verify. Default $w_g = 0.15$ (0.5 when the chat message asks for high protein or low calories).
  These are design choices and were not tuned, because we have no logged user targets to tune against.
- **Cold start (new user).** With no likes, the CF side is replaced by popularity,
  $\log(1 + \#\text{ratings})$. If the user typed preferences, $h = 0.6\,\tilde s^{cb} + 0.4\,\widetilde{pop}$;
  with nothing at all, the list is popularity only. With fewer than 5 likes, CF is averaged with popularity,
  because in our cold-start experiment popularity beat CF when only 3 ratings were known.
- **Cold start (new item).** The 1,182 extra Indian recipes have no ratings. Their CF score is 0 and their
  popularity is set to the median of rated recipes, so being new is neither a penalty nor a boost.

---

## 7. Constraint layer (`nutrimate/constraints.py`)

Hard rules are applied **before** ranking, so a high score can never bring back a forbidden recipe.

| Rule | How it is checked |
|---|---|
| Diet | eggetarian: no meat or fish word; vegetarian: also no egg (Indian convention); vegan: also no dairy or honey |
| Allergens | 9 groups (peanut, tree nuts, dairy, egg, gluten, soy, fish, shellfish, sesame), each a keyword list. "nuts" means peanut and tree nuts |
| Disliked ingredients | free text, matched as a whole word with simple plurals |
| Max kcal / min protein per serving | numeric comparison; a recipe with **unknown** value fails the rule |
| Max time, meal type, cuisine (when asked for "only this cuisine") | field comparison |
| Missing ingredient list | if the user has any diet or exclusion rule, recipes with no ingredient list are removed |

Details that matter:

- Matching uses the recipe name, ingredient list **and the instruction steps**. During inspection we found
  recipes where an ingredient appears only in the steps (the ingredient and quantity lists have different
  lengths in many rows), so the ingredient list alone is not enough.
- Word boundaries and exception phrases avoid obvious false matches: "eggplant" is not egg, "nutmeg" and
  "coconut" are not tree nuts, "peanut butter" and "coconut milk" are not dairy, "rice flour" and "besan" are
  not gluten. In the other direction we stay conservative: "flour" counts as gluten, "worcestershire" counts as
  fish.
- A chat message can add rules but cannot remove an allergen or relax the diet saved in the profile.
- This is text matching on user-written recipes. It reduces risk; it does not guarantee allergy safety, and
  the app says so on every card.

---

## 8. Diversity (MMR) and explanations

From the top 100 candidates, Maximal Marginal Relevance picks the list one item at a time:

$$i^* = \arg\max_{i \notin S}\Big[\lambda\,\text{final}_{u,i} - (1 - \lambda)\max_{j \in S} \cos(\mathbf{x}_i, \mathbf{x}_j)\Big], \qquad \lambda = 0.7$$

so the list does not fill up with five versions of the same curry.

Explanations are templates filled only from the returned row: diet and exclusion checks that passed, kcal and
protein with their basis and source, cuisine and Food.com tags, ingredients shared with recipes the user liked,
the number of ratings and their mean, and whether CF contributed. If a value is missing the text says
"not available".

---

## 9. GenAI layer and guardrails (`nutrimate/parser.py`)

Flow: message, then parser, then JSON validated against a JSON Schema, then constraint layer and ranker,
then template explanation.

- **Schema.** Nine required fields: `meal_type`, `diet`, `cuisine`, `max_calories` (50 to 3000),
  `min_protein_g` (0 to 200), `max_minutes` (1 to 1440), `excluded_ingredients` (up to 20 strings),
  `prefer_high_protein`, `prefer_low_calorie`. Enums for categorical fields; no extra keys allowed.
- **Default parser: rules.** Regular expressions for diets, calorie and protein amounts, time, cuisines,
  meal words and exclusion phrases ("avoid", "no", "without", "allergic to", "X-free"). It works offline.
- **Optional LLM.** If `NUTRIMATE_LLM_API_KEY` or `OPENAI_API_KEY` is set, an OpenAI-compatible model is asked
  for the same JSON (temperature 0, JSON mode). Its output is treated as untrusted input: it must pass the
  schema or the rule-based result is used. Only the user's message is sent; no nutrition data goes in or out.
- **Asking instead of guessing.** "something under 700" has no unit, so the assistant asks whether 700 kcal
  was meant. An out-of-range number is dropped with a question. Assumptions that are made are stated, for
  example "Food.com does not separate lunch and dinner, so I searched main dishes" or "took 'quick' to mean
  30 minutes or less".
- **No invented facts.** Every number in a reply comes from the returned recipe row and is labelled with its
  basis and source.

---

## 10. Macro tracking (`nutrimate/tracker.py`)

$$\text{per serving: } m = q_{servings} \times v_{serving} \qquad\qquad \text{per 100 g: } m = \frac{q_{grams}}{100} \times v_{100g}$$

Each log row stores the per-unit values, the basis and the source at the time of logging, so editing the
quantity recomputes correctly and a later data update does not silently change history. A NULL value is
reported as "n items with unknown value not counted" next to the total. Remaining = target minus consumed,
shown only when the user set a target. Data is kept in a local SQLite file (`data/nutrimate_app.db`).

---

## 11. Evaluation protocol (`scripts/evaluate.py`)

- **Split.** Per user, ratings are sorted by time: the oldest 70% train, the next 10% validation, the newest
  20% test (161,009 / 21,067 / 43,766 ratings). Predicting later ratings from earlier ones avoids the leakage
  of a random row split.
- **Tuning.** SVD factors and $w_{cf}$ were chosen on validation, then all models were refit on
  train + validation and scored once on test.
- **Ranking task.** For each of the 3,665 test users with at least one relevant test item, rank all 7,903
  recipes except the ones already rated in train/validation; take the top 10. Relevant = test rating of 4 or 5.
  Because 95% of ratings are 4 or 5, this is close to predicting "will the user review this recipe next".
- **Metrics.** Precision@10, Recall@10, NDCG@10 with binary gains, catalogue coverage (share of recipes
  that appear in at least one top-10 list), intra-list diversity (1 minus mean pairwise cosine of the
  content vectors, on every ninth user), RMSE and MAE for star prediction.
- **Reference point.** A random ranker would get an expected precision@10 of about 0.0015 (mean of
  relevant test items over unseen recipes per user).

### 11.1 Ranking results (test split)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity@10 |
|---|---|---|---|---|---|
| Popularity | 0.0220 | 0.0255 | 0.0293 | 0.0066 | 0.929 |
| Content only | 0.0055 | 0.0048 | 0.0065 | 0.3714 | 0.663 |
| CF only (implicit SVD, k = 16) | 0.0236 | 0.0271 | 0.0305 | 0.0401 | 0.936 |
| Hybrid w_cf = 0.25 | 0.0134 | 0.0133 | 0.0175 | 0.3443 | 0.706 |
| Hybrid w_cf = 0.50 | 0.0222 | 0.0247 | 0.0294 | 0.1869 | 0.838 |
| Hybrid w_cf = 0.75 (chosen on validation) | 0.0232 | 0.0264 | 0.0305 | 0.0593 | 0.909 |
| Hybrid w_cf = 0.75 + MMR (lambda 0.7) | 0.0235 | 0.0270 | 0.0308 | 0.0569 | 0.939 |

(w_cf = 0 equals content only and w_cf = 1 equals CF only.)

On validation, NDCG@10 was 0.0198 (popularity), 0.0045 (content), 0.0203 (CF), 0.0198 / 0.0209 for hybrid
0.5 / 0.75, which is why 0.75 was chosen.

### 11.2 Rating prediction (test split, 43,766 ratings)

| Model | RMSE | MAE |
|---|---|---|
| Global mean | 0.5764 | 0.4062 |
| User + item bias (damped) | 0.5344 | 0.3373 |
| Biased MF, 32 factors, SGD | 0.5372 | 0.3326 |

### 11.3 Cold start: only each user's first 3 ratings known

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Popularity | 0.0190 | 0.0235 | 0.0251 | 0.0015 |
| Content only | 0.0030 | 0.0032 | 0.0037 | 0.6839 |
| CF only (fold-in) | 0.0146 | 0.0184 | 0.0200 | 0.0716 |
| Hybrid w_cf = 0.75 | 0.0140 | 0.0174 | 0.0193 | 0.1782 |

### 11.4 Constraint check (hybrid top-10, 300 sampled test users per profile)

| Constraint profile | Share of catalogue allowed | Violation rate without filter | With filter |
|---|---|---|---|
| vegetarian, no peanut | 36.9% | 88.8% | 0.0% |
| vegan | 15.3% | 95.8% | 0.0% |
| gluten-free, at most 600 kcal | 44.8% | 74.1% | 0.0% |
| dairy-free, at least 20 g protein | 11.6% | 81.5% | 0.0% |
| no egg, no mushroom, at most 45 min | 42.3% | 70.0% | 0.0% |

A violation is a returned recipe that breaks a machine-checkable rule as defined in section 7, checked row by
row with `check_recipe`. Zero with the filter is expected by construction; the useful number is the left
column, which shows how often a ranker without the constraint layer would have shown something the user
said they cannot eat. 15,000 filtered recommendations were checked in total.

### 11.5 What the results say

- **CF carries the accuracy.** CF only beats popularity slightly (NDCG 0.0305 vs 0.0293) while recommending
  six times more of the catalogue (coverage 0.040 vs 0.007). Both are about 15 times better than random.
- **Content alone is weak at predicting what people review next** on Food.com. Its value is coverage (37% of
  the catalogue), working for brand-new recipes, and using typed preferences before any history exists.
- **The hybrid at 0.75 matches CF on NDCG and raises coverage by about 48%** (0.059 vs 0.040). Adding MMR
  raised list diversity from 0.909 to 0.939 without losing accuracy (NDCG 0.0308). We do not claim the hybrid is
  more accurate than CF; on this data it is equally accurate and less repetitive.
- **Rating prediction barely improves on biases.** With 80% five-star ratings there is little variance left
  for latent factors: MF has slightly better MAE (0.333 vs 0.337) but slightly worse RMSE (0.537 vs 0.534).
  This is why NutriMate ranks with the implicit model and reports RMSE only for completeness.
- **Cold start:** with 3 known ratings, popularity is the strongest signal (NDCG 0.0251 vs 0.0200 for CF),
  which is why the app mixes popularity into CF until a user has 5 likes.
- **Constraints matter:** without the filter, 70% to 96% of top-10 items would break the user's rules.

---

## 12. Limitations

- **Offline metrics only.** No user study has been run. Offline accuracy on historical reviews does not
  prove that people would like the suggestions or eat better.
- **Ratings are skewed and sparse** (0.92% density even after the 20/15 core), and Food.com reviewers are not
  a typical Indian user base. Absolute precision values are low, which is normal for top-10 over about 8,000
  items.
- **Recipe nutrition is site-computed per serving**, serving sizes vary and are missing for about a third of
  recipes, and 176 values were removed as implausible. Values are estimates, not lab measurements.
- **Allergen and diet detection is keyword based** on user-written text. Unusual names or missing
  ingredients can slip through; conservative lists also exclude some safe recipes.
- **Cuisine labels** come from tags; 63% of recipes have no cuisine tag and show as Unspecified.
- **Goal-fit weights and the cold-start popularity weight are untuned** design choices.
- The demo is single-user and local (Streamlit + SQLite), not a deployed multi-user service.

## 13. Future work

- A FastAPI backend with the same `nutrimate` package behind it and a React front end, as in the original plan.
- Map recipe ingredients to INDB / USDA foods to compute nutrition from quantities and fill missing values,
  with a confidence label.
- Add Indian user preference data (a small survey or app logs) and retrain CF with app feedback.
- Learn the hybrid and goal weights per user from feedback, for example with a simple bandit.
- Barcode lookup through Open Food Facts for packaged food, with its data quality shown.
- A short user study measuring perceived relevance and how easy the tracker is to use.

## 14. References

1. B. P. Majumder, S. Li, J. Ni, J. McAuley. Generating Personalized Recipes from Historical User Preferences. EMNLP 2019.
2. A. Irkaal. Food.com Recipes and Reviews. Kaggle dataset, CC0.
3. A. Vijayakumar, H. B. Dubasi, A. Awasthi, L. M. Jaacks. Development of an Indian Food Composition Database. Current Developments in Nutrition, 2024. doi:10.1016/j.cdnut.2024.103790
4. U.S. Department of Agriculture, Agricultural Research Service. FoodData Central, SR Legacy, 2018.
5. P. Cremonesi, Y. Koren, R. Turrin. Performance of Recommender Algorithms on Top-N Recommendation Tasks. RecSys 2010 (PureSVD).
6. Y. Koren, R. Bell, C. Volinsky. Matrix Factorization Techniques for Recommender Systems. IEEE Computer, 2009.
7. J. Carbonell, J. Goldstein. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. SIGIR 1998.
