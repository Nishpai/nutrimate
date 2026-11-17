"""Hard dietary constraints.

Everything here is deterministic keyword matching on the recipe text that the
dataset gives us (name, ingredient list and instruction steps). We scan the
instructions too because the Food.com ingredient lists are sometimes incomplete
(we found recipes where an ingredient appears only in the steps).

This is text matching, not a food-safety guarantee. If a recipe does not mention
an allergen we can only say "not found in the listed text", never "safe".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

MEAT_FISH = [
    "chicken", "beef", "pork", "bacon", "ham", "turkey", "lamb", "mutton", "veal", "goat meat",
    "sausage", "pepperoni", "salami", "prosciutto", "pancetta", "chorizo", "kielbasa", "bratwurst",
    "hot dog", "frankfurter", "meat", "meatball", "steak", "sirloin", "brisket", "ribs", "chuck",
    "ground round", "venison", "bison", "rabbit", "duck", "goose", "quail", "liver", "jerky",
    "lard", "suet", "gelatin", "gelatine", "worcestershire", "keema", "kheema", "hamburger",
    "moose", "elk", "deer", "boar", "pheasant", "partridge", "ostrich", "pepperoni", "spam",
    # fish and seafood
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "sardine", "anchovy", "anchovies",
    "mackerel", "haddock", "snapper", "catfish", "pomfret", "surimi", "shrimp", "prawn", "crab",
    "lobster", "scallop", "clam", "mussel", "oyster", "squid", "calamari", "octopus", "crawfish",
    "crayfish", "caviar",
]

DAIRY = [
    "milk", "butter", "buttermilk", "cheese", "cream", "yogurt", "yoghurt", "curd", "dahi", "ghee",
    "paneer", "whey", "casein", "parmesan", "mozzarella", "cheddar", "ricotta", "feta", "brie",
    "custard", "half-and-half", "half and half", "khoya", "mawa", "malai", "kefir", "lassi",
]

EGG = ["egg", "eggs", "egg white", "egg yolk", "mayonnaise", "mayo", "meringue", "eggnog"]

VEGAN_EXTRA = ["honey"]

ALLERGEN_GROUPS = {
    "peanut": ["peanut", "groundnut", "monkey nut", "arachis"],
    "tree_nut": [
        "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "filbert", "macadamia",
        "brazil nut", "pine nut", "praline", "marzipan", "nutella", "nut", "mixed nuts",
    ],
    "dairy": DAIRY,
    "egg": EGG,
    "gluten": [
        "wheat", "flour", "bread", "breadcrumb", "breadcrumbs", "panko", "pasta", "spaghetti",
        "macaroni", "noodle", "couscous", "barley", "rye", "semolina", "bulgur", "seitan", "cracker",
        "tortilla", "soy sauce", "beer", "malt", "maida", "atta", "sooji", "suji", "rava",
        "farro", "spelt", "pita", "naan", "roti", "chapati", "biscuit", "cake mix", "pie crust",
    ],
    "soy": ["soy", "soya", "soybean", "tofu", "tempeh", "edamame", "miso", "tamari"],
    "fish": [
        "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "sardine", "anchovy",
        "anchovies", "mackerel", "haddock", "snapper", "catfish", "pomfret", "worcestershire",
    ],
    "shellfish": [
        "shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel", "oyster", "crawfish",
        "crayfish", "squid", "calamari", "octopus",
    ],
    "sesame": ["sesame", "tahini", "gingelly", "til"],
}

# Phrases removed before matching a group, because they contain a group word but
# are not that thing (for example peanut butter is not dairy, eggplant is not egg).
NOT_THIS_GROUP = {
    "dairy": [
        "peanut butter", "almond butter", "cashew butter", "nut butter", "apple butter",
        "cocoa butter", "butternut", "butter beans", "butter bean", "coconut milk",
        "coconut cream", "almond milk", "soy milk", "soymilk", "rice milk", "oat milk",
        "cashew milk", "cream of tartar", "vegan butter", "vegan cheese", "vegan cream cheese",
        "dairy-free", "non-dairy", "nondairy",
    ],
    "egg": ["eggplant", "egg-free", "egg free", "egg replacer", "egg substitute"],
    "gluten": [
        "rice flour", "almond flour", "coconut flour", "chickpea flour", "gram flour",
        "besan", "corn flour", "cornflour", "tapioca flour", "potato flour", "gluten-free",
        "gluten free", "rice noodle", "rice noodles", "corn tortilla", "corn tortillas",
        "buckwheat flour", "sorghum flour", "millet flour", "ragi flour", "jowar flour",
    ],
    "meat": ["meatless", "vegetarian sausage", "veggie burger", "mock meat", "vegan sausage"],
    "tree_nut": ["nutmeg", "coconut", "doughnut", "butternut", "nut-free", "nut free"],
}

# Indian convention: "vegetarian" means no meat, fish or egg. "eggetarian" allows egg.
DIETS = ("any", "eggetarian", "vegetarian", "vegan")
DIET_RANK = {"any": 0, "eggetarian": 1, "vegetarian": 2, "vegan": 3}

# Labels shown in the app for the allergen group keys
GROUP_LABELS = {
    "peanut": "peanut", "tree_nut": "tree nuts", "any_nut": "nuts (peanut and tree nuts)",
    "dairy": "dairy", "egg": "egg", "gluten": "gluten (wheat, barley, rye)", "soy": "soy",
    "fish": "fish", "shellfish": "shellfish", "sesame": "sesame",
}


def _pattern(words: list[str]) -> re.Pattern:
    """Word-boundary regex that also allows simple plurals (egg/eggs, cashew/cashews)."""
    parts = sorted({re.escape(w.lower()) for w in words}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")(?:s|es)?\b")


def _strip(text: str, phrases: list[str]) -> str:
    for p in phrases:
        text = text.replace(p, " ")
    return text


_MEAT_RE = _pattern(MEAT_FISH)
_VEGAN_RE = _pattern(DAIRY + EGG + VEGAN_EXTRA)
_GROUP_RE = {g: _pattern(words) for g, words in ALLERGEN_GROUPS.items()}


def recipe_text(name: str, ingredients: str, instructions: str) -> str:
    """The text we search for constraint checks: name, ingredients and steps, lowercased."""
    return " ".join([name or "", (ingredients or "").replace("|", " ; "), instructions or ""]).lower()


def has_meat_or_fish(text: str) -> bool:
    return bool(_MEAT_RE.search(_strip(text, NOT_THIS_GROUP["meat"])))


def has_egg(text: str) -> bool:
    return bool(_GROUP_RE["egg"].search(_strip(text, NOT_THIS_GROUP["egg"])))


def has_animal_product(text: str) -> bool:
    """Dairy, egg or honey (meat/fish is checked separately)."""
    t = _strip(text, NOT_THIS_GROUP["dairy"] + NOT_THIS_GROUP["egg"])
    return bool(_VEGAN_RE.search(t))


def allergens_in(text: str) -> list[str]:
    found = []
    for group, rx in _GROUP_RE.items():
        if rx.search(_strip(text, NOT_THIS_GROUP.get(group, []))):
            found.append(group)
    return found


def normalize_exclusion(term: str) -> str:
    """Map what a user types (e.g. 'peanuts', 'nuts', 'gluten') to an allergen group if possible."""
    t = term.strip().lower()
    aliases = {
        "peanuts": "peanut", "peanut": "peanut", "groundnut": "peanut", "groundnuts": "peanut",
        "nuts": "any_nut", "nut": "any_nut", "any_nut": "any_nut", "tree nut": "tree_nut", "tree nuts": "tree_nut",
        "tree_nut": "tree_nut", "dairy": "dairy", "lactose": "dairy", "milk": "dairy",
        "egg": "egg", "eggs": "egg", "gluten": "gluten", "wheat": "gluten", "soy": "soy",
        "soya": "soy", "fish": "fish", "shellfish": "shellfish", "seafood": "shellfish",
        "sesame": "sesame",
    }
    return aliases.get(t, t)


def term_matches(text: str, term: str) -> bool:
    """True if a user exclusion (allergen group or free ingredient word) appears in the text."""
    key = normalize_exclusion(term)
    if key == "any_nut":  # people who say "no nuts" usually mean peanuts too
        return term_matches(text, "peanut") or term_matches(text, "tree_nut")
    if key in _GROUP_RE:
        return bool(_GROUP_RE[key].search(_strip(text, NOT_THIS_GROUP.get(key, []))))
    if not key:
        return False
    return bool(_pattern([key]).search(text))


# ---------------------------------------------------------------------------
# The constraint object and the filter
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    diet: str = "any"                      # any / eggetarian / vegetarian / vegan
    exclude: list[str] = field(default_factory=list)   # allergens and disliked ingredients
    max_calories: float | None = None      # per serving, kcal
    min_protein_g: float | None = None     # per serving, grams
    max_minutes: float | None = None
    meal_type: str | None = None
    cuisine: str | None = None             # soft by default, see hard_cuisine
    hard_cuisine: bool = False

    def describe(self) -> list[str]:
        out = []
        if self.diet != "any":
            out.append(self.diet)
        if self.exclude:
            out.append("excluding " + ", ".join(self.exclude))
        if self.max_calories is not None:
            out.append(f"at most {self.max_calories:g} kcal per serving")
        if self.min_protein_g is not None:
            out.append(f"at least {self.min_protein_g:g} g protein per serving")
        if self.max_minutes is not None:
            out.append(f"ready in {self.max_minutes:g} min or less")
        if self.meal_type:
            out.append(f"meal type: {self.meal_type}")
        if self.cuisine and self.hard_cuisine:
            out.append(f"cuisine: {self.cuisine}")
        return out


def check_recipe(row, c: Constraints) -> list[str]:
    """Return the list of rules this single recipe breaks (empty list = passes).

    Used both by the filter and by the evaluation to count violations.
    Missing nutrient values break a nutrient rule, because we cannot confirm it.
    """
    problems = []
    text = row["search_text"]
    ingredients = row["ingredients"]
    if (c.diet != "any" or c.exclude) and (not isinstance(ingredients, str) or not ingredients):
        problems.append("ingredients unknown")
    if c.diet != "any" and has_meat_or_fish(text):
        problems.append("contains meat or fish")
    if c.diet in ("vegetarian", "vegan") and has_egg(text):
        problems.append("contains egg")
    if c.diet == "vegan" and has_animal_product(text):
        problems.append("contains dairy, egg or honey")
    for term in c.exclude:
        if term_matches(text, term):
            problems.append(f"mentions {term}")
    if c.max_calories is not None:
        kcal = row["calories_kcal"]
        if kcal is None or pd.isna(kcal):
            problems.append("calories unknown")
        elif kcal > c.max_calories:
            problems.append("too many calories")
    if c.min_protein_g is not None:
        prot = row["protein_g"]
        if prot is None or pd.isna(prot):
            problems.append("protein unknown")
        elif prot < c.min_protein_g:
            problems.append("not enough protein")
    if c.max_minutes is not None:
        mins = row["total_minutes"]
        if mins is None or pd.isna(mins) or mins > c.max_minutes:
            problems.append("takes too long or time unknown")
    if c.meal_type and row["meal_type"] != c.meal_type:
        problems.append("different meal type")
    if c.cuisine and c.hard_cuisine and row["cuisine"] != c.cuisine:
        problems.append("different cuisine")
    return problems


def filter_recipes(recipes: pd.DataFrame, c: Constraints) -> tuple[pd.DataFrame, dict]:
    """Apply the hard constraints. Returns the passing rows and counts of why rows were removed."""
    mask = np.ones(len(recipes), dtype=bool)
    removed = {}

    def apply(name, bad):
        nonlocal mask
        bad = np.asarray(bad, dtype=bool) & mask
        if bad.any():
            removed[name] = int(bad.sum())
        mask &= ~bad

    if c.diet != "any" or c.exclude:
        # no ingredient list means we cannot check anything, so the recipe is not shown
        apply("ingredients unknown", (recipes["ingredients"].fillna("") == "").to_numpy())
    if c.diet != "any":
        apply("meat or fish", ~recipes["is_eggetarian"].to_numpy(dtype=bool))
    if c.diet in ("vegetarian", "vegan"):
        apply("egg", ~recipes["is_vegetarian"].to_numpy(dtype=bool))
    if c.diet == "vegan":
        apply("dairy, egg or honey", ~recipes["is_vegan"].to_numpy(dtype=bool))
    texts = recipes["search_text"]
    for term in c.exclude:
        apply(f"mentions {term}", texts.map(lambda t, term=term: term_matches(t, term)).to_numpy())
    if c.max_calories is not None:
        kcal = recipes["calories_kcal"]
        apply("calories unknown", kcal.isna().to_numpy())
        apply("over calorie limit", (kcal > c.max_calories).to_numpy())
    if c.min_protein_g is not None:
        prot = recipes["protein_g"]
        apply("protein unknown", prot.isna().to_numpy())
        apply("under protein minimum", (prot < c.min_protein_g).to_numpy())
    if c.max_minutes is not None:
        mins = recipes["total_minutes"]
        apply("too long or time unknown", (mins.isna() | (mins > c.max_minutes)).to_numpy())
    if c.meal_type:
        apply("different meal type", (recipes["meal_type"] != c.meal_type).to_numpy())
    if c.cuisine and c.hard_cuisine:
        apply("different cuisine", (recipes["cuisine"] != c.cuisine).to_numpy())
    return recipes[mask], removed
