"""Content-based model: TF-IDF over recipe text and tags, plus a scaled nutrient vector.

Each recipe becomes one row vector:
    [ TF-IDF(name + ingredients + tags)  |  a * standardized(log(1 + nutrients)) ]
and the row is L2-normalized, so a dot product between rows is cosine similarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

NUTRIENT_FEATURES = ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


def recipe_document(row) -> str:
    """The text the TF-IDF model sees for one recipe."""
    tags = [f"tag_{_slug(k)}" for k in row["keywords"].split("|") if k]
    tags.append(f"cat_{_slug(row['category'])}")
    if row["cuisine"] != "Unspecified":
        tags.append(f"cuisine_{_slug(row['cuisine'])}")
    ingredients = row["ingredients"].replace("|", " , ")
    return f"{row['name']} . {ingredients} . {' '.join(tags)}"


class ContentModel:
    def __init__(self, nutrient_weight: float = 0.5, max_features: int = 30000):
        self.nutrient_weight = nutrient_weight
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b", ngram_range=(1, 2), min_df=3,
            max_features=max_features, sublinear_tf=True, stop_words="english",
        )
        self.scaler = StandardScaler()
        self.X = None  # item matrix, rows L2-normalized

    def fit(self, recipes: pd.DataFrame) -> "ContentModel":
        docs = recipes.apply(recipe_document, axis=1)
        tfidf = self.vectorizer.fit_transform(docs)
        nut = np.log1p(recipes[NUTRIENT_FEATURES].to_numpy(dtype=float))
        known = ~np.isnan(nut).any(axis=1)
        self.scaler.fit(nut[known])
        z = self.scaler.transform(np.nan_to_num(nut))
        # For similarity only: a recipe with unknown nutrition gets the neutral vector 0.
        # This never shows up as a displayed value.
        z[~known] = 0.0
        z *= self.nutrient_weight / np.sqrt(len(NUTRIENT_FEATURES))
        self.X = normalize(sparse.hstack([tfidf, sparse.csr_matrix(z)]).tocsr())
        return self

    # --- user profiles -----------------------------------------------------
    def profile_from_items(self, item_idx, weights=None):
        """Weighted average of item vectors (positive for liked, negative for disliked)."""
        item_idx = np.asarray(item_idx, dtype=int)
        if len(item_idx) == 0:
            return None
        w = np.ones(len(item_idx)) if weights is None else np.asarray(weights, dtype=float)
        vec = sparse.csr_matrix(w) @ self.X[item_idx]
        return normalize(vec)

    def profile_from_text(self, text: str):
        """A query vector built from typed preferences such as cuisines and liked ingredients."""
        if not text.strip():
            return None
        t = self.vectorizer.transform([text])
        pad = sparse.csr_matrix((1, self.X.shape[1] - t.shape[1]))
        vec = sparse.hstack([t, pad]).tocsr()
        return normalize(vec) if vec.nnz else None

    def scores(self, profile) -> np.ndarray:
        """Cosine similarity of every recipe to the profile (zeros if there is no profile)."""
        if profile is None:
            return np.zeros(self.X.shape[0])
        return np.asarray((self.X @ profile.T).todense()).ravel()

    def similarity(self, idx_a, idx_b) -> np.ndarray:
        return (self.X[idx_a] @ self.X[idx_b].T).toarray()
