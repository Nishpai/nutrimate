"""Hybrid recommender: constraint filter -> content + CF (or popularity) -> goal fit -> MMR.

    hybrid = (1 - w_cf) * content + w_cf * cf
    final  = (1 - w_goal) * hybrid + w_goal * goal   (goal only when targets are set)
           + cuisine bonus (soft preference)

Every component is min-max normalized over the candidates that passed the hard
constraints, so the weights mean the same thing for every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config
from .constraints import Constraints, filter_recipes
from .content import ContentModel
from .collab import ImplicitSVD
from .data import load_interactions, load_recipes

CUISINE_BONUS = 0.10


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def popularity_scores(recipes: pd.DataFrame, interactions: pd.DataFrame) -> np.ndarray:
    """log(1 + number of ratings). Recipes with no ratings in our data (the extra Indian
    recipes) get the median value as a neutral prior, so being new is not a penalty."""
    counts = interactions.groupby("recipe_id").size()
    n = recipes.recipe_id.map(counts)
    pop = np.log1p(n.fillna(0).to_numpy())
    rated = n.notna().to_numpy()
    pop[~rated] = np.median(pop[rated])
    return pop


def mmr_select(candidates, relevance, sim_fn, k, lam):
    """Maximal Marginal Relevance: pick items that are relevant but not too similar
    to the ones already picked. score = lam * rel - (1 - lam) * max_sim_to_selected."""
    candidates = list(candidates)
    rel = dict(zip(candidates, relevance))
    if len(candidates) <= k:
        return candidates
    sim = sim_fn(candidates, candidates)
    pos = {c: j for j, c in enumerate(candidates)}
    selected = [max(candidates, key=lambda c: rel[c])]
    remaining = [c for c in candidates if c != selected[0]]
    while len(selected) < k and remaining:
        sel_pos = [pos[s] for s in selected]
        best, best_val = None, -np.inf
        for c in remaining:
            val = lam * rel[c] - (1 - lam) * sim[pos[c], sel_pos].max()
            if val > best_val:
                best, best_val = c, val
        selected.append(best)
        remaining.remove(best)
    return selected


@dataclass
class UserState:
    liked_ids: list = field(default_factory=list)       # recipe ids the user liked (history + feedback)
    disliked_ids: list = field(default_factory=list)
    preference_text: str = ""                           # cuisines and liked ingredients typed in the profile
    daily_kcal: float | None = None
    daily_protein_g: float | None = None
    prefer_high_protein: bool = False
    prefer_low_calorie: bool = False


class Recommender:
    def __init__(self, recipes, content: ContentModel, cf: ImplicitSVD, popularity: np.ndarray):
        self.recipes = recipes
        self.content = content
        self.cf = cf
        self.pop = popularity
        self.id_to_idx = {rid: i for i, rid in enumerate(recipes.recipe_id)}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, models_dir=config.MODELS_DIR):
        recipes = load_recipes()
        content = joblib.load(models_dir / "content_model.joblib")
        cf = joblib.load(models_dir / "cf_implicit_svd.joblib")
        pop = np.load(models_dir / "popularity.npy")
        return cls(recipes, content, cf, pop)

    @classmethod
    def build(cls, recipes=None, interactions=None, n_factors=64):
        """Fit all models in memory (used by train.py and by tests)."""
        recipes = load_recipes() if recipes is None else recipes
        interactions = load_interactions() if interactions is None else interactions
        content = ContentModel().fit(recipes)
        id_to_idx = {rid: i for i, rid in enumerate(recipes.recipe_id)}
        users = {u: j for j, u in enumerate(interactions.user_id.unique())}
        cf = ImplicitSVD(n_factors=n_factors).fit(
            interactions.user_id.map(users).to_numpy(), interactions.recipe_id.map(id_to_idx).to_numpy(),
            len(users), len(recipes))
        return cls(recipes, content, cf, popularity_scores(recipes, interactions))

    # ------------------------------------------------------------- components
    def idx(self, ids):
        return [self.id_to_idx[r] for r in ids if r in self.id_to_idx]

    def content_scores(self, user: UserState):
        liked, disliked = self.idx(user.liked_ids), self.idx(user.disliked_ids)
        parts = []
        if liked or disliked:
            w = [1.0] * len(liked) + [-0.5] * len(disliked)
            parts.append(self.content.profile_from_items(liked + disliked, w))
        txt = self.content.profile_from_text(user.preference_text)
        if txt is not None:
            parts.append(txt * 0.7)
        if not parts:
            return np.zeros(len(self.recipes)), "none"
        prof = parts[0]
        for p in parts[1:]:
            prof = prof + p
        return self.content.scores(prof), "profile"

    def cf_scores(self, user: UserState):
        """CF from liked items the CF model knows; popularity when there are none (cold start)."""
        liked = [i for i in self.idx(user.liked_ids) if self.cf.trained_items[i]]
        disliked = [i for i in self.idx(user.disliked_ids) if self.cf.trained_items[i]]
        if not liked:
            return self.pop.copy(), "popularity (cold start)"
        hist = liked + disliked
        w = [1.0] * len(liked) + [-0.5] * len(disliked)
        cf = self.cf.scores(hist, w)
        if len(liked) < config.CF_MIN_HISTORY:
            return 0.5 * minmax(cf) + 0.5 * minmax(self.pop), "collaborative + popularity (short history)"
        return cf, "collaborative"

    def goal_scores(self, rows: pd.DataFrame, user: UserState):
        """How well each candidate fits the per-meal share of the user's daily targets.
        Unknown nutrients score 0 here (we do not reward what we cannot verify)."""
        subs = []
        kcal, prot = rows.calories_kcal.to_numpy(float), rows.protein_g.to_numpy(float)
        if user.daily_kcal:
            t = user.daily_kcal / config.MEALS_PER_DAY
            subs.append(np.clip(1 - np.abs(kcal - t) / t, 0, 1))
        if user.daily_protein_g:
            t = user.daily_protein_g / config.MEALS_PER_DAY
            subs.append(np.clip(prot / t, 0, 1))
        if user.prefer_high_protein:
            density = prot / np.maximum(kcal, 50) * 100  # g protein per 100 kcal
            # half for protein density (8 g per 100 kcal = full marks), half for the amount (25 g = full)
            subs.append(0.5 * np.clip(density / 8, 0, 1) + 0.5 * np.clip(prot / 25, 0, 1))
        if user.prefer_low_calorie:
            subs.append(np.clip(1 - kcal / 800, 0, 1))
        if not subs:
            return None
        return np.nan_to_num(np.mean(subs, axis=0), nan=0.0)

    # ------------------------------------------------------------- recommend
    def recommend(self, user: UserState, c: Constraints, k=config.TOP_K, w_cf=config.DEFAULT_W_CF,
                  w_goal=config.DEFAULT_W_GOAL, use_mmr=True, mmr_lambda=config.MMR_LAMBDA,
                  pool=100):
        passed, removed = filter_recipes(self.recipes, c)
        seen = set(user.liked_ids) | set(user.disliked_ids)
        passed = passed[~passed.recipe_id.isin(seen)]
        info = {"removed": removed, "n_candidates": len(passed), "cuisine_relaxed": False}
        if passed.empty:
            info.update(content_mode="none", cf_mode="none")
            return passed.assign(), info

        idx = passed.index.to_numpy()
        content_raw, content_mode = self.content_scores(user)
        cf_raw, cf_mode = self.cf_scores(user)
        content_n = minmax(content_raw[idx])
        cf_n = minmax(cf_raw[idx])
        if content_mode == "none":  # nothing to compare against yet: rely on the CF/popularity side
            hybrid = cf_n
        elif cf_mode.startswith("popularity"):
            # cold start: typed preferences are the only personal signal, so popularity gets less weight
            w_pop = config.COLD_START_POP_WEIGHT
            hybrid = (1 - w_pop) * content_n + w_pop * cf_n
        else:
            hybrid = (1 - w_cf) * content_n + w_cf * cf_n
        goal = self.goal_scores(passed, user)
        final = hybrid if goal is None else (1 - w_goal) * hybrid + w_goal * goal
        cuisine_match = np.zeros(len(idx))
        if c.cuisine and not c.hard_cuisine:
            cuisine_match = (passed.cuisine.to_numpy() == c.cuisine).astype(float)
            final = final + CUISINE_BONUS * cuisine_match

        order = np.argsort(-final)[:pool]
        if use_mmr:
            picked_pos = mmr_select(list(order), final[order],
                                    lambda a, b: self.content.similarity(idx[a], idx[b]), k, mmr_lambda)
        else:
            picked_pos = list(order[:k])

        out = passed.iloc[picked_pos].copy()
        out["score_content"] = content_n[picked_pos]
        out["score_cf"] = cf_n[picked_pos]
        out["score_goal"] = np.nan if goal is None else goal[picked_pos]
        out["score_hybrid"] = hybrid[picked_pos]
        out["score_final"] = final[picked_pos]
        out["cuisine_match"] = cuisine_match[picked_pos] > 0
        info.update(content_mode=content_mode, cf_mode=cf_mode,
                    w_cf=(config.COLD_START_POP_WEIGHT if cf_mode.startswith("popularity") and content_mode != "none"
                          else w_cf),
                    w_goal=(w_goal if goal is not None else 0.0))
        return out, info
