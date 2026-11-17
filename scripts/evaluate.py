"""Offline evaluation on a per-user temporal split.

Steps
  1. split each user's ratings by time: train 70% / validation 10% / test 20%
  2. on validation: choose the number of SVD factors and look at hybrid weights
  3. refit on train + validation, evaluate on test:
       - rating prediction (RMSE, MAE): global mean, bias baseline, biased MF
       - top-10 ranking: popularity, content-only, CF-only, hybrid at several weights, hybrid + MMR
       - cold-start: same users but only their first 3 ratings are known
       - constraint check: how many top-10 items break a rule with and without the filter
Results go to results/*.csv, results/*.png and results/summary.json.
"""

import json
import time

import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from nutrimate import config  # noqa: E402
from nutrimate.collab import BiasedMF, ImplicitSVD, bias_baseline  # noqa: E402
from nutrimate.constraints import Constraints, check_recipe, filter_recipes  # noqa: E402
from nutrimate.content import ContentModel  # noqa: E402
from nutrimate.data import load_interactions, load_recipes  # noqa: E402
from nutrimate.hybrid import mmr_select  # noqa: E402
from nutrimate.metrics import (intra_list_diversity, mae, precision_recall_ndcg, rmse,  # noqa: E402
                               temporal_split)

K = config.TOP_K
REL = config.RELEVANT_RATING
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42
RES = config.RESULTS_DIR


def row_minmax(S):
    """Min-max normalize each row, ignoring NaN (masked) entries."""
    lo = np.nanmin(S, axis=1, keepdims=True)
    hi = np.nanmax(S, axis=1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-12)


class Setup:
    """Index maps and matrices for one train/eval pair."""

    def __init__(self, recipes, train, target):
        self.n_items = len(recipes)
        self.item_of = {rid: i for i, rid in enumerate(recipes.recipe_id)}
        users = sorted(target.user_id.unique())
        self.users = users
        self.uidx = {u: j for j, u in enumerate(users)}
        all_train_users = {u: j for j, u in enumerate(sorted(train.user_id.unique()))}
        self.train_u = train.user_id.map(all_train_users).to_numpy()
        self.train_i = train.recipe_id.map(self.item_of).to_numpy()
        self.n_train_users = len(all_train_users)
        tr = train[train.user_id.isin(self.uidx)]
        rows = tr.user_id.map(self.uidx).to_numpy()
        cols = tr.recipe_id.map(self.item_of).to_numpy()
        # histories of evaluated users: all train items (0/1), liked items (rating >= 4)
        self.H = sparse.csr_matrix((np.ones(len(tr)), (rows, cols)), shape=(len(users), self.n_items))
        self.H.data[:] = 1
        liked = tr.rating.to_numpy() >= REL
        self.L = sparse.csr_matrix((np.ones(liked.sum()), (rows[liked], cols[liked])),
                                   shape=(len(users), self.n_items))
        rel = target[target.rating >= REL]
        self.relevant = rel.groupby("user_id").recipe_id.apply(lambda s: {self.item_of[r] for r in s}).to_dict()
        self.pop = np.log1p(np.bincount(self.train_i, minlength=self.n_items).astype(float))


def score_matrices(setup, content, cf, H=None, L=None, batch=600):
    """Yield (user rows, content scores, cf scores) in batches, train items masked to -inf."""
    H = setup.H if H is None else H
    L = setup.L if L is None else L
    for start in range(0, H.shape[0], batch):
        h, l = H[start:start + batch], L[start:start + batch]
        prof_src = l.copy()
        empty = np.asarray(prof_src.sum(axis=1)).ravel() == 0
        if empty.any():  # no liked items: fall back to everything the user rated
            prof_src = prof_src.tolil()
            prof_src[np.where(empty)[0]] = h[np.where(empty)[0]]
            prof_src = prof_src.tocsr()
        prof = normalize(prof_src @ content.X)
        cs = np.asarray((prof @ content.X.T).todense())
        fs = cf.batch_scores(h)
        mask = h.toarray() > 0
        cs[mask] = -np.inf
        fs[mask] = -np.inf
        yield start, cs, fs, mask


def evaluate_lists(setup, lists, content, users_subset=None):
    """lists: dict name -> array (n_users, K) of item indices."""
    rows = []
    for name, top in lists.items():
        P, R, N, ild = [], [], [], []
        for j, u in enumerate(setup.users):
            if u not in setup.relevant:
                continue
            p, r, n = precision_recall_ndcg(top[j], setup.relevant[u], K)
            P.append(p), R.append(r), N.append(n)
        sample = range(0, len(setup.users), max(1, len(setup.users) // 400))
        for j in sample:
            ild.append(intra_list_diversity(top[j], content.similarity))
        rows.append({
            "model": name, f"precision@{K}": np.mean(P), f"recall@{K}": np.mean(R), f"ndcg@{K}": np.mean(N),
            "coverage": len(np.unique(top)) / setup.n_items, f"diversity@{K}": np.mean(ild),
            "users_evaluated": len(P),
        })
    return pd.DataFrame(rows)


def rank_all(setup, content, cf, weights, mmr=None, H=None, L=None):
    """Top-K lists for popularity, content, CF and hybrid weights (and optional hybrid+MMR)."""
    n = (setup.H if H is None else H).shape[0]
    lists = {"popularity": np.zeros((n, K), int), "content-only": np.zeros((n, K), int),
             "cf-only": np.zeros((n, K), int)}
    for w in weights:
        lists[f"hybrid w_cf={w}"] = np.zeros((n, K), int)
    if mmr is not None:
        lists[f"hybrid w_cf={mmr[0]} + MMR"] = np.zeros((n, K), int)
    for start, cs, fs, mask in score_matrices(setup, content, cf, H, L):
        pop = np.where(mask, -np.inf, setup.pop[None, :])
        sl = slice(start, start + cs.shape[0])
        lists["popularity"][sl] = np.argsort(-pop, axis=1)[:, :K]
        lists["content-only"][sl] = np.argsort(-cs, axis=1)[:, :K]
        lists["cf-only"][sl] = np.argsort(-fs, axis=1)[:, :K]
        csn = row_minmax(np.where(mask, np.nan, cs))
        fsn = row_minmax(np.where(mask, np.nan, fs))
        csn, fsn = np.nan_to_num(csn, nan=-1), np.nan_to_num(fsn, nan=-1)
        for w in weights:
            hy = (1 - w) * csn + w * fsn
            lists[f"hybrid w_cf={w}"][sl] = np.argsort(-hy, axis=1)[:, :K]
        if mmr is not None:
            w, lam = mmr
            hy = (1 - w) * csn + w * fsn
            pool = np.argsort(-hy, axis=1)[:, :50]
            for r in range(hy.shape[0]):
                picked = mmr_select(list(pool[r]), hy[r, pool[r]], content.similarity, K, lam)
                lists[f"hybrid w_cf={w} + MMR"][start + r] = picked
    return lists


def fit_cf(setup, k):
    return ImplicitSVD(n_factors=k, random_state=SEED).fit(
        setup.train_u, setup.train_i, setup.n_train_users, setup.n_items)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    recipes = load_recipes()
    inter = load_interactions()
    inter["split"] = temporal_split(inter)
    inter[["user_id", "recipe_id", "split"]].to_csv(config.SPLIT_FILE, index=False)
    counts = inter.split.value_counts().to_dict()
    print("split sizes:", counts)

    content = ContentModel().fit(recipes)  # content features use no ratings, so no leakage

    # ---------------- 1. validation: choose SVD factors and look at weights
    train = inter[inter.split == "train"]
    val = inter[inter.split == "val"]
    s_val = Setup(recipes, train, val)
    tune = []
    for k in [16, 32, 64, 128]:
        cf = fit_cf(s_val, k)
        res = evaluate_lists(s_val, {"cf": rank_all(s_val, content, cf, [])["cf-only"]}, content)
        tune.append({"n_factors": k, "val_ndcg@10": res[f"ndcg@{K}"][0], "val_recall@10": res[f"recall@{K}"][0]})
        print(f"  val: SVD k={k}  NDCG@10={tune[-1]['val_ndcg@10']:.4f}")
    tune = pd.DataFrame(tune)
    tune.to_csv(RES / "svd_factor_tuning.csv", index=False)
    best_k = int(tune.sort_values("val_ndcg@10", ascending=False).iloc[0].n_factors)
    cf_val = fit_cf(s_val, best_k)
    val_res = evaluate_lists(s_val, rank_all(s_val, content, cf_val, WEIGHTS), content)
    val_res.to_csv(RES / "validation_weight_sweep.csv", index=False)
    hyb = val_res[val_res.model.str.startswith("hybrid")]
    best_w = float(hyb.sort_values(f"ndcg@{K}", ascending=False).iloc[0].model.split("=")[1])
    print(f"  chosen on validation: k={best_k}, w_cf={best_w}")

    # ---------------- 2. test: refit on train + val
    trainval = inter[inter.split != "test"]
    test = inter[inter.split == "test"]
    s_test = Setup(recipes, trainval, test)
    cf = fit_cf(s_test, best_k)
    lists = rank_all(s_test, content, cf, WEIGHTS, mmr=(best_w, config.MMR_LAMBDA))
    ranking = evaluate_lists(s_test, lists, content)
    ranking.to_csv(RES / "ranking_metrics_test.csv", index=False)
    # reference point: expected precision@10 of a random ranker over the unseen recipes
    seen = np.asarray(s_test.H.sum(axis=1)).ravel()
    rand_p = np.mean([len(s_test.relevant[u]) / (s_test.n_items - seen[j])
                      for j, u in enumerate(s_test.users) if u in s_test.relevant])
    print(f"  random ranker, expected precision@10: {rand_p:.4f}")
    print(ranking.round(4).to_string(index=False))

    # ---------------- 3. rating prediction
    users = {u: j for j, u in enumerate(sorted(inter.user_id.unique()))}
    items = s_test.item_of
    tu, ti = trainval.user_id.map(users).to_numpy(), trainval.recipe_id.map(items).to_numpy()
    tr = trainval.rating.to_numpy(float)
    eu, ei = test.user_id.map(users).to_numpy(), test.recipe_id.map(items).to_numpy()
    er = test.rating.to_numpy(float)
    base = bias_baseline(tu, ti, tr, len(users), len(items))
    print("  training biased MF (SGD) ...")
    mf = BiasedMF(n_factors=32, epochs=25, random_state=SEED).fit(tu, ti, tr, len(users), len(items))
    rating = pd.DataFrame([
        {"model": "global mean", "rmse": rmse(er, np.full(len(er), tr.mean())), "mae": mae(er, np.full(len(er), tr.mean()))},
        {"model": "user + item bias", "rmse": rmse(er, base(eu, ei)), "mae": mae(er, base(eu, ei))},
        {"model": "biased MF (32 factors, SGD)", "rmse": rmse(er, mf.predict(eu, ei)), "mae": mae(er, mf.predict(eu, ei))},
    ])
    rating["test_ratings"] = len(er)
    rating.to_csv(RES / "rating_metrics_test.csv", index=False)
    print(rating.round(4).to_string(index=False))

    # ---------------- 4. cold start: only the first 3 train ratings of each test user are known
    tv = trainval.sort_values("ts")
    first3 = tv.groupby("user_id").head(3)
    first3 = first3[first3.user_id.isin(s_test.uidx)]
    rows = first3.user_id.map(s_test.uidx).to_numpy()
    cols = first3.recipe_id.map(items).to_numpy()
    H3 = sparse.csr_matrix((np.ones(len(first3)), (rows, cols)), shape=s_test.H.shape)
    lk = first3.rating.to_numpy() >= REL
    L3 = sparse.csr_matrix((np.ones(lk.sum()), (rows[lk], cols[lk])), shape=s_test.H.shape)
    # items rated later in train/val are not "unseen" for these users, but they are also not test
    # items, so they simply cannot count as hits. Relevant items stay the test items.
    cold_lists = rank_all(s_test, content, cf, [best_w], H=H3, L=L3)
    cold = evaluate_lists(s_test, cold_lists, content)
    cold.to_csv(RES / "cold_start_3_ratings.csv", index=False)
    print("cold start (3 known ratings):")
    print(cold.round(4).to_string(index=False))

    # ---------------- 5. constraint check on hybrid top-10, with and without the filter
    profiles = {
        "vegetarian, no peanut": Constraints(diet="vegetarian", exclude=["peanut"]),
        "vegan": Constraints(diet="vegan"),
        "gluten-free, <=600 kcal": Constraints(exclude=["gluten"], max_calories=600),
        "dairy-free, >=20 g protein": Constraints(exclude=["dairy"], min_protein_g=20),
        "no egg, no mushroom, <=45 min": Constraints(exclude=["egg", "mushroom"], max_minutes=45),
    }
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(s_test.users), size=min(300, len(s_test.users)), replace=False)
    Hs, Ls = s_test.H[sample], s_test.L[sample]
    sub = Setup.__new__(Setup)
    sub.__dict__.update(s_test.__dict__)
    sub.H, sub.L = Hs, Ls
    crow = []
    for pname, c in profiles.items():
        allowed, _ = filter_recipes(recipes, c)
        allow_mask = np.zeros(len(recipes), bool)
        allow_mask[allowed.index.to_numpy()] = True
        v_off = v_on = n_off = n_on = 0
        for start, cs, fs, mask in score_matrices(sub, content, cf):
            csn = np.nan_to_num(row_minmax(np.where(mask, np.nan, cs)), nan=-1)
            fsn = np.nan_to_num(row_minmax(np.where(mask, np.nan, fs)), nan=-1)
            hy = (1 - best_w) * csn + best_w * fsn
            top_off = np.argsort(-hy, axis=1)[:, :K]
            top_on = np.argsort(-np.where(allow_mask[None, :], hy, -np.inf), axis=1)[:, :K]
            for r in range(hy.shape[0]):
                for i in top_off[r]:
                    v_off += bool(check_recipe(recipes.iloc[i], c))
                    n_off += 1
                for i in top_on[r]:
                    v_on += bool(check_recipe(recipes.iloc[i], c))
                    n_on += 1
        crow.append({"constraint_profile": pname, "catalogue_share_allowed": allow_mask.mean(),
                     "violation_rate_without_filter": v_off / n_off,
                     "violation_rate_with_filter": v_on / n_on, "items_checked": n_on})
    cons_df = pd.DataFrame(crow)
    cons_df.to_csv(RES / "constraint_check.csv", index=False)
    print(cons_df.round(4).to_string(index=False))

    # ---------------- charts
    make_charts(ranking, cold, rating, best_w)

    summary = {
        "split_sizes": counts, "test_users_evaluated": int(ranking.users_evaluated.iloc[0]),
        "random_expected_precision@10": round(float(rand_p), 5),
        "chosen_svd_factors": best_k, "chosen_w_cf_on_validation": best_w, "k": K,
        "relevant_rating_threshold": REL, "seconds": round(time.time() - t0, 1),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def make_charts(ranking, cold, rating, best_w):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1. model comparison
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = ranking.model.tolist()
    x = np.arange(len(names))
    for j, (m, c) in enumerate([(f"precision@{K}", "#4C72B0"), (f"recall@{K}", "#55A868"), (f"ndcg@{K}", "#C44E52")]):
        ax.bar(x + (j - 1) * 0.27, ranking[m], width=0.27, label=m, color=c)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Top-10 ranking on the temporal test split")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, ranking[[f"precision@{K}", f"recall@{K}", f"ndcg@{K}"]].to_numpy().max() * 1.25)
    fig.tight_layout()
    fig.savefig(RES / "ranking_comparison.png", dpi=150)
    plt.close(fig)

    # 2. weight sweep
    sweep = ranking[ranking.model.str.match(r"hybrid w_cf=[0-9.]+$")].copy()
    sweep["w"] = sweep.model.str.split("=").str[1].astype(float)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(sweep.w, sweep[f"ndcg@{K}"], "o-", color="#C44E52", label=f"NDCG@{K}")
    ax1.plot(sweep.w, sweep[f"recall@{K}"], "s-", color="#55A868", label=f"Recall@{K}")
    ax1.set_xlabel("w_cf (0 = content only, 1 = CF only)")
    ax1.set_ylabel("score")
    ax2 = ax1.twinx()
    ax2.plot(sweep.w, sweep.coverage, "^--", color="#8172B2", label="coverage")
    ax2.set_ylabel("catalogue coverage")
    ax1.axvline(best_w, color="grey", lw=0.8, ls=":")
    lines = [ln for ln in ax1.get_lines() + ax2.get_lines() if not ln.get_label().startswith("_")]
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="center right")
    ax1.set_title("Hybrid weight sweep (test split)")
    fig.tight_layout()
    fig.savefig(RES / "hybrid_weight_sweep.png", dpi=150)
    plt.close(fig)

    # 3. cold start
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(cold.model, cold[f"ndcg@{K}"], color="#4C72B0")
    ax.set_ylabel(f"NDCG@{K}")
    ax.set_title("Cold start: only 3 ratings known per user")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(RES / "cold_start.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
