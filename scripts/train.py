"""Fit the models used by the app on all processed interactions and save them to models/.

(Offline evaluation in evaluate.py fits its own copies on the training split only.)
"""

import json
import time

import _path  # noqa: F401
import joblib
import numpy as np

from nutrimate.config import MODELS_DIR, RESULTS_DIR
from nutrimate.hybrid import Recommender


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    n_factors = 64
    tuned = RESULTS_DIR / "svd_factor_tuning.csv"
    if tuned.exists():  # use the factor count chosen on the validation split, if evaluate.py ran
        import pandas as pd
        tab = pd.read_csv(tuned)
        n_factors = int(tab.sort_values("val_ndcg@10", ascending=False).iloc[0]["n_factors"])
    rec = Recommender.build(n_factors=n_factors)
    joblib.dump(rec.content, MODELS_DIR / "content_model.joblib", compress=3)
    joblib.dump(rec.cf, MODELS_DIR / "cf_implicit_svd.joblib", compress=3)
    np.save(MODELS_DIR / "popularity.npy", rec.pop)
    info = {
        "recipes": len(rec.recipes),
        "tfidf_vocabulary": len(rec.content.vectorizer.vocabulary_),
        "item_vector_dims": int(rec.content.X.shape[1]),
        "cf_factors": n_factors,
        "cf_items_with_ratings": int(rec.cf.trained_items.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
