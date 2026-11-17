"""Collaborative filtering models, written with numpy/scipy/sklearn only.

1. ImplicitSVD ("PureSVD"): truncated SVD of the binary user x recipe matrix. Used for
   ranking. A user's scores are r_u V V^T, where r_u is the user's 0/1 history and V
   holds the item factors. Because the user vector is computed from the history on the
   fly ("folding in"), it also works for a new app user after a few likes.

2. BiasedMF: explicit rating prediction r_hat = mu + b_u + b_i + p_u . q_i trained with
   SGD. Used to report RMSE/MAE on held-out ratings.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class ImplicitSVD:
    def __init__(self, n_factors: int = 64, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.V = None  # (n_items, k)

    def fit(self, user_idx, item_idx, n_users, n_items) -> "ImplicitSVD":
        R = sparse.csr_matrix((np.ones(len(user_idx)), (user_idx, item_idx)), shape=(n_users, n_items))
        R.data[:] = 1.0  # duplicates collapse to 1
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        svd.fit(R)
        self.V = svd.components_.T
        self.trained_items = np.asarray(R.sum(axis=0)).ravel() > 0
        return self

    def scores(self, history_idx, history_weights=None) -> np.ndarray:
        """Scores for all items given a list of item indices the user interacted with."""
        history_idx = np.asarray(history_idx, dtype=int)
        if len(history_idx) == 0:
            return np.zeros(self.V.shape[0])
        w = np.ones(len(history_idx)) if history_weights is None else np.asarray(history_weights, float)
        user_vec = w @ self.V[history_idx]
        return self.V @ user_vec

    def batch_scores(self, R_users: sparse.csr_matrix) -> np.ndarray:
        """Scores for many users at once (rows of R_users are 0/1 histories)."""
        return np.asarray((R_users @ self.V) @ self.V.T)


class BiasedMF:
    def __init__(self, n_factors=32, lr=0.01, reg=0.05, epochs=25, random_state=42):
        self.k, self.lr, self.reg, self.epochs = n_factors, lr, reg, epochs
        self.rng = np.random.default_rng(random_state)

    def fit(self, u, i, r, n_users, n_items, verbose=False):
        self.mu = float(np.mean(r))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = self.rng.normal(0, 0.05, (n_users, self.k))
        self.Q = self.rng.normal(0, 0.05, (n_items, self.k))
        order = np.arange(len(r))
        for epoch in range(self.epochs):
            self.rng.shuffle(order)
            sq = 0.0
            for idx in order:
                uu, ii = u[idx], i[idx]
                pu, qi = self.P[uu], self.Q[ii]
                err = r[idx] - (self.mu + self.bu[uu] + self.bi[ii] + pu @ qi)
                sq += err * err
                self.bu[uu] += self.lr * (err - self.reg * self.bu[uu])
                self.bi[ii] += self.lr * (err - self.reg * self.bi[ii])
                self.P[uu] = pu + self.lr * (err * qi - self.reg * pu)
                self.Q[ii] = qi + self.lr * (err * pu - self.reg * qi)
            if verbose:
                print(f"  epoch {epoch + 1:2d}  train RMSE {np.sqrt(sq / len(r)):.4f}")
        return self

    def predict(self, u, i):
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        return np.clip(pred, 1, 5)


def bias_baseline(u, i, r, n_users, n_items, reg=10.0):
    """Damped user/item bias baseline: mu + b_u + b_i (no latent factors)."""
    mu = float(np.mean(r))
    bi = np.bincount(i, weights=r - mu, minlength=n_items) / (np.bincount(i, minlength=n_items) + reg)
    resid = r - mu - bi[i]
    bu = np.bincount(u, weights=resid, minlength=n_users) / (np.bincount(u, minlength=n_users) + reg)
    return lambda uu, ii: np.clip(mu + bu[uu] + bi[ii], 1, 5)
