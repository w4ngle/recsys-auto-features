"""User-macro ranking metrics; only users with at least one positive are evaluated."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def per_user_metrics(X, score, k=10):
    score = np.asarray(score, dtype=float)
    if k < 1 or score.ndim != 1 or len(score) != len(X) or not np.isfinite(score).all():
        raise ValueError("Expected k >= 1 and one finite score per row")
    if not X.label.isin([0, 1]).all() or X.uid.isna().any():
        raise ValueError("Expected binary labels and non-null user IDs")
    d = pd.DataFrame({"uid": X.uid.to_numpy(), "y": X.label.to_numpy(), "s": score})
    # Stable input order breaks exact ties without changing unequal scores.
    d = d.sort_values(["uid", "s"], ascending=[True, False], kind="stable")
    d["r"] = d.groupby("uid").cumcount() + 1
    npos = d.groupby("uid").y.sum()
    npos = npos[npos > 0]
    if npos.empty:
        raise ValueError("No users with positive labels")
    top = d[d.r <= k]
    dcg = (top.y / np.log2(top.r + 1)).groupby(top.uid).sum().reindex(npos.index, fill_value=0)
    ideal = np.cumsum(1 / np.log2(np.arange(2, k + 2)))
    idcg = ideal[np.minimum(npos.to_numpy(), k).astype(int) - 1]
    hits = top.groupby("uid").y.sum().reindex(npos.index, fill_value=0)
    return pd.DataFrame({f"ndcg@{k}": dcg / idcg, f"recall@{k}": hits / npos})


def rank_metrics(X, score, k=10):
    return {name: float(value) for name, value in per_user_metrics(X, score, k).mean().items()}


def report(name, X, score):
    r = {"AUC": float(roc_auc_score(X.label, score)), **rank_metrics(X, score)}
    print(name, {key: round(value, 4) for key, value in r.items()})
    return r


def paired_bootstrap(X, scores, ref, n=2000, k=10, seed=0):
    """Resample paired per-user NDCG differences; return draws for existing callers."""
    if ref not in scores or n < 1:
        raise ValueError("Reference must exist and n must be positive")
    metric = f"ndcg@{k}"
    values = {name: per_user_metrics(X, score, k)[metric] for name, score in scores.items()}
    delta = pd.DataFrame({name: value - values[ref] for name, value in values.items() if name != ref})
    rng = np.random.default_rng(seed)
    draws = np.empty((n, len(delta.columns)))
    for i in range(n):
        draws[i] = delta.iloc[rng.integers(len(delta), size=len(delta))].mean().to_numpy()
    return {name: draws[:, i].tolist() for i, name in enumerate(delta.columns)}
