"""AUC и метрики ранжирования внутри каждого юзера."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rank_metrics(X, score, k=10):
    """NDCG@k и Recall@k: для каждого юзера ранжируем его кандидатов по score; берём только юзеров с позитивом."""
    d = pd.DataFrame({"uid": X.uid.values, "y": X.label.values, "s": np.asarray(score, dtype="float64")})
    d["s"] += np.random.default_rng(0).random(len(d)) * 1e-9          # случайный разбор ничьих
    d = d.sort_values(["uid", "s"], ascending=[True, False])
    d["r"] = d.groupby("uid").cumcount() + 1

    npos = d.groupby("uid").y.sum()
    npos = npos[npos > 0]
    top = d[(d.r <= k) & d.uid.isin(npos.index)]

    dcg = (top.y / np.log2(top.r + 1)).groupby(top.uid).sum().reindex(npos.index).fillna(0)
    ideal = np.cumsum(1 / np.log2(np.arange(2, k + 2)))
    idcg = pd.Series(ideal[np.minimum(npos.values, k).astype(int) - 1], index=npos.index)
    hits = top.y.groupby(top.uid).sum().reindex(npos.index).fillna(0)
    return {f"ndcg@{k}": float((dcg / idcg).mean()), f"recall@{k}": float((hits / npos).mean())}


def report(name, X, score):
    r = {"AUC": float(roc_auc_score(X.label, score))}
    r.update(rank_metrics(X, score))
    print(f"{name:32s}", {k: round(v, 4) for k, v in r.items()})
    return r
