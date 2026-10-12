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


def paired_bootstrap(X, scores, ref, n=100, k=10, seed=0):
    """Разница NDCG@k между каждой моделью и ref с 95%-интервалом (пересэмплирование юзеров)."""
    rng = np.random.default_rng(seed)
    d = X[["uid", "label"]].reset_index(drop=True).copy()
    for nm, s in scores.items():
        d[nm] = np.asarray(s)
    groups = {u: g for u, g in d.groupby("uid")}
    users = np.array(list(groups))
    diffs = {nm: [] for nm in scores if nm != ref}
    for _ in range(n):
        pick = rng.choice(users, len(users))
        Xb = pd.concat([groups[u].assign(uid=i) for i, u in enumerate(pick)], ignore_index=True)
        base_v = rank_metrics(Xb, Xb[ref].values, k)[f"ndcg@{k}"]
        for nm in diffs:
            diffs[nm].append(rank_metrics(Xb, Xb[nm].values, k)[f"ndcg@{k}"] - base_v)
    for nm, v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{nm:14s} - {ref}: среднее {np.mean(v):+.4f}, 95% интервал [{lo:+.4f}, {hi:+.4f}]")
    return diffs
