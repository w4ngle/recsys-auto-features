"""Обучение LightGBM и оценка на val/test."""
import lightgbm as lgb
from sklearn.metrics import log_loss
from .metrics import report

NON_FEATURES = ("uid", "item_id", "label", "artist_id", "album_id")


def feature_columns(X, exclude_prefixes=()):
    return [c for c in X.columns if c not in NON_FEATURES and not c.startswith(tuple(exclude_prefixes))]


def fit_eval(name, feats, Xtr, Xva, Xte, params=None):
    p = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    p.update(params or {})
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr[feats], Xtr.label, eval_set=[(Xva[feats], Xva.label)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    out = {}
    for nm, X in [("val", Xva), ("test", Xte)]:
        s = m.predict_proba(X[feats])[:, 1]
        out[nm] = report(f"{name} [{nm}]", X, s)
        out[nm]["logloss"] = float(log_loss(X.label, s))
    return m, out
