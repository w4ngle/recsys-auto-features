"""Ручные признаки. Главное правило: признаки считаются ТОЛЬКО по событиям строго до среза (cutoff)."""
import pandas as pd
from .data import DAY

WINDOWS = (1, 7, 30, None)  # дни; None = за всё время. Этот список будем подбирать автоматически.


def window(L, a, b):
    """Прослушивания в [a, b): кандидаты, которых мы оцениваем."""
    return L[(L.timestamp >= a) & (L.timestamp < b)]


def build(L, K, cand, cutoff, windows=WINDOWS):
    """User / item / user-item признаки + конверсии + средняя доля прослушивания в прошлом."""
    HL, HK = L[L.timestamp < cutoff], K[K.timestamp < cutoff]
    assert (HL.timestamp < cutoff).all() and (HK.timestamp < cutoff).all()   # защита от утечки
    assert (cand.timestamp >= cutoff).all()

    X = cand[["uid", "item_id", "is_organic", "label"]].copy()
    for w in windows:
        lo = 0 if w is None else cutoff - w * DAY
        l, k = HL[HL.timestamp >= lo], HK[HK.timestamp >= lo]
        wn = "all" if w is None else f"{w}d"

        for ent, keys in [("u", ["uid"]), ("i", ["item_id"]), ("ui", ["uid", "item_id"])]:
            for ev, h in [("listens", l), ("likes", k)]:
                col = f"{ent}_{ev}_{wn}"
                c = h.groupby(keys).size().rename(col).reset_index()
                X = X.merge(c, on=keys, how="left")
                X[col] = X[col].fillna(0)

        for ent in ["u", "i"]:
            X[f"{ent}_cr_{wn}"] = X[f"{ent}_likes_{wn}"] / (X[f"{ent}_listens_{wn}"] + 1)

        for ent, key in [("u", "uid"), ("i", "item_id")]:
            col = f"{ent}_played_{wn}"
            c = l.groupby(key).played_ratio_pct.mean().rename(col).reset_index()
            X = X.merge(c, on=key, how="left")      # NaN оставляем, бустинг с ними справляется
    return X


def add_indirect(L, K, X, cutoff, maps, windows=WINDOWS):
    """Косвенные признаки: юзер x артист (uar_), юзер x альбом (ual_). L и K должны содержать artist_id/album_id."""
    HL, HK = L[L.timestamp < cutoff], K[K.timestamp < cutoff]
    names = {"artist_id": "ar", "album_id": "al"}
    for gcol, gname in names.items():
        if gcol not in maps:
            continue
        X[gcol] = X.item_id.map(maps[gcol]).fillna(-1).astype("int64")
        for w in windows:
            lo = 0 if w is None else cutoff - w * DAY
            wn = "all" if w is None else f"{w}d"
            for ev, h in [("listens", HL[HL.timestamp >= lo]), ("likes", HK[HK.timestamp >= lo])]:
                col = f"u{gname}_{ev}_{wn}"
                c = h.groupby(["uid", gcol]).size().rename(col).reset_index()
                X = X.merge(c, on=["uid", gcol], how="left")
                X[col] = X[col].fillna(0)
            X[f"u{gname}_cr_{wn}"] = X[f"u{gname}_likes_{wn}"] / (X[f"u{gname}_listens_{wn}"] + 1)
    return X


def build_train(L, K, t_val, maps=None, n_cuts=4, step=7, horizon=3600):
    """Обучающая выборка из нескольких недельных срезов перед t_val.
    Пары (юзер, трек), которые юзер уже лайкал до среза, выкидываем: новым лайком они стать не могут."""
    if not 0 <= horizon < step * DAY:
        raise ValueError("horizon must be shorter than the training window")
    parts = []
    for k in range(1, n_cuts + 1):
        c = t_val - k * step * DAY
        X = build(L, K, window(L, c, c + step * DAY - horizon), c)
        X = X[X.ui_likes_all == 0]
        if maps:
            X = add_indirect(L, K, X, c, maps)
        parts.append(X)
    return pd.concat(parts, ignore_index=True)
