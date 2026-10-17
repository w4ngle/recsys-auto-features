"""Автоматическая генерация признаков по шаблону.

Шаблон признака:  (сущность) x (событие) x (окно) [x (агрегат)]
  сущность: u (юзер), i (трек), ui (юзер-трек), uar (юзер-артист), ual (юзер-альбом)
  событие:  listens / likes
  окно:     любой набор дней из AUTO_WINDOWS (None = за всё время)
  агрегат:  count; плюс производные cr = likes / (listens + 1) и средняя доля прослушивания (для u и i)

Правило без утечек то же: история строго до cutoff.
"""
import numpy as np
import pandas as pd
from .data import DAY

AUTO_WINDOWS = (1, 3, 7, 14, 21, 30, 60, 90, 180, None)

ENTITIES = {
    "u":   ["uid"],
    "i":   ["item_id"],
    "ui":  ["uid", "item_id"],
    "uar": ["uid", "artist_id"],
    "ual": ["uid", "album_id"],
}


def _lookup(X, keys, series):
    """Быстро подставить значения series (индекс = ключи) в строки X без merge."""
    if len(keys) == 1:
        return X[keys[0]].map(series).values
    mi = pd.MultiIndex.from_frame(X[keys])
    return series.reindex(mi).values


def wname(w):
    return "all" if w is None else f"{w}d"


def build_auto(L, K, cand, cutoff, maps=None, windows=AUTO_WINDOWS, entities=None):
    entities = entities or ENTITIES
    HL, HK = L[L.timestamp < cutoff], K[K.timestamp < cutoff]
    assert HL.timestamp.max() < cutoff and HK.timestamp.max() < cutoff   # защита от утечки
    assert cand.timestamp.min() >= cutoff

    X = cand[["uid", "item_id", "is_organic", "label"]].copy()
    if maps:
        for gcol, mp in maps.items():
            X[gcol] = X.item_id.map(mp).fillna(-1).astype("int64")

    cols = {}
    for w in windows:
        lo = 0 if w is None else cutoff - w * DAY
        l, k = HL[HL.timestamp >= lo], HK[HK.timestamp >= lo]
        wn = wname(w)
        for ent, keys in entities.items():
            if not all(c in X.columns for c in keys):
                continue
            for ev, h in (("listens", l), ("likes", k)):
                s = h.groupby(keys).size()
                cols[f"{ent}_{ev}_{wn}"] = np.nan_to_num(_lookup(X, keys, s).astype("float64"), nan=0.0)
            cols[f"{ent}_cr_{wn}"] = cols[f"{ent}_likes_{wn}"] / (cols[f"{ent}_listens_{wn}"] + 1)
        for ent, key in (("u", "uid"), ("i", "item_id")):
            s = l.groupby(key).played_ratio_pct.mean()
            cols[f"{ent}_played_{wn}"] = _lookup(X, [key], s).astype("float64")

    F = pd.DataFrame(cols, index=X.index).astype("float32")
    return pd.concat([X, F], axis=1)


def build_train_auto(L, K, t_val, maps=None, n_cuts=4, step=7, windows=AUTO_WINDOWS):
    from .features import window
    parts = []
    for k in range(1, n_cuts + 1):
        c = t_val - k * step * DAY
        X = build_auto(L, K, window(L, c, c + step * DAY), c, maps, windows)
        parts.append(X[X.ui_likes_all == 0])          # убираем уже лайкнутые пары
    return pd.concat(parts, ignore_index=True)
