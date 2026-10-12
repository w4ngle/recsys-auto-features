"""Загрузка Yambda, подвыборка пользователей, метки и сплит по времени."""
import numpy as np
import pandas as pd
from datasets import load_dataset

DAY = 86400  # timestamp в Yambda измеряется в секундах


def load_yambda(name, size="50m"):
    """name: listens.parquet, likes.parquet и т.д."""
    return load_dataset("yandex/yambda", data_dir=f"flat/{size}", data_files=name)["train"].to_pandas()


def load_mapping(fname):
    """artist_item_mapping.parquet / album_item_mapping.parquet -> Series item_id -> group_id."""
    mp = load_dataset("yandex/yambda", data_files=fname)["train"].to_pandas()
    group_col = [c for c in mp.columns if c != "item_id"][0]
    return mp.drop_duplicates("item_id").set_index("item_id")[group_col]


def sample_users(listens, likes, n_users=2000, seed=42):
    """Берём n_users случайных юзеров со ВСЕЙ их историей (а не случайные строки)."""
    rng = np.random.default_rng(seed)
    users = rng.choice(listens.uid.unique(), n_users, replace=False)
    return listens[listens.uid.isin(users)].copy(), likes[likes.uid.isin(users)].copy()


def make_labels(L, K, horizon=3600):
    """label=1, если юзер лайкнул трек в течение horizon секунд ПОСЛЕ начала прослушивания."""
    like_ts = K.groupby(["uid", "item_id"]).timestamp.min().rename("like_ts").reset_index()
    L = L.merge(like_ts, on=["uid", "item_id"], how="left")
    dt = L.like_ts.astype("float") - L.timestamp.astype("float")
    L["label"] = ((dt >= 0) & (dt <= horizon)).astype("int8")
    return L.drop(columns="like_ts")


def add_groups(L, K, maps):
    """Добавляем artist_id / album_id к прослушиваниям и лайкам. maps = {"artist_id": Series, ...}."""
    for gcol, mp in maps.items():
        L[gcol] = L.item_id.map(mp).fillna(-1).astype("int64")
        K[gcol] = K.item_id.map(mp).fillna(-1).astype("int64")
    return L, K


def time_split(L, val_days=14, test_days=14):
    """Границы сплита по времени: train < t_val <= val < t_test <= test."""
    t_end = int(L.timestamp.max())
    t_test = t_end - test_days * DAY
    t_val = t_test - val_days * DAY
    return t_val, t_test, t_end
