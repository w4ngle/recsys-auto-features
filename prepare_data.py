"""Rebuild caches with a one-hour label observation gap. Run from project root."""
from pathlib import Path
from src.data import load_yambda, sample_users, make_labels, load_mapping, add_groups, time_split
from src.features import build_train, build, add_indirect, window
from src.auto_features import build_train_auto, build_auto


def candidate_window(L, start, end, horizon=3600):
    if horizon < 0 or end - horizon <= start:
        raise ValueError('Insufficient observation interval for label horizon')
    return window(L, start, end - horizon)


def main():
    horizon = 3600
    L, K = sample_users(load_yambda('listens.parquet'), load_yambda('likes.parquet'), 5000, 42)
    L = make_labels(L, K, horizon=horizon)
    maps = {name: load_mapping(filename) for name, filename in (
        ('artist_id', 'artist_item_mapping.parquet'), ('album_id', 'album_item_mapping.parquet'))}
    L, K = add_groups(L, K, maps)
    t_val, t_test, t_end = time_split(L)
    out = Path('data/rebuilt')
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(build_train(L,K,t_val,maps=maps,horizon=horizon),
              build_train_auto(L,K,t_val,maps,horizon=horizon))]
    for start, end in ((t_val,t_test), (t_test,t_end+1)):
        cand = candidate_window(L, start, end, horizon)
        A = build(L,K,cand,start)
        A = add_indirect(L,K,A[A.ui_likes_all.eq(0)].copy(),start,maps)
        B = build_auto(L,K,cand,start,maps)
        pairs.append((A,B[B.ui_likes_all.eq(0)]))
    for stem, (A,B) in zip(('Xtr','Xva','Xte'),pairs):
        A, B = A.reset_index(drop=True), B.reset_index(drop=True)
        assert A[['uid','item_id','label']].equals(B[['uid','item_id','label']])
        A.to_parquet(out / f'{stem}_5k.parquet',index=False)
        B.to_parquet(out / f'{stem}_a_5k.parquet',index=False)
    print('Rebuilt tables:',out.resolve())

if __name__ == '__main__': main()
