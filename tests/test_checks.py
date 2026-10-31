import unittest
import numpy as np
import pandas as pd
from src.metrics import rank_metrics, paired_bootstrap
from src.features import build
from src.auto_features import build_auto

class Checks(unittest.TestCase):
    def test_metrics(self):
        X = pd.DataFrame({'uid': [1,1,2,2,3], 'label': [1,0,0,1,0]})
        good = np.array([1,0,0,1,0])
        bad = 1-good
        self.assertEqual(rank_metrics(X, good, 1), {'ndcg@1':1., 'recall@1':1.})
        self.assertEqual(rank_metrics(X, bad, 1)['ndcg@1'], 0.)
        b = paired_bootstrap(X, {'good':good,'bad':bad}, 'good', n=20,k=1)
        np.testing.assert_equal(b['bad'], [-1.]*20)
        self.assertEqual(rank_metrics(X, np.zeros(5), 1)['ndcg@1'], .5)
        with self.assertRaises(ValueError): rank_metrics(X, [np.nan]*5)
        with self.assertRaises(ValueError): rank_metrics(X.assign(label=0), good)

    def test_features_and_cutoff(self):
        L = pd.DataFrame({'uid':[1,1,2], 'item_id':[10,10,20], 'timestamp':[1,10,10],
                          'label':[0,1,0], 'is_organic':[1,0,1], 'played_ratio_pct':[50,90,80]})
        K = pd.DataFrame({'uid':[1], 'item_id':[10], 'timestamp':[10]})
        cand = L[L.timestamp >= 10]
        a = build(L,K,cand,10,windows=(1,None))
        b = build_auto(L,K,cand,10,windows=(1,None))
        for c in a.columns:
            np.testing.assert_allclose(a[c], b[c], equal_nan=True)
        self.assertEqual(a.u_likes_all.sum(),0)
        self.assertEqual(a.u_listens_all.sum(),1)


class BoundaryChecks(unittest.TestCase):
    def test_label_horizon_gap(self):
        from prepare_data import candidate_window
        L = pd.DataFrame({'timestamp':[0, 6399, 6400, 9999, 10000]})
        self.assertEqual(candidate_window(L,0,10000).timestamp.tolist(),[0,6399])
        with self.assertRaises(ValueError): candidate_window(L,0,100)

class TrainBoundaryChecks(unittest.TestCase):
    def test_train_builders_exclude_label_horizon(self):
        from src.features import build_train
        from src.auto_features import build_train_auto
        L = pd.DataFrame({'uid':[1]*4, 'item_id':[10]*4,
                          'timestamp':[1, 15000, 19000, 20000],
                          'label':[0]*4,'is_organic':[1]*4,'played_ratio_pct':[50]*4})
        K = pd.DataFrame({'uid':pd.Series(dtype='int64'),
                          'item_id':pd.Series(dtype='int64'),
                          'timestamp':pd.Series(dtype='int64')})
        # cutoff=10000; window ends at 20000, observed labels require 3600 seconds.
        step = 10000/86400
        A = build_train(L,K,20000,n_cuts=1,step=step)
        B = build_train_auto(L,K,20000,n_cuts=1,step=step)
        self.assertEqual(len(A),1)
        self.assertEqual(len(B),1)

if __name__ == '__main__': unittest.main()
