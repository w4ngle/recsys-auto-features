"""New, auditable search protocol; not a reconstruction of the historical study."""
import lightgbm as lgb
import optuna
from .metrics import rank_metrics


def tune(Xtr, Xva, feats, name, report_dir, n_trials=30, seed=42):
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    study = optuna.create_study(direction='maximize', study_name=name,
        storage=f'sqlite:///{report_dir / "optuna.sqlite3"}',
        sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', .01, .1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 127),
            min_child_samples=trial.suggest_int('min_child_samples', 20, 300),
            colsample_bytree=trial.suggest_float('colsample_bytree', .4, 1.),
            subsample=trial.suggest_float('subsample', .5, 1.),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10., log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10., log=True))
        model = lgb.LGBMClassifier(n_estimators=1000, subsample_freq=1,
                                  random_state=seed, n_jobs=4, verbosity=-1, **params)
        model.fit(Xtr[feats], Xtr.label, eval_set=[(Xva[feats], Xva.label)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr('best_iteration', model.best_iteration_)
        return rank_metrics(Xva, model.predict_proba(Xva[feats])[:, 1])['ndcg@10']

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    study.trials_dataframe().to_csv(report_dir / f'{name}_trials.csv', index=False)
    return study.best_params
