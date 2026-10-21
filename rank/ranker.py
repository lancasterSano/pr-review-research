"""LightGBM lambdarank wrapper with PR groups and early stopping on validation."""
import lightgbm as lgb

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1, 3, 5], learning_rate=0.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=4,
              lambdarank_truncation_level=10)


def _groups(df):
    # rows must be contiguous per (repo, pr): sort first
    return df.groupby(["repo", "pr"], sort=False).size().to_numpy()


def _prep(df, features):
    df = df.sort_values(["repo", "pr"], kind="stable")
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return df, X, y, _groups(df)


def train(train_df, val_df, features, num_rounds=2000, early_stopping=100, params=None):
    p = dict(PARAMS); p.update(params or {})
    _, Xt, yt, gt = _prep(train_df, features)
    _, Xv, yv, gv = _prep(val_df, features)
    dtrain = lgb.Dataset(Xt, yt, group=gt, feature_name=features, free_raw_data=False)
    dval = lgb.Dataset(Xv, yv, group=gv, reference=dtrain, feature_name=features, free_raw_data=False)
    model = lgb.train(p, dtrain, num_boost_round=num_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    return model


def predict(model, df, features):
    return model.predict(df[features].to_numpy(dtype=float), num_iteration=model.best_iteration)


def importance(model, features, kind="gain"):
    vals = model.feature_importance(importance_type=kind)
    tot = vals.sum() or 1
    return {f: float(v / tot) for f, v in zip(features, vals)}
