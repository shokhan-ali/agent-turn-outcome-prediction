"""
Early prediction of user cancellation in AI coding-agent turns.
Trains and evaluates: prior, heuristic, logistic regression, LightGBM, GRU, and a GRU+LightGBM ensemble.
Writes results/*.json, results/*.csv and figures/*.png.
"""
import os, json, glob, time, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AGENT_DATA", os.path.join(HERE, "data"))
RES = HERE; os.makedirs(RES, exist_ok=True)
SEED = 0
TRAIN_DAYS = ["2026-06-01", "2026-06-02", "2026-06-03"]
VAL_DAYS = ["2026-06-04"]
TEST_DAYS = ["2026-06-05", "2026-06-06", "2026-06-07"]
CAT = ["model", "last_tool_cat", "prev_outcome"]
DROP = {"turn", "session", "date", "y", "remaining_calls", "remaining_tokens"}


def load(days, target="cancel"):
    out = []
    for d in days:
        p = pd.read_parquet(os.path.join(DATA, f"prefix_{d}.parquet"))
        t = pd.read_parquet(os.path.join(DATA, f"turns_{d}.parquet"), columns=["turn", "outcome"])
        p = p.merge(t, on="turn", how="left")
        if target == "failure": p["y"] = (p.outcome == "Failure").astype(np.int8)
        out.append(p.drop(columns=["outcome"]))
    return pd.concat(out, ignore_index=True)


def ece(y, p, bins=15):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -1, 2
    idx = np.digitize(p, q[1:-1]); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    top = np.argsort(-p)[: max(1, int(0.05 * len(p)))]
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)) if p.min() >= 0 and p.max() <= 1 else None,
                ece=ece(y, p) if p.min() >= 0 and p.max() <= 1 else None,
                tpr_at_fpr10=float(np.interp(0.10, fpr, tpr)), prec_top5=float(y[top].mean()),
                base_rate=float(y.mean()), n=int(len(y)))


def boot_auc_diff(y, p1, p2, groups, B=200, seed=SEED):
    """Session-level (cluster) bootstrap of AUC(p1) - AUC(p2) using multinomial session weights."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True); G = inv.max() + 1
    d = []
    for _ in range(B):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[inv].astype(float)   # rows inherit their session's count
        if w[y == 1].sum() == 0: continue
        d.append(roc_auc_score(y, p1, sample_weight=w) - roc_auc_score(y, p2, sample_weight=w))
    d = np.array(d)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_le0=float((d <= 0).mean()))


def prep(df, cats=None):
    X = df[[c for c in df.columns if c not in DROP]].copy()
    for c in CAT:
        X[c] = X[c].astype("category") if cats is None else pd.Categorical(X[c].where(X[c].isin(cats[c])), categories=cats[c])
    for c in X.columns:
        if c not in CAT: X[c] = X[c].astype(np.float32)
    return X


def lr_matrix(X, cols=None, scaler=None):
    Z = X.drop(columns=CAT).to_numpy(dtype=np.float32, na_value=np.nan)
    Z = np.nan_to_num(np.sign(Z) * np.log1p(np.abs(Z)), nan=0.0)          # heavy tails
    D = pd.get_dummies(X[CAT].astype(str), dtype=np.float32)
    names = list(X.drop(columns=CAT).columns) + list(D.columns)
    M = pd.DataFrame(np.hstack([Z, D.to_numpy()]), columns=names)
    if cols is not None: M = M.reindex(columns=cols, fill_value=0.0)
    if scaler is None: scaler = StandardScaler().fit(M.values)
    return scaler.transform(M.values).astype(np.float32), list(M.columns), scaler


def fit_all(tr, va, te, Xtr, Xva, Xte, ytr, yva, yte, sfx, t0=None):
    t0 = t0 or time.time()
    P = {}
    # heuristic baseline: elapsed wall-clock time of the turn so far (longer = riskier)
    P["Heuristic (elapsed time)"] = te.elapsed_s.values.astype(float)

    # logistic regression (fitted on a random subsample of 600k training rows to bound memory)
    sub = np.sort(np.random.default_rng(SEED).permutation(len(tr))[:600_000])
    Mtr, cols, sc = lr_matrix(Xtr.iloc[sub]); Mva, _, _ = lr_matrix(Xva, cols, sc); Mte, _, _ = lr_matrix(Xte, cols, sc)
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
    lr.fit(Mtr, ytr[sub])
    P["Logistic regression"] = lr.predict_proba(Mte)[:, 1]
    p_lr_va = lr.predict_proba(Mva)[:, 1]

    # LightGBM (early stopping on validation day)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=SEED, verbose=-1, num_threads=2)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CAT); dva = lgb.Dataset(Xva, yva, reference=dtr)
    gbm = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    p_gbm_va = gbm.predict(Xva, num_iteration=gbm.best_iteration)
    p_gbm_te = gbm.predict(Xte, num_iteration=gbm.best_iteration)
    P["LightGBM"] = p_gbm_te
    gbm.save_model(os.path.join(RES, f"lightgbm_model{sfx}.txt"))
    print("lgbm best iter", gbm.best_iteration, "t", time.time() - t0, flush=True)

    # isotonic calibration of LightGBM on the validation day
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_gbm_va, yva)
    P["LightGBM (calibrated)"] = iso.predict(p_gbm_te)

    # GRU sequence model (+ ensemble) if sequences are available
    try:
        from agent_gru import train_gru
        p_gru_va, p_gru_te = train_gru(tr.iloc[sub].reset_index(drop=True), va, te, Mtr, Mva, Mte, DATA, TRAIN_DAYS, VAL_DAYS, TEST_DAYS, seed=SEED, tag=sfx)
        P["GRU (sequence)"] = p_gru_te
        iso_e = IsotonicRegression(out_of_bounds="clip").fit(0.5 * p_gbm_va + 0.5 * p_gru_va, yva)
        P["Ensemble (GRU + LightGBM)"] = iso_e.predict(0.5 * p_gbm_te + 0.5 * p_gru_te)
    except Exception as e:
        print("GRU skipped:", repr(e), flush=True)

    return P


def main(target):
    t0 = time.time()
    tr, va, te = load(TRAIN_DAYS, target), load(VAL_DAYS, target), load(TEST_DAYS, target)
    sfx = "" if target == "cancel" else "_failure"
    print("rows", len(tr), len(va), len(te), "pos rate", tr.y.mean(), va.y.mean(), te.y.mean(), flush=True)
    keepcols = ["turn", "session", "k", "y", "remaining_tokens", "remaining_calls", "elapsed_s"]
    Xtr = prep(tr); cats = {c: Xtr[c].cat.categories for c in CAT}
    Xva, Xte = prep(va, cats), prep(te, cats)
    ytr, yva, yte = tr.y.values, va.y.values, te.y.values
    tr, va, te = tr[keepcols], va[keepcols], te[keepcols]
    import gc; gc.collect()
    P = {}
    cache = os.path.join(RES, f"cache_preds{sfx}.npz")
    if os.path.exists(cache):
        P = np.load(cache, allow_pickle=True)["P"].item()
        gbm = lgb.Booster(model_file=os.path.join(RES, f"lightgbm_model{sfx}.txt"))

    if not P:
        P = fit_all(tr, va, te, Xtr, Xva, Xte, ytr, yva, yte, sfx, t0)
        gbm = lgb.Booster(model_file=os.path.join(RES, f"lightgbm_model{sfx}.txt"))
        np.savez(cache, P=np.array(P, dtype=object), best=0)
    # ---- overall and per-checkpoint metrics
    out = {"overall": {m: metrics(yte, p) for m, p in P.items()}, "per_k": {}}
    for k in sorted(te.k.unique()):
        msk = te.k.values == k
        out["per_k"][int(k)] = {m: metrics(yte[msk], p[msk]) for m, p in P.items()}
    groups = te.session.values
    out["boot"] = {f"LightGBM - {m}": boot_auc_diff(yte, P["LightGBM"], P[m], groups)
                   for m in P if m not in ("LightGBM", "LightGBM (calibrated)")}
    if "Ensemble (GRU + LightGBM)" in P:
        out["boot"]["Ensemble - LightGBM"] = boot_auc_diff(yte, P["Ensemble (GRU + LightGBM)"], P["LightGBM"], groups)
    out["lgbm_best_iter"] = int(gbm.current_iteration())
    out["splits"] = dict(train_rows=len(tr), val_rows=len(va), test_rows=len(te),
                         train_turns=int(tr.turn.nunique()), test_turns=int(te.turn.nunique()),
                         train_sessions=int(tr.session.nunique()), test_sessions=int(te.session.nunique()))
    json.dump(out, open(os.path.join(RES, f"model_results{sfx}.json"), "w"), indent=1)

    # ---- save test predictions for utility analysis and figures
    keep = te[["turn", "session", "k", "y", "remaining_tokens", "remaining_calls"]].copy()
    for m, p in P.items(): keep[m] = p
    keep.to_parquet(os.path.join(RES, f"test_predictions{sfx}.parquet"))

    # ---- SHAP explanations on a test sample
    import shap
    samp = Xte.sample(n=min(20000, len(Xte)), random_state=SEED)
    sv = shap.TreeExplainer(gbm).shap_values(samp)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(RES, f"shap_importance{sfx}.csv"))
    np.save(os.path.join(RES, f"shap_values{sfx}.npy"), sv); samp.to_parquet(os.path.join(RES, f"shap_sample{sfx}.parquet"))
    print(json.dumps(out["overall"], indent=1)); print(imp.head(15)); print("total time", time.time() - t0)


if __name__ == "__main__":
    import sys
    for target in (sys.argv[1:] or ["cancel", "failure"]):
        main(target)
