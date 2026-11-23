"""Figures (both targets side by side) and the what-if utility analysis (RQ4) from the saved test predictions."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

HERE = os.path.dirname(os.path.abspath(__file__))
RES = HERE; FIG = HERE; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Serif", "axes.grid": True, "grid.alpha": .3, "figure.dpi": 200})
C = {"Heuristic (elapsed time)": "#999999", "Logistic regression": "#edc948", "LightGBM": "#4e79a7",
     "GRU (sequence)": "#59a14f", "Ensemble (GRU + LightGBM)": "#e15759"}
T = {"": "User cancellation", "_failure": "Failure"}
R = {s: json.load(open(os.path.join(RES, f"model_results{s}.json"))) for s in T}
P = {s: pd.read_parquet(os.path.join(RES, f"test_predictions{s}.parquet")) for s in T}

# Fig 3: ROC and PR, rows = targets
fig, ax = plt.subplots(2, 2, figsize=(7, 5.6))
for r, s in enumerate(T):
    y = P[s].y.values
    for m in C:
        f, t, _ = roc_curve(y, P[s][m]); ax[r, 0].plot(f, t, color=C[m], lw=1.1, label=f"{m} ({R[s]['overall'][m]['auc']:.3f})")
        p, q, _ = precision_recall_curve(y, P[s][m]); ax[r, 1].plot(q, p, color=C[m], lw=1.1, label=f"{m} ({R[s]['overall'][m]['ap']:.3f})")
    ax[r, 0].plot([0, 1], [0, 1], "k--", lw=.6); ax[r, 1].axhline(y.mean(), color="k", ls="--", lw=.6)
    ax[r, 0].set_ylabel(f"{T[s]}\ntrue-positive rate"); ax[r, 1].set_ylabel("precision")
    ax[r, 0].legend(fontsize=5.5, title="ROC-AUC", title_fontsize=6); ax[r, 1].legend(fontsize=5.5, title="average precision", title_fontsize=6)
ax[1, 0].set_xlabel("false-positive rate"); ax[1, 1].set_xlabel("recall")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_roc_pr.png")); plt.close()

# Fig 4: AUC and AP lift by checkpoint
fig, ax = plt.subplots(2, 2, figsize=(7, 4.8))
for r, s in enumerate(T):
    ks = sorted(int(k) for k in R[s]["per_k"])
    for m in C:
        ax[r, 0].plot(ks, [R[s]["per_k"][str(k)][m]["auc"] for k in ks], "o-", ms=2.5, color=C[m], label=m)
        ax[r, 1].plot(ks, [R[s]["per_k"][str(k)][m]["ap"] / R[s]["per_k"][str(k)][m]["base_rate"] for k in ks], "o-", ms=2.5, color=C[m])
    ax[r, 1].axhline(1, color="k", ls="--", lw=.7)
    ax[r, 0].set_ylabel(f"{T[s]}\nROC-AUC"); ax[r, 1].set_ylabel("average precision / base rate")
    for a in ax[r]: a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels(ks)
ax[0, 0].legend(fontsize=5.5)
for a in ax[1]: a.set_xlabel("checkpoint k (LLM calls observed)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_by_checkpoint.png")); plt.close()

# Fig 5: SHAP importance (top 12) for both targets
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
for i, s in enumerate(T):
    imp = pd.read_csv(os.path.join(RES, f"shap_importance{s}.csv"), index_col=0).iloc[:, 0].head(12)[::-1]
    ax[i].barh(imp.index, imp.values, color="#4e79a7"); ax[i].set_title(T[s], fontsize=9)
    ax[i].set_xlabel("mean |SHAP value| (log-odds)"); ax[i].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_shap.png")); plt.close()

# Fig 6: calibration
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9))
for i, s in enumerate(T):
    y = P[s].y.values; lim = {"": .3, "_failure": .2}[s]
    for m, ls in [("LightGBM", "-"), ("LightGBM (calibrated)", "--"), ("Ensemble (GRU + LightGBM)", ":")]:
        fy, fx = calibration_curve(y, P[s][m], n_bins=15, strategy="quantile")
        ax[i].plot(fx, fy, ls, marker="o", ms=2.5, label=m)
    ax[i].plot([0, lim], [0, lim], "k:", lw=.7); ax[i].set_xlim(0, lim); ax[i].set_ylim(0, lim)
    ax[i].set_title(T[s], fontsize=9); ax[i].set_xlabel("predicted probability"); ax[i].set_ylabel("observed frequency")
ax[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_calibration.png")); plt.close()


def utility(df, score):
    """What-if (upper bound): intervene the first time a turn's risk exceeds a threshold."""
    df = df.sort_values(["turn", "k"])
    npos = df[df.y == 1].turn.nunique(); nneg = df[df.y == 0].turn.nunique()
    pos_tokens = df[df.y == 1].groupby("turn").remaining_tokens.max().sum()
    rows = []
    for thr in np.unique(np.quantile(df[score], np.linspace(0.70, 0.999, 60))):
        first = df[df[score] >= thr].groupby("turn").head(1)
        tp, fp = first[first.y == 1], first[first.y == 0]
        rows.append(dict(threshold=float(thr), flagged_turns=int(len(first)), recall_turns=len(tp) / npos,
                         precision_turns=len(tp) / max(1, len(first)), false_alarm_rate=len(fp) / nneg,
                         tokens_saved=float(tp.remaining_tokens.sum()),
                         tokens_saved_share=float(tp.remaining_tokens.sum() / pos_tokens)))
    return pd.DataFrame(rows)


best = {"": "Ensemble (GRU + LightGBM)", "_failure": "LightGBM (calibrated)"}
fig, ax = plt.subplots(figsize=(3.8, 2.9))
summary = {}
for s, col in zip(T, ["#e15759", "#4e79a7"]):
    U = utility(P[s], best[s]); U.to_csv(os.path.join(RES, f"utility_curve{s}.csv"), index=False)
    ax.plot(100 * U.false_alarm_rate, 100 * U.tokens_saved_share, "o-", ms=2, color=col, label=f"{T[s]} ({best[s]})")
    summary[T[s]] = {}
    for far in (0.01, 0.02, 0.05, 0.10):
        r = U.iloc[(U.false_alarm_rate - far).abs().argmin()]
        summary[T[s]][far] = {k: float(r[k]) for k in ("false_alarm_rate", "recall_turns", "precision_turns", "tokens_saved_share", "tokens_saved")}
ax.set_xlim(0, 20); ax.set_xlabel("turns flagged that end normally (%)"); ax.set_ylabel("post-flag tokens of target turns (%)")
ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_utility.png")); plt.close()
json.dump(summary, open(os.path.join(RES, "utility_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
