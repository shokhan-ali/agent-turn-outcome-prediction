"""Workload characterisation of user cancellation (RQ1). Writes results/characterisation.json and figures."""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AGENT_DATA", os.path.join(HERE, "data"))
RES = HERE; FIG = HERE
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Serif", "axes.grid": True, "grid.alpha": .3, "figure.dpi": 200})

t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "turns_*.parquet")))], ignore_index=True)
t["tokens"] = t.prompt_tokens + t.completion_tokens
t["cancel"] = (t.outcome == "UserCancel").astype(int)
C = {}
C["n_turns"] = int(len(t)); C["n_sessions"] = int(t.session.nunique())
C["n_llm_calls"] = int(t.n_calls.sum()); C["n_tool_calls"] = int(t.n_tool_calls.sum())
C["outcome_share"] = t.outcome.value_counts(normalize=True).round(5).to_dict()
C["token_share"] = (t.groupby("outcome").tokens.sum() / t.tokens.sum()).round(5).to_dict()
C["llm_time_share"] = (t.groupby("outcome").llm_time_s.sum() / t.llm_time_s.sum()).round(5).to_dict()
C["median_calls"] = t.groupby("outcome").n_calls.median().to_dict()
C["mean_calls"] = t.groupby("outcome").n_calls.mean().round(2).to_dict()
C["median_duration_s"] = t.groupby("outcome").duration_s.median().round(1).to_dict()
c = t[t.cancel == 1]
C["cancel_tokens_total"] = float(c.tokens.sum())
C["cancel_position_quartiles"] = np.percentile(c.cancel_idx + 1, [25, 50, 75, 90]).tolist()
C["cancel_at_first_call"] = float(((c.cancel_idx == 0)).mean())
C["cancel_after_k_calls"] = {k: float((c.cancel_idx >= k).mean()) for k in (1, 2, 4, 8, 16)}
# context effects
t = t.sort_values(["session", "turn_index"])
t["prev_cancel"] = t.groupby("session").cancel.shift(1)
C["rate_by_prev"] = t.groupby(t.prev_cancel.fillna(-1)).cancel.mean().round(4).to_dict()
C["rate_by_turn_index"] = t.groupby(t.turn_index.clip(upper=9)).cancel.mean().round(4).to_dict()
top = t.model.value_counts().index[:10]
C["rate_by_model"] = t[t.model.isin(top)].groupby("model").cancel.agg(["mean", "size"]).round(4).to_dict()
C["rate_by_day"] = t.groupby("date").cancel.mean().round(4).to_dict()
C["turns_by_day"] = t.groupby("date").size().to_dict()
json.dump(C, open(os.path.join(RES, "characterisation.json"), "w"), indent=1, default=float)
print(json.dumps(C, indent=1, default=float))

# Figure: cancellation position and rate by call count
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
pos = np.clip(c.cancel_idx + 1, 1, 40)
ax[0].hist(pos, bins=np.arange(0.5, 41.5), color="#e15759")
ax[0].set_xlabel("cancelled at LLM call (40 = 40+)"); ax[0].set_ylabel("cancelled turns")
grp = t.groupby(t.n_calls.clip(upper=40)).cancel.mean()
ax[1].plot(grp.index, 100 * grp.values, color="#4e79a7"); ax[1].set_xlabel("LLM calls per turn (40 = 40+)")
ax[1].set_ylabel("cancelled turns (%)")
r = t.groupby(t.turn_index.clip(upper=9)).cancel.mean()
ax[2].bar(r.index, 100 * r.values, color="#59a14f"); ax[2].set_xlabel("turn index (0 = first, 9 = 9+)")
ax[2].set_ylabel("cancelled turns (%)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_characterisation.png")); plt.close()
