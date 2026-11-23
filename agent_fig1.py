"""Figure 1: a coding-agent turn, checkpoints and the prediction pipeline."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = HERE; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
def box(x, y, w, h, t, fc, ec, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
# (a) timeline of one turn
ax.text(1, 50, "(a) One agent turn and its checkpoints", fontsize=8, weight="bold")
ax.annotate("", (97, 38), (2, 38), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555"))
ax.text(97, 35.5, "time", ha="right", fontsize=6.5, color="#555")
x = 4; segs = [("LLM 1", 7, "llm"), ("tools", 6, "tool"), ("LLM 2", 6, "llm"), ("tools", 9, "tool"), ("LLM 3", 7, "llm"),
               ("tools", 5, "tool"), ("LLM 4", 8, "llm"), ("tools", 7, "tool"), ("LLM 5", 6, "llm")]
starts = []
for t, w, kind in segs:
    if kind == "llm": starts.append(x)
    box(x, 39.5, w - 0.6, 5, t, "#dbe7f5" if kind == "llm" else "#e9f5e6", "#4e79a7" if kind == "llm" else "#59a14f", 6.5)
    x += w
ax.text(x + 1, 42, "user presses\nStop", fontsize=6.5, color="#c0392b", va="center")
for k, s in zip([1, 2, 3, 4], starts[1:]):
    yy = 32.2 if k % 2 else 28.6
    ax.plot([s, s], [yy + 2.2, 46.5], ls="--", lw=0.8, color="#e15759")
    ax.text(s, yy, f"k={k}: call {k + 1} starts", ha="center", fontsize=5.8, color="#c0392b")
ax.text(4, 24.5, "At checkpoint k only completed events are used (Proposition 1). The model outputs the risk that the turn\nwill eventually be cancelled or fail.", fontsize=6.3)
# (b) pipeline
ax.text(1, 19.5, "(b) Prediction pipeline", fontsize=8, weight="bold")
steps = [("Production traces\n301k sessions\n1.18M turns", "#f5f5f5", "#777"),
         ("Leakage-free\nprefix features\n(72) + event\nsequences", "#dbe7f5", "#4e79a7"),
         ("Models: LR,\nLightGBM, GRU,\nensemble", "#fdf0ef", "#e15759"),
         ("Isotonic\ncalibration\n(validation day)", "#fff6dd", "#c9a227"),
         ("Risk score\n+ SHAP\nexplanation", "#e9f5e6", "#59a14f"),
         ("Intervention:\nconfirm,\nde-prioritise,\nrelease KV cache", "#f5f5f5", "#777")]
w, gap, y0 = 14.2, 2.0, 2
for i, (t, fc, ec) in enumerate(steps):
    xi = 1 + i * (w + gap); box(xi, y0, w, 13, t, fc, ec, 6.0)
    if i < len(steps) - 1: ax.annotate("", (xi + w + gap - 0.2, y0 + 6.5), (xi + w + 0.3, y0 + 6.5), arrowprops=dict(arrowstyle="-|>", lw=0.8))
fig.savefig(os.path.join(FIG, "fig1_overview.png"), bbox_inches="tight"); print("ok")
