# Early Prediction of User Cancellation and Failure in Production AI Coding-Agent Turns

Code, trained models and results for the paper:

> **Omar, S.A.** *Can We See It Coming? Early Machine-Learning Prediction of User Cancellation and Failure in 1.2 Million Production AI Coding-Agent Turns.* 2026 (under review).

**Author:** Shokhan Ali Omar, Dukan Technical Institute, Sulaimani Polytechnic University, Dukan, Kurdistan Region, Iraq · shokhan.omar@spu.edu.iq

## Overview

AI coding agents complete a developer's request by running a loop of language-model calls and tool executions. Sometimes the developer stops the agent part-way (**cancellation**), and sometimes the turn ends in a system or model error (**failure**). Both waste inference capacity. This repository predicts, while a turn is still running, whether it will be cancelled or will fail. It uses only privacy-preserving telemetry (timings, token counts, tool names and outcomes); no prompts, code or responses are needed.

The study uses the public **GitHub Copilot Coding Agent Traces 2026** release (Azure Public Dataset): 301,026 sessions, 1,182,650 turns, 9.3 million LLM calls and 8.6 million tool calls recorded between 1 and 7 June 2026.

## Method in brief

- **At-risk formulation:** at checkpoint *k* ∈ {1, 2, 3, 4, 6, 8, 12, 16}, only turns that have issued more than *k* LLM calls and have not yet been cancelled are scored. Features use only events completed before the (*k*+1)-th call starts, so no information from the future leaks in (Proposition 1 in the paper).
- **Features:** 72 features in eight groups (progress, LLM timing, tokens, prompt structure, tools, errors, model and session context), plus a sequence of the last 48 events.
- **Models:** an elapsed-time heuristic, logistic regression, LightGBM, a GRU over the event sequence, and a GRU + LightGBM ensemble, all calibrated with isotonic regression.
- **Evaluation:** temporal split (train 1–3 June, validation 4 June, test 5–7 June); ROC-AUC, average precision, Brier score and ECE; session-level cluster-bootstrap confidence intervals; SHAP explanations; an upper-bound what-if analysis of early intervention.

## Main results (test set, 906,098 checkpoint rows)

| Target | Base rate | Best model | ROC-AUC | Average precision |
|---|---|---|---|---|
| User cancellation | 4.78% | Ensemble (GRU + LightGBM) | 0.691 | 0.104 (2.2× base rate) |
| Failure | 1.38% | LightGBM | 0.783 | 0.176 (12.7× base rate) |

Failures are much more predictable than cancellations. Flagging fewer than 1% of normal turns would catch about a third of failing turns and up to 15% of their remaining tokens (upper bound).

## Repository contents

All files are in the main folder.

| File | Purpose |
|---|---|
| `agent_parse.py` | Converts the raw JSONL shards into per-day turn, prefix-feature and sequence files |
| `agent_characterise.py` | Descriptive statistics and Figure 2 |
| `agent_models.py` | Heuristic, logistic regression, LightGBM, ensemble, bootstrap and SHAP |
| `agent_gru.py` | GRU sequence model (imported by `agent_models.py`) |
| `agent_figures.py` | Figures 3–7 and the what-if utility analysis |
| `agent_fig1.py` | Figure 1 (overview diagram) |
| `model_results*.json`, `characterisation.json`, `utility_summary.json` | Metrics reported in the paper |
| `shap_importance*.csv`, `utility_curve*.csv` | SHAP importance and utility curves |
| `lightgbm_model*.txt`, `gru_model*.pt` | Trained models (no suffix = cancellation; `_failure` = failure) |
| `fig1_overview.png` … `fig7_utility.png` | Figures used in the paper |

The processed data is written to a `data/` folder that `agent_parse.py` creates; it is not included because of its size.

## How to reproduce

1. **Install the requirements** (Python 3.11 or newer):

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the raw traces.** Get the release `ghcp-coding-agent-2026` from the Azure Public Dataset repository (https://github.com/Azure/AzurePublicDataset). Keep its folder layout, which looks like `date=2026-06-01/shard-00000.jsonl.gz`. The compressed data is about 3.1 GB.

3. **Parse the traces** into processed files (about 16 minutes on a two-core machine):

   ```bash
   python agent_parse.py /path/to/ghcp-coding-agent-2026 data
   ```

4. **Characterise the data** (Figure 2):

   ```bash
   python agent_characterise.py
   ```

5. **Train and evaluate all models** for both targets:

   ```bash
   python agent_models.py cancel failure
   ```

6. **Draw the figures and run the what-if analysis:**

   ```bash
   python agent_figures.py
   python agent_fig1.py
   ```

If the processed data is stored somewhere other than `data/`, set the environment variable `AGENT_DATA` to that folder. All random seeds are fixed. Peak memory use is about 6 GB.

## Using the trained models

```python
import lightgbm as lgb
booster = lgb.Booster(model_file="lightgbm_model_failure.txt")
# X: a DataFrame with the 72 features produced by agent_parse.py
# p = booster.predict(X)
```

## Data licence and citation

The traces are released by Microsoft under a CC-BY licence. If you use them, please cite: Liu, B., et al., *Agentic coding in the wild: Characterizing GitHub Copilot traces at production scale.* arXiv preprint arXiv:2608.00101, 2026.

If you use this code, please cite the paper above.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`).
