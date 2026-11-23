"""
Parse the GitHub Copilot coding-agent traces (Azure Public Dataset, June 2026) into
  (1) a turn table   : one row per user turn (outcome, size, cost)            -> turns.parquet
  (2) a prefix table : one row per (turn, checkpoint k) with features computed
                       ONLY from the first k LLM calls of the turn and the tool
                       batches they triggered (no information from the future) -> prefix.parquet
  (3) event sequences: the first K_SEQ events of every turn, for the GRU model -> seq.npz

Usage: python agent_parse.py <trace_dir> <out_dir>
"""
import sys, os, json, gzip, glob, math
from multiprocessing import Pool
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

CHECKPOINTS = (1, 2, 3, 4, 6, 8, 12, 16)
K_SEQ = 48
TOOL_VOCAB = ["get_file", "run_command_in_terminal", "replace_string_in_file", "code_search", "run_build",
              "file_search", "apply_patch", "create_file", "update_plan_progress", "multi_replace_string_in_file",
              "get_errors", "edit_file", "get_files_in_project", "get_projects_in_solution", "find_symbol",
              "run_tests", "plan", "get_symbols_by_name", "finish_plan", "remove_file", "read_file"]
EDIT_TOOLS = {"replace_string_in_file", "apply_patch", "create_file", "multi_replace_string_in_file", "edit_file", "remove_file"}
EXEC_TOOLS = {"run_command_in_terminal", "run_build", "run_tests"}
READ_TOOLS = {"get_file", "code_search", "file_search", "read_file", "find_symbol", "get_symbols_by_name",
              "get_files_in_project", "get_projects_in_solution", "get_errors"}
MD_TYPES = ["History", "FunctionCalls", "Context", "System", "UserMessage", "IdeState",
            "RepositoryWideCustomInstructions", "AgentSkillsInstructions", "PathSpecificCustomInstructions"]


def ts(s):  # ISO timestamp -> seconds (float)
    return pd.Timestamp(s).value / 1e9 if s else math.nan


def slog(x):
    return math.log1p(x) if x == x and x > 0 else 0.0


def start_time(c):
    return ts(c["timestamp"]) - (c.get("duration_ms") or 0) / 1000


def tok(c, k):
    v = (c.get("tokens") or {}).get(k)
    return float(v) if v is not None else math.nan


def turn_outcome(calls, batches):
    if any(c.get("result") == "UserCancel" for c in calls) or any(b.get("result") == "UserCancel" for b in batches):
        return "UserCancel"
    last = calls[-1].get("result") if calls else None
    return last or "Unknown"


def prefix_features(calls, batches, k, sess_ctx, t0):
    """Features observable at the moment the (k+1)-th LLM call of the turn is issued."""
    pc = calls[:k]
    # Call timestamps mark call completion, so call k+1 starts at its timestamp minus its duration.
    t_obs = start_time(calls[k])                                   # observation time
    bts = [b for b in batches if ts(b["timestamp"]) + (b.get("duration_ms") or 0) / 1000 <= t_obs]
    tools = [fc for b in bts for fc in (b.get("function_calls") or [])]
    f = {"k": k}
    f["elapsed_s"] = t_obs - t0
    f["gap_last_s"] = t_obs - ts(pc[-1]["timestamp"])
    d = np.array([(c.get("duration_ms") or 0) / 1000 for c in pc])
    f["llm_time_s"], f["llm_dur_mean"], f["llm_dur_max"], f["llm_dur_last"] = d.sum(), d.mean(), d.max(), d[-1]
    tb = np.array([(b.get("duration_ms") or 0) / 1000 for b in bts]) if bts else np.zeros(1)
    f["tool_time_s"], f["tool_batch_max"] = tb.sum(), tb.max()
    f["n_batches"] = len(bts)
    f["n_tool_calls"] = len(tools)
    pr = np.array([tok(c, "prompt") for c in pc]); co = np.array([tok(c, "completion") for c in pc])
    ca = np.array([tok(c, "cached") for c in pc])
    f["prompt_first"], f["prompt_last"] = pr[0], pr[-1]
    f["prompt_growth"] = pr[-1] - pr[0]
    f["prompt_sum"], f["completion_sum"], f["completion_last"] = np.nansum(pr), np.nansum(co), co[-1]
    f["completion_mean"] = np.nanmean(co) if np.isfinite(co).any() else math.nan
    f["cached_pct_last"] = ca[-1] / pr[-1] if pr[-1] and np.isfinite(ca[-1]) else math.nan
    f["cached_pct_mean"] = np.nanmean(ca / np.where(pr > 0, pr, np.nan)) if np.isfinite(pr).any() else math.nan
    f["tokens_missing"] = int(np.isnan(pr).sum())
    md = pc[-1].get("message_metadata") or []
    f["md_segments_last"] = len(md)
    for t in MD_TYPES:
        f["md_" + t] = sum(1 for m in md if m.get("type") == t)
    f["llm_failures"] = sum(1 for c in pc if c.get("result") not in ("Success", None))
    f["tool_failures"] = sum(1 for t in tools if t.get("status") == 2)
    f["batch_failures"] = sum(1 for b in bts if b.get("result") not in ("Success", None))
    f["autocontinue_false"] = sum(1 for b in bts if b.get("auto_continue") is False)
    f["autocontinue_true"] = sum(1 for b in bts if b.get("auto_continue") is True)
    names = [t.get("name") for t in tools]
    for t in TOOL_VOCAB:
        f["tool_" + t] = names.count(t)
    f["tool_other"] = sum(1 for n in names if n not in TOOL_VOCAB)
    f["tools_distinct"] = len(set(names))
    f["n_edit"] = sum(1 for n in names if n in EDIT_TOOLS)
    f["n_exec"] = sum(1 for n in names if n in EXEC_TOOLS)
    f["n_read"] = sum(1 for n in names if n in READ_TOOLS)
    f["tool_repeat_frac"] = 1 - len(set(names)) / len(names) if names else 0.0
    last = names[-1] if names else "none"
    f["last_tool_cat"] = ("edit" if last in EDIT_TOOLS else "exec" if last in EXEC_TOOLS
                          else "read" if last in READ_TOOLS else "none" if last == "none" else "other")
    mods = [c.get("model") or "" for c in pc]
    f["model"] = mods[-1]
    f["model_switches"] = sum(1 for a, b in zip(mods, mods[1:]) if a != b)
    f.update(sess_ctx)
    return f


def seq_events(calls, batches, t_stop):
    """Time-ordered event tokens (LLM calls and tool calls) that occur before t_stop."""
    ev = []
    for c in calls:
        t = ts(c["timestamp"])          # completion time
        if t > t_stop: break
        p, q = tok(c, "prompt"), tok(c, "completion")
        ev.append((t, 0, 1 + min(MODELS.get(c.get("model") or "", 0), 39),
                   [slog((c.get("duration_ms") or 0) / 1000), slog(p) if p == p else 0.0,
                    slog(q) if q == q else 0.0, float(c.get("result") not in ("Success", None)), 1.0]))
    for b in batches:
        t = ts(b["timestamp"]) + (b.get("duration_ms") or 0) / 1000
        if t > t_stop: continue
        for fc in b.get("function_calls") or []:
            n = fc.get("name")
            ev.append((t, 1, 41 + (TOOL_VOCAB.index(n) if n in TOOL_VOCAB else len(TOOL_VOCAB)),
                       [slog((fc.get("duration_ms") or 0) / 1000), 0.0, 0.0, float(fc.get("status") == 2), 0.0]))
    ev.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in ev], [e[3] for e in ev]


MODELS = {f"Model {chr(65 + i)}": i + 1 for i in range(26)}
MODELS.update({f"Model {chr(65 + i)}{chr(65 + j)}": 27 + 26 * i + j for i in range(2) for j in range(26)})


def process_shard(path):
    turns, prefixes, seq_ids, seq_k, seq_tok, seq_num = [], [], [], [], [], []
    for line in gzip.open(path, "rt"):
        s = json.loads(line)
        date = s.get("date")
        prev_out, prev_end, prev_calls, n_prev_cancel = "none", None, 0, 0
        tl = []
        for t in s["turns"]:
            calls = sorted(t.get("llm_calls") or [], key=lambda c: c["timestamp"])
            if calls: tl.append((start_time(calls[0]), t, calls))
        tl.sort(key=lambda x: x[0])
        for ti, (t0, t, calls) in enumerate(tl):
            batches = t.get("tool_batches") or []
            bbr = {}
            for b in batches: bbr.setdefault(b.get("request_id"), []).append(b)
            out = turn_outcome(calls, batches)
            cancel_idx = next((i for i, c in enumerate(calls) if c.get("result") == "UserCancel"), None)
            t_end = max([ts(c["timestamp"]) for c in calls] +
                        [ts(b["timestamp"]) + (b.get("duration_ms") or 0) / 1000 for b in batches])
            tid = f"{s['session_id']}:{t['turn_id']}"
            n = len(calls)
            turns.append(dict(turn=tid, session=s["session_id"], date=date, turn_index=ti, n_calls=n,
                              n_tool_calls=sum(len(b.get("function_calls") or []) for b in batches),
                              prompt_tokens=np.nansum([tok(c, "prompt") for c in calls]),
                              completion_tokens=np.nansum([tok(c, "completion") for c in calls]),
                              cached_tokens=np.nansum([tok(c, "cached") for c in calls]),
                              llm_time_s=sum((c.get("duration_ms") or 0) for c in calls) / 1000,
                              tool_time_s=sum((b.get("duration_ms") or 0) for b in batches) / 1000,
                              duration_s=t_end - t0, outcome=out, cancel_idx=-1 if cancel_idx is None else cancel_idx,
                              model=calls[0].get("model") or "", hour=pd.Timestamp(t0, unit="s").hour))
            ctx = dict(turn_index=ti, prev_outcome=prev_out, prev_n_calls=prev_calls,
                       idle_before_s=(t0 - prev_end) if prev_end is not None else -1.0,
                       n_prev_cancels=n_prev_cancel, hour=pd.Timestamp(t0, unit="s").hour)
            y = int(out == "UserCancel")
            # observable prefix must end before the cancelling call (if any)
            limit = n - 1 if cancel_idx is None else cancel_idx
            for k in CHECKPOINTS:
                if k <= limit and k < n:   # turn still running after k calls (at risk)
                    f = prefix_features(calls, batches, k, ctx, t0)
                    f.update(turn=tid, session=s["session_id"], date=date, y=y,
                             remaining_calls=n - k,
                             remaining_tokens=float(np.nansum([tok(c, "prompt") for c in calls[k:]]) +
                                                    np.nansum([tok(c, "completion") for c in calls[k:]])))
                    prefixes.append(f)
            t_stop = start_time(calls[limit]) if limit < n else math.inf
            tk, nm = seq_events(calls, batches, t_stop)
            seq_ids.append(tid); seq_k.append(min(limit, n - 1))
            pad = K_SEQ - min(len(tk), K_SEQ)
            seq_tok.append((tk[:K_SEQ] + [0] * pad)); seq_num.append((nm[:K_SEQ] + [[0.0] * 5] * pad))
            prev_out, prev_end, prev_calls = out, t_end, n
            n_prev_cancel += int(out == "UserCancel")
    return (pd.DataFrame(turns), pd.DataFrame(prefixes),
            (seq_ids, np.array(seq_k, dtype=np.int16), np.array(seq_tok, dtype=np.int16), np.array(seq_num, dtype=np.float32)))


def shrink(df):
    for c in df.columns:
        if df[c].dtype == "float64": df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64": df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for day in sorted(glob.glob(os.path.join(src, "date=*"))):
        tag = os.path.basename(day).split("=")[1]
        if os.path.exists(os.path.join(dst, f"prefix_{tag}.parquet")): continue
        shards = sorted(glob.glob(os.path.join(day, "shard-*.jsonl.gz")))
        with Pool(2) as p:
            res = p.map(process_shard, shards, chunksize=2)
        shrink(pd.concat([r[0] for r in res], ignore_index=True)).to_parquet(os.path.join(dst, f"turns_{tag}.parquet"))
        shrink(pd.concat([r[1] for r in res], ignore_index=True)).to_parquet(os.path.join(dst, f"prefix_{tag}.parquet"))
        ids = sum([r[2][0] for r in res], [])
        np.savez_compressed(os.path.join(dst, f"seq_{tag}.npz"), ids=np.array(ids),
                            tok=np.concatenate([r[2][2] for r in res]), num=np.concatenate([r[2][3] for r in res]))
        print(tag, sum(len(r[0]) for r in res), "turns", sum(len(r[1]) for r in res), "prefix rows", flush=True)
        del res
