"""GRU sequence model over the event stream observed before checkpoint k, fused with the static features."""
import os, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

L = 32          # events fed to the GRU (most recent ones before the checkpoint)
N_TOK = 64


def load_seq(data, days):
    ids, tok, num = [], [], []
    for d in days:
        z = np.load(os.path.join(data, f"seq_{d}.npz"))
        ids.append(z["ids"]); tok.append(z["tok"]); num.append(z["num"])
    ids = np.concatenate(ids)
    return {t: i for i, t in enumerate(ids)}, np.concatenate(tok), np.concatenate(num)


class Builder:
    """Builds right-aligned event windows for (turn, k) rows, masking events after the k-th LLM call."""
    def __init__(self, df, data, days):
        index, self.tok, self.num = load_seq(data, days)
        self.si = np.array([index[t] for t in df.turn.values]); self.k = df.k.values.astype(int)
        self.cum = np.cumsum(self.num[:, :, 4], axis=1)          # running count of LLM-call events

    def batch(self, rows):
        si, k = self.si[rows], self.k[rows]
        tok, num, cum = self.tok[si].astype(np.int64), self.num[si], self.cum[si]
        cnt = ((cum <= k[:, None]) & (tok > 0)).sum(1)          # events are time-ordered and contiguous
        idx = cnt[:, None] - L + np.arange(L)[None, :]
        valid = idx >= 0; idx = np.clip(idx, 0, tok.shape[1] - 1)
        T = np.take_along_axis(tok, idx, 1) * valid
        X = np.take_along_axis(num, idx[:, :, None], 1) * valid[:, :, None]
        return torch.from_numpy(T), torch.from_numpy(X.astype(np.float32))


class Net(nn.Module):
    def __init__(self, d_static, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(N_TOK, emb, padding_idx=0)
        self.gru = nn.GRU(emb + 5, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + d_static, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, T, X, S):
        h, _ = self.gru(torch.cat([self.emb(T), X], -1))
        return self.head(torch.cat([h[:, -1], S], -1)).squeeze(-1)


def predict(net, b, S, bs=4096):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(S), bs):
            r = np.arange(i, min(i + bs, len(S)))
            T, X = b.batch(r)
            out.append(torch.sigmoid(net(T, X, torch.from_numpy(S[r]).float())).numpy())
    return np.concatenate(out)


def train_gru(tr, va, te, Mtr, Mva, Mte, data, dtr, dva, dte, seed=0, epochs=3, max_train=600_000, bs=1024, tag=""):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    btr, bva, bte = Builder(tr, data, dtr), Builder(va, data, dva), Builder(te, data, dte)
    ytr = tr.y.values.astype(np.float32)
    net = Net(Mtr.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    pool = rng.permutation(len(tr))[:max_train]
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); perm = rng.permutation(pool)
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            T, X = btr.batch(r)
            loss = lossf(net(T, X, torch.from_numpy(Mtr[r]).float()), torch.from_numpy(ytr[r]))
            opt.zero_grad(); loss.backward(); opt.step()
        pva = predict(net, bva, Mva); auc = roc_auc_score(va.y.values, pva)
        print(f"GRU epoch {ep + 1}: val AUC {auc:.4f}", flush=True)
        if auc > best: best, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save(best_state, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gru_model{tag}.pt"))
    return predict(net, bva, Mva), predict(net, bte, Mte)
