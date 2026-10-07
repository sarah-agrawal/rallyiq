"""A small causal Transformer that reads a rally shot by shot.

At shot t it can only look at shots 1..t (a causal mask), so it never sees the future.
It predicts the same 3 outcomes as the XGBoost model, but with the whole rally history.
"""
import numpy as np
import torch
import torch.nn as nn

MAX_LEN = 72


class RallyTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.position = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x, pad_mask):
        length = x.shape[1]
        pos = torch.arange(length, device=x.device)
        h = self.input(x) + self.position(pos)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(h)


def make_sequences(X_scaled, y, rally_ids):
    """Group shot rows (already in rally order) into one sequence per rally."""
    import pandas as pd
    groups = pd.Series(np.arange(len(rally_ids))).groupby(np.asarray(rally_ids), sort=False).groups
    seqs, labels, rows = [], [], []
    for idx in groups.values():
        idx = np.asarray(idx)[:MAX_LEN]
        seqs.append(X_scaled[idx])
        labels.append(y[idx])
        rows.append(idx)
    return seqs, labels, rows


def pad_batch(seqs, labels, device):
    n, length, f = len(seqs), max(len(s) for s in seqs), seqs[0].shape[1]
    x = np.zeros((n, length, f), dtype=np.float32)
    yb = np.full((n, length), -100, dtype=np.int64)
    pad = np.ones((n, length), dtype=bool)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        x[i, :len(s)] = s
        yb[i, :len(l)] = l
        pad[i, :len(s)] = False
    return (torch.tensor(x, device=device), torch.tensor(yb, device=device),
            torch.tensor(pad, device=device))


def train_transformer(train_seqs, train_labels, val_seqs, val_labels, n_features,
                      epochs=25, batch_size=64, lr=1e-3, device=None, seed=42, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RallyTransformer(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best, best_state, patience = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_seqs))
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            x, yb, pad = pad_batch([train_seqs[i] for i in b], [train_labels[i] for i in b], device)
            logits = model(x, pad)
            loss = loss_fn(logits.reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss = evaluate_loss(model, val_seqs, val_labels, device, loss_fn)
        log(f"epoch {epoch + 1:2d}  val loss {val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return model, device


@torch.no_grad()
def evaluate_loss(model, seqs, labels, device, loss_fn, batch_size=256):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(seqs), batch_size):
        x, yb, pad = pad_batch(seqs[start:start + batch_size], labels[start:start + batch_size], device)
        logits = model(x, pad)
        n = (yb != -100).sum().item()
        total += loss_fn(logits.reshape(-1, 3), yb.reshape(-1)).item() * n
        count += n
    return total / count


@torch.no_grad()
def predict(model, seqs, device, batch_size=256):
    model.eval()
    out = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        x, _, pad = pad_batch(chunk, [np.zeros(len(s), dtype=np.int64) for s in chunk], device)
        probs = torch.softmax(model(x, pad), dim=-1).cpu().numpy()
        out.extend(probs[i, :len(s)] for i, s in enumerate(chunk))
    return out
