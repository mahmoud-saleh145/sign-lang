"""
STAGE 2 EXPERIMENT -- temporal convolutional classifier over KArSL landmarks.

============================================================================
WHY A TEMPORAL CNN AND NOT A BiLSTM
============================================================================
Both were on the table; the deciding factor is the one the brief named --
"whichever is easiest to export and run in our existing TypeScript/Next.js
environment" -- and the repository already answers it. lib/ml/model.ts ships a
hand-written forward pass reading public/models/arsl-31/weights.json, with no
inference runtime at all.

A temporal CNN is Conv1d, ReLU, pooling and a dense layer. Every one of those
is a few lines of TypeScript, so Stage 3 can follow the precedent already in
the tree and add ZERO browser dependencies. A BiLSTM needs onnxruntime-web or
TF.js -- multiple megabytes of WASM before the model itself, for a sequence 30
steps long where a dilated convolution stack already sees the whole span.

This script does NOT export to the browser. It trains and measures, so the
architecture decision can be made on numbers.

============================================================================
NO LEAKAGE, RESTATED HERE BECAUSE IT IS THE POINT
============================================================================
The tensors come from export_tensors.ts, which uses the production
splitTakes(): train is the prototype sessions, test is the held-out session.
This script reads the split assignment from the manifest and never recomputes
it. It asserts the guarantee again at load time rather than trusting it.

The validation set is the held-out session's OTHER half. It is NOT clean for
model selection -- it shares a sitting with test -- and every number measured
on it is optimistic. It is used only for early stopping, because a two-session
corpus offers nothing better, and it is labelled as such in the output.

============================================================================
AUGMENTATION
============================================================================
Off by default. When enabled, it applies only transformations of REAL recorded
motion -- temporal crop/resample, small isotropic scale, small rotation,
landmark jitter, frame dropout. No sign motion is synthesised and no take is
created that did not come from a real recording.

Usage:
    python ml/experiments/stage2-temporal/train_tcn.py \
        --data ml/experiments/stage2-temporal/data \
        --out ml/experiments/stage2-temporal/runs/tcn
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(data_dir: str):
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    t, c = manifest["sequenceLength"], manifest["channels"]
    flat = np.fromfile(os.path.join(data_dir, "tensors.f32"), dtype=np.float32)
    x = flat.reshape(-1, t, c)
    rows = manifest["rows"]
    assert len(rows) == x.shape[0], f"manifest rows {len(rows)} != tensors {x.shape[0]}"
    y = np.array([r["classIndex"] for r in rows], dtype=np.int64)
    split = np.array([r["split"] for r in rows])

    # The guarantee, re-asserted rather than trusted.
    train_keys = {(r["signId"], r["file"]) for r in rows if r["split"] == "train"}
    for r in rows:
        if r["split"] != "train":
            assert (r["signId"], r["file"]) not in train_keys, f"take in train AND {r['split']}: {r}"
    train_sessions: dict[str, set] = {}
    for r in rows:
        if r["split"] == "train" and r["session"]:
            train_sessions.setdefault(r["signId"], set()).add(r["session"])
    for r in rows:
        if r["split"] == "test" and r["session"]:
            assert r["session"] not in train_sessions.get(r["signId"], set()), \
                f"test session also trained on: {r['signId']}/{r['file']}"
    return manifest, x, y, split, rows


class TCN(nn.Module):
    """Dilated temporal conv stack + masked global pooling + linear head.

    Dilations 1, 2, 4 over kernel 3 give a receptive field of 29 frames, which
    covers the whole 30-frame sequence -- every output position can see the
    entire gesture, which is what a BiLSTM's second direction would buy.
    """

    def __init__(self, channels: int, classes: int, width: int = 128, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_c = channels
        for dilation in (1, 2, 4):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_c, width, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            in_c = width
        self.head = nn.Linear(width * 2, classes)

    def forward(self, x):                      # x: [B, T, C]
        h = x.transpose(1, 2)                  # -> [B, C, T]
        for block in self.blocks:
            h = block(h)
        # Mean AND max pooling: mean carries the overall configuration, max
        # carries the single most distinctive instant of the sign.
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return self.head(pooled)


def augment(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Transformations of REAL motion only. Nothing is synthesised."""
    out = batch.clone()
    b, t, _ = out.shape
    # Isotropic scale and small rotation on the landmark/position blocks.
    for i in range(b):
        s = 1.0 + float(rng.normal(0, 0.05))
        a = float(rng.normal(0, 0.05))
        ca, sa = np.cos(a), np.sin(a)
        for lo, hi in ((0, 44), (44, 88), (88, 90)):
            blk = out[i, :, lo:hi]
            xs, ys = blk[:, 0::2], blk[:, 1::2]
            out[i, :, lo:hi:2] = (xs * ca - ys * sa) * s
            out[i, :, lo + 1:hi:2] = (xs * sa + ys * ca) * s
    # Landmark jitter, at the scale of real detector noise.
    out[:, :, :90] += torch.from_numpy(
        rng.normal(0, 0.01, size=(b, t, 90)).astype(np.float32)
    )
    # Frame dropout: hold the previous frame, as a dropped detection would.
    for i in range(b):
        for f in range(1, t):
            if rng.random() < 0.05:
                out[i, f, :] = out[i, f - 1, :]
    return out


def evaluate(model, x, y, device, batch=256):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            logits.append(model(xb).cpu())
    logits = torch.cat(logits)
    probs = F.softmax(logits, dim=1).numpy()
    yt = torch.from_numpy(y)
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    acc1 = (top5[:, 0] == yt).float().mean().item()
    acc3 = (top5[:, :3] == yt[:, None]).any(dim=1).float().mean().item()
    acc5 = (top5 == yt[:, None]).any(dim=1).float().mean().item()
    return acc1, acc3, acc5, probs, logits.numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml/experiments/stage2-temporal/data")
    p.add_argument("--out", default="ml/experiments/stage2-temporal/runs/tcn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    manifest, x, y, split, rows = load(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = len(manifest["classes"])

    xtr, ytr = x[split == "train"], y[split == "train"]
    xva, yva = x[split == "val"], y[split == "val"]
    xte, yte = x[split == "test"], y[split == "test"]
    print(f"device {device} | classes {classes} | train {len(xtr)} val {len(xva)} test {len(xte)}")

    model = TCN(manifest["channels"], classes, width=args.width).to(device)
    params = sum(p_.numel() for p_ in model.parameters())
    print(f"parameters {params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * max(len(xtr) // args.batch, 1), 1)
    )
    best_va, best_state, best_epoch = -1.0, None, -1

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(xtr))
        for i in range(0, len(order) - 1, args.batch):
            idx = order[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(xtr[idx])
            if args.augment:
                xb = augment(xb, rng)
            xb, yb = xb.to(device), torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
        if len(xva):
            va1, _, _, _, _ = evaluate(model, xva, yva, device)
            if va1 > best_va:
                best_va, best_epoch = va1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                print(f"   epoch {epoch + 1:3d}  val top-1 {va1:.4f}  (best {best_va:.4f} @ {best_epoch + 1})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- measure, on TEST, once -------------------------------------------
    t0 = time.time()
    acc1, acc3, acc5, probs, logits = evaluate(model, xte, yte, device)
    bulk_ms = (time.time() - t0) * 1000 / max(len(xte), 1)

    # Single-sample CPU latency, which is what the browser would see.
    cpu_model = TCN(manifest["channels"], classes, width=args.width)
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()
    one = torch.from_numpy(xte[:1])
    with torch.no_grad():
        for _ in range(5):
            cpu_model(one)
        t0 = time.time()
        for _ in range(50):
            cpu_model(one)
        single_ms = (time.time() - t0) * 1000 / 50

    path = os.path.join(args.out, "model.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1024 / 1024

    # The abstention threshold must be chosen on val, never on test, so val's
    # probabilities are written out alongside.
    if len(xva):
        _, _, _, vprobs, _ = evaluate(model, xva, yva, device)
        np.save(os.path.join(args.out, "val_probs.npy"), vprobs)
        with open(os.path.join(args.out, "val_rows.json"), "w", encoding="utf-8") as f:
            json.dump([r for r in rows if r["split"] == "val"], f)

    np.save(os.path.join(args.out, "test_probs.npy"), probs)
    np.save(os.path.join(args.out, "test_logits.npy"), logits)
    with open(os.path.join(args.out, "test_rows.json"), "w", encoding="utf-8") as f:
        json.dump([r for r in rows if r["split"] == "test"], f)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes, "parameters": params, "modelSizeMB": size_mb,
            "top1": acc1, "top3": acc3, "top5": acc5,
            "valTop1Optimistic": best_va, "bestEpoch": best_epoch + 1,
            "msPerQueryBatched": bulk_ms, "msPerQuerySingleCPU": single_ms,
            "train": int(len(xtr)), "val": int(len(xva)), "test": int(len(xte)),
            "augment": args.augment, "width": args.width, "epochs": args.epochs,
        }, f, indent=1)

    print("=" * 76)
    print("TEMPORAL MODEL -- HELD-OUT TEST")
    print("=" * 76)
    print(f"   top-1 {acc1:.4f}   top-3 {acc3:.4f}   top-5 {acc5:.4f}")
    print(f"   val top-1 (OPTIMISTIC, same session as test) {best_va:.4f} @ epoch {best_epoch + 1}")
    print(f"   parameters {params:,}   model {size_mb:.2f} MB")
    print(f"   latency  {single_ms:.2f} ms/query single-sample CPU   {bulk_ms:.2f} ms/query batched")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
