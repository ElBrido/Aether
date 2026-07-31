#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrenamiento X0: pretraining y SFT de chat con loss solo en respuesta.

  python train_aether_x.py --smoke
  python train_aether_x.py --mode chat --data chat.jsonl --steps 1000
  python train_aether_x.py --mode pretrain --cache kairos_tokens_es.bin --steps 5000

JSONL chat: {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
O: {"prompt":"...", "response":"..."}
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from aether_v4 import KairosTokenizer, amp_autocast, make_grad_scaler, lm_loss
from aether_x import AetherX, X0_CFG


class CacheDataset(IterableDataset):
    def __init__(self, path, seq_len, seed=42):
        self.path, self.seq_len, self.seed = path, seq_len, seed
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        hi = max(1, len(mm) - self.seq_len - 2)
        while True:
            i = int(rng.integers(0, hi))
            yield torch.from_numpy(np.asarray(mm[i:i + self.seq_len + 2], dtype=np.int64))


def format_messages(tok, messages):
    ids, labels = [], []
    for m in messages:
        role = m["role"].lower()
        tag = "<SYS>" if role == "system" else "<USR>" if role == "user" else "<AST>"
        head, body = tok.encode_with_specials(tag), tok.encode(m.get("content", ""))
        ids.extend(head + body)
        labels.extend([-100] * len(head) + (body if tag == "<AST>" else [-100] * len(body)))
    ids.append(3); labels.append(3 if any(m["role"].lower() == "assistant" for m in messages) else -100)
    return ids, labels


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=512):
        self.rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            messages = row.get("messages") or [
                {"role": "user", "content": row.get("prompt", "")},
                {"role": "assistant", "content": row.get("response", "")},
            ]
            ids, labels = format_messages(tok, messages)
            if len(ids) >= 3: self.rows.append((ids[:max_len], labels[:max_len]))
        if not self.rows: raise RuntimeError("chat.jsonl no contiene ejemplos validos")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def pad_chat(batch):
    n = max(len(x[0]) for x in batch)
    x = torch.zeros(len(batch), n, dtype=torch.long)
    y = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids); y[i, :len(labels)] = torch.tensor(labels)
    return x, y


def chat_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def build_model(tok, args, device):
    cfg = dict(X0_CFG, hidden_dim=args.hidden, ffn_dim=args.ffn,
               memory_slots=args.slots, scratch_size=args.scratch)
    return AetherX(max(args.vocab, tok.vocab_size), cfg).to(device)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = KairosTokenizer.load(args.tokenizer); model = build_model(tok, args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = make_grad_scaler(enabled=device == "cuda"); model.train()
    if args.mode == "chat":
        loader = DataLoader(ChatDataset(args.data, tok, args.seq_len), batch_size=args.batch,
                            shuffle=True, collate_fn=pad_chat)
    else:
        if not args.cache: raise RuntimeError("--cache es obligatorio en mode=pretrain")
        loader = DataLoader(CacheDataset(args.cache, args.seq_len, args.seed), batch_size=args.batch)
    it = iter(loader); out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        batch = next(it)
        if args.mode == "chat": x, target = batch; x, target = x.to(device), target.to(device)
        else:
            batch = batch.to(device); x, target = batch[:, :-2], batch[:, 1:-1]
        opt.zero_grad(set_to_none=True)
        with amp_autocast(enabled=device == "cuda"):
            logits = model(x)
            main = chat_loss(logits, target) if args.mode == "chat" else lm_loss(logits, target, 1e-4)
            loss = main + args.halt_weight * model.cell.route_penalty()
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step == 1 or step % args.log_every == 0:
            s = model.cell.stats()
            print(f"paso {step:>6}/{args.steps} | loss {main.item():.4f} | ppl {math.exp(min(main.item(), 20)):.1f} | deep {s['deep_ratio']:.2%}")
            model.cell.reset_stats()
        if step % args.save_every == 0 or step == args.steps:
            torch.save(dict(model=model.state_dict(), cfg=model.cfg, step=step,
                            tokenizer=args.tokenizer, mode=args.mode), out)
            print(f"checkpoint: {out}")
    return model, tok


def smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, cfg).to(device).train(); x = torch.randint(0, 128, (2, 24), device=device)
    y = m(x); loss = lm_loss(y[:, :-1], x[:, 1:]) + 1e-3 * m.cell.route_penalty(); loss.backward()
    print(f"train smoke | loss={loss.item():.4f} | params={m.count_parameters():,} | OK")


def main():
    ap = argparse.ArgumentParser("train AETHER-X X0")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--mode", choices=["pretrain", "chat"], default="chat")
    ap.add_argument("--data", default="chat.jsonl"); ap.add_argument("--cache", default=None)
    ap.add_argument("--tokenizer", default="kairos_tokenizer.json"); ap.add_argument("--out", default="checkpoints/x0.pt")
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=2048); ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--scratch", type=int, default=32); ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--halt-weight", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args()
    if args.smoke: smoke(); return
    train(args)


if __name__ == "__main__": main()
