#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento X0: pretraining y SFT de chat con loss solo en la respuesta.

  python train_aether_x.py --smoke
  python train_aether_x.py --mode chat --data chat.jsonl --steps 1000
  python train_aether_x.py --mode pretrain --cache kairos_tokens_es.bin --steps 5000

JSONL de chat, una linea por ejemplo:
  {"messages":[{"role":"system","content":"..."},
               {"role":"user","content":"..."},
               {"role":"assistant","content":"..."}]}
O la forma corta: {"prompt":"...", "response":"..."}
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

from aether_v4 import (EOS_ID, KairosTokenizer, amp_autocast, lm_loss,
                       make_grad_scaler)
from aether_x import AetherX, X0_CFG


def cycle(loader):
    """Repite el dataset para siempre.

    Sin esto, `next(it)` revienta con StopIteration en cuanto el chat.jsonl
    da una vuelta completa, que con pocos ejemplos pasa en segundos.
    """
    while True:
        for batch in loader:
            yield batch


class CacheDataset(IterableDataset):
    """Ventanas aleatorias sobre el cache binario de tokens ya validado."""

    def __init__(self, path, seq_len, seed=42):
        self.path, self.seq_len, self.seed = path, seq_len, seed
        self.n_tokens = Path(path).stat().st_size // 2

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        hi = max(1, len(mm) - self.seq_len - 2)
        while True:
            i = int(rng.integers(0, hi))
            yield torch.from_numpy(np.asarray(mm[i:i + self.seq_len + 2], dtype=np.int64))


def format_messages(tok, messages):
    """Etiqueta SOLO los tokens del assistant; el resto es contexto (-100)."""
    ids, labels = [], []
    has_assistant = False
    for m in messages:
        role = m.get("role", "user").lower()
        tag = "<SYS>" if role == "system" else "<USR>" if role == "user" else "<AST>"
        head = tok.encode_with_specials(tag)
        body = tok.encode(m.get("content", ""))
        ids.extend(head + body)
        if tag == "<AST>":
            has_assistant = True
            labels.extend([-100] * len(head) + body)
        else:
            labels.extend([-100] * (len(head) + len(body)))
    ids.append(EOS_ID)
    labels.append(EOS_ID if has_assistant else -100)
    return ids, labels


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=512):
        self.rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or [
                {"role": "user", "content": row.get("prompt", "")},
                {"role": "assistant", "content": row.get("response", "")},
            ]
            ids, labels = format_messages(tok, messages)
            ids, labels = ids[:max_len], labels[:max_len]
            # descarta ejemplos sin ningun token supervisado: solo meten ruido
            if len(ids) >= 3 and any(l != -100 for l in labels[1:]):
                self.rows.append((ids, labels))
        if not self.rows:
            raise RuntimeError(f"{path} no contiene ejemplos validos con respuesta")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def pad_chat(batch):
    n = max(len(x[0]) for x in batch)
    x = torch.zeros(len(batch), n, dtype=torch.long)          # PAD_ID = 0
    y = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x, y


def chat_loss(logits, labels):
    tgt = labels[:, 1:].reshape(-1)
    if int((tgt != -100).sum()) == 0:
        # cross_entropy con todo ignorado devuelve NaN y envenena el scaler
        return logits.sum() * 0.0
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           tgt, ignore_index=-100)


def lr_at(step, args):
    """Warmup lineal + coseno. Una celda recurrente sin warmup se desestabiliza."""
    if step <= args.warmup:
        return args.lr * step / max(1, args.warmup)
    p = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * p)))


def build_model(tok, args, device):
    cfg = dict(X0_CFG, hidden_dim=args.hidden, ffn_dim=args.ffn,
               memory_slots=args.slots, scratch_size=args.scratch)
    return AetherX(max(args.vocab, tok.vocab_size), cfg).to(device)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tok = KairosTokenizer.load(args.tokenizer)
    model = build_model(tok, args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    scaler = make_grad_scaler(enabled=device == "cuda")
    model.train()

    if args.mode == "chat":
        ds = ChatDataset(args.data, tok, args.seq_len)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            collate_fn=pad_chat)
        print(f"chat: {len(ds)} ejemplos | vocab {model.vocab_size} "
              f"| params {model.count_parameters():,}")
    else:
        if not args.cache:
            raise RuntimeError("--cache es obligatorio en mode=pretrain")
        ds = CacheDataset(args.cache, args.seq_len, args.seed)
        loader = DataLoader(ds, batch_size=args.batch)
        print(f"pretrain: {ds.n_tokens/1e6:.1f}M tokens en cache "
              f"| params {model.count_parameters():,}")

    it = cycle(loader)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(it)
        if args.mode == "chat":
            x, target = batch[0].to(device), batch[1].to(device)
        else:
            batch = batch.to(device)
            x, target = batch[:, :-2], batch[:, 1:-1]

        opt.zero_grad(set_to_none=True)
        with amp_autocast(enabled=device == "cuda"):
            logits = model(x)
            main = (chat_loss(logits, target) if args.mode == "chat"
                    else lm_loss(logits, target, 1e-4))
            loss = main + args.halt_weight * model.cell.route_penalty()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            s = model.cell.stats()
            print(f"paso {step:>6}/{args.steps} | loss {main.item():.4f} "
                  f"| ppl {math.exp(min(main.item(), 20)):>8.1f} "
                  f"| deep {s['deep_ratio']:.1%} | lr {lr:.2e}")
            model.cell.reset_stats()

        if step % args.save_every == 0 or step == args.steps:
            torch.save(dict(model=model.state_dict(), cfg=model.cfg, step=step,
                            tokenizer=args.tokenizer, mode=args.mode), out)
            print(f"checkpoint: {out}")

    return model, tok


def smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, cfg).to(device).train()
    x = torch.randint(0, 128, (2, 24), device=device)
    loss = lm_loss(m(x)[:, :-1], x[:, 1:]) + 1e-3 * m.cell.route_penalty()
    loss.backward()

    # el masking del chat tiene que cobrar SOLO la respuesta
    labels = torch.full((2, 24), -100, dtype=torch.long, device=device)
    labels[:, 12:] = x[:, 12:]
    cl = chat_loss(m(x), labels)
    empty = chat_loss(m(x), torch.full_like(labels, -100))
    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(cl)) and float(empty) == 0.0
    print(f"train smoke | loss={loss.item():.4f} | chat_loss={cl.item():.4f} "
          f"| batch_vacio={float(empty):.1f} | params={m.count_parameters():,} "
          f"| {'OK' if ok else 'FALLA'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser("train AETHER-X X0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["pretrain", "chat"], default="chat")
    ap.add_argument("--data", default="chat.jsonl")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tokenizer", default="kairos_tokenizer.json")
    ap.add_argument("--out", default="checkpoints/x0.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--scratch", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--halt-weight", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(smoke())
    train(args)


if __name__ == "__main__":
    main()
