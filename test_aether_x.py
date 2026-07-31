#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests rapidos: no entrenes horas si esto esta rojo."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from aether_v4 import lm_loss
from aether_x import AetherX, X0_CFG


def check(name, value):
    print(f"  [{'OK' if value else 'FALLA'}] {name}")
    return bool(value)


def main():
    torch.manual_seed(123)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, cfg).to(dev).eval()
    ids = torch.randint(0, 128, (2, 31), device=dev)
    ok = True

    with torch.no_grad():
        full = m(ids, m.init_state(2, dev), adaptive=False)
        st = m.init_state(2, dev)
        pieces = []
        for i in range(ids.shape[1]):
            z, st = m.step(ids[:, i:i + 1], st, adaptive=False)
            pieces.append(z)
        err = (full - torch.cat(pieces, 1)).abs().max().item()
    ok &= check(f"forward == step (err={err:.2e})", err < 1e-5)

    with torch.no_grad():
        a = m(ids[:, :4], m.init_state(2, dev), return_state=True)[1]
        b = m(ids, m.init_state(2, dev), return_state=True)[1]
    ok &= check(
        "estado constante con 4 o 31 tokens",
        all(x.shape == y.shape for x, y in zip(a.__dict__.values(), b.__dict__.values())),
    )

    mt = AetherX(128, cfg).to(dev).train()
    logits = mt(ids)
    loss = lm_loss(logits[:, :-1], ids[:, 1:]) + 1e-3 * mt.cell.route_penalty()
    loss.backward()
    ok &= check("loss finita", bool(torch.isfinite(loss)))
    ok &= check(
        "embedding y celda reciben gradiente",
        all(
            p.grad is not None
            for p in mt.parameters()
            if p is not mt.cell.halt.weight and p is not mt.cell.halt.bias
        ),
    )
    ok &= check("router halt recibe gradiente", mt.cell.halt.weight.grad is not None)

    with torch.no_grad():
        m.cell.halt.bias.fill_(-10.0)
        z, st = m(ids[:, :5], m.init_state(2, dev), return_state=True, adaptive=True)
    ratio = m.cell.stats()["deep_ratio"]
    ok &= check(f"ruta deep activa ({ratio:.0%})", ratio > 0.9)
    ok &= check("logits sin NaN", bool(torch.isfinite(z).all()))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "state.pt"
        torch.save({k: v.cpu() for k, v in st.__dict__.items()}, p)
        raw = torch.load(p, weights_only=False)
        ok &= check("estado serializable", raw["slots"].shape == st.slots.shape)

    print("\n" + ("TODOS LOS TESTS X0 PASARON" if ok else "HAY TESTS EN ROJO"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
