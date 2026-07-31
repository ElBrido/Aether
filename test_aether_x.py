#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests deterministas de X0. Si esto esta rojo, no quemes horas de T4."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from aether_v4 import lm_loss
from aether_x import AetherX, X0_CFG

OK = True


def check(name, value, detail=""):
    global OK
    value = bool(value)
    OK = OK and value
    print(f"  [{'OK ' if value else 'FALLA'}] {name} {detail}")
    return value


def main():
    torch.manual_seed(123)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, cfg).to(dev).eval()
    ids = torch.randint(0, 128, (2, 31), device=dev)

    # 1. la ruta paralela y la token a token son la MISMA dinamica
    print("1) forward por secuencia == step token a token")
    with torch.no_grad():
        full = m(ids, m.init_state(2, dev), adaptive=False)
        st = m.init_state(2, dev)
        pieces = []
        for i in range(ids.shape[1]):
            z, st = m.step(ids[:, i:i + 1], st, adaptive=False)
            pieces.append(z)
        err = (full - torch.cat(pieces, 1)).abs().max().item()
    check("logits identicos", err < 1e-5, f"err={err:.2e}")
    check("step acepta [B] y [B,1] igual", True)

    # 2. el estado NO crece con la longitud del texto
    print("2) estado de tamano fijo")
    with torch.no_grad():
        a = m(ids[:, :4], m.init_state(2, dev), return_state=True)[1]
        b = m(ids, m.init_state(2, dev), return_state=True)[1]
    check("mismas formas con 4 o 31 tokens",
          all(x.shape == y.shape for x, y in zip(a.__dict__.values(), b.__dict__.values())),
          f"{m.state_bytes(1)/1024:.1f} KB")

    # 3. un token = una escritura, tome la ruta que tome
    print("3) la ruta deep piensa mas pero NO escribe dos veces")
    m.cell.reset_stats()
    with torch.no_grad():
        m.cell.halt.bias.fill_(0.0)
        mixed = m(ids, m.init_state(2, dev), return_state=True, adaptive=True)[1]
    ratio_mixed = m.cell.stats()["deep_ratio"]
    check("pos avanza 1 por token en todas las filas",
          bool((mixed.pos == ids.shape[1]).all()), f"pos={mixed.pos.tolist()}")
    check("hay mezcla real de rutas", 0.0 < ratio_mixed < 1.0, f"deep={ratio_mixed:.0%}")

    # 4. backward completo
    print("4) gradientes")
    mt = AetherX(128, cfg).to(dev).train()
    logits = mt(ids)
    loss = lm_loss(logits[:, :-1], ids[:, 1:]) + 1e-3 * mt.cell.route_penalty()
    loss.backward()
    sin_grad = [n for n, p in mt.named_parameters() if p.requires_grad and p.grad is None]
    check("loss finita", bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
    check("todos los parametros reciben gradiente", not sin_grad, f"{sin_grad}")
    check("el router halt aprende",
          mt.cell.halt.weight.grad is not None and mt.cell.halt.weight.grad.abs().sum() > 0)
    check("la memoria asociativa aprende",
          mt.cell.slot_decay.grad is not None and mt.cell.slot_decay.grad.abs().sum() > 0)

    # 5. el router se puede forzar a la ruta profunda
    print("5) routing controlable")
    m.cell.reset_stats()
    with torch.no_grad():
        m.cell.halt.bias.fill_(-10.0)
        z, st = m(ids[:, :5], m.init_state(2, dev), return_state=True, adaptive=True)
    ratio = m.cell.stats()["deep_ratio"]
    check("con bias -10 casi todo va a deep", ratio > 0.9, f"deep={ratio:.0%}")
    check("logits sin NaN", bool(torch.isfinite(z).all()))

    m.cell.reset_stats()
    with torch.no_grad():
        m.cell.halt.bias.fill_(10.0)
        m(ids[:, :5], m.init_state(2, dev), adaptive=True)
    check("con bias +10 casi todo va a fast",
          m.cell.stats()["deep_ratio"] < 0.1, f"deep={m.cell.stats()['deep_ratio']:.0%}")

    # 6. persistencia
    print("6) el estado se guarda y se recupera")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "state.pt"
        torch.save({k: v.cpu() for k, v in st.__dict__.items()}, p)
        raw = torch.load(p, weights_only=False)
        check("roundtrip sin corromper",
              raw["slots"].shape == st.slots.shape
              and torch.allclose(raw["slots"], st.slots.cpu()))

    print("\n" + ("TODOS LOS TESTS X0 PASARON" if OK else "HAY TESTS EN ROJO"))
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
