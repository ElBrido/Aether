#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AETHER-X X0: celda recurrente compartida para modelos pequenos.

X0 aisla la arquitectura antes de mezclar bytes, memoria externa, MoE o MTP.
Usa el tokenizer BPE actual a proposito: si algo falla aqui, es la celda.

INVARIANTES QUE LOS TESTS DEFIENDEN
-----------------------------------
1. forward(seq) == step token a token, con el mismo modo adaptativo.
2. Un token = UNA escritura de memoria, tome la ruta fast o la deep.
   La ruta deep piensa mas, NO recuerda distinto. Si dejaramos que el
   segundo pase escribiera otra vez, el scratchpad consumiria dos ranuras
   por token y las filas del batch quedarian desalineadas entre si.
3. El estado es de tamano FIJO: no crece con la longitud del texto.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether_v4 import BOS_ID, EOS_ID, KairosTokenizer, RMSNorm, lm_loss

X0_CFG = dict(vocab_size=16_000, hidden_dim=512, ffn_dim=2048,
              memory_slots=8, scratch_size=32, deep_threshold=0.50,
              tie_embeddings=True)


@dataclass
class XState:
    fast: torch.Tensor      # [B, D]      memoria de trabajo
    slots: torch.Tensor     # [B, M, D]   memoria asociativa
    scratch: torch.Tensor   # [B, W, D]   ventana literal acotada
    pos: torch.Tensor       # [B]         tokens vistos


def _select(st: XState, idx) -> XState:
    return XState(st.fast[idx], st.slots[idx], st.scratch[idx], st.pos[idx])


class SharedAetherCell(nn.Module):
    """Celda compartida: lee memoria, piensa, y escribe UNA vez por token."""

    def __init__(self, d, ffn_dim, slots, scratch, deep_threshold=0.5):
        super().__init__()
        self.d, self.n_slots, self.scratch_size = d, slots, scratch
        self.deep_threshold = deep_threshold
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.fast_gate = nn.Linear(2 * d, d)
        self.write_gate = nn.Linear(2 * d, 1)
        self.slot_decay = nn.Parameter(torch.full((slots,), 0.95))
        self.fuse = nn.Linear(4 * d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.w1 = nn.Linear(d, ffn_dim, bias=False)
        self.w2 = nn.Linear(d, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, d, bias=False)
        self.halt = nn.Linear(d, 1)
        self._deep_count = None      # tensor: evita sincronizar GPU por token
        self._total_tokens = 0
        self._halt_sum = None
        self._halt_count = 0
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Nace estable: la celda arranca cerca de la identidad.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.w3.weight)

    def init_state(self, batch, device, dtype) -> XState:
        return XState(
            torch.zeros(batch, self.d, device=device, dtype=dtype),
            torch.zeros(batch, self.n_slots, self.d, device=device, dtype=dtype),
            torch.zeros(batch, self.scratch_size, self.d, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=torch.long),
        )

    # -- lectura + computo: NO toca el estado ---------------------------------
    def _read(self, x, st):
        """Devuelve (h, v, read, write_w). Puro: se puede llamar varias veces."""
        xn = self.norm(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        scale = self.d ** -0.5

        # lee la memoria asociativa con q
        read_w = (torch.einsum("bd,bmd->bm", q, st.slots) * scale).softmax(-1)
        read = torch.einsum("bm,bmd->bd", read_w, st.slots)

        # lee el scratchpad literal (ventana fija, no KV cache creciente)
        local_w = (torch.einsum("bd,bwd->bw", q, st.scratch) * scale).softmax(-1)
        local = torch.einsum("bw,bwd->bd", local_w, st.scratch)

        h = x + self.out(self.fuse(torch.cat([x, st.fast, read, local], -1)))
        z = self.ffn_norm(h)
        h = h + self.w3(F.silu(self.w1(z)) * self.w2(z))

        # direcciona la ESCRITURA con k, no con q: leer y escribir son cosas
        # distintas y asi el tercio k del proyector deja de ser peso muerto.
        write_w = (torch.einsum("bd,bmd->bm", k, st.slots) * scale).softmax(-1)
        return h, v, read, write_w

    # -- escritura: exactamente una por token ---------------------------------
    def _commit(self, h, v, read, write_w, st) -> XState:
        gate = torch.sigmoid(self.write_gate(torch.cat([h, read], -1)))
        decay = self.slot_decay.clamp(0.80, 0.999).view(1, -1, 1)
        slots = decay * st.slots + (1.0 - decay) * gate[:, None, :] \
            * write_w[:, :, None] * (v[:, None, :] - st.slots)
        fg = torch.sigmoid(self.fast_gate(torch.cat([h, st.fast], -1)))
        fast = fg * st.fast + (1.0 - fg) * h
        scratch = torch.cat([st.scratch[:, 1:], h[:, None, :]], 1)
        return XState(fast, slots, scratch, st.pos + 1)

    def step(self, x, st: XState, adaptive: bool = True):
        h, v, read, write_w = self._read(x, st)

        if adaptive:
            p_fast = torch.sigmoid(self.halt(x).squeeze(-1))
            deep = p_fast < self.deep_threshold
            c = deep.sum()
            self._deep_count = c if self._deep_count is None else self._deep_count + c
            self._total_tokens += int(x.shape[0])
            hm = p_fast.mean()
            self._halt_sum = hm if self._halt_sum is None else self._halt_sum + hm
            self._halt_count += 1
            if deep.any():
                idx = deep.nonzero(as_tuple=False).squeeze(-1)
                # segundo pase de pensamiento sobre el MISMO estado de entrada
                h2, _, _, _ = self._read(h[idx], _select(st, idx))
                p = p_fast[idx].unsqueeze(-1)
                h = h.clone()
                h[idx] = p * h[idx] + (1.0 - p) * h2

        return h, self._commit(h, v, read, write_w, st)

    def route_penalty(self, clear: bool = True):
        """Coste explicito de pensar. Sin esto el router se va siempre a deep."""
        if self._halt_sum is None:
            return torch.zeros((), device=self.halt.weight.device)
        out = self._halt_sum / max(1, self._halt_count)
        if clear:  # el grafo es de este batch, no lo arrastramos al siguiente
            self._halt_sum = None
            self._halt_count = 0
        return out

    def reset_stats(self):
        self._deep_count = None
        self._total_tokens = 0
        self._halt_count = 0
        self._halt_sum = None

    def stats(self):
        deep = 0 if self._deep_count is None else int(self._deep_count.item())
        return dict(deep_tokens=deep, total_tokens=self._total_tokens,
                    deep_ratio=deep / max(1, self._total_tokens))


class AetherX(nn.Module):
    """Backbone X0: comparte la celda, no duplica un stack Transformer."""

    def __init__(self, vocab_size, cfg: Optional[dict] = None):
        super().__init__()
        c = dict(X0_CFG)
        c.update(cfg or {})
        self.cfg, self.vocab_size = c, vocab_size
        d = c["hidden_dim"]
        self.d_model = d
        self.embedding = nn.Embedding(vocab_size, d)
        self.emb_norm = RMSNorm(d)
        self.cell = SharedAetherCell(d, c["ffn_dim"], c["memory_slots"],
                                     c["scratch_size"], c["deep_threshold"])
        self.final_norm = RMSNorm(d)
        self.fc_out = nn.Linear(d, vocab_size, bias=False)
        if c.get("tie_embeddings", True):
            self.fc_out.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)

    def init_state(self, batch=1, device=None, dtype=None) -> XState:
        p = next(self.parameters())
        return self.cell.init_state(batch, device or p.device, dtype or p.dtype)

    def state_bytes(self, batch=1, dtype=torch.float32) -> int:
        b = torch.tensor([], dtype=dtype).element_size()
        per_row = self.d_model * (1 + self.cell.n_slots + self.cell.scratch_size)
        return batch * (b * per_row + 8)

    def forward(self, idx, state: Optional[XState] = None,
                return_state: bool = False, adaptive: bool = True):
        b, t = idx.shape
        x = self.emb_norm(self.embedding(idx))
        st = state or self.init_state(b, x.device, x.dtype)
        out = []
        for j in range(t):
            h, st = self.cell.step(x[:, j], st, adaptive)
            out.append(h)
        logits = self.fc_out(self.final_norm(torch.stack(out, 1)))
        return (logits, st) if return_state else logits

    @torch.no_grad()
    def step(self, idx_t, state: XState, adaptive: bool = True):
        """Acepta [B] o [B,1]; devuelve logits [B,1,V] para poder concatenar."""
        if idx_t.ndim == 2:
            if idx_t.shape[1] != 1:
                raise ValueError(f"step espera [B] o [B,1], recibio {tuple(idx_t.shape)}")
            idx_t = idx_t[:, 0]
        if idx_t.ndim != 1:
            raise ValueError(f"step espera [B] o [B,1], recibio {tuple(idx_t.shape)}")
        x = self.emb_norm(self.embedding(idx_t))
        h, state = self.cell.step(x, state, adaptive)
        return self.fc_out(self.final_norm(h)).unsqueeze(1), state

    @torch.no_grad()
    def generate(self, tok: KairosTokenizer, prompt: str, state=None,
                 max_tokens: int = 160, temperature: float = 0.8,
                 top_p: float = 0.95):
        self.eval()
        dev = next(self.parameters()).device
        ids = tok.encode_with_specials(prompt) or [BOS_ID]
        logits, state = self(torch.tensor([ids], device=dev), state, True)
        last, out = logits[0, -1], []
        for _ in range(max_tokens):
            z = last.float().reshape(-1)          # [V] plano: evita indexar mal
            if temperature <= 0:
                nxt = int(z.argmax(-1).item())
            else:
                probs = (z / temperature).softmax(-1)
                if 0 < top_p < 1:
                    sp, si = probs.sort(descending=True)
                    sp = sp * ((sp.cumsum(-1) - sp) <= top_p)
                    sp = sp / sp.sum().clamp_min(1e-9)
                    nxt = int(si[torch.multinomial(sp, 1)].item())
                else:
                    nxt = int(torch.multinomial(probs, 1).item())
            if nxt == EOS_ID:
                break
            out.append(nxt)
            nl, state = self.step(torch.tensor([nxt], device=dev), state, True)
            last = nl[0, -1]
        return tok.decode(out), state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke(device="cpu") -> int:
    torch.manual_seed(7)
    c = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, c).to(device).train()
    ids = torch.randint(0, 128, (2, 19), device=device)

    logits, st = m(ids, return_state=True)
    loss = lm_loss(logits[:, :-1], ids[:, 1:]) + 1e-3 * m.cell.route_penalty()
    loss.backward()

    with torch.no_grad():
        m.eval()
        a = m(ids, m.init_state(2, device), adaptive=False)
        ss = m.init_state(2, device)
        ys = []
        for j in range(ids.shape[1]):
            y, ss = m.step(ids[:, j:j + 1], ss, adaptive=False)
            ys.append(y)
        err = (a - torch.cat(ys, 1)).abs().max().item()

    ok = (torch.isfinite(loss).item() and err < 1e-4
          and st.fast.shape == (2, 96) and int(ss.pos[0]) == ids.shape[1])
    print(f"X0 smoke | params={m.count_parameters():,} | loss={loss.item():.4f} "
          f"| step_err={err:.2e} | estado={m.state_bytes(1)/1024:.1f} KB "
          f"| {'OK' if ok else 'FALLA'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser("AETHER-X X0")
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        raise SystemExit(smoke("cuda" if torch.cuda.is_available() else "cpu"))
    print("Corre: python aether_x.py --smoke")


if __name__ == "__main__":
    main()
