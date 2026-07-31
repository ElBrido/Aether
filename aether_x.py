#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AETHER-X X0: celda recurrente compartida para modelos pequenos."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from aether_v4 import KairosTokenizer, RMSNorm, lm_loss

X0_CFG = dict(vocab_size=16_000, hidden_dim=512, ffn_dim=2048,
              memory_slots=8, scratch_size=32, deep_threshold=0.50,
              tie_embeddings=True)

@dataclass
class XState:
    fast: torch.Tensor
    slots: torch.Tensor
    scratch: torch.Tensor
    pos: torch.Tensor

def _select(st: XState, idx):
    return XState(st.fast[idx], st.slots[idx], st.scratch[idx], st.pos[idx])

def _merge(base: XState, sub: XState, idx):
    def put(a, b):
        z = a.clone(); z[idx] = b; return z
    return XState(put(base.fast, sub.fast), put(base.slots, sub.slots),
                  put(base.scratch, sub.scratch), put(base.pos, sub.pos))

class SharedAetherCell(nn.Module):
    """Celda compartida con memoria fast, asociativa y scratchpad literal."""
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
        self._deep_tokens = 0; self._total_tokens = 0
        self._halt_sum = None; self._halt_count = 0
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.w3.weight)

    def init_state(self, batch, device, dtype):
        return XState(
            torch.zeros(batch, self.d, device=device, dtype=dtype),
            torch.zeros(batch, self.n_slots, self.d, device=device, dtype=dtype),
            torch.zeros(batch, self.scratch_size, self.d, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=torch.long))

    def _core(self, x, st) -> Tuple[torch.Tensor, XState]:
        xn = self.norm(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        scale = self.d ** -0.5
        sim = torch.einsum("bd,bmd->bm", q, st.slots) * scale
        read_w = sim.softmax(-1)
        read = torch.einsum("bm,bmd->bd", read_w, st.slots)
        ss = torch.einsum("bd,bwd->bw", q, st.scratch) * scale
        local = torch.einsum("bw,bwd->bd", ss.softmax(-1), st.scratch)
        h = x + self.out(self.fuse(torch.cat([x, st.fast, read, local], -1)))
        z = self.ffn_norm(h)
        h = h + self.w3(F.silu(self.w1(z)) * self.w2(z))
        write = torch.sigmoid(self.write_gate(torch.cat([h, read], -1)))
        decay = self.slot_decay.clamp(0.80, 0.999).view(1, -1, 1)
        slots = decay * st.slots + (1.0 - decay) * write[:, None, :] * read_w[:, :, None] * (v[:, None, :] - st.slots)
        fg = torch.sigmoid(self.fast_gate(torch.cat([h, st.fast], -1)))
        fast = fg * st.fast + (1.0 - fg) * h
        scratch = torch.cat([st.scratch[:, 1:], h[:, None, :]], 1)
        return h, XState(fast, slots, scratch, st.pos + 1)

    def step(self, x, st, adaptive=True):
        h1, s1 = self._core(x, st)
        if not adaptive: return h1, s1
        p_fast = torch.sigmoid(self.halt(x).squeeze(-1))
        deep = p_fast < self.deep_threshold
        self._deep_tokens += int(deep.detach().sum().item()); self._total_tokens += int(x.shape[0])
        self._halt_sum = p_fast.mean() if self._halt_sum is None else self._halt_sum + p_fast.mean()
        self._halt_count += 1
        if not deep.any(): return h1, s1
        idx = deep.nonzero(as_tuple=False).squeeze(-1)
        h2, s2 = self._core(h1[idx], _select(s1, idx))
        p = p_fast[idx].unsqueeze(-1)
        h = h1.clone(); h[idx] = p * h1[idx] + (1.0 - p) * h2
        return h, _merge(s1, s2, idx)

    def route_penalty(self, clear=True):
        if self._halt_sum is None:
            return torch.zeros((), device=self.halt.weight.device)
        out = self._halt_sum / max(1, self._halt_count)
        if clear:
            self._halt_sum = None; self._halt_count = 0
        return out

    def reset_stats(self):
        self._deep_tokens = self._total_tokens = self._halt_count = 0
        self._halt_sum = None

    def stats(self):
        return dict(deep_tokens=self._deep_tokens, total_tokens=self._total_tokens,
                    deep_ratio=self._deep_tokens / max(1, self._total_tokens))

class AetherX(nn.Module):
    """Backbone X0: comparte la celda, no duplica un stack Transformer."""
    def __init__(self, vocab_size, cfg: Optional[dict] = None):
        super().__init__()
        c = dict(X0_CFG); c.update(cfg or {})
        self.cfg, self.vocab_size = c, vocab_size
        d = c["hidden_dim"]; self.d_model = d
        self.embedding = nn.Embedding(vocab_size, d)
        self.emb_norm = RMSNorm(d)
        self.cell = SharedAetherCell(d, c["ffn_dim"], c["memory_slots"], c["scratch_size"], c["deep_threshold"])
        self.final_norm = RMSNorm(d)
        self.fc_out = nn.Linear(d, vocab_size, bias=False)
        if c.get("tie_embeddings", True): self.fc_out.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)

    def init_state(self, batch=1, device=None, dtype=None):
        p = next(self.parameters())
        return self.cell.init_state(batch, device or p.device, dtype or p.dtype)

    def state_bytes(self, batch=1, dtype=torch.float32):
        b = torch.tensor([], dtype=dtype).element_size()
        return batch * (b * (self.d_model + self.cell.n_slots * self.d_model + self.cell.scratch_size * self.d_model) + 8)

    def forward(self, idx, state: Optional[XState] = None, return_state=False, adaptive=True):
        b, t = idx.shape
        x = self.emb_norm(self.embedding(idx)); st = state or self.init_state(b, x.device, x.dtype)
        out = []
        for j in range(t):
            h, st = self.cell.step(x[:, j], st, adaptive); out.append(h)
        logits = self.fc_out(self.final_norm(torch.stack(out, 1)))
        return (logits, st) if return_state else logits

    @torch.no_grad()
    def step(self, idx_t, state, adaptive=True):
        """Procesa [B] o [B,1]; normaliza la dimension singleton antes de la celda."""
        if idx_t.ndim == 2:
            if idx_t.shape[1] != 1:
                raise ValueError(f"step espera [B] o [B,1], recibio {tuple(idx_t.shape)}")
            idx_t = idx_t[:, 0]
        if idx_t.ndim != 1:
            raise ValueError(f"step espera [B] o [B,1], recibio {tuple(idx_t.shape)}")
        x = self.emb_norm(self.embedding(idx_t))
        h, state = self.cell.step(x, state, adaptive)
        return self.fc_out(self.final_norm(h)), state

    @torch.no_grad()
    def generate(self, tok: KairosTokenizer, prompt, state=None, max_tokens=160, temperature=0.8, top_p=0.95):
        self.eval(); dev = next(self.parameters()).device
        ids = tok.encode_with_specials(prompt) or [2]
        logits, state = self(torch.tensor([ids], device=dev), state, True)
        last, out = logits[:, -1], []
        for _ in range(max_tokens):
            z = last.float()
            if temperature <= 0: nxt = int(z.argmax(-1).item())
            else:
                p = (z / temperature).softmax(-1)
                if 0 < top_p < 1:
                    sp, si = p.sort(descending=True); keep = (sp.cumsum(-1) - sp) <= top_p
                    sp = sp * keep; sp = sp / sp.sum().clamp_min(1e-9); nxt = int(si[torch.multinomial(sp, 1)].item())
                else: nxt = int(torch.multinomial(p, 1).item())
            if nxt == 3: break
            out.append(nxt); last, state = self.step(torch.tensor([[nxt]], device=dev), state, adaptive=True)
        return tok.decode(out), state

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke(device="cpu"):
    torch.manual_seed(7); c = dict(X0_CFG, hidden_dim=96, ffn_dim=256, memory_slots=4, scratch_size=8)
    m = AetherX(128, c).to(device).train(); ids = torch.randint(0, 128, (2, 19), device=device)
    logits, st = m(ids, return_state=True); loss = lm_loss(logits[:, :-1], ids[:, 1:]) + 1e-3 * m.cell.route_penalty(); loss.backward()
    with torch.no_grad():
        m.eval(); a = m(ids, m.init_state(2, device), adaptive=False); ss = m.init_state(2, device); ys = []
        for j in range(ids.shape[1]): y, ss = m.step(ids[:, j:j + 1], ss, adaptive=False); ys.append(y)
        err = (a - torch.cat(ys, 1)).abs().max().item()
    ok = torch.isfinite(loss).item() and err < 1e-4 and st.fast.shape == (2, 96)
    print(f"X0 smoke | params={m.count_parameters()} | loss={loss.item():.4f} | step_err={err:.2e} | {'OK' if ok else 'FALLA'}")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser("AETHER-X X0"); ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke: raise SystemExit(smoke("cuda" if torch.cuda.is_available() else "cpu"))
    print("Corre: python aether_x.py --smoke")

if __name__ == "__main__": main()
