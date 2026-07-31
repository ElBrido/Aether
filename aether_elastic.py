#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 K A I R O S  /  A.E.T.H.E.R.  v4.1  --  "CAPACIDAD ELASTICA"
===============================================================================
 Autor : brido

 QUE RESUELVE ESTO
 -----------------
 v4.0 ya tiene memoria de tamano FIJO (el estado .kmem). Bien.
 Lo que NO tiene es CAPACIDAD elastica: el numero de parametros esta clavado
 en CFG y no se puede subir sin tirar el checkpoint a la basura.

 Este modulo agrega tres ejes de crecimiento, todos FUNCTION-PRESERVING
 (el modelo crece SIN que la loss pegue un brinco):

   1. ANCHO EFECTIVO  -> banco de expertos MoE que crece de 0 a N sin limite.
                         Solo top_k expertos se ejecutan por token, asi que
                         el COMPUTO por token no crece. El disco si.
   2. PROFUNDIDAD     -> insercion de capas nuevas inicializadas a IDENTIDAD.
   3. ESPECIALIZACION -> S0 / LoRA por dominio (ya existe en v4.0).

 LA PARTE HONESTA (leela, no la saltes)
 --------------------------------------
 * "Crecer sin limite" = crecer en DISCO sin limite. Los parametros ACTIVOS
   por token siguen teniendo que caber en VRAM. Un banco de 50 GB es viable;
   50 GB activos por token en una T4 NO. Aqui el limite lo pone top_k, no el
   tamano del banco.
 * El offload a disco cuesta latencia. En Kaggle vas a ver ~1-3 GB/s reales
   de disco. Por eso los expertos son CHICOS (d_ff ~512) y hay cache LRU.
 * Entrenamiento con expertos en disco + DDP = infierno. Regla practica:
   entrena en modo RESIDENT (los expertos viven en VRAM, son chicos), y usa
   OFFLOAD solo en inferencia o cuando el banco ya no cabe.
   Con d=1024, d_ff=512: cada experto = 1.57M params = 3.1 MB en fp16.
   64 expertos por capa MoE ya son ~200 MB. Cabe de sobra.
 * Crecer INVALIDA el estado del optimizador de los tensores que cambiaron de
   forma. Hay que reconstruir el optimizador despues de cada crecimiento.
   `maybe_grow()` te devuelve True justo para eso.
 * grad_checkpoint + MoE: la aux loss se registraria dos veces (forward doble).
   Si activas expertos, deja grad_checkpoint=False o cobra la aux fuera.

 USO
 ---
   python aether_elastic.py --self-test        # valida TODA la matematica
   python aether_elastic.py --demo             # demo de crecimiento + reporte

   # en codigo:
   from aether_elastic import (ELASTIC, GrowthLedger, elastify, grow_depth,
                               grow_experts, collect_aux, maybe_grow,
                               capacity_report, ExpertBank)
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aether_v4 import (CFG, AetherEngine, GatedDeltaNetLayer, RMSNorm,
                       SlidingWindowAttentionLayer, SwiGLU, human, lm_loss,
                       log)


# =============================================================================
# CONFIG ELASTICA
# =============================================================================
ELASTIC = dict(
    # -- banco de expertos ----------------------------------------------------
    expert_dir=str(Path(CFG["checkpoint_dir"]).parent / "experts"),
    expert_d_ff=512,          # chico a proposito: granularidad > tamano
    top_k=2,                  # expertos activos por token (compute FIJO)
    cache_experts=128,        # cuantos expertos caben en VRAM en modo offload
    birth_steps=500,          # rampa de entrada de un experto recien nacido
    birth_penalty=30.0,       # logit inicial del experto nuevo (-30 => prob ~0)
    aux_weight=1e-2,          # load balancing (Switch-style)

    # -- politica de crecimiento ----------------------------------------------
    grow_every=2_000,         # cada cuantos pasos evaluar crecimiento
    grow_experts_per_event=4, # expertos nuevos por capa MoE en cada evento
    grow_depth_every=0,       # 0 = nunca crecer en profundidad automaticamente
    grow_depth_n=2,
    max_experts_per_layer=256,
    grow_patience=2,          # eventos seguidos sin mejorar loss antes de crecer
    grow_min_improve=0.01,    # mejora minima de loss para NO crecer
)


# =============================================================================
# I. EXPERTO
# =============================================================================
class ExpertFFN(nn.Module):
    """
    Un experto = un SwiGLU chico, guardado como matrices planas para poder
    serializarlo a disco de un jalon (un solo bloque contiguo fp16).

    w3 arranca en CERO: un experto recien nacido aporta EXACTAMENTE 0.
    Ese es el truco que hace que crecer no mueva la loss ni un decimal.
    """

    def __init__(self, d_model: int, d_ff: int, zero_init: bool = True):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w3 = nn.Parameter(torch.empty(d_model, d_ff))
        nn.init.normal_(self.w1, 0.0, 0.02)
        nn.init.normal_(self.w2, 0.0, 0.02)
        if zero_init:
            nn.init.zeros_(self.w3)
        else:
            nn.init.normal_(self.w3, 0.0, 0.02)

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.w1)) * F.linear(x, self.w2),
                        self.w3)

    @property
    def numel(self) -> int:
        return self.w1.numel() + self.w2.numel() + self.w3.numel()

    @torch.no_grad()
    def to_flat(self) -> np.ndarray:
        parts = [w.detach().float().reshape(-1).cpu() for w in
                 (self.w1, self.w2, self.w3)]
        return torch.cat(parts).to(torch.float16).numpy()

    @torch.no_grad()
    def from_flat(self, flat: np.ndarray) -> "ExpertFFN":
        t = torch.from_numpy(np.asarray(flat)).float()
        n1 = self.w1.numel()
        n2 = n1 + self.w2.numel()
        self.w1.copy_(t[:n1].view_as(self.w1))
        self.w2.copy_(t[n1:n2].view_as(self.w2))
        self.w3.copy_(t[n2:].view_as(self.w3))
        return self


# =============================================================================
# II. BANCO DE EXPERTOS EN DISCO
# =============================================================================
class ExpertBank:
    """
    Almacen de expertos en disco + cache LRU en VRAM.

    Layout:
        <dir>/manifest.json
        <dir>/L003_E000017.bin      (fp16 plano: [w1 | w2 | w3])

    Un archivo por experto a proposito: agregar uno es un append, borrarlo es
    un unlink, y subirlo a HF es un upload suelto. Nada de reescribir un blob
    de 50 GB para tocar 3 MB.

    NO es un nn.Module: no aparece en state_dict ni lo toca DDP.
    """

    def __init__(self, directory: str, d_model: int, d_ff: int,
                 cache_experts: int = 128):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.d_model, self.d_ff = d_model, d_ff
        self.cache_experts = cache_experts
        self._cache: "OrderedDict[Tuple[int,int], ExpertFFN]" = OrderedDict()
        self.hits, self.misses = 0, 0
        mf = self.dir / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            self.counts = {int(k): v for k, v in m.get("counts", {}).items()}
        else:
            self.counts = {}
            self._flush_manifest()

    # -- manifest -------------------------------------------------------------
    def _flush_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(dict(
            d_model=self.d_model, d_ff=self.d_ff,
            counts={str(k): v for k, v in self.counts.items()},
            updated_at=time.time()), indent=2))

    def path(self, layer: int, eid: int) -> Path:
        return self.dir / f"L{layer:03d}_E{eid:06d}.bin"

    def count(self, layer: int) -> int:
        return self.counts.get(layer, 0)

    # -- alta / lectura / escritura -------------------------------------------
    def create(self, layer: int, n: int) -> List[int]:
        """Crea n expertos nuevos (w3=0) y los deja escritos en disco."""
        base = self.count(layer)
        for j in range(n):
            e = ExpertFFN(self.d_model, self.d_ff, zero_init=True)
            e.to_flat().tofile(self.path(layer, base + j))
        self.counts[layer] = base + n
        self._flush_manifest()
        return list(range(base, base + n))

    def read(self, layer: int, eid: int, device="cpu") -> ExpertFFN:
        p = self.path(layer, eid)
        if not p.exists():
            raise FileNotFoundError(f"experto ausente: {p}")
        flat = np.fromfile(p, dtype=np.float16)
        e = ExpertFFN(self.d_model, self.d_ff, zero_init=True)
        e.from_flat(flat)
        return e.to(device)

    def write(self, layer: int, eid: int, expert: ExpertFFN) -> None:
        expert.to_flat().tofile(self.path(layer, eid))
        if eid >= self.count(layer):
            self.counts[layer] = eid + 1
            self._flush_manifest()

    # -- cache LRU (modo offload, solo inferencia) -----------------------------
    def acquire(self, layer: int, eid: int, device) -> ExpertFFN:
        key = (layer, eid)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        e = self.read(layer, eid, device)
        e.eval()
        for p in e.parameters():
            p.requires_grad_(False)
        self._cache[key] = e
        while len(self._cache) > self.cache_experts:
            self._cache.popitem(last=False)
        return e

    def clear_cache(self):
        self._cache.clear()

    # -- contabilidad ----------------------------------------------------------
    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("L*_E*.bin"))

    def total_experts(self) -> int:
        return sum(self.counts.values())

    def expert_params(self) -> int:
        return 3 * self.d_model * self.d_ff

    # -- puente resident <-> disco ---------------------------------------------
    def dump_model(self, model: nn.Module) -> int:
        """Vuelca a disco todos los expertos residentes del modelo."""
        n = 0
        for m in model.modules():
            if isinstance(m, ElasticFFN) and m.mode == "resident":
                for eid, e in enumerate(m.experts):
                    self.write(m.layer_id, eid, e)
                    n += 1
        return n

    def load_model(self, model: nn.Module, device=None) -> int:
        """Carga desde disco a los expertos residentes del modelo."""
        n = 0
        for m in model.modules():
            if isinstance(m, ElasticFFN) and m.mode == "resident":
                for eid, e in enumerate(m.experts):
                    flat = np.fromfile(self.path(m.layer_id, eid),
                                       dtype=np.float16)
                    e.from_flat(flat)
                    if device:
                        e.to(device)
                    n += 1
        return n


# =============================================================================
# III. FFN ELASTICA (MoE que crece)
# =============================================================================
class ElasticFFN(nn.Module):
    """
    Reemplaza al SwiGLU denso de una capa por:

        y = shared(x)  +  sum_{e in topk} g_e * expert_e(x)

    `shared` es el SwiGLU ORIGINAL, intacto: garantiza que siempre hay una ruta
    densa que funciona aunque el router se equivoque. Los expertos son la parte
    que crece.

    COMPUTO POR TOKEN: shared + top_k expertos. CONSTANTE, no depende de cuantos
    expertos tengas en el banco. Puedes tener 4 o 4.000.

    NACIMIENTO SUAVE: un experto nuevo tiene w3=0 (aporta 0 exacto) y ademas un
    penalty de -30 en el logit del router que se va a 0 en `birth_steps`. Sin
    esa rampa, meter expertos nuevos reparte de golpe la masa del softmax y te
    mueve la loss de los expertos viejos. Con ella, crecer es un no-op perfecto
    y luego el router decide solito si el nuevo sirve.
    """

    def __init__(self, shared: nn.Module, layer_id: int, d_model: int,
                 d_ff: int, top_k: int = 2, bank: Optional[ExpertBank] = None,
                 birth_steps: int = 500, birth_penalty: float = 30.0):
        super().__init__()
        self.shared = shared
        self.layer_id = layer_id
        self.d_model, self.d_ff = d_model, d_ff
        self.top_k = top_k
        self.birth_steps = max(1, int(birth_steps))
        self.birth_penalty = float(birth_penalty)
        self.mode = "resident"          # "resident" | "offload"
        self.bank = bank                # no es Module: no entra al state_dict
        self.experts = nn.ModuleList()
        self.w_router = nn.Parameter(torch.zeros(0, d_model))
        self.register_buffer("birth_left", torch.zeros(0), persistent=True)
        self._aux = None
        self._n_offload = 0             # cuantos expertos ve en modo offload

    # -- introspeccion ---------------------------------------------------------
    @property
    def n_experts(self) -> int:
        return self._n_offload if self.mode == "offload" else len(self.experts)

    def extra_repr(self):
        return (f"layer={self.layer_id}, experts={self.n_experts}, "
                f"top_k={self.top_k}, d_ff={self.d_ff}, mode={self.mode}")

    # -- crecimiento -----------------------------------------------------------
    @torch.no_grad()
    def grow(self, n_new: int) -> int:
        dev = self.w_router.device
        for _ in range(n_new):
            self.experts.append(
                ExpertFFN(self.d_model, self.d_ff, zero_init=True).to(dev))
        new_rows = torch.zeros(n_new, self.d_model, device=dev)
        self.w_router = nn.Parameter(torch.cat([self.w_router.data, new_rows]))
        self.birth_left = torch.cat([
            self.birth_left.to(dev),
            torch.full((n_new,), float(self.birth_steps), device=dev)])
        if self.bank is not None:
            base = self.bank.count(self.layer_id)
            need = len(self.experts) - base
            if need > 0:
                self.bank.create(self.layer_id, need)
        return len(self.experts)

    # -- modos -----------------------------------------------------------------
    def set_offload(self, on: bool):
        if on:
            assert self.bank is not None, "offload necesita un ExpertBank"
            if len(self.experts):
                for eid, e in enumerate(self.experts):
                    self.bank.write(self.layer_id, eid, e)
            self._n_offload = max(self.bank.count(self.layer_id),
                                  len(self.experts))
            self.experts = nn.ModuleList()
            self.mode = "offload"
        else:
            assert self.bank is not None, "para volver a resident hace falta banco"
            dev = self.w_router.device
            self.experts = nn.ModuleList(
                [self.bank.read(self.layer_id, eid, dev)
                 for eid in range(self.bank.count(self.layer_id))])
            for e in self.experts:
                for p in e.parameters():
                    p.requires_grad_(True)
            self.mode = "resident"

    # -- forward ---------------------------------------------------------------
    def _expert(self, eid: int, device):
        if self.mode == "resident":
            return self.experts[eid]
        return self.bank.acquire(self.layer_id, eid, device)

    def forward(self, x):
        y = self.shared(x)
        E = self.n_experts
        if E == 0:
            self._aux = None
            return y
        if self.mode == "offload" and self.training:
            raise RuntimeError(
                "modo offload es solo para inferencia. Llama set_offload(False) "
                "antes de entrenar (los expertos son chicos, caben en VRAM).")

        shp = x.shape
        xf = x.reshape(-1, self.d_model)
        N = xf.shape[0]
        k = min(self.top_k, E)

        logits = F.linear(xf.float(), self.w_router.float())        # [N, E]
        pen = -self.birth_penalty * (self.birth_left.float()
                                     / self.birth_steps)            # [E]
        logits = logits + pen[None, :]
        probs = logits.softmax(dim=-1)
        tw, ti = probs.topk(k, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(xf)
        for eid in ti.unique().tolist():
            hit = (ti == eid)
            rows, slot = hit.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            w = tw[rows, slot].unsqueeze(-1).to(xf.dtype)
            expert = self._expert(int(eid), xf.device)
            out = out.index_add(0, rows, expert(xf[rows]) * w)

        # load balancing (Switch): penaliza que todo caiga en el mismo experto
        with torch.enable_grad() if self.training else torch.no_grad():
            f = torch.zeros(E, device=xf.device, dtype=probs.dtype)
            f = f.index_add(0, ti.reshape(-1),
                            torch.ones(ti.numel(), device=xf.device,
                                       dtype=probs.dtype)) / max(1, N * k)
            P = probs.mean(dim=0)
            self._aux = E * (f.detach() * P).sum()

        if self.training and self.birth_left.numel():
            self.birth_left = (self.birth_left - 1.0).clamp_min(0.0)

        return y + out.view(shp)


def collect_aux(model: nn.Module, reset: bool = True) -> torch.Tensor:
    """Suma las aux losses de todas las capas MoE. Sumala a tu loss principal:
           loss = lm_loss(...) + ELASTIC['aux_weight'] * collect_aux(model)
    Sin esto el router colapsa: todos los tokens al mismo experto y el resto
    del banco se queda de adorno ocupando disco."""
    tot, dev = None, None
    for m in model.modules():
        if isinstance(m, ElasticFFN) and m._aux is not None:
            tot = m._aux if tot is None else tot + m._aux
            dev = m._aux.device
            if reset:
                m._aux = None
    if tot is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device if p is not None else "cpu")
    return tot


def router_stats(model: nn.Module) -> Dict[int, int]:
    return {m.layer_id: m.n_experts
            for m in model.modules() if isinstance(m, ElasticFFN)}


def set_offload(model: nn.Module, on: bool):
    for m in model.modules():
        if isinstance(m, ElasticFFN):
            m.set_offload(on)
    return model


# =============================================================================
# IV. OPERACIONES DE CRECIMIENTO
# =============================================================================
def elastify(model: AetherEngine, bank: Optional[ExpertBank] = None,
             d_ff: Optional[int] = None, top_k: int = 2,
             targets: str = "gdn", birth_steps: int = 500,
             birth_penalty: float = 30.0) -> AetherEngine:
    """
    Convierte los FFN densos en FFN elasticas. FUNCTION-PRESERVING exacto:
    arranca con CERO expertos, asi que el forward es identico al de v4.0.

    targets: "gdn" (solo capas recurrentes), "attn", o "all".
    """
    d_ff = d_ff or ELASTIC["expert_d_ff"]
    want = {"gdn": ("gdn",), "attn": ("attn",), "all": ("gdn", "attn")}[targets]
    n = 0
    for i, lyr in enumerate(model.layers):
        if model.kinds[i] not in want:
            continue
        if isinstance(lyr.ffn, ElasticFFN):
            continue
        lyr.ffn = ElasticFFN(lyr.ffn, layer_id=i, d_model=model.d_model,
                             d_ff=d_ff, top_k=top_k, bank=bank,
                             birth_steps=birth_steps,
                             birth_penalty=birth_penalty)
        n += 1
    log.info(f"elastify: {n} capas ahora son MoE elastica "
             f"(d_ff={d_ff}, top_k={top_k}, targets={targets})")
    return model


def grow_experts(model: AetherEngine, n_new: int = 4,
                 layers: Optional[List[int]] = None,
                 max_per_layer: int = 256) -> int:
    """Agrega n_new expertos a cada capa MoE. Aporte inicial = 0 EXACTO."""
    added = 0
    for m in model.modules():
        if not isinstance(m, ElasticFFN):
            continue
        if layers is not None and m.layer_id not in layers:
            continue
        room = max_per_layer - m.n_experts
        if room <= 0:
            continue
        m.grow(min(n_new, room))
        added += min(n_new, room)
    if added:
        log.info(f"grow_experts: +{added} expertos | "
                 f"por capa: {router_stats(model)}")
    return added


@torch.no_grad()
def _identity_init(layer: nn.Module) -> nn.Module:
    """Deja la capa como IDENTIDAD exacta: out = x.
    o_proj=0 -> h = x ; ffn.w3=0 -> out = h. Ninguno tiene bias, asi que es
    identidad literal, no 'aproximadamente'.
    Los gradientes NO mueren: o_proj recibe grad desde el primer paso (su
    entrada no es cero), y en cuanto o_proj se despega, el resto tambien."""
    layer.o_proj.weight.zero_()
    ffn = layer.ffn
    if isinstance(ffn, ElasticFFN):
        ffn.shared.w3.weight.zero_()
    else:
        ffn.w3.weight.zero_()
    return layer


def grow_depth(model: AetherEngine, n_new: int = 2, kind: str = "gdn",
               where: str = "interleave") -> AetherEngine:
    """
    Inserta n_new capas nuevas inicializadas a IDENTIDAD.

    Reindexa model.s0 (esta indexado por posicion de capa: si insertas en medio
    y no lo remapeas, cada S0 entrenado queda pegado a la capa equivocada; es
    el bug silencioso mas facil de cometer aqui) y remapea ltm_at.

    where: "interleave" (reparte las nuevas a lo largo del stack, mejor senal)
           o "end" (todas al final).
    """
    assert kind in ("gdn", "attn")
    cfg = model.cfg
    dev = next(model.parameters()).device
    old_layers, old_kinds = list(model.layers), list(model.kinds)
    L = len(old_layers)
    total_new = L + n_new
    ds = 1.0 / math.sqrt(2 * total_new)

    if where == "end":
        insert_at = [L] * n_new
    else:
        step = max(1, L // (n_new + 1))
        insert_at = [min(L, (j + 1) * step) for j in range(n_new)]

    # detectar si el modelo ya es elastico para envolver tambien las nuevas
    proto = next((m for m in model.modules() if isinstance(m, ElasticFFN)), None)

    def _new_layer():
        if kind == "gdn":
            lyr = GatedDeltaNetLayer(model.d_model, cfg["num_heads"],
                                     cfg["chunk_size"], cfg["conv_kernel"], ds)
        else:
            lyr = SlidingWindowAttentionLayer(model.d_model, cfg["num_heads"],
                                              cfg["window_size"],
                                              cfg["rope_theta"], ds)
        return _identity_init(lyr).to(dev)

    new_layers, new_kinds, old_to_new = [], [], {}
    for i in range(L + 1):
        for j, at in enumerate(insert_at):
            if at == i:
                new_layers.append(_new_layer())
                new_kinds.append(kind)
        if i < L:
            old_to_new[i] = len(new_layers)
            new_layers.append(old_layers[i])
            new_kinds.append(old_kinds[i])

    # -- reindexar S0 ---------------------------------------------------------
    old_s0 = {int(k): v for k, v in model.s0.items()}
    new_s0 = nn.ParameterDict()
    new_to_old = {v: k for k, v in old_to_new.items()}
    for ni, kd in enumerate(new_kinds):
        if kd != "gdn":
            continue
        oi = new_to_old.get(ni)
        if oi is not None and oi in old_s0:
            new_s0[str(ni)] = old_s0[oi]
        else:
            new_s0[str(ni)] = nn.Parameter(
                torch.zeros(1, model.n_heads, model.dh, model.dh, device=dev))

    # -- reindexar los layer_id de las FFN elasticas --------------------------
    for ni, lyr in enumerate(new_layers):
        if isinstance(getattr(lyr, "ffn", None), ElasticFFN):
            lyr.ffn.layer_id = ni
        elif proto is not None and new_kinds[ni] in ("gdn",) and \
                ni not in new_to_old:
            lyr.ffn = ElasticFFN(lyr.ffn, layer_id=ni, d_model=model.d_model,
                                 d_ff=proto.d_ff, top_k=proto.top_k,
                                 bank=proto.bank,
                                 birth_steps=proto.birth_steps,
                                 birth_penalty=proto.birth_penalty).to(dev)

    model.layers = nn.ModuleList(new_layers)
    model.kinds = new_kinds
    model.s0 = new_s0
    model.n_layers = len(new_layers)
    model.ltm_at = old_to_new.get(model.ltm_at, -1) if model.ltm_at >= 0 else -1
    log.info(f"grow_depth: {L} -> {model.n_layers} capas (+{n_new} {kind}, "
             f"identidad exacta) | ltm_at={model.ltm_at}")
    return model


# =============================================================================
# V. LEDGER DE CRECIMIENTO  (sin esto, tu checkpoint es basura)
# =============================================================================
class GrowthLedger:
    """
    Registro ordenado de TODO lo que le creciste al modelo.

    Por que es obligatorio: un checkpoint de un modelo crecido tiene una forma
    que NO se puede deducir de CFG. Si guardas solo el state_dict, al recargar
    construyes el modelo base y el load_state_dict te explota (o peor: pasa con
    strict=False y entrenas medio modelo sin darte cuenta).

    Con el ledger: construyes el base, REPRODUCES el crecimiento, y recien ahi
    cargas los pesos con strict=True. Determinista y auditable.
    """

    def __init__(self, events: Optional[List[dict]] = None):
        self.events: List[dict] = list(events or [])

    def record(self, op: str, step: int = -1, **kwargs):
        self.events.append(dict(op=op, step=step, kwargs=kwargs,
                                at=time.time()))
        return self

    def to_list(self) -> List[dict]:
        return self.events

    def apply(self, model: AetherEngine,
              bank: Optional[ExpertBank] = None) -> AetherEngine:
        for ev in self.events:
            op, kw = ev["op"], dict(ev["kwargs"])
            if op == "elastify":
                elastify(model, bank=bank, **kw)
            elif op == "grow_experts":
                grow_experts(model, **kw)
            elif op == "grow_depth":
                grow_depth(model, **kw)
            else:
                raise ValueError(f"evento de crecimiento desconocido: {op}")
        return model

    def summary(self) -> str:
        if not self.events:
            return "(sin crecimiento)"
        return " -> ".join(f"{e['op']}@{e['step']}" for e in self.events)


def rebuild_from_ledger(vocab_size: int, cfg: dict, ledger: GrowthLedger,
                        bank: Optional[ExpertBank] = None) -> AetherEngine:
    """Modelo base + replay del ledger = la forma exacta del checkpoint."""
    model = AetherEngine(vocab_size, cfg)
    ledger.apply(model, bank)
    return model


def save_elastic(path, model: AetherEngine, ledger: GrowthLedger, step: int,
                 cfg: dict, extra: Optional[dict] = None,
                 bank: Optional[ExpertBank] = None) -> Path:
    raw = model.module if hasattr(model, "module") else model
    if bank is not None:
        bank.dump_model(raw)
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    torch.save(dict(model=raw.state_dict(), ledger=ledger.to_list(),
                    step=step, cfg=cfg, elastic=dict(ELASTIC),
                    **(extra or {})), tmp)
    tmp.replace(p)
    return p


def load_elastic(path, cfg: Optional[dict] = None,
                 bank: Optional[ExpertBank] = None, device="cpu",
                 strict: bool = True):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg or sd["cfg"]
    ledger = GrowthLedger(sd.get("ledger", []))
    vocab = sd["model"]["embedding.weight"].shape[0]
    model = rebuild_from_ledger(vocab, cfg, ledger, bank)
    model.load_state_dict(sd["model"], strict=strict)
    log.info(f"cargado paso {sd['step']:,} | crecimiento: {ledger.summary()}")
    return model.to(device), ledger, sd


# =============================================================================
# VI. POLITICA DE CRECIMIENTO AUTOMATICO
# =============================================================================
class GrowthPolicy:
    """
    Crece cuando la loss deja de bajar, no cada N pasos porque si.

    Un modelo que todavia esta mejorando NO necesita mas parametros: necesita
    mas tokens. Meterle capacidad ahi solo te quita tok/s. La senal correcta es
    el estancamiento: si en `patience` ventanas seguidas la loss no bajo al
    menos `min_improve`, ya saturaste la capacidad actual y toca crecer.
    """

    def __init__(self, ec: Optional[dict] = None):
        ec = ec or ELASTIC
        self.every = ec["grow_every"]
        self.n_experts = ec["grow_experts_per_event"]
        self.max_per_layer = ec["max_experts_per_layer"]
        self.depth_every = ec["grow_depth_every"]
        self.depth_n = ec["grow_depth_n"]
        self.patience = ec["grow_patience"]
        self.min_improve = ec["grow_min_improve"]
        self.best = float("inf")
        self.stale = 0

    def should(self, step: int, loss: float) -> Tuple[bool, bool]:
        """devuelve (crecer_expertos, crecer_profundidad)"""
        if step <= 0 or step % self.every:
            return False, False
        if loss < self.best - self.min_improve:
            self.best, self.stale = loss, 0
            return False, False
        self.stale += 1
        if self.stale < self.patience:
            return False, False
        self.stale = 0
        deep = bool(self.depth_every) and (step % self.depth_every == 0)
        return True, deep


def maybe_grow(model: AetherEngine, ledger: GrowthLedger, policy: GrowthPolicy,
               step: int, loss: float) -> bool:
    """
    Devuelve True si el modelo CRECIO. Si te devuelve True estas OBLIGADO a:
      1. reconstruir el optimizador (build_optimizers)
      2. reconstruir el EMA
      3. re-envolver en DDP si estas en multi-GPU
    Los estados de Adam de un tensor que cambio de forma no sirven. Reusarlos
    es como reusar un checkpoint con otro tokenizer: silencioso y letal.
    """
    do_e, do_d = policy.should(step, loss)
    if not (do_e or do_d):
        return False
    raw = model.module if hasattr(model, "module") else model
    if do_d:
        grow_depth(raw, policy.depth_n)
        ledger.record("grow_depth", step=step, n_new=policy.depth_n)
    if do_e:
        n = grow_experts(raw, policy.n_experts,
                         max_per_layer=policy.max_per_layer)
        if n:
            ledger.record("grow_experts", step=step, n_new=policy.n_experts,
                          max_per_layer=policy.max_per_layer)
    log.info(f"CRECIMIENTO @ paso {step:,} | "
             f"{human(raw.count_parameters())} params | "
             f"reconstruye optimizador+EMA+DDP")
    return True


# =============================================================================
# VII. CONTABILIDAD  (la tabla que hay que enseniar, sin humo)
# =============================================================================
def capacity_report(model: AetherEngine, bank: Optional[ExpertBank] = None,
                    verbose: bool = True) -> dict:
    raw = model.module if hasattr(model, "module") else model
    moe = [m for m in raw.modules() if isinstance(m, ElasticFFN)]
    dense = raw.count_parameters()
    resident_experts = sum(sum(e.numel for e in m.experts) for m in moe)
    per_expert = (3 * raw.d_model * moe[0].d_ff) if moe else 0
    banked = (bank.total_experts() * bank.expert_params()) if bank else 0
    n_bank = bank.total_experts() if bank else sum(m.n_experts for m in moe)

    # parametros ACTIVOS por token: densos - expertos residentes + top_k*capa
    active = dense - resident_experts + sum(
        min(m.top_k, m.n_experts) * per_expert for m in moe)
    total = dense - resident_experts + max(banked, resident_experts)

    rep = dict(
        capas_moe=len(moe),
        expertos_totales=n_bank,
        params_por_experto=per_expert,
        params_activos_por_token=active,
        params_totales=total,
        ratio_capacidad=(total / max(1, active)),
        estado_persistente_bytes=raw.state_bytes(1),
        disco_banco_bytes=bank.disk_bytes() if bank else 0,
    )
    if verbose:
        log.info("=" * 70)
        log.info(f"  capas MoE            : {rep['capas_moe']}")
        log.info(f"  expertos en el banco : {rep['expertos_totales']}")
        log.info(f"  ACTIVOS por token    : {human(active)} params  <- esto es "
                 f"lo que tiene que caber en la T4")
        log.info(f"  TOTALES (capacidad)  : {human(total)} params  <- esto es "
                 f"lo que crece sin limite")
        log.info(f"  ratio capacidad/comp : {rep['ratio_capacidad']:.2f}x")
        log.info(f"  estado persistente   : "
                 f"{rep['estado_persistente_bytes']/1e6:.2f} MB (FIJO)")
        log.info(f"  banco en disco       : "
                 f"{rep['disco_banco_bytes']/1e6:.2f} MB")
        log.info("=" * 70)
    return rep


# =============================================================================
# VIII. SELF-TEST  (corre esto ANTES de quemar horas de T4)
# =============================================================================
def _tiny_cfg(**over):
    c = dict(CFG)
    c.update(hidden_dim=64, num_layers=4, num_heads=4, chunk_size=8,
             window_size=16, ltm_dim=32, seq_len=32, use_ltm=False,
             mtp_weight=0.0, grad_checkpoint=False, tie_embeddings=True)
    c.update(over)
    return c


def self_test() -> int:
    import shutil
    import tempfile

    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"  [{'OK ' if cond else 'FALLA'}] {name} {detail}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="aether_elastic_"))
    print(f"\ndispositivo: {dev} | torch {torch.__version__} | tmp {tmp}\n")

    cfg = _tiny_cfg()
    V = 128
    ids = torch.randint(0, V, (2, 24), device=dev)

    # -- 1: elastify no cambia NADA -------------------------------------------
    print("1) elastify es function-preserving exacto")
    m = AetherEngine(V, cfg).to(dev).eval()
    with torch.no_grad():
        base = m(ids).clone()
    bank = ExpertBank(str(tmp / "experts"), m.d_model, 32, cache_experts=8)
    elastify(m, bank=bank, d_ff=32, top_k=2, targets="gdn")
    m.to(dev).eval()
    with torch.no_grad():
        after = m(ids)
    e = (base - after).abs().max().item()
    chk("logits identicos con 0 expertos", e == 0.0, f"err={e:.2e}")

    # -- 2: crecer expertos no mueve la loss ----------------------------------
    print("2) grow_experts es function-preserving exacto")
    grow_experts(m, 4)
    m.to(dev).eval()
    with torch.no_grad():
        after2 = m(ids)
    e = (base - after2).abs().max().item()
    chk("logits identicos al nacer 4 expertos", e < 1e-5, f"err={e:.2e}")

    # -- 3: crecer SOBRE expertos ya entrenados tampoco mueve nada ------------
    print("3) segundo crecimiento con expertos ya 'entrenados'")
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, ElasticFFN):
                mod.birth_left.zero_()                      # ya nacieron
                for ex in mod.experts:
                    ex.w3.normal_(0, 0.02)
                mod.w_router.normal_(0, 0.5)
        trained = m(ids).clone()
    grow_experts(m, 3)
    m.to(dev).eval()
    with torch.no_grad():
        after3 = m(ids)
    e = (trained - after3).abs().max().item()
    chk("la rampa de nacimiento aisla a los nuevos", e < 1e-5, f"err={e:.2e}")

    # -- 4: el compute por token NO crece -------------------------------------
    print("4) compute por token constante mientras la capacidad sube")
    r1 = capacity_report(m, bank, verbose=False)
    grow_experts(m, 16)
    r2 = capacity_report(m, bank, verbose=False)
    chk("params activos por token iguales",
        r1["params_activos_por_token"] == r2["params_activos_por_token"],
        f"{human(r1['params_activos_por_token'])}")
    chk("params totales suben",
        r2["params_totales"] > r1["params_totales"],
        f"{human(r1['params_totales'])} -> {human(r2['params_totales'])}")

    # -- 5: gradientes -----------------------------------------------------
    print("5) backward: los expertos y el router reciben gradiente")
    mt = m.train()
    for mod in mt.modules():
        if isinstance(mod, ElasticFFN):
            mod.birth_left.zero_()
    lg = mt(ids[:, :-1])
    loss = lm_loss(lg, ids[:, 1:], 1e-4) + ELASTIC["aux_weight"] * collect_aux(mt)
    loss.backward()
    routers = [mod.w_router for mod in mt.modules() if isinstance(mod, ElasticFFN)]
    chk("loss finita", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    chk("el router tiene gradiente",
        all(p.grad is not None and p.grad.abs().sum() > 0 for p in routers))
    used = [ex for mod in mt.modules() if isinstance(mod, ElasticFFN)
            for ex in mod.experts if ex.w3.grad is not None
            and ex.w3.grad.abs().sum() > 0]
    chk("al menos un experto se entrena", len(used) > 0, f"{len(used)} expertos")
    mt.zero_grad(set_to_none=True)

    # -- 6: grow_depth es identidad exacta ------------------------------------
    print("6) grow_depth: capas nuevas = identidad exacta")
    m2 = AetherEngine(V, cfg).to(dev).eval()
    with torch.no_grad():
        b2 = m2(ids).clone()
    n_before = m2.n_layers
    grow_depth(m2, 2, kind="gdn", where="interleave")
    m2.to(dev).eval()
    with torch.no_grad():
        a2 = m2(ids)
    e = (b2 - a2).abs().max().item()
    chk("logits identicos", e < 1e-5, f"err={e:.2e}")
    chk("el stack crecio", m2.n_layers == n_before + 2,
        f"{n_before} -> {m2.n_layers}")
    chk("hay un S0 por capa recurrente",
        len(m2.s0) == sum(1 for k in m2.kinds if k == "gdn"))

    # -- 7: S0 sigue pegado a SU capa -----------------------------------------
    print("7) reindexado de S0: cada estado sigue con su capa")
    m3 = AetherEngine(V, cfg).to(dev)
    marks = {}
    with torch.no_grad():
        for kk in list(m3.s0.keys()):
            m3.s0[kk].fill_(float(kk) + 1.0)
            marks[int(kk)] = float(kk) + 1.0
    old_kinds = list(m3.kinds)
    grow_depth(m3, 2, kind="gdn", where="interleave")
    vals = sorted(float(v.flatten()[0]) for v in m3.s0.values())
    esperados = sorted(list(marks.values()) + [0.0, 0.0])
    chk("las marcas sobrevivieron sin mezclarse",
        all(abs(a - b) < 1e-6 for a, b in zip(vals, esperados)),
        f"{vals}")

    # -- 8: banco en disco, roundtrip exacto ----------------------------------
    print("8) banco en disco: escribir y releer no corrompe")
    ex = ExpertFFN(m.d_model, 32)
    with torch.no_grad():
        ex.w1.normal_(0, 0.1); ex.w2.normal_(0, 0.1); ex.w3.normal_(0, 0.1)
    bank.write(99, 0, ex)
    back = bank.read(99, 0, "cpu")
    e = max((ex.w1 - back.w1).abs().max().item(),
            (ex.w2 - back.w2).abs().max().item(),
            (ex.w3 - back.w3).abs().max().item())
    chk("roundtrip fp16 dentro de tolerancia", e < 1e-3, f"err={e:.2e}")

    # -- 9: offload da el MISMO resultado que resident ------------------------
    print("9) offload (disco+LRU) == resident, bit a bit razonable")
    m4 = AetherEngine(V, cfg).to(dev).eval()
    bank4 = ExpertBank(str(tmp / "experts4"), m4.d_model, 32, cache_experts=2)
    elastify(m4, bank=bank4, d_ff=32, top_k=2)
    grow_experts(m4, 6)
    m4.to(dev).eval()
    with torch.no_grad():
        for mod in m4.modules():
            if isinstance(mod, ElasticFFN):
                mod.birth_left.zero_()
                for exx in mod.experts:
                    exx.w3.normal_(0, 0.05)
                mod.w_router.normal_(0, 0.5)
        y_res = m4(ids).clone()
    set_offload(m4, True)
    with torch.no_grad():
        y_off = m4(ids)
    e = (y_res - y_off).abs().max().item()
    chk("mismo forward con expertos en disco", e < 5e-2, f"err={e:.2e}")
    chk("la cache LRU evicta de verdad", len(bank4._cache) <= 2,
        f"residentes={len(bank4._cache)} hits={bank4.hits} miss={bank4.misses}")
    chk("offload en training esta bloqueado",
        _raises(lambda: m4.train()(ids)))
    m4.eval()

    # -- 10: ledger reproduce la forma exacta ---------------------------------
    print("10) ledger: rebuild + load_state_dict(strict=True)")
    m5 = AetherEngine(V, cfg).to(dev)
    led = GrowthLedger()
    bank5 = ExpertBank(str(tmp / "experts5"), m5.d_model, 32)
    elastify(m5, bank=bank5, d_ff=32, top_k=2, targets="gdn")
    led.record("elastify", step=0, d_ff=32, top_k=2, targets="gdn")
    grow_experts(m5, 3); led.record("grow_experts", step=100, n_new=3)
    grow_depth(m5, 2);   led.record("grow_depth", step=200, n_new=2)
    grow_experts(m5, 2); led.record("grow_experts", step=300, n_new=2)
    m5.to(dev).eval()
    with torch.no_grad():
        for mod in m5.modules():
            if isinstance(mod, ElasticFFN):
                mod.birth_left.zero_()
                for exx in mod.experts:
                    exx.w3.normal_(0, 0.05)
        y5 = m5(ids).clone()
    ck = save_elastic(tmp / "elastic.pt", m5, led, 300, cfg)
    m6, led6, _ = load_elastic(ck, cfg, bank=None, device=dev, strict=True)
    m6.eval()
    with torch.no_grad():
        y6 = m6(ids)
    e = (y5 - y6).abs().max().item()
    chk("forma reconstruida sin strict=False", True, f"eventos={len(led6.events)}")
    chk("mismos logits tras recargar", e < 1e-5, f"err={e:.2e}")

    # -- 11: la memoria persistente NO se contamina ---------------------------
    print("11) crecer no rompe la promesa de estado de tamano FIJO")
    sb_before = m2.state_bytes(1)
    grow_experts(m2, 8) if any(isinstance(x, ElasticFFN) for x in m2.modules()) \
        else None
    sb_after = m2.state_bytes(1)
    chk("los expertos no entran al .kmem", sb_before == sb_after,
        f"{sb_before/1e3:.1f} KB")

    # -- 12: politica de crecimiento ------------------------------------------
    print("12) politica: crece por estancamiento, no por calendario")
    pol = GrowthPolicy(dict(ELASTIC, grow_every=10, grow_patience=2,
                            grow_min_improve=0.01, grow_depth_every=0))
    seq = [(10, 5.0), (20, 4.0), (30, 3.99), (40, 3.985)]
    got = [pol.should(s, l)[0] for s, l in seq]
    chk("no crece mientras la loss baja", got[:2] == [False, False])
    chk("crece tras 2 ventanas estancadas", got[3] is True, f"{got}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("TODOS LOS TESTS PASARON, ya puedes crecer en Kaggle"
                  if ok_all else "HAY TESTS EN ROJO, no entrenes todavia"))
    return 0 if ok_all else 1


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


# =============================================================================
# IX. DEMO
# =============================================================================
def demo():
    """Muestra la curva capacidad-vs-compute con la config real de 350M."""
    cfg = dict(CFG)
    model = AetherEngine(16_000, cfg)
    bank = ExpertBank(ELASTIC["expert_dir"], model.d_model,
                      ELASTIC["expert_d_ff"])
    ledger = GrowthLedger()
    elastify(model, bank=bank, d_ff=ELASTIC["expert_d_ff"],
             top_k=ELASTIC["top_k"], targets="gdn")
    ledger.record("elastify", step=0, d_ff=ELASTIC["expert_d_ff"],
                  top_k=ELASTIC["top_k"], targets="gdn")
    print("\nbase (0 expertos):")
    capacity_report(model, bank)
    for n in (8, 32, 128):
        grow_experts(model, n - max(m.n_experts for m in model.modules()
                                    if isinstance(m, ElasticFFN)))
        print(f"\ncon {n} expertos por capa MoE:")
        capacity_report(model, bank)


def main():
    ap = argparse.ArgumentParser("AETHER v4.1 elastico")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if a.demo:
        demo()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
