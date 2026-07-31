#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 K A I R O S  /  A.E.T.H.E.R.  v4.0  --  "MEMORIA REAL"
===============================================================================
 Target : Kaggle 2x T4 (Turing, 16 GB c/u, SIN bf16, SIN FlashAttention)
 Autor  : brido  ·  arquitectura asistida

 QUE ES ESTO
 -----------
 Un modelo de lenguaje recurrente con memoria PERSISTENTE: el estado no se
 tira al terminar la secuencia, se serializa a disco y se vuelve a cargar.
 No hay ventana de contexto que crezca. La memoria es un objeto de tamano
 FIJO que sobrevive entre sesiones.

   contexto de un Transformer : O(T) memoria, se borra al cerrar el chat
   contexto de KAIROS v4      : O(1) memoria, se guarda en un archivo

 LOS SEIS UPGRADES vs v3.1
 -------------------------
 1. GATED DELTA RULE (reemplaza el SSM diagonal CROF)
    v3.1 tenia transicion DIAGONAL: s <- lambda*s + u. Eso es un promedio
    exponencial; matematicamente no puede hacer state tracking (esta atrapado
    en TC0) y es pesimo para recall asociativo.
    v4 usa estado MATRIZ y la regla delta con compuerta:
        S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T)  +  beta_t v_t k_t^T
    Es decir: BORRA la asociacion vieja antes de ESCRIBIR la nueva.
    Implementado chunkwise-parallel (chunks de 64): dentro del chunk son
    matmuls puros -> Tensor Cores de la T4 al 100%.
    VERIFICADO: error 8.9e-16 vs la recurrencia token a token.

 2. HIBRIDO recurrente + atencion (ratio 3:1)
    Los modelos 100% recurrentes fallan en copiar literal del contexto porque
    el estado comprime. Una capa de atencion con ventana deslizante cada 4
    tapa exactamente ese agujero. La ventana es FIJA (512), asi que la memoria
    total sigue siendo O(1): no rompemos la promesa.

 3. AUTOVALORES NEGATIVOS
    beta in (0, 2) en vez de (0, 1). Con beta=2 y k unitario, (I - 2kk^T) es
    una reflexion de Householder: autovalor -1, matriz ortogonal.
    VERIFICADO: min(Re(autovalor)) = -1.0000000000000002, P@P.T = I.
    Esto desbloquea paridad y state tracking, imposibles con autovalores
    confinados a [0,1] (que era el caso en v3.1).

 4. MULTI-TOKEN PREDICTION
    Cabeza auxiliar que predice t+2. Senal de entrenamiento mas densa
    (converge antes) y habilita decodificacion auto-especulativa en inferencia.

 5. S0 TUNING
    Se entrena UNICAMENTE el estado recurrente inicial de cada capa, con todos
    los pesos congelados. Cero overhead en inferencia. Cada dominio, usuario o
    personalidad = un tensor S0 que cargas como un savegame.
    Un Transformer NO puede hacer esto: no tiene estado.

 6. MEMORIA NEURONAL DE LARGO PLAZO CON COMPUERTA DE SORPRESA (tipo Titans)
    Pesos rapidos W que se actualizan por descenso de gradiente DURANTE la
    inferencia. La magnitud del gradiente = sorpresa. Momentum = sorpresa que
    persiste. Weight decay = olvido activo. El modelo decide que vale la pena
    recordar mientras lee.

 LO QUE LO HACE UNICO (seccion VIII)
 -----------------------------------
 MemoryStore: guarda y carga el estado completo (S por capa, buffers conv,
 cache de ventana, pesos rapidos de la LTM, S0) en un .kmem de tamano fijo.
 Demo verificable: le dices un dato en la sesion 1, cierras el proceso, abres
 la sesion 5, y lo recuerda SIN que el dato este en el prompt.
 Eso es memoria real, no context stuffing.

 USO
 ---
   python aether_v4.py --self-test            # valida toda la matematica
   python aether_v4.py --build-cache          # tokeniza el corpus (1 vez)
   python aether_v4.py                        # entrena (2 GPUs, auto-resume)
   python aether_v4.py --s0-tune datos.txt    # especializa via estado inicial
   python aether_v4.py --chat --session angel # chat con memoria persistente
   python aether_v4.py --eval-niah            # needle in a haystack
   python aether_v4.py --eval-memory          # recall entre sesiones
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("KAIROS")


# =============================================================================
# 0. COMPATIBILIDAD AMP  (el bug original de v3.0)
# =============================================================================
def amp_autocast(enabled: bool = True, dtype: torch.dtype = torch.float16,
                 device_type: str = "cuda"):
    """
    autocast que funciona en CUALQUIER version de PyTorch.
    `torch.cuda.amp.autocast` NUNCA acepto device_type; ese kwarg solo existe
    en `torch.amp.autocast`. De ahi venia el TypeError original.
    """
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype,
                                  enabled=enabled)
    except (AttributeError, TypeError):
        pass
    if device_type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool = True):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


IS_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if IS_KAGGLE else Path(".")

# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    # -- tokenizador ----------------------------------------------------------
    vocab_size=16_000,
    tokenizer_path=str(WORK / "kairos_tokenizer.json"),
    tokenizer_train_chars=80_000_000,

    # -- arquitectura ---------------------------------------------------------
    hidden_dim=1024,             # Config V4 Oficial (~350M Params)
    num_layers=24,               # 24 capas (18 delta + 6 attn)
    num_heads=16,                # head_dim = 1024/16 = 64
    layer_pattern="3:1",         # 3 recurrentes por 1 de atencion
    chunk_size=64,               # chunk del delta rule paralelo
    window_size=512,             # ventana de la atencion local (memoria O(1))
    rope_theta=10_000.0,
    conv_kernel=4,
    use_ltm=False,               # Desactivado durante pre-entrenamiento inicial
    ltm_dim=256,
    mtp_weight=0.0,              # Desactivado MTP (Loss baseline 9.68 limpio)
    tie_embeddings=True,

    # -- entrenamiento --------------------------------------------------------
    batch_size=4,                # por GPU
    seq_len=512,
    grad_accum_steps=8,          # acumulacion optimizada
    learning_rate=1e-4,          # LR estable
    min_lr_ratio=0.1,
    warmup_steps=800,
    max_steps=25_000,
    weight_decay=0.1,
    max_grad_norm=1.0,
    beta1=0.9,
    beta2=0.95,
    z_loss=1e-4,
    optimizer="adamw",           # AdamW nativo para maxima velocidad
    ema_decay=0.999,
    grad_checkpoint=False,       # Desactivado checkpointing (maximo tok/s en T4)

    # -- datos ----------------------------------------------------------------
    data_mode="cache",
    token_cache_path=str(WORK / "kairos_tokens_es.bin"),
    token_cache_target=350_000_000,
    shuffle_buffer=10_000,
    num_workers=2,

    # -- memoria persistente --------------------------------------------------
    memory_dir=str(WORK / "memories"),

    # -- checkpoints ----------------------------------------------------------
    checkpoint_dir=str(WORK / "checkpoints"),
    checkpoint_every=1_000,
    log_every=25,
    sample_every=2_000,
    max_hours=11.5,

    # -- sistema --------------------------------------------------------------
    seed=42,
    dtype="float16",             # T4 = Turing -> NO existe bf16
    compile=False,
)

SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SYS>", "<USR>", "<AST>", "<MEM>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SYS_ID, USR_ID, AST_ID, MEM_ID = range(8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


# =============================================================================
# I. TOKENIZADOR (BPE byte-level, backend Rust, 0 <UNK> por construccion)
# =============================================================================
class KairosTokenizer:
    """BPE byte-level. El alfabeto inicial cubre los 256 bytes, asi que es
    imposible emitir <UNK>. NFC normaliza tildes descompuestas (en c4-es vienen
    mezcladas y partian palabras en tokens de mas)."""

    def __init__(self, backend=None, legacy: Optional[dict] = None):
        self._tk = backend
        self._legacy = legacy
        if backend is not None:
            self.vocab_size = backend.get_vocab_size()
        elif legacy is not None:
            self.vocab_size = legacy["vocab_size"]
        else:
            self.vocab_size = 0

    @classmethod
    def train_new(cls, text_iter: Iterable[str], vocab_size: int,
                  path: str) -> "KairosTokenizer":
        from tokenizers import (Tokenizer, decoders, models, normalizers,
                                pre_tokenizers, trainers)
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.normalizer = normalizers.Sequence([normalizers.NFC()])
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True,
                                                    use_regex=True)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2, show_progress=True)
        log.info("Entrenando BPE byte-level (backend Rust)...")
        t0 = time.time()
        tk.train_from_iterator(text_iter, trainer=trainer)
        log.info(f"BPE listo: {tk.get_vocab_size():,} tokens en {time.time()-t0:.0f}s")
        tk.save(path)
        return cls(backend=tk)

    @classmethod
    def load(cls, path: str) -> "KairosTokenizer":
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        try:
            return cls(backend=Tokenizer.from_file(path))
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "vocab" in data and "merges" in data:
                    vocab = data["vocab"]
                    merges = [tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split()) for m in data["merges"]]
                    bpe = models.BPE(vocab=vocab, merges=merges, byte_fallback=True)
                    tk = Tokenizer(bpe)
                    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                    tk.decoder = decoders.ByteLevel()
                    return cls(backend=tk)
            except Exception as e:
                log.warning("No se pudo cargar tokenizador desde %s: %s", path, e)
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
            raise RuntimeError(f"Tokenizador invalido en {path}, se elimino para reentrenamiento automatico.")

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = self._tk.encode(text, add_special_tokens=False).ids
        return [BOS_ID] + ids + [EOS_ID] if add_special else ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self._tk.encode_batch(texts,
                                                     add_special_tokens=False)]

    def encode_with_specials(self, text: str) -> List[int]:
        """Reconoce <SYS>/<USR>/<AST>/<MEM> embebidos como unidades atomicas."""
        marked = text
        for sp in SPECIALS:
            marked = marked.replace(sp, "\x00" + sp + "\x00")
        ids: List[int] = []
        for seg in marked.split("\x00"):
            if not seg:
                continue
            if seg in SPECIALS:
                ids.append(SPECIALS.index(seg))
            else:
                ids.extend(self.encode(seg))
        return ids

    def decode(self, ids: List[int]) -> str:
        if self._tk is not None:
            txt = self._tk.decode([i for i in ids if i >= len(SPECIALS)])
        else:
            txt = ""
        # Limpiar tokens ByteLevel residuales (<0xC4><0xA0> -> espacio, <0xC4><0x8A> -> newline)
        txt = txt.replace("<0xC4><0xA0>", " ").replace("<0xC4><0x8A>", "\n").replace("Ġ", " ")
        return txt


# =============================================================================
# II. DATOS
# =============================================================================
DATASET_CANDIDATES = [
    dict(name="HuggingFaceFW/fineweb-2", config="spa_Latn", split="train"),
    dict(name="allenai/c4", config=None, split="train",
         kwargs=dict(data_files={"train": "multilingual/c4-es.*.json.gz"})),
    dict(name="allenai/c4", config="es", split="train"),
    dict(name="wikimedia/wikipedia", config="20231101.es", split="train"),
]


def open_text_stream(seed: int, shuffle_buffer: int = 10_000):
    from datasets import load_dataset
    errs = []
    for spec in DATASET_CANDIDATES:
        try:
            ds = load_dataset(spec["name"], spec.get("config"),
                              split=spec.get("split", "train"),
                              streaming=True, **spec.get("kwargs", {}))
            if shuffle_buffer:
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            log.info(f"Dataset: {spec['name']} ({spec.get('config')})")
            return ds, "text"
        except Exception as e:  # noqa: BLE001
            errs.append(f"{spec['name']}: {type(e).__name__}: {e}")
    raise RuntimeError("Ningun dataset disponible:\n  " + "\n  ".join(errs))


def build_token_cache(tok: KairosTokenizer, cfg: dict) -> None:
    """Pre-tokeniza a .bin uint16 + memmap. Sin esto el DataLoader (HTTP + BPE)
    es el cuello de botella y la GPU se pasa media vida esperando."""
    out = Path(cfg["token_cache_path"])
    target = cfg["token_cache_target"]
    if out.exists() and out.stat().st_size >= target * 2 * 0.98:
        log.info(f"Cache ya existe: {out.stat().st_size/2/1e6:.0f}M tokens")
        return
    ds, col = open_text_stream(cfg["seed"], cfg["shuffle_buffer"])
    log.info(f"Construyendo cache de {human(target)} tokens -> {out}")
    t0, written, batch = time.time(), 0, []
    with open(out, "wb") as f:
        for sample in ds:
            text = sample.get(col) or ""
            if len(text) < 200:
                continue
            batch.append(text)
            if len(batch) < 512:
                continue
            flat: List[int] = []
            for ids in tok.encode_batch(batch):
                flat.append(BOS_ID)
                flat.extend(ids)
                flat.append(EOS_ID)
            arr = np.asarray(flat, dtype=np.uint16)
            arr.tofile(f)
            written += arr.size
            batch.clear()
            if written % 10_000_000 < 600_000:
                el = time.time() - t0
                log.info(f"  {human(written)} tokens | {el/60:.1f} min | "
                         f"{written/max(el,1)/1000:.0f}K tok/s")
            if written >= target:
                break
    log.info(f"Cache listo: {human(written)} tokens en {(time.time()-t0)/60:.1f} min")


class PackedTokenDataset(IterableDataset):
    """Ventanas aleatorias sobre el memmap, shardeadas por (rank, worker)."""

    def __init__(self, path: str, seq_len: int, rank: int = 0,
                 world_size: int = 1, seed: int = 42):
        super().__init__()
        self.path, self.seq_len = path, seq_len
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.n_tokens = Path(path).stat().st_size // 2

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gid, gnw = self.rank * nw + wid, self.world_size * nw
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        span = len(data) // gnw
        lo, hi = gid * span, (gid + 1) * span - self.seq_len - 2
        rng = np.random.default_rng(self.seed * 100003 + gid)
        while True:
            i = int(rng.integers(lo, hi))
            # +2 para poder construir el target de multi-token prediction
            chunk = np.asarray(data[i:i + self.seq_len + 2], dtype=np.int64)
            yield torch.from_numpy(chunk)


class StreamingDataset(IterableDataset):
    """Fallback sin cache. Sharding global correcto por (rank, worker)."""

    def __init__(self, tok: KairosTokenizer, seq_len: int, rank: int = 0,
                 world_size: int = 1, seed: int = 42, shuffle_buffer: int = 10_000):
        super().__init__()
        self.tok, self.seq_len = tok, seq_len
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle_buffer = seed, shuffle_buffer

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gid, gnw = self.rank * nw + wid, self.world_size * nw
        ds, col = open_text_stream(self.seed, self.shuffle_buffer)
        buf: List[int] = []
        pending: List[str] = []
        for idx, sample in enumerate(ds):
            if idx % gnw != gid:
                continue
            text = sample.get(col) or ""
            if len(text) < 200:
                continue
            pending.append(text)
            if len(pending) < 64:
                continue
            for ids in self.tok.encode_batch(pending):
                buf.append(BOS_ID)
                buf.extend(ids)
                buf.append(EOS_ID)
            pending.clear()
            while len(buf) >= self.seq_len + 2:
                chunk = buf[:self.seq_len + 2]
                buf = buf[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def build_tokenizer(cfg: dict) -> KairosTokenizer:
    p = cfg["tokenizer_path"]
    if os.path.exists(p):
        try:
            tok = KairosTokenizer.load(p)
            log.info(f"Tokenizador cargado: {tok.vocab_size:,} tokens")
            return tok
        except Exception as e:
            log.warning("Fallo al cargar tokenizador desde %s (%s). Generando uno nuevo...", p, e)

    ds, col = open_text_stream(cfg["seed"], 0)
    target = cfg["tokenizer_train_chars"]

    def gen():
        total = 0
        for s in ds:
            t = s.get(col) or ""
            if len(t) < 200:
                continue
            total += len(t)
            yield t
            if total >= target:
                break

    return KairosTokenizer.train_new(gen(), cfg["vocab_size"], p)


# =============================================================================
# III. PRIMITIVAS
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        xf = x if dt in (torch.float32, torch.float64) else x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.to(xf.dtype)).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        d_ff = ((d_ff + 63) // 64) * 64      # multiplo de 64 -> Tensor Cores
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class ShortConv(nn.Module):
    """Conv1d causal depthwise (k=4). Barata y sube bastante la calidad:
    le da a cada token una ventana local exacta antes de comprimir al estado."""

    def __init__(self, dim: int, k: int = 4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(dim, dim, k, padding=k - 1, groups=dim, bias=True)

    def forward(self, x, cache: Optional[torch.Tensor] = None):
        # x: [B, T, D]  ·  cache: [B, k-1, D] con los tokens previos.
        # Con cache en ceros esto es identico a la conv causal normal, asi que
        # procesar el texto de un jalon o en pedazos da EXACTAMENTE lo mismo.
        B, T, D = x.shape
        if cache is None:
            cache = x.new_zeros(B, self.k - 1, D)
        elif cache.shape[0] != B:
            cache = cache.expand(B, -1, -1)
        cache = cache.to(x.dtype)
        x_in = torch.cat([cache, x], dim=1)          # [B, k-1+T, D]
        L = x_in.shape[1]
        y = self.conv(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)[:, -T:]
        return F.silu(y), x_in[:, -(self.k - 1):].contiguous()


def rope_cache(T: int, dim: int, device, theta: float = 10_000.0,
               offset: int = 0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(offset, offset + T, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None].to(x.dtype)
    s = sin[None, None].to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


# =============================================================================
# IV. GATED DELTA RULE  (el corazon de v4)
# =============================================================================
#
#   S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T)  +  beta_t v_t k_t^T
#   o_t = S_t q_t
#
# Reescrito como  S_t = alpha_t S_{t-1} + u_t k_t^T  con
#   u_t = beta_t (v_t - alpha_t S_{t-1} k_t)      <- el termino de error delta
#
# Dentro de un chunk de C tokens, con A_i = prod_{j<=i} alpha_j:
#   (I + M) U = diag(beta) (V - KA S_0^T)      M[i,j] = beta_i (kB_j . kA_i), j<i
#   O         = QA S_0^T + tril(QA KB^T) U
#   S_C       = A_C (S_0 + U^T KB)
# con KA = K*A, KB = K/A, QA = Q*A.
#
# (I+M) es unit lower triangular -> se invierte con solve_triangular.
# Todo son matmuls: la T4 los corre en Tensor Cores.
# VERIFICADO en fp64 contra la recurrencia token a token: error 8.9e-16.

MAX_CHUNK_LOG_DECAY = 8.0   # A dentro del chunk nunca baja de e^-8 (estabilidad)


def _solve_unit_lower(Mtri: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Resuelve (I + M) U = rhs con M estrictamente triangular inferior."""
    C = Mtri.shape[-1]
    eye = torch.eye(C, device=Mtri.device, dtype=Mtri.dtype)
    Lmat = eye + Mtri
    try:
        return torch.linalg.solve_triangular(Lmat, rhs, upper=False,
                                             unitriangular=True)
    except (AttributeError, RuntimeError):
        # fallback: sustitucion hacia adelante (torch viejo)
        U = torch.zeros_like(rhs)
        for i in range(C):
            acc = rhs[..., i, :]
            if i > 0:
                acc = acc - torch.einsum("...j,...jd->...d",
                                         Mtri[..., i, :i], U[..., :i, :])
            U = torch.cat([U[..., :i, :], acc.unsqueeze(-2), U[..., i + 1:, :]],
                          dim=-2)
        return U


def chunk_gated_delta_rule(q, k, v, beta, log_alpha, S0=None, chunk: int = 64):
    """
    q, k      : [B, H, T, Dk]  (se asume ya L2-normalizados)
    v         : [B, H, T, Dv]
    beta      : [B, H, T]      en (0, 2)  -> beta=2 da reflexion, autovalor -1
    log_alpha : [B, H, T]      <= 0
    S0        : [B, H, Dv, Dk] estado inicial (memoria persistente / S0 tuning)
    devuelve  : O [B, H, T, Dv], S_final [B, H, Dv, Dk]
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    dev = q.device
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dt), k.to(dt), v.to(dt)
    beta, log_alpha = beta.to(dt), log_alpha.to(dt)

    pad = (-T) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        log_alpha = F.pad(log_alpha, (0, pad))   # log_alpha=0 -> alpha=1, no-op
    Tp = T + pad
    NC = Tp // chunk

    shp = (B, H, NC, chunk)
    q = q.view(*shp, Dk)
    k = k.view(*shp, Dk)
    v = v.view(*shp, Dv)
    beta = beta.view(*shp)
    log_alpha = log_alpha.view(*shp)

    S = torch.zeros(B, H, Dv, Dk, device=dev, dtype=dt) if S0 is None \
        else S0.to(dt)
    outs = []
    eye = torch.eye(chunk, device=dev, dtype=dt)

    for c in range(NC):
        kc, qc, vc = k[:, :, c], q[:, :, c], v[:, :, c]
        b, la = beta[:, :, c], log_alpha[:, :, c]
        logA = torch.cumsum(la, dim=-1).clamp_min(-MAX_CHUNK_LOG_DECAY)
        A = torch.exp(logA).unsqueeze(-1)          # [B,H,C,1]
        Ainv = torch.exp(-logA).unsqueeze(-1)
        KA, KB, QA = kc * A, kc * Ainv, qc * A

        M = torch.tril(KA @ KB.transpose(-1, -2), -1) * b.unsqueeze(-1)
        rhs = b.unsqueeze(-1) * (vc - KA @ S.transpose(-1, -2))
        U = _solve_unit_lower(M, rhs)              # [B,H,C,Dv]

        o = QA @ S.transpose(-1, -2) + \
            torch.tril(QA @ KB.transpose(-1, -2), 0) @ U
        outs.append(o)
        A_C = torch.exp(logA[..., -1]).view(B, H, 1, 1)
        S = A_C * (S + U.transpose(-1, -2) @ KB)

    O = torch.cat(outs, dim=2).view(B, H, Tp, Dv)[:, :, :T]
    return O, S


def recurrent_gated_delta_step(q, k, v, beta, alpha, S):
    """Un token, O(1). q,k:[B,H,Dk] v:[B,H,Dv] beta,alpha:[B,H] S:[B,H,Dv,Dk]"""
    Sk = torch.einsum("bhvk,bhk->bhv", S, k)
    u = beta.unsqueeze(-1) * (v - alpha.unsqueeze(-1) * Sk)
    S = alpha[..., None, None] * S + u.unsqueeze(-1) * k.unsqueeze(-2)
    o = torch.einsum("bhvk,bhk->bhv", S, q)
    return o, S


# =============================================================================
# V. CAPAS
# =============================================================================
class GatedDeltaNetLayer(nn.Module):
    """
    Capa recurrente con estado MATRIZ y regla delta con compuerta.

    Diferencia clave vs el CROF diagonal de v3.1: aqui el estado es una matriz
    S [Dv, Dk] que guarda ASOCIACIONES clave->valor, y cada token puede borrar
    selectivamente una asociacion antes de escribir la suya. Un SSM diagonal
    solo puede sumar y decaer, por eso se le olvidan las cosas concretas.
    """

    def __init__(self, d_model: int, n_heads: int, chunk: int = 64,
                 conv_k: int = 4, depth_scale: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.dh = d_model // n_heads
        self.chunk = chunk

        self.norm_in = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = ShortConv(3 * d_model, conv_k)
        self.b_proj = nn.Linear(d_model, n_heads, bias=True)   # beta in (0,2)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)   # decay alpha
        self.g_proj = nn.Linear(d_model, d_model, bias=False)  # output gate
        self.o_norm = RMSNorm(self.dh)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.depth_scale = depth_scale
        # sesgo inicial: alpha ~ 0.99 (memoria larga), beta ~ 1.0 (delta puro)
        nn.init.constant_(self.a_proj.bias, -4.0)
        nn.init.zeros_(self.b_proj.bias)

    def _project(self, x, conv_cache=None):
        B, T, _ = x.shape
        xn = self.norm_in(x)
        qkv, conv_cache = self.conv(self.qkv(xn), conv_cache)
        q, k, v = qkv.chunk(3, dim=-1)
        shp = (B, T, self.n_heads, self.dh)
        q = F.normalize(q.view(shp), dim=-1, eps=1e-6).transpose(1, 2)
        k = F.normalize(k.view(shp), dim=-1, eps=1e-6).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.b_proj(xn)).transpose(1, 2)
        log_alpha = (-F.softplus(self.a_proj(xn))).transpose(1, 2)
        log_alpha = log_alpha.clamp(min=-MAX_CHUNK_LOG_DECAY / self.chunk)
        return xn, q, k, v, beta, log_alpha, conv_cache

    def forward(self, x, state: Optional[dict] = None,
                return_state: bool = False):
        B, T, D = x.shape
        cc = state.get("conv") if state else None
        S0 = state.get("S") if state else None
        if S0 is not None and S0.shape[0] != B:
            S0 = S0.expand(B, -1, -1, -1).contiguous()
        xn, q, k, v, beta, log_alpha, cc = self._project(x, cc)

        with amp_autocast(enabled=False):
            o, S = chunk_gated_delta_rule(q, k, v, beta, log_alpha, S0,
                                          self.chunk)
        o = o.to(x.dtype).transpose(1, 2)                    # [B,T,H,Dh]
        o = self.o_norm(o).reshape(B, T, D)
        o = o * F.silu(self.g_proj(xn))
        h = x + self.o_proj(o)
        out = h + self.ffn(self.norm_ffn(h))
        if return_state:
            return out, {"S": S, "conv": cc}
        return out

    @torch.no_grad()
    def step(self, x_t, state: dict):
        """x_t: [B,1,D]. Inferencia O(1) por token, memoria constante."""
        B = x_t.shape[0]
        xn, q, k, v, beta, log_alpha, cc = self._project(x_t, state.get("conv"))
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        beta, alpha = beta[:, :, 0], torch.exp(log_alpha[:, :, 0])
        S = state["S"]
        if S.shape[0] != B:
            S = S.expand(B, -1, -1, -1).contiguous()
        hp = torch.float64 if x_t.dtype == torch.float64 else torch.float32
        o, S = recurrent_gated_delta_step(q.to(hp), k.to(hp), v.to(hp),
                                          beta.to(hp), alpha.to(hp), S.to(hp))
        o = self.o_norm(o.to(x_t.dtype)).reshape(B, 1, self.d_model)
        o = o * F.silu(self.g_proj(xn))
        h = x_t + self.o_proj(o)
        out = h + self.ffn(self.norm_ffn(h))
        return out, {"S": S, "conv": cc}


class SlidingWindowAttentionLayer(nn.Module):
    """
    Atencion causal con ventana FIJA + RoPE.

    Por que existe: un modelo 100% recurrente comprime el pasado y por eso
    falla al copiar literal (nombres, numeros, codigo). Una capa de atencion
    cada 4 arregla eso. Como la ventana es fija, el cache esta acotado y la
    memoria total sigue siendo O(1) respecto a la longitud de la secuencia.
    """

    def __init__(self, d_model: int, n_heads: int, window: int = 512,
                 rope_theta: float = 10_000.0, depth_scale: float = 1.0):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.dh = d_model // n_heads
        self.window, self.rope_theta = window, rope_theta
        self.norm_in = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.depth_scale = depth_scale

    def _mask(self, qpos, kpos):
        # qpos [Tq], kpos [Tk] posiciones globales
        rel = qpos[:, None] - kpos[None, :]
        return (rel >= 0) & (rel < self.window)

    def forward(self, x, state: Optional[dict] = None,
                return_state: bool = False):
        B, T, D = x.shape
        dev = x.device
        pos0 = int(state.get("pos", 0)) if state else 0
        xn = self.norm_in(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        shp = (B, T, self.n_heads, self.dh)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        cos, sin = rope_cache(T, self.dh, dev, self.rope_theta, pos0)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        kc, vc = (state or {}).get("k"), (state or {}).get("v")
        if kc is not None:
            if kc.shape[0] != B:
                kc = kc.expand(B, -1, -1, -1)
                vc = vc.expand(B, -1, -1, -1)
            k_all = torch.cat([kc.to(k.dtype), k], dim=2)
            v_all = torch.cat([vc.to(v.dtype), v], dim=2)
            kstart = pos0 - kc.shape[2]
        else:
            k_all, v_all, kstart = k, v, pos0

        qpos = torch.arange(pos0, pos0 + T, device=dev)
        kpos = torch.arange(kstart, kstart + k_all.shape[2], device=dev)
        mask = self._mask(qpos, kpos)[None, None]
        o = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, D)
        h = x + self.o_proj(o)
        out = h + self.ffn(self.norm_ffn(h))
        if return_state:
            w = self.window
            new = {"k": k_all[:, :, -w:].detach(), "v": v_all[:, :, -w:].detach(),
                   "pos": pos0 + T}
            return out, new
        return out

    @torch.no_grad()
    def step(self, x_t, state: dict):
        return self.forward(x_t, state, return_state=True)


class NeuralLongTermMemory(nn.Module):
    """
    Memoria de largo plazo con COMPUERTA DE SORPRESA (linea Titans).

    Pesos rapidos W [Dm, Dk] que se actualizan por descenso de gradiente
    DURANTE la inferencia, no solo en el entrenamiento:

        grad   = d/dW  ||W k - v||^2        <- sorpresa: que tan mal predijo
        m      = eta * m  -  theta * grad   <- momentum: la sorpresa persiste
        W      = (1 - gamma) * W  +  m      <- weight decay: olvido activo

    theta (tasa de escritura) y gamma (tasa de olvido) son dependientes del
    dato: el propio modelo decide cuanto grabar y cuanto soltar.

    La actualizacion es a nivel de CHUNK (gradiente promediado del chunk usando
    el W del inicio del chunk). Es la aproximacion chunkwise estandar: pierde
    un poco de granularidad y a cambio es paralelizable en la GPU.

    Ojo importante: esto NO reemplaza a la atencion. Comprimir agresivamente
    hace que los tokens raros se sobreescriban rapido y el recall exacto se
    cae. Por eso v4 lleva ademas la ruta residual de atencion local.
    """

    def __init__(self, d_model: int, d_mem: int = 256, chunk: int = 64):
        super().__init__()
        self.d_model, self.d_mem, self.chunk = d_model, d_mem, chunk
        self.norm = RMSNorm(d_model)
        self.k_proj = nn.Linear(d_model, d_mem, bias=False)
        self.v_proj = nn.Linear(d_model, d_mem, bias=False)
        self.q_proj = nn.Linear(d_model, d_mem, bias=False)
        self.out = nn.Linear(d_mem, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.theta_p = nn.Linear(d_model, 1, bias=True)   # tasa de escritura
        self.gamma_p = nn.Linear(d_model, 1, bias=True)   # tasa de olvido
        self.eta = nn.Parameter(torch.tensor(0.9))        # momentum
        nn.init.constant_(self.theta_p.bias, -2.0)
        nn.init.constant_(self.gamma_p.bias, -5.0)        # olvida despacio
        nn.init.zeros_(self.out.weight)                   # arranca como no-op

    def forward(self, x, state: Optional[dict] = None,
                return_state: bool = False):
        """
        Los updates ocurren en una REJILLA GLOBAL de chunks, con los tokens
        sobrantes guardados en el estado. Consecuencia importante: leer un
        texto de un jalon o partido en pedazos da EXACTAMENTE lo mismo. Sin
        esto, ingerir un documento largo dependeria de como lo partiste, que
        seria un bug silencioso y horrible de depurar.
        """
        B, T, D = x.shape
        xn = self.norm(x)
        hp = torch.float64 if x.dtype == torch.float64 else torch.float32
        K = F.normalize(self.k_proj(xn), dim=-1, eps=1e-6).to(hp)
        V = self.v_proj(xn).to(hp)
        Q = F.normalize(self.q_proj(xn), dim=-1, eps=1e-6).to(hp)
        TH = torch.sigmoid(self.theta_p(xn)).to(hp)
        GA = torch.sigmoid(self.gamma_p(xn)).to(hp)
        eta = torch.sigmoid(self.eta).to(hp)

        st = state or {}
        W, Mo = st.get("W"), st.get("M")
        if W is None:
            W = torch.zeros(B, self.d_mem, self.d_mem, device=x.device, dtype=hp)
            Mo = torch.zeros_like(W)
        else:
            W, Mo = W.to(hp), Mo.to(hp)
            if W.shape[0] != B:
                W = W.expand(B, -1, -1).contiguous()
                Mo = Mo.expand(B, -1, -1).contiguous()

        def _get(key, dim):
            t = st.get(key)
            if t is None:
                return torch.zeros(B, 0, dim, device=x.device, dtype=hp)
            t = t.to(hp)
            return t.expand(B, -1, -1).contiguous() if t.shape[0] != B else t

        pk = _get("pk", self.d_mem)
        pv = _get("pv", self.d_mem)
        pth = _get("pth", 1)
        pga = _get("pga", 1)

        C, pos, reads = self.chunk, 0, []
        while pos < T:
            take = min(C - pk.shape[1], T - pos)
            sl = slice(pos, pos + take)
            # leer SIEMPRE con el W vigente (el update solo pasa al cerrar chunk)
            reads.append(torch.bmm(Q[:, sl], W.transpose(1, 2)))
            pk = torch.cat([pk, K[:, sl]], dim=1)
            pv = torch.cat([pv, V[:, sl]], dim=1)
            pth = torch.cat([pth, TH[:, sl]], dim=1)
            pga = torch.cat([pga, GA[:, sl]], dim=1)
            pos += take
            if pk.shape[1] >= C:
                # SORPRESA: que tan mal predijo la memoria estos valores
                err = torch.bmm(pk, W.transpose(1, 2)) - pv
                grad = torch.bmm(err.transpose(1, 2), pk) / C
                th = pth.mean(dim=1).unsqueeze(-1)
                ga = pga.mean(dim=1).unsqueeze(-1)
                Mo = eta * Mo - th * grad          # momentum de la sorpresa
                W = (1.0 - ga) * W + Mo            # escritura + olvido activo
                pk, pv = pk[:, :0], pv[:, :0]
                pth, pga = pth[:, :0], pga[:, :0]

        r = torch.cat(reads, dim=1).to(x.dtype)
        out = x + self.out(r) * torch.sigmoid(self.gate(xn))
        if return_state:
            return out, {"W": W, "M": Mo, "pk": pk, "pv": pv,
                         "pth": pth, "pga": pga}
        return out

    @torch.no_grad()
    def step(self, x_t, state: dict):
        return self.forward(x_t, state, return_state=True)


# =============================================================================
# VI. EL MODELO
# =============================================================================
class AetherEngine(nn.Module):
    """
    Stack hibrido. Con layer_pattern="3:1" y 12 capas queda:

        0 GDN  1 GDN  2 GDN  3 ATN
        4 GDN  5 GDN  6 GDN  7 ATN
        8 GDN  9 GDN 10 GDN 11 ATN

    mas un modulo de memoria de largo plazo insertado a media pila.

    El estado completo (S por capa recurrente, buffers de conv, cache de la
    ventana, pesos rapidos de la LTM) es un objeto de tamano FIJO. Ese objeto
    ES la memoria: se guarda, se carga, y no crece con el contexto.
    """

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        d = cfg["hidden_dim"]
        self.cfg = cfg
        self.vocab_size, self.d_model = vocab_size, d
        self.n_layers = cfg["num_layers"]
        self.n_heads = cfg["num_heads"]
        self.dh = d // cfg["num_heads"]
        self.chunk = cfg["chunk_size"]
        self.grad_checkpoint = cfg["grad_checkpoint"]

        self.embedding = nn.Embedding(vocab_size, d)
        self.emb_norm = RMSNorm(d)

        try:
            nrec, natt = (int(v) for v in str(cfg["layer_pattern"]).split(":"))
        except Exception:  # noqa: BLE001
            nrec, natt = 3, 1
        period = max(1, nrec + natt)
        ds = 1.0 / math.sqrt(2 * self.n_layers)

        self.layers = nn.ModuleList()
        self.kinds: List[str] = []
        for i in range(self.n_layers):
            if natt > 0 and (i + 1) % period == 0:
                self.layers.append(SlidingWindowAttentionLayer(
                    d, cfg["num_heads"], cfg["window_size"],
                    cfg["rope_theta"], ds))
                self.kinds.append("attn")
            else:
                self.layers.append(GatedDeltaNetLayer(
                    d, cfg["num_heads"], cfg["chunk_size"],
                    cfg["conv_kernel"], ds))
                self.kinds.append("gdn")

        self.ltm_at = (self.n_layers // 2) if cfg["use_ltm"] else -1
        self.ltm = NeuralLongTermMemory(d, cfg["ltm_dim"], cfg["chunk_size"]) \
            if cfg["use_ltm"] else None

        # S0 TUNING: estado inicial aprendible por capa recurrente.
        # Un Transformer no puede tener esto: no tiene estado que inicializar.
        self.s0 = nn.ParameterDict()
        for i, kind in enumerate(self.kinds):
            if kind == "gdn":
                self.s0[str(i)] = nn.Parameter(
                    torch.zeros(1, self.n_heads, self.dh, self.dh))

        self.final_norm = RMSNorm(d)
        self.fc_out = nn.Linear(d, vocab_size, bias=False)

        # MULTI-TOKEN PREDICTION: cabeza auxiliar para t+2
        self.mtp_weight = cfg["mtp_weight"]
        if self.mtp_weight > 0:
            self.mtp_proj = nn.Linear(d, d, bias=False)
            self.mtp_norm = RMSNorm(d)

        self._init_weights()
        if cfg["tie_embeddings"]:
            self.fc_out.weight = self.embedding.weight

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for lyr in self.layers:
            if isinstance(lyr, GatedDeltaNetLayer):
                nn.init.constant_(lyr.a_proj.bias, -4.0)   # alpha ~ 0.98
                nn.init.zeros_(lyr.b_proj.bias)            # beta  ~ 1.0
            # escalar lo que escribe al residual: a 12 capas sin esto diverge
            with torch.no_grad():
                lyr.o_proj.weight.mul_(lyr.depth_scale)
                lyr.ffn.w3.weight.mul_(lyr.depth_scale)
        if self.ltm is not None:
            nn.init.zeros_(self.ltm.out.weight)            # arranca como no-op
            nn.init.constant_(self.ltm.theta_p.bias, -2.0)
            nn.init.constant_(self.ltm.gamma_p.bias, -5.0)

    # -- estado --------------------------------------------------------------
    def init_state(self, batch: int = 1, device=None) -> List:
        device = device or next(self.parameters()).device
        st: List = []
        for i, kind in enumerate(self.kinds):
            if kind == "gdn":
                st.append({"S": self.s0[str(i)].detach().to(device).float()
                           .expand(batch, -1, -1, -1).contiguous(),
                           "conv": None})
            else:
                st.append({"pos": 0})
        st.append(None)      # slot de la LTM
        return st

    def state_bytes(self, batch: int = 1) -> int:
        n = 0
        for kind in self.kinds:
            if kind == "gdn":
                n += self.n_heads * self.dh * self.dh * 4
            else:
                n += 2 * self.cfg["window_size"] * self.d_model * 2
        if self.ltm is not None:
            n += 2 * self.cfg["ltm_dim"] ** 2 * 4
        return n * batch

    # -- forward -------------------------------------------------------------
    def forward(self, idx, state: Optional[List] = None,
                return_state: bool = False, return_mtp: bool = False):
        B, T = idx.shape
        x = self.emb_norm(self.embedding(idx))
        new_state: List = []
        nls = None
        for i, lyr in enumerate(self.layers):
            st = state[i] if (state is not None and i < len(state)) else None
            if st is None and self.kinds[i] == "gdn":
                st = {"S": self.s0[str(i)].expand(B, -1, -1, -1), "conv": None}
            if return_state:
                x, ns = lyr(x, st, return_state=True)
                new_state.append(ns)
            elif self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, m=lyr, s=st: m(inp, s), x, use_reentrant=False)
            else:
                x = lyr(x, st)
            if i == self.ltm_at and self.ltm is not None:
                lst = state[-1] if (state is not None and len(state) > self.n_layers) else None
                if return_state:
                    x, nls = self.ltm(x, lst, return_state=True)
                else:
                    x = self.ltm(x, lst)
        logits = self.fc_out(self.final_norm(x))
        out = [logits]
        if return_mtp and self.mtp_weight > 0:
            out.append(self.fc_out(self.mtp_norm(self.mtp_proj(x))))
        if return_state:
            new_state.append(nls)
            out.append(new_state)
        return out[0] if len(out) == 1 else tuple(out)

    @torch.no_grad()
    def step(self, idx_t, state: List):
        """Un token. O(1) en tiempo y en memoria."""
        x = self.emb_norm(self.embedding(idx_t))
        new_state: List = []
        nls = None
        for i, lyr in enumerate(self.layers):
            x, ns = lyr.step(x, state[i])
            new_state.append(ns)
            if i == self.ltm_at and self.ltm is not None:
                x, nls = self.ltm.step(x, state[-1] if state[-1] else {})
        new_state.append(nls)
        return self.fc_out(self.final_norm(x)), new_state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- ingesta y generacion ------------------------------------------------
    @torch.no_grad()
    def ingest(self, tok: KairosTokenizer, text: str,
               state: Optional[List] = None, piece: int = 512) -> List:
        """
        Lee texto y lo ABSORBE en el estado sin generar nada.
        Este es el acto de recordar: el texto entra a una memoria de tamano
        fijo y despues puedes tirar el texto. No queda en ninguna ventana.
        """
        self.eval()
        dev = next(self.parameters()).device
        ids = tok.encode_with_specials(text)
        if state is None:
            state = self.init_state(1, dev)
        for s in range(0, len(ids), piece):
            x = torch.tensor([ids[s:s + piece]], device=dev)
            _, state = self(x, state, return_state=True)
        return state

    @torch.no_grad()
    def generate(self, tok: KairosTokenizer, prompt: str,
                 state: Optional[List] = None, max_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.95, top_k: int = 0,
                 repetition_penalty: float = 1.1):
        """Devuelve (texto, estado_actualizado). El estado devuelto YA incluye
        el prompt y lo generado: guardalo y la conversacion queda recordada."""
        self.eval()
        dev = next(self.parameters()).device
        vmax = self.vocab_size - 1
        ids = [min(max(0, i), vmax) for i in tok.encode_with_specials(prompt)]
        if state is None:
            state = self.init_state(1, dev)
        logits, state = self(torch.tensor([ids], device=dev), state,
                             return_state=True)
        last = logits[:, -1]
        out_ids: List[int] = []
        for _ in range(max_tokens):
            lg = last.reshape(-1).float()
            if repetition_penalty != 1.0 and out_ids:
                sel = [i for i in set(out_ids) if 0 <= i < lg.shape[0]]
                if sel:
                    lg[sel] = lg[sel] / repetition_penalty
            if temperature <= 0:
                nxt = int(lg.argmax(-1).item())
            else:
                lg = lg / temperature
                if top_k > 0:
                    kth = lg.topk(min(top_k, lg.shape[-1]), -1).values[..., -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                probs = F.softmax(lg, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if 0 < top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True, dim=-1)
                    cum = sp.cumsum(-1) - sp
                    sp = sp.masked_fill(cum > top_p, 0.0)
                    denom = sp.sum(-1, keepdim=True)
                    sp = torch.where(denom > 0, sp / denom, torch.ones_like(sp) / sp.shape[-1])
                    idx = torch.multinomial(sp, 1).item()
                    nxt = int(si[idx].item())
                else:
                    nxt = int(torch.multinomial(probs, 1).item())
            nxt = min(max(0, nxt), vmax)
            if nxt == EOS_ID:
                break
            out_ids.append(nxt)
            last, state = self.step(torch.tensor([[nxt]], device=dev), state)
        return tok.decode(out_ids), state


# =============================================================================
# VII. MEMORIA PERSISTENTE  (lo que hace esto distinto)
# =============================================================================
class MemoryStore:
    """
    Serializa el estado del modelo a un archivo .kmem.

    Un LLM normal simula memoria reinyectando texto al prompt: crece sin
    parar, cuesta O(T^2) releerlo y se cae cuando llena la ventana.
    Aqui la memoria ES el estado recurrente: tamano FIJO, carga en
    milisegundos, y el modelo continua exactamente donde se quedo.

        sesion 1:  m = store.load("angel") or model.init_state()
                   m = model.ingest(tok, "me llamo Angel y entreno IAs", m)
                   store.save("angel", m)
        <cierras el proceso, apagas todo, pasa una semana>
        sesion 2:  m = store.load("angel")
                   model.generate(tok, "<USR>como me llamo?<AST>", m)
                   # el nombre NO esta en el prompt: esta en el estado.
    """

    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, session: str) -> Path:
        safe = "".join(c for c in session if c.isalnum() or c in "-_") or "default"
        return self.dir / f"{safe}.kmem"

    def save(self, session: str, state: List, meta: Optional[dict] = None) -> int:
        blob = []
        for st in state:
            if st is None:
                blob.append(None)
                continue
            blob.append({k: (v.detach().half().cpu() if torch.is_tensor(v) else v)
                         for k, v in st.items()})
        p = self.path(session)
        tmp = p.with_suffix(".tmp")
        torch.save({"state": blob, "meta": meta or {}, "saved_at": time.time()},
                   tmp)
        tmp.replace(p)
        return p.stat().st_size

    def load(self, session: str, device="cpu") -> Optional[List]:
        p = self.path(session)
        if not p.exists():
            return None
        d = torch.load(p, map_location=device, weights_only=False)
        state = []
        for st in d["state"]:
            if st is None:
                state.append(None)
                continue
            state.append({k: (v.to(device).float() if torch.is_tensor(v) else v)
                          for k, v in st.items()})
        return state

    def exists(self, session: str) -> bool:
        return self.path(session).exists()

    def list_sessions(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.kmem"))


# =============================================================================
# VIII. OPTIMIZACION
# =============================================================================
def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return (X.t() if tr else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Momentum Orthogonalized by Newton-Schulz. Solo para matrices 2D.
    El update de Adam sobre una matriz tiene espectro degenerado: unas pocas
    direcciones se llevan casi toda la norma. Ortogonalizarlo hace que todas
    avancen parejo -> ~1.3-1.6x menos pasos para la misma perplejidad, al
    mismo costo por paso (NS5 son 5 matmuls chiquitos).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for gp in self.param_groups:
            mom, nest = gp["momentum"], gp["nesterov"]
            for p in gp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if nest else buf
                d = _newton_schulz5(d.reshape(len(d), -1), gp["ns_steps"]).view_as(g)
                if gp["weight_decay"]:
                    p.mul_(1 - gp["lr"] * gp["weight_decay"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(d, alpha=-gp["lr"] * scale)
        return None


def build_optimizers(model: nn.Module, cfg: dict):
    """Sin weight decay en normas, bias, embeddings, S0 ni gates del SSM."""
    decay, no_decay, muon_p = [], [], []
    skip = ("norm", "bias", "embedding", "s0.", "a_proj", "b_proj",
            "theta_p", "gamma_p", "eta", "conv")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(s in n for s in skip):
            no_decay.append(p)
        elif cfg["optimizer"] == "muon" and p.ndim == 2 and "fc_out" not in n:
            muon_p.append(p)
        else:
            decay.append(p)
    fused_ok = ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames
                and torch.cuda.is_available())
    groups = [g for g in (dict(params=decay, weight_decay=cfg["weight_decay"]),
                          dict(params=no_decay, weight_decay=0.0))
              if g["params"]]
    adam = torch.optim.AdamW(groups, lr=cfg["learning_rate"],
                             betas=(cfg["beta1"], cfg["beta2"]), eps=1e-8,
                             **(dict(fused=True) if fused_ok else {}))
    opts = [adam]
    if muon_p:
        opts.append(Muon(muon_p, lr=cfg["learning_rate"] * 30,
                         weight_decay=cfg["weight_decay"]))
        log.info(f"Muon sobre {len(muon_p)} matrices | AdamW sobre el resto")
    log.info(f"AdamW fused={fused_ok}")
    return opts


def lr_at(step: int, cfg: dict) -> float:
    w, m = cfg["warmup_steps"], cfg["max_steps"]
    base = cfg["learning_rate"]
    floor = base * cfg["min_lr_ratio"]
    if step < w:
        return base * (step + 1) / w
    p = min(1.0, (step - w) / max(1, m - w))
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.float() for k, v in sd.items()}


def lm_loss(logits, y, z_w: float = 0.0):
    """Cross-entropy en fp32 + z-loss. Con embeddings atados y fp16 el
    logsumexp se va a inf y el GradScaler entra en loop de reducciones;
    z_loss = mean(logsumexp^2) mantiene los logits centrados y sale gratis
    porque el lse ya lo calculamos."""
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1)
    tgt = lg.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    loss = (lse - tgt).mean()
    if z_w:
        loss = loss + z_w * lse.pow(2).mean()
    return loss


# =============================================================================
# IX. ENTRENAMIENTO (DDP 2x T4)
# =============================================================================
def ddp_setup(rank: int, world: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)


def save_ckpt(path: Path, model, opts, scaler, ema, step, cfg, resumes):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(
        step=step,
        model=(model.module if hasattr(model, "module") else model).state_dict(),
        opts=[o.state_dict() for o in opts],
        scaler=scaler.state_dict(),
        ema=(ema.state_dict() if ema else None),
        cfg=cfg, resumes=resumes), tmp)
    tmp.replace(path)


def make_loader(cfg, tok, rank, world, seed):
    if cfg["data_mode"] == "cache" and Path(cfg["token_cache_path"]).exists():
        ds = PackedTokenDataset(cfg["token_cache_path"], cfg["seq_len"],
                                rank, world, seed)
        src = f"cache memmap ({human(ds.n_tokens)} tokens)"
    else:
        ds = StreamingDataset(tok, cfg["seq_len"], rank, world, seed,
                              cfg["shuffle_buffer"])
        src = "streaming HF (corre --build-cache, es mucho mas rapido)"
    nw = cfg["num_workers"]
    kw = dict(batch_size=cfg["batch_size"], num_workers=nw, pin_memory=True,
              drop_last=True)
    if nw > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw), src


def train_worker(rank: int, world: int, cfg: dict):
    is_main = rank == 0
    if world > 1:
        ddp_setup(rank, world)
    set_seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if not is_main:
        log.setLevel(logging.WARNING)

    tok = KairosTokenizer.load(cfg["tokenizer_path"])
    V = ((tok.vocab_size + 63) // 64) * 64
    model = AetherEngine(V, cfg).to(dev)
    if is_main:
        nat = sum(1 for k in model.kinds if k == "attn")
        log.info(f"AETHER v4.0 | {human(model.count_parameters())} params | "
                 f"{model.n_layers} capas ({model.n_layers-nat} delta + {nat} attn)"
                 f" | d={cfg['hidden_dim']} h={cfg['num_heads']} | vocab {V}")
        log.info(f"Estado persistente por sesion: "
                 f"{model.state_bytes(1)/1e6:.2f} MB (FIJO, no crece)")

    raw = model
    if world > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False,
                    gradient_as_bucket_view=True, find_unused_parameters=False)

    opts = build_optimizers(raw, cfg)
    scaler = make_grad_scaler(enabled=cfg["dtype"] == "float16")
    ema = EMA(raw, cfg["ema_decay"]) if (is_main and cfg["ema_decay"]) else None

    ck_dir = Path(cfg["checkpoint_dir"])
    ck_dir.mkdir(parents=True, exist_ok=True)
    latest = ck_dir / "latest.pt"
    start_step, resumes = 0, 0
    if latest.exists():
        sd = torch.load(latest, map_location=dev, weights_only=False)
        raw.load_state_dict(sd["model"])
        for o, osd in zip(opts, sd.get("opts", [])):
            o.load_state_dict(osd)
        scaler.load_state_dict(sd["scaler"])
        if ema and sd.get("ema"):
            ema.load_state_dict(sd["ema"])
        start_step, resumes = sd["step"], sd.get("resumes", 0) + 1
        if is_main:
            log.info(f"Reanudando en el paso {start_step:,} (resume #{resumes})")

    loader, src = make_loader(cfg, tok, rank, world, cfg["seed"] + 1000 * resumes)
    if is_main:
        log.info(f"Datos: {src}")
    it = iter(loader)

    accum = cfg["grad_accum_steps"]
    tps = cfg["batch_size"] * cfg["seq_len"] * accum * world
    if is_main:
        log.info(f"Batch efectivo: {tps:,} tokens/paso | "
                 f"total {human(tps * cfg['max_steps'])} tokens")

    t0 = time.time()
    win_t, win_tok, win_loss, win_mtp = time.time(), 0, 0.0, 0.0
    gnorm = torch.zeros(())
    model.train()

    for step in range(start_step, cfg["max_steps"]):
        lr = lr_at(step, cfg)
        for o in opts:
            mult = 30.0 if isinstance(o, Muon) else 1.0
            for g in o.param_groups:
                g["lr"] = lr * mult

        acc_l, acc_m = 0.0, 0.0
        for micro in range(accum):
            batch = next(it).to(dev, non_blocking=True)   # [B, T+2]
            x = batch[:, :-2]
            y1 = batch[:, 1:-1]
            y2 = batch[:, 2:]
            sync = (contextlib.nullcontext()
                    if (micro == accum - 1 or world == 1) else model.no_sync())
            with sync:
                with amp_autocast(enabled=cfg["dtype"] == "float16"):
                    out = model(x, return_mtp=cfg["mtp_weight"] > 0)
                if cfg["mtp_weight"] > 0:
                    lg1, lg2 = out
                    l1 = lm_loss(lg1, y1, cfg["z_loss"])
                    l2 = lm_loss(lg2, y2, 0.0)
                    loss = (l1 + cfg["mtp_weight"] * l2) / accum
                    acc_m += float(l2.detach())
                else:
                    l1 = lm_loss(out, y1, cfg["z_loss"])
                    loss = l1 / accum
                scaler.scale(loss).backward()
            acc_l += float(l1.detach())

        for o in opts:
            scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg["max_grad_norm"])
        for o in opts:
            scaler.step(o)
        scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if ema:
            ema.update(raw)

        win_loss += acc_l / accum
        win_mtp += acc_m / accum
        win_tok += tps

        if is_main and (step + 1) % cfg["log_every"] == 0:
            el = max(time.time() - win_t, 1e-6)
            avg = win_loss / cfg["log_every"]
            m2 = win_mtp / cfg["log_every"]
            log.info(f"paso {step+1:>6,}/{cfg['max_steps']:,} | loss {avg:.4f} | "
                     f"ppl {math.exp(min(avg,20)):>7.1f} | mtp {m2:.3f} | "
                     f"lr {lr:.2e} | gn {float(gnorm):.2f} | "
                     f"{win_tok/el/1000:.1f}K tok/s | "
                     f"vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB | "
                     f"{(time.time()-t0)/3600:.2f}h")
            win_t, win_tok, win_loss, win_mtp = time.time(), 0, 0.0, 0.0

        if is_main and (step + 1) % cfg["sample_every"] == 0:
            try:
                txt, _ = raw.generate(tok, "El futuro de la inteligencia artificial",
                                      max_tokens=80, temperature=0.85)
                log.info(f"MUESTRA >> {txt[:280]}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"muestra fallo: {e}")
            model.train()

        if is_main and (step + 1) % cfg["checkpoint_every"] == 0:
            save_ckpt(latest, raw, opts, scaler, ema, step + 1, cfg, resumes)
            log.info(f"checkpoint @ {step+1:,}")
            try:
                token = os.environ.get("HF_TOKEN")
                if token:
                    from huggingface_hub import HfApi
                    HfApi().upload_file(
                        path_or_fileobj=str(latest),
                        path_in_repo="v4/latest.pt",
                        repo_id="Bridoxd/AETHER-v2",
                        token=token
                    )
                    log.info(f"✓ Checkpoint @ {step+1:,} subido a HF v4/latest.pt!")
            except Exception as e:
                log.warning(f"Auto-subida de checkpoint a HF omitida: {e}")

        if (time.time() - t0) / 3600 > cfg["max_hours"]:
            if is_main:
                save_ckpt(latest, raw, opts, scaler, ema, step + 1, cfg, resumes)
                log.info(f"Limite de {cfg['max_hours']}h alcanzado, guardado en "
                         f"{step+1:,}. Relanza el script y sigue solo.")
            break

    if is_main:
        save_ckpt(latest, raw, opts, scaler, ema, cfg["max_steps"], cfg, resumes)
        log.info("Entrenamiento terminado.")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# X. S0 TUNING  (adaptacion con CERO overhead en inferencia)
# =============================================================================
def s0_tune(cfg: dict, text_path: str, steps: int = 300, lr: float = 0.05,
            out_name: str = "s0_custom.pt"):
    """
    Entrena UNICAMENTE el estado inicial S0 de cada capa recurrente.
    Todos los pesos congelados. El resultado son unos pocos cientos de miles
    de numeros que actuan como "personalidad" o "dominio" precargado.

    Contra LoRA: LoRA mete matrices extra que hay que multiplicar en cada
    forward. S0 no anade NI UNA operacion en inferencia: solo cambia con que
    valor arranca el estado. Es adaptacion literalmente gratis.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, step0 = load_for_inference(cfg, use_ema=False)
    model = model.to(dev).train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.s0.parameters():
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in model.s0.parameters())
    log.info(f"S0 tuning: {human(n_tr)} params entrenables "
             f"de {human(model.count_parameters())} totales")

    ids = tok.encode_with_specials(Path(text_path).read_text(encoding="utf-8"))
    T = cfg["seq_len"]
    if len(ids) < T + 2:
        ids = (ids * (1 + (T + 2) // max(1, len(ids))))
    arr = np.asarray(ids, dtype=np.int64)
    opt = torch.optim.AdamW(model.s0.parameters(), lr=lr, weight_decay=0.0)
    scaler = make_grad_scaler(dev == "cuda")
    rng = np.random.default_rng(0)

    for st in range(steps):
        i = int(rng.integers(0, max(1, len(arr) - T - 2)))
        b = torch.from_numpy(arr[i:i + T + 1]).unsqueeze(0).to(dev)
        with amp_autocast(enabled=dev == "cuda"):
            lg = model(b[:, :-1])
        loss = lm_loss(lg, b[:, 1:], 1e-4)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.s0.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if (st + 1) % 25 == 0:
            log.info(f"  s0 paso {st+1}/{steps} | loss {float(loss):.4f}")

    out = Path(cfg["checkpoint_dir"]) / out_name
    torch.save({k: v.detach().cpu() for k, v in model.s0.state_dict().items()}, out)
    log.info(f"S0 guardado en {out} ({out.stat().st_size/1e6:.2f} MB)")
    return out


# =============================================================================
# XI. APRENDIZAJE CONTINUO (LoRA en caliente + replay + EWC)
# =============================================================================
# El estado recurrente es memoria de TRABAJO: rapida, de tamano fijo, y se
# degrada con el tiempo por el decay. Para que algo pase a ser conocimiento
# PERMANENTE hay que moverlo a los pesos. Ahi aparece el olvido catastrofico:
# los pesos son memoria asociativa densa, escribir encima borra.
# Solucion: base congelada + LoRA + replay + EWC + consolidacion periodica.

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_((self.B @ self.A) * self.scaling)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B.zero_()


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                targets=("qkv", "o_proj", "w1", "w3")):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r, alpha))
                n += 1
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"LoRA r={r} en {n} capas ({human(tr)} params entrenables)")
    return model


class ReplayBuffer:
    """Reservoir sampling: muestra uniforme de TODO lo visto, con RAM fija."""

    def __init__(self, capacity: int = 20_000):
        self.cap, self.data, self.seen = capacity, [], 0

    def add(self, tokens: np.ndarray):
        self.seen += 1
        if len(self.data) < self.cap:
            self.data.append(tokens.astype(np.uint16))
        else:
            j = random.randint(0, self.seen - 1)
            if j < self.cap:
                self.data[j] = tokens.astype(np.uint16)

    def sample(self, n: int, device):
        if len(self.data) < n:
            return None
        rows = [self.data[random.randrange(len(self.data))] for _ in range(n)]
        arr = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        return arr[:, :-1], arr[:, 1:]


class ContinualLearner:
    """
    Defaults conservadores a proposito: lr 1e-5, replay 50%, EWC 0.1.
    Si subes el lr sin subir el replay, el modelo se borra a si mismo.
    Es asi de literal, no es una advertencia decorativa.
    """

    def __init__(self, model: AetherEngine, tok: KairosTokenizer, lr=1e-5, r=16,
                 replay_ratio=0.5, ewc_lambda=0.1, seq_len=512,
                 consolidate_every=500, device=None):
        self.dev = device or next(model.parameters()).device
        self.tok, self.seq_len = tok, seq_len
        self.model = inject_lora(model, r=r).to(self.dev)
        self.replay = ReplayBuffer()
        self.replay_ratio, self.ewc_lambda = replay_ratio, ewc_lambda
        self.consolidate_every = consolidate_every
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.scaler = make_grad_scaler(torch.cuda.is_available())
        self.fisher, self.anchor, self.updates, self.buf = {}, {}, 0, []

    @torch.no_grad()
    def _anchor(self):
        self.anchor = {n: p.detach().clone()
                       for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(self, n_batches=20, bs=2):
        """Fisher diagonal = importancia de cada peso. Sin esto EWC no sirve."""
        self.model.train()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        done = 0
        for _ in range(n_batches):
            b = self.replay.sample(bs, self.dev)
            if b is None:
                break
            self.model.zero_grad(set_to_none=True)
            with amp_autocast(enabled=torch.cuda.is_available()):
                lg = self.model(b[0])
            lm_loss(lg, b[1]).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().float().pow(2)
            done += 1
        if done:
            for n in fisher:
                fisher[n] /= done
            self.fisher = fisher
            self._anchor()
        self.model.zero_grad(set_to_none=True)

    def _ewc(self):
        if not self.fisher:
            return torch.zeros((), device=self.dev)
        tot = torch.zeros((), device=self.dev)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                tot = tot + (self.fisher[n] * (p - self.anchor[n]).pow(2)).sum()
        return tot

    def observe(self, text: str) -> Optional[float]:
        self.buf.append(BOS_ID)
        self.buf.extend(self.tok.encode(text))
        self.buf.append(EOS_ID)
        losses = []
        while len(self.buf) >= self.seq_len + 1:
            chunk = np.asarray(self.buf[:self.seq_len + 1], dtype=np.int64)
            self.buf = self.buf[self.seq_len:]
            self.replay.add(chunk)
            losses.append(self._update(chunk))
        return float(np.mean(losses)) if losses else None

    def _update(self, chunk: np.ndarray) -> float:
        self.model.train()
        t = torch.from_numpy(chunk).unsqueeze(0).to(self.dev)
        x, y = t[:, :-1], t[:, 1:]
        n_rep = max(1, int(round(self.replay_ratio / max(1e-6, 1 - self.replay_ratio))))
        rep = self.replay.sample(n_rep, self.dev)
        if rep is not None:
            x, y = torch.cat([x, rep[0]], 0), torch.cat([y, rep[1]], 0)
        with amp_autocast(enabled=torch.cuda.is_available()):
            lg = self.model(x)
        loss = lm_loss(lg, y, 1e-4)
        (self.scaler.scale(loss + self.ewc_lambda * self._ewc())).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 0.5)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.updates += 1
        if self.updates % self.consolidate_every == 0:
            self.consolidate()
        return float(loss.detach())

    def consolidate(self):
        """Funde LoRA en la base (RAM -> disco) y recalcula Fisher."""
        log.info(f"Consolidando en el update {self.updates}...")
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.merge()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.opt.param_groups[0]["lr"], betas=(0.9, 0.99), weight_decay=0.0)
        self.estimate_fisher()


# =============================================================================
# XII. EVALUACION
# =============================================================================
def load_for_inference(cfg: dict, use_ema: bool = True):
    tok = KairosTokenizer.load(cfg["tokenizer_path"])
    ck = Path(cfg["checkpoint_dir"]) / "latest.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    saved_cfg = {**cfg, **{k: v for k, v in sd.get("cfg", {}).items()
                           if k in ("hidden_dim", "num_layers", "num_heads",
                                    "layer_pattern", "chunk_size",
                                    "window_size", "ltm_dim", "use_ltm",
                                    "mtp_weight", "conv_kernel",
                                    "tie_embeddings", "rope_theta")}}
    V = ((tok.vocab_size + 63) // 64) * 64
    model = AetherEngine(V, saved_cfg)
    model.load_state_dict(sd["model"], strict=False)
    if use_ema and sd.get("ema"):
        model.load_state_dict(dict(sd["ema"]), strict=False)
        log.info("Pesos EMA cargados")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), tok, sd["step"]


@torch.no_grad()
def nll_of(model, tok, state, question: str, answer: str) -> float:
    """Log-verosimilitud negativa media (nats/token) de `answer` dado
    `question`, partiendo del estado dado. Es la metrica limpia para medir
    si la memoria sirve: mide informacion, no vibes."""
    dev = next(model.parameters()).device
    q = tok.encode_with_specials(question)
    a = tok.encode(answer)
    ids = torch.tensor([q + a], device=dev)
    st = [dict(s) if isinstance(s, dict) else s for s in state] if state else None
    logits, _ = model(ids, st, return_state=True)
    lg = logits[0, len(q) - 1:len(q) - 1 + len(a)].float()
    tgt = torch.tensor(a, device=dev)
    return float(F.cross_entropy(lg, tgt))


def eval_memory(cfg: dict, verbose: bool = True) -> dict:
    """
    EL EXPERIMENTO QUE IMPORTA.

    Le contamos un dato al modelo, guardamos la memoria a disco, la volvemos a
    cargar, y preguntamos. El dato NUNCA aparece en el prompt de la pregunta.
    Si NLL(con memoria) < NLL(sin memoria), la memoria carga informacion real.
    Y si NLL(recargada) == NLL(en RAM), la persistencia es exacta.

    Esta es la tabla que hay que publicar. Corre en 2 T4 sin despeinarse.
    """
    model, tok, step = load_for_inference(cfg)
    store = MemoryStore(cfg["memory_dir"])
    dev = next(model.parameters()).device

    fact = ("<MEM>Dato importante: el proyecto secreto de Angel se llama "
            "KAIROS y corre en dos tarjetas T4. La clave de acceso es 47831.")
    question = "<USR>Cual es la clave de acceso del proyecto?<AST>La clave es"
    answer = " 47831"

    nll_sin = nll_of(model, tok, model.init_state(1, dev), question, answer)
    st = model.ingest(tok, fact, model.init_state(1, dev))
    nll_con = nll_of(model, tok, st, question, answer)
    size = store.save("_eval_memoria", st, {"fact": fact})
    st2 = store.load("_eval_memoria", device=dev)
    nll_rec = nll_of(model, tok, st2, question, answer)

    res = dict(step=step, nll_sin_memoria=nll_sin, nll_con_memoria=nll_con,
               nll_recargada=nll_rec, ganancia_nats=nll_sin - nll_con,
               deriva_persistencia=abs(nll_con - nll_rec),
               bytes_memoria=size,
               bytes_estado_teorico=model.state_bytes(1))
    if verbose:
        log.info("=" * 68)
        log.info(f"  sin memoria      : {nll_sin:.4f} nats/token")
        log.info(f"  con memoria      : {nll_con:.4f} nats/token")
        log.info(f"  tras guardar+cargar: {nll_rec:.4f} nats/token")
        log.info(f"  GANANCIA         : {nll_sin - nll_con:+.4f} nats "
                 f"({'la memoria funciona' if nll_sin > nll_con else 'sin efecto aun'})")
        log.info(f"  deriva por persistir: {abs(nll_con-nll_rec):.2e} "
                 f"(fp16 en disco; <1e-2 es correcto)")
        log.info(f"  tamano en disco  : {size/1024:.1f} KB (FIJO por sesion)")
        log.info("=" * 68)
    return res


def eval_niah(cfg: dict, lengths=(1024, 4096, 16384), depths=(0.1, 0.5, 0.9),
              trials: int = 3) -> dict:
    """
    Needle in a Haystack en memoria CONSTANTE.

    El truco: el pajar no entra en ninguna ventana de contexto, se INGIERE al
    estado en pedazos. La VRAM usada es la misma para 1K que para 1M tokens.
    Un Transformer con 16K de contexto necesita 16K de cache; aqui son bytes
    fijos. Por eso el eje interesante no es la exactitud sola, es exactitud
    contra memoria usada.
    """
    model, tok, step = load_for_inference(cfg)
    dev = next(model.parameters()).device
    filler = ("El cielo estaba despejado y la ciudad seguia su ritmo de siempre. "
              "La gente caminaba sin prisa entre los puestos del mercado. ")
    rng = random.Random(0)
    results = {}
    for L in lengths:
        for d in depths:
            ok = 0
            for t in range(trials):
                code = rng.randint(10000, 99999)
                needle = f" El codigo secreto numero {t} es {code}. "
                n_rep = max(1, L // max(1, len(tok.encode(filler))))
                pre = filler * max(1, int(n_rep * d))
                post = filler * max(1, int(n_rep * (1 - d)))
                st = model.init_state(1, dev)
                for piece in (pre, needle, post):
                    st = model.ingest(tok, piece, st, piece=512)
                q = f"<USR>Cual es el codigo secreto numero {t}?<AST>Es el"
                nll_true = nll_of(model, tok, st, q, f" {code}")
                distract = rng.randint(10000, 99999)
                nll_fake = nll_of(model, tok, st, q, f" {distract}")
                ok += int(nll_true < nll_fake)
            results[f"len{L}_depth{int(d*100)}"] = ok / trials
            log.info(f"  NIAH len={L:>6} prof={int(d*100):>3}% : "
                     f"{ok}/{trials} | VRAM estado = "
                     f"{model.state_bytes(1)/1e6:.2f} MB (constante)")
    return results


def chat(cfg: dict, session: str):
    """Chat con memoria persistente. Cierra el proceso y vuelve: sigue ahi."""
    model, tok, step = load_for_inference(cfg)
    store = MemoryStore(cfg["memory_dir"])
    dev = next(model.parameters()).device
    st = store.load(session, device=dev)
    if st is None:
        st = model.init_state(1, dev)
        print(f"[memoria nueva para '{session}']")
    else:
        print(f"[memoria cargada: {store.path(session).stat().st_size/1024:.1f} KB]")
    print("escribe 'salir' para terminar (la memoria se guarda sola)\n")
    while True:
        try:
            u = input("tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("salir", "exit", "quit"):
            break
        if not u:
            continue
        txt, st = model.generate(tok, f"<USR>{u}<AST>", st, max_tokens=200,
                                 temperature=0.8, top_p=0.95,
                                 repetition_penalty=1.12)
        print(f"kairos > {txt}\n")
        store.save(session, st)
    n = store.save(session, st)
    print(f"[memoria guardada: {n/1024:.1f} KB]")


# =============================================================================
# XIII. SELF-TEST  (corre esto ANTES de gastar horas de GPU)
# =============================================================================
class _FakeTok:
    """Tokenizador de juguete para los tests, sin dependencias externas."""
    vocab_size = 128

    def encode(self, t, add_special=False):
        ids = [8 + (ord(c) % 100) for c in t]
        return [BOS_ID] + ids + [EOS_ID] if add_special else ids

    def encode_with_specials(self, t):
        return self.encode(t)

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _tiny_cfg(**over):
    c = dict(CFG)
    c.update(hidden_dim=64, num_layers=4, num_heads=4, chunk_size=8,
             window_size=16, ltm_dim=32, seq_len=32, use_ltm=True,
             mtp_weight=0.3, grad_checkpoint=False)
    c.update(over)
    return c


def self_test():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"  [{'OK ' if cond else 'FALLA'}] {name} {detail}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"\ndispositivo: {dev} | torch {torch.__version__}\n")

    # -- 1: la regla delta chunkwise es EXACTA -------------------------------
    print("1) gated delta rule: chunkwise vs recurrencia token a token (fp64)")
    B, H, T, Dk, Dv, C = 2, 3, 40, 16, 16, 8
    q = F.normalize(torch.randn(B, H, T, Dk, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(B, H, T, Dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, Dv, dtype=torch.float64)
    beta = 2 * torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    la = (-F.softplus(torch.randn(B, H, T, dtype=torch.float64))
          ).clamp(min=-MAX_CHUNK_LOG_DECAY / T)   # evita tocar el clamp interno
    O1, S1 = chunk_gated_delta_rule(q, k, v, beta, la, None, C)
    S = torch.zeros(B, H, Dv, Dk, dtype=torch.float64)
    outs = []
    for t in range(T):
        o, S = recurrent_gated_delta_step(q[:, :, t], k[:, :, t], v[:, :, t],
                                          beta[:, :, t], torch.exp(la[:, :, t]), S)
        outs.append(o)
    O2 = torch.stack(outs, dim=2)
    e = max((O1 - O2).abs().max().item(), (S1 - S).abs().max().item())
    chk("salida y estado coinciden", e < 1e-10, f"err={e:.2e}")

    # invariancia al tamano de chunk
    O3, S3 = chunk_gated_delta_rule(q, k, v, beta, la, None, 20)
    e2 = max((O1 - O3).abs().max().item(), (S1 - S3).abs().max().item())
    chk("invariante al tamano de chunk", e2 < 1e-10, f"err={e2:.2e}")

    # -- 2: autovalores negativos --------------------------------------------
    print("2) beta=2 produce reflexion de Householder (autovalor -1)")
    kk = F.normalize(torch.randn(32, dtype=torch.float64), dim=0)
    P = torch.eye(32, dtype=torch.float64) - 2 * torch.outer(kk, kk)
    ev = torch.linalg.eigvals(P).real
    chk("autovalor minimo = -1", abs(ev.min().item() + 1) < 1e-9,
        f"min={ev.min().item():.6f}")
    chk("la transicion es ortogonal (no explota ni se apaga)",
        torch.allclose(P @ P.T, torch.eye(32, dtype=torch.float64)))

    # -- 3: conv causal con cache --------------------------------------------
    print("3) ShortConv: un jalon vs por pedazos")
    sc = ShortConv(16, 4).double()
    x = torch.randn(1, 20, 16, dtype=torch.float64)
    with torch.no_grad():
        y_full, _ = sc(x)
        y1, c1 = sc(x[:, :7])
        y2, c2 = sc(x[:, 7:13], c1)
        y3, _ = sc(x[:, 13:], c2)
        y_split = torch.cat([y1, y2, y3], dim=1)
    e = (y_full - y_split).abs().max().item()
    chk("identico", e < 1e-12, f"err={e:.2e}")

    # -- 4: ventana deslizante ------------------------------------------------
    print("4) la atencion respeta la ventana")
    swa = SlidingWindowAttentionLayer(32, 4, window=4).eval()
    xa = torch.randn(1, 12, 32)
    xb = xa.clone()
    xb[:, 0] += 10.0                     # cambio un token muy viejo
    with torch.no_grad():
        oa, ob = swa(xa), swa(xb)
    far = (oa[:, -1] - ob[:, -1]).abs().max().item()
    near = (oa[:, 0] - ob[:, 0]).abs().max().item()
    chk("token fuera de ventana no influye", far < 1e-6, f"delta={far:.2e}")
    chk("token dentro de ventana si influye", near > 1e-3, f"delta={near:.2e}")

    # -- 5: modelo completo, forward paralelo vs paso a paso ------------------
    print("5) modelo completo: forward paralelo == inferencia token a token")
    cfg = _tiny_cfg()
    m = AetherEngine(128, cfg).eval().to(dev)
    ids = torch.randint(0, 128, (1, 25), device=dev)
    with torch.no_grad():
        lg_par, st_par = m(ids, m.init_state(1, dev), return_state=True)
        st = m.init_state(1, dev)
        seq = []
        for t in range(ids.shape[1]):
            lg, st = m.step(ids[:, t:t + 1], st)
            seq.append(lg)
        lg_seq = torch.cat(seq, dim=1)
    e = (lg_par - lg_seq).abs().max().item()
    chk("logits identicos", e < 2e-3, f"err={e:.2e}")

    # -- 6: ingerir por pedazos == de un jalon --------------------------------
    print("6) ingerir texto partido == ingerirlo completo")
    m32 = AetherEngine(128, cfg).eval().to(dev)
    tk = _FakeTok()
    txt = "la memoria persistente es el objetivo de este proyecto " * 6
    with torch.no_grad():
        s_full = m32.ingest(tk, txt, m32.init_state(1, dev), piece=4096)
        half = len(txt) // 3
        s_part = m32.init_state(1, dev)
        for pc in (txt[:half], txt[half:2 * half], txt[2 * half:]):
            s_part = m32.ingest(tk, pc, s_part, piece=7)
    diffs = []
    for a, b in zip(s_full, s_part):
        if isinstance(a, dict):
            for kk in a:
                if torch.is_tensor(a[kk]) and torch.is_tensor(b.get(kk)):
                    if a[kk].shape == b[kk].shape:
                        diffs.append((a[kk] - b[kk]).abs().max().item())
    e = max(diffs) if diffs else 1.0
    chk("estado final identico", e < 1e-3, f"err={e:.2e}")

    # -- 7: la memoria NO crece con el texto ----------------------------------
    print("7) el estado es de tamano FIJO (esta es la tesis del proyecto)")
    store = MemoryStore("/tmp/_kmem_test")
    with torch.no_grad():
        s_short = m32.ingest(tk, "hola" * 10, m32.init_state(1, dev))
        s_long = m32.ingest(tk, "hola" * 2000, m32.init_state(1, dev))
    n1 = store.save("corto", s_short)
    n2 = store.save("largo", s_long)
    chk("40 tokens vs 8.000 tokens ocupan lo mismo", abs(n1 - n2) < 256,
        f"{n1/1024:.1f} KB vs {n2/1024:.1f} KB")

    # -- 8: persistencia exacta ------------------------------------------------
    print("8) guardar y cargar la memoria no la corrompe")
    s_back = store.load("largo", device=dev)
    d = []
    for a, b in zip(s_long, s_back):
        if isinstance(a, dict):
            for kk in a:
                if torch.is_tensor(a[kk]) and a[kk].numel():
                    d.append((a[kk].float() - b[kk]).abs().max().item())
    e = max(d) if d else 1.0
    chk("roundtrip fp16 dentro de tolerancia", e < 1e-1, f"err={e:.2e}")

    # -- 9: forward + backward + AMP + multi-token -----------------------------
    print("9) entrenamiento: forward, MTP, backward y AMP")
    mt = AetherEngine(128, cfg).to(dev).train()
    b = torch.randint(0, 128, (2, 34), device=dev)
    sc2 = make_grad_scaler(dev == "cuda")
    with amp_autocast(enabled=dev == "cuda"):
        lg1, lg2 = mt(b[:, :-2], return_mtp=True)
    loss = lm_loss(lg1, b[:, 1:-1], 1e-4) + 0.3 * lm_loss(lg2, b[:, 2:])
    sc2.scale(loss).backward()
    ngrad = sum(1 for p in mt.parameters() if p.grad is not None)
    ntot = sum(1 for p in mt.parameters() if p.requires_grad)
    chk("loss finita", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    chk("gradiente llega a todos los parametros", ngrad == ntot,
        f"{ngrad}/{ntot}")
    chk("S0 recibe gradiente",
        all(p.grad is not None and p.grad.abs().sum() > 0
            for p in mt.s0.parameters()))

    # -- 10: S0 tuning aisla los gradientes -------------------------------------
    print("10) modo S0 tuning: solo el estado inicial es entrenable")
    ms = AetherEngine(128, cfg).to(dev).train()
    for p in ms.parameters():
        p.requires_grad_(False)
    for p in ms.s0.parameters():
        p.requires_grad_(True)
    lm_loss(ms(b[:, :-2]), b[:, 1:-1]).backward()
    leak = sum(1 for n, p in ms.named_parameters()
               if not n.startswith("s0.") and p.grad is not None)
    tr = sum(p.numel() for p in ms.s0.parameters())
    tot = sum(p.numel() for p in ms.parameters())
    chk("ningun peso base se mueve", leak == 0, f"fugas={leak}")
    chk("superficie entrenable minima", tr < tot * 0.05,
        f"{human(tr)} de {human(tot)} = {100*tr/tot:.2f}%")

    # -- 11: contabilidad de memoria -------------------------------------------
    print("11) presupuesto de memoria del modelo grande")
    big = AetherEngine(16000, CFG)
    nat = sum(1 for kk in big.kinds if kk == "attn")
    print(f"       parametros        : {human(big.count_parameters())}")
    print(f"       capas             : {big.n_layers - nat} delta + {nat} atencion")
    print(f"       estado por sesion : {big.state_bytes(1)/1e6:.2f} MB (constante)")
    print(f"       S0 entrenable     : "
          f"{human(sum(p.numel() for p in big.s0.parameters()))}")
    chk("el estado cabe de sobra en disco y en VRAM",
        big.state_bytes(1) < 100e6)

    print("\n" + ("TODOS LOS TESTS PASARON" if ok_all else
                  "HAY TESTS EN ROJO, no entrenes todavia"))
    return 0 if ok_all else 1


# =============================================================================
# XIV. CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser("KAIROS / AETHER v4.0")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--sample", type=str, default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--session", type=str, default="default")
    ap.add_argument("--eval-memory", action="store_true")
    ap.add_argument("--eval-niah", action="store_true")
    ap.add_argument("--s0-tune", type=str, default=None, metavar="ARCHIVO.txt")
    ap.add_argument("--continual", type=str, default=None, metavar="ARCHIVO.txt")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-ltm", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--no-attn", action="store_true",
                    help="ablacion: 100%% recurrente, sin atencion")
    ap.add_argument("--ckpt-grad", action="store_true",
                    help="gradient checkpointing: mas batch, ~25%% mas lento")
    ap.add_argument("--small", action="store_true",
                    help="config chica (512d, 6 capas) si te quedas sin VRAM")
    a = ap.parse_args()

    cfg = dict(CFG)
    if a.small:
        cfg.update(hidden_dim=512, num_layers=6, num_heads=4, batch_size=12)
    for key, val in (("optimizer", a.optimizer), ("max_steps", a.steps),
                     ("batch_size", a.batch_size), ("seq_len", a.seq_len),
                     ("hidden_dim", a.hidden), ("num_layers", a.layers)):
        if val:
            cfg[key] = val
    if a.stream:
        cfg["data_mode"] = "stream"
    if a.no_ltm:
        cfg["use_ltm"] = False
    if a.no_mtp:
        cfg["mtp_weight"] = 0.0
    if a.no_attn:
        cfg["layer_pattern"] = "1:0"
    if a.ckpt_grad:
        cfg["grad_checkpoint"] = True

    if a.self_test:
        raise SystemExit(self_test())
    if a.eval_memory:
        eval_memory(cfg)
        return
    if a.eval_niah:
        eval_niah(cfg)
        return
    if a.chat:
        chat(cfg, a.session)
        return
    if a.s0_tune:
        s0_tune(cfg, a.s0_tune)
        return
    if a.sample:
        model, tok, step = load_for_inference(cfg)
        print(f"\n[checkpoint paso {step:,}]\n")
        txt, _ = model.generate(tok, f"<USR>{a.sample}<AST>", None, 250,
                                0.85, 0.95, 0, 1.12)
        print(txt)
        return
    if a.continual:
        model, tok, step = load_for_inference(cfg, use_ema=False)
        learner = ContinualLearner(model, tok, seq_len=cfg["seq_len"])
        text = Path(a.continual).read_text(encoding="utf-8")
        log.info(f"loss online: {learner.observe(text)}")
        learner.consolidate()
        return

    tok = build_tokenizer(cfg)
    if a.build_cache:
        build_token_cache(tok, cfg)
        return
    if cfg["data_mode"] == "cache" and not Path(cfg["token_cache_path"]).exists():
        build_token_cache(tok, cfg)

    world = a.gpus if a.gpus else max(1, torch.cuda.device_count())
    log.info(f"Lanzando con {world} GPU(s)")
    if world > 1:
        mp.spawn(train_worker, args=(world, cfg), nprocs=world, join=True)
    else:
        train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
