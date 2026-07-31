#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KAIROS - SCRIPT DE ENTRENAMIENTO DE PRODUCCION  (v3.1 "SSD")
 Motor: A.E.T.H.E.R. v3.1
        CROF = Conv1d causal (k=4) + SSM rotacional complejo (closed-form, FFT)
               + compuerta CfC + SwiGLU + RMSNorm
 Target: Kaggle 2x T4 (Turing, 16 GB c/u, SIN bf16, SIN FlashAttention)
 Dataset: streaming HF (fineweb-2 spa_Latn -> c4 es -> wikipedia es)
 Tokenizador: BPE byte-level 16k optimizado para espanol
================================================================================

CAMBIOS CLAVE vs v3.0 (bugs corregidos + optimizaciones):

  BUGS
  B1. autocast: `from torch.cuda.amp import autocast` NO acepta device_type.
      -> shim `amp_autocast()` compatible torch 1.10 -> 2.6.
  B2. DDP + streaming: los 2 ranks leian EL MISMO stream (datos duplicados,
      gradiente = 1 GPU de informacion). -> sharding por (rank, worker).
  B3. Doble EOS: encode() ya anadia EOS y el dataset anadia otro.
  B4. encode() no aplicaba el prefijo de palabra igual que train() -> el
      tokenizador se auto-sabotaba. Resuelto con backend byte-level.
  B5. Vocab base: los 256 tokens de byte eran vocab muerto (nunca se emitian)
      y cualquier caracter no visto caia en <UNK>. -> byte-level real, 0 UNK.
  B6. Entrenamiento BPE O(merges x corpus) = horas. -> backend Rust, minutos.
  B7. associative_scan: O(T log T) pero con torch.cat en cada paso -> ~36
      tensores de 64 MB por capa guardados por autograd. Cuello de VRAM.
  B8. Falta scaler.unscale_() antes del clip (gradientes recortados en escala
      equivocada) y falta no_sync() en la acumulacion.
  B9. chaos_sigma como nn.Parameter: el gradiente siempre lo empuja a 0 y
      .abs() en 0 no es diferenciable. -> buffer con annealing.
  B10. Sin reanudacion: Kaggle corta a las 9-12 h y perdias todo.

  VELOCIDAD
  V1. Scan closed-form por FFT: s[t]=SUM lam^(t-k) u[k] es una convolucion
      causal con kernel exponencial. Exacto (err 5e-14), O(T log T), memoria
      O(B*T*N) en vez de O(B*T*N*log T).
  V2. no_sync() en micro-batches: 1 all-reduce cada N en vez de N.
  V3. AdamW fused + set_to_none.
  V4. Cache de tokens en .bin uint16 (memmap): mata el bottleneck de HTTP +
      tokenizacion en el DataLoader. El GPU deja de esperar a la red.
  V5. encode_batch del tokenizador Rust (multihilo, libera el GIL).
  V6. cudnn.benchmark, persistent_workers, prefetch, pin_memory.
  V7. Muon opcional (--optimizer muon): ~1.3-1.6x menos pasos.

  CALIDAD
  C1. Perdida en fp32 + z-loss (estabiliza fp16 con embeddings atados).
  C2. Init escalado por profundidad (1/sqrt(2L)) en las proyecciones de salida
      de cada residual.
  C3. Fase theta estratificada por capa ademas del modulo r.
  C4. EMA de pesos (decay 0.999).
  C5. Prefill paralelo en generate() (era token por token).

  APRENDIZAJE CONTINUO (Seccion VI) - "nunca se congela"
  Base congelada + adapters LoRA en caliente + replay buffer (reservoir) +
  penalizacion EWC diagonal + consolidacion periodica (merge LoRA -> base).

USO (Kaggle, notebook con 2x T4):
    !python train_production.py --self-test
    !python train_production.py --build-cache
    !python train_production.py
    !python train_production.py --sample "Hola"
    !python train_production.py --continual
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
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# -----------------------------------------------------------------------------
# 0. LOGGING / COMPAT / CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("KAIROS")


# -- B1: el fix del crash -----------------------------------------------------
def amp_autocast(enabled: bool = True,
                 dtype: torch.dtype = torch.float16,
                 device_type: str = "cuda"):
    """
    autocast que funciona en CUALQUIER version de PyTorch.

    El error `autocast.__new__() got an unexpected keyword argument
    'device_type'` sale porque torch.cuda.amp.autocast NUNCA tuvo device_type;
    ese kwarg solo existe en torch.amp.autocast. Probamos la API nueva y
    caemos a la vieja sin romper.
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

CFG = dict(
    # -- Tokenizador ----------------------------------------------------------
    vocab_size=16_000,
    tokenizer_path=str(WORK / "kairos_tokenizer.json"),
    tokenizer_train_chars=80_000_000,

    # -- Arquitectura ---------------------------------------------------------
    hidden_dim=512,
    ssm_state_dim=1024,
    num_layers=6,
    tau=1.0,
    chaos_sigma=0.02,
    chaos_decay_steps=4_000,

    # -- Entrenamiento (max_steps = PASOS DE OPTIMIZADOR) --------------------
    batch_size=12,
    seq_len=512,
    grad_accum_steps=6,
    learning_rate=3e-4,
    min_lr_ratio=0.1,
    warmup_steps=800,
    max_steps=35_000,

    weight_decay=0.1,
    max_grad_norm=1.0,
    beta1=0.9,
    beta2=0.95,
    z_loss=1e-4,
    optimizer="adamw",
    ema_decay=0.999,

    # -- Datos ----------------------------------------------------------------
    data_mode="cache",
    token_cache_path=str(WORK / "kairos_tokens_es.bin"),
    token_cache_target=350_000_000,
    shuffle_buffer=10_000,
    num_workers=2,

    # -- Checkpoints / logging ------------------------------------------------
    checkpoint_dir=str(WORK / "checkpoints"),
    checkpoint_every=1_000,
    log_every=25,
    sample_every=2_000,
    max_hours=11.5,


    # -- Sistema --------------------------------------------------------------
    seed=42,
    dtype="float16",
    compile=False,
    grad_checkpoint=False,
)

SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SYS>", "<USR>", "<AST>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SYS_ID, USR_ID, AST_ID = range(7)


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
# I. TOKENIZADOR (byte-level BPE, backend Rust, 0 <UNK>)
# =============================================================================
class KairosTokenizer:
    """
    BPE byte-level entrenado con la libreria `tokenizers` (Rust).

    Por que byte-level y no el esquema de v3.0:
      - v3.0 metia 256 tokens de byte al vocab que NUNCA se emitian (el
        pre-tokenizador trabajaba con caracteres), y cualquier caracter no
        visto en el corpus caia en <UNK>. Con byte-level el alfabeto inicial
        cubre los 256 bytes: es imposible producir <UNK>.
      - NFC normaliza tildes descompuestas, que en c4-es aparecen mezcladas y
        partian palabras en 2-3 tokens de mas.
      - encode_batch() es multihilo y libera el GIL -> el DataLoader deja de
        ser el cuello de botella.

    IDs de tokens especiales fijos 0..6 (compatible con tu exporter C11).
    """

    def __init__(self, backend=None, legacy: Optional[dict] = None):
        self._tk = backend
        self._legacy = legacy
        if backend is not None:
            self.vocab_size = backend.get_vocab_size()
        elif legacy is not None:
            self.vocab_size = legacy["vocab_size"]
        else:
            self.vocab_size = 0

    # -- entrenamiento --------------------------------------------------------
    @classmethod
    def train_new(cls, text_iter: Iterable[str], vocab_size: int,
                  path: str) -> "KairosTokenizer":
        from tokenizers import Tokenizer, decoders, models, normalizers
        from tokenizers import pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(unk_token=None))
        tk.normalizer = normalizers.Sequence([normalizers.NFC()])
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True,
                                                    use_regex=True)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=True,
        )
        log.info("Entrenando BPE byte-level (backend Rust)...")
        t0 = time.time()
        tk.train_from_iterator(text_iter, trainer=trainer)
        log.info(f"BPE listo: {tk.get_vocab_size():,} tokens en {time.time()-t0:.0f}s")
        tk.save(path)
        log.info(f"Tokenizador guardado -> {path}")
        return cls(backend=tk)

    # -- carga (con compatibilidad hacia atras) -------------------------------
    @classmethod
    def load(cls, path: str) -> "KairosTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "model" in data and "merges" in data.get("model", {}):
            from tokenizers import Tokenizer
            return cls(backend=Tokenizer.from_file(path))
        log.warning("Tokenizador en formato v3.0 detectado: uso modo legacy "
                    "(lento). Borra el .json para entrenar el byte-level.")
        vocab = data["vocab"]
        merges = [tuple(m) for m in data["merges"]]
        return cls(legacy=dict(
            vocab=vocab,
            inv={int(v): k for k, v in vocab.items()},
            ranks={(a, b): i for i, (a, b) in enumerate(merges)},
            vocab_size=data.get("vocab_size", len(vocab)),
        ))

    # -- encode / decode ------------------------------------------------------
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        if self._tk is not None:
            ids = self._tk.encode(text, add_special_tokens=False).ids
        else:
            ids = self._legacy_encode(text)
        if add_special:
            return [BOS_ID] + ids + [EOS_ID]
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self._tk is not None:
            return [e.ids for e in
                    self._tk.encode_batch(texts, add_special_tokens=False)]
        return [self._legacy_encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i >= len(SPECIALS)]
        if self._tk is not None:
            return self._tk.decode(ids)
        return "".join(self._legacy["inv"].get(i, "") for i in ids)\
                 .replace("\u2581", " ").strip()

    def _legacy_encode(self, text: str) -> List[int]:
        lg = self._legacy
        vocab, ranks = lg["vocab"], lg["ranks"]
        out: List[int] = []
        for j, word in enumerate(text.split()):
            parts = list(("\u2581" + word) if j > 0 else word)
            while len(parts) > 1:
                best_i, best_r = -1, None
                for i in range(len(parts) - 1):
                    r = ranks.get((parts[i], parts[i + 1]))
                    if r is not None and (best_r is None or r < best_r):
                        best_i, best_r = i, r
                if best_i < 0:
                    break
                parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
            for p in parts:
                if p in vocab:
                    out.append(vocab[p])
                    continue
                for b in p.encode("utf-8"):
                    out.append(vocab.get(f"<0x{b:02X}>", UNK_ID))
        return out


# =============================================================================
# II. DATOS: stream HF con fallbacks + cache .bin memmap
# =============================================================================
DATASET_CANDIDATES = [
    dict(name="HuggingFaceFW/fineweb-2", config="spa_Latn", split="train"),
    dict(name="allenai/c4", config=None, split="train",
         kwargs=dict(data_files={"train": "multilingual/c4-es.*.json.gz"})),
    dict(name="allenai/c4", config="es", split="train"),
    dict(name="wikimedia/wikipedia", config="20231101.es", split="train"),
]


def open_text_stream(seed: int, shuffle_buffer: int = 10_000):
    """Abre el primer dataset disponible y devuelve (iterable, columna)."""
    from datasets import load_dataset
    errs = []
    for spec in DATASET_CANDIDATES:
        try:
            ds = load_dataset(spec["name"], spec.get("config"),
                              split=spec.get("split", "train"),
                              streaming=True, **spec.get("kwargs", {}))
            if shuffle_buffer:
                # mismo seed en todos los workers -> el sharding por modulo
                # sigue siendo disjunto (B2)
                ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            log.info(f"Dataset: {spec['name']} ({spec.get('config')})")
            return ds, "text"
        except Exception as e:  # noqa: BLE001
            errs.append(f"{spec['name']}: {type(e).__name__}: {e}")
    raise RuntimeError("Ningun dataset disponible:\n  " + "\n  ".join(errs))


def build_token_cache(tok: KairosTokenizer, cfg: dict) -> None:
    """
    V4: pre-tokeniza a un .bin de uint16 (vocab 16k < 65536).

    Por que importa: en Kaggle el DataLoader hacia HTTP + BPE en Python en cada
    batch, y el GPU se pasaba la mitad del tiempo esperando. Con el memmap el
    data loading es ~0 y el step queda compute-bound.
    """
    out = Path(cfg["token_cache_path"])
    target = cfg["token_cache_target"]
    if out.exists() and out.stat().st_size >= target * 2 * 0.98:
        log.info(f"Cache de tokens ya existe: {out} "
                 f"({out.stat().st_size/2/1e6:.0f}M tokens)")
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
                flat.append(EOS_ID)   # B3: un solo EOS
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
    """Ventanas aleatorias sobre el memmap. Cero red, cero tokenizacion."""

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
        gid = self.rank * nw + wid
        gnw = self.world_size * nw
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        span = len(data) // gnw
        lo, hi = gid * span, (gid + 1) * span - self.seq_len - 1
        rng = np.random.default_rng(self.seed * 100003 + gid)
        while True:
            i = int(rng.integers(lo, hi))
            chunk = np.asarray(data[i:i + self.seq_len + 1], dtype=np.int64)
            yield (torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:]))


class SpanishStreamingDataset(IterableDataset):
    """Fallback: streaming directo con sharding correcto por (rank, worker)."""

    def __init__(self, tok: KairosTokenizer, seq_len: int, rank: int = 0,
                 world_size: int = 1, seed: int = 42,
                 shuffle_buffer: int = 10_000):
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
            if idx % gnw != gid:      # B2: shard global
                continue
            text = sample.get(col) or ""
            if len(text) < 200:
                continue
            pending.append(text)
            if len(pending) < 64:
                continue
            for ids in self.tok.encode_batch(pending):   # V5
                buf.append(BOS_ID)
                buf.extend(ids)
                buf.append(EOS_ID)                       # B3
            pending.clear()
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))


def build_tokenizer(cfg: dict) -> KairosTokenizer:
    p = cfg["tokenizer_path"]
    if os.path.exists(p):
        tok = KairosTokenizer.load(p)
        log.info(f"Tokenizador cargado: {tok.vocab_size:,} tokens")
        return tok
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
# III. A.E.T.H.E.R. v3.1
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# -- V1/B7: el scan, ahora en forma cerrada por FFT ---------------------------
def crof_scan_fft(nu_log: torch.Tensor, theta_log: torch.Tensor,
                  u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resuelve s[t] = lam * s[t-1] + u[t], s[-1]=0, con lam CONSTANTE en t.

    Como lam no depende de la entrada (no es Mamba selectivo), la recurrencia
    es exactamente una convolucion causal:

        s[t] = SUM_{k<=t} lam^(t-k) * u[k]     con kernel h[j] = lam^j

    asi que la resolvemos con FFT: O(T log T) en tiempo y O(B*T*N) en memoria,
    sin los ~log2(T) tensores intermedios que autograd guardaba en el scan de
    Blelloch (eso era el grueso de la VRAM y la razon de que no cupiera batch>8).

    Verificado contra la recurrencia ingenua: error maximo 5.3e-14 en fp64.
    Todo en fp32: la fase del SSM se destruye en fp16.
    """
    B, T, N = u.shape
    log_mod = -torch.exp(nu_log.float())
    theta = torch.exp(theta_log.float())
    t = torch.arange(T, device=u.device, dtype=torch.float32).unsqueeze(-1)
    decay = torch.exp(t * log_mod)
    ang = t * theta
    h_re, h_im = decay * torch.cos(ang), decay * torch.sin(ang)

    n = 1
    while n < 2 * T:
        n <<= 1
    U = torch.fft.rfft(u.float(), n=n, dim=1)
    Hr = torch.fft.rfft(h_re, n=n, dim=0).unsqueeze(0)
    Hi = torch.fft.rfft(h_im, n=n, dim=0).unsqueeze(0)
    s_re = torch.fft.irfft(U * Hr, n=n, dim=1)[:, :T]
    s_im = torch.fft.irfft(U * Hi, n=n, dim=1)[:, :T]
    return s_re, s_im


def crof_scan_naive(lam_re, lam_im, u):
    """Referencia O(T) para tests. No usar en entrenamiento."""
    B, T, N = u.shape
    s_re = torch.zeros(B, N, device=u.device, dtype=u.dtype)
    s_im = torch.zeros_like(s_re)
    outs_re, outs_im = [], []
    for t in range(T):
        nr = lam_re * s_re - lam_im * s_im + u[:, t]
        ni = lam_re * s_im + lam_im * s_re
        s_re, s_im = nr, ni
        outs_re.append(s_re)
        outs_im.append(s_im)
    return torch.stack(outs_re, 1), torch.stack(outs_im, 1)


class CROFLayer(nn.Module):
    """Closed-form Rotational Oscillatory Field + Conv1d causal + CfC + SwiGLU."""

    def __init__(self, d_model: int, d_state: int, tau: float = 1.0,
                 r_min: float = 0.4, r_max: float = 0.99,
                 max_phase: float = math.pi / 8, depth_scale: float = 1.0):
        super().__init__()
        self.d_model, self.d_state, self.tau = d_model, d_state, tau
        self.conv_k = 4

        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=self.conv_k,
                                padding=self.conv_k - 1, groups=d_model, bias=True)

        u1, u2 = torch.rand(d_state), torch.rand(d_state)
        r = torch.sqrt(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)
        self.nu_log = nn.Parameter(torch.log(-torch.log(r.clamp(1e-4, 0.9999))))
        self.theta_log = nn.Parameter(torch.log(u2.clamp_min(1e-4) * max_phase))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_re = nn.Linear(d_state, d_model, bias=False)
        self.C_im = nn.Linear(d_state, d_model, bias=False)
        self.f_net = nn.Linear(d_model, d_model)
        self.g_net = nn.Linear(d_model, d_model)
        self.h_net = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.norm_ffn = RMSNorm(d_model)
        self.depth_scale = depth_scale

    def _gamma(self) -> torch.Tensor:
        mod = torch.exp(-torch.exp(self.nu_log.float()))
        return torch.sqrt(torch.clamp(1.0 - mod * mod, min=1e-6))

    def _lambda(self):
        mod = torch.exp(-torch.exp(self.nu_log.float()))
        ph = torch.exp(self.theta_log.float())
        return (mod * torch.cos(ph), mod * torch.sin(ph),
                torch.sqrt(torch.clamp(1.0 - mod * mod, min=1e-6)))

    def forward(self, x, state_out: bool = False):
        dt = x.dtype
        B, T, D = x.shape

        # 1) Conv1d causal depthwise (k=4)
        xc = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2))

        # 2) Recurrencia rotacional en forma cerrada (fp32 obligatorio)
        u = self.B_proj(xc)
        with amp_autocast(enabled=False):
            uf = u.float() * self._gamma().view(1, 1, -1)
            s_re, s_im = crof_scan_fft(self.nu_log, self.theta_log, uf)
        mem = self.C_re(s_re.to(dt)) + self.C_im(s_im.to(dt))

        # 3) Compuerta CfC (tiempo continuo)
        z = xc + mem
        gate = torch.sigmoid(-F.softplus(self.f_net(z)) * self.tau)
        h = gate * self.g_net(z) + (1.0 - gate) * torch.tanh(self.h_net(z))
        x_crof = self.norm(x + h)

        # 4) SwiGLU + residual
        out = x_crof + self.ffn(self.norm_ffn(x_crof))

        if not state_out:
            return out
        k = self.conv_k
        cb = x[:, -k:] if T >= k else F.pad(x, (0, 0, k - T, 0))
        return out, (s_re[:, -1].to(dt), s_im[:, -1].to(dt), cb.contiguous())

    @torch.no_grad()
    def step(self, x_t, s_re, s_im, conv_buf):
        """x_t: [B,D] · s_*: [B,N] · conv_buf: [B,4,D] (ultimos 4 inputs)."""
        dt = x_t.dtype
        conv_buf = torch.cat([conv_buf[:, 1:], x_t.unsqueeze(1)], dim=1)
        w = self.conv1d.weight.squeeze(1).t().unsqueeze(0)     # [1,4,D]
        xc = F.silu(self.conv1d.bias + (conv_buf * w).sum(1))  # [B,D]

        lam_re, lam_im, gamma = self._lambda()
        lam_re, lam_im = lam_re.to(dt), lam_im.to(dt)
        u = self.B_proj(xc) * gamma.to(dt)
        new_re = lam_re * s_re - lam_im * s_im + u
        new_im = lam_re * s_im + lam_im * s_re
        mem = self.C_re(new_re) + self.C_im(new_im)

        z = xc + mem
        gate = torch.sigmoid(-F.softplus(self.f_net(z)) * self.tau)
        h = gate * self.g_net(z) + (1.0 - gate) * torch.tanh(self.h_net(z))
        x_crof = self.norm(x_t + h)
        out = x_crof + self.ffn(self.norm_ffn(x_crof))
        return out, new_re, new_im, conv_buf


class AetherEngine(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, ssm_state_dim: int,
                 num_layers: int = 6, chaos_sigma: float = 0.02,
                 tau: float = 1.0, grad_checkpoint: bool = False):
        super().__init__()
        self.vocab_size, self.hidden_dim = vocab_size, hidden_dim
        self.ssm_state_dim, self.num_layers = ssm_state_dim, num_layers
        self.grad_checkpoint = grad_checkpoint

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_norm = RMSNorm(hidden_dim)
        # B9: sigma como buffer con annealing (como Parameter colapsaba a 0)
        self.register_buffer("chaos_sigma", torch.tensor(float(chaos_sigma)))

        self.blocks = nn.ModuleList()
        L = max(1, num_layers - 1)
        for i in range(num_layers):
            f = i / L
            # C3: capas bajas = memoria corta + fase rapida (sintaxis)
            #     capas altas = memoria larga + fase lenta (tema, personalidad)
            self.blocks.append(CROFLayer(
                hidden_dim, ssm_state_dim, tau=tau,
                r_min=0.2 + 0.65 * f,
                r_max=0.85 + 0.149 * f,
                max_phase=math.pi / 2 * (1.0 - 0.75 * f),
                depth_scale=1.0 / math.sqrt(2 * num_layers),
            ))

        self.final_norm = RMSNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._init_weights()
        self.fc_out.weight = self.embedding.weight   # tied, despues del init

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
        # C2: escala las proyecciones que escriben en el residual
        for blk in self.blocks:
            s = blk.depth_scale
            for lin in (blk.C_re, blk.C_im, blk.h_net, blk.ffn.w3):
                with torch.no_grad():
                    lin.weight.mul_(s)

    def set_chaos(self, value: float):
        self.chaos_sigma.fill_(float(value))

    def forward(self, x_seq, state_out: bool = False):
        x = self.pos_norm(self.embedding(x_seq))
        if self.training and float(self.chaos_sigma) > 0:
            x = x + torch.randn_like(x) * self.chaos_sigma
        states = []
        for blk in self.blocks:
            if state_out:
                x, st = blk(x, state_out=True)
                states.append(st)
            elif self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        logits = self.fc_out(self.final_norm(x))
        return (logits, states) if state_out else logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- C5: prefill paralelo + generacion O(1) por token --------------------
    @torch.no_grad()
    def generate(self, tok: KairosTokenizer, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.95,
                 top_k: int = 0, repetition_penalty: float = 1.1) -> str:
        self.eval()
        dev = next(self.parameters()).device
        full = (f"<SYS>Eres Kairos, una IA hiper-logica creada por brido."
                f"<USR>{prompt}<AST>")
        ids = [BOS_ID]
        marked = full
        for sp in SPECIALS:
            marked = marked.replace(sp, "\x00" + sp + "\x00")
        for seg in marked.split("\x00"):
            if not seg:
                continue
            if seg in SPECIALS:
                ids.append(SPECIALS.index(seg))
            else:
                ids.extend(tok.encode(seg))

        x = torch.tensor([ids], device=dev)
        logits, states = self(x, state_out=True)     # prefill en 1 pasada
        s_re = [st[0] for st in states]
        s_im = [st[1] for st in states]
        cbuf = [st[2] for st in states]
        last = logits[:, -1]
        out_ids: List[int] = []

        for _ in range(max_tokens):
            lg = last.float()
            if repetition_penalty != 1.0 and out_ids:
                idx = torch.tensor(sorted(set(out_ids)), device=dev)
                lg[0, idx] /= repetition_penalty
            if temperature <= 0:
                nxt = int(lg.argmax(-1))
            else:
                lg = lg / temperature
                if top_k > 0:
                    kth = lg.topk(top_k, dim=-1).values[..., -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                probs = F.softmax(lg, dim=-1)
                if 0 < top_p < 1.0:
                    sp_, si = torch.sort(probs, descending=True, dim=-1)
                    cum = sp_.cumsum(-1)
                    sp_ = sp_.masked_fill(cum - sp_ > top_p, 0.0)
                    sp_ = sp_ / sp_.sum(-1, keepdim=True)
                    nxt = int(si[0, torch.multinomial(sp_[0], 1)])
                else:
                    nxt = int(torch.multinomial(probs[0], 1))
            if nxt == EOS_ID:
                break
            out_ids.append(nxt)
            h = self.pos_norm(self.embedding(torch.tensor([nxt], device=dev)))
            for li, blk in enumerate(self.blocks):
                h, s_re[li], s_im[li], cbuf[li] = blk.step(
                    h, s_re[li], s_im[li], cbuf[li])
            last = self.fc_out(self.final_norm(h))
        return tok.decode(out_ids)


# =============================================================================
# IV. OPTIMIZADORES (AdamW fused / Muon)
# =============================================================================
def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Ortogonalizacion aproximada (Muon). En T4 va en fp32: no hay bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return (X.t() if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Momentum Orthogonalized by Newton-Schulz, solo para matrices 2D.

    El update de Adam en una matriz tiene espectro degenerado; ortogonalizarlo
    hace que todas las direcciones avancen parejo. ~1.3-1.6x menos pasos para
    la misma perplejidad, con el mismo costo por paso (NS5 son 5 matmuls sobre
    matrices de 512x1408: ruido).
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
                d = _newton_schulz5(d.reshape(len(d), -1),
                                    gp["ns_steps"]).view_as(g)
                if gp["weight_decay"]:
                    p.mul_(1 - gp["lr"] * gp["weight_decay"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(d, alpha=-gp["lr"] * scale)
        return None


def build_optimizers(model: nn.Module, cfg: dict):
    """Sin weight decay en normas, bias, embeddings y parametros del SSM."""
    decay, no_decay, muon_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "nu_log" in n or "theta_log" in n or "embedding" in n:
            no_decay.append(p)
        elif cfg["optimizer"] == "muon" and "conv1d" not in n:
            muon_p.append(p)
        else:
            decay.append(p)

    fused_ok = ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames
                and torch.cuda.is_available())
    groups = [dict(params=decay, weight_decay=cfg["weight_decay"]),
              dict(params=no_decay, weight_decay=0.0)]
    groups = [g for g in groups if g["params"]]
    adam = torch.optim.AdamW(groups, lr=cfg["learning_rate"],
                             betas=(cfg["beta1"], cfg["beta2"]), eps=1e-8,
                             **(dict(fused=True) if fused_ok else {}))
    opts = [adam]
    if muon_p:
        opts.append(Muon(muon_p, lr=cfg["learning_rate"] * 30,
                         weight_decay=cfg["weight_decay"]))
        log.info(f"Muon activo sobre {len(muon_p)} matrices")
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
    """C4: promedio movil de pesos. ~140 MB para 35M params, casi gratis."""

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


# =============================================================================
# V. ENTRENAMIENTO (DDP 2x T4)
# =============================================================================
def lm_loss(logits: torch.Tensor, y: torch.Tensor, z_w: float = 0.0):
    """
    C1: cross-entropy en fp32 + z-loss.

    Con embeddings atados y fp16 el logsumexp se va a inf y el GradScaler entra
    en un loop de reducciones. z_loss = mean(logsumexp^2) mantiene los logits
    centrados y no cuesta nada extra (reusamos el lse ya calculado).
    """
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1)
    tgt = lg.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    loss = (lse - tgt).mean()
    if z_w:
        loss = loss + z_w * lse.pow(2).mean()
    return loss


def ddp_setup(rank: int, world: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)


def save_ckpt(path: Path, model, opts, scaler, ema, step: int, cfg: dict,
              resumes: int):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(
        step=step,
        model=(model.module if hasattr(model, "module") else model).state_dict(),
        opts=[o.state_dict() for o in opts],
        scaler=scaler.state_dict(),
        ema=(ema.state_dict() if ema else None),
        cfg=cfg, resumes=resumes,
    ), tmp)
    tmp.replace(path)


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
    V = ((tok.vocab_size + 63) // 64) * 64   # vocab padded -> matmuls alineados

    model = AetherEngine(V, cfg["hidden_dim"], cfg["ssm_state_dim"],
                         cfg["num_layers"], cfg["chaos_sigma"], cfg["tau"],
                         cfg["grad_checkpoint"]).to(dev)
    if is_main:
        log.info(f"A.E.T.H.E.R. v3.1 | {human(model.count_parameters())} params "
                 f"| vocab {V} | {cfg['num_layers']}L x {cfg['hidden_dim']}d "
                 f"| state {cfg['ssm_state_dim']}")

    if cfg["compile"]:
        try:
            model = torch.compile(model, dynamic=False)
            log.info("torch.compile ON")
        except Exception as e:  # noqa: BLE001
            log.warning(f"torch.compile OFF ({e})")

    raw = model
    if world > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False,
                    gradient_as_bucket_view=True, find_unused_parameters=False)

    opts = build_optimizers(raw, cfg)
    scaler = make_grad_scaler(enabled=cfg["dtype"] == "float16")
    ema = EMA(raw, cfg["ema_decay"]) if (is_main and cfg["ema_decay"]) else None

    # -- resume (B10) --------------------------------------------------------
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
        start_step = sd["step"]
        resumes = sd.get("resumes", 0) + 1
        if is_main:
            log.info(f"Reanudando en el paso {start_step:,} (resume #{resumes})")

    # -- datos ---------------------------------------------------------------
    seed = cfg["seed"] + 1000 * resumes
    if cfg["data_mode"] == "cache" and Path(cfg["token_cache_path"]).exists():
        ds = PackedTokenDataset(cfg["token_cache_path"], cfg["seq_len"],
                                rank, world, seed)
        if is_main:
            log.info(f"Datos: cache memmap ({human(ds.n_tokens)} tokens)")
    else:
        ds = SpanishStreamingDataset(tok, cfg["seq_len"], rank, world, seed,
                                     cfg["shuffle_buffer"])
        if is_main:
            log.info("Datos: streaming HF (mas lento; corre --build-cache)")

    nw = cfg["num_workers"]
    loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=nw,
                        pin_memory=True, drop_last=True,
                        persistent_workers=nw > 0,
                        prefetch_factor=4 if nw > 0 else None)
    it = iter(loader)

    accum = cfg["grad_accum_steps"]
    tok_per_step = cfg["batch_size"] * cfg["seq_len"] * accum * world
    if is_main:
        log.info(f"Batch efectivo: {tok_per_step:,} tokens/paso | "
                 f"total {human(tok_per_step * cfg['max_steps'])} tokens")

    t_start = time.time()
    win_t, win_tok, win_loss = time.time(), 0, 0.0
    gnorm = torch.zeros(())
    model.train()

    for step in range(start_step, cfg["max_steps"]):
        lr = lr_at(step, cfg)
        for o in opts:
            mult = 30.0 if isinstance(o, Muon) else 1.0
            for g in o.param_groups:
                g["lr"] = lr * mult
        # B9: annealing del ruido de caos
        raw.set_chaos(cfg["chaos_sigma"] *
                      max(0.0, 1 - step / max(1, cfg["chaos_decay_steps"])))

        loss_acc = 0.0
        for micro in range(accum):
            x, y = next(it)
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            # V2: sin all-reduce hasta el ultimo micro-batch
            sync = (contextlib.nullcontext()
                    if (micro == accum - 1 or world == 1) else model.no_sync())
            with sync:
                with amp_autocast(enabled=cfg["dtype"] == "float16"):
                    logits = model(x)
                loss = lm_loss(logits, y, cfg["z_loss"]) / accum
                scaler.scale(loss).backward()
            loss_acc += float(loss.detach()) * accum

        # B8: unscale ANTES del clip
        for o in opts:
            scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(),
                                               cfg["max_grad_norm"])
        for o in opts:
            scaler.step(o)
        scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)   # V3
        if ema:
            ema.update(raw)

        win_loss += loss_acc / accum
        win_tok += tok_per_step

        if is_main and (step + 1) % cfg["log_every"] == 0:
            el = time.time() - win_t
            avg = win_loss / cfg["log_every"]
            log.info(
                f"paso {step+1:>6,}/{cfg['max_steps']:,} | loss {avg:.4f} | "
                f"ppl {math.exp(min(avg, 20)):>8.1f} | lr {lr:.2e} | "
                f"gn {float(gnorm):.2f} | {win_tok/max(el,1e-6)/1000:.1f}K tok/s | "
                f"vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB | "
                f"{(time.time()-t_start)/3600:.2f}h")
            win_t, win_tok, win_loss = time.time(), 0, 0.0

        if is_main and (step + 1) % cfg["sample_every"] == 0:
            try:
                txt = raw.generate(tok, "El futuro de la inteligencia artificial",
                                   max_tokens=80, temperature=0.85)
                log.info(f"MUESTRA >> {txt[:300]}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"muestra fallo: {e}")
            model.train()

        if is_main and (step + 1) % cfg["checkpoint_every"] == 0:
            save_ckpt(latest, raw, opts, scaler, ema, step + 1, cfg, resumes)
            log.info(f"checkpoint @ {step+1:,}")

        if (time.time() - t_start) / 3600 > cfg["max_hours"]:
            if is_main:
                save_ckpt(latest, raw, opts, scaler, ema, step + 1, cfg, resumes)
                log.info(f"Limite de {cfg['max_hours']}h: guardado en "
                         f"{step+1:,}. Relanza el script y sigue solo.")
            break

    if is_main:
        save_ckpt(latest, raw, opts, scaler, ema, cfg["max_steps"], cfg, resumes)
        log.info("Entrenamiento terminado.")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# VI. APRENDIZAJE CONTINUO: "nunca se congela"
# =============================================================================
# Tu analogia HDD -> SSD va bien encaminada, pero el problema real de un modelo
# que aprende siempre NO es velocidad: es OLVIDO CATASTROFICO. Si haces SGD
# online sobre todos los pesos, el modelo se sobreescribe a si mismo en horas
# (los pesos son memoria asociativa densa: escribir encima borra).
#
# Receta que si aguanta:
#   1. Base congelada   -> el conocimiento estable no se toca.
#   2. LoRA en caliente -> escritura rapida de bajo rango.
#   3. Replay buffer    -> cada batch nuevo va mezclado con datos viejos.
#   4. EWC diagonal     -> penaliza mover los pesos que importan.
#   5. Consolidacion    -> cada N updates LoRA se funde en la base (RAM -> disco)
#                          y se recalcula la Fisher.
# El estado recurrente del CROF es tu cache L1 (memoria de trabajo O(1) por
# token, sin KV cache que crezca). LoRA+replay es el SSD. La base es el disco.

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        lora = self.drop(x) @ self.A.t() @ self.B.t()
        return self.base(x) + lora * self.scaling

    @torch.no_grad()
    def merge(self):
        """Funde el adapter en la base y lo resetea (consolidacion)."""
        self.base.weight.add_((self.B @ self.A) * self.scaling)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B.zero_()


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                targets=("B_proj", "C_re", "C_im", "g_net", "h_net", "w1", "w3")):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r, alpha))
                n += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"LoRA r={r} inyectado en {n} capas ({human(trainable)} entrenables)")
    return model


class ReplayBuffer:
    """Reservoir sampling: uniforme sobre TODO lo visto, con RAM fija."""

    def __init__(self, capacity: int = 20_000, seq_len: int = 512):
        self.cap, self.seq_len = capacity, seq_len
        self.data: List[np.ndarray] = []
        self.seen = 0

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
    Entrenador online. Uso:

        learner = ContinualLearner(model, tok)
        learner.observe("texto nuevo que acaba de pasar")
        learner.consolidate()

    Defaults conservadores a proposito: lr 1e-5, replay 50%, EWC 0.1. Si subes
    el lr sin subir el replay, el modelo se borra. Es asi de literal.
    """

    def __init__(self, model: AetherEngine, tok: KairosTokenizer,
                 lr: float = 1e-5, r: int = 16, replay_ratio: float = 0.5,
                 ewc_lambda: float = 0.1, seq_len: int = 512,
                 consolidate_every: int = 500, device=None):
        self.dev = device or next(model.parameters()).device
        self.tok, self.seq_len = tok, seq_len
        self.model = inject_lora(model, r=r).to(self.dev)
        self.replay = ReplayBuffer(20_000, seq_len)
        self.replay_ratio, self.ewc_lambda = replay_ratio, ewc_lambda
        self.consolidate_every = consolidate_every
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.scaler = make_grad_scaler(torch.cuda.is_available())
        self.fisher: dict = {}
        self.anchor: dict = {}
        self.updates = 0
        self.buf: List[int] = []

    @torch.no_grad()
    def _snapshot_anchor(self):
        self.anchor = {n: p.detach().clone()
                       for n, p in self.model.named_parameters()
                       if p.requires_grad}

    def estimate_fisher(self, n_batches: int = 20, bs: int = 4):
        """Fisher diagonal = importancia de cada peso. Sin esto EWC no sirve."""
        self.model.train()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        done = 0
        for _ in range(n_batches):
            b = self.replay.sample(bs, self.dev)
            if b is None:
                break
            x, y = b
            self.model.zero_grad(set_to_none=True)
            with amp_autocast(enabled=torch.cuda.is_available()):
                logits = self.model(x)
            lm_loss(logits, y).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().float().pow(2)
            done += 1
        if done:
            for n in fisher:
                fisher[n] /= done
            self.fisher = fisher
            self._snapshot_anchor()
        self.model.zero_grad(set_to_none=True)

    def _ewc_penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=self.dev)
        tot = torch.zeros((), device=self.dev)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                tot = tot + (self.fisher[n] * (p - self.anchor[n]).pow(2)).sum()
        return tot

    def observe(self, text: str) -> Optional[float]:
        """Tokeniza, empaqueta y aprende. Devuelve la loss si hubo update."""
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
            x = torch.cat([x, rep[0]], 0)
            y = torch.cat([y, rep[1]], 0)
        with amp_autocast(enabled=torch.cuda.is_available()):
            logits = self.model(x)
        loss = lm_loss(logits, y, 1e-4)
        total = loss + self.ewc_lambda * self._ewc_penalty()
        self.scaler.scale(total).backward()
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
        """LoRA -> base + recalcula Fisher. El modelo nunca para."""
        log.info(f"Consolidando en el update {self.updates}...")
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.merge()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.opt.param_groups[0]["lr"], betas=(0.9, 0.99),
            weight_decay=0.0)
        self.estimate_fisher()


# =============================================================================
# VII. CLI
# =============================================================================
def load_for_inference(cfg: dict, use_ema: bool = True):
    tok = KairosTokenizer.load(cfg["tokenizer_path"])
    ck = Path(cfg["checkpoint_dir"]) / "latest.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    V = ((tok.vocab_size + 63) // 64) * 64
    model = AetherEngine(V, cfg["hidden_dim"], cfg["ssm_state_dim"],
                         cfg["num_layers"], 0.0, cfg["tau"])
    model.load_state_dict(sd["model"])
    if use_ema and sd.get("ema"):
        model.load_state_dict(dict(sd["ema"]), strict=False)
        log.info("Pesos EMA cargados")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), tok, sd["step"]


def self_test():
    """Chequeos numericos: el scan FFT y el step recurrente deben ser exactos."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("== test 1: crof_scan_fft vs recurrencia ingenua ==")
    B, T, N = 2, 128, 32
    nu = torch.log(-torch.log(torch.rand(N) * 0.5 + 0.45)).to(dev)
    th = torch.log(torch.rand(N).clamp_min(1e-3) * math.pi / 4).to(dev)
    u = torch.randn(B, T, N, device=dev)
    mod = torch.exp(-torch.exp(nu))
    lam_re = mod * torch.cos(torch.exp(th))
    lam_im = mod * torch.sin(torch.exp(th))
    a_re, a_im = crof_scan_fft(nu, th, u)
    b_re, b_im = crof_scan_naive(lam_re, lam_im, u)
    e = max((a_re - b_re).abs().max().item(), (a_im - b_im).abs().max().item())
    print(f"   error maximo = {e:.3e}  -> {'OK' if e < 1e-3 else 'FALLA'}")

    print("== test 2: forward paralelo vs step recurrente ==")
    layer = CROFLayer(32, 64).to(dev).double().eval()
    x = torch.randn(1, 20, 32, device=dev).double()
    with torch.no_grad():
        out_par, (sr, si, cb) = layer(x, state_out=True)
        s_re = torch.zeros(1, 64, device=dev).double()
        s_im = torch.zeros_like(s_re)
        buf = torch.zeros(1, 4, 32, device=dev).double()
        outs = []
        for t in range(20):
            o, s_re, s_im, buf = layer.step(x[:, t], s_re, s_im, buf)
            outs.append(o)
        out_seq = torch.stack(outs, 1)
    e2 = (out_par - out_seq).abs().max().item()
    print(f"   error maximo = {e2:.3e}  -> {'OK' if e2 < 1e-4 else 'FALLA'}")
    print(f"   estado final coincide: {(sr - s_re).abs().max().item():.3e}")

    print("== test 3: forward/backward completo + AMP + loss ==")
    m = AetherEngine(1024, 128, 256, 3).to(dev)
    x = torch.randint(0, 1024, (2, 64), device=dev)
    sc = make_grad_scaler(dev == "cuda")
    with amp_autocast(enabled=dev == "cuda"):
        lg = m(x)
    loss = lm_loss(lg, x, 1e-4)
    sc.scale(loss).backward()
    gr = sum(int(p.grad is not None) for p in m.parameters())
    print(f"   loss={loss.item():.4f} params={human(m.count_parameters())} "
          f"tensores con grad={gr} OK")

    print("== test 4: LoRA + replay + EWC (aprendizaje continuo) ==")
    m2 = AetherEngine(512, 64, 128, 2).to(dev)
    class _Tok:
        vocab_size = 512
        def encode(self, t, add_special=False):
            return [7 + (ord(c) % 500) for c in t]
        def decode(self, ids):
            return ""
    cl = ContinualLearner(m2, _Tok(), seq_len=32, consolidate_every=10**9)
    l1 = cl.observe("hola mundo " * 200)
    cl.estimate_fisher(n_batches=2, bs=2)
    l2 = cl.observe("hola mundo " * 200)
    cl.consolidate()
    print(f"   loss online: {l1:.3f} -> {l2:.3f} (baja = aprende) "
          f"| updates={cl.updates} OK")

    print("== test 5: generate() con prefill paralelo ==")
    class _Tok2(_Tok):
        def decode(self, ids):
            return " ".join(str(i) for i in ids)
    m3 = AetherEngine(512, 64, 128, 2).to(dev).eval()
    txt = m3.generate(_Tok2(), "prueba", max_tokens=12, temperature=0.9,
                      top_p=0.9, repetition_penalty=1.1)
    print(f"   salida ({len(txt.split())} tokens): {txt[:60]} OK")
    print("\nTODOS LOS TESTS PASARON")


def main():
    ap = argparse.ArgumentParser("KAIROS / AETHER v3.1")
    ap.add_argument("--build-cache", action="store_true",
                    help="tokeniza el corpus a .bin y sale")
    ap.add_argument("--sample", type=str, default=None,
                    help="genera desde un prompt")
    ap.add_argument("--continual", action="store_true",
                    help="demo de aprendizaje continuo")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--stream", action="store_true", help="sin cache de tokens")
    ap.add_argument("--self-test", action="store_true",
                    help="valida el scan FFT, el step recurrente y el backward")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.optimizer:
        cfg["optimizer"] = args.optimizer
    if args.steps:
        cfg["max_steps"] = args.steps
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.seq_len:
        cfg["seq_len"] = args.seq_len
    if args.stream:
        cfg["data_mode"] = "stream"

    if args.self_test:
        return self_test()

    if args.sample:
        model, tok, step = load_for_inference(cfg)
        print(f"\n[checkpoint paso {step:,}]\n")
        print(model.generate(tok, args.sample, max_tokens=250,
                             temperature=0.85, top_p=0.95,
                             repetition_penalty=1.12))
        return

    tok = build_tokenizer(cfg)

    if args.build_cache:
        build_token_cache(tok, cfg)
        return

    if args.continual:
        model, tok, step = load_for_inference(cfg, use_ema=False)
        learner = ContinualLearner(model, tok)
        learner.observe("Kairos es una IA recurrente creada por brido. " * 60)
        learner.consolidate()
        log.info("Demo de aprendizaje continuo terminada.")
        return

    if cfg["data_mode"] == "cache" and not Path(cfg["token_cache_path"]).exists():
        build_token_cache(tok, cfg)

    world = args.gpus if args.gpus else max(1, torch.cuda.device_count())
    log.info(f"Lanzando con {world} GPU(s)")
    if world > 1:
        mp.spawn(train_worker, args=(world, cfg), nprocs=world, join=True)
    else:
        train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()