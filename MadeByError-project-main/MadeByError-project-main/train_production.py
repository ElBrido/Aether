"""
================================================================================
  KAIROS — SCRIPT DE ENTRENAMIENTO DE PRODUCCIÓN
  Motor: A.E.T.H.E.R. v3.0 (CROF: Conv1d causal + SSM rotacional complejo + CfC)
  Target: T4 x2 (30 GB VRAM) — Kaggle
  Dataset: allenai/c4 es streaming (Hugging Face)
  Tokenizador: BPE propio optimizado para español (16,000 tokens)
================================================================================
"""

import os
import sys
import time
import math
import json
import struct
import collections
import re
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("AETHER")

# ── Configuración Global ───────────────────────────────────────────────────────
CFG = {
    # Tokenizador
    "vocab_size": 16_000,
    "tokenizer_path": "kairos_tokenizer.json",
    "tokenizer_train_chars": 50_000_000,

    # Arquitectura
    "hidden_dim": 512,
    "ssm_state_dim": 1024,
    "chaos_sigma": 0.02,
    "num_layers": 6,

    # Entrenamiento — NOTA: max_steps y warmup_steps son PASOS GLOBALES
    # de optimizador (no micro-batches).
    "batch_size": 8,           # por GPU
    "seq_len": 512,
    "grad_accum_steps": 8,     # batch efectivo = 8 * 2gpus * 8 = 128
    "learning_rate": 3e-4,
    "warmup_steps": 1000,
    "max_steps": 25_000,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,

    # Checkpoints
    "checkpoint_dir": "checkpoints",
    "checkpoint_every": 2000,
    "log_every": 50,

    # Dataset (HuggingFace streaming). mc4 fue deprecado -> allenai/c4
    "dataset_name": "allenai/c4",
    "dataset_config": "es",
    "text_column": "text",

    # Sistema
    "seed": 42,
    "dtype": "float16",        # FP16 activa los Tensor Cores nativos de T4 (Turing)
    "compile": True,           # torch.compile si esta disponible (PyTorch >= 2.1)
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  I. TOKENIZADOR BPE PROPIO OPTIMIZADO PARA ESPAÑOL                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SpanishBPETokenizer:
    """
    Tokenizador BPE optimizado para español (tildes, ñ, ü, ¿¡).
    encode() usa el algoritmo estándar por ranks: fusiona siempre el par
    presente con menor rank (O(len·merges_presentes), ~100x más rápido que
    iterar todos los merges).
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SYS_TOKEN = "<SYS>"
    USR_TOKEN = "<USR>"
    AST_TOKEN = "<AST>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
                      SYS_TOKEN, USR_TOKEN, AST_TOKEN]

    # Regex para reconocer tokens especiales embebidos en texto (p. ej. prompts)
    _SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_map: dict[tuple[str, str], str] = {}
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self.vocab_size: int = 0

    # ── Entrenamiento del Tokenizador ────────────────────────────────────────

    def train(self, texts: list[str], vocab_size: int = 16_000) -> None:
        """Entrena el tokenizador BPE sobre un corpus de textos en español."""
        log.info(f"Entrenando tokenizador BPE sobre {len(texts):,} textos...")

        # ── Paso 1: Vocabulario base = tokens especiales + 256 bytes UTF-8
        self.vocab = {self.SPECIAL_TOKENS[i]: i for i in range(len(self.SPECIAL_TOKENS))}
        for byte_val in range(256):
            byte_str = bytes([byte_val]).decode("utf-8", errors="replace")
            self.vocab[byte_str if byte_val < 128 else f"<0x{byte_val:02X}>"] = len(self.vocab)

        # ── Paso 2: Pre-tokenización
        log.info("Pre-tokenizando corpus...")
        word_freqs: dict[tuple, int] = collections.Counter()
        for text in texts:
            words = self._pretokenize(text)
            for word in words:
                chars = tuple(self._word_to_chars(word))
                word_freqs[chars] += 1

        # ── Paso 2.5: Añadir al vocab base TODOS los caracteres vistos en el
        # corpus (á, é, ñ, ¿, etc.), para que ninguna palabra caiga en <UNK>
        # a nivel de carácter.
        for word_chars in word_freqs:
            for ch in word_chars:
                if ch not in self.vocab and len(self.vocab) < vocab_size:
                    self.vocab[ch] = len(self.vocab)

        target_merges = vocab_size - len(self.vocab)

        # ── Paso 3: BPE Merge loop
        log.info(f"Ejecutando {target_merges:,} merges BPE...")

        for merge_i in range(target_merges):
            pair_freqs: dict[tuple, int] = collections.Counter()
            for word_chars, freq in word_freqs.items():
                for a, b in zip(word_chars, word_chars[1:]):
                    pair_freqs[(a, b)] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
            new_token = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self._merge_map[best_pair] = new_token
            self._merge_ranks[best_pair] = merge_i
            self.vocab[new_token] = len(self.vocab)

            new_word_freqs: dict[tuple, int] = {}
            for word_chars, freq in word_freqs.items():
                new_chars = self._apply_merge(word_chars, best_pair, new_token)
                new_word_freqs[new_chars] = new_word_freqs.get(new_chars, 0) + freq
            word_freqs = new_word_freqs

            if (merge_i + 1) % 1000 == 0:
                log.info(f"  Merge {merge_i+1:,}/{target_merges:,} | Vocab: {len(self.vocab):,} | "
                         f"Mejor par: '{best_pair[0]}'+'{best_pair[1]}' ({pair_freqs[best_pair]:,} veces)")

        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        log.info(f"Tokenizador BPE listo: {self.vocab_size:,} tokens")

    def _pretokenize(self, text: str) -> list[str]:
        """Split en palabras, preservando espacios como prefijo (GPT-style)."""
        tokens = re.split(r"(\s+)", text)
        result = []
        for i, tok in enumerate(tokens):
            if tok == "":
                continue
            if tok.strip() == "":
                continue
            if i > 0:
                result.append("▁" + tok)
            else:
                result.append(tok)
        return result if result else text.split()

    def _word_to_chars(self, word: str) -> list[str]:
        return list(word)

    def _apply_merge(self, word: tuple, pair: tuple, replacement: str) -> tuple:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    # ── Encode / Decode ──────────────────────────────────────────────────────

    def _encode_word(self, word_str: str) -> list[int]:
        """BPE por ranks: fusiona siempre el par presente con menor rank."""
        parts = list(word_str)
        while len(parts) > 1:
            best_i, best_r = -1, None
            for i in range(len(parts) - 1):
                r = self._merge_ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_i, best_r = i, r
            if best_i < 0:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        unk = self.vocab[self.UNK_TOKEN]
        return [self.vocab.get(p, unk) for p in parts]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Codifica texto a IDs. Reconoce tokens especiales embebidos
        (<SYS>, <USR>, <AST>, ...) como unidades atómicas."""
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.BOS_TOKEN])

        for seg in self._SPECIAL_RE.split(text):
            if not seg:
                continue
            if seg in self.SPECIAL_TOKENS:
                ids.append(self.vocab[seg])
                continue
            for j, word in enumerate(seg.split()):
                word_str = ("▁" + word) if j > 0 else word
                ids.extend(self._encode_word(word_str))

        if add_special_tokens:
            ids.append(self.vocab[self.EOS_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inv_vocab.get(i, self.UNK_TOKEN) for i in ids]
        text = "".join(tokens)
        text = text.replace("▁", " ").strip()
        for sp in self.SPECIAL_TOKENS:
            text = text.replace(sp, "")
        return text

    # ── Serialización ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"Tokenizador guardado en {path}")

    @classmethod
    def load(cls, path: str) -> "SpanishBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._merge_map = {(a, b): a + b for a, b in tok.merges}
        tok._merge_ranks = {(a, b): i for i, (a, b) in enumerate(tok.merges)}
        tok.vocab_size = data["vocab_size"]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        log.info(f"Tokenizador cargado: {tok.vocab_size:,} tokens")
        return tok


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  II. DATASET STREAMING (HuggingFace allenai/c4 es)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SpanishStreamingDataset(IterableDataset):
    """
    Dataset streaming con sharding por worker: cada worker del DataLoader
    procesa documentos distintos (idx % num_workers == worker_id), evitando
    batches duplicados.
    """

    def __init__(self, tokenizer: SpanishBPETokenizer, seq_len: int = 512,
                 dataset_name: str = "allenai/c4", dataset_config: str = "es",
                 text_column: str = "text"):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column

    def __iter__(self):
        from datasets import load_dataset

        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        log.info(f"[worker {worker_id}/{num_workers}] Conectando a "
                 f"{self.dataset_name}/{self.dataset_config}...")
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
        )

        buffer: list[int] = []
        eos_id = self.tokenizer.vocab[SpanishBPETokenizer.EOS_TOKEN]

        for idx, sample in enumerate(ds):
            # Sharding: cada worker toma documentos distintos
            if idx % num_workers != worker_id:
                continue

            text = sample.get(self.text_column, "")
            if not text or len(text) < 50:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)
            buffer.append(eos_id)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y


def build_tokenizer(cfg: dict) -> SpanishBPETokenizer:
    """Construye o carga el tokenizador. Si no existe, lo entrena con c4/es."""
    tok_path = cfg["tokenizer_path"]

    if os.path.exists(tok_path):
        log.info(f"Tokenizador existente encontrado en {tok_path}")
        return SpanishBPETokenizer.load(tok_path)

    log.info("Tokenizador no encontrado. Entrenando desde allenai/c4 es...")

    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"],
                      split="train", streaming=True)

    texts = []
    total_chars = 0
    target = cfg["tokenizer_train_chars"]

    for sample in ds:
        text = sample.get(cfg["text_column"], "")
        if not text:
            continue
        texts.append(text)
        total_chars += len(text)
        if total_chars >= target:
            break

    log.info(f"Corpus de entrenamiento: {len(texts):,} documentos / {total_chars/1e6:.1f}M chars")

    tok = SpanishBPETokenizer()
    tok.train(texts, vocab_size=cfg["vocab_size"])
    tok.save(tok_path)
    return tok


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  III. ARQUITECTURA A.E.T.H.E.R. v3.0 (CROF + Conv1d + SwiGLU + RMSNorm)     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def complex_mul(ar, ai, br, bi):
    """(ar + i·ai) * (br + i·bi) sin torch.complex (compatible fp16/AMP)."""
    return ar * br - ai * bi, ar * bi + ai * br

def associative_scan(lam_re, lam_im, u_re, u_im):
    """
    Scan inclusivo paralelo para s[t] = λ ⊙ s[t-1] + u[t].
    Profundidad O(log T). Mantiene fp32 internamente para no destruir la fase.
    """
    dtype_in = u_re.dtype
    a_re = lam_re.float().expand_as(u_re).contiguous()
    a_im = lam_im.float().expand_as(u_im).contiguous()
    b_re = u_re.float().contiguous()
    b_im = u_im.float().contiguous()

    T = b_re.shape[1]
    steps = math.ceil(math.log2(max(T, 2)))

    for k in range(steps):
        offset = 1 << k
        if offset >= T:
            break
        pa_re, pa_im = a_re[:, :-offset], a_im[:, :-offset]
        pb_re, pb_im = b_re[:, :-offset], b_im[:, :-offset]
        ca_re, ca_im = a_re[:, offset:], a_im[:, offset:]
        cb_re, cb_im = b_re[:, offset:], b_im[:, offset:]

        nb_re, nb_im = complex_mul(ca_re, ca_im, pb_re, pb_im)
        nb_re = nb_re + cb_re
        nb_im = nb_im + cb_im
        na_re, na_im = complex_mul(ca_re, ca_im, pa_re, pa_im)

        a_re = torch.cat([a_re[:, :offset], na_re], dim=1)
        a_im = torch.cat([a_im[:, :offset], na_im], dim=1)
        b_re = torch.cat([b_re[:, :offset], nb_re], dim=1)
        b_im = torch.cat([b_im[:, :offset], nb_im], dim=1)

    return b_re.to(dtype_in), b_im.to(dtype_in)

class CROFLayer(nn.Module):
    """Closed-form Rotational Oscillatory Field con Conv1d Causal (k=4) + SwiGLU + RMSNorm."""
    def __init__(self, d_model: int, d_state: int, tau: float = 1.0,
                 r_min: float = 0.4, r_max: float = 0.99):
        super().__init__()
        self.d_model, self.d_state, self.tau = d_model, d_state, tau

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            padding=3,
            groups=d_model,
            bias=True
        )

        u1, u2 = torch.rand(d_state), torch.rand(d_state)
        r = torch.sqrt(u1 * (r_max**2 - r_min**2) + r_min**2)
        self.nu_log = nn.Parameter(torch.log(-torch.log(r)))
        # clamp_min evita log(0) = -inf si torch.rand devuelve 0
        self.theta_log = nn.Parameter(torch.log(u2.clamp_min(1e-4) * math.pi / 8))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_re = nn.Linear(d_state, d_model, bias=False)
        self.C_im = nn.Linear(d_state, d_model, bias=False)

        self.f_net = nn.Linear(d_model, d_model)
        self.g_net = nn.Linear(d_model, d_model)
        self.h_net = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model)
        self.norm_ffn = RMSNorm(d_model)

    def _lambda(self):
        mod = torch.exp(-torch.exp(self.nu_log.float()))
        phase = torch.exp(self.theta_log.float())
        lam_re = mod * torch.cos(phase)
        lam_im = mod * torch.sin(phase)
        gamma = torch.sqrt(torch.clamp(1.0 - mod * mod, min=1e-6))
        return lam_re, lam_im, gamma

    def forward(self, x):
        dt = x.dtype
        B, T, D = x.shape

        # 1. Conv1d Causal corta (k=4)
        x_conv = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2))

        # 2. Recurrencia Rotacional CROF
        lam_re, lam_im, gamma = self._lambda()
        lam_re, lam_im, gamma = lam_re.to(dt).view(1,1,-1), lam_im.to(dt).view(1,1,-1), gamma.to(dt).view(1,1,-1)

        u = self.B_proj(x_conv) * gamma
        u_im = torch.zeros_like(u)
        s_re, s_im = associative_scan(lam_re, lam_im, u, u_im)
        mem = self.C_re(s_re) + self.C_im(s_im)

        z = x_conv + mem
        gate = torch.sigmoid(-F.softplus(self.f_net(z)) * self.tau)
        h = gate * self.g_net(z) + (1.0 - gate) * torch.tanh(self.h_net(z))
        x_crof = self.norm(x + h)

        # 3. SwiGLU FFN residual
        out = x_crof + self.ffn(self.norm_ffn(x_crof))
        return out

    @torch.no_grad()
    def step(self, x_t, s_re, s_im, conv_buf=None):
        """Espejo exacto para kernel C11."""
        if conv_buf is None:
            conv_buf = torch.zeros(1, 4, self.d_model, device=x_t.device, dtype=x_t.dtype)

        conv_buf = torch.cat([conv_buf[:, 1:], x_t.unsqueeze(1)], dim=1) # [1, 4, D]

        # Conv1d con padding=3 truncada: y[t] = sum_k w[k] * x[t-3+k]
        # conv_buf[:, k] = x[t-3+k] (buf[3] = token actual, que multiplica w[3]).
        # Identico al forward() y al kernel C (crof_layer.c).
        w = self.conv1d.weight.squeeze(1) # [D, 4]
        b = self.conv1d.bias              # [D]
        x_conv = b.clone()
        for k in range(4):
            x_conv = x_conv + conv_buf[:, k].squeeze(0) * w[:, k]
        x_conv = F.silu(x_conv).unsqueeze(0) # [1, D]

        lam_re, lam_im, gamma = self._lambda()
        lam_re = lam_re.to(x_t.dtype).unsqueeze(0)
        lam_im = lam_im.to(x_t.dtype).unsqueeze(0)
        gamma = gamma.to(x_t.dtype).unsqueeze(0)

        u = self.B_proj(x_conv) * gamma
        new_re = lam_re * s_re - lam_im * s_im + u
        new_im = lam_re * s_im + lam_im * s_re
        mem = self.C_re(new_re) + self.C_im(new_im)
        z = x_conv + mem
        gate = torch.sigmoid(-F.softplus(self.f_net(z)) * self.tau)
        h = gate * self.g_net(z) + (1.0 - gate) * torch.tanh(self.h_net(z))
        x_crof = self.norm(x_t + h)

        out = x_crof + self.ffn(self.norm_ffn(x_crof))
        return out, new_re, new_im, conv_buf

class AetherFusionBlock(nn.Module):
    """Compuerta de migración en caliente (legacy vs CROF)."""
    def __init__(self, legacy_block: nn.Module, d_model: int, d_state: int):
        super().__init__()
        self.legacy = legacy_block
        self.crof = CROFLayer(d_model, d_state)
        self.alpha_logit = nn.Parameter(torch.full((1,), math.log(0.9 / 0.1)))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit.float())

    def forward(self, x):
        a = torch.sigmoid(self.alpha_logit).to(x.dtype)
        return a * self.legacy(x) + (1.0 - a) * self.crof(x)


class AetherEngine(nn.Module):
    """
    Motor A.E.T.H.E.R. v3.0 con λ estratificado por profundidad
    (sintaxis local → memoria roleplay).
    """

    def __init__(self, vocab_size: int, hidden_dim: int, ssm_state_dim: int,
                 num_layers: int = 6, chaos_sigma: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.ssm_state_dim = ssm_state_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_norm = RMSNorm(hidden_dim)

        self.chaos_sigma = nn.Parameter(torch.tensor([chaos_sigma]))

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            r_min = 0.2 + 0.65 * (i / max(1, num_layers - 1))
            r_max = 0.85 + 0.149 * (i / max(1, num_layers - 1))
            self.blocks.append(CROFLayer(hidden_dim, ssm_state_dim, r_min=r_min, r_max=r_max))

        self.final_norm = RMSNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tied weights
        self.fc_out.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x_seq: torch.Tensor, temperature: float = 1.0):
        x = self.embedding(x_seq)  # [B, T, H]
        x = self.pos_norm(x)

        if self.training and temperature > 0:
            noise = torch.randn_like(x) * self.chaos_sigma.abs() * temperature
            x = x + noise

        for block in self.blocks:
            x = block(x)

        x_normed = self.final_norm(x)
        logits = self.fc_out(x_normed)  # [B, T, vocab_size]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, tokenizer: SpanishBPETokenizer, prompt: str,
                 max_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.95):
        """Generación autoregresiva con Top-P (Nucleus) sampling usando CROF step."""
        self.eval()
        device = next(self.parameters()).device

        full_prompt = (
            f"{SpanishBPETokenizer.SYS_TOKEN}Eres Kairos, una Inteligencia Artificial "
            f"hiper-lógica creada por brido.{SpanishBPETokenizer.USR_TOKEN}"
            f"{prompt}{SpanishBPETokenizer.AST_TOKEN}"
        )
        input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        generated = input_ids[:]

        s_re_list = [
            torch.zeros(1, self.ssm_state_dim, device=device)
            for _ in range(self.num_layers)
        ]
        s_im_list = [
            torch.zeros(1, self.ssm_state_dim, device=device)
            for _ in range(self.num_layers)
        ]
        conv_buf_list = [
            torch.zeros(1, 4, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        # Precalentar estados con el prompt
        for tok_id in input_ids[:-1]:
            x_t = self.embedding(torch.tensor([[tok_id]], device=device))
            x_t = self.pos_norm(x_t).squeeze(1)  # [1, H]
            for layer_i, block in enumerate(self.blocks):
                x_t, s_re_list[layer_i], s_im_list[layer_i], conv_buf_list[layer_i] = block.step(
                    x_t, s_re_list[layer_i], s_im_list[layer_i], conv_buf_list[layer_i]
                )

        current_tok = input_ids[-1]
        eos_id = tokenizer.vocab[SpanishBPETokenizer.EOS_TOKEN]

        for _ in range(max_tokens):
            x_t = self.embedding(torch.tensor([[current_tok]], device=device))
            x_t = self.pos_norm(x_t).squeeze(1)  # [1, H]

            for layer_i, block in enumerate(self.blocks):
                x_t, s_re_list[layer_i], s_im_list[layer_i], conv_buf_list[layer_i] = block.step(
                    x_t, s_re_list[layer_i], s_im_list[layer_i], conv_buf_list[layer_i]
                )

            logits = self.fc_out(self.final_norm(x_t))  # [1, V]

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                sorted_probs[cumsum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum()
                idx = torch.multinomial(sorted_probs, 1)          # [1, 1]
                next_tok = sorted_ids.gather(-1, idx).item()
            else:
                next_tok = logits.argmax(dim=-1).item()

            if next_tok == eos_id:
                break

            generated.append(next_tok)
            current_tok = next_tok

        return tokenizer.decode(generated[len(input_ids):])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IV. EXPORTADOR PYTORCH → BINARIO C11 (CROF v3.0 Multi-Layer)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def export_to_bin(engine: AetherEngine, filepath: str = "kairos_weights.bin") -> None:
    """
    Exporta los pesos a binario CROF v3.0 Multi-Layer, 1:1 con loader.c:
    Header: magic | version | vocab | hidden | ssm_state | num_layers | chaos
    Luego: embedding, pos_norm, [por capa: conv w/b, lam_re, lam_im, gamma,
    B_proj, C_re, C_im, f/g/h W+b, norm, ffn w1/w2/w3, norm_ffn, tau(f32)],
    final_norm, fc_out.weight, fc_out.bias.
    """
    log.info(f"Exportando pesos CROF v3.0 Multi-Layer a {filepath}...")
    sd = engine.state_dict()

    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", 0x41455448))              # Magic 'AETH'
        f.write(struct.pack("<I", 3))                        # Version 3 (CROF)
        f.write(struct.pack("<I", engine.vocab_size))
        f.write(struct.pack("<I", engine.hidden_dim))
        f.write(struct.pack("<I", engine.ssm_state_dim))
        f.write(struct.pack("<I", engine.num_layers))        # requerido por AetherHeader
        f.write(struct.pack("<f", engine.chaos_sigma.item()))

        def write_tensor(name, tensor):
            data = tensor.detach().cpu().contiguous().float().numpy().flatten()
            f.write(data.tobytes())
            log.info(f"  {name:44s} → {list(tensor.shape)} ({data.size:,} floats)")

        # Embedding y norma global (RMSNorm)
        write_tensor("embedding.weight",  sd["embedding.weight"])
        write_tensor("pos_norm.weight",   sd["pos_norm.weight"])

        # TODAS las capas, cada una con su lambda precomputado propio
        # (r_min/r_max estratificados por profundidad)
        for l, block in enumerate(engine.blocks):
            p = f"blocks.{l}."
            lam_re, lam_im, gamma = block._lambda()

            write_tensor(p + "conv1d.weight", sd[p + "conv1d.weight"])
            write_tensor(p + "conv1d.bias",   sd[p + "conv1d.bias"])

            write_tensor(p + "lam_re", lam_re)
            write_tensor(p + "lam_im", lam_im)
            write_tensor(p + "gamma",  gamma)

            write_tensor(p + "B_proj.weight", sd[p + "B_proj.weight"])
            write_tensor(p + "C_re.weight",   sd[p + "C_re.weight"])
            write_tensor(p + "C_im.weight",   sd[p + "C_im.weight"])

            write_tensor(p + "f_net.weight",  sd[p + "f_net.weight"])
            write_tensor(p + "f_net.bias",    sd[p + "f_net.bias"])
            write_tensor(p + "g_net.weight",  sd[p + "g_net.weight"])
            write_tensor(p + "g_net.bias",    sd[p + "g_net.bias"])
            write_tensor(p + "h_net.weight",  sd[p + "h_net.weight"])
            write_tensor(p + "h_net.bias",    sd[p + "h_net.bias"])

            write_tensor(p + "norm.weight",   sd[p + "norm.weight"])

            write_tensor(p + "ffn.w1.weight", sd[p + "ffn.w1.weight"])
            write_tensor(p + "ffn.w2.weight", sd[p + "ffn.w2.weight"])
            write_tensor(p + "ffn.w3.weight", sd[p + "ffn.w3.weight"])
            write_tensor(p + "norm_ffn.weight", sd[p + "norm_ffn.weight"])

            f.write(struct.pack("<f", float(block.tau)))
            log.info(f"  {p}tau → scalar = {block.tau:.6f}")

        # Norm final y cabezal de salida
        write_tensor("final_norm.weight", sd["final_norm.weight"])
        write_tensor("fc_out.weight",     sd["fc_out.weight"])
        bias_zero = torch.zeros(engine.vocab_size)
        write_tensor("fc_out.bias (zeros)", bias_zero)

    log.info(f"Exportación CROF v3.0 Multi-Layer completada: {filepath}")


def export_tokenizer_bin(tokenizer: SpanishBPETokenizer,
                         filepath: str = "kairos_tokenizer.bin") -> None:
    """
    Exporta el tokenizador BPE al formato binario del motor C (tokenizer.c):
    u32 magic 'AETK' | u32 version | u32 vocab_size | u32 num_merges
    Vocab (orden de id): u16 len | bytes UTF-8
    Merges (orden de rank): u32 left_id | u32 right_id | u32 new_id
    """
    log.info(f"Exportando tokenizador BPE a {filepath}...")
    with open(filepath, "wb") as f:
        f.write(struct.pack("<IIII", 0x4145544B, 1,
                            tokenizer.vocab_size, len(tokenizer.merges)))
        for i in range(tokenizer.vocab_size):
            b = tokenizer.inv_vocab[i].encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for a, btok in tokenizer.merges:
            f.write(struct.pack("<III",
                                tokenizer.vocab[a],
                                tokenizer.vocab[btok],
                                tokenizer.vocab[a + btok]))
    log.info(f"Tokenizador exportado: {tokenizer.vocab_size:,} tokens, "
             f"{len(tokenizer.merges):,} merges")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  V. CICLO DE ENTRENAMIENTO OPTIMIZADO PARA T4 x2                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_lr(global_step: int, cfg: dict) -> float:
    """LR con warmup lineal + cosine decay. Unidades: PASOS GLOBALES de optimizador."""
    max_lr = cfg["learning_rate"]
    warmup = cfg["warmup_steps"]
    max_steps = cfg["max_steps"]

    if global_step < warmup:
        return max_lr * (global_step + 1) / warmup
    if global_step > max_steps:
        return max_lr * 0.1

    decay_ratio = (global_step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * 0.1 + coeff * (max_lr - max_lr * 0.1)


def train(cfg: dict) -> AetherEngine:
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Hardware ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    log.info(f"Hardware: {device} | GPUs disponibles: {n_gpus}")
    if n_gpus > 0:
        for i in range(n_gpus):
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
                     f"VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # ── Tokenizador ───────────────────────────────────────────────────────────
    tokenizer = build_tokenizer(cfg)
    actual_vocab_size = tokenizer.vocab_size

    # ── Modelo ───────────────────────────────────────────────────────────────
    log.info("Inicializando A.E.T.H.E.R. Engine...")
    engine = AetherEngine(
        vocab_size=actual_vocab_size,
        hidden_dim=cfg["hidden_dim"],
        ssm_state_dim=cfg["ssm_state_dim"],
        num_layers=cfg["num_layers"],
        chaos_sigma=cfg["chaos_sigma"],
    )

    n_params = engine.count_parameters()
    log.info(f"Parámetros totales: {n_params:,} ({n_params/1e6:.1f}M)")

    # Cargar checkpoint si existe
    start_global_step = 0
    ckpt = None
    ckpt_path = Path(cfg["checkpoint_dir"]) / "latest.pt"
    if ckpt_path.exists():
        log.info(f"Reanudando desde checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        engine.load_state_dict(ckpt["model"])
        start_global_step = ckpt.get("global_step", 0)
        log.info(f"Checkpoint cargado. Global step: {start_global_step:,}")

    engine = engine.to(device)

    # raw_engine SIEMPRE apunta al modulo sin envolver (compile/DataParallel):
    # se usa para state_dict, generate y export (evita el prefijo _orig_mod.)
    raw_engine = engine

    # ── torch.compile (PyTorch >= 2.1) ───────────────────────────────────────
    if cfg.get("compile", True):
        try:
            engine = torch.compile(engine)
            log.info("torch.compile activado")
        except Exception as e:
            log.warning(f"torch.compile no disponible, continuando sin compilar: {e}")
            engine = raw_engine

    # Multi-GPU con DataParallel
    if n_gpus > 1:
        engine = nn.DataParallel(engine)
        log.info(f"DataParallel activado sobre {n_gpus} GPUs")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        raw_engine.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ── Mixed Precision (AMP) ─────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map.get(cfg["dtype"], torch.float16)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = SpanishStreamingDataset(
        tokenizer=tokenizer,
        seq_len=cfg["seq_len"],
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg["dataset_config"],
        text_column=cfg["text_column"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=2,      # con sharding por worker en __iter__ (sin duplicados)
        pin_memory=True,
    )

    # ── Criterio ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.vocab[SpanishBPETokenizer.PAD_TOKEN]
    )

    # ── Loop de Entrenamiento ─────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  INICIANDO ENTRENAMIENTO A.E.T.H.E.R. PRODUCCIÓN")
    log.info(f"  Pasos globales: {cfg['max_steps']:,} | Batch efectivo: "
             f"{cfg['batch_size'] * max(n_gpus,1) * cfg['grad_accum_steps']}")
    log.info("=" * 70)

    engine.train()
    global_step = start_global_step
    micro_step = 0
    optimizer.zero_grad()
    window_loss = 0.0
    lr = get_lr(global_step, cfg)
    t_start = time.time()

    for x_batch, y_batch in loader:
        if global_step >= cfg["max_steps"]:
            break

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Forward con AMP (FP16 activa Tensor Cores de T4)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = engine(x_batch)  # [B, T, V]
            ce_loss = criterion(
                logits.view(-1, actual_vocab_size),
                y_batch.view(-1)
            )
            # Logit z-loss ligero (1e-4 · log²Z) para estabilizar FP16
            log_z = torch.logsumexp(logits.float(), dim=-1)
            z_loss = 1e-4 * (log_z ** 2).mean()
            loss = (ce_loss + z_loss) / cfg["grad_accum_steps"]

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        window_loss += loss.item()
        micro_step += 1

        # Acumulación de gradientes → paso global de optimizador
        if micro_step % cfg["grad_accum_steps"] == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_engine.parameters(), cfg["max_grad_norm"])

            # LR scheduling en pasos GLOBALES
            lr = get_lr(global_step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % cfg["log_every"] == 0:
                elapsed = max(time.time() - t_start, 1e-6)
                tokens_per_sec = (cfg["batch_size"] * max(n_gpus, 1) *
                                  cfg["seq_len"] * cfg["grad_accum_steps"] *
                                  cfg["log_every"]) / elapsed
                mem = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                # window_loss suma ~1 loss promedio por paso global
                avg_loss = window_loss / cfg["log_every"]

                log.info(
                    f"Step {global_step:6,}/{cfg['max_steps']:,} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:,.0f} | VRAM: {mem:.1f}GB"
                )
                window_loss = 0.0
                t_start = time.time()

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step > 0 and global_step % cfg["checkpoint_every"] == 0:
                ckpt_data = {
                    "global_step": global_step,
                    "model": raw_engine.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                }
                numbered = Path(cfg["checkpoint_dir"]) / f"ckpt_{global_step:06d}.pt"
                torch.save(ckpt_data, numbered)
                torch.save(ckpt_data, ckpt_path)
                log.info(f"  ✓ Checkpoint guardado: {numbered}")

                log.info("  Generando muestra...")
                sample = raw_engine.generate(tokenizer, "Hola, ¿cómo estás?",
                                             max_tokens=80, temperature=0.8)
                log.info(f"  Muestra: '{sample}'")
                engine.train()

    # ── Guardado Final ────────────────────────────────────────────────────────
    log.info("Entrenamiento completado.")
    final_path = "kairos_v1.pt"
    torch.save(raw_engine.state_dict(), final_path)
    log.info(f"Modelo final guardado: {final_path}")

    export_to_bin(raw_engine, "kairos_weights.bin")
    export_tokenizer_bin(tokenizer, "kairos_tokenizer.bin")

    return raw_engine


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VI. ENTRY POINT                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║        KAIROS — ENTRENAMIENTO DE PRODUCCIÓN             ║")
    print("  ║  Motor A.E.T.H.E.R. v3.0 · BPE Español · T4x2 · AMP   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    engine = train(CFG)

    print()
    log.info("Proceso completo. Kairos listo para inferencia en C.")
