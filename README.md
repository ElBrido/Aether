# 🧬 A.E.T.H.E.R. v3.0 — CROF Architecture

**Arquitectura CROF (Closed-form Rotational Oscillatory Field)**

Motor de inferencia de tiempo continuo en forma cerrada que reemplaza la integración numérica ODE por números complejos y recurrencia rotacional, logrando latencia fija $\mathcal{O}(1)$ por token sin NFE explosivo y retención de memoria inquebrantable.

> Creado por **brido** · Inferencia en C11 puro · Entrenamiento en PyTorch

---

## Arquitectura CROF v3.0

```
Token → Embedding → PosNorm → [Inyección de Caos] → CROFLayer (Associative Scan) → FinalNorm → Head → Token
```

| Componente | Función | Complejidad |
|---|---|---|
| **CROFLayer** | Campo Oscilatorio Rotacional en Forma Cerrada con números complejos | $\mathcal{O}(1)$ latencia / token |
| **Associative Scan** | Scan inclusivo paralelo de estado rotacional complejo $s[t] = \lambda \odot s[t-1] + u[t]$ | $\mathcal{O}(\log T)$ scan paralelo |
| **Inyección de Caos** | Ruido gaussiano $\mathcal{N}(0, \sigma^2 \cdot T)$ en embeddings | Parámetro $\sigma$ aprendible |
| **AetherFusionBlock** | Compuerta de migración en caliente $\alpha \cdot \text{legacy} + (1-\alpha) \cdot \text{crof}$ | Fase de migración suave |

## Pipeline de Desarrollo

```
┌─────────────────────────┐        ┌─────────────────────────┐
│  Fase A: Laboratorio    │        │  Fase B: Producción     │
│  (PyTorch + GPU)        │        │  (C11 Puro + CPU)       │
│                         │        │                         │
│  train_production.py    │───────>│  src/*.c                │
│  ├─ Entrena con CROF    │ .bin   │  ├─ Lee binario (v3.0)  │
│  ├─ Precomputa λ_re/im  │ ────>  │  ├─ Carga con punteros  │
│  └─ Test de paridad     │        │  └─ Step rotacional C11 │
└─────────────────────────┘        └─────────────────────────┘
```

## Estructura del Repositorio

```
proyecto-aether/
├── include/
│   └── aether_core.h      # Header maestro (structs CROF v3.0 y prototipos)
├── src/
│   ├── tensor.c            # Operaciones de tensor, LayerNorm y activaciones (sigmoid, tanh, softplus)
│   ├── crof_layer.c        # Capa CROF en forma cerrada (recurrencia oscilatoria compleja)
│   ├── embedding.c         # Tabla de embeddings (token_id → vector)
│   ├── sampler.c           # Muestreo multinomial con Temperatura y Top-P
│   ├── loader.c            # Cargador binario (.bin v3.0 → pesos en memoria)
│   ├── ewc.c               # Elastic Weight Consolidation (cimientos)
│   ├── aether.c            # Composición del motor Kairos/AETHER v3.0
│   └── main.c              # Entry point, parity test, generación interactiva
├── train_production.py     # Script de entrenamiento PyTorch (CROF + Spanish BPE 16K)
├── Makefile                # Build system (gcc, O3, debug, asan)
└── README.md
```

## Uso Rápido

### 1. Entrenar en Kaggle (PyTorch)

```bash
pip install torch datasets huggingface_hub
python train_production.py
```

Esto genera:
- `kairos_v1.pt` — Checkpoint StateDict PyTorch
- `kairos_weights.bin` — Binario CROF v3.0 plano para motor en C
- `kairos_tokenizer.json` — Tokenizador BPE de 16,000 tokens en español

### 2. Compilar Motor C11

```bash
make clean && make
```

### 3. Ejecutar Inferencia

```bash
# Con pesos entrenados:
./aether_engine kairos_weights.bin

# Sin pesos (demo con pesos aleatorios):
./aether_engine
```

## Formato Binario CROF v3.0 (.bin)

```
Offset  Contenido                         Tipo
──────  ────────────────────────────────  ────────
0x00    Magic: 0x41455448 ('AETH')        uint32
0x04    Version: 3                        uint32
0x08    vocab_size                        uint32
0x0C    hidden_dim                        uint32
0x10    ssm_state_dim (d_state)           uint32
0x14    chaos_sigma                       float32
0x18    embedding.weight [V × H]          float32[]
...     pos_norm.weight / bias [H]        float32[]
...     blocks.0.lam_re [S]               float32[] (precomputado _lambda())
...     blocks.0.lam_im [S]               float32[] (precomputado _lambda())
...     blocks.0.gamma [S]                float32[] (precomputado _lambda())
...     blocks.0.B_proj.weight [S × H]    float32[]
...     blocks.0.C_re.weight [H × S]      float32[]
...     blocks.0.C_im.weight [H × S]      float32[]
...     blocks.0.f_net.weight [H × H]     float32[]
...     blocks.0.f_net.bias [H]           float32[]
...     blocks.0.g_net.weight [H × H]     float32[]
...     blocks.0.g_net.bias [H]           float32[]
...     blocks.0.h_net.weight [H × H]     float32[]
...     blocks.0.h_net.bias [H]           float32[]
...     blocks.0.norm.weight / bias [H]   float32[]
...     blocks.0.tau                      float32
...     final_norm.weight / bias [H]      float32[]
...     fc_out.weight [V × H]             float32[]
...     fc_out.bias [V]                   float32[]
```

---

*Kairos / A.E.T.H.E.R. v3.0 — Memoria Rotacional Compleja en Forma Cerrada por brido.*