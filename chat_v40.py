#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KAIROS v4.0 - CHAT INTERACTIVO CON RAZONAMIENTO (<think>...</think>)
================================================================================
"""

import os
import torch
from huggingface_hub import hf_hub_download
from aether_v4 import AetherEngine, KairosTokenizer, CFG as BASE_CFG

SYSTEM_PROMPT = "Eres Kairos, una inteligencia artificial de razonamiento lógico, útil y honesta creada por brido."

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================================================")
    print("  KAIROS v4.0 - CHAT CON PENSAMIENTO EN CADENA (<think>...</think>)")
    print(f"  Dispositivo: {device}")
    print("==================================================================\n")

    # 1. Cargar Tokenizador
    tok_path = "kairos_tokenizer.json"
    if not os.path.exists(tok_path):
        try:
            tok_path = hf_hub_download(repo_id="Bridoxd/AETHER-v2", filename="v4/kairos_tokenizer.json")
        except Exception:
            tok_path = hf_hub_download(repo_id="Bridoxd/AETHER-v2", filename="kairos_tokenizer.json")

    tok = KairosTokenizer.load(tok_path)
    vocab_size = ((tok.vocab_size + 63) // 64) * 64

    # 2. Cargar modelo v4.0 Instruct
    ckpt_path = "checkpoints_instruct40/latest.pt"
    if not os.path.exists(ckpt_path):
        print("Descargando `v4/kairos_instruct_v40.pt` desde Hugging Face...")
        ckpt_path = hf_hub_download(repo_id="Bridoxd/AETHER-v2", filename="v4/kairos_instruct_v40.pt")

    print(f"Cargando pesos de Kairos v4.0 desde '{ckpt_path}'...")
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = sd.get("model", sd)
    
    # Auto-detectar dimensiones exactas del modelo cargado (35M vs 350M vs 1B)
    vocab_size = state_dict["embedding.weight"].shape[0] if "embedding.weight" in state_dict else 16000
    hidden_dim = state_dict["embedding.weight"].shape[1] if "embedding.weight" in state_dict else 1024
    layer_idxs = [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("layers.")]
    num_layers = (max(layer_idxs) + 1) if layer_idxs else 24
    num_heads = max(1, hidden_dim // 64)

    cfg = dict(BASE_CFG)
    cfg["hidden_dim"] = hidden_dim
    cfg["num_layers"] = num_layers
    cfg["num_heads"] = num_heads

    print(f"-> Arquitectura detectada: {hidden_dim}d, {num_layers} capas, {num_heads} heads ({vocab_size} vocab)")

    model = AetherEngine(vocab_size=vocab_size, cfg=cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("\n¡KAIROS LISTO! Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        try:
            user_input = input("\nUsuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                break

            prompt = f"<SYS>{SYSTEM_PROMPT}<USR>{user_input}<AST>"
            print("\nKairos: ", end="", flush=True)
            res, _ = model.generate(tok, prompt, max_tokens=250, temperature=0.7)
            print(res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
