#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KAIROS - INTERFACE DE CHAT INTERACTIVO (INSTRUCT FINE-TUNED)
 Usa el Tokenizador de Rust de Hugging Face para Respuestas 100% Impecables
================================================================================
"""

import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from train_production import AetherEngine, KairosTokenizer

SYSTEM_PROMPT = "Eres Kairos, una Inteligencia Artificial hiper-logica y servicial creada por brido."

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================================================")
    print("  KAIROS CHAT INTERACTIVO (A.E.T.H.E.R. v3.1 CROF)")
    print(f"  Dispositivo: {device}")
    print("==================================================================\n")

    # 1. Cargar Tokenizador oficial
    tok_path = "kairos_tokenizer.json"
    if not os.path.exists(tok_path):
        tok_path = hf_hub_download(repo_id="Bridoxd/AETHER-v2", filename="kairos_tokenizer.json")

    tok = KairosTokenizer.load(tok_path)
    vocab_size = ((tok.vocab_size + 63) // 64) * 64

    # 2. Cargar modelo de Chat entrenado
    ckpt_path = "checkpoints_instruct/kairos_instruct_latest.pt"
    if not os.path.exists(ckpt_path):
        print("Descargando `kairos_instruct_latest.pt` desde Hugging Face...")
        ckpt_path = hf_hub_download(repo_id="Bridoxd/AETHER-v2", filename="kairos_instruct_latest.pt")

    print(f"Cargando modelo desde '{ckpt_path}'...")
    model = AetherEngine(vocab_size=vocab_size, hidden_dim=512, ssm_state_dim=1024, num_layers=6).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    print("\n¡KAIROS LISTO PARA CHATEAR! Escribe 'salir' para terminar.\n")

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                break

            prompt = f"<SYS>{SYSTEM_PROMPT}<USR>{user_input}<AST>"
            response = model.generate(
                tok,
                prompt,
                max_tokens=150,
                temperature=0.35,
                top_p=0.90,
                repetition_penalty=1.15
            )
            print(f"Kairos: {response.strip()}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
