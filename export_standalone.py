#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 EXPORTADOR INDEPENDIENTE DE PESOS (0 DEPENDENCIAS DE TORCH)
 Convierte `latest.zip` / `latest.pt` -> `kairos_weights.bin`
 Funciona con la libreria estandar de Python (zipfile + pickle + numpy)
================================================================================
"""

import zipfile
import pickle
import io
import struct
import sys
import os
import numpy as np

AETHER_MAGIC = 0x41455448
AETHER_VERSION = 3

class PyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                return {'storage': storage, 'offset': storage_offset, 'size': size, 'stride': stride}
            return rebuild
        if module == 'torch' and 'Storage' in name:
            return lambda *args: name
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return pid

def get_tensor_numpy(z, prefix, t_dict):
    st_tuple = t_dict['storage']
    key = str(st_tuple[2])
    raw_data = z.read(prefix + 'data/' + key)
    
    # dtype por defecto float32
    dtype = np.float32
    if 'FloatStorage' in str(st_tuple[1]):
        dtype = np.float32
    elif 'HalfStorage' in str(st_tuple[1]):
        dtype = np.float16

    arr = np.frombuffer(raw_data, dtype=dtype)
    offset = t_dict.get('offset', 0)
    shape = t_dict['size']
    numel = int(np.prod(shape)) if shape else 1
    
    arr = arr[offset:offset + numel]
    if dtype == np.float16:
        arr = arr.astype(np.float32)
        
    return arr.reshape(shape) if shape else arr

def export_standalone(zip_path="latest.zip", out_bin="kairos_weights.bin"):
    if not os.path.exists(zip_path):
        if os.path.exists("latest.pt"):
            zip_path = "latest.pt"
        else:
            print(f"[ERROR] No se encontro '{zip_path}' ni 'latest.pt'")
            return False

    print(f"[EXPORT] Leyendo '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        prefix = [n for n in z.namelist() if n.endswith('data.pkl')][0].rsplit('data.pkl', 1)[0]
        pkl_data = z.read(prefix + 'data.pkl')
        unp = PyTorchUnpickler(io.BytesIO(pkl_data))
        sd = unp.load()
        state = sd.get('ema', sd.get('model', sd))

        # Detectar dimensiones
        emb_arr = get_tensor_numpy(z, prefix, state['embedding.weight'])
        vocab_size, hidden_dim = emb_arr.shape

        num_layers = 0
        while f"blocks.{num_layers}.norm.weight" in state:
            num_layers += 1

        def get_weight(p_key):
            if p_key in state:
                return get_tensor_numpy(z, prefix, state[p_key])
            base_key = p_key.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
            if base_key in state:
                base_w = get_tensor_numpy(z, prefix, state[base_key])
                if ".weight" in p_key:
                    a_key = p_key.replace(".weight", ".A")
                    b_key = p_key.replace(".weight", ".B")
                    if a_key in state and b_key in state:
                        A = get_tensor_numpy(z, prefix, state[a_key])
                        B = get_tensor_numpy(z, prefix, state[b_key])
                        scale = 32.0 / 16.0
                        return base_w + (B @ A) * scale
                return base_w
            raise KeyError(p_key)

        b0_B = get_weight('blocks.0.B_proj.weight')
        ssm_state_dim = b0_B.shape[0]

        chaos_sigma = 0.02
        if 'chaos_sigma' in state:
            cs = state['chaos_sigma']
            if isinstance(cs, dict):
                chaos_sigma = float(get_tensor_numpy(z, prefix, cs))
            else:
                chaos_sigma = float(cs)

        print(f"[EXPORT] Parametros del modelo detectados:")
        print(f"  vocab_size    = {vocab_size}")
        print(f"  hidden_dim    = {hidden_dim}")
        print(f"  ssm_state_dim = {ssm_state_dim}")
        print(f"  num_layers    = {num_layers}")
        print(f"  chaos_sigma   = {chaos_sigma:.6f}")

        def write_arr(f, arr, name=""):
            arr32 = arr.astype(np.float32).flatten()
            f.write(arr32.tobytes())

        print(f"[EXPORT] Escribiendo '{out_bin}' (Fusionando LoRA si existe)...")
        with open(out_bin, "wb") as f:
            # Header (28 bytes: 6x uint32_t + 1x float)
            f.write(struct.pack("<IIIIIIf", AETHER_MAGIC, AETHER_VERSION, vocab_size, hidden_dim, ssm_state_dim, num_layers, chaos_sigma))

            # Embeddings y normas iniciales
            write_arr(f, emb_arr, "embedding.weight")
            write_arr(f, get_weight('pos_norm.weight'), "pos_norm.weight")

            # Bloques CROF
            for l in range(num_layers):
                p = f"blocks.{l}."
                write_arr(f, get_weight(p + "conv1d.weight"), p + "conv1d.weight")
                write_arr(f, get_weight(p + "conv1d.bias"), p + "conv1d.bias")

                # Reconstruir lam_re, lam_im, gamma desde nu_log / theta_log
                nu_log = get_weight(p + "nu_log")
                theta_log = get_weight(p + "theta_log")
                mod = np.exp(-np.exp(nu_log))
                ph = np.exp(theta_log)
                lam_re = mod * np.cos(ph)
                lam_im = mod * np.sin(ph)
                gamma = np.sqrt(np.maximum(1.0 - mod * mod, 1e-6))

                write_arr(f, lam_re, p + "lam_re")
                write_arr(f, lam_im, p + "lam_im")
                write_arr(f, gamma, p + "gamma")

                write_arr(f, get_weight(p + "B_proj.weight"), p + "B_proj.weight")
                write_arr(f, get_weight(p + "C_re.weight"), p + "C_re.weight")
                write_arr(f, get_weight(p + "C_im.weight"), p + "C_im.weight")

                write_arr(f, get_weight(p + "f_net.weight"), p + "f_net.weight")
                write_arr(f, get_weight(p + "f_net.bias"), p + "f_net.bias")
                write_arr(f, get_weight(p + "g_net.weight"), p + "g_net.weight")
                write_arr(f, get_weight(p + "g_net.bias"), p + "g_net.bias")
                write_arr(f, get_weight(p + "h_net.weight"), p + "h_net.weight")
                write_arr(f, get_weight(p + "h_net.bias"), p + "h_net.bias")

                write_arr(f, get_weight(p + "norm.weight"), p + "norm.weight")

                write_arr(f, get_weight(p + "ffn.w1.weight"), p + "ffn.w1.weight")
                write_arr(f, get_weight(p + "ffn.w2.weight"), p + "ffn.w2.weight")
                write_arr(f, get_weight(p + "ffn.w3.weight"), p + "ffn.w3.weight")
                write_arr(f, get_weight(p + "norm_ffn.weight"), p + "norm_ffn.weight")

                # Tau (scalar = 1.0)
                f.write(struct.pack("<f", 1.0))

            # Norm final y cabezal de salida
            write_arr(f, get_weight('final_norm.weight'), "final_norm.weight")
            
            if 'fc_out.weight' in state or 'fc_out.base.weight' in state:
                write_arr(f, get_weight('fc_out.weight'), "fc_out.weight")
            else:
                write_arr(f, emb_arr, "fc_out.weight")

            if 'fc_out.bias' in state or 'fc_out.base.bias' in state:
                write_arr(f, get_weight('fc_out.bias'), "fc_out.bias")
            else:
                write_arr(f, np.zeros(vocab_size, dtype=np.float32), "fc_out.bias")


    print(f"[EXPORT] ÉXITO: '{out_bin}' generado ({os.path.getsize(out_bin) / (1024*1024):.1f} MB)")
    return True

if __name__ == "__main__":
    zip_p = sys.argv[1] if len(sys.argv) > 1 else "latest.zip"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "kairos_weights.bin"
    export_standalone(zip_p, out_p)
