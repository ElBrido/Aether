import zipfile, pickle, io, struct
import numpy as np

# Load export_standalone state
with zipfile.ZipFile('kairos_pure_chat_latest.pt', 'r') as z:
    pkl_name = [f for f in z.namelist() if f.endswith('data.pkl') or f.endswith('archive/data.pkl')][0]
    prefix = pkl_name.replace('data.pkl', '')
    
    class PyTorchUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return lambda storage, storage_offset, size, stride, requires_grad, backward_hooks: {'storage': storage, 'offset': storage_offset, 'size': size, 'stride': stride}
            if module == 'torch' and 'Storage' in name:
                return lambda *args: name
            return super().find_class(module, name)
        def persistent_load(self, pid):
            return pid

    unp = PyTorchUnpickler(io.BytesIO(z.read(pkl_name)))
    sd = unp.load()

ema = sd.get('ema', sd.get('model'))
print("EMA keys count:", len(ema))

# Read binary weights file header to verify
with open('kairos_weights.bin', 'rb') as f:
    magic, ver, vocab, dim, ssm, layers, chaos = struct.unpack('<IIIIIIf', f.read(28))
    print(f"Header: magic={hex(magic)}, ver={ver}, vocab={vocab}, dim={dim}, ssm={ssm}, layers={layers}")

# Check parity test input: tokens [1, 1, 1]
# Let's inspect what logits PyTorch produces vs C11!
