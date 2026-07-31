import json
import struct

def export_tokenizer_bin(json_path="kairos_tokenizer.json", bin_path="kairos_tokenizer.bin"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    merges = data["merges"]

    inv_vocab = {int(v): k for k, v in vocab.items()}
    vocab_size = len(inv_vocab)
    num_merges = len(merges)

    magic = 0x4B544542
    version = 2

    byte_to_id = [0] * 256
    for i in range(256):
        ch = chr(i)
        if ch in vocab:
            byte_to_id[i] = vocab[ch]
        else:
            byte_to_id[i] = 1

    with open(bin_path, "wb") as f:
        # 1. Header (16 bytes)
        f.write(struct.pack("<IIII", magic, version, vocab_size, num_merges))

        # 2. Table byte_to_id[256] (1024 bytes)
        for i in range(256):
            f.write(struct.pack("<I", byte_to_id[i]))

        # 3. Vocab
        for i in range(vocab_size):
            tok_str = inv_vocab.get(i, "")
            tok_bytes = tok_str.encode("utf-8")
            f.write(struct.pack("<H", len(tok_bytes)))
            f.write(tok_bytes)

        # 4. Merges
        for m in merges:
            p1, p2 = m[0], m[1]
            merged_str = p1 + p2
            id_left = vocab.get(p1, 1)
            id_right = vocab.get(p2, 1)
            id_new = vocab.get(merged_str, 1)
            f.write(struct.pack("<III", id_left, id_right, id_new))

    print(f"[OK] '{bin_path}' generado exitosamente ({vocab_size} tokens, {num_merges} merges)")

if __name__ == "__main__":
    export_tokenizer_bin()
