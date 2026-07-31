# AETHER-X X0: quickstart

## 1. Validacion local

Corre primero:

```bash
python aether_x.py --smoke
python test_aether_x.py
python train_aether_x.py --smoke
```

Los tests comprueban forward contra step, estado fijo, backward, gradiente del router fast/deep y roundtrip del estado.

## 2. Chat espanol pequeno

Crea `chat.jsonl` con una linea por ejemplo:

```json
{"messages":[{"role":"system","content":"Eres KAIROS, un asistente claro y breve."},{"role":"user","content":"Hola, quien eres?"},{"role":"assistant","content":"Soy KAIROS, un asistente creado por Angel."}]}
```

Entrena primero una corrida corta:

```bash
python train_aether_x.py --mode chat \
  --tokenizer kairos_tokenizer.json \
  --data chat.jsonl \
  --hidden 512 --ffn 2048 --slots 8 --scratch 32 \
  --batch 8 --seq-len 256 --steps 1000 \
  --out checkpoints/x0_chat.pt
```

La loss se calcula solo sobre los tokens de `assistant`. El router tiene una penalizacion pequena para no usar `deep` siempre.

## 3. Pretraining

Usa el cache binario ya validado con el mismo tokenizer:

```bash
python train_aether_x.py --mode pretrain \
  --tokenizer kairos_tokenizer.json \
  --cache kairos_tokens_es.bin \
  --hidden 512 --ffn 2048 \
  --batch 8 --seq-len 256 --steps 5000 \
  --out checkpoints/x0_pretrain.pt
```

No mezcles checkpoint de otro tokenizer. X0 usa BPE a proposito para medir la celda sin meter el problema del segmentador aprendido.

## 4. Presupuesto de memoria

El estado por sesion es:

```python
model.state_bytes(1)
```

No aumenta con la cantidad de tokens ingeridos. El scratchpad es fijo y protege copia local; la memoria asociativa guarda patrones, no un KV cache creciente.

## 5. Orden correcto

1. Pasa smoke tests.
2. Entrena 100-1000 pasos y revisa loss, deep ratio, VRAM y tok/s.
3. Evalua chat con preguntas nunca vistas.
4. Compara contra v4 tiny con los mismos datos y tokens.
5. Solo despues agrega segmentacion por bytes, memoria externa y capacidad elastica.

X0 no pretende saber todo con 35M. Pretende aprender español conversacional limpio y usar después memoria, recuperación y herramientas para ampliar su alcance sin inflar el core.
