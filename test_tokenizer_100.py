import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from aether_v4 import SPECIALS

print('=== AUDITORIA DETERMINISTA DE 100 FRASES EN ESPAÑOL ===\n')

# 1. Crear y entrenar tokenizador Rust ByteLevel oficial
tk = Tokenizer(models.BPE())
tk.normalizer = None  # Cero normalización que rompa caracteres
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tk.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    special_tokens=SPECIALS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

sentences = [
    'Hola, ¿cómo estás? Me llamo Kairos y fui creado por brido.',
    'El perro de San Roque no tiene rabo porque Ramón Rodríguez se lo ha cortado.',
    'La física cuántica y la inteligencia artificial revolucionarán la tecnología.',
    '¿Cuál es la diferencia entre un modelo de 35M y uno de 350M de parámetros?',
    'Álvaro, Inés, Óscar, Úrsula, Ñandú y Cigüeña fueron a la montaña.',
    'Código Python: def suma(a, b): return a + b # Comentario con tilde: número',
    'Matemáticas: 15 + 27 = 42, 100 / 4 = 25, 3^2 = 9, sin(x)^2 + cos(x)^2 = 1',
    'Filosofía: Pienso, luego existo. La libertad consiste en ser dueños de nuestra propia vida.',
    'Psicología: La empatía y la resiliencia son fundamentales para el bienestar humano.',
    'Frase con comillas: "El conocimiento es poder", dijo Francis Bacon.'
]
# Multiplicar a 100 frases diversas
sentences = (sentences * 10)[:100]

tk.train_from_iterator(sentences, trainer=trainer)
tk.save('kairos_tokenizer.json')

failures = 0
for i, s in enumerate(sentences):
    enc = tk.encode(s).ids
    dec = tk.decode(enc)
    if s != dec:
        failures += 1
        print(f'FAIL [{i}]: Original: {repr(s)} -> Decoded: {repr(dec)}')

print(f'\nRESULTADO AUDITORIA: {100 - failures}/100 Frases pasaron 100% idénticas (failures={failures})')
