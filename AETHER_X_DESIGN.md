# A.E.T.H.E.R. X

## Tesis

AETHER-X no debe ser un Transformer con menos atención. Debe tratar el texto como un flujo de información y gastar capacidad solo cuando hace falta.

La promesa medible es:

- memoria de trabajo constante respecto a la longitud del texto;
- capacidad total ampliable en disco, con cómputo activo acotado;
- menos operaciones sobre texto predecible y más sobre texto difícil;
- ejecución viable en GPU pequeña y, después, en CPU;
- memoria persistente separada de los pesos y de la ventana de contexto.

No se promete "infinito" sin almacenamiento. Se separan tres presupuestos: pesos totales, parámetros activos por token y estado persistente.

## Qué se conserva de v4.0

1. Gated Delta Rule como baseline de memoria asociativa.
2. Estado serializable `.kmem` y pruebas de persistencia entre procesos.
3. entrenamiento chunkwise para paralelizar en T4;
4. S0 tuning como estado inicial especializado;
5. checkpoints reproducibles y tests deterministas.

`mtp_weight=0.0` se mantiene en la baseline para medir velocidad y loss limpias. MTP se reintroduce después como ablación, no mezclado con la primera prueba de arquitectura.

## AETHER-X: piezas nuevas

### 1. ByteFlow Adaptive Segmenter

Entrada UTF-8 cruda, sin BPE obligatorio. Un encoder pequeño acumula bytes y aprende cuándo cerrar un segmento según sorpresa, compresibilidad y cambio semántico. El backbone trabaja sobre segmentos latentes, no sobre cada byte.

Debe existir una ruta de escape byte-a-byte para caracteres raros, código y texto corrupto. La granularidad no es fija: palabras sencillas pueden formar un segmento largo; nombres, números y código reciben segmentos cortos.

Objetivo auxiliar: reconstrucción exacta de bytes + penalización suave por demasiados segmentos + target de compresión con curriculum.

### 2. AETHER Cell recurrente

Una celda compartida reemplaza el stack fijo de bloques. Mantiene tres estados:

- `F`: memoria rápida local, actualizada en cada segmento;
- `S`: memoria asociativa de mediano plazo, actualización delta de bajo rango;
- `P`: memoria persistente comprimida, escrita solo cuando la compuerta de sorpresa lo justifica.

La celda produce `x_next` y un score de trabajo. No usa atención global como camino principal.

### 3. Profundidad dinámica por token

Cada segmento puede hacer 1, 2, 4 o más iteraciones de la celda compartida. Un token fácil sale pronto; uno ambiguo, raro o con alta pérdida recibe más cómputo. La decisión usa un halting gate monotónico y una penalización explícita por FLOPs.

El batch se mantiene eficiente con máscaras de tokens activos y buckets por profundidad. No se acepta un bucle Python por token en la versión de producción.

### 4. Memoria asociativa externa

La memoria persistente no intenta meter todos los hechos en un tensor. El modelo genera registros `{key, value, salience, time, provenance}` y los consolida en páginas comprimidas. En inferencia solo se cargan las páginas top-k relevantes, con cache LRU.

El `.kmem` contiene estado neuronal, índice y páginas. El texto original no se reinyecta como prompt salvo que una evaluación lo pida explícitamente.

### 5. Capacidad elástica real

El banco de expertos/adapters de `aether_elastic.py` queda como backend de capacidad:

- expertos pequeños por dominio o habilidad;
- router que activa top-k constante;
- expertos fríos en disco y hot-set en VRAM;
- ledger obligatorio para reconstruir la forma del modelo;
- crecimiento solo después de estancamiento comprobado.

La capacidad crece en disco. Los parámetros activos por token y el presupuesto de VRAM siguen acotados.

### 6. Ruta CPU futura

Después de validar calidad en FP16, se prueba una rama ternaria o 2-bit para matrices seleccionadas: router, proyecciones de la celda y expertos fríos. No se cuantiza antes de tener una baseline estable. La meta es usar kernels de suma/resta SIMD, no fingir que una matriz ternaria sigue siendo una GEMM FP16.

## Objetivos de entrenamiento

La pérdida total inicial será:

`L = L_byte + lambda_seg * L_segment + lambda_mem * L_memory + lambda_compute * L_compute + lambda_aux * L_router`

En la primera corrida solo se activan `L_byte` y una versión pequeña de `L_segment`. Memoria, crecimiento, MTP y ternarización se agregan uno por uno.

## Orden de iteración

### X0: celda mínima

- d=128, 2 capas lógicas compartidas, estado F+S;
- entrada BPE actual para aislar arquitectura;
- sin LTM, sin MoE, sin MTP;
- comparar contra v4.0 tiny con mismos tokens y semillas;
- tests: forward chunk vs step, gradientes, estabilidad y estado constante.

### X1: cómputo adaptativo

- añadir halting por segmento;
- medir FLOPs reales, tokens/s y distribución de profundidad;
- exigir que la calidad no caiga con el mismo presupuesto de FLOPs;
- eliminar cualquier router colapsado o que siempre use la profundidad máxima.

### X2: segmentación aprendida

- reemplazar BPE por bytes + segmenter;
- evaluar bits-per-byte, compresión, tildes, ñ, código, números y texto ruidoso;
- comparar contra BPE y byte-level fijo, no solo contra una muestra generada.

### X3: memoria persistente

- conectar P y páginas externas;
- evaluar NIAH, hechos contradictorios, actualización y olvido;
- medir ganancia NLL con memoria, deriva tras guardar/cargar y bytes por recuerdo.

### X4: capacidad elástica

- insertar banco de expertos/adapters;
- crecer únicamente por estancamiento;
- reconstruir optimizer, EMA y DDP tras cada evento;
- comparar calidad por parámetros activos, no por parámetros totales.

### X5: CPU

- exportar estado y pesos a formato compacto;
- probar 8-bit, 4-bit y ternario por separado;
- implementar kernel SIMD solo después de fijar el formato;
- comparar latencia por token, RAM y pérdida contra FP16.

## Ventajas esperadas frente a un Transformer

- no necesita KV cache que crezca con el contexto;
- no depende de un vocabulario BPE fijo;
- computa menos en texto fácil;
- puede recordar entre procesos mediante estado y páginas persistentes;
- puede crecer sin cargar todo el banco en VRAM;
- comparte la celda y reduce parámetros redundantes;
- tiene una ruta clara a CPU, baja precisión y almacenamiento barato;
- la granularidad del procesamiento se adapta al idioma, dominio y dificultad;
- puede separar memoria de trabajo, conocimiento permanente y capacidad física.

## Costes inevitables

- el segmenter aprendido hace el entrenamiento más delicado que BPE;
- la recurrencia y el halting dinámico pueden perder throughput si se implementan con kernels ingenuos;
- la memoria comprimida puede olvidar o mezclar hechos si no hay recuperación y consolidación correctas;
- copiar texto literal largo puede ser peor sin una memoria local especializada;
- crecer cambia la forma del modelo y complica optimizer, EMA, DDP y checkpoints;
- ternarizar puede bajar calidad y exige entrenamiento consciente de cuantización;
- habrá más código y más tests que en un Transformer estándar;
- al principio PyTorch puede ser más lento hasta escribir kernels fused.

## Gate de calidad

No se declara victoria por texto bonito. Cada versión debe reportar:

- loss y perplexity o bits-per-byte;
- tokens/segmentos por segundo;
- FLOPs estimados y FLOPs realmente ejecutados;
- VRAM, RAM y tamaño del `.kmem`;
- exactitud de tokenizer/bytes;
- NIAH y recall entre sesiones;
- estabilidad al partir el texto en chunks distintos;
- calidad después de crecer, guardar, cargar y cuantizar.

La regla es simple: si una mejora no supera a v4.0 bajo el mismo presupuesto de datos y cómputo, se revierte. AETHER-X debe ser diferente, sí, pero primero tiene que ser mediblemente mejor.