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

## Principio para modelos pequeños

Un modelo de 35M no puede memorizar el conocimiento general de un modelo de miles de millones de parámetros. Sí puede ser un asistente coherente si dejamos de exigirle que sea simultáneamente cerebro, base de datos, calculadora y buscador.

AETHER-X separa cuatro funciones:

1. **Core lingüístico:** español, sintaxis, tono y conversación.
2. **Razonador latente:** pocos pasos internos para planear sin imprimir todo el razonamiento.
3. **Memoria externa:** hechos del usuario y conocimiento recuperable en páginas compactas.
4. **Herramientas/copy path:** números, nombres, código y texto literal no se fuerzan a pasar por una memoria comprimida.

Así el modelo pequeño aprende a hablar bien; la memoria y las herramientas le dan alcance sin inflar el core. La promesa realista de 35M es un asistente conversacional especializado y claro, no un modelo general que sepa todo sin ayuda.

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

### 3. Scratchpad local acotado

No eliminamos toda mezcla local. Cada segmento mantiene un scratchpad de 32-128 unidades recientes, con atención o mezcla lineal pequeña. Se descarta o resume al cerrar una página.

Esto protege la copia literal y la coherencia de frases sin pagar KV cache global. Para texto predecible se usa solo la celda; para código, números y nombres se activa el modo exacto.

### 4. Profundidad dinámica por segmento

Cada segmento puede hacer 1, 2, 4 o más iteraciones de la celda compartida. Un segmento fácil sale pronto; uno ambiguo, raro o con alta pérdida recibe más cómputo. La decisión usa un halting gate monotónico y una penalización explícita por FLOPs.

El batch se mantiene eficiente con buckets por profundidad y máscaras de segmentos activos. No se acepta un bucle Python por token en la versión de producción.

Para evitar que el halting destruya la velocidad, X1 empieza con solo dos rutas estáticas: `fast` y `deep`, agrupadas por batch. El halting libre de muchos niveles se prueba después.

### 5. Razonamiento latente corto

Entre la pregunta y la respuesta el modelo puede ejecutar 2-4 pasos internos sobre un vector de estado sin emitir texto. Los pasos se entrenan con respuestas verificables y consistencia, no con la obligación de imitar cada palabra del razonamiento de un profesor.

La salida visible sigue siendo supervisada directamente. El razonamiento latente es una ayuda, no evidencia de que el modelo "piense" mágicamente.

### 6. Memoria asociativa externa

La memoria persistente no intenta meter todos los hechos en un tensor. El modelo genera registros `{key, value, salience, time, provenance}` y los consolida en páginas comprimidas. En inferencia solo se cargan las páginas top-k relevantes, con cache LRU.

El `.kmem` contiene estado neuronal, índice y páginas. El texto original no se reinyecta como prompt salvo que una evaluación lo pida explícitamente.

Las páginas tienen dos rutas: `exact` para nombres, números y claves, y `semantic` para preferencias y resúmenes. Así evitamos que una compresión neuronal destruya datos que deben copiarse literalmente.

### 7. Capacidad elástica real

El banco de expertos/adapters de `aether_elastic.py` queda como backend de capacidad:

- expertos pequeños por dominio o habilidad;
- router que activa top-k constante;
- expertos fríos en disco y hot-set en VRAM;
- ledger obligatorio para reconstruir la forma del modelo;
- crecimiento solo después de estancamiento comprobado.

La capacidad crece en disco. Los parámetros activos por token y el presupuesto de VRAM siguen acotados.

### 8. Ruta CPU futura

Después de validar calidad en FP16, se prueba una rama ternaria o 2-bit para matrices seleccionadas: router, proyecciones de la celda y expertos fríos. No se cuantiza antes de tener una baseline estable. La meta es usar kernels de suma/resta SIMD, no fingir que una matriz ternaria sigue siendo una GEMM FP16.

## Cómo reducimos los riesgos

| Riesgo | Diseño preventivo | Prueba que lo puede tumbar |
|---|---|---|
| Segmentador difícil | empezar con BPE como teacher, byte fallback, curriculum de compresión | BPB peor que BPE tras mismo presupuesto |
| Bytes lentos | segmentación temprana, cache binario de bytes/segmentos y packing por longitud | tok/s menor que baseline sin recuperar calidad |
| Mala copia literal | scratchpad local + ruta exacta para spans raros | exact-match de nombres, números y código |
| Memoria que olvida | páginas exact/semantic, salience, timestamp, contradicción y replay | NLL con memoria no mejora o mezcla hechos |
| Halting irregular | primero fast/deep en buckets, coste FLOPs explícito | utilización GPU o throughput cae >15% |
| Offload lento | hot-set LRU, prefetch por router, páginas contiguas y expertos pequeños | cache miss dispara latencia >2x |
| Crecimiento rompe training | ledger, checkpoint transaccional, reconstrucción de optimizer/EMA/DDP | no se puede recargar con `strict=True` |
| Ternario baja calidad | solo después de FP16, QAT por etapas y precisión mixta | pérdida o chat empeoran más del umbral |
| PyTorch lento | primero referencia correcta, después kernels fused y C++/CUDA | no se acepta optimización sin benchmark reproducible |
| Demasiado código | cada módulo nace con self-test y ablación independiente | cualquier módulo sin métrica se elimina |

## Modelo pequeño: cómo hacerlo hablar bien

### Arquitectura compacta recomendada

Configuración inicial `X0-35M`:

- 128-256 dimensiones latentes;
- 8-12 capas lógicas compartidas, no 24 capas únicas;
- 2 cabezas de memoria asociativa de bajo rango;
- scratchpad local de 64 unidades;
- salida byte-level o vocabulario pequeño validado, no una matriz gigante innecesaria;
- embeddings y salida factorizados o atados;
- adapters de habilidad cargables bajo demanda;
- estado persistente separado del core.

El objetivo no es que 35M memorice más. Es que desperdicie menos parámetros en capas repetidas, vocabulario fijo, contexto redundante y respuestas largas de razonamiento.

### Entrenamiento por fases

#### Fase A: lenguaje español limpio

Entrenar sobre datos filtrados y deduplicados: conversación natural, prosa, instrucciones, preguntas y respuestas, código corto y texto con tildes/ñ. Menos basura web, más tokens útiles. Medir bits-per-byte además de loss.

#### Fase B: distilación de comportamiento

Un teacher grande produce respuestas, correcciones, pares de preferencia, explicaciones compactas y ejemplos de rechazo. El estudiante aprende logits o distribuciones cuando sea posible, no solo el texto final.

Se usa generación activa: el teacher genera ejemplos donde el estudiante falla, en vez de fabricar millones de ejemplos aleatorios. El objetivo es transferir estilo, coherencia y hábitos de respuesta, no copiar conocimiento infinito.

#### Fase C: conversación española

SFT con loss en la respuesta, formato estable `<SYS> <USR> <AST>`, turnos cortos y largos, preguntas ambiguas, corrección de errores, "no sé", seguimiento de contexto y respuestas concisas. Mezclar datos reales curados con datos sintéticos revisados.

#### Fase D: razonamiento y herramientas

Entrenar al modelo a decidir entre responder, recuperar memoria, usar calculadora/código o pedir aclaración. La respuesta final se entrena con verificación. Los números y nombres se copian por la ruta exacta.

#### Fase E: memoria persistente

Entrenar escritura, recuperación, actualización, contradicción y olvido con episodios artificiales. La métrica no es que repita un resumen bonito: es que baje NLL de la respuesta correcta y conserve exactitud después de guardar/cargar.

#### Fase F: compresión

Primero FP16 estable, después 8-bit, 4-bit y finalmente ternario en módulos seleccionados. Cada reducción debe compararse en calidad, RAM, latencia y pérdida, no solo en tamaño del archivo.

## Por qué esto puede superar a un Transformer en un modelo pequeño

- El core no carga con todo el conocimiento: recupera lo necesario.
- La celda compartida convierte más parámetros en profundidad de procesamiento útil.
- El razonamiento latente evita gastar tokens visibles en pasos internos.
- La ruta exacta protege nombres, números y código.
- El modelo aprende cuándo usar memoria o herramientas, en vez de alucinar.
- El entrenamiento se enfoca en datos de alta señal y fallos reales del estudiante.
- La memoria persistente puede crecer sin aumentar los pesos activos.
- El costo de inferencia se adapta a la dificultad.

## Límites que no vamos a maquillar

- 35M no será un modelo general comparable a 7B sin memoria o herramientas.
- Una memoria externa no aumenta la capacidad de razonamiento del core por arte de magia.
- Distilar respuestas malas produce un modelo pequeño que habla bonito y se equivoca con confianza.
- La recurrencia puede perder copia literal si el scratchpad y la ruta exacta fallan.
- Segmentar bytes puede ser peor que BPE en la primera iteración.
- El halting adaptativo puede ser más lento si se implementa con kernels ingenuos.
- Los expertos en disco no son gratis: cache miss significa latencia.
- La cuantización extrema puede exigir sacrificar calidad.

## Orden de iteración

### X0: celda mínima

- d=128 o 192;
- 2-4 celdas compartidas con estado `F+S`;
- BPE actual para aislar arquitectura;
- sin LTM, sin MoE, sin MTP y sin segmentador aprendido;
- comparar contra v4.0 tiny con mismos tokens y semillas;
- tests: forward chunk vs step, gradientes, estabilidad y estado constante.

### X0-chat: capacidad conversacional mínima

- dataset español curado de alta señal;
- destilación de respuestas y correcciones;
- SFT con masking correcto;
- formato de chat estable;
- evaluación humana y automática de coherencia, no solo loss.

### X1: cómputo adaptativo seguro

- dos buckets: `fast` y `deep`;
- medir FLOPs reales, tokens/s y distribución de rutas;
- exigir calidad igual o mejor al mismo presupuesto de FLOPs;
- eliminar cualquier router colapsado o que siempre use la ruta profunda.

### X2: segmentación aprendida

- reemplazar BPE por bytes + segmenter;
- mantener byte fallback;
- evaluar bits-per-byte, compresión, tildes, ñ, código, números y texto ruidoso;
- comparar contra BPE y byte-level fijo.

### X3: memoria persistente

- conectar `P` y páginas exact/semantic;
- evaluar NIAH, hechos contradictorios, actualización y olvido;
- medir ganancia NLL con memoria, deriva tras guardar/cargar y bytes por recuerdo.

### X4: capacidad elástica

- insertar banco de expertos/adapters;
- crecer únicamente por estancamiento comprobado;
- reconstruir optimizer, EMA y DDP tras cada evento;
- comparar calidad por parámetros activos, no por parámetros totales.

### X5: CPU

- exportar estado y pesos a formato compacto;
- probar 8-bit, 4-bit y ternario por separado;
- implementar kernel SIMD solo después de fijar el formato;
- comparar latencia por token, RAM y pérdida contra FP16.

## Gate de calidad

No se declara victoria por texto bonito. Cada versión debe reportar:

- loss y perplexity o bits-per-byte;
- tokens/segmentos por segundo;
- FLOPs estimados y FLOPs realmente ejecutados;
- VRAM, RAM y tamaño del `.kmem`;
- exactitud de tokenizer/bytes;
- NIAH y recall entre sesiones;
- exactitud de nombres, números, código y acentos;
- coherencia de chat en español;
- estabilidad al partir el texto en chunks distintos;
- calidad después de crecer, guardar, cargar y cuantizar.

La regla es simple: si una mejora no supera a v4.0 bajo el mismo presupuesto de datos y cómputo, se revierte. AETHER-X debe ser diferente, sí, pero primero tiene que ser mediblemente mejor.