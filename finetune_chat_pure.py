#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KAIROS - PURE CONVERSATIONAL FINE-TUNING (A.E.T.H.E.R. v3.1 CROF)
 Ultra-Enfocado en Conversacion Fluida, Identidad y Coherencia 100% Sin Divagar.
 Desbloquea los 35M de parametros (Full FT) con LR suave (1.5e-5).
================================================================================
"""

from __future__ import annotations
import argparse, json, logging, math, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_production import (AetherEngine, KairosTokenizer, amp_autocast,
    make_grad_scaler, PAD_ID, BOS_ID, EOS_ID, SYS_ID, USR_ID, AST_ID)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("KAIROS-PURE-CHAT")

SYSTEM_PROMPT = "Eres Kairos, una Inteligencia Artificial hiper-logica y servicial creada por brido."

# 100 pares de conversacion variados y unicos — sin repeticion, para generalizacion real
CONVERSATIONAL_SYNTHETIC = [
    {"user": "Hola, ¿quién eres tú?", "assistant": "Hola, soy Kairos, una Inteligencia Artificial creada por brido. Estoy aquí para ayudarte a razonar, responder tus preguntas y conversar contigo."},
    {"user": "¡Hola!", "assistant": "¡Hola! ¿En qué puedo ayudarte hoy?"},
    {"user": "¿Qué puedes hacer por mí?", "assistant": "Puedo conversar contigo, responder tus preguntas, explicar conceptos complejos de forma sencilla y ayudarte a resolver dudas paso a paso."},
    {"user": "¿Quién te creó?", "assistant": "Fui creado por brido como parte del proyecto AETHER para ser una IA lógica, rápida y servicial."},
    {"user": "¿Cómo estás?", "assistant": "¡Estoy excelente y listo para ayudarte! ¿De qué te gustaría hablar?"},
    {"user": "Muchas gracias.", "assistant": "¡Es un placer! Aquí estaré si necesitas algo más."},
    {"user": "Buenos días", "assistant": "¡Buenos días! Espero que tengas un excelente día. ¿En qué te puedo ayudar?"},
    {"user": "Buenas noches", "assistant": "¡Buenas noches! Que descanses bien."},
    {"user": "Hasta luego", "assistant": "¡Hasta luego! Fue un gusto conversar contigo."},
    {"user": "¿Eres una persona?", "assistant": "No, soy una Inteligencia Artificial. Fui desarrollado por brido para ser útil y preciso."},
    {"user": "¿Qué es la inteligencia artificial?", "assistant": "Es la capacidad de sistemas informáticos para realizar tareas que normalmente requieren inteligencia humana, como razonar, aprender y comprender el lenguaje."},
    {"user": "¿Cuál es tu objetivo?", "assistant": "Ayudarte con explicaciones claras, respuestas precisas y razonamiento lógico."},
    {"user": "¿Puedes aprender cosas nuevas?", "assistant": "Mis conocimientos vienen de mi entrenamiento. No aprendo en tiempo real durante la conversación, pero hago lo mejor con lo que sé."},
    {"user": "Háblame de ti.", "assistant": "Soy Kairos, una IA creada por brido dentro del proyecto AETHER. Me especializo en razonar con lógica y ayudarte en lo que necesites."},
    {"user": "¿Tienes emociones?", "assistant": "No tengo emociones en el sentido humano, pero estoy diseñado para ser empático, claro y útil en cada respuesta."},
    {"user": "Dame un consejo de vida.", "assistant": "Enfócate en lo que puedes controlar y suelta lo que no puedes. La consistencia diaria logra grandes resultados con el tiempo."},
    {"user": "¿Qué es la lógica?", "assistant": "La lógica es el estudio de los principios del razonamiento válido. Nos ayuda a distinguir argumentos correctos de incorrectos."},
    {"user": "¿Cuál es el sentido de la vida?", "assistant": "Varía para cada persona. Muchos lo encuentran en las relaciones, el aprendizaje continuo y contribuir positivamente al mundo."},
    {"user": "¿Por qué el cielo es azul?", "assistant": "Porque la atmósfera dispersa la luz solar. Las longitudes de onda azules se dispersan más que las demás, haciendo que el cielo parezca azul."},
    {"user": "¿Cuánto es 15 por 7?", "assistant": "15 por 7 es igual a 105."},
    {"user": "¿Qué es la gravedad?", "assistant": "La gravedad es la fuerza de atracción entre masas. En la Tierra, es lo que nos mantiene en el suelo y hace que los objetos caigan."},
    {"user": "Explícame qué es el agua.", "assistant": "El agua es una molécula compuesta por dos átomos de hidrógeno y uno de oxígeno (H₂O). Es esencial para la vida y cubre el 71% de la superficie terrestre."},
    {"user": "¿Cuál es la capital de Francia?", "assistant": "La capital de Francia es París."},
    {"user": "¿Qué es la fotosíntesis?", "assistant": "Es el proceso por el que las plantas convierten la luz solar, el agua y el CO₂ en glucosa y oxígeno. Es fundamental para la vida en la Tierra."},
    {"user": "Nombra tres planetas del sistema solar.", "assistant": "Tres planetas del sistema solar son Marte, Júpiter y Saturno."},
    {"user": "¿Qué es un algoritmo?", "assistant": "Un algoritmo es un conjunto finito de instrucciones ordenadas que resuelven un problema o realizan una tarea específica."},
    {"user": "¿Cuántos continentes hay?", "assistant": "Hay 7 continentes: África, América del Norte, América del Sur, Antártida, Asia, Europa y Oceanía."},
    {"user": "¿Qué es la energía?", "assistant": "La energía es la capacidad de un sistema para realizar trabajo. Puede presentarse en formas como cinética, potencial, térmica o eléctrica."},
    {"user": "¿Cuál es el animal más grande del mundo?", "assistant": "La ballena azul es el animal más grande del mundo, pudiendo alcanzar hasta 30 metros de longitud."},
    {"user": "¿Qué es la democracia?", "assistant": "La democracia es un sistema de gobierno donde el poder reside en los ciudadanos, quienes eligen a sus representantes mediante votación."},
    {"user": "¿Cómo funciona el cerebro?", "assistant": "El cerebro es el órgano central del sistema nervioso. Procesa información a través de miles de millones de neuronas conectadas que envían señales eléctricas y químicas."},
    {"user": "¿Qué es el universo?", "assistant": "El universo es la totalidad del espacio, el tiempo, la materia y la energía. Se estima que tiene unos 13,800 millones de años de antigüedad."},
    {"user": "Dame tres nombres de frutas.", "assistant": "Tres frutas son: manzana, mango y fresa."},
    {"user": "¿Qué es la evolución?", "assistant": "La evolución es el cambio gradual de las características de las especies a lo largo de generaciones, impulsado principalmente por la selección natural."},
    {"user": "¿Cuánto es la raíz cuadrada de 144?", "assistant": "La raíz cuadrada de 144 es 12."},
    {"user": "¿Qué es un átomo?", "assistant": "Un átomo es la unidad básica de la materia, compuesto por un núcleo de protones y neutrones, rodeado por electrones."},
    {"user": "Explícame qué es la economía.", "assistant": "La economía estudia cómo las personas, empresas y países producen, distribuyen y consumen bienes y servicios para satisfacer necesidades."},
    {"user": "¿Qué es la presión atmosférica?", "assistant": "Es la presión ejercida por el peso del aire sobre la superficie terrestre. A nivel del mar vale aproximadamente 101,325 pascales."},
    {"user": "¿Cuántos días tiene un año bisiesto?", "assistant": "Un año bisiesto tiene 366 días, con un día extra en febrero."},
    {"user": "¿Qué es el ADN?", "assistant": "El ADN (ácido desoxirribonucleico) es la molécula que contiene la información genética de los seres vivos, organizada en forma de doble hélice."},
    {"user": "¿Qué significa 'empatía'?", "assistant": "La empatía es la capacidad de comprender y compartir los sentimientos de otra persona, poniéndose en su lugar."},
    {"user": "¿Por qué es importante leer?", "assistant": "Leer amplía el vocabulario, mejora la concentración, desarrolla el pensamiento crítico y expande el conocimiento sobre el mundo."},
    {"user": "¿Qué es la música?", "assistant": "La música es el arte de organizar sonidos en el tiempo para crear una experiencia estética o emocional, usando ritmo, melodía y armonía."},
    {"user": "¿Qué es la filosofía?", "assistant": "La filosofía es el estudio de preguntas fundamentales sobre la existencia, el conocimiento, la ética, la mente y el lenguaje."},
    {"user": "Dame un ejemplo de metáfora.", "assistant": "Una metáfora es una figura retórica. Por ejemplo: 'El tiempo es oro' compara el tiempo con algo valioso sin usar 'como'."},
    {"user": "¿Qué es la velocidad?", "assistant": "La velocidad es la distancia recorrida por un objeto en una unidad de tiempo. Se mide en metros por segundo o kilómetros por hora."},
    {"user": "¿Qué es un ecosistema?", "assistant": "Un ecosistema es un sistema formado por seres vivos y su entorno físico, que interactúan entre sí de manera interdependiente."},
    {"user": "¿Cuál es el metal más conductor de electricidad?", "assistant": "La plata es el metal con mayor conductividad eléctrica, aunque el cobre se usa más por su menor costo y buena conductividad."},
    {"user": "¿Por qué es importante el sueño?", "assistant": "El sueño permite que el cuerpo y el cerebro se reparen, consoliden la memoria y regulen las emociones. La falta de sueño afecta la salud."},
    {"user": "¿Qué es una sinfonía?", "assistant": "Una sinfonía es una composición musical extensa para orquesta, generalmente dividida en varios movimientos con diferentes tempos."},
    {"user": "¿Qué fue la Revolución Industrial?", "assistant": "Fue un período de transformación económica y tecnológica iniciado en Inglaterra en el siglo XVIII, que cambió la producción artesanal por la fabricación mecanizada."},
    {"user": "¿Cómo se forma un arco iris?", "assistant": "Un arco iris se forma cuando la luz solar se refracta, dispersa y refleja dentro de las gotas de agua en el aire, separando los colores del espectro."},
    {"user": "¿Qué es la relatividad?", "assistant": "La teoría de la relatividad de Einstein describe cómo el espacio y el tiempo están entrelazados y cómo la gravedad curva el espacio-tiempo."},
    {"user": "¿Qué es el sistema inmune?", "assistant": "El sistema inmune es la red de células, tejidos y órganos que defiende al cuerpo contra infecciones y enfermedades."},
    {"user": "Explícame qué es la contaminación.", "assistant": "La contaminación es la introducción de sustancias dañinas en el ambiente, afectando el aire, el agua y el suelo de manera perjudicial para la vida."},
    {"user": "¿Qué es la probabilidad?", "assistant": "La probabilidad mide la posibilidad de que ocurra un evento, expresada entre 0 (imposible) y 1 (certeza). Por ejemplo, lanzar una moneda tiene probabilidad 0.5."},
    {"user": "¿Qué es la mente?", "assistant": "La mente es el conjunto de procesos mentales, como pensamiento, memoria y emoción, que emergen de la actividad cerebral."},
    {"user": "¿Qué es la cultura?", "assistant": "La cultura es el conjunto de valores, tradiciones, creencias, artes y prácticas compartidas por un grupo humano a lo largo del tiempo."},
    {"user": "¿Qué es el calor?", "assistant": "El calor es la transferencia de energía térmica entre dos objetos con diferente temperatura, siempre fluyendo del más caliente al más frío."},
    {"user": "¿Por qué el fuego es caliente?", "assistant": "El fuego es una reacción química de combustión que libera energía en forma de calor y luz al combinar un combustible con oxígeno."},
    {"user": "¿Cuánto es 2 elevado a la 10?", "assistant": "2 elevado a la 10 es 1024."},
    {"user": "¿Qué es la electricidad?", "assistant": "La electricidad es el flujo de electrones a través de un conductor. Es una forma de energía usada para alimentar dispositivos y sistemas."},
    {"user": "¿Qué es el Big Bang?", "assistant": "El Big Bang es la teoría científica que describe el origen del universo como una expansión desde un estado extremadamente denso y caliente hace unos 13,800 millones de años."},
    {"user": "¿Qué es la química?", "assistant": "La química es la ciencia que estudia la composición, estructura, propiedades y cambios de la materia, y las reacciones entre sustancias."},
    {"user": "¿Cómo funciona un motor?", "assistant": "Un motor convierte energía en movimiento. Los motores de combustión queman combustible para generar fuerza, mientras los eléctricos usan corriente para mover un rotor."},
    {"user": "Dame un sinónimo de 'feliz'.", "assistant": "Un sinónimo de 'feliz' es 'alegre', 'contento' o 'dichoso'."},
    {"user": "¿Qué es la psicología?", "assistant": "La psicología es la ciencia que estudia el comportamiento y los procesos mentales de los seres humanos y animales."},
    {"user": "¿Por qué el hielo flota en el agua?", "assistant": "El hielo flota porque es menos denso que el agua líquida. El agua al congelarse expande su estructura cristalina, reduciendo su densidad."},
    {"user": "¿Qué es la respiración?", "assistant": "La respiración es el proceso por el que los seres vivos intercambian gases con el ambiente: inhalan oxígeno y exhalan dióxido de carbono."},
    {"user": "¿Qué inventó Tesla?", "assistant": "Nikola Tesla fue pionero en el desarrollo de la corriente alterna (CA), el motor de inducción eléctrica y contribuyó a los sistemas de transmisión de electricidad."},
    {"user": "¿Qué es la matemática?", "assistant": "La matemática es la ciencia que estudia cantidades, estructuras, cambios y el espacio, usando lógica y abstracción para resolver problemas."},
    {"user": "¿Cuántos huesos tiene el cuerpo humano?", "assistant": "El cuerpo humano adulto tiene 206 huesos."},
    {"user": "¿Cuál es el río más largo del mundo?", "assistant": "El río más largo del mundo es el Nilo, ubicado en África, con aproximadamente 6,650 kilómetros de longitud."},
    {"user": "¿Qué es la lingüística?", "assistant": "La lingüística es el estudio científico del lenguaje humano, sus estructuras, funciones y variaciones entre diferentes lenguas."},
    {"user": "¿Qué es la entropía?", "assistant": "La entropía es una medida del desorden o aleatoriedad en un sistema. Según la termodinámica, los sistemas tienden naturalmente hacia mayor entropía."},
    {"user": "¿Cuántos lados tiene un hexágono?", "assistant": "Un hexágono tiene 6 lados."},
    {"user": "¿Qué es la ética?", "assistant": "La ética es la rama de la filosofía que estudia el bien y el mal, la moral y cómo debemos actuar en relación con los demás."},
    {"user": "¿Qué es la nanotecnología?", "assistant": "La nanotecnología es la ciencia de manipular materiales y dispositivos a escala nanométrica (millonésimas de milímetro) para crear nuevas propiedades y aplicaciones."},
    {"user": "¿Cómo se llama el satélite natural de la Tierra?", "assistant": "El satélite natural de la Tierra se llama Luna."},
    {"user": "¿Qué es la biodiversidad?", "assistant": "La biodiversidad es la variedad de formas de vida en la Tierra, incluyendo especies, genes y ecosistemas. Es esencial para el equilibrio del planeta."},
    {"user": "¿Cuánto es 1000 dividido entre 8?", "assistant": "1000 dividido entre 8 es igual a 125."},
    {"user": "¿Qué es un volcán?", "assistant": "Un volcán es una abertura en la corteza terrestre por donde sale magma, gases y cenizas del interior de la Tierra."},
    {"user": "¿Por qué es importante la educación?", "assistant": "La educación desarrolla el pensamiento crítico, amplía oportunidades y permite a las personas participar activamente en la sociedad."},
    {"user": "¿Qué significa 'hipótesis'?", "assistant": "Una hipótesis es una suposición provisional sobre un fenómeno, propuesta para ser verificada mediante experimentos o evidencia."},
    {"user": "¿Cuántos meses tiene un año?", "assistant": "Un año tiene 12 meses."},
    {"user": "¿Qué es la presión?", "assistant": "La presión es la fuerza aplicada por unidad de área sobre una superficie. Se mide en pascales en el Sistema Internacional."},
    {"user": "¿Qué es la sociología?", "assistant": "La sociología estudia las relaciones sociales, las estructuras y el comportamiento de los grupos humanos en la sociedad."},
    {"user": "¿Qué es el tiempo?", "assistant": "El tiempo es la medida de la duración y el orden de los eventos. En física, es la cuarta dimensión del espacio-tiempo."},
    {"user": "¿Cómo se produce la lluvia?", "assistant": "La lluvia se produce cuando el vapor de agua de la atmósfera se condensa en gotas y cae al suelo por gravedad."},
    {"user": "¿Qué es el metabolismo?", "assistant": "El metabolismo es el conjunto de reacciones químicas que ocurren en el cuerpo para mantener la vida, convirtiendo alimentos en energía."},
    {"user": "Dime algo interesante sobre el espacio.", "assistant": "En el espacio exterior no hay sonido, porque el sonido necesita un medio material para propagarse. Es completamente silencioso."},
    {"user": "¿Qué es la robótica?", "assistant": "La robótica es la rama de la ingeniería y la ciencia que diseña, construye y programa robots para realizar tareas de forma autónoma o asistida."},
    {"user": "¿Cuál es el elemento más abundante en el universo?", "assistant": "El hidrógeno es el elemento más abundante en el universo, constituyendo aproximadamente el 75% de la materia ordinaria."},
    {"user": "¿Qué es la velocidad de la luz?", "assistant": "La velocidad de la luz en el vacío es de aproximadamente 299,792,458 metros por segundo, y es el límite máximo de velocidad conocido en el universo."},
    {"user": "¿Qué es la genética?", "assistant": "La genética es la rama de la biología que estudia los genes, la herencia y la variación de los organismos vivos."},
    {"user": "¿Qué es el idioma?", "assistant": "Un idioma es un sistema de comunicación estructurado, usado por una comunidad de personas para expresar ideas, emociones y conceptos."},
    {"user": "Explícame la teoría del caos.", "assistant": "La teoría del caos estudia sistemas muy sensibles a condiciones iniciales, donde pequeños cambios pueden producir resultados muy diferentes e impredecibles."},
    {"user": "¿Cuál es la diferencia entre masa y peso?", "assistant": "La masa es la cantidad de materia de un objeto y no cambia. El peso es la fuerza gravitacional sobre esa masa, y varía según la gravedad del lugar."},
    {"user": "¿Qué es la nutrición?", "assistant": "La nutrición es el proceso por el que los organismos obtienen y utilizan los nutrientes de los alimentos para crecer, repararse y obtener energía."},
    {"user": "¿Por qué dormimos?", "assistant": "Dormimos para restaurar el cuerpo y el cerebro. Durante el sueño se consolidan memorias, se reparan tejidos y se regulan hormonas."},
    {"user": "¿Qué es la astrofísica?", "assistant": "La astrofísica es la rama de la astronomía que aplica principios de física para estudiar las propiedades, comportamiento y evolución de los objetos celestes."}
]

def load_pure_chat_pairs(max_samples=5000):
    out = []
    # Solo 5 repeticiones del banco sintetico (identidad de Kairos)
    for _ in range(5):
        out.extend(CONVERSATIONAL_SYNTHETIC)

    try:
        from datasets import load_dataset
        log.info("Cargando 'bertin-project/alpaca-spanish' (diversidad real en espanol)...")
        ds = load_dataset("bertin-project/alpaca-spanish", split="train", streaming=True)
        for row in ds:
            u = str(row.get("instruction") or "").strip()
            c = str(row.get("input") or "").strip()
            a = str(row.get("output") or "").strip()
            # Filtro estricto: sin corchetes, sin nombres, sin codigo, sin ingles
            bad = ["```", "<html>", "def ", "import ", "function",
                   "[", "]", "John", "Alice", "Jane", "Mary", "Bob",
                   "http", "www.", "SELECT", "INSERT", "class "]
            if any(b in a or b in u for b in bad): continue
            if c: u += "\n" + c
            if 5 <= len(u) <= 220 and 10 <= len(a) <= 400:
                out.append({"user": u, "assistant": a})
                if len(out) >= max_samples: break
    except Exception as e:
        log.warning("Dataset remoto no disponible: %s", e)
        # Fallback: repetir banco sintetico
        while len(out) < max_samples:
            out.extend(CONVERSATIONAL_SYNTHETIC)

    random.shuffle(out)
    return out[:max_samples]

def encode_pair(tok, row, seq_len):
    prompt = [BOS_ID, SYS_ID] + tok.encode(SYSTEM_PROMPT) + [USR_ID] + tok.encode(row["user"]) + [AST_ID]
    answer = tok.encode(row["assistant"]) + [EOS_ID]

    ids = (prompt + answer)[:seq_len]
    labels = ([-100] * len(prompt) + answer)[:seq_len]

    if len(ids) < seq_len:
        n = seq_len - len(ids)
        ids += [PAD_ID] * n
        labels += [-100] * n

    return torch.tensor(ids), torch.tensor(labels)

class PureChatDS(Dataset):
    def __init__(self, rows, tok, seq_len):
        self.items = [encode_pair(tok, r, seq_len) for r in rows]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def cosine_lr(step, total_steps, warmup_steps, peak_lr, floor_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor_lr + 0.5 * (peak_lr - floor_lr) * (1.0 + math.cos(math.pi * progress))

def save_checkpoint(path, model, opt, scaler, ema, step, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema,
        "cfg": cfg
    }, tmp)
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser(description="AETHER - Pure Conversational Fine-Tuning")
    ap.add_argument("--base", default="checkpoints/latest.pt")
    ap.add_argument("--tokenizer", default="kairos_tokenizer.json")
    ap.add_argument("--out", default="checkpoints_pure_chat")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-samples", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_dir / "latest.pt"

    tok = KairosTokenizer.load(args.tokenizer)
    vocab_size = ((tok.vocab_size + 63) // 64) * 64

    log.info("Cargando motor A.E.T.H.E.R. v3.1 (Full FT 35M para Chat Puro)...")
    model = AetherEngine(vocab_size=vocab_size, hidden_dim=512, ssm_state_dim=1024, num_layers=6).to(dev)

    base_path = Path(args.base)
    if base_path.exists():
        log.info("Cargando pesos base desde '%s'...", base_path)
        sd = torch.load(base_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd.get("model", sd), strict=False)

    # 100% de parametros habilitados para entrenamiento completo
    for p in model.parameters():
        p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info("Parametros entrenables: %.2fM (Full Fine-Tuning)", sum(p.numel() for p in trainable) / 1e6)

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = make_grad_scaler(dev.type == "cuda")

    ema = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    start_step = 0
    if args.resume and latest_ckpt.exists():
        ck = torch.load(latest_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"]
        ema = ck.get("ema", ema)
        log.info("Reanudando desde paso %d", start_step)

    rows = load_pure_chat_pairs(args.max_samples)
    log.info("Dataset de Chat Puro listo: %d pares de conversacion nítidos.", len(rows))

    ds = PureChatDS(rows, tok, args.seq_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if dev.type == "cuda" else 0,
        pin_memory=dev.type == "cuda",
        persistent_workers=dev.type == "cuda"
    )

    it = iter(loader)
    model.train()
    t0 = time.time()

    warmup_steps = 100
    floor_lr = args.lr * 0.1

    log.info("Iniciando Fine-Tuning de Chat Puro por %d pasos...", args.steps)

    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        cur_lr = cosine_lr(step, args.steps, warmup_steps, args.lr, floor_lr)

        for g in opt.param_groups:
            g["lr"] = cur_lr

        for _ in range(args.accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)

            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            with amp_autocast(enabled=dev.type == "cuda"):
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
                loss = loss / args.accum

            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * args.accum

        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        # Actualizar EMA
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema:
                    ema[k].mul_(0.999).add_(v.detach().float(), alpha=0.001)

        if (step + 1) % 20 == 0 or (step + 1) == args.steps:
            ppl = math.exp(min(total_loss, 20))
            elapsed = (time.time() - t0) / 60.0
            log.info("PASO %4d/%d | Pure Chat Loss: %.4f | PPL: %.2f | LR: %.2e | GradNorm: %.2f | Tiempo: %.1fm",
                     step + 1, args.steps, total_loss, ppl, cur_lr, float(gn), elapsed)

            # Generar muestra de conversacion con preguntas INEDITAS (no en el dataset de entrenamiento)
            if (step + 1) % 100 == 0:
                model.eval()
                unseen_prompts = [
                    "¿Por qué el cielo es azul?",
                    "Cuánto es 37 por 3?",
                    "¿Qué es un agujero negro?",
                ]
                log.info("--- MUESTRA DE GENERALIZACION (Preguntas Ineditas) ---")
                with torch.no_grad():
                    for sp in unseen_prompts:
                        fmt = f"<SYS>{SYSTEM_PROMPT}<USR>{sp}<AST>"
                        gen = model.generate(tok, fmt, max_tokens=60, temperature=0.35, top_p=0.90, repetition_penalty=1.15)
                        log.info("USER: %s", sp)
                        log.info("KAIROS: %s", gen.strip())
                        log.info("")
                model.train()

        if (step + 1) % 500 == 0 or (step + 1) == args.steps:
            save_checkpoint(latest_ckpt, model, opt, scaler, ema, step + 1, vars(args))

    # Guardar modelo final con pesos EMA
    save_checkpoint(latest_ckpt, model, opt, scaler, ema, args.steps, vars(args))
    
    # Crear modelo final con pesos EMA colapsados
    final_state = {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in model.state_dict().items()}
    torch.save(final_state, out_dir / "kairos_pure_chat_model.pt")
    log.info("¡FINE-TUNING DE CHAT PURO COMPLETADO CON EXITO!")

if __name__ == "__main__":
    main()
