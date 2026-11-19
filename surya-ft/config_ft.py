#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_ft.py

Configuración del fine-tuning LoRA de Surya 2: rutas, modelo base, traducción de las
regiones PAGE XML a etiquetas de Surya e hiperparámetros de entrenamiento. Reutiliza
corpus, publicaciones y SPLIT_JSON de config.py. Es el único fichero que hay que editar.

Uso:
    python config_ft.py    # muestra las rutas resueltas y los comandos del flujo
"""

from pathlib import Path

import config as base

# ══════════════════════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════════════════════

SURYA_ROOT = base.SURYA_ROOT
CORPUS_DIR = base.CORPUS_DIR
PUBLICACIONES = base.PUBLICACIONES
SPLIT_JSON = getattr(base, "SPLIT_JSON", SURYA_ROOT / "split.json")

SFT_DIR = SURYA_ROOT / "surya-ft" / "data" / "sft"
RUNS_DIR = SURYA_ROOT / "surya-ft" / "runs" / "finetune"
RUN_NAME = "lora_v1"

HF_CACHE = Path.home() / ".cache" / "huggingface"
MODEL_OUT_NAME = "surya-ft"

# Debe coincidir con el modelo que usa tu instalación de Surya:
#   python -c "from surya.settings import settings; print(settings.SURYA_MODEL_CHECKPOINT)"
BASE_MODEL = "datalab-to/surya-ocr-2"

# ══════════════════════════════════════════════════════════════════════════════
# CONTRATO DE SURYA 2 (no cambiar: son los prompts y el preprocesado de entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

LAYOUT_PROMPT = (
    "Output the layout of this image as JSON. Each entry is a dict with "
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)
BLOCK_PROMPT = "OCR this block image to HTML."

SCALE_MAX = (3072, 2048)
SCALE_MIN = (1792, 28)
GRID = 28
BLOCK_PAD = 4
MIN_PIXELS = 3136
MAX_PIXELS = 6291456
MAX_BLOQUES_PAGINA = 200

# ══════════════════════════════════════════════════════════════════════════════
# TRADUCCIÓN DE TU ANOTACIÓN PAGE XML -> ETIQUETAS DE SURYA
# ══════════════════════════════════════════════════════════════════════════════
# Clave: atributo type de TextRegion en minúsculas, o el tag de la región sin
# "Region" en minúsculas (image, table, advert...). Los tipos que no estén aquí
# (separator, noise, unknown...) se descartan y se cuentan en el informe.
# Surya solo admite estas 19 etiquetas: Caption, Footnote, Equation-Block, List-Group,
# Page-Header, Page-Footer, Image, Section-Header, Table, Text, Complex-Block,
# Code-Block, Form, Table-Of-Contents, Figure, Chemical-Block, Diagram, Bibliography,
# Blank-Page.

REGION_A_LABEL = {
    "paragraph": "Text", "text": "Text", "other": "Text", "marginalia": "Text",
    "catch-word": "Text", "drop-capital": "Text", "signature-mark": "Text",
    "list-label": "List-Group", "toc-entry": "Table-Of-Contents",
    "heading": "Section-Header",
    "header": "Page-Header", "footer": "Page-Footer", "page-number": "Page-Footer",
    "caption": "Caption", "credit": "Caption",
    "footnote": "Footnote", "footnote-continued": "Footnote", "endnote": "Footnote",
    "image": "Image", "graphic": "Figure", "linedrawing": "Figure", "chart": "Figure",
    "table": "Table", "maths": "Equation-Block",
    "advert": "Complex-Block",
}

SIN_OCR = {"Figure", "Image", "Diagram", "Blank-Page"}

ETIQUETA_HTML = {"Section-Header": "h2"}

# ══════════════════════════════════════════════════════════════════════════════
# CONTEO DE TOKENS (campo "count" del layout)
# ══════════════════════════════════════════════════════════════════════════════
# Con --tokenizer se cuentan los tokens reales del HTML de cada bloque; sin él se
# estima con CHARS_PER_TOKEN. El resultado se redondea a COUNT_ROUND (las salidas del
# modelo base que has visto van en múltiplos de 10).

CHARS_PER_TOKEN = 3.5
COUNT_ROUND = 10
MIN_LADO_BLOQUE = 12

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LR = 1e-4
EPOCHS = 2
GRAD_ACCUM = 16
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.0
BLOCKS_PER_LAYOUT = 4
EVAL_EVERY = 50
EVAL_SAMPLES = 40
MAX_SEQ_TOKENS = 16384
LOG_EVERY = 5
SEED = 42
END_TOKEN = None


# ══════════════════════════════════════════════════════════════════════════════
# Ayuda: python config_surya_ft.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = lambda x: str(x).replace("\\", "/")
    print(f"Raíz proyecto : {p(SURYA_ROOT)}")
    print(f"Corpus        : {p(CORPUS_DIR)}  {'(existe)' if CORPUS_DIR.exists() else '(NO EXISTE)'}")
    print(f"Split JSON    : {p(SPLIT_JSON)}  {'(existe)' if SPLIT_JSON.exists() else '(NO EXISTE)'}")
    print(f"Datos SFT     : {p(SFT_DIR)}")
    print(f"Run           : {p(RUNS_DIR / RUN_NAME)}")
    print(f"Modelo base   : {BASE_MODEL}\n")

    print("# 1. Datos de entrenamiento (layout + bloques) desde PAGE XML y split.json")
    print("python 01_build_data.py --dry-run")
    print(f"python 01_build_data.py --tokenizer {BASE_MODEL}\n")
    print("# 2. Parar el servidor de Surya (libera la VRAM) y entrenar")
    print("docker stop $(docker ps -q --filter name=surya-vllm)")
    print("python 02_train_surya_lora.py --smoke")
    print("python 02_train_surya_lora.py\n")
    print("# 3. Fusionar la LoRA con el modelo base y comprobar que genera JSON válido")
    print("python 03_merge_lora.py --probar <pagina.jpg>\n")
    print("# 4. Evaluar el modelo afinado con Surya y comparar")
    print(f"export SURYA_MODEL_CHECKPOINT=/root/.cache/huggingface/{MODEL_OUT_NAME}")
    print("export VLLM_ENABLE_MTP=0")
    print("python 04_batch_process.py eval --dataset <COCO>/test --out <resultados>/eval_ft")
    print("python 05_compare_metrics.py <resultados>/eval_base/metricas_resumen.csv "
          "<resultados>/eval_ft/metricas_resumen.csv")