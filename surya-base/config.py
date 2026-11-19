#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Configuración del pipeline de layout con Surya 2: rutas, publicaciones,
split, clases y mapa de etiquetas.

Uso:
    Editar solo este fichero.
    python config.py    # muestra las rutas resueltas y los comandos del flujo
"""

import json
import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════════════════════

import re
import sys

def _ruta(s: str) -> Path:
    """Convierte 'C:/...' en '/mnt/c/...' cuando se ejecuta en WSL/Linux."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m and sys.platform != "win32":
        s = f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return Path(s)

SURYA_ROOT = _ruta(os.environ.get("SURYA_ROOT", "C:/Users/Uned/Documents/ocr/surya"))

# Corpus de imágenes + PAGE-XML. Estructura esperada:
#   CORPUS_DIR/<publicacion>/jpg/<stem>.jpg
#   CORPUS_DIR/<publicacion>/page/<stem>.xml
CORPUS_DIR  = SURYA_ROOT / "corpus"
CORPUS_NAME = "corpus"

# Dataset COCO que genera 00b_pagexml_to_coco.py:
#   COCO_DIR/{train,val,test}/images/  y  COCO_DIR/{train,val,test}/instances.json
COCO_DIR = SURYA_ROOT / "data" / f"{CORPUS_NAME}_coco"

RUNS_DIR       = SURYA_ROOT / "runs"
SPLIT_JSON     = SURYA_ROOT / "split.json"
LABEL_MAP_FILE = SURYA_ROOT / "label_map.json"

# Opcional: carpeta del dataset YOLO ya generado, solo para verificar que el split
# de COCO coincide con el suyo (00b --check-yolo). None = no verificar.
YOLO_DATA_DIR = None

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICACIONES DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════
# 'estratificar' → 80-10-10, 'solo_train' → todo a train.
# Mantén esta lista sincronizada con la de tu pipeline olmOCR.

PUBLICACIONES = {
    "accion-libertaria":     "estratificar",
    "boletín-de-la-cámara":  "estratificar",
    "excelsior":             "estratificar",
    "fémina":                "estratificar",
    "filipinas":             "estratificar",
    "filipinas-ante-europa": "estratificar",
    "heraldo-de-la-mujer":   "estratificar",
    "hispanidad":            "estratificar",
    "la-malasia":            "estratificar",
    "la-vanguardia":         "estratificar",
    "semana":                "estratificar",
    "sorpresas-chicago":     "estratificar",
    "other":                 "solo_train",
}

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT — deben coincidir exactamente con olmOCR para que train/val/test sean
# los mismos conjuntos de páginas. NO los cambies.
# ══════════════════════════════════════════════════════════════════════════════

SPLIT_TRAIN       = 0.80
SPLIT_VALIDATION  = 0.10
SPLIT_TEST        = 0.10
SPLIT_SEED        = 42
SPLIT_MIN_PAGINAS = 10

# ══════════════════════════════════════════════════════════════════════════════
# CLASES DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

# 'tag'  → text / graphic / unknown
# 'fino' → TextRegion dividido por su atributo type (paragraph, heading, header...)
#          y resto de regiones por su tag (image, separator, table, advert...)
MODO_CLASES       = "fino"
MIN_PAGINAS_CLASE = 3          # excluye clases presentes en menos de N páginas

# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

ENGINE           = "layout"    # "layout" (VLM) | "fast" (FastLayoutPredictor) | "fullpage" (layout + OCR)
CHECKPOINT       = None        # None = modelo por defecto de Surya; o ruta a un checkpoint propio
EVAL_BATCH       = 4
VIZ_PAGES        = 20          # páginas con las cajas dibujadas
RESIZE_LONG_SIDE = 0           # reduce el lado largo a N px antes de inferir (0 = no)

# ══════════════════════════════════════════════════════════════════════════════
# TRADUCCIÓN DE ETIQUETAS: Surya 2 → clases de TU dataset
# ══════════════════════════════════════════════════════════════════════════════
# Los nombres de la derecha deben existir en COCO_DIR/classes.json; los que no, se
# descartan al evaluar (eval_surya_layout.py cuenta cuántas). Revisa «Etiquetas
# devueltas» al final de la evaluación por si tu versión usa otras.

_LABEL_MAP_TAG = {
    "Text": "text", "SectionHeader": "text", "PageHeader": "text", "PageFooter": "text",
    "Caption": "text", "Footnote": "text", "ListGroup": "text", "Equation": "text",
    "Form": "text", "Code": "text", "TableOfContents": "text",
    "Picture": "graphic", "Figure": "graphic", "Diagram": "graphic", "Table": "graphic",
}
_LABEL_MAP_FINO = {
    "Text": "paragraph", "ListGroup": "paragraph", "TableOfContents": "paragraph",
    "SectionHeader": "heading",
    "PageHeader": "header", "PageFooter": "footer",
    "Caption": "caption", "Footnote": "footnote",
    "Picture": "image", "Figure": "image", "Diagram": "image", "Table": "table",
}


def label_map_surya() -> dict:
    """Mapa etiquetas Surya → clases del dataset, filtrado por las que existan en classes.json."""
    mapa = dict(_LABEL_MAP_FINO if MODO_CLASES == "fino" else _LABEL_MAP_TAG)
    classes_json = COCO_DIR / "classes.json"
    if classes_json.exists():
        existentes = set(json.loads(classes_json.read_text(encoding="utf-8")))
        mapa = {k: v for k, v in mapa.items() if v in existentes}
    return mapa


def escribir_label_map() -> Path:
    LABEL_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_FILE.write_text(
        json.dumps(label_map_surya(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return LABEL_MAP_FILE


# ══════════════════════════════════════════════════════════════════════════════
# Ayuda: python config_surya.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = lambda x: str(x).replace("\\", "/")
    print(f"Raíz proyecto : {p(SURYA_ROOT)}")
    print(f"Corpus        : {p(CORPUS_DIR)}  {'(existe)' if CORPUS_DIR.exists() else '(NO EXISTE)'}")
    print(f"COCO          : {p(COCO_DIR)}")
    print(f"Modo de clases: {MODO_CLASES}\n")

    print("# 1. Generar el dataset COCO (con verificación del split si hay dataset YOLO)")
    check = f" --check-yolo {p(YOLO_DATA_DIR)}" if YOLO_DATA_DIR else ""
    print(f"python 01_pagexml_to_coco.py --dry-run{check}")
    print("python 01_pagexml_to_coco.py\n")
    print("# 2. Escribir label_map.json (tras el paso 1: filtra con classes.json)")
    print('python -c "import config as c; print(c.escribir_label_map())"\n')

    ckpt = f" --checkpoint {p(CHECKPOINT)}" if CHECKPOINT else ""
    resize = f" --resize-long-side {RESIZE_LONG_SIDE}" if RESIZE_LONG_SIDE else ""
    print("# 3. Evaluar en test (compara los tres motores)")
    for eng in ("layout", "fast", "fullpage"):
        print(
            f"python 02_batch-process.py --images {p(COCO_DIR)}/test/images "
            f"--annotations {p(COCO_DIR)}/test/instances.json --label-map {p(LABEL_MAP_FILE)} "
            f"--engine {eng} --batch-size {EVAL_BATCH} --viz {VIZ_PAGES}{resize}{ckpt} "
            f"--out {p(RUNS_DIR)}/surya_{eng}"
        )