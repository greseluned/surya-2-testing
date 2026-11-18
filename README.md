# surya-2-testing

Evaluación y fine-tuning (LoRA) de [Surya 2](https://huggingface.co/datalab-to/surya-ocr-2) sobre un corpus de prensa histórica en español anotado con PAGE XML. Este trabajo se centra en la transcripción automática de prensa histórica en lengua española publicada entre 1850 y 1945 (Asia, España y el Caribe hispánico). El repositorio compara el modelo base con el modelo afinado sobre exactamente las mismas páginas de test, midiendo CER y WER por publicación.

Surya 2 trabaja en dos pasos: primero detecta el layout de la página (regiones, orden de lectura) y después transcribe cada bloque a HTML. Aquí se usa siempre ese modo (layout y OCR por bloque), nunca el OCR de página completa, y la salida es un PAGE XML por imagen.

## Estructura

```
surya-2-testing/
├── corpus//<publicación>/{jpg,page}/           # Imágenes JPG y PAGE XML
├── surya-base/                                 # Baseline de Surya 2
  ├── config.py
  ├── scripts/                                  # 01_pagexml_to_coco.py, 02_batch-process.py
  ├── data/corpus_coco/                         # Dataset COCO generado a partir de split.json
  └── results/                                  # PAGE XML, textos y métricas de la evaluación base
├── surya-ft/                                   # Fine-tuning y evaluación del modelo afinado
  ├── config.py, config_ft.py
  ├── scripts/                                  # 01_build_data.py ... 05_compare_metrics.py, utils.py
  ├── data/sft/                                 # JSONL de entrenamiento (layout y bloques)
  └── results/                                  # PAGE XML, textos y métricas de la evaluación del modelo afinado
├── CITATION.cff
├── split.json                                  # Reparto train/val/test por página (compartido por todo el proyecto)
├── comparacion.csv                             # Comparación final entre el modelo base y el afinado
└── README.md
```

## Corpus y reparto

Doce publicaciones más una miscelánea (solo train), con partición estratificada 80-10-10 por cabecera: 736 páginas de train, 90 de validación y 90 de test. Las publicaciones más representadas son *Semana* (334), *Fémina* (204) e *Hispanidad* (154); el resto tiene entre 5 y 50 ejemplares.

| Publicación | Split |
|---|---|
| *Acción Libertaria* | 80-10-10 |
| *Boletín de la Cámara de Comercio Española de Filipinas* | 80-10-10 |
| *Excelsior* | 80-10-10 |
| *Fémina* | 80-10-10 |
| *Filipinas* | 80-10-10 |
| *Filipinas Ante Europa* | 80-10-10 |
| *Heraldo de la Mujer* | 80-10-10 |
| *Hispanidad* | 80-10-10 |
| *La Malasia* | 80-10-10 |
| *La Vanguardia* | 80-10-10 |
| *Semana* | 80-10-10 |
| *Sorpresas Chicago* | 80-10-10 |
| Miscelánea (*other*) | solo train |

Las publicaciones marcadas como `solo_train` tienen material insuficiente para dividir en splits (< 10 páginas válidas).

`split.json` asigna cada página (por su nombre sin extensión) a `train`, `val` o `test`, con semilla 42. Es la única fuente de verdad del reparto: tanto el dataset COCO de la línea base como los datos de fine-tuning se generan a partir de él, para que ambos modelos se evalúen sobre las mismas páginas.

## Flujo de trabajo

Los comandos se lanzan desde WSL con el entorno virtual activado. Las rutas se configuran en los `config.py` de cada carpeta.

### 1. Línea base (`surya-base`)

```bash
python surya-base/scripts/01_pagexml_to_coco.py
python surya-base/scripts/02_batch-process.py eval \
    --dataset surya-base/data/corpus_coco/test --out surya-base/results
```

`01_pagexml_to_coco.py` convierte los PAGE XML en un dataset COCO siguiendo `split.json` y se detiene si faltan páginas de val o test. `02_batch-process.py eval` procesa el test con Surya, escribe un PAGE XML por página y calcula CER y WER contra las anotaciones de referencia.

### 2. Fine-tuning (`surya-ft`)

```bash
python surya-ft/scripts/01_build_data.py
python surya-ft/scripts/02_train_lora.py
python surya-ft/scripts/03_merge.py
```

- `01_build_data.py` genera, por split, un JSONL de layout (página completa a JSON con etiqueta, caja y número de tokens, en orden de lectura) y otro de bloques (recorte a HTML con la transcripción), con los prompts exactos de Surya.
- `02_train_lora.py` entrena una LoRA mezclando las dos tareas. Evalúa en validación cada cierto número de pasos y guarda `adapter_best`, `adapter_last` y un estado para reanudar con `--resume`.
- `03_merge.py` fusiona la LoRA con el modelo base y guarda un modelo completo que Surya puede servir con vLLM. Con `--probar` comprueba que el layout generado es JSON válido.

Los hiperparámetros están en `config_ft.py`:

```python
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
```

### 3. Evaluación del modelo afinado y comparación

```bash
export SURYA_MODEL_CHECKPOINT=/root/.cache/huggingface/surya-ft
export VLLM_ENABLE_MTP=0
python surya-ft/scripts/04_batch_process.py eval \
    --sft-data surya-ft/data/sft --out surya-ft/results
python surya-ft/scripts/05_compare_metrics.py \
    surya-base/results/metricas_resumen.csv surya-ft/results/metricas_resumen.csv \
    --out comparacion.csv
```

`04_batch_process.py` es la misma evaluación que en la línea base, pero toma las páginas de test de los JSONL de fine-tuning. `05_compare_metrics.py` compara los dos `metricas_resumen.csv` y muestra, por publicación, el CER y el WER micro de cada modelo y su diferencia (negativa significa mejora).

## Resultados

Cada evaluación guarda en su carpeta `results/`:

- `<publicación>/<página>.xml`: PAGE XML predicho.
- `<publicación>/_texto/`: texto de referencia (`.gt.txt`) y predicho (`.pred.txt`) de cada página, para revisar las diferencias.
- `<publicación>/_layout/`: imágenes con el layout dibujado (si se ejecuta con `--images`).
- `metricas_paginas.csv`: CER y WER por página.
- `metricas_resumen.csv`: CER y WER por publicación, total y media de publicaciones (micro y macro).

La comparación entre ambos modelos está en `comparacion.csv`. Las diferencias negativas indican mejora del modelo afinado.

| Publicación | Páginas | CER base | CER afinado | CER dif. | WER base | WER afinado | WER dif. |
|---|---:|---:|---:|---:|---:|---:|---:|
| accion-libertaria | 4 | 0.2959 | 0.2318 | -0.0641 | 0.4033 | 0.2597 | -0.1436 |
| excelsior | 5 | 0.0210 | 0.0701 | 0.0491 | 0.1104 | 0.1004 | -0.0100 |
| filipinas | 4 | 0.0872 | 0.0750 | -0.0122 | 0.2162 | 0.1855 | -0.0307 |
| filipinas-ante-europa | 2 | 0.0395 | 0.0261 | -0.0134 | 0.1125 | 0.0496 | -0.0629 |
| fémina | 20 | 0.1468 | 0.1311 | -0.0157 | 0.2570 | 0.2091 | -0.0479 |
| heraldo-de-la-mujer | 2 | 0.7828 | 0.7827 | -0.0001 | 0.9501 | 0.9501 | 0.0000 |
| hispanidad | 15 | 0.0465 | 0.0491 | 0.0026 | 0.1057 | 0.0609 | -0.0448 |
| la-malasia | 2 | 0.0517 | 0.0357 | -0.0160 | 0.1514 | 0.0569 | -0.0945 |
| la-vanguardia | 1 | 0.2886 | 0.5357 | 0.2471 | 0.4157 | 0.5817 | 0.1660 |
| semana | 33 | 0.0648 | 0.0558 | -0.0090 | 0.1527 | 0.0884 | -0.0643 |
| sorpresas-chicago | 2 | 0.0267 | 0.0191 | -0.0076 | 0.1114 | 0.0833 | -0.0281 |
| **TOTAL** | 90 | 0.1306 | 0.1295 | -0.0011 | 0.2247 | 0.1709 | -0.0538 |
| **MEDIA_PUBLICACIONES** | 90 | 0.1683 | 0.1829 | 0.0146 | 0.2715 | 0.2387 | -0.0328 |

CER y WER micro.


## Requisitos

Python con `transformers`, `peft`, `torch`, `Pillow`, `rapidfuzz` y `surya-ocr`, y Docker para servir Surya con vLLM.

## Hardware

Todos los experimentos (entrenamiento, inferencia y evaluación) se ejecutaron en un portátil **MSI Vector 16 HX AI A2XW** con procesador **Intel Core Ultra 9** y GPU **NVIDIA GeForce RTX 5090 Laptop**, bajo Ubuntu-22.04.

## Licencia
 
El código, la documentación, las anotaciones y los resultados de este repositorio se distribuyen bajo la licencia [Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
 
Surya conserva sus propias licencias: el código, [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), y los pesos del modelo, [AI Pubs Open RAIL-M (modificada)](https://huggingface.co/datalab-to/chandra-ocr-2/blob/main/LICENSE). Esta licencia no las sustituye ni las modifica.

## Autoría

```text
Nuria Lucia Pirvu Curtusan
```

## Financiación

Trabajo desarrollado en el marco del proyecto GRESEL-UNED «Narrativas poscoloniales en periódicos en español de Asia, España y el Caribe hispánico» (PID2023-151280OB-C22), financiado por el Ministerio de Ciencia e Innovación / AEI.

## Referencias

- Garcia-Serrano, A. M., López Adrados, H., & Zabala Ortiz, B. (2026). Prensa histórica: Gold Standard con 9 números de "Acción Libertaria" (V1) (Version V1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21076010
- Garcia-Serrano, A. M., Torrente Gómez de la Venta, C., Martín Piquero, A., López Adrados, H., Linda Gaia, Z., & Alonso Aparicio, S. (2026). Gold Standard de "Semana, 1949" prensa histórica (V1) (Version V1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21245557
- Paruchuri, V., & Datalab Team. (2025). Surya: A lightweight framework for analyzing documents and PDFs at scale (Version 1.0.0) [Software]. https://github.com/datalab-to/surya
