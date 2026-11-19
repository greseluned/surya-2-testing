#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_merge.py

Fusiona la LoRA entrenada con el modelo base y guarda un modelo completo que Surya puede
servir con vLLM, en HF_CACHE/MODEL_OUT_NAME (~/.cache/huggingface/surya-ft). Si el modelo
no incluye la cabeza MTP, deja a 0 sus contadores en config.json.

Uso:
    python 03_merge.py
    python 03_merge.py --probar corpus/semana/jpg/0010_Semana-0010.jpg   # comprueba que el layout sale en JSON válido

Opciones: --adapter (def. adapter_best), --out

Después, para evaluar:
    export SURYA_MODEL_CHECKPOINT=/root/.cache/huggingface/surya-ft
    export VLLM_ENABLE_MTP=0
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from utils import (
    a_dispositivo, cargar_config, cargar_modelo, cargar_procesador, codificar, escalar,
)


def claves_mtp(destino):
    from safetensors import safe_open

    claves = []
    for f in sorted(destino.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as g:
            claves += [k for k in g.keys() if "mtp" in k.lower()]
    return claves


def limpiar_config(obj):
    cambios = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("num_nextn_predict_layers", "mtp_num_hidden_layers") and isinstance(v, int) and v > 0:
                obj[k] = 0
                cambios += 1
            else:
                cambios += limpiar_config(v)
    elif isinstance(obj, list):
        for v in obj:
            cambios += limpiar_config(v)
    return cambios


def ajustar_mtp(destino):
    mtp = claves_mtp(destino)
    if mtp:
        print(f"El modelo guardado contiene {len(mtp)} tensores MTP; el config.json no se toca.")
        return
    ruta = destino / "config.json"
    config = json.loads(ruta.read_text(encoding="utf-8"))
    cambios = limpiar_config(config)
    if cambios:
        ruta.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sin cabeza MTP en los pesos: {cambios} contador(es) de MTP puestos a 0 en config.json.")


def probar(modelo, procesador, ruta_imagen, cfg, max_nuevos):
    import torch

    modelo.to("cuda")
    modelo.eval()
    imagen = escalar(Image.open(ruta_imagen).convert("RGB"), cfg.SCALE_MAX, cfg.SCALE_MIN, cfg.GRID)
    enc = a_dispositivo(codificar(procesador, imagen, cfg.LAYOUT_PROMPT), "cuda")
    n = enc["input_ids"].shape[1]
    with torch.no_grad():
        salida = modelo.generate(**enc, max_new_tokens=max_nuevos, do_sample=False)
    texto = procesador.tokenizer.decode(salida[0, n:], skip_special_tokens=True)
    print(f"\nSalida del layout ({salida.shape[1] - n} tokens):\n{texto[:800]}{' …' if len(texto) > 800 else ''}")
    try:
        bloques = json.loads(texto)
        print(f"\nJSON válido con {len(bloques)} bloques.")
    except json.JSONDecodeError as e:
        print(f"\nAVISO: la salida no es JSON válido ({e}). Revisa el token de fin y el entrenamiento.")


def main():
    ap = argparse.ArgumentParser(description="Fusiona LoRA + modelo base de Surya 2")
    ap.add_argument("--config", default="config_ft")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="default: RUNS_DIR/RUN_NAME/adapter_best")
    ap.add_argument("--out", default=None, help="default: HF_CACHE/MODEL_OUT_NAME")
    ap.add_argument("--probar", default=None, help="imagen de una página para probar el layout del modelo fusionado")
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    modelo_base = args.model or cfg.BASE_MODEL
    adaptador = Path(args.adapter) if args.adapter else Path(cfg.RUNS_DIR) / cfg.RUN_NAME / "adapter_best"
    destino = Path(args.out) if args.out else Path(cfg.HF_CACHE) / cfg.MODEL_OUT_NAME
    if not adaptador.exists():
        sys.exit(f"ERROR: no existe el adaptador {adaptador}")

    from peft import PeftModel

    procesador = cargar_procesador(modelo_base, cfg.MIN_PIXELS, cfg.MAX_PIXELS)
    base = cargar_modelo(modelo_base)
    fusionado = PeftModel.from_pretrained(base, str(adaptador)).merge_and_unload()

    destino.mkdir(parents=True, exist_ok=True)
    fusionado.save_pretrained(destino, safe_serialization=True)
    procesador.save_pretrained(destino)
    ajustar_mtp(destino)
    print(f"\nModelo fusionado guardado en {destino}")
    print(f"Para Surya: export SURYA_MODEL_CHECKPOINT=/root/.cache/huggingface/{destino.name}")

    if args.probar:
        probar(fusionado, procesador, args.probar, cfg, args.max_new_tokens)


if __name__ == "__main__":
    main()