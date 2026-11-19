#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Funciones compartidas por 02_train_lora.py y 03_merge.py: carga de config_ft, del modelo
y del procesador, preprocesado de imágenes idéntico al de Surya (escalado y recorte de
bloques) y codificación de muestras de entrenamiento. No se ejecuta directamente.
"""

import importlib
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def cargar_config(nombre="config_ft"):
    try:
        return importlib.import_module(nombre)
    except ModuleNotFoundError as e:
        sys.exit(f"ERROR: no se pudo importar '{nombre}': {e}")


def escalar(img, max_size=(3072, 2048), min_size=(1792, 28), grid_size=28):
    width, height = img.size
    if width <= 0 or height <= 0:
        return img
    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1
    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size
    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def recortar(img, caja, pad=4):
    x0 = max(0, int(caja[0]) - pad)
    y0 = max(0, int(caja[1]) - pad)
    x1 = min(img.size[0], int(caja[2]) + pad)
    y1 = min(img.size[1], int(caja[3]) + pad)
    if x1 <= x0 or y1 <= y0:
        return img.crop((0, 0, 1, 1))
    return img.crop((x0, y0, x1, y1))


def cargar_modelo(ruta, device_map=None):
    import torch
    import transformers

    clase = next(
        (getattr(transformers, n) for n in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText")
         if hasattr(transformers, n)),
        None,
    )
    if clase is None:
        sys.exit("ERROR: esta versión de transformers no tiene AutoModelForMultimodalLM ni "
                 "AutoModelForImageTextToText; actualiza con `pip install -U transformers`")
    opciones = {} if device_map is None else {"device_map": device_map}
    try:
        return clase.from_pretrained(ruta, dtype=torch.bfloat16, **opciones)
    except TypeError:
        return clase.from_pretrained(ruta, torch_dtype=torch.bfloat16, **opciones)


def cargar_procesador(ruta, min_pixels=3136, max_pixels=6291456):
    from transformers import AutoProcessor

    procesador = AutoProcessor.from_pretrained(ruta)
    ip = procesador.image_processor
    for nombre, valor in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if hasattr(ip, nombre):
            setattr(ip, nombre, valor)
    tam = getattr(ip, "size", None)
    if isinstance(tam, dict):
        if "shortest_edge" in tam:
            tam["shortest_edge"] = min_pixels
        if "longest_edge" in tam:
            tam["longest_edge"] = max_pixels
    return procesador


def token_fin(procesador, modelo, forzado=None):
    tok = procesador.tokenizer
    if forzado:
        return forzado
    ids = getattr(getattr(modelo, "generation_config", None), "eos_token_id", None)
    if isinstance(ids, (list, tuple)) and ids:
        ids = ids[0]
    if isinstance(ids, int):
        t = tok.convert_ids_to_tokens(ids)
        if t:
            return t
    return tok.eos_token


def codificar(procesador, imagen, prompt, respuesta=None, fin=None):
    import torch

    mensajes = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    texto = procesador.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    enc = procesador(text=[texto], images=[imagen], return_tensors="pt")
    if respuesta is None:
        return enc
    tok = procesador.tokenizer
    ids = tok(respuesta + (fin or tok.eos_token), add_special_tokens=False, return_tensors="pt")["input_ids"]
    n = enc["input_ids"].shape[1]
    for k, v in list(enc.items()):
        if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == n:
            enc[k] = torch.cat([v, torch.zeros(1, ids.shape[1], dtype=v.dtype)], dim=1)
    enc["labels"] = torch.cat([torch.full((1, n), -100, dtype=ids.dtype), ids], dim=1)
    enc["input_ids"] = torch.cat([enc["input_ids"], ids], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return enc


def a_dispositivo(enc, dispositivo):
    import torch

    salida = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            salida[k] = v.to(dispositivo, dtype=torch.bfloat16) if v.is_floating_point() else v.to(dispositivo)
    return salida