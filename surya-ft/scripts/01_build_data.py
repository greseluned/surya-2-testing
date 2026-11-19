#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_data.py

Convierte los PAGE XML del corpus en los datos de fine-tuning, siguiendo split.json.
Por cada split genera dos JSONL con los prompts exactos de Surya:
    <split>_layout.jsonl   página completa -> JSON con etiqueta, caja y nº de tokens, en orden de lectura
    <split>_block.jsonl    recorte de un bloque -> HTML con su transcripción
También escribe sft_report.json con el resumen y los tipos de región descartados.

Uso:
    python 01_build_data.py --dry-run
    python 01_build_data.py --tokenizer datalab-to/surya-ocr-2

Opciones: --faltantes train, --publicaciones, --split-json, --output
"""

import argparse
import html
import json
import math
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from utils import cargar_config


def nfc(s):
    return unicodedata.normalize("NFC", s)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def leer_puntos(s):
    pts = []
    for p in s.strip().split():
        if "," not in p:
            continue
        try:
            x, y = p.split(",")[:2]
            pts.append((float(x), float(y)))
        except ValueError:
            continue
    return pts


def tipo_region(elem, tag):
    if tag == "TextRegion":
        t = elem.get("type")
        if t:
            return t.strip().lower()
        m = re.search(r"structure\s*\{[^}]*type:([^;}]+)", elem.get("custom", "") or "")
        return m.group(1).strip().lower() if m else "text"
    return tag[: -len("Region")].lower()


def idx(ch):
    try:
        return int(ch.get("index", 0))
    except (ValueError, TypeError):
        return 0


def orden_lectura(root):
    ro = next((e for e in root.iter() if local(e.tag) == "ReadingOrder"), None)
    if ro is None:
        return {}
    refs = []

    def visitar(grupo):
        indexados, planos = [], []
        for ch in grupo:
            t = local(ch.tag)
            if t in ("RegionRefIndexed", "OrderedGroupIndexed", "UnorderedGroupIndexed"):
                indexados.append(ch)
            elif t in ("RegionRef", "OrderedGroup", "UnorderedGroup"):
                planos.append(ch)
        for ch in sorted(indexados, key=idx) + planos:
            if local(ch.tag).startswith("RegionRef"):
                if ch.get("regionRef"):
                    refs.append(ch.get("regionRef"))
            else:
                visitar(ch)

    visitar(ro)
    return {r: i for i, r in enumerate(refs)}


def texto_region(elem):
    lineas = []
    for tl in elem:
        if local(tl.tag) != "TextLine":
            continue
        for te in tl:
            if local(te.tag) == "TextEquiv":
                for u in te:
                    if local(u.tag) == "Unicode" and u.text and u.text.strip():
                        lineas.append(" ".join(u.text.split()))
                break
    if lineas:
        return " ".join(lineas)
    for te in elem:
        if local(te.tag) == "TextEquiv":
            for u in te:
                if local(u.tag) == "Unicode" and u.text:
                    return " ".join(u.text.split())
            break
    return ""


def leer_pagina(ruta_xml):
    try:
        root = ET.parse(ruta_xml).getroot()
    except ET.ParseError:
        return None
    page = next((e for e in root.iter() if local(e.tag) == "Page"), None)
    if page is None:
        return None
    try:
        w, h = int(page.get("imageWidth", 0)), int(page.get("imageHeight", 0))
    except (ValueError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None
    orden = orden_lectura(root)
    regiones = []
    for elem in root.iter():
        tag = local(elem.tag)
        if not tag.endswith("Region") or tag == "Region":
            continue
        coords = next((c for c in elem if local(c.tag) == "Coords"), None)
        if coords is None:
            continue
        pts = leer_puntos(coords.get("points", ""))
        if len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            continue
        regiones.append(
            {
                "tipo": tipo_region(elem, tag),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "texto": texto_region(elem) if tag == "TextRegion" else "",
                "orden": orden.get(elem.get("id")),
            }
        )
    return {"w": w, "h": h, "regiones": regiones}


def ordenar(regiones):
    con = sorted((r for r in regiones if r["orden"] is not None), key=lambda r: r["orden"])
    sin = [r for r in regiones if r["orden"] is None]
    return con + sin


def crear_contador(nombre_tokenizer, chars_por_token, redondeo):
    if nombre_tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(nombre_tokenizer)
        medir = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    else:
        medir = lambda s: len(s) / chars_por_token
    return lambda s: min(10000, max(redondeo, int(round(medir(s) / redondeo)) * redondeo))


def normalizar(v, tam):
    return max(0, min(1000, round(v / tam * 1000)))


def construir_pagina(par, cfg, corpus, contar):
    pagina = leer_pagina(par["xml"])
    if pagina is None:
        return None
    with Image.open(par["jpg"]) as im:
        ancho, alto = im.size
    sx, sy = ancho / pagina["w"], alto / pagina["h"]
    rel = par["jpg"].relative_to(corpus).as_posix()
    entradas, bloques, descartes = [], [], Counter()

    for r in ordenar(pagina["regiones"]):
        etiqueta = cfg.REGION_A_LABEL.get(r["tipo"])
        if etiqueta is None:
            descartes[r["tipo"]] += 1
            continue
        x0, y0, x1, y1 = r["bbox"][0] * sx, r["bbox"][1] * sy, r["bbox"][2] * sx, r["bbox"][3] * sy
        n = [normalizar(x0, ancho), normalizar(y0, alto), normalizar(x1, ancho), normalizar(y1, alto)]
        if n[2] <= n[0] or n[3] <= n[1]:
            descartes["caja_diminuta"] += 1
            continue
        cuenta = 0
        if etiqueta not in cfg.SIN_OCR and r["texto"]:
            etq = cfg.ETIQUETA_HTML.get(etiqueta, "p")
            respuesta = f"<{etq}>{html.escape(r['texto'], quote=False)}</{etq}>"
            cuenta = contar(respuesta)
            caja = [int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))]
            if caja[2] - caja[0] >= cfg.MIN_LADO_BLOQUE and caja[3] - caja[1] >= cfg.MIN_LADO_BLOQUE:
                bloques.append(
                    {
                        "id": f"{par['pub']}/{par['stem']}#{len(bloques)}",
                        "task": "block",
                        "image": rel,
                        "crop": caja,
                        "prompt": cfg.BLOCK_PROMPT,
                        "answer": respuesta,
                        "label": etiqueta,
                        "publication": par["pub"],
                    }
                )
        entradas.append({"label": etiqueta, "bbox": " ".join(map(str, n)), "count": cuenta})

    if not entradas:
        return None
    if len(entradas) > cfg.MAX_BLOQUES_PAGINA:
        descartes["pagina_con_demasiados_bloques"] += 1
        return {"capa": None, "bloques": [], "descartes": descartes}
    capa = {
        "id": f"{par['pub']}/{par['stem']}#layout",
        "task": "layout",
        "image": rel,
        "crop": None,
        "prompt": cfg.LAYOUT_PROMPT,
        "answer": json.dumps(entradas, ensure_ascii=False),
        "publication": par["pub"],
    }
    return {"capa": capa, "bloques": bloques, "descartes": descartes}


def escanear(publicaciones, corpus):
    pares = []
    for nombre in publicaciones:
        dir_jpg, dir_page = corpus / nombre / "jpg", corpus / nombre / "page"
        if not dir_jpg.exists() or not dir_page.exists():
            print(f"  [AVISO] {nombre}: carpetas jpg/ o page/ no encontradas, saltando.")
            continue
        n = 0
        for jpg in sorted(p for p in dir_jpg.glob("*") if p.suffix.lower() in (".jpg", ".jpeg")):
            xml = dir_page / (jpg.stem + ".xml")
            if xml.exists():
                pares.append({"pub": nombre, "stem": jpg.stem, "jpg": jpg, "xml": xml})
                n += 1
        print(f"  {nombre:<28}  {n:>4} pares")
    return pares


def cargar_split(ruta):
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    for stem, s in datos.items():
        if s not in ("train", "val", "test"):
            sys.exit(f"ERROR: valor de split inválido en {ruta}: {stem!r} -> {s!r}")
    return {nfc(k): v for k, v in datos.items()}


def asignar(pares, split, faltantes):
    res = {"train": [], "val": [], "test": []}
    sin_split = []
    for par in pares:
        s = split.get(nfc(par["stem"]))
        if s is None:
            sin_split.append(par["stem"])
            if faltantes == "train":
                res["train"].append(par)
            continue
        res[s].append(par)
    if sin_split:
        destino = "enviadas a train" if faltantes == "train" else "omitidas"
        print(f"  [AVISO] {len(sin_split)} páginas del corpus no están en el JSON ({destino}): "
              f"{', '.join(sin_split[:10])}{' …' if len(sin_split) > 10 else ''}")
    return res


def escribir_jsonl(ruta, registros):
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="PAGE XML + split.json -> JSONL de fine-tuning de Surya 2")
    ap.add_argument("--config", default="config_ft")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--output", default=None, help="default: config.SFT_DIR")
    ap.add_argument("--split-json", default=None, help="default: config.SPLIT_JSON")
    ap.add_argument("--faltantes", choices=["omitir", "train"], default="omitir")
    ap.add_argument("--publicaciones", nargs="+", default=None)
    ap.add_argument("--tokenizer", default=None, help="tokenizer para contar tokens reales (p. ej. datalab-to/surya-ocr-2)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    corpus = Path(args.corpus) if args.corpus else Path(cfg.CORPUS_DIR)
    salida = Path(args.output) if args.output else Path(cfg.SFT_DIR)
    ruta_split = Path(args.split_json) if args.split_json else Path(cfg.SPLIT_JSON)
    if not corpus.exists():
        sys.exit(f"ERROR: directorio corpus no encontrado: {corpus}")
    if not ruta_split.exists():
        sys.exit(f"ERROR: no existe el JSON de split: {ruta_split}")

    publicaciones = cfg.PUBLICACIONES
    if args.publicaciones:
        publicaciones = {k: v for k, v in publicaciones.items() if k in args.publicaciones}

    print(f"Escaneando corpus en {corpus} …")
    pares = escanear(publicaciones, corpus)
    if not pares:
        sys.exit("ERROR: no se encontraron pares JPG+XML.")

    print(f"\nAplicando split desde {ruta_split} …")
    splits = asignar(pares, cargar_split(ruta_split), args.faltantes)

    contar = crear_contador(args.tokenizer, cfg.CHARS_PER_TOKEN, cfg.COUNT_ROUND)
    modo_conteo = f"tokenizer {args.tokenizer}" if args.tokenizer else f"estimación {cfg.CHARS_PER_TOKEN} caracteres/token"
    print(f"Conteo de tokens: {modo_conteo}\n")

    datos = {s: {"layout": [], "block": []} for s in splits}
    descartes, etiquetas, paginas_omitidas = Counter(), defaultdict(Counter), []
    for s, lista in splits.items():
        for par in lista:
            res = construir_pagina(par, cfg, corpus, contar)
            if res is None:
                paginas_omitidas.append(par["stem"])
                continue
            descartes.update(res["descartes"])
            if res["capa"] is None:
                paginas_omitidas.append(par["stem"])
                continue
            datos[s]["layout"].append(res["capa"])
            datos[s]["block"].extend(res["bloques"])
            for e in json.loads(res["capa"]["answer"]):
                etiquetas[s][e["label"]] += 1

    informe = {
        "split_json": str(ruta_split),
        "conteo_tokens": modo_conteo,
        "paginas_omitidas": paginas_omitidas,
        "tipos_descartados": dict(descartes),
        "splits": {
            s: {
                "paginas": len(d["layout"]),
                "bloques_ocr": len(d["block"]),
                "etiquetas_layout": dict(etiquetas[s]),
            }
            for s, d in datos.items()
        },
    }

    for s, d in datos.items():
        print(f"  {s:<6} {len(d['layout']):>5} páginas   {len(d['block']):>6} bloques con transcripción")
    if descartes:
        print(f"\n  Regiones descartadas por tipo: {dict(descartes)}")
    if paginas_omitidas:
        print(f"  Páginas omitidas ({len(paginas_omitidas)}): {', '.join(paginas_omitidas[:10])}"
              f"{' …' if len(paginas_omitidas) > 10 else ''}")

    if args.dry_run:
        print("\n[DRY-RUN] No se ha escrito nada.")
        return

    salida.mkdir(parents=True, exist_ok=True)
    for s, d in datos.items():
        escribir_jsonl(salida / f"{s}_layout.jsonl", d["layout"])
        escribir_jsonl(salida / f"{s}_block.jsonl", d["block"])
    (salida / "sft_report.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDatos escritos en {salida}")


if __name__ == "__main__":
    main()