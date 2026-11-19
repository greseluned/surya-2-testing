#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_pagexml_to_coco.py

Convierte los PAGE-XML del corpus a un dataset COCO (train/val/test) siguiendo
el split de split.json (config.SPLIT_JSON o --split-json). Cada anotación incluye
bbox, polígono y orden de lectura si el XML lo trae.

Uso (desde WSL, con el entorno activado):
    python surya/scripts/01_pagexml_to_coco.py --dry-run
    python surya/scripts/01_pagexml_to_coco.py

Opciones: --modo-clases {tag,fino}, --split-json RUTA, --split-aleatorio (ignora split.json),
          --publicaciones, --symlinks, --check-yolo DIR
Requisitos: Pillow

Regenera por completo las carpetas images/ de cada split. Si falta algún JPG o XML
de val/test, se detiene sin escribir nada.
"""

import argparse
import importlib
import json
import random
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def cargar_config(nombre: str):
    for n in (nombre,):
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    sys.exit(f"ERROR: no se pudo importar el módulo de configuración '{nombre}'")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSEO PAGE-XML
# ══════════════════════════════════════════════════════════════════════════════
TAG_A_CLASE_TAG = {
    "TextRegion": "text",
    "ImageRegion": "graphic",
    "TableRegion": "graphic",
    "GraphicRegion": "graphic",
    "UnknownRegion": "unknown",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _leer_puntos(s: str) -> list[tuple[float, float]]:
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


def _tipo_texto(elem) -> str | None:
    t = elem.get("type")
    if t:
        return t.strip().lower()
    m = re.search(r"structure\s*\{[^}]*type:([^;}]+)", elem.get("custom", "") or "")
    return m.group(1).strip().lower() if m else None


def _clase_region(elem, tag: str, modo: str) -> str | None:
    if modo == "tag":
        return TAG_A_CLASE_TAG.get(tag)
    if tag == "TextRegion":
        return _tipo_texto(elem) or "text"
    if tag.endswith("Region") and tag != "Region":
        return tag[: -len("Region")].lower()
    return None


def _idx(ch) -> int:
    try:
        return int(ch.get("index", 0))
    except (ValueError, TypeError):
        return 0


def _orden_lectura(root) -> dict[str, int]:
    """region_id -> posición de lectura (0-based), aplanando grupos anidados."""
    ro = next((e for e in root.iter() if _local(e.tag) == "ReadingOrder"), None)
    if ro is None:
        return {}
    refs: list[str] = []

    def visit(group):
        indexed, plain = [], []
        for ch in group:
            t = _local(ch.tag)
            if t in ("RegionRefIndexed", "OrderedGroupIndexed", "UnorderedGroupIndexed"):
                indexed.append(ch)
            elif t in ("RegionRef", "OrderedGroup", "UnorderedGroup"):
                plain.append(ch)
        for ch in sorted(indexed, key=_idx) + plain:
            if _local(ch.tag).startswith("RegionRef"):
                ref = ch.get("regionRef")
                if ref:
                    refs.append(ref)
            else:
                visit(ch)

    visit(ro)
    return {r: i for i, r in enumerate(refs)}


def parsear_pagina(ruta_xml: Path, modo: str) -> dict | None:
    """Devuelve {w, h, regiones} o None si el XML no es válido (mismo criterio que el script YOLO)."""
    try:
        root = ET.parse(ruta_xml).getroot()
    except ET.ParseError:
        return None
    page = next((e for e in root.iter() if _local(e.tag) == "Page"), None)
    if page is None:
        return None
    try:
        w = int(page.get("imageWidth", 0))
        h = int(page.get("imageHeight", 0))
    except (ValueError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None

    orden = _orden_lectura(root)
    regiones = []
    for elem in root.iter():
        clase = _clase_region(elem, _local(elem.tag), modo)
        if clase is None:
            continue
        coords = elem.find("{*}Coords")
        if coords is None:
            continue
        pts = _leer_puntos(coords.get("points", ""))
        if len(pts) < 2:
            continue
        pts = [(min(max(x, 0.0), w), min(max(y, 0.0), h)) for x, y in pts]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            continue
        rid = elem.get("id")
        regiones.append(
            {
                "clase": clase,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "poly": pts if len(pts) >= 3 else None,
                "region_id": rid,
                "orden": orden.get(rid),
            }
        )
    return {"w": w, "h": h, "regiones": regiones}


# ══════════════════════════════════════════════════════════════════════════════
# 2. ESCANEO (mismo recorrido y orden que 00_pagexml_to_yolo.py)
# ══════════════════════════════════════════════════════════════════════════════
def escanear(publicaciones: dict, corpus_dir: Path, modo: str):
    pares, conteo = [], defaultdict(int)
    for nombre, modo_split in publicaciones.items():
        dir_jpg = corpus_dir / nombre / "jpg"
        dir_page = corpus_dir / nombre / "page"
        if not dir_jpg.exists() or not dir_page.exists():
            print(f"  [AVISO] {nombre}: carpetas jpg/ o page/ no encontradas, saltando.")
            continue
        jpgs = sorted(
            {p.resolve(): p for p in dir_jpg.glob("*") if p.suffix.lower() in (".jpg", ".jpeg")}.values()
        )
        n = 0
        for jpg in jpgs:
            xml = dir_page / (jpg.stem + ".xml")
            if not xml.exists():
                continue
            pagina = parsear_pagina(xml, modo)
            if pagina is None:
                print(f"  [AVISO] {jpg.stem}: XML inválido o sin dimensiones, omitido.")
                continue
            for cls in {r["clase"] for r in pagina["regiones"]}:
                conteo[cls] += 1
            pares.append({"pub": nombre, "modo": modo_split, "stem": jpg.stem, "jpg": jpg, "pagina": pagina})
            n += 1
        print(f"  {nombre:<28}  {n:>4} pares")
    return pares, dict(conteo)


# ══════════════════════════════════════════════════════════════════════════════
# 3a. SPLIT DESDE JSON (por defecto): {stem: "train" | "val" | "test"}
# ══════════════════════════════════════════════════════════════════════════════
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def cargar_split_json(ruta: Path) -> dict[str, str]:
    """stem (normalizado NFC) -> 'train' | 'val' | 'test'."""
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: no se pudo leer {ruta}: {e}")
    validos = {"train", "val", "test"}
    split = {}
    for stem, s in datos.items():
        if s not in validos:
            sys.exit(f"ERROR: valor de split inválido en {ruta}: {stem!r} -> {s!r}")
        split[_nfc(stem)] = s
    return split


def _muestra(lista, n=10) -> str:
    extra = f" … (+{len(lista) - n})" if len(lista) > n else ""
    return ", ".join(lista[:n]) + extra


def asignar_splits_json(pares, split_stem: dict[str, str], faltantes: str):
    res = {"train": [], "val": [], "test": []}
    sin_split, usados = [], set()

    for par in pares:
        clave = _nfc(par["stem"])
        s = split_stem.get(clave)
        if s is None:
            sin_split.append(par["stem"])
            if faltantes == "train":
                res["train"].append(par)
            continue
        usados.add(clave)
        res[s].append(par)

    sin_par = sorted(set(split_stem) - usados)
    repetidos = sorted(s for s, n in Counter(_nfc(p["stem"]) for p in pares).items() if n > 1)

    if sin_split:
        destino = "enviadas a train" if faltantes == "train" else "omitidas"
        print(f"  [AVISO] {len(sin_split)} páginas del corpus no están en el JSON ({destino}): "
              f"{_muestra(sin_split)}")
    if sin_par:
        print(f"  [AVISO] {len(sin_par)} stems del JSON no tienen par JPG+XML válido "
              f"(esperable si usas --publicaciones): {_muestra(sin_par)}")
    if repetidos:
        print(f"  [AVISO] stems repetidos entre publicaciones (el JSON no puede distinguirlos, "
              f"todos reciben el mismo split): {_muestra(repetidos)}")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 3b. SPLIT ALEATORIO (--split-aleatorio) — réplica de 00_pagexml_to_yolo.py
#     (NO tocar el uso de rng)
# ══════════════════════════════════════════════════════════════════════════════
def split_publicacion(pares_pub, modo, nombre, rng, cfg):
    n = len(pares_pub)
    if modo == "solo_train" or n < cfg.SPLIT_MIN_PAGINAS:
        if modo == "estratificar" and n < cfg.SPLIT_MIN_PAGINAS:
            print(f"  [AVISO] {nombre}: solo {n} págs (<{cfg.SPLIT_MIN_PAGINAS}), todo a train.")
        shuffled = pares_pub[:]
        rng.shuffle(shuffled)
        return shuffled, [], []

    shuffled = pares_pub[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(n * cfg.SPLIT_TEST))
    n_val = max(1, round(n * cfg.SPLIT_VALIDATION))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def asignar_splits(pares, rng, cfg):
    por_pub, modos = defaultdict(list), {}
    for par in pares:
        por_pub[par["pub"]].append(par)
        modos[par["pub"]] = par["modo"]
    res = {"train": [], "val": [], "test": []}
    for nombre, lista in sorted(por_pub.items()):
        train, val, test = split_publicacion(lista, modos[nombre], nombre, rng, cfg)
        res["train"].extend(train)
        res["val"].extend(val)
        res["test"].extend(test)
    for lst in res.values():
        rng.shuffle(lst)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DE COCO
# ══════════════════════════════════════════════════════════════════════════════
def nombre_imagen(par) -> str:
    # Prefijo con la publicación para evitar colisiones de nombre entre cabeceras
    return f"{par['pub']}__{par['jpg'].name}"


def escribir_split(split, pares, clase_a_id, out_dir, symlinks):
    dir_img = out_dir / split / "images"
    if dir_img.exists():
        shutil.rmtree(dir_img)  # elimina restos de repartos anteriores
    dir_img.mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"description": f"prensa hispanica - {split}"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": ""} for n, i in clase_a_id.items()],
    }
    ann_id, n_desajuste, n_orden_completo = 1, 0, 0
    por_clase = defaultdict(int)

    for img_id, par in enumerate(pares, start=1):
        nombre = nombre_imagen(par)
        dst = dir_img / nombre
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if symlinks:
            dst.symlink_to(par["jpg"].resolve())
        else:
            shutil.copy2(par["jpg"], dst)

        with Image.open(par["jpg"]) as im:
            w_real, h_real = im.size
        w_xml, h_xml = par["pagina"]["w"], par["pagina"]["h"]
        sx, sy = w_real / w_xml, h_real / h_xml
        if abs(w_real - w_xml) > 1 or abs(h_real - h_xml) > 1:
            n_desajuste += 1  # las coordenadas se reescalan a la imagen real

        coco["images"].append(
            {
                "id": img_id,
                "file_name": nombre,
                "width": w_real,
                "height": h_real,
                "publication": par["pub"],
                "stem": par["stem"],
            }
        )

        anns_pagina = []
        for r in par["pagina"]["regiones"]:
            cid = clase_a_id.get(r["clase"])
            if cid is None:
                continue
            x0, y0, x1, y1 = r["bbox"]
            x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round((x1 - x0) * (y1 - y0), 2),
                "iscrowd": 0,
                "region_id": r["region_id"],
                "segmentation": [],
            }
            if r["poly"]:
                ann["segmentation"] = [[round(c, 2) for x, y in r["poly"] for c in (x * sx, y * sy)]]
            if r["orden"] is not None:
                ann["reading_order"] = r["orden"]
            anns_pagina.append(ann)
            por_clase[r["clase"]] += 1
            ann_id += 1
        if anns_pagina and all("reading_order" in a for a in anns_pagina):
            n_orden_completo += 1
        coco["annotations"].extend(anns_pagina)

    (out_dir / split / "instances.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return {
        "paginas": len(pares),
        "regiones": len(coco["annotations"]),
        "paginas_con_orden_lectura_completo": n_orden_completo,
        "paginas_con_tamano_distinto_xml_vs_jpg": n_desajuste,
        "por_clase": dict(por_clase),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. VERIFICACIÓN CONTRA EL DATASET YOLO
# ══════════════════════════════════════════════════════════════════════════════
def verificar_yolo(splits, yolo_dir: Path) -> bool:
    ok = True
    print(f"\nVerificando split contra {yolo_dir} …")
    for split in ("train", "val", "test"):
        d = yolo_dir / split / "images"
        if not d.exists():
            print(f"  [{split}] no existe {d}")
            ok = False
            continue
        yolo = {_nfc(p.stem) for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        mio = {_nfc(p["stem"]) for p in splits[split]}
        if yolo == mio:
            print(f"  [{split}] OK  ({len(mio)} páginas idénticas)")
        else:
            ok = False
            print(f"  [{split}] DIFIERE: solo COCO={len(mio - yolo)}  solo YOLO={len(yolo - mio)}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="PAGE-XML → COCO siguiendo el split del proyecto")
    ap.add_argument("--config", default="config", help="módulo de configuración (default: config)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--output", default=None, help="default: config.COCO_DIR")
    ap.add_argument("--modo-clases", choices=["tag", "fino"], default=None, help="default: config.MODO_CLASES (o tag)")
    ap.add_argument("--clases", nargs="+", default=None)
    ap.add_argument("--min-paginas-clase", type=int, default=None, help="default: config.MIN_PAGINAS_CLASE (o 3)")
    ap.add_argument("--semilla", type=int, default=None, help="solo con --split-aleatorio; default: config.SPLIT_SEED")
    ap.add_argument("--publicaciones", nargs="+", default=None)
    ap.add_argument("--symlinks", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-yolo", default=None, help="carpeta data/<corpus> del script YOLO para verificar el split")
    ap.add_argument("--split-json", default=None, help="JSON {stem: train|val|test}; default: config.SPLIT_JSON")
    ap.add_argument("--split-aleatorio", action="store_true",
                    help="ignora el JSON y recalcula el split con semilla (como 00_pagexml_to_yolo.py)")
    ap.add_argument("--faltantes", choices=["omitir", "train"], default="omitir",
                    help="qué hacer con las páginas que no aparecen en el JSON (default: omitir)")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    semilla = args.semilla if args.semilla is not None else cfg.SPLIT_SEED
    corpus_dir = Path(args.corpus) if args.corpus else cfg.CORPUS_DIR
    modo_clases = args.modo_clases or getattr(cfg, "MODO_CLASES", "tag")
    if args.min_paginas_clase is None:
        args.min_paginas_clase = getattr(cfg, "MIN_PAGINAS_CLASE", 3)
    out_dir = (
        Path(args.output)
        if args.output
        else Path(cfg.COCO_DIR)
    )
    if not corpus_dir.exists():
        sys.exit(f"ERROR: directorio corpus no encontrado: {corpus_dir}")

    ruta_json = None
    if not args.split_aleatorio:
        candidato = args.split_json or getattr(cfg, "SPLIT_JSON", None)
        if not candidato:
            sys.exit("ERROR: config no define SPLIT_JSON y no has pasado --split-json.\n"
                     "       Para recalcular el split al azar usa --split-aleatorio explícitamente.")
        ruta_json = Path(candidato)
        if not ruta_json.exists():
            sys.exit(f"ERROR: no existe el JSON de split: {ruta_json}\n"
                     f"       (usa --split-json RUTA o --split-aleatorio)")
            
    publicaciones = cfg.PUBLICACIONES
    if args.publicaciones:
        publicaciones = {k: v for k, v in publicaciones.items() if k in args.publicaciones}

    rng = random.Random(semilla)
    print(f"Escaneando corpus en {corpus_dir} (modo de clases: {modo_clases}) …")
    pares, conteo = escanear(publicaciones, corpus_dir, modo_clases)
    if not pares:
        sys.exit("ERROR: no se encontraron pares JPG+XML.")
    print(f"\n  {len(pares)} pares JPG+XML válidos.")

    if args.clases:
        validas = [c for c in args.clases if c in conteo]
    else:
        validas = [c for c, n in conteo.items() if n >= args.min_paginas_clase]
    excluidas = sorted(set(conteo) - set(validas))
    if not validas:
        sys.exit("ERROR: no quedan clases válidas.")
    clase_a_id = {c: i for i, c in enumerate(sorted(validas), start=1)}

    print("\n  Clases finales (id COCO desde 1):")
    for c, i in clase_a_id.items():
        print(f"    {i:>2}: {c:<20} en {conteo[c]:>4} páginas")
    if excluidas:
        print(f"  Excluidas: {excluidas}")

    if ruta_json is not None:
        origen_split = str(ruta_json)
        print(f"\nAplicando split desde {ruta_json} …")
        split_stem = cargar_split_json(ruta_json)
        splits = asignar_splits_json(pares, split_stem, args.faltantes)
        if not args.publicaciones:
            hay_dif = False
            for s in ("val", "test"):
                esperado = {k for k, v in split_stem.items() if v == s}
                obtenido = {_nfc(p["stem"]) for p in splits[s]}
                if esperado != obtenido:
                    hay_dif = True
                    print(f"  [ERROR] {s}: solo JSON={len(esperado - obtenido)} "
                          f"solo COCO={len(obtenido - esperado)}")
            if hay_dif:
                sys.exit("ERROR: el COCO no coincide con split.json (¿faltan JPG/XML?). No se escribe nada.")
    else:
        origen_split = f"aleatorio (semilla {semilla})"
        print(f"\n[AVISO] Split ALEATORIO (semilla={semilla}); NO coincide con split.json.")
        splits = asignar_splits(pares, rng, cfg)

    for s, lst in splits.items():
        print(f"  {s:<6} {len(lst):>4} páginas")

    if args.check_yolo:
        verificar_yolo(splits, Path(args.check_yolo))

    if args.dry_run:
        print("\n[DRY-RUN] No se ha escrito nada.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEscribiendo dataset COCO en {out_dir} …")
    stats = {s: escribir_split(s, lst, clase_a_id, out_dir, args.symlinks) for s, lst in splits.items()}
    (out_dir / "classes.json").write_text(json.dumps(clase_a_id, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "informe_coco.json").write_text(
        json.dumps(
            {
                "origen_split": origen_split,
                "semilla": semilla if ruta_json is None else None,
                "modo_clases": modo_clases,
                "clases_detectadas": conteo,
                "clases_excluidas": excluidas,
                "stats_por_split": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for s, st in stats.items():
        print(
            f"  {s:<6} {st['paginas']:>4} págs  {st['regiones']:>6} regiones  "
            f"orden de lectura completo en {st['paginas_con_orden_lectura_completo']} págs  "
            f"tamaño XML≠JPG en {st['paginas_con_tamano_distinto_xml_vs_jpg']}"
        )
    print("\nListo.")


if __name__ == "__main__":
    main()