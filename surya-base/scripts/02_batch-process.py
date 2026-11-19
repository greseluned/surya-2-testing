#!/usr/bin/env python3
"""
02_batch-process.py

Procesa páginas con Surya 2 (layout y OCR por bloque) y escribe un PAGE XML por imagen.
  eval   procesa un split COCO y calcula CER/WER contra los PAGE XML de referencia
  infer  procesa una carpeta de imágenes, excluyendo el test set

Uso (desde WSL, con el entorno activado):
    python surya/scripts/02_batch-process.py eval  --dataset surya/data/corpus_coco/test --out surya/results/eval
    python surya/scripts/02_batch-process.py infer --input surya/corpus \
        --exclude-dataset surya/data/corpus_coco/test --out surya/results/infer

Opciones: --testset TXT (alternativa a --dataset), --only-metrics, --overwrite, --limit N,
          --images, --budget-factor / --budget-extra (prueba 2 y 300 si salen bloques cortados)

Usa una carpeta --out nueva en cada evaluación: los XML que ya haya no se borran.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean
from typing import Optional
import xml.etree.ElementTree as ET

# Versión del esquema PAGE. 2013-07-15 es la que exporta Transkribus; cámbiala
# si tus XML de referencia usan otra (p. ej. 2019-07-15).
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

VISUAL_LABELS = {"Picture", "Figure", "Diagram"}
REGION_TYPE = {  # etiqueta de Surya -> atributo type de TextRegion en PAGE
    "Text": "paragraph",
    "SectionHeader": "heading",
    "PageHeader": "header",
    "PageFooter": "footer",
    "Caption": "caption",
    "Footnote": "footnote",
}


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def to_local_path(s: str) -> Path:
    """Convierte 'C:/Users/...' en '/mnt/c/Users/...' cuando se ejecuta en WSL."""
    s = s.strip().strip('"')
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m and sys.platform != "win32":
        s = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return Path(s)


def gt_xml_path(img: Path) -> Path:
    """Misma ruta cambiando la carpeta 'jpg' por 'page' y la extensión por .xml."""
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "jpg":
            parts[i] = "page"
            break
    return Path(*parts).with_suffix(".xml")


def pub_of(img: Path) -> str:
    return img.parent.parent.name if img.parent.name.lower() == "jpg" else img.parent.name


def out_xml_path(img: Path, out_dir: Path, root: Optional[Path] = None) -> Path:
    if root is not None:  # modo infer: refleja la estructura de la carpeta de entrada
        rel = img.relative_to(root)
        parts = [p for p in rel.parts[:-1] if p.lower() != "jpg"]
        return out_dir.joinpath(*parts, img.stem + ".xml")
    # modo eval con lista de rutas: <out>/<publicación>/<nombre>.xml
    return out_dir / pub_of(img) / (img.stem + ".xml")


@dataclass
class Task:
    img: Path
    out: Path
    gt: Optional[Path] = None
    pub: str = ""
    stem: str = ""  # nombre base original (sin el prefijo <publicación>__ del dataset)
    name: str = ""  # nombre de archivo original (para imageFilename y los avisos)

    def __post_init__(self):
        self.stem = self.stem or self.img.stem
        self.name = self.name or self.img.name


def read_path_list(p: str) -> list:
    lines = Path(p).read_text(encoding="utf-8").splitlines()
    return [to_local_path(l) for l in lines if l.strip()]


# --------------------------------------------------------------------------- #
# HTML (salida de Surya) -> texto plano
# --------------------------------------------------------------------------- #
class _HtmlToText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
    CELL = {"td", "th"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.out.append("\n")
        elif tag in self.CELL:
            self.out.append(" ")

    def handle_endtag(self, tag):
        if tag in self.BLOCK and tag != "br":
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


def html_to_text(s: str) -> str:
    p = _HtmlToText()
    p.feed(s or "")
    p.close()
    lines = [" ".join(l.split()) for l in "".join(p.out).splitlines()]
    return "\n".join(l for l in lines if l)


# --------------------------------------------------------------------------- #
# PAGE XML: escritura y lectura
# --------------------------------------------------------------------------- #
def g(obj, key, default=None):
    """Lee un campo tanto de un objeto como de un dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def write_page_xml(path: Path, image_name: str, width: int, height: int, blocks) -> None:
    ET.register_namespace("", PAGE_NS)
    q = lambda tag: f"{{{PAGE_NS}}}{tag}"
    now = datetime.now().isoformat(timespec="seconds")

    root = ET.Element(q("PcGts"))
    meta = ET.SubElement(root, q("Metadata"))
    ET.SubElement(meta, q("Creator")).text = "surya-ocr 2 (layout + OCR por bloque)"
    ET.SubElement(meta, q("Created")).text = now
    ET.SubElement(meta, q("LastChange")).text = now

    page = ET.SubElement(
        root, q("Page"),
        imageFilename=image_name, imageWidth=str(width), imageHeight=str(height),
    )

    def order_key(b):
        ro = g(b, "reading_order")
        return ro if ro is not None else 10**9  # sort estable: conserva el orden recibido

    ordered = sorted(blocks, key=order_key)

    group = ET.SubElement(
        ET.SubElement(page, q("ReadingOrder")),
        q("OrderedGroup"), id="ro1", caption="Regions reading order",
    )
    for i in range(len(ordered)):
        ET.SubElement(group, q("RegionRefIndexed"), index=str(i), regionRef=f"r{i}")

    for i, b in enumerate(ordered):
        label = g(b, "label", "Text")
        points = " ".join(f"{round(x)},{round(y)}" for x, y in g(b, "polygon"))
        custom = f"readingOrder {{index:{i};}} structure {{type:{label};}}"

        if label in VISUAL_LABELS or g(b, "skipped", False):
            region = ET.SubElement(page, q("ImageRegion"), id=f"r{i}", custom=custom)
            ET.SubElement(region, q("Coords"), points=points)
            continue

        region = ET.SubElement(
            page, q("TextRegion"),
            id=f"r{i}", type=REGION_TYPE.get(label, "other"), custom=custom,
        )
        ET.SubElement(region, q("Coords"), points=points)
        conf = g(b, "confidence")
        te = ET.SubElement(region, q("TextEquiv"))
        if conf is not None:
            te.set("conf", f"{float(conf):.4f}")
        html = g(b, "html", "") or ""
        corte = html.find("<|im_end|>")
        if corte != -1:
            print(f"  [aviso] bucle de generación en {image_name}, bloque {i}", file=sys.stderr)
            html = html[:corte]
        ET.SubElement(te, q("Unicode")).text = html_to_text(html)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _first_unicode(el) -> str:
    for te in _children(el, "TextEquiv"):
        for u in _children(te, "Unicode"):
            if u.text and u.text.strip():
                return u.text
        break  # solo la primera TextEquiv
    return ""


def _region_text(region) -> str:
    lines = [_first_unicode(tl) for tl in _children(region, "TextLine")]
    lines = [l for l in lines if l.strip()]
    if lines:
        return " ".join(lines)
    return _first_unicode(region)  # si no hay líneas, texto a nivel de región


def read_page_text(path: Path) -> str:
    """Texto de la página en orden de lectura (ReadingOrder si existe, si no orden del XML).
    Sirve tanto para las referencias de Transkribus como para nuestras predicciones.
    Solo se leen TextRegion (las ImageRegion no cuentan)."""
    root = ET.parse(path).getroot()
    regions = {}
    for el in root.iter():
        if _local(el.tag) == "TextRegion":
            regions[el.get("id")] = _region_text(el)

    indexed = []
    for el in root.iter():
        if _local(el.tag) == "RegionRefIndexed":
            indexed.append((int(el.get("index", 0)), el.get("regionRef")))
    indexed.sort()  # asume un OrderedGroup plano, que es lo que exporta Transkribus

    ids = [r for _, r in indexed if r in regions]
    ids += [r for r in regions if r not in ids]
    return " ".join(regions[r] for r in ids)


# --------------------------------------------------------------------------- #
# Procesado con Surya
# --------------------------------------------------------------------------- #
def process(tasks, batch_size: int, err_log: Path, budget=(1.0, 100)) -> None:
    from PIL import Image
    import surya.recognition as surya_recognition
    from surya.inference import SuryaInferenceManager
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor

    factor, extra = budget
    if (factor, extra) != (1.0, 100):
        surya_recognition.image_token_budget = lambda count, ceiling=4096, floor=64: min(
            max(int(count * factor) + extra, floor), ceiling)
        print(f"Límite de tokens por bloque: {factor} x count + {extra}")

    manager = SuryaInferenceManager()
    layout_pred = LayoutPredictor(manager)
    rec_pred = RecognitionPredictor(manager)

    def fail(msg: str):
        print(f"  [error] {msg}", file=sys.stderr)
        err_log.parent.mkdir(parents=True, exist_ok=True)
        with open(err_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{msg}\n")

    def run(batch):
        imgs = [Image.open(t.img).convert("RGB") for t in batch]  # PIL: seguro con rutas no ASCII
        layouts = layout_pred(imgs)
        ok = []
        for i, lay in enumerate(layouts):
            if g(lay, "error", False):
                fail(f"layout fallido: {batch[i].img}")
            else:
                ok.append(i)
        if not ok:
            return
        # Pasar `layouts` fuerza el modo por bloques: una llamada de OCR por bloque de layout
        preds = rec_pred([imgs[i] for i in ok], [layouts[i] for i in ok])
        for i, pred in zip(ok, preds):
            blocks = g(pred, "blocks", [])
            n_err = sum(bool(g(b, "error", False)) for b in blocks)
            if n_err:
                fail(f"{n_err} bloque(s) con error de OCR: {batch[i].img}")
            w, h = imgs[i].size
            write_page_xml(batch[i].out, batch[i].name, w, h, blocks)

    total = len(tasks)
    for k in range(0, total, batch_size):
        batch = tasks[k:k + batch_size]
        print(f"[{min(k + batch_size, total)}/{total}] {batch[0].name} ...", flush=True)
        try:
            run(batch)
        except Exception as e:  # si falla el lote, aislar la imagen problemática
            print(f"  lote fallido ({e!r}); reintento una a una", file=sys.stderr)
            for t in batch:
                try:
                    run([t])
                except Exception as e2:
                    fail(f"{t.img}: {e2!r}")


# --------------------------------------------------------------------------- #
# Imágenes con el layout dibujado (se generan a partir del PAGE XML)
# --------------------------------------------------------------------------- #
LABEL_COLORS = {
    "Text": "#1f77b4", "SectionHeader": "#2ca02c", "PageHeader": "#d62728",
    "PageFooter": "#d62728", "Caption": "#9467bd", "Footnote": "#17becf",
    "Table": "#8c564b", "Picture": "#ff7f0e", "Figure": "#ff7f0e", "Diagram": "#ff7f0e",
}


def draw_layout(img_path: Path, xml_path: Path, out_path: Path) -> None:
    """Dibuja cada región con su etiqueta y su número de orden de lectura."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(24, img.width // 70))
    except OSError:
        font = ImageFont.load_default()

    root = ET.parse(xml_path).getroot()
    order = {el.get("regionRef"): int(el.get("index", 0))
             for el in root.iter() if _local(el.tag) == "RegionRefIndexed"}

    for el in root.iter():
        if _local(el.tag) not in ("TextRegion", "ImageRegion"):
            continue
        coords = _children(el, "Coords")
        if not coords:
            continue
        pts = [tuple(float(v) for v in p.split(",")) for p in coords[0].get("points", "").split()]
        if len(pts) < 3:
            continue
        m = re.search(r"structure \{type:([^;]+);", el.get("custom", ""))
        label = m.group(1) if m else _local(el.tag)
        color = LABEL_COLORS.get(label, "#7f7f7f")

        draw.polygon(pts, fill=color + "30")  # relleno translúcido
        draw.line(pts + [pts[0]], fill=color, width=max(3, img.width // 600))
        tag = f"{order.get(el.get('id'), '?')} {label}"
        x0, y0 = pts[0]
        draw.rectangle(draw.textbbox((x0, y0), tag, font=font), fill=color)
        draw.text((x0, y0), tag, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=85)


# --------------------------------------------------------------------------- #
# Evaluación (CER / WER)
# --------------------------------------------------------------------------- #
def norm(t: str) -> str:
    return " ".join(_nfc(t).split())


def evaluate(tasks, out_dir: Path) -> None:
    from rapidfuzz.distance import Levenshtein

    rows = []
    for t in tasks:
        if not (t.gt and t.gt.exists()):
            print(f"[aviso] sin XML de referencia: {t.name}  (buscado en {t.gt})")
            continue
        if not t.out.exists():
            print(f"[aviso] sin predicción: {t.name}")
            continue
        ref = norm(read_page_text(t.gt))
        hyp = norm(read_page_text(t.out))
        if not ref:
            print(f"[aviso] referencia vacía: {t.gt.name}")
            continue
        rw, hw = ref.split(), hyp.split()
        ce, we = Levenshtein.distance(ref, hyp), Levenshtein.distance(rw, hw)
        rows.append({
            "imagen": t.name, "publicacion": t.pub,
            "ref_chars": len(ref), "ref_words": len(rw),
            "char_edits": ce, "word_edits": we,
            "cer": round(ce / len(ref), 4), "wer": round(we / max(len(rw), 1), 4),
        })
        txt_dir = t.out.parent / "_texto"  # para inspeccionar diferencias a mano
        txt_dir.mkdir(exist_ok=True)
        (txt_dir / f"{t.stem}.gt.txt").write_text(ref, encoding="utf-8")
        (txt_dir / f"{t.stem}.pred.txt").write_text(hyp, encoding="utf-8")

    if not rows:
        print("No hay páginas para evaluar.")
        return

    def agg(name, rs):
        rc = sum(r["ref_chars"] for r in rs)
        rwd = sum(r["ref_words"] for r in rs)
        return {
            "publicacion": name, "paginas": len(rs),
            "CER_micro": round(sum(r["char_edits"] for r in rs) / rc, 4),
            "WER_micro": round(sum(r["word_edits"] for r in rs) / max(rwd, 1), 4),
            "CER_macro": round(mean(r["cer"] for r in rs), 4),
            "WER_macro": round(mean(r["wer"] for r in rs), 4),
        }

    groups = defaultdict(list)
    for r in rows:
        groups[r["publicacion"]].append(r)
    per_pub = [agg(p, rs) for p, rs in sorted(groups.items())]
    media_pub = {  # cada publicación pesa igual, tenga las páginas que tenga
        "publicacion": "MEDIA_PUBLICACIONES", "paginas": len(rows),
        **{k: round(mean(s[k] for s in per_pub), 4)
           for k in ("CER_micro", "WER_micro", "CER_macro", "WER_macro")},
    }
    summary = per_pub + [agg("TOTAL", rows), media_pub]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("metricas_paginas.csv", rows), ("metricas_resumen.csv", summary)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    print("\n{:<24}{:>8}{:>11}{:>11}{:>11}{:>11}".format(
        "publicación", "páginas", "CER micro", "WER micro", "CER macro", "WER macro"))
    for s in summary:
        print("{:<24}{:>8}{:>11.2%}{:>11.2%}{:>11.2%}{:>11.2%}".format(
            s["publicacion"], s["paginas"], s["CER_micro"], s["WER_micro"],
            s["CER_macro"], s["WER_macro"]))
    print(f"\nCSV guardados en {out_dir}")


# --------------------------------------------------------------------------- #
# Construcción de tareas
# --------------------------------------------------------------------------- #
def load_coco_images(dataset_dir: str) -> list:
    inst = to_local_path(dataset_dir) / "instances.json"
    if not inst.exists():
        sys.exit(f"ERROR: no existe {inst} (¿has generado el dataset con 00b_pagexml_to_coco.py?)")
    return json.loads(inst.read_text(encoding="utf-8"))["images"]


def build_tasks_dataset(args):
    """Test set = imágenes de <dataset>/images; referencia = <corpus>/<pub>/page/<stem>.xml."""
    ds = to_local_path(args.dataset)
    corpus = (to_local_path(args.corpus) if args.corpus
              else Path(__file__).resolve().parent.parent / "corpus")
    print(f"Dataset: {ds}\nCorpus (referencias PAGE XML): {corpus}")
    tasks = []
    for im in sorted(load_coco_images(args.dataset), key=lambda i: i["file_name"]):
        fn = im["file_name"]
        img = ds / "images" / fn
        if not img.exists():
            print(f"[aviso] no existe la imagen del dataset: {img}")
            continue
        pub = im.get("publication") or fn.split("__", 1)[0]
        name = fn[len(pub) + 2:] if fn.startswith(pub + "__") else fn
        stem = im.get("stem") or Path(name).stem
        tasks.append(Task(img, Path(args.out) / pub / f"{stem}.xml",
                          corpus / pub / "page" / f"{stem}.xml", pub, stem, name))
    return tasks


def build_tasks_eval(args):
    tasks, seen = [], set()
    for img in read_path_list(args.testset):
        if img in seen:
            continue
        seen.add(img)
        if not img.exists():
            print(f"[aviso] no existe la imagen: {img}")
            continue
        tasks.append(Task(img, out_xml_path(img, Path(args.out)), gt_xml_path(img), pub_of(img)))
    return tasks


def build_tasks_infer(args):
    root = to_local_path(args.input).resolve()
    excluded = set()
    if args.exclude:
        excluded = {p.resolve() for p in read_path_list(args.exclude)}
    excluded_names = set()  # nombres del dataset: <publicación>__<archivo>
    for d in args.exclude_dataset or []:
        excluded_names |= {_nfc(im["file_name"]) for im in load_coco_images(d)}

    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    kept = [p for p in imgs
            if p.resolve() not in excluded
            and _nfc(f"{pub_of(p)}__{p.name}") not in excluded_names]
    print(f"{len(imgs)} imágenes encontradas, {len(imgs) - len(kept)} excluidas (test set), "
          f"{len(kept)} a procesar")
    return [Task(p, out_xml_path(p, Path(args.out), root=root), pub=pub_of(p)) for p in kept]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True, help="carpeta de salida")
        p.add_argument("--batch-size", type=int, default=8, help="páginas por lote (def. 8)")
        p.add_argument("--overwrite", action="store_true", help="reprocesar aunque el XML ya exista")
        p.add_argument("--limit", type=int, help="procesar solo las N primeras (para pruebas)")
        p.add_argument("--images", action="store_true",
                       help="guardar imágenes con el layout dibujado en <salida>/<pub>/_layout/")
        p.add_argument("--budget-factor", type=float, default=1.0,
                       help="límite de tokens por bloque = factor x count + extra (def. 1.0, como Surya)")
        p.add_argument("--budget-extra", type=int, default=100,
                       help="margen fijo de tokens por bloque (def. 100, como Surya)")

    ev = sub.add_parser("eval", help="procesar el test set y calcular CER/WER")
    src = ev.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="carpeta del split COCO (con images/ e instances.json), p. ej. .../corpus_coco/test")
    src.add_argument("--testset", help="txt con una ruta de imagen por línea")
    ev.add_argument("--corpus", help="raíz del corpus con <pub>/page/*.xml (solo con --dataset; "
                                     "def.: carpeta 'corpus' junto a 'scripts')")
    ev.add_argument("--only-metrics", action="store_true",
                    help="no procesar nada; calcular métricas con los XML ya generados")
    common(ev)

    inf = sub.add_parser("infer", help="procesar una carpeta de imágenes")
    inf.add_argument("--input", required=True, help="carpeta con imágenes (búsqueda recursiva)")
    inf.add_argument("--exclude-dataset", nargs="+",
                     help="una o varias carpetas de split COCO cuyas imágenes se excluyen (p. ej. .../test)")
    inf.add_argument("--exclude", help="txt con rutas a excluir")
    common(inf)

    args = ap.parse_args()
    if args.cmd == "eval":
        tasks = build_tasks_dataset(args) if args.dataset else build_tasks_eval(args)
    else:
        tasks = build_tasks_infer(args)
    if args.limit:
        tasks = tasks[:args.limit]

    only_metrics = args.cmd == "eval" and args.only_metrics
    if not only_metrics:
        todo = tasks if args.overwrite else [t for t in tasks if not t.out.exists()]
        print(f"{len(tasks) - len(todo)} ya procesadas (se saltan), {len(todo)} pendientes")
        if todo:
            process(todo, args.batch_size, Path(args.out) / "errores.log", budget=(args.budget_factor, args.budget_extra))

    if args.images:  # se dibuja desde el XML, así también sirve para páginas ya procesadas
        n = 0
        for t in tasks:
            dest = t.out.parent / "_layout" / f"{t.stem}.jpg"
            if t.out.exists() and (args.overwrite or not dest.exists()):
                try:
                    draw_layout(t.img, t.out, dest)
                    n += 1
                except Exception as e:
                    print(f"[aviso] no se pudo dibujar {t.name}: {e!r}")
        print(f"{n} imágenes con layout guardadas")

    if args.cmd == "eval":
        evaluate(tasks, Path(args.out))


if __name__ == "__main__":
    main()