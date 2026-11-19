#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_compare_metrics.py

Compara los metricas_resumen.csv del modelo base y del afinado. Muestra por publicación
el CER y el WER micro de cada uno y su diferencia (negativa = mejora).

Uso:
    python 05_compare_metrics.py resultados/eval_base/metricas_resumen.csv resultados/eval_ft/metricas_resumen.csv
    python 05_compare_metrics.py base.csv ft.csv --out comparacion.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r["publicacion"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description="Compara métricas de dos evaluaciones")
    ap.add_argument("base")
    ap.add_argument("afinado")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for ruta in (args.base, args.afinado):
        if not Path(ruta).exists():
            sys.exit(f"ERROR: no existe {ruta}")
    base, afinado = leer(args.base), leer(args.afinado)
    nombres = [n for n in base if n in afinado]

    filas = []
    for n in nombres:
        fila = {"publicacion": n, "paginas": afinado[n]["paginas"]}
        for m in ("CER_micro", "WER_micro"):
            b, a = float(base[n][m]), float(afinado[n][m])
            fila[f"{m}_base"], fila[f"{m}_afinado"], fila[f"{m}_dif"] = round(b, 4), round(a, 4), round(a - b, 4)
        filas.append(fila)

    print("{:<24}{:>8}{:>11}{:>11}{:>10}{:>11}{:>11}{:>10}".format(
        "publicación", "páginas", "CER base", "CER ft", "dif", "WER base", "WER ft", "dif"))
    for f in filas:
        print("{:<24}{:>8}{:>11.2%}{:>11.2%}{:>+10.2%}{:>11.2%}{:>11.2%}{:>+10.2%}".format(
            f["publicacion"], f["paginas"],
            f["CER_micro_base"], f["CER_micro_afinado"], f["CER_micro_dif"],
            f["WER_micro_base"], f["WER_micro_afinado"], f["WER_micro_dif"]))

    if args.out and filas:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\nTabla guardada en {args.out}")


if __name__ == "__main__":
    main()