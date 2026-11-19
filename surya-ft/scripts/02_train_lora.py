#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_lora.py

Entrena una LoRA sobre Surya 2 con los JSONL de 01_build_data.py, mezclando las dos
tareas (layout de página y OCR por bloque). Cada EVAL_EVERY pasos evalúa en validación
y guarda adapter_last, adapter_best y estado.pt para poder reanudar.

Uso:
    python 02_train_lora.py --smoke      # prueba rápida de 2 pasos
    python 02_train_lora.py
    python 02_train_lora.py --resume     # continúa tras un reinicio

Opciones: --tasks {layout,block}, --epochs, --lr, --rank, --run
Salida: RUNS_DIR/<run>/ (adapter_best/, adapter_last/, estado.pt, train_log.jsonl)

Para el servidor de Surya antes de entrenar (docker stop ...): vLLM ocupa la VRAM.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import (
    a_dispositivo, cargar_config, cargar_modelo, cargar_procesador, codificar, escalar, recortar, token_fin,
)


def valor(a, b):
    return b if a is None else a


def identidad(x):
    return x


def leer_jsonl(ruta):
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def registros_epoca(capas, bloques, tareas, bloques_por_capa, rng):
    reg = []
    if "layout" in tareas:
        reg += capas
    if "block" in tareas:
        n = len(bloques) if "layout" not in tareas else min(len(bloques), int(bloques_por_capa * len(capas)))
        reg += rng.sample(bloques, n)
    rng.shuffle(reg)
    return reg


class SFTDataset(Dataset):
    def __init__(self, registros, corpus, procesador, fin, max_tokens, cfg):
        self.registros = registros
        self.corpus = corpus
        self.procesador = procesador
        self.fin = fin
        self.max_tokens = max_tokens
        self.cfg = cfg

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, i):
        r = self.registros[i]
        imagen = Image.open(self.corpus / r["image"]).convert("RGB")
        if r["crop"]:
            imagen = recortar(imagen, r["crop"], self.cfg.BLOCK_PAD)
        imagen = escalar(imagen, self.cfg.SCALE_MAX, self.cfg.SCALE_MIN, self.cfg.GRID)
        enc = codificar(self.procesador, imagen, r["prompt"], r["answer"], self.fin)
        if enc["input_ids"].shape[1] > self.max_tokens:
            return None
        return enc, r["task"]


def preparar_lora(modelo, args):
    from peft import LoraConfig, get_peft_model

    if args.target_modules:
        hojas = [h.strip() for h in args.target_modules.split(",") if h.strip()]
    else:
        hojas = sorted(
            {
                n.rsplit(".", 1)[-1]
                for n, m in modelo.named_modules()
                if isinstance(m, torch.nn.Linear) and not re.search(r"visual|vision|mtp|lm_head|embed", n)
            }
        )
    if not hojas:
        sys.exit("ERROR: no se encontraron capas lineales del modelo de lenguaje; usa --target-modules")
    patron = rf"(?!.*(?:visual|vision|mtp|lm_head|embed)).*\.(?:{'|'.join(map(re.escape, hojas))})"
    print(f"LoRA sobre capas: {hojas}")

    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()
    config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=patron, task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, config)
    modelo.print_trainable_parameters()
    return modelo


@torch.no_grad()
def evaluar(modelo, procesador, fin, val, corpus, max_tokens, cfg, dispositivo):
    modelo.eval()
    res = {}
    for tarea, regs in val.items():
        if not regs:
            continue
        ds = SFTDataset(regs, corpus, procesador, fin, max_tokens, cfg)
        perdidas = []
        for i in range(len(ds)):
            item = ds[i]
            if item is None:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perdidas.append(modelo(**a_dispositivo(item[0], dispositivo)).loss.item())
        if perdidas:
            res[tarea] = sum(perdidas) / len(perdidas)
    modelo.train()
    if res:
        res["media"] = sum(res.values()) / len(res)
    return res


def guardar_estado(run, modelo, opt, sched, paso, epoca, vistas, mejor):
    modelo.save_pretrained(run / "adapter_last")
    torch.save(
        {"opt": opt.state_dict(), "sched": sched.state_dict(), "paso": paso, "epoca": epoca,
         "vistas": vistas, "mejor": mejor},
        run / "estado.pt",
    )


def diagnostico(procesador, enc, fin):
    ids = enc["input_ids"][0]
    con_perdida = int((enc["labels"][0] != -100).sum())
    print(f"  tokens totales: {ids.shape[0]}   con pérdida: {con_perdida}")
    if "image_grid_thw" in enc:
        print(f"  image_grid_thw: {enc['image_grid_thw'].tolist()}")
    print(f"  token de fin usado: {fin!r}   eos_token del tokenizer: {procesador.tokenizer.eos_token!r}")
    print(f"  final de la secuencia: {procesador.tokenizer.decode(ids[-12:])!r}")


def entrenar(args, cfg):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    corpus, data, run = Path(args.corpus), Path(args.data), Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    capas = leer_jsonl(data / "train_layout.jsonl")
    bloques = leer_jsonl(data / "train_block.jsonl")
    val = {"layout": leer_jsonl(data / "val_layout.jsonl"), "block": leer_jsonl(data / "val_block.jsonl")}
    val = {k: random.Random(0).sample(v, min(len(v), args.eval_samples)) for k, v in val.items()}
    if not (capas or bloques):
        sys.exit(f"ERROR: no hay datos de entrenamiento en {data} (ejecuta 10_build_sft_data.py)")

    n_epoca = len(registros_epoca(capas, bloques, args.tasks, args.blocks_per_layout, random.Random(0)))
    pasos_epoca = math.ceil(n_epoca / args.grad_accum)
    total = pasos_epoca * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    print(f"Muestras por época: {n_epoca}   pasos por época: {pasos_epoca}   pasos totales: {total}")

    procesador = cargar_procesador(args.model, cfg.MIN_PIXELS, cfg.MAX_PIXELS)
    dispositivo = "cuda:0"
    modelo = cargar_modelo(args.model, device_map={"": 0})
    fin = token_fin(procesador, modelo, args.end_token)
    modelo = preparar_lora(modelo, args)
    modelo.train()

    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * args.warmup_ratio)), total)

    paso, epoca0, vistas0, mejor = 0, 0, 0, float("inf")
    if args.resume and (run / "estado.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(modelo, load_file(str(run / "adapter_last" / "adapter_model.safetensors")))
        estado = torch.load(run / "estado.pt", map_location="cpu")
        opt.load_state_dict(estado["opt"])
        sched.load_state_dict(estado["sched"])
        paso, epoca0, vistas0, mejor = estado["paso"], estado["epoca"], estado["vistas"], estado["mejor"]
        print(f"Reanudado en paso {paso} (época {epoca0}, {vistas0} muestras vistas)")

    primera = registros_epoca(capas, bloques, args.tasks, args.blocks_per_layout, random.Random(args.seed))[:1]
    if primera:
        item = SFTDataset(primera, corpus, procesador, fin, 10**9, cfg)[0]
        print("Diagnóstico de la primera muestra:")
        diagnostico(procesador, item[0], fin)

    log = open(run / "train_log.jsonl", "a", encoding="utf-8")
    perdidas = {"layout": [], "block": []}
    micro, t0 = 0, time.time()
    terminado = paso >= total

    for epoca in range(epoca0, args.epochs):
        if terminado:
            break
        registros = registros_epoca(capas, bloques, args.tasks, args.blocks_per_layout, random.Random(args.seed + epoca))
        inicio = vistas0 if epoca == epoca0 else 0
        loader = DataLoader(
            SFTDataset(registros[inicio:], corpus, procesador, fin, args.max_tokens, cfg),
            batch_size=None, shuffle=False, num_workers=args.workers, collate_fn=identidad,
        )
        vistas = inicio
        for item in loader:
            vistas += 1
            if item is None:
                continue
            enc, tarea = item
            with torch.autocast("cuda", dtype=torch.bfloat16):
                salida = modelo(**a_dispositivo(enc, dispositivo))
            (salida.loss / args.grad_accum).backward()
            perdidas[tarea].append(salida.loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            paso += 1

            if paso == 1 or paso % args.log_every == 0:
                medias = {k: round(sum(v) / len(v), 4) for k, v in perdidas.items() if v}
                linea = {"paso": paso, "epoca": epoca, "lr": sched.get_last_lr()[0], "loss": medias,
                         "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                         "min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(linea))
                log.write(json.dumps(linea) + "\n")
                log.flush()
                perdidas = {"layout": [], "block": []}

            if paso % args.eval_every == 0 or paso >= total:
                res = evaluar(modelo, procesador, fin, val, corpus, args.max_tokens, cfg, dispositivo)
                print(f"  validación paso {paso}: {json.dumps({k: round(v, 4) for k, v in res.items()})}")
                log.write(json.dumps({"paso": paso, "val": res}) + "\n")
                log.flush()
                if res and res["media"] < mejor:
                    mejor = res["media"]
                    modelo.save_pretrained(run / "adapter_best")
                guardar_estado(run, modelo, opt, sched, paso, epoca, vistas, mejor)

            if paso >= total:
                terminado = True
                break

    if micro % args.grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    modelo.save_pretrained(run / "adapter_last")
    if not (run / "adapter_best").exists():
        modelo.save_pretrained(run / "adapter_best")
    log.close()
    print(f"\nListo. Adaptadores en {run} (mejor pérdida de validación: {mejor:.4f})")


def main():
    ap = argparse.ArgumentParser(description="Fine-tuning LoRA de Surya 2 (layout + OCR por bloque)")
    ap.add_argument("--config", default="config_ft")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--tasks", nargs="+", choices=["layout", "block"], default=["layout", "block"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--blocks-per-layout", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--target-modules", default=None, help="nombres de capas separados por comas (por defecto, automático)")
    ap.add_argument("--end-token", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida: 2 pasos y validación mínima")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    args.model = valor(args.model, cfg.BASE_MODEL)
    args.data = valor(args.data, str(cfg.SFT_DIR))
    args.corpus = valor(args.corpus, str(cfg.CORPUS_DIR))
    args.run_dir = str(Path(cfg.RUNS_DIR) / valor(args.run, cfg.RUN_NAME))
    args.epochs = valor(args.epochs, cfg.EPOCHS)
    args.lr = valor(args.lr, cfg.LR)
    args.rank = valor(args.rank, cfg.LORA_R)
    args.alpha = valor(args.alpha, cfg.LORA_ALPHA)
    args.dropout = valor(args.dropout, cfg.LORA_DROPOUT)
    args.grad_accum = valor(args.grad_accum, cfg.GRAD_ACCUM)
    args.blocks_per_layout = valor(args.blocks_per_layout, cfg.BLOCKS_PER_LAYOUT)
    args.eval_every = valor(args.eval_every, cfg.EVAL_EVERY)
    args.eval_samples = valor(args.eval_samples, cfg.EVAL_SAMPLES)
    args.max_tokens = valor(args.max_tokens, cfg.MAX_SEQ_TOKENS)
    args.seed = valor(args.seed, cfg.SEED)
    args.warmup_ratio = cfg.WARMUP_RATIO
    args.weight_decay = cfg.WEIGHT_DECAY
    args.log_every = cfg.LOG_EVERY
    args.end_token = valor(args.end_token, cfg.END_TOKEN)

    if args.smoke:
        args.max_steps, args.grad_accum, args.eval_samples, args.eval_every = 2, 2, 2, 1
        args.run_dir = str(Path(args.run_dir).parent / (Path(args.run_dir).name + "_smoke"))

    if not torch.cuda.is_available():
        sys.exit("ERROR: no hay GPU CUDA disponible")
    entrenar(args, cfg)


if __name__ == "__main__":
    main()