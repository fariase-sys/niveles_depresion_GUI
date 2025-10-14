#!/usr/bin/env python3
"""
Extracción de tweets en español con categorías de palabras clave
(Depresión clara, Depresión leve, Sin depresión).
Se utiliza el dataset de ejemplo `pysentimiento/spanish-tweets`.
"""

import re
import csv
import argparse
from datasets import load_dataset
from tqdm import tqdm

# ---------- 1. KEYWORDS (grupos) ----------
# Se definen tres grupos de palabras clave:
#   - Palabras sin depresión
#   - Palabras de depresión leve
#   - Palabras de depresión clara
# (Aquí se deben completar manualmente según la investigación)

KEYWORDS_SIN_DEPRESION = ["..."]      # Ejemplo: frases de bienestar
KEYWORDS_DEPRESION_LEVE = ["..."]     # Ejemplo: expresiones de tristeza moderada
KEYWORDS_DEPRESION_CLARA = ["..."]    # Ejemplo: ideación suicida o desesperanza fuerte

# Selección de grupo de keywords (modificar según necesidad)
KEYWORDS = KEYWORDS_DEPRESION_CLARA

# ---------- 2. EXCLUSIONES ----------
# Palabras o expresiones que deben descartarse:
# - Otros términos no relevantes
EXCLUSIONES = ["..."]

# ---------- 3. FUNCIONES ----------
def build_patterns(keywords):
    simple = [k for k in keywords if "_" not in k]
    cooc   = [k for k in keywords if "_" in k]
    re_simple = re.compile("|".join(map(re.escape, simple)), flags=re.IGNORECASE)
    return re_simple, cooc

def build_exclusion(exclusiones):
    return re.compile("|".join(map(re.escape, exclusiones)), flags=re.IGNORECASE)

def passes_cooc(text, cooc_list):
    text = text.lower()
    for cooc in cooc_list:
        parts = cooc.lower().split("_")
        if all(p in text for p in parts):
            return True
    return False

# ---------- 4. PIPELINE ----------
def main(output, max_tweets=50_000):
    re_simple, cooc = build_patterns(KEYWORDS)
    re_exc = build_exclusion(EXCLUSIONES)

    # Cargar dataset de ejemplo (se puede reemplazar con otro dataset si se requiere)
    ds = load_dataset("pysentimiento/spanish-tweets", split="train", streaming=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = None
        saved = 0
        for row in tqdm(ds, desc="Guardando tweets", total=max_tweets):
            if saved >= max_tweets:
                break

            text = row.get("text") or row.get("tweet") or ""
            if not text:
                continue

            # EXCLUSIONES
            if re_exc.search(text):
                continue

            # MATCH con palabras clave
            if re_simple.search(text) or passes_cooc(text, cooc):
                if writer is None:
                    fields = [k for k in ("id", "tweet_id", "created_at", "author_id", "lang", "text") if k in row]
                    if "text" not in fields: fields.append("text")
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()

                out_row = {k: row.get(k, "") for k in writer.fieldnames}
                writer.writerow(out_row)
                saved += 1

    print(f"✅ Guardados {saved} tweets en {output}")

# ---------- 5. CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extracción de tweets según categorías de depresión")
    ap.add_argument("--output", default=" dataset_tweets.csv", help="Archivo CSV de salida")
    ap.add_argument("--max", type=int, default=50_000, help="Número máximo de tweets a descargar")
    args = ap.parse_args()
    main(args.output, args.max)
