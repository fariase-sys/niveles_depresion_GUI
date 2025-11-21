import pandas as pd
import re
from collections import Counter

df = pd.read_csv("dataset_tweets_sin_duplicados.csv", sep=";")

def limpiar_texto(texto):
    texto = str(texto).lower()                           # minúsculas
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto) # quitar links
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)         # quitar signos y números
    texto = re.sub(r"\s+", " ", texto).strip()            # quitar espacios extras
    return texto

df["CLEAN_TEXT"] = df["TEXT_TWEET"].apply(limpiar_texto)

for clase in sorted(df["SENTIMENT"].unique()):
    subset = df[df["SENTIMENT"] == clase]

    texto_unido = " ".join(subset["CLEAN_TEXT"])
    palabras = texto_unido.split()
    contador = Counter(palabras)
    top_500 = contador.most_common(500)
    df_top = pd.DataFrame(top_500, columns=["PALABRA", "CANTIDAD"])
    nombre_archivo = f"top500_clase{clase}.csv"
    df_top.to_csv(nombre_archivo, sep=";", index=False, encoding="utf-8-sig")

    print(f"✅ Archivo generado: {nombre_archivo}")
