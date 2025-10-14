import pandas as pd

# Leer CSV separado por ";"
df = pd.read_csv("dataset_tweets_final.csv", sep=";")

# --------- 1. Contar duplicados por clase antes de borrar ---------
duplicados = df[df.duplicated(subset=["TEXT_TWEET"], keep=False)]
duplicados_por_clase = duplicados.groupby("SENTIMENT").size()

# --------- 2. Eliminar duplicados (manteniendo el primero) ---------
df_limpio = df.drop_duplicates(subset=["TEXT_TWEET"], keep="first")

# --------- 3. Contar totales después de limpiar ---------
total_tweets = len(df_limpio)
tweets_por_clase = df_limpio["SENTIMENT"].value_counts().sort_index()

# --------- 4. Mostrar resultados ---------
print("📌 Tweets duplicados eliminados por clase:")
for clase in sorted(duplicados_por_clase.index):
    print(f"   Clase {clase}: {duplicados_por_clase[clase]} eliminados")

print("\n✅ Totales después de limpiar duplicados:")
print(f"   Total de tweets: {total_tweets}")
for clase in sorted(tweets_por_clase.index):
    print(f"   Clase {clase}: {tweets_por_clase[clase]} tweets")

# --------- 5. Guardar CSV limpio ---------
df_limpio.to_csv("dataset_tweets_sin_duplicados.csv", sep=";", index=False, encoding="utf-8-sig")
print("\n💾 Archivo guardado como: dataset_tweets_sin_duplicados.csv")
