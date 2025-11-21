import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_tweets_final.csv", sep=";")

df["NUM_PALABRAS"] = df["TEXT_TWEET"].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(12,6))
plt.hist(df["NUM_PALABRAS"], bins=range(1, df["NUM_PALABRAS"].max() + 2),
         color="#2196F3", edgecolor="black", alpha=0.7)

plt.xlabel("Número de palabras", fontsize=12)
plt.ylabel("Cantidad de tweets", fontsize=12)
plt.xticks(range(1, df["NUM_PALABRAS"].max() + 1, 2))  # ticks cada 2 palabras (ajustable)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
