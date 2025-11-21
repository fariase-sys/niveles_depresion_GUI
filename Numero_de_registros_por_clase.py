import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_tweets_final.csv", sep=";")

conteo_clases = df["SENTIMENT"].value_counts().sort_index()

colores = ['#4CAF50', '#FFC107', '#F44336']  # verde, amarillo, rojo

plt.figure(figsize=(8,6))

bars = plt.bar(conteo_clases.index.astype(str), conteo_clases.values,
               color=colores, width=0.6)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval),
             ha="center", va="bottom", fontsize=11, color="black")

plt.xticks(conteo_clases.index.astype(str),
           ["0 - Sin depresión", "1 - Depresión leve", "2 - Depresión clara"],
           fontsize=11)
plt.ylabel("Número de registros", fontsize=12)
plt.xlabel("Clase", fontsize=12)

plt.tight_layout()
plt.show()
