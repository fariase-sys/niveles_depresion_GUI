import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_tweets_final.csv", sep=";")

df["NUM_PALABRAS"] = df["TEXT_TWEET"].astype(str).apply(lambda x: len(x.split()))

colores = {
    0: "#4CAF50",  # Verde
    1: "#FFC107",  # Amarillo
    2: "#F44336"   # Rojo
}

resultados = {}
for clase in sorted(df["SENTIMENT"].unique()):
    subset = df[df["SENTIMENT"] == clase]
    moda = subset["NUM_PALABRAS"].mode()[0]
    cantidad = (subset["NUM_PALABRAS"] == moda).sum()
    resultados[clase] = (moda, cantidad)

clases = list(resultados.keys())
modas = [resultados[c][0] for c in clases]
cantidades = [resultados[c][1] for c in clases]
colores_barras = [colores[c] for c in clases]

plt.figure(figsize=(10,6))
bars = plt.bar(clases, cantidades, color=colores_barras)

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(cantidades)*0.02),
             f"{cantidades[i]} tweets\n({modas[i]} palabras)",
             ha="center", va="bottom", fontsize=10, color="black")

plt.xlabel("Clases (SENTIMENT)", fontsize=12)
plt.ylabel("Número de Tweets", fontsize=12)
plt.xticks(clases, ["0: Sin depresión", "1: Depresión leve", "2: Depresión clara"], fontsize=11)

plt.ylim(0, max(cantidades) * 1.2)

plt.tight_layout()
plt.show()
