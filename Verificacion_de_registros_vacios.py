import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_tweets_final.csv", sep=";")

total_por_columna = df.shape[0]
totales = [total_por_columna] * len(df.columns)
vacios = df.isnull().sum().tolist()

colores_totales = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722']
colores_vacios  = ['#81C784', '#64B5F6', '#FFD54F', '#BA68C8', '#FF8A65']

x = range(len(df.columns))
bar_width = 0.35

plt.figure(figsize=(12,6))

bars1 = plt.bar([i - bar_width/2 for i in x], totales, width=bar_width,
                color=colores_totales[:len(df.columns)])

bars2 = plt.bar([i + bar_width/2 for i in x], vacios, width=bar_width,
                color=colores_vacios[:len(df.columns)])

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval),
             ha="center", va="bottom", fontsize=10, color="black")

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval),
             ha="center", va="bottom", fontsize=10, color="black")

plt.xticks(x, df.columns, rotation=20, fontsize=11)
plt.ylabel("Número de valores", fontsize=12)
plt.xlabel("CARACTERÍSTICAS", fontsize=12)
plt.title("Totales vs Valores Vacíos por Columna", fontsize=14, weight="bold")

plt.tight_layout()
plt.show()
