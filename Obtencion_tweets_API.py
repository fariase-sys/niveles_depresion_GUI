# ============================
# Librerías necesarias
# ============================
import requests
import pandas as pd
import time

# ============================
# Autenticación con X API
# ============================
BEARER_TOKEN = "BEARER_TOKEN"   #Bearer Token real
headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

# ============================
# Definición de keywords
# ============================
# Se definen tres grupos de palabras clave:
#   - Palabras sin depresión
#   - Palabras de depresión leve
#   - Palabras de depresión clara
# (Aquí se deben colocar manualmente las palabras que correspondan en cada categoría)

keywords_sin_depresion = ["..."]      # Ejemplo: frases positivas o de bienestar
keywords_depresion_leve = ["..."]     # Ejemplo: expresiones de tristeza moderada
keywords_depresion_clara = ["..."]    # Ejemplo: frases con ideación suicida o desesperanza profunda

# Selección del conjunto de keywords a utilizar (modificar según la categoría deseada)
keywords = keywords_sin_depresion

# Construcción del query (se unen palabras con OR, en español y sin retweets)
query = " OR ".join([f'"{kw}"' for kw in keywords]) + " lang:es -is:retweet"

# ============================
# Configuración del endpoint de búsqueda
# ============================
search_url = "https://api.twitter.com/2/tweets/search/recent"
query_params = {
    "query": query,
    "tweet.fields": "id,text,author_id,created_at,lang",
    "max_results": 100   # máximo permitido por request
}

# ============================
# Variables de control
# ============================
tweets_data = []
max_requests = 50     #Número máximo de solicitudes (≈ 5000 tweets si max_results=100)
requests_done = 0
next_token = None

# ============================
# Bucle para obtener tweets
# ============================
while requests_done < max_requests:
    if next_token:
        query_params["next_token"] = next_token

    response = requests.get(search_url, headers=headers, params=query_params)

    if response.status_code == 200:
        data = response.json()
        tweets = data.get("data", [])
        tweets_data.extend(tweets)

        meta = data.get("meta", {})
        next_token = meta.get("next_token")
        requests_done += 1

        print(f"Solicitud {requests_done} completada, {len(tweets)} tweets obtenidos")

        if not next_token:
            break

        time.sleep(2)  # Pausa entre solicitudes

    elif response.status_code == 429:
        print("⚠️ Límite alcanzado. Esperando 60 segundos...")
        time.sleep(60)
    else:
        print(f"Error: {response.text}")
        break

# ============================
# Guardar resultados en CSV
# ============================
df = pd.DataFrame(tweets_data)

if not df.empty:
    # Renombrar columnas y seleccionar solo las necesarias
    df = df.rename(columns={
        "id": "ID_TWEET",
        "text": "TEXT_TWEET",
        "author_id": "AUTHOR_ID",
        "created_at": "CREATE_AT"
    })[["ID_TWEET", "TEXT_TWEET", "AUTHOR_ID", "CREATE_AT"]]

    df.to_csv("tweets_categoria.csv", index=False, encoding="utf-8-sig")
    print("✅ Archivo guardado: dataset_tweets.csv")
else:
    print("⚠️ No se recolectaron tweets.")
