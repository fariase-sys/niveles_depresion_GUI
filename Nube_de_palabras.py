import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

MASK_PATH = "imagen.png"              # imagen para la forma de la nube
CSV_PATH  = "top500_clase0.csv"       # archivo CSV con columnas PALABRA;CANTIDAD
OUT_FILE  = "nube_4k.png"             # salida PNG en 4K
SVG_FILE  = "nube_4k.svg"             # salida SVG vectorial
PDF_FILE  = "nube_4k.pdf"             # salida PDF vectorial

df = pd.read_csv(CSV_PATH, sep=";")
frecuencias = dict(zip(df["PALABRA"], df["CANTIDAD"]))

img = Image.open(MASK_PATH).convert("RGB")
arr = np.array(img)

gris = arr.mean(axis=2)
mask_bool = gris > 180   # zona blanca = nube
mask = mask_bool.astype(np.uint8) * 255

px_blancos = mask.sum() // 255
if px_blancos < 5000:
    scale = 3
    new_size = (img.width*scale, img.height*scale)
    img  = img.resize(new_size, Image.LANCZOS)
    mask = Image.fromarray(mask).resize(new_size, Image.NEAREST)
    mask = np.array(mask)

print(f"Píxeles útiles (máscara): {mask.sum()//255}")

img_viva = ImageEnhance.Color(img).enhance(1.8)
img_viva = ImageEnhance.Brightness(img_viva).enhance(1.2)
arr_viva = np.array(img_viva)
color_gen = ImageColorGenerator(arr_viva)

wc = WordCloud(
    background_color=None,
    mode="RGBA",
    mask=mask,
    max_words=500,
    relative_scaling=0.5,
    min_font_size=2,
    color_func=color_gen,
    contour_width=0,
    scale=2
).generate_from_frequencies(frecuencias)

fig, ax = plt.subplots(figsize=(img.width/50, img.height/50), dpi=600)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
plt.tight_layout(pad=0)

plt.savefig(OUT_FILE, bbox_inches="tight", pad_inches=0, dpi=600, transparent=True)
plt.savefig(PDF_FILE, bbox_inches="tight", pad_inches=0, dpi=600, transparent=True)

svg_content = wc.to_svg(embed_font=True)
with open(SVG_FILE, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"✅ Guardado en máxima calidad: {OUT_FILE}, {SVG_FILE}, {PDF_FILE}")
plt.show()
