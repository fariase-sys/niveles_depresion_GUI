# =============================
# 1) Preparación e instalación
# =============================
!pip install --quiet --upgrade transformers datasets scikit-learn accelerate safetensors

import os
os.environ["WANDB_DISABLED"] = "true"   # desactiva Weights & Biases logging

import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil

# =============================
# 2) Cargar dataset ya disponible
# =============================
csv_filename = "dataset_tweets_normalizados.csv"
df = pd.read_csv(csv_filename, dtype=str)

# Validar columnas necesarias
required_cols = {"CLEAN_TEXT", "SENTIMENT"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"El CSV debe contener las columnas: {required_cols}")

# Seleccionar solo texto limpio y labels
df = df[["CLEAN_TEXT", "SENTIMENT"]].rename(columns={"CLEAN_TEXT":"text", "SENTIMENT":"labels"})
df = df.dropna(subset=["text","labels"]).copy()
df["text"] = df["text"].astype(str)
df["labels"] = df["labels"].astype(int)

print("✅ Registros disponibles:", len(df))
num_labels = df["labels"].nunique()
print("✅ Número de clases detectadas:", num_labels)

# =============================
# 3) Tokenizer
# =============================
tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# =============================
# 4) Guardar tensores completos antes del split
# =============================
full_ds = Dataset.from_pandas(df[["text","labels"]])
full_ds = full_ds.map(tokenize_batch, batched=True)

full_tokenized = pd.DataFrame(full_ds)
full_tokenized.to_csv("tweets_tensores_completo.csv", sep=";", index=False)

# =============================
# 5) Split estratificado y guardar CSV
# =============================
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["labels"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["labels"], random_state=42)

print("📊 Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

train_df.to_csv("Train.csv", sep=";", index=False)
val_df.to_csv("Validation.csv", sep=";", index=False)
test_df.to_csv("Test.csv", sep=";", index=False)

# =============================
# 6) Convertir a Dataset y tokenizar splits
# =============================
train_ds = Dataset.from_pandas(train_df[["text","labels"]])
val_ds   = Dataset.from_pandas(val_df[["text","labels"]])
test_ds  = Dataset.from_pandas(test_df[["text","labels"]])

train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds   = val_ds.map(tokenize_batch, batched=True)
test_ds  = test_ds.map(tokenize_batch, batched=True)

# Mantener solo las columnas necesarias
keep_cols = ["input_ids","attention_mask","labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds   = val_ds.remove_columns([c for c in val_ds.column_names   if c not in keep_cols])
test_ds  = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

all_tokenized = pd.concat([
    pd.DataFrame(train_ds).assign(split="train"),
    pd.DataFrame(val_ds).assign(split="val"),
    pd.DataFrame(test_ds).assign(split="test")
])
all_tokenized.to_csv("tweets_tokenizados.csv", sep=";", index=False)

# =============================
# 7) Modelo BETO
# =============================
model = BertForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-uncased",
    num_labels=num_labels
)

# =============================
# 8) Métricas
# =============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# =============================
# 9) Argumentos de entrenamiento
# =============================
common_kwargs = dict(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none"
)

try:
    training_args = TrainingArguments(**{**common_kwargs, "evaluation_strategy":"epoch", "save_strategy":"epoch"})
except TypeError:
    training_args = TrainingArguments(**{**common_kwargs, "eval_strategy":"epoch", "save_strategy":"epoch"})

print("✅ TrainingArguments creados con éxito.")

# =============================
# 10) Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =============================
# 11) Entrenamiento
# =============================
train_result = trainer.train()
print("🏋️ Entrenamiento finalizado.")

# =============================
# 12) Evaluación en test
# =============================
metrics = trainer.evaluate(test_ds)
print("📊 Resultados en test:", metrics)

# =============================
# 13) Guardar modelo final
# =============================
output_dir = "./Modelo_deteccion_niveles_depresion"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
shutil.make_archive("Modelo_deteccion_niveles_depresion", 'zip', output_dir)
print("✅ Modelo guardado y comprimido en Modelo_deteccion_niveles_depresion.zip")
