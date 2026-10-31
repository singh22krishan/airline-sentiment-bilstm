from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEED = 42
MAX_WORDS = 10_000
MAX_LEN = 40

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Tweets.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(MAX_LEN,), dtype="int32"),
            Embedding(input_dim=MAX_WORDS, output_dim=128),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ],
        name="airline_sentiment_bilstm",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/Tweets.csv. Add a dataset containing 'text' and "
            "'airline_sentiment' columns before running training."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"text", "airline_sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["text", "airline_sentiment"]].dropna().copy()
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    plt.figure(figsize=(6, 4))
    order = df["airline_sentiment"].value_counts().index
    sns.countplot(x="airline_sentiment", data=df, order=order, color="skyblue")
    plt.title("Class Distribution - Airline Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=300)
    plt.close()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["airline_sentiment"])
    X = df["clean_text"].to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def encode(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train_pad = encode(X_train)
    X_val_pad = encode(X_val)
    X_test_pad = encode(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(len(encoder.classes_))
    with open(OUTPUT_DIR / "model_summary.txt", "w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=12,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves_lstm.png", dpi=300)
    plt.close()

    probabilities = model.predict(X_test_pad, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics_lstm.csv", index=False)

    report = classification_report(
        y_test, predictions, target_names=encoder.classes_, zero_division=0
    )
    (OUTPUT_DIR / "classification_report_lstm.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - BiLSTM")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_lstm.png", dpi=300)
    plt.close()

    model.save(MODEL_DIR / "airline_sentiment_bilstm.keras")
    (MODEL_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (MODEL_DIR / "labels.json").write_text(
        json.dumps(encoder.classes_.tolist(), indent=2), encoding="utf-8"
    )

    print(pd.DataFrame([metrics]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
