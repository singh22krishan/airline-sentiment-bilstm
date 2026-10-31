# Airline Sentiment BiLSTM

An NLP/deep-learning portfolio project for classifying airline-related text into **negative**, **neutral**, and **positive** sentiment using a Bidirectional LSTM built with TensorFlow/Keras.

## Highlights

- Text cleaning for URLs, mentions, hashtags, punctuation, and whitespace
- Stratified 70/15/15 train/validation/test split
- Keras tokenization with out-of-vocabulary handling
- Class weighting to address sentiment imbalance
- 128-dimensional embeddings and a Bidirectional LSTM
- Early stopping and validation monitoring
- Confusion matrix, classification report, macro metrics, and weighted metrics
- Reproducible random seed

## Recorded academic-project results

The supplied project outputs record these held-out test results:

| Metric | Result |
| --- | ---: |
| Accuracy | 76.64% |
| Macro precision | 69.96% |
| Macro recall | 75.25% |
| Macro F1 | 71.94% |
| Weighted F1 | 77.39% |

Per-class F1 scores recorded in the supplied classification report were **0.85 negative**, **0.60 neutral**, and **0.71 positive**. These values are retained from the original academic run; they are not claimed as independently re-trained results for this cleaned portfolio version.

## Architecture

```text
Token sequence (length 40)
        ↓
Embedding (10,000 × 128)
        ↓
Bidirectional LSTM (64 units/direction)
        ↓
Dropout (0.5)
        ↓
Dense (32, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (3, Softmax)
```

The cleaned model definition contains **1,383,043 trainable parameters**.

## Privacy-conscious portfolio version

The original classroom notebook contained student identifiers, local computer paths, notebook execution outputs, and a raw social-media dataset. Those are intentionally excluded here. The public repository contains source code and aggregate evaluation artifacts only.

The original work was completed in an academic group context. This repository is a cleaned portfolio presentation of the implementation and results and does not rewrite the historical classroom collaboration.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place an authorized dataset at `data/Tweets.csv` with `text` and `airline_sentiment` columns, then run:

```bash
python src/train.py
```

Generated model files are intentionally git-ignored. The training script saves a modern `.keras` model plus JSON tokenizer/label artifacts locally.

## Portfolio skills demonstrated

Python · TensorFlow/Keras · NLP · Bidirectional LSTM · Text Preprocessing · Imbalanced Classification · Scikit-learn · Model Evaluation · Data Visualization
