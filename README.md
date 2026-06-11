# Mobile App Sentiment Analysis using Bidirectional LSTM

A deep learning NLP pipeline that automatically classifies mobile application reviews as **Positive** or **Negative**. Built on a Bidirectional LSTM architecture, the model learns contextual language patterns in user feedback — handling negation, sarcasm, and imbalanced real-world data to achieve strong classification performance.

---

## Problem Statement

Mobile apps receive thousands of reviews daily. Manually reading them to understand user sentiment is impossible at scale. This project builds an automated pipeline that classifies any review as positive or negative, enabling product teams to monitor user satisfaction without reading individual reviews.

---

## Key Features

- **Bidirectional LSTM** — reads sequences in both forward and backward directions, capturing context that unidirectional models miss (e.g. "fixed all the bugs" correctly classified as positive)
- **Robust preprocessing** — text tokenization, out-of-vocabulary (`<OOV>`) handling, and uniform padding
- **Overfitting protection** — L1/L2 kernel regularization, Dropout layers, and EarlyStopping with `restore_best_weights=True`
- **Imbalance handling** — achieves balanced precision/recall on a 74/26 positive/negative split without artificial class-weight injection

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data & Evaluation | Pandas, NumPy, scikit-learn |

---

## Model Architecture

```
Input → Embedding → Bidirectional LSTM → Dropout → Dense (L1/L2) → Sigmoid Output
```

- Vocabulary size: 5,000 most frequent tokens
- OOV token: `<OOV>` for unknown words at inference time
- Regularization: L1 + L2 kernel constraints on Dense layer
- Dropout: applied after LSTM layer
- Output: sigmoid activation for binary classification

---

## Results

Evaluated on a held-out test set of **5,184 reviews** (20% of dataset).

| Metric | Score |
|--------|-------|
| Test Accuracy | **87.52%** |
| Macro Average F1 | **0.83** |
| Negative F1 | 0.75 |
| Positive F1 | 0.92 |

```
              precision    recall  f1-score   support

    Negative       0.75      0.75      0.75      1305
    Positive       0.92      0.92      0.92      3879

    accuracy                           0.88      5184
   macro avg       0.83      0.83      0.83      5184
weighted avg       0.88      0.88      0.88      5184
```

> **Baseline comparison:** A standard dense feed-forward network on the same data capped at 75% accuracy — barely above the naive majority-class baseline. The Bidirectional LSTM raised this to 87.52%.

---

## Training Details

- Early stopping triggered at **Epoch 10**, rolling back to best weights at **Epoch 5**
- Confusion matrix shows symmetrical errors: 322 False Positives vs 325 False Negatives — the default 0.5 threshold required no adjustment
- Training stopped automatically before overfitting, with validation loss plateauing cleanly

---

## Wins & Challenges

### What worked well

**Breaking the baseline bottleneck**
The dataset is heavily imbalanced (74% positive). A dense feed-forward network capped at 75% — barely above naive guessing. The Bidirectional LSTM broke through this to 87.52% by learning sequential patterns rather than treating text as a bag of words.

**Contextual sarcasm and negation handling**
Early model iterations misclassified reviews like *"fixed all the bugs"* as negative due to the word "bugs". The LSTM's sequential memory learned word pairings in context, correctly flipping these reviews to positive with 71.35% confidence.

**Balanced errors without class weighting**
The final confusion matrix shows near-symmetrical errors (322 FP vs 325 FN), proving the model learned a genuinely balanced decision boundary rather than biasing toward the majority class.

**Clean early stopping**
`EarlyStopping` with `restore_best_weights=True` stopped training at Epoch 10 and automatically rolled back to the Epoch 5 weights — no manual intervention required.

### Challenges overcome

**Vocabulary noise control**
Thousands of unique typos, slang terms, and structural noise risked bloating the vocabulary. Resolved by enforcing a strict `num_words=5000` cap to isolate the highest-signal recurring patterns.

**Continuous vs binary evaluation conflicts**
Scikit-learn metrics crashed on raw sigmoid probability outputs (floats) instead of hard binary classes. Resolved by applying boolean threshold filtering — `.astype(int)` — to bridge Keras outputs with scikit-learn's evaluation pipeline.

**Overfitting pressure from recurrent layers**
Early iterations showed training accuracy climbing past 91% while validation loss drifted upward. Resolved by combining L1/L2 kernel weight constraints with Dropout layers to limit the expressive capacity of the recurrent layers.

---

## How to Reproduce

```bash
pip install tensorflow pandas numpy scikit-learn
```

Run the notebook:

```
sentiment_bilstm.ipynb
```

---

## Dataset

- **Source:** Google Play Store (Safaricom MyOne app)
- **Raw reviews scraped:** 50,000
- **After cleaning (min 3 words):** 25,918
- **Labels:** Positive (4–5 stars) / Negative (1–3 stars)
- **Split:** 80% train / 20% test (stratified)

---

## Known Limitations

- Negative recall of 0.75 means the model still misses 25% of negative reviews — an improvement over the Word2Vec + LightGBM baseline (0.64) but not yet production-grade for high-stakes negative feedback detection
- The model was trained on English-language reviews; Swahili and Sheng reviews may tokenize poorly with the current vocabulary
- Static vocabulary — the model cannot adapt to new slang or product-specific terms without retraining

---

## Next Steps

- Fine-tune DistilBERT on the same dataset to compare transformer vs LSTM performance
- Add SHAP token-level explainability to identify which words drive predictions
- Deploy as a FastAPI endpoint with a live Gradio demo
- Extend to multilingual classification using XLM-RoBERTa to handle Swahili and Sheng reviews
