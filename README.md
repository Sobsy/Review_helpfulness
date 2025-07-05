# 🧠 Review Helpfulness Prediction (Magazines)

This project builds a **hybrid deep learning model** to predict the **helpfulness score** (0 to 1) of magazine reviews using:

- 📝 **Review Text** (title + body)
- 🔍 **Hand-crafted features** (e.g., readability, rating, punctuation)
- 🤖 **BERT + Bi-GRU** for deep language understanding

---

## 🚀 Project Overview

Given a dataset of magazine reviews with fields like rating, review text, and helpful votes, we:

- **Preprocess text** and extract linguistic + structural features
- **Use PEGASUS summarization** for long reviews
- **Convert text to embeddings using BERT**
- **Model sequence dynamics with Bi-GRU**
- **Fuse with numerical features**
- **Train a regression model to predict helpfulness scores**

---

## 📊 Dataset Fields

Used columns from the dataset:

- `rating`: User's rating (1 to 5)
- `title`: Title of the review
- `text`: Full review content
- `helpful_vote`: Number of helpful votes (scaled to [0, 1])
- `verified_purchase`: Whether the user was a verified buyer

---

## 🧱 Model Architecture

```
[Text] ──▶ [PEGASUS Summary > BERT > Bi-GRU] ─┐
                                             ├─▶ [Concat + Self-Attention] ─▶ [MLP] ─▶ Helpfulness Score
[Features: readability, rating, etc.] ────────┘
```

**Key Components:**
- **PEGASUS**: Abstractive summarization for reviews >128 tokens
- **BERT**: Deep semantic embeddings
- **Bi-GRU**: Temporal structure modeling
- **Handcrafted Features**: Readability, rating, punctuation, etc.
- **Self-Attention Layer**: Focuses on combined features
- **Regression Output**: Predicts a score ∈ [0, 1]

---

## 🧪 Sample Features

Hand-crafted features used:

| Feature | Description |
|--------|-------------|
| `length` | Number of words |
| `readability` | Flesch Reading Ease score |
| `rating` | Normalized (1–5) |
| `verified_purchase` | Binary |
| `punctuation` | Count of `!` and `?` |

---

---

## 📈 Evaluation

- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Evaluation on 90/10 train-test split

---

## 📦 Inference Example

```python
text = "A fascinating and well-researched piece!"
features = [12, 78.5, 1.0, 1, 2]  # length, readability, rating, verified, punctuation
score = predict_helpfulness(text, features)
print("Predicted helpfulness:", round(score, 3))
```
