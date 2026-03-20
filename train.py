# train.py
# Phelix - Train and save the fraud detection model

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

from data.training_data import TRAINING_DATA

# ── 1. Clean Text ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# ── 2. Load Data ──────────────────────────────────────────────
df = pd.DataFrame(TRAINING_DATA)
df['clean'] = df['text'].apply(clean_text)
print(f"Data loaded: {len(df)} examples")

# ── 3. Split ──────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['intent']
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── 4. Vectorize ──────────────────────────────────────────────
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['clean'])
X_test = vectorizer.transform(test_df['clean'])
y_train = train_df['intent']
y_test = test_df['intent']

# ── 5. Train ──────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained!")

# ── 6. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {round(accuracy * 100, 1)}%")
print("\nFull Report:")
print(classification_report(y_test, y_pred))

# ── 7. Save ───────────────────────────────────────────────────
bundle = {
    "model": model,
    "vectorizer": vectorizer,
    "labels": sorted(df['intent'].unique().tolist())
}

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phelix_model.pkl')
joblib.dump(bundle, model_path)
print(f"Model saved to phelix_model.pkl")