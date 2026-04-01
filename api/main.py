# main.py
# Phelix - Fraud Detection API

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Train or load model on startup ────────────────────────────
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phelix_model.pkl")

def train_model():
    from data.training_data import TRAINING_DATA
    df = pd.DataFrame(TRAINING_DATA)
    df['clean'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean'])
    y = df['intent']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "labels": sorted(df['intent'].unique().tolist())
    }
    joblib.dump(bundle, MODEL_PATH)
    print("Model trained and saved!")
    return bundle

def load_or_train():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        return joblib.load(MODEL_PATH)
    else:
        print("No model found — training now...")
        return train_model()

bundle = load_or_train()

# ── Helper functions ──────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def predict(text):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    cleaned = clean_text(text)
    vector = bundle["vectorizer"].transform([cleaned])
    intent = bundle["model"].predict(vector)[0]
    proba = bundle["model"].predict_proba(vector)[0]
    confidence = round(float(max(proba)) * 100, 1)

    if confidence >= 75:
        return {
            "intent": intent,
            "confidence": confidence,
            "input": text,
            "method": "ml_model"
        }

    # LLM fallback
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""You are a fraud detection assistant for a bank.

Classify the following customer message into exactly one of these intents:
- legitimate_inquiry
- suspicious_transfer
- account_takeover
- phishing_attempt
- dispute_claim
- unusual_activity
- account_recovery

Customer message: "{text}"

Reply with ONLY the intent label. Nothing else."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        llm_intent = response.choices[0].message.content.strip().lower()
        return {
            "intent": llm_intent,
            "confidence": confidence,
            "input": text,
            "method": "llm_fallback"
        }
    except Exception:
        return {
            "intent": intent,
            "confidence": confidence,
            "input": text,
            "method": "ml_model"
        }

# ── Actions ───────────────────────────────────────────────────
ACTIONS = {
    "legitimate_inquiry":  "Route to customer service queue",
    "suspicious_transfer": "Hold transaction and flag for review",
    "account_takeover":    "Lock account and alert security team",
    "phishing_attempt":    "Warn customer and log incident",
    "dispute_claim":       "Open dispute ticket and notify disputes team",
    "unusual_activity":    "Flag for fraud analyst review",
    "account_recovery":    "Route to identity verification team"
}

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "product": "Phelix",
        "description": "Fraud Detection Intent Classifier",
        "version": "1.0.0"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Phelix API is running"})

@app.route("/intents")
def intents():
    return jsonify({"intents": list(ACTIONS.keys())})

@app.route("/predict", methods=["POST"])
def predict_intent():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No data received."}), 400

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field."}), 400

    text = data["text"].strip()

    if not text:
        return jsonify({"error": "'text' field cannot be empty."}), 400

    try:
        result = predict(text)
        result["recommended_action"] = ACTIONS.get(result["intent"], "Route to general support")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
