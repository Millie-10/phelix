# predict.py
# Phelix - Prediction pipeline with LLM fallback

import os
import re
import joblib
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment variables ────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Load model ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "phelix_model.pkl")
bundle = joblib.load(MODEL_PATH)

# ── Confidence threshold ──────────────────────────────────────
# Below this score the LLM takes over
CONFIDENCE_THRESHOLD = 75.0

# ── Clean text ────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# ── ML prediction ─────────────────────────────────────────────
def ml_predict(text):
    cleaned = clean_text(text)
    vector = bundle["vectorizer"].transform([cleaned])
    intent = bundle["model"].predict(vector)[0]
    proba = bundle["model"].predict_proba(vector)[0]
    confidence = round(float(max(proba)) * 100, 1)
    return intent, confidence

# ── LLM fallback ──────────────────────────────────────────────
def llm_predict(text):
    prompt = f"""You are a fraud detection assistant for a bank.
    
Classify the following customer message into exactly one of these intents:
- legitimate_inquiry: normal customer service question
- suspicious_transfer: urgent or unusual transfer request
- account_takeover: someone else accessing the account
- phishing_attempt: scam or social engineering attempt
- dispute_claim: incorrect charge or billing error
- unusual_activity: transaction inconsistent with customer behaviour
- account_recovery: forgotten password or locked account

Customer message: "{text}"

Reply with ONLY the intent label. Nothing else."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20
    )

    intent = response.choices[0].message.content.strip().lower()
    return intent

# ── Main predict function ─────────────────────────────────────
def predict(text):
    # Step 1 — try the fast ML model first
    intent, confidence = ml_predict(text)

    # Step 2 — if confident enough return immediately
    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "intent": intent,
            "confidence": confidence,
            "input": text,
            "method": "ml_model"
        }

    # Step 3 — low confidence, ask the LLM
    print(f"Low confidence ({confidence}%) — escalating to LLM...")
    llm_intent = llm_predict(text)

    return {
        "intent": llm_intent,
        "confidence": confidence,
        "input": text,
        "method": "llm_fallback"
    }


# ── Test it ───────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "what are the charges for international transfers",
        "I want to transfer all my money right now no questions",
        "someone logged into my account from another country",
        "there is a transaction on my account I cannot explain",
        "I was debited twice for the same transaction",
        "I forgot my internet banking password",
    ]

    print("Testing Phelix prediction pipeline...\n")
    for text in tests:
        result = predict(text)
        print(f"Input:      {text[:60]}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Method:     {result['method']}")
        print()
