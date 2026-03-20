# main.py
# Phelix - Fraud Detection API

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model.predict import predict

app = Flask(__name__)
CORS(app)

ACTIONS = {
    "legitimate_inquiry":  "Route to customer service queue",
    "suspicious_transfer": "Hold transaction and flag for review",
    "account_takeover":    "Lock account and alert security team",
    "phishing_attempt":    "Warn customer and log incident",
    "dispute_claim":       "Open dispute ticket and notify disputes team",
    "unusual_activity":    "Flag for fraud analyst review",
    "account_recovery":    "Route to identity verification team"
}

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
    return jsonify({
        "intents": list(ACTIONS.keys())
    })

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

if __name__ == "__main__":
    print("Starting Phelix API on http://localhost:5000")
    app.run(debug=True, port=5000)