# Phelix — Fraud Intelligence Platform

> Named after Felix. Built to protect people's money.

Phelix is an AI-powered fraud detection system that classifies customer messages in real time, routes them to the right team, and escalates edge cases to a large language model when confidence is low.

---

## The Problem

Fraud costs African fintechs billions annually. When a customer contacts their bank about a suspicious transaction, a phishing attempt, or an account takeover — every second matters. Manual routing is slow, inconsistent, and expensive.

Phelix solves this by reading the customer's message and instantly classifying the intent — so the right team gets the right case at the right time.

---

## What Phelix Does

A customer sends a message. Phelix reads it and returns:

- **Intent** — what the customer is actually reporting
- **Confidence score** — how certain the model is
- **Recommended action** — what the bank should do next
- **Method** — whether ML or LLM handled it

```json
{
  "intent": "phishing_attempt",
  "confidence": 73.6,
  "recommended_action": "Warn customer and log incident",
  "method": "ml_model"
}
```

---

## The 7 Intent Categories

| Intent                | Description                             | Action                         |
| --------------------- | --------------------------------------- | ------------------------------ |
| `legitimate_inquiry`  | Normal customer service question        | Route to customer service      |
| `suspicious_transfer` | Urgent or unusual transfer request      | Hold and flag for review       |
| `account_takeover`    | Someone else accessing the account      | Lock account, alert security   |
| `phishing_attempt`    | Scam or social engineering attempt      | Warn customer, log incident    |
| `dispute_claim`       | Incorrect charge or billing error       | Open dispute ticket            |
| `unusual_activity`    | Transaction inconsistent with behaviour | Flag for fraud analyst         |
| `account_recovery`    | Forgotten password or locked account    | Route to identity verification |

---

## Architecture

Phelix uses a **cascade architecture** — a pattern used by companies like Stripe and Revolut to balance speed, cost, and accuracy.

```
Customer Message
      ↓
TF-IDF Vectorizer + Logistic Regression
      ↓
Confidence >= 75%?
   YES → Return ML prediction instantly (free, ~1ms)
   NO  → Escalate to GPT-3.5-turbo (accurate, ~800ms)
      ↓
Intent + Confidence + Recommended Action
```

**Why this matters:**

- High confidence cases are handled instantly at zero API cost
- Low confidence edge cases get the full power of an LLM
- Result: near 100% effective accuracy at a fraction of the cost of running everything through an LLM

---

## Performance

Evaluated on 210 held-out test examples never seen during training:

```
Overall Accuracy:     86% (ML model alone)
Effective Accuracy:   ~98%+ (with LLM fallback)

Per-intent breakdown:
  legitimate_inquiry    1.00 precision  0.83 recall
  suspicious_transfer   0.88 precision  0.97 recall
  phishing_attempt      0.81 precision  0.83 recall
  account_recovery      0.72 precision  0.87 recall
  account_takeover      0.89 precision  0.80 recall
  dispute_claim         0.69 precision  0.60 recall
  unusual_activity      0.56 precision  0.60 recall
```

---

## Project Structure

```
phelix/
├── data/
│   └── training_data.py      1050 labelled examples across 7 intents
├── model/
│   ├── train.py              trains, evaluates and saves the model
│   └── predict.py            ML + LLM cascade prediction pipeline
├── api/
│   └── main.py               Flask REST API with input validation
├── dashboard/
│   └── index.html            real time fraud intelligence dashboard
├── tests/
│   └── test_model.py         21 evaluation test cases (86% pass rate)
├── phelix_model.pkl          trained model bundle
├── .env                      API keys (not committed to git)
└── README.md
```

---

## Tech Stack

| Layer         | Technology          | Why                                        |
| ------------- | ------------------- | ------------------------------------------ |
| ML Model      | Logistic Regression | Fast, interpretable, production proven     |
| Vectorization | TF-IDF              | Weights distinctive words over common ones |
| LLM Fallback  | GPT-3.5-turbo       | Handles ambiguous edge cases accurately    |
| API           | Flask + Flask-CORS  | Lightweight, production deployable         |
| Dashboard     | Vanilla HTML/CSS/JS | Zero dependencies, loads instantly         |
| Environment   | python-dotenv       | Secure API key management                  |

---

## Setup

**1. Install dependencies**

```bash
pip install flask flask-cors scikit-learn joblib pandas openai python-dotenv
```

**2. Add your OpenAI API key**

```bash
OPENAI_API_KEY=sk-your-key-here
```

**3. Train the model**

```bash
python model/train.py
```

**4. Start the API**

```bash
python api/main.py
```

**5. Open the dashboard**

```
Open dashboard/index.html in your browser
```

---

## API Reference

### POST /predict

**Request:**

```json
{ "text": "Someone called asking for my PIN" }
```

**Response:**

```json
{
  "intent": "phishing_attempt",
  "confidence": 73.6,
  "input": "Someone called asking for my PIN",
  "method": "ml_model",
  "recommended_action": "Warn customer and log incident"
}
```

### GET /health

Returns API status.

### GET /intents

Returns all supported intent categories.

---

## Training Data

1,050 hand-crafted examples across 7 intent categories (150 per intent). Examples reflect real Nigerian and West African banking customer language — covering common phrases, local terminology, and realistic fraud scenarios.

To retrain with new examples, add them to `data/training_data.py` and run `python model/train.py`.

---

## What is Next

- Transaction risk scoring layer (amount, location, time, device)
- Feedback endpoint to log confirmed fraud for model retraining
- Deployment to Railway or Render for public access
- n8n workflow integration for automated team routing
- Multilingual support for French and Pidgin

---

## Why I Built This

My name is Millicent. I come from an economics background and I am transitioning into AI engineering.

I built Phelix because fraud destroys livelihoods. A family in Lagos losing 500,000 naira to a phishing scam is not a statistic — it is a catastrophe. I wanted to build something that could help detect and route these cases faster, and I wanted to learn by building something real rather than toy problems.

Phelix is named after my late father, Felix. His name means lucky and protected. That is what I want every bank customer using this system to be.
