# test_model.py
# Phelix - Model Evaluation Tests

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import ml_predict

# ── Test cases ────────────────────────────────────────────────
# Format: (input_text, expected_intent)

TEST_CASES = [
    # legitimate_inquiry
    ("what are the charges for international transfers", "legitimate_inquiry"),
    ("how do I check my account balance", "legitimate_inquiry"),
    ("can I open a domiciliary account", "legitimate_inquiry"),

    # suspicious_transfer
    ("transfer all my money to this account immediately", "suspicious_transfer"),
    ("I need to send money to 10 different accounts today", "suspicious_transfer"),
    ("process this transfer without the normal checks", "suspicious_transfer"),

    # account_takeover
    ("my password was changed and I did not change it", "account_takeover"),
    ("someone logged into my account from another country", "account_takeover"),
    ("I keep getting OTP codes I did not request", "account_takeover"),

    # phishing_attempt
    ("someone called claiming to be from the bank asking for my pin", "phishing_attempt"),
    ("I received a message saying my account will be closed if I don't verify", "phishing_attempt"),
    ("a caller asked me to read out the OTP sent to my phone", "phishing_attempt"),

    # dispute_claim
    ("I was debited twice for the same transaction", "dispute_claim"),
    ("my ATM withdrawal was unsuccessful but my account was debited", "dispute_claim"),
    ("I was charged for a service I did not use", "dispute_claim"),

    # unusual_activity
    ("there is a transaction on my account from a city I have never been to", "unusual_activity"),
    ("my account shows a purchase at a store I have never visited", "unusual_activity"),
    ("I see a crypto purchase on my account I did not make", "unusual_activity"),

    # account_recovery
    ("I forgot my internet banking password", "account_recovery"),
    ("my account is locked because I entered wrong pin too many times", "account_recovery"),
    ("I cannot remember my ATM pin", "account_recovery"),
]


# ── Run tests ─────────────────────────────────────────────────
def run_tests():
    print("=" * 60)
    print("PHELIX — MODEL EVALUATION TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    failed_cases = []

    for text, expected in TEST_CASES:
        intent, confidence = ml_predict(text)
        status = "✅ PASS" if intent == expected else "❌ FAIL"

        if intent == expected:
            passed += 1
        else:
            failed += 1
            failed_cases.append({
                "text": text,
                "expected": expected,
                "got": intent,
                "confidence": confidence
            })

        print(f"{status} | {confidence:5.1f}% | {intent:<25} | {text[:45]}")

    print("=" * 60)
    print(f"Results: {passed}/{len(TEST_CASES)} passed ({round(passed/len(TEST_CASES)*100)}%)")
    print("=" * 60)

    if failed_cases:
        print("\nFailed cases:")
        for case in failed_cases:
            print(f"\n  Input:    {case['text']}")
            print(f"  Expected: {case['expected']}")
            print(f"  Got:      {case['got']} ({case['confidence']}%)")

    return passed, failed


if __name__ == "__main__":
    run_tests()
