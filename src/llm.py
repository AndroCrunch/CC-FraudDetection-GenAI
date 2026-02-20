from typing import Dict, Any

ANALYST_PROMPT = """You are a fraud risk analyst assistant.
You will receive a JSON evidence object.

Rules:
- Use ONLY fields from the evidence JSON.
- If information is missing, say "Not provided".
- Output MUST be valid JSON that matches the schema.

Schema:
{
  "analyst_explanation": {
    "summary": "string",
    "evidence_bullets": ["string", "..."],
    "top_drivers": [{"feature":"string","direction":"increase_risk|decrease_risk","reason":"string"}],
    "recommended_checks": ["string", "..."]
  },
  "plain_language_explanation": {
    "summary": "string",
    "why_it_matters": ["string", "..."],
    "next_steps": ["string", "..."]
  }
}
"""

def generate_explanations_with_llm(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace the body of this function with your real LLM call later.
    For MVP without API, we return a deterministic, grounded template
    that only uses fields from the evidence object.
    """
    drivers = evidence.get("top_model_drivers", [])[:3]
    derived = evidence.get("derived_signals", {})
    tx = evidence.get("transaction", {})
    risk = float(evidence.get("risk", {}).get("risk_score", 0.0))

    out = {
        "analyst_explanation": {
            "summary": f"Alert scored {risk:.3f}. Key drivers suggest elevated risk relative to learned patterns.",
            "evidence_bullets": [
                f"Amount={tx.get('amount', 0.0):.2f}, amt_robust_z={derived.get('amt_robust_z', 0.0):.2f}",
                f"velocity_10m={derived.get('velocity_10m', 1)}, is_new_device={derived.get('is_new_device', 0)}, is_new_ip={derived.get('is_new_ip', 0)}, is_geo_jump={derived.get('is_geo_jump', 0)}",
                f"merchant_fraud_rate={derived.get('merchant_fraud_rate', 0.0):.4f}, ip_fraud_rate={derived.get('ip_fraud_rate', 0.0):.4f}, device_fraud_rate={derived.get('device_fraud_rate', 0.0):.4f}"
            ],
            "top_drivers": [
                {
                    "feature": d.get("feature", "Not provided"),
                    "direction": d.get("direction", "increase_risk"),
                    "reason": "High absolute SHAP contribution in evidence.top_model_drivers"
                }
                for d in drivers
            ],
            "recommended_checks": [
                "Verify cardholder authentication outcome (Not provided in evidence).",
                "Check whether device_id/ip_id were previously associated with confirmed fraud in internal systems (Not provided).",
                "Review linked recent transactions for the same card_id around the same time."
            ]
        },
        "plain_language_explanation": {
            "summary": f"This transaction looks unusual enough to trigger a high-risk alert (score {risk:.2f}).",
            "why_it_matters": [
                "Some signals indicate behavior that differs from what is normally seen for similar transactions.",
                "There may be signs like unusual amount/tempo or a new access context (device/IP/geo), depending on the evidence."
            ],
            "next_steps": [
                "Confirm if the customer recognizes the transaction.",
                "Look for other recent transactions that share the same device, IP, or merchant."
            ]
        }
    }
    return out

def validate_json(obj: Dict[str, Any]) -> None:
    # Minimal validation (extend later)
    if "analyst_explanation" not in obj:
        raise ValueError("Missing analyst_explanation")
    if "plain_language_explanation" not in obj:
        raise ValueError("Missing plain_language_explanation")
