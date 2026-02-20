import json
import numpy as np
import pandas as pd
from datetime import datetime

def top_shap_features(shap_row: np.ndarray, feature_names: list[str], k: int = 8):
    idx = np.argsort(np.abs(shap_row))[::-1][:k]
    out = []
    for i in idx:
        out.append({
            ""feature"": feature_names[i],
            ""shap_value"": float(shap_row[i]),
            ""direction"": ""increase_risk"" if shap_row[i] > 0 else ""decrease_risk""
        })
    return out

def build_evidence_row(row: pd.Series, risk_score: float, shap_row: np.ndarray, feature_names: list[str]) -> dict:
    tx = {
        ""time"": float(row[""Time""]),
        ""amount"": float(row[""Amount""]),
        ""card_id"": int(row[""card_id""]),
        ""merchant_id"": int(row[""merchant_id""]),
        ""device_id"": int(row[""device_id""]),
        ""ip_id"": int(row[""ip_id""]),
        ""geo_id"": int(row[""geo_id""]),
    }

    derived = {
        ""is_new_device"": int(row.get(""is_new_device"", 0)),
        ""is_new_ip"": int(row.get(""is_new_ip"", 0)),
        ""is_geo_jump"": int(row.get(""is_geo_jump"", 0)),
        ""velocity_10m"": int(row.get(""velocity_10m"", 1)),
        ""amt_robust_z"": float(row.get(""amt_robust_z"", 0.0)),
        ""merchant_fraud_rate"": float(row.get(""merchant_fraud_rate"", 0.0)),
        ""ip_fraud_rate"": float(row.get(""ip_fraud_rate"", 0.0)),
        ""device_fraud_rate"": float(row.get(""device_fraud_rate"", 0.0)),
    }

    return {
        ""alert_id"": f""ALERT-{int(row.name)}"",
        ""generated_at_utc"": datetime.utcnow().isoformat() + ""Z"",
        ""risk"": {""risk_score"": float(risk_score)},
        ""transaction"": tx,
        ""derived_signals"": derived,
        ""top_model_drivers"": top_shap_features(shap_row, feature_names, k=8)
    }

def write_jsonl(records: list[dict], out_path: str):
    with open(out_path, ""w"", encoding=""utf-8"") as f:
        for r in records:
            f.write(json.dumps(r) + ""\n"")
