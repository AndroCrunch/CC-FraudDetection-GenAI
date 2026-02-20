import joblib
import shap
import pandas as pd

def compute_shap_for_alerts(model_path: str, enc_path: str, df: pd.DataFrame, max_background: int = 2000):
    model = joblib.load(model_path)
    enc = joblib.load(enc_path)
    features = enc[""features""]

    X = df[features]

    bg = X.sample(min(len(X), max_background), random_state=42)
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation=""tree_path_dependent"")

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, features
