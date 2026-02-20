import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from .config import Config
from .enrich import enrich_transactions
from .features import add_derived_features, compute_train_only_rates, apply_rates

FEATURE_EXCLUDE = {"Class"}

def train_pipeline(csv_path: str, cfg: Config, model_out: str, enc_out: str) -> None:
    df = pd.read_csv(csv_path)

    df, _ = enrich_transactions(df, cfg)
    df = add_derived_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["Class"]
    )

    rates = compute_train_only_rates(train_df)
    train_df = apply_rates(train_df, rates)
    test_df = apply_rates(test_df, rates)

    features = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]

    X_train = train_df[features]
    y_train = train_df["Class"].astype(int)

    X_test = test_df[features]
    y_test = test_df["Class"].astype(int)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=cfg.seed
    )
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, p_test)
    roc_auc = roc_auc_score(y_test, p_test)

    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    joblib.dump(model, model_out)
    joblib.dump({"features": features, "rates": rates}, enc_out)


