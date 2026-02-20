import numpy as np
import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features:
      - velocity_10m: number of tx in last 10 minutes per card_id
      - amt_robust_z: robust z-score of Amount per card_id using MAD
      - is_new_device / is_new_ip: first time a (card_id, device_id/ip_id) combination appears
      - is_geo_jump: geo differs from the previous geo for the same card_id

    Requires columns: Time, Amount, card_id, device_id, ip_id, geo_id
    """
    out = df.copy()

    # Sort by card/time for temporal features
    out = out.sort_values(["card_id", "Time"]).reset_index(drop=True)

    # new device/ip flags per card
    out["is_new_device"] = out.groupby(["card_id", "device_id"]).cumcount().eq(0).astype(int)
    out["is_new_ip"] = out.groupby(["card_id", "ip_id"]).cumcount().eq(0).astype(int)

    # geo jump vs previous transaction geo per card
    prev_geo = out.groupby("card_id")["geo_id"].shift(1)
    out["is_geo_jump"] = (prev_geo.notna() & (out["geo_id"] != prev_geo)).astype(int)

    # robust z-score of amount per card using MAD
    grp = out.groupby("card_id")["Amount"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (np.abs(s - np.median(s))).median() + 1e-6)
    out["amt_robust_z"] = (out["Amount"] - med) / (1.4826 * mad)

    # velocity_10m (600 seconds window) per card using two-pointer per card
    times = out["Time"].to_numpy()
    velocity = np.ones(len(out), dtype=np.int32)

    for cid, idxs in out.groupby("card_id").indices.items():
        idxs = np.array(idxs)
        t = times[idxs]
        j = 0
        for k in range(len(idxs)):
            while t[k] - t[j] > 600:
                j += 1
            velocity[idxs[k]] = k - j + 1

    out["velocity_10m"] = velocity
    return out

def compute_train_only_rates(train_df: pd.DataFrame) -> dict:
    """
    Computes fraud rates per merchant/ip/device on TRAIN ONLY (to avoid leakage).
    Returns dicts: merchant_rate, ip_rate, device_rate, global_rate.
    """
    y = train_df["Class"].astype(int)

    merchant_rate = train_df.groupby("merchant_id")["Class"].mean().to_dict()
    ip_rate = train_df.groupby("ip_id")["Class"].mean().to_dict()
    device_rate = train_df.groupby("device_id")["Class"].mean().to_dict()
    global_rate = float(y.mean())

    return {
        "merchant_rate": merchant_rate,
        "ip_rate": ip_rate,
        "device_rate": device_rate,
        "global_rate": global_rate
    }

def apply_rates(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """
    Applies train-only rate mappings to any dataframe (train/test).
    Unseen IDs fall back to global_rate.
    """
    out = df.copy()
    g = rates["global_rate"]
    out["merchant_fraud_rate"] = out["merchant_id"].map(rates["merchant_rate"]).fillna(g)
    out["ip_fraud_rate"] = out["ip_id"].map(rates["ip_rate"]).fillna(g)
    out["device_fraud_rate"] = out["device_id"].map(rates["device_rate"]).fillna(g)
    return out
