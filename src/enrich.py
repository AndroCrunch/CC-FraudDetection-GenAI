import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import Config

@dataclass
class EnrichmentArtifacts:
    card_id: np.ndarray
    merchant_id: np.ndarray
    device_id: np.ndarray
    ip_id: np.ndarray
    geo_id: np.ndarray

def enrich_transactions(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, EnrichmentArtifacts]:
    """"""
    Enriches credit card transactions with synthetic categorical IDs:
    card_id, merchant_id, device_id, ip_id, geo_id.

    Assumptions:
    - We don't have real entities in the public dataset, so we simulate.
    - Each transaction is assigned a card_id; merchants are biased; devices/IPs mostly stable per card.
    """"""
    rng = np.random.default_rng(cfg.seed)
    n = len(df)

    # Assign each row to a synthetic cardholder/card_id
    card_id = rng.integers(0, cfg.n_cards, size=n)

    # Merchants: heavy-tailed distribution (few merchants get lots of traffic)
    merchant_pop = np.arange(cfg.n_merchants)
    weights = 1.0 / (1.0 + merchant_pop)  # Zipf-ish
    weights = weights / weights.sum()
    merchant_id = rng.choice(merchant_pop, size=n, p=weights)

    # Each card has a "primary" device, ip, geo
    primary_device = rng.integers(0, cfg.n_devices, size=cfg.n_cards)
    primary_ip = rng.integers(0, cfg.n_ips, size=cfg.n_cards)
    primary_geo = rng.integers(0, cfg.n_geos, size=cfg.n_cards)

    device_id = primary_device[card_id].copy()
    ip_id = primary_ip[card_id].copy()
    geo_id = primary_geo[card_id].copy()

    # Occasionally new device/IP/geo jumps
    new_device_mask = rng.random(n) < cfg.prob_new_device_per_txn
    device_id[new_device_mask] = rng.integers(0, cfg.n_devices, size=new_device_mask.sum())

    new_ip_mask = rng.random(n) < cfg.prob_new_ip_per_txn
    ip_id[new_ip_mask] = rng.integers(0, cfg.n_ips, size=new_ip_mask.sum())

    travel_mask = rng.random(n) < cfg.prob_travel_geo_jump
    geo_id[travel_mask] = rng.integers(0, cfg.n_geos, size=travel_mask.sum())

    out = df.copy()
    out[""card_id""] = card_id
    out[""merchant_id""] = merchant_id
    out[""device_id""] = device_id
    out[""ip_id""] = ip_id
    out[""geo_id""] = geo_id

    return out, EnrichmentArtifacts(card_id, merchant_id, device_id, ip_id, geo_id)
