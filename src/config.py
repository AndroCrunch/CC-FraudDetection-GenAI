from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2

    # Synthetic enrichment sizes
    n_cards: int = 20000
    n_merchants: int = 3000
    n_devices: int = 60000
    n_ips: int = 80000
    n_geos: int = 500

    # Behavior parameters
    prob_new_device_per_txn: float = 0.02
    prob_new_ip_per_txn: float = 0.03
    prob_travel_geo_jump: float = 0.01

    # Alerting
    alert_threshold: float = 0.90  # probability threshold for alerts
    top_k_alerts: int = 200

