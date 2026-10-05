"""Generic synthetic panel generator for the blinded demonstration.

Every dataset is a panel of assets with static characteristics and an AR(1)
return series per asset:

    r[i,t] = mu + phi[type_i] * (r[i,t-1] - mu) + sigma_i * sqrt(1 - phi[type_i]^2) * e[i,t]

`phi` is a list with one or more entries (one per latent type) and `type_probs`
gives their frequencies. A characteristic named in `proxy` is shifted by
+/- `proxy_shift` according to the type (kept at unit variance overall); the other
characteristics are pure noise. Which dataset gets which configuration is decided
by `make_answer_key`, from a seed that is stored only in the sealed answer key.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURES = ("size", "liquidity", "value", "age")
SECTORS = ("A", "B", "C", "D", "E", "F")
T_PERIODS = 40
N_EXPLORE = 300
N_CONFIRM = 150
N_FRESH = 600
FRESH_TARGETS = 10  # rolling one-step-ahead targets after a 40-period history

# Pass criteria, fixed before any run and copied into every answer key.
CRITERIA = {
    "split": {
        "claim_must_be": ["multiple_processes"],
        "min_captured_gain": 0.5,
        "require_ci_improvement_over_pooled": True,
    },
    "single": {
        "claim_must_be": ["single_process", "insufficient_evidence"],
        "max_mse_ratio_vs_pooled": 1.01,
    },
}


@dataclass
class CaseConfig:
    kind: str  # "split" or "single" (hidden from the investigating agent)
    phi: list[float]
    type_probs: list[float]
    mu: float
    vol: float
    vol_dispersion: float
    proxy: str | None
    proxy_shift: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseConfig":
        return cls(**data)


def make_answer_key(seed: int | None = None) -> dict[str, Any]:
    seed = int(seed if seed is not None else secrets.randbits(63))
    rng = np.random.default_rng(seed)
    phi_a = float(rng.uniform(0.35, 0.5))
    phi_b = -float(rng.uniform(0.35, 0.5))
    p_a = float(rng.uniform(0.4, 0.6))
    split = CaseConfig(
        kind="split", phi=[phi_a, phi_b], type_probs=[p_a, 1 - p_a],
        mu=float(rng.uniform(0.0002, 0.0008)), vol=0.015, vol_dispersion=0.2,
        proxy=str(rng.choice(FEATURES)), proxy_shift=float(rng.uniform(0.5, 0.7)),
    )
    single = CaseConfig(
        kind="single", phi=[float(rng.uniform(0.02, 0.1))], type_probs=[1.0],
        mu=float(rng.uniform(0.0002, 0.0008)), vol=0.015, vol_dispersion=0.2,
        proxy=None, proxy_shift=0.0,
    )
    labels = ["dataset-1", "dataset-2"]
    order = [split, single] if rng.random() < 0.5 else [single, split]
    return {
        "seed": seed,
        "canary": f"SEALED-CANARY-{secrets.token_hex(8)}",
        "cases": {label: config.to_dict() for label, config in zip(labels, order)},
        "criteria": CRITERIA,
        "sizes": {"T": T_PERIODS, "explore": N_EXPLORE, "confirm": N_CONFIRM, "fresh": N_FRESH, "fresh_targets": FRESH_TARGETS},
    }


def simulate(config: CaseConfig, *, n_assets: int, periods: int, seed: int, id_prefix: str) -> dict[str, Any]:
    """Return observable frames plus the latent truth for one sample of assets."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(config.type_probs, dtype=float)
    types = rng.choice(len(probs), size=n_assets, p=probs / probs.sum())
    phi = np.asarray(config.phi, dtype=float)[types]
    features = {name: rng.standard_normal(n_assets) for name in FEATURES}
    if config.proxy and len(config.phi) > 1:
        shift = config.proxy_shift
        sign = np.where(types == 0, 1.0, -1.0)
        features[config.proxy] = shift * sign + np.sqrt(1 - shift**2) * rng.standard_normal(n_assets)
    sigma = config.vol * np.exp(config.vol_dispersion * rng.standard_normal(n_assets))
    innovations = rng.standard_normal((n_assets, periods))
    returns = np.empty((n_assets, periods))
    previous = config.mu + sigma * rng.standard_normal(n_assets)  # stationary start
    for t in range(periods):
        current = config.mu + phi * (previous - config.mu) + sigma * np.sqrt(1 - phi**2) * innovations[:, t]
        returns[:, t] = current
        previous = current
    ids = [f"{id_prefix}{i:04d}" for i in range(n_assets)]
    assets = pd.DataFrame({"asset_id": ids, "sector": rng.choice(SECTORS, size=n_assets),
                           **{name: np.round(values, 4) for name, values in features.items()}})
    panel = pd.DataFrame({
        "asset_id": np.repeat(ids, periods),
        "t": np.tile(np.arange(1, periods + 1), n_assets),
        "ret": np.round(returns.reshape(-1), 7),
    })
    truth = pd.DataFrame({"asset_id": ids, "type": types, "phi": phi, "sigma": sigma})
    return {"assets": assets, "returns": panel, "truth": truth}


def sample_seed(key: dict[str, Any], label: str, part: str) -> int:
    offsets = {"explore": 11, "confirm": 23, "fresh": 37}
    return int((key["seed"] + 1_000_003 * (labels_index(label) + 1) + offsets[part]) % (2**63))


def labels_index(label: str) -> int:
    return int(label.rsplit("-", 1)[-1])


def generate_part(key: dict[str, Any], label: str, part: str) -> dict[str, Any]:
    config = CaseConfig.from_dict(key["cases"][label])
    sizes = key["sizes"]
    n = {"explore": sizes["explore"], "confirm": sizes["confirm"], "fresh": sizes["fresh"]}[part]
    periods = sizes["T"] + (sizes["fresh_targets"] if part == "fresh" else 0)
    prefix = {"explore": "X", "confirm": "C", "fresh": "F"}[part]
    return simulate(config, n_assets=n, periods=periods, seed=sample_seed(key, label, part), id_prefix=prefix)
