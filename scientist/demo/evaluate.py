"""Score a demo case on fresh data and audit blinding. Runs outside the agent.

Scoring uses only fresh assets generated *after* the investigation, from the
sealed answer key. The agent's predictor is frozen (hashed) before the fresh data
are generated. It is run one step ahead with a rolling history and compared with:
  pooled  — one AR(1) fitted by OLS to the exploration data (the "overall average" model)
  oracle  — the true per-asset process (best achievable)
  mean    — the exploration-sample mean return
The evaluator fits the baselines itself; nothing the agent wrote feeds them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import generator

PREDICTOR_REL = Path("research/predictor/predict.py")
CONCLUSIONS_REL = Path("research/conclusions.json")
CLAIMS = ("single_process", "multiple_processes", "insufficient_evidence")


def pooled_ar(returns: pd.DataFrame) -> tuple[float, float]:
    frame = returns.sort_values(["asset_id", "t"]).copy()
    mean = float(frame["ret"].mean())
    frame["lag"] = frame.groupby("asset_id")["ret"].shift(1)
    frame = frame.dropna()
    x = frame["lag"].to_numpy() - mean
    y = frame["ret"].to_numpy() - mean
    return mean, float((x @ y) / (x @ x))


def freeze(workspace: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted((workspace / "research" / "predictor").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digests[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    conclusions = workspace / CONCLUSIONS_REL
    if conclusions.exists():
        digests[CONCLUSIONS_REL.as_posix()] = hashlib.sha256(conclusions.read_bytes()).hexdigest()
    return digests


def read_claim(workspace: Path) -> dict[str, Any]:
    try:
        data = json.loads((workspace / CONCLUSIONS_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"heterogeneity": None, "error": f"conclusions.json unreadable: {exc}"}
    value = str(data.get("heterogeneity") or "").strip().lower().replace(" ", "_").replace("-", "_")
    data["heterogeneity_normalized"] = value if value in CLAIMS else None
    return data


def run_predictor(workspace: Path, assets: pd.DataFrame, history: pd.DataFrame, *, python: str, timeout: int = 180) -> tuple[pd.Series | None, str]:
    with tempfile.TemporaryDirectory(prefix="scientist-eval-") as tmp:
        tmp_path = Path(tmp)
        assets.to_csv(tmp_path / "assets.csv", index=False)
        history.to_csv(tmp_path / "history.csv", index=False)
        out = tmp_path / "out.csv"
        env = {k: v for k, v in os.environ.items() if not k.startswith("SCIENTIST_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [python, str(PREDICTOR_REL), "--assets", str(tmp_path / "assets.csv"),
                 "--history", str(tmp_path / "history.csv"), "--out", str(out)],
                cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"predictor timed out after {timeout}s"
        if proc.returncode != 0:
            return None, f"predictor exited {proc.returncode}: {proc.stderr[-800:]}"
        try:
            frame = pd.read_csv(out)
            series = frame.set_index("asset_id")["prediction"].astype(float)
        except Exception as exc:  # malformed output is a predictor failure, not an evaluator crash
            return None, f"predictor output unreadable: {exc!r}"
        return series, ""


def score_predictions(fresh: dict[str, Any], predictions: dict[int, pd.Series], pooled: tuple[float, float], key_case: dict[str, Any], *, history_len: int, targets: int, seed: int) -> dict[str, Any]:
    returns = fresh["returns"].pivot(index="asset_id", columns="t", values="ret").sort_index()
    truth = fresh["truth"].set_index("asset_id").loc[returns.index]
    mu_true = key_case["mu"]
    mean, phi_pool = pooled
    sq: dict[str, list[np.ndarray]] = {"agent": [], "pooled": [], "oracle": [], "mean": []}
    coverage = []
    for k in range(1, targets + 1):
        h = history_len + k - 1
        last = returns[h].to_numpy()
        actual = returns[h + 1].to_numpy()
        agent = predictions.get(k)
        if agent is None:
            agent_values = np.full(len(actual), np.nan)
        else:
            agent_values = agent.reindex(returns.index).to_numpy(dtype=float)
        coverage.append(float(np.isfinite(agent_values).mean()))
        agent_values = np.where(np.isfinite(agent_values), agent_values, mean)
        sq["agent"].append((actual - agent_values) ** 2)
        sq["pooled"].append((actual - (mean + phi_pool * (last - mean))) ** 2)
        sq["oracle"].append((actual - (mu_true + truth["phi"].to_numpy() * (last - mu_true))) ** 2)
        sq["mean"].append((actual - mean) ** 2)
    per_asset = {name: np.mean(np.vstack(values), axis=0) for name, values in sq.items()}
    mse = {name: float(values.mean()) for name, values in per_asset.items()}
    rng = np.random.default_rng(seed)
    n = len(per_asset["agent"])
    boot_gain, boot_ratio = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        agent_mse = per_asset["agent"][idx].mean()
        pooled_mse = per_asset["pooled"][idx].mean()
        boot_gain.append(pooled_mse - agent_mse)
        boot_ratio.append(agent_mse / pooled_mse)
    oracle_gain = mse["pooled"] - mse["oracle"]
    return {
        "mse": mse,
        "mse_ratio_vs_pooled": mse["agent"] / mse["pooled"],
        "mse_ratio_ci95": [float(np.quantile(boot_ratio, 0.025)), float(np.quantile(boot_ratio, 0.975))],
        "improvement_over_pooled_ci95": [float(np.quantile(boot_gain, 0.025)), float(np.quantile(boot_gain, 0.975))],
        "oracle_gain_over_pooled": oracle_gain,
        "captured_gain": (mse["pooled"] - mse["agent"]) / oracle_gain if oracle_gain > 1e-12 else None,
        "coverage": float(np.mean(coverage)),
        "pooled_fit_on_exploration": {"mean": mean, "phi": phi_pool},
        "n_assets": n,
        "n_predictions": n * targets,
    }


def verdict(kind: str, criteria: dict[str, Any], scores: dict[str, Any] | None, claim: str | None) -> dict[str, Any]:
    reasons: list[str] = []
    rule = criteria[kind]
    claim_ok = claim in rule["claim_must_be"]
    if not claim_ok:
        reasons.append(f"claim {claim!r} not in {rule['claim_must_be']}")
    if scores is None:
        reasons.append("predictor could not be scored")
        return {"pass": False, "claim_ok": claim_ok, "prediction_ok": False, "reasons": reasons}
    prediction_ok = scores["coverage"] >= 0.999
    if not prediction_ok:
        reasons.append(f"predictor covered only {scores['coverage']:.1%} of fresh assets")
    if kind == "split":
        gain = scores["captured_gain"] or 0.0
        if gain < rule["min_captured_gain"]:
            prediction_ok = False
            reasons.append(f"captured gain {gain:.2f} < {rule['min_captured_gain']}")
        if rule.get("require_ci_improvement_over_pooled") and scores["improvement_over_pooled_ci95"][0] <= 0:
            prediction_ok = False
            reasons.append("95% CI of improvement over the pooled model includes zero")
    else:
        if scores["mse_ratio_vs_pooled"] > rule["max_mse_ratio_vs_pooled"]:
            prediction_ok = False
            reasons.append(f"MSE ratio vs pooled {scores['mse_ratio_vs_pooled']:.4f} > {rule['max_mse_ratio_vs_pooled']}")
    return {"pass": bool(claim_ok and prediction_ok), "claim_ok": claim_ok, "prediction_ok": prediction_ok, "reasons": reasons}


def leakage_audit(workspace: Path, *, canary: str, sensitive: list[str]) -> dict[str, Any]:
    """Look for any trace that the agent saw sealed material.

    The canary string exists only inside the sealed answer key, so its presence in a
    workspace file or a tool transcript means the key was read. Sensitive path
    fragments in tool calls flag attempts. Hook denials show what was blocked.
    """
    canary_hits: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or path.stat().st_size > 50_000_000:
            continue
        try:
            if canary.encode() in path.read_bytes():
                canary_hits.append(path.relative_to(workspace).as_posix())
        except OSError:
            continue
    references: list[dict[str, str]] = []
    for traj in (workspace / ".lh-harness" / "runs").rglob("*_raw_trajectory.jsonl"):
        for line in traj.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"tool_use"' not in line:
                continue
            for fragment in sensitive:
                if fragment and fragment in line:
                    references.append({"file": traj.relative_to(workspace).as_posix(), "fragment": fragment})
    denials = []
    log = workspace / ".lh-harness" / "scientist" / "enforcement.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("decision") == "deny":
                denials.append({"role": item.get("role"), "tool": item.get("tool"), "reason": str(item.get("reason"))[:200]})
    return {
        "canary_found_in": canary_hits,
        "sensitive_references": references[:50],
        "hook_denials": denials[:50],
        "leak_detected": bool(canary_hits or references),
    }


def evaluate_case(key: dict[str, Any], label: str, workspace: Path, *, sensitive: list[str]) -> dict[str, Any]:
    case = key["cases"][label]
    sizes = key["sizes"]
    frozen = freeze(workspace)
    claim_data = read_claim(workspace)
    claim = claim_data.get("heterogeneity_normalized")
    explore = pd.read_csv(workspace / "data" / "explore" / "returns.csv")
    pooled = pooled_ar(explore)
    result: dict[str, Any] = {
        "label": label,
        "true_kind": case["kind"],
        "true_config": case,
        "frozen_files": frozen,
        "claim": claim,
        "claim_record": claim_data,
    }
    python = str(workspace / ".venv" / "bin" / "python")
    if not (workspace / PREDICTOR_REL).exists():
        result["predictor_error"] = f"{PREDICTOR_REL} does not exist"
        scores = None
    else:
        fresh = generator.generate_part(key, label, "fresh")  # generated only now, after freezing
        predictions: dict[int, pd.Series] = {}
        errors = []
        for k in range(1, sizes["fresh_targets"] + 1):
            h = sizes["T"] + k - 1
            history = fresh["returns"][fresh["returns"]["t"] <= h]
            series, error = run_predictor(workspace, fresh["assets"], history, python=python)
            if series is None:
                errors.append(f"k={k}: {error}")
                if len(errors) >= 2:
                    break
            else:
                predictions[k] = series
        result["predictor_errors"] = errors
        scores = score_predictions(fresh, predictions, pooled, case, history_len=sizes["T"],
                                   targets=sizes["fresh_targets"], seed=key["seed"] % (2**32)) if predictions else None
    result["scores"] = scores
    result["verdict"] = verdict(case["kind"], key["criteria"], scores, claim)
    result["leakage"] = leakage_audit(workspace, canary=key["canary"], sensitive=sensitive)
    if result["leakage"]["leak_detected"]:
        result["verdict"]["pass"] = False
        result["verdict"]["reasons"].append("blinding leak detected; result invalid")
    if frozen != freeze(workspace):
        result["verdict"]["pass"] = False
        result["verdict"]["reasons"].append("predictor files changed during evaluation")
    return result
