#!/usr/bin/env python3
"""Turn a directory of recovery shards into one verdict.

Coverage is the statistic that tests the *network*: a calibrated likelihood
puts the truth inside its 94% interval 94% of the time, however wide that
interval is. Contraction is the statistic that tests the *design*: it says how
much the data narrowed the posterior relative to the prior. Reading them
together is what separates the two failure modes:

                    narrow posterior          wide posterior
    covers truth    identifiable, correct     unidentifiable, but honest
    misses truth    THE NETWORK IS WRONG      wrong and vague

So coverage is gated and contraction is only ever reported. A wide-but-covering
posterior is the model telling the truth about what the data support, and
failing a network for it would be punishing honesty.

The analytical arm anchors the comparison. It shares data and priors with the
network arms and contains no network, so a shortfall both arms show is the
design's identifiability limit, while one only a network shows is the network's.

Usage:

    uv run --group validate python validation/aggregate_recovery.py \\
        --shard-dir results/ --out results/recovery_report.json
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import typer

logger = logging.getLogger("aggregate_recovery")
app = typer.Typer(add_completion=False)

NOMINAL_COVERAGE = 0.94

# A fit that did not converge is evidence about the sampler, not about the
# network, so it is excluded rather than counted as a failure. Thresholds are
# the conventional ones; the divergence bar matches the spine's rlssm skill.
MAX_RHAT = 1.01
MIN_ESS = 400.0
MAX_DIVERGENCE_RATE = 0.05

# |z| > 2: the posterior mean sits more than two posterior sds from the truth.
# Same rule as HSSM's own scripts/addm_parameter_recovery.py.
MAX_ABS_Z = 2.0


def _binomial_band(n: int, p: float = NOMINAL_COVERAGE, n_se: float = 2.0):
    """Two-SE band around nominal coverage for `n` datasets.

    With 20 datasets the SE is 5.3 points, so a point estimate of "coverage is
    0.90" is not evidence of anything. The gate has to be a band or it is noise
    amplification.
    """
    if n == 0:
        return (0.0, 1.0)
    se = math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - n_se * se), min(1.0, p + n_se * se))


def load_shards(shard_dir: Path) -> list[dict]:
    shards = []
    for path in sorted(shard_dir.glob("recovery_*.json")):
        try:
            shards.append(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            logger.warning(f"skipping unreadable shard {path.name}: {e}")
    return shards


def summarise(shards: list[dict]) -> dict:
    """Per (likelihood, design, parameter) recovery statistics."""
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    errors: list[dict] = []
    diverged: dict[tuple[str, str], int] = defaultdict(int)
    attempted: dict[tuple[str, str], int] = defaultdict(int)

    for shard in shards:
        key2 = (shard.get("likelihood", "?"), shard.get("design", "?"))
        attempted[key2] += 1
        if "error" in shard:
            errors.append(
                {
                    "design": shard.get("design"),
                    "likelihood": shard.get("likelihood"),
                    "dataset_index": shard.get("dataset_index"),
                    "error": shard["error"],
                }
            )
            continue
        if shard["sampler"]["divergence_rate"] > MAX_DIVERGENCE_RATE:
            diverged[key2] += 1
            continue
        for label, rec in shard["parameters"].items():
            if "error" in rec:
                continue
            cells[(*key2, label)].append(rec)

    summary = {}
    for (likelihood, design, label), records in sorted(cells.items()):
        usable = [
            r for r in records if r["rhat"] <= MAX_RHAT and r["ess_bulk"] >= MIN_ESS
        ]
        n = len(usable)
        entry = {
            "n_fits": len(records),
            "n_converged": n,
            "coverage": None,
            "coverage_band": None,
            "bias_rate": None,
            "mean_abs_z": None,
            "median_contraction": None,
            "truth_recovered_corr": None,
        }
        if n:
            covered = sum(r["covered"] for r in usable)
            entry["coverage"] = covered / n
            entry["coverage_band"] = list(_binomial_band(n))
            entry["bias_rate"] = sum(abs(r["z"]) > MAX_ABS_Z for r in usable) / n
            entry["mean_abs_z"] = sum(abs(r["z"]) for r in usable) / n
            contractions = sorted(r["contraction"] for r in usable)
            entry["median_contraction"] = contractions[n // 2]
            entry["truth_recovered_corr"] = _corr(
                [r["truth"] for r in usable], [r["mean"] for r in usable]
            )
        summary[f"{likelihood}|{design}|{label}"] = entry

    return {
        "cells": summary,
        "errors": errors,
        "excluded_for_divergences": {f"{k[0]}|{k[1]}": v for k, v in diverged.items()},
        "attempted": {f"{k[0]}|{k[1]}": v for k, v in attempted.items()},
    }


def _corr(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, or None when either side is effectively constant."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-24 or syy < 1e-24:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def verdict(summary: dict) -> tuple[bool, list[str]]:
    """Judge each network arm against nominal coverage and the analytical arm.

    Two ways to fail, both about calibration rather than sharpness:
      * coverage below the binomial band around nominal, and
      * a bias rate worse than the analytical arm on the same cell.
    A cell the analytical arm also fails is reported, not charged to the
    network.
    """
    failures: list[str] = []
    cells = summary["cells"]

    for key, entry in cells.items():
        likelihood, design, label = key.split("|")
        if likelihood == "analytical" or entry["coverage"] is None:
            continue
        reference = cells.get(f"analytical|{design}|{label}")

        low, _ = entry["coverage_band"]
        if entry["coverage"] < low:
            ref_cov = reference["coverage"] if reference else None
            if ref_cov is not None and ref_cov < low:
                continue  # the analytical arm misses it too: not the network
            failures.append(
                f"{design}/{label}: coverage {entry['coverage']:.2f} below the "
                f"{low:.2f} floor"
                + (f" while analytical reaches {ref_cov:.2f}" if ref_cov else "")
            )

        if reference and reference["bias_rate"] is not None:
            if entry["bias_rate"] > reference["bias_rate"] + 0.15:
                failures.append(
                    f"{design}/{label}: bias rate {entry['bias_rate']:.2f} vs "
                    f"analytical {reference['bias_rate']:.2f}"
                )

    return not failures, failures


@app.command()
def main(
    shard_dir: Path = typer.Option(..., exists=True, file_okay=False),
    out: Path | None = typer.Option(
        None, help="[default: <shard-dir>/recovery_report.json]"
    ),
    log_level: str = typer.Option("WARNING"),
):
    """Aggregate recovery shards into a report and a verdict."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.WARNING))
    shards = load_shards(shard_dir)
    if not shards:
        raise typer.BadParameter(f"No recovery_*.json under {shard_dir}")

    summary = summarise(shards)
    passed, failures = verdict(summary)
    report = {
        "schema_version": 1,
        "n_shards": len(shards),
        "nominal_coverage": NOMINAL_COVERAGE,
        "thresholds": {
            "max_rhat": MAX_RHAT,
            "min_ess": MIN_ESS,
            "max_divergence_rate": MAX_DIVERGENCE_RATE,
            "max_abs_z": MAX_ABS_Z,
        },
        "passed": passed,
        "failures": failures,
        **summary,
    }

    destination = out or (shard_dir / "recovery_report.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "report": str(destination),
                "n_shards": len(shards),
                "failures": failures,
            }
        )
    )


if __name__ == "__main__":
    app()
