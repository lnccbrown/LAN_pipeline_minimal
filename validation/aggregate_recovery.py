#!/usr/bin/env python3
"""Turn a directory of recovery shards into one verdict, for any model.

The recipe is written out in `recovery_designs.py`; this file is the part that
turns a pile of fits into a pass/fail. Nothing here knows what model it is
reading — shards carry their own model name and the cells are keyed by it, so
one directory can hold a whole catalogue.

Coverage is the statistic that tests the *likelihood*: a calibrated one puts
the truth inside its 94% interval 94% of the time, however wide that interval
is. Contraction is the statistic that tests the *design*: it says how much the
data narrowed the posterior relative to the prior. Reading them together is
what separates the two failure modes:

                    narrow posterior          wide posterior
    covers truth    identifiable, correct     unidentifiable, but honest
    misses truth    THE NETWORK IS WRONG      wrong and vague

So coverage is gated and contraction is only ever reported. A wide-but-covering
posterior is the model telling the truth about what the data support, and
failing a network for it would be punishing honesty.

Attribution needs a reference that contains no network, and there are two,
tried in that order:

  1. **The analytical arm**, where the model has one. It shares data and priors
     with the network arm, so a shortfall both arms show is the design's
     identifiability limit while one only the network shows is the network's.
  2. **The ladder**, for the majority of models that have no analytical form.
     A design limit relaxes when you add trials or conditions; a broken
     likelihood does not. So a shortfall that a richer rung repairs is charged
     to the design, and one that is flat across the whole ladder is charged to
     the network — with the weaker attribution said out loud in the failure
     text, because it is weaker.

Usage:

    uv run --group validate python validation/aggregate_recovery.py \\
        --shard-dir results/ --out results/recovery_report.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent))
import recovery_designs as rd  # noqa: E402

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


def _key(shard: dict) -> tuple[str, str, str]:
    """(model, likelihood, design) — the cell prefix a shard belongs to.

    `model` is part of the key so one shard directory can hold a whole
    catalogue without cells from different models colliding. Shards written
    before the model field existed all came from ddm_sdv.
    """
    return (
        shard.get("model", "ddm_sdv"),
        shard.get("likelihood", "?"),
        shard.get("design", "?"),
    )


def summarise(shards: list[dict]) -> dict:
    """Per (model, likelihood, design, parameter) recovery statistics."""
    cells: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    errors: list[dict] = []
    diverged: dict[tuple[str, str, str], int] = defaultdict(int)
    attempted: dict[tuple[str, str, str], int] = defaultdict(int)

    for shard in shards:
        key2 = _key(shard)
        attempted[key2] += 1
        if "error" in shard:
            errors.append(
                {
                    "model": key2[0],
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
    for (model, likelihood, design, label), records in sorted(cells.items()):
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
        summary[f"{model}|{likelihood}|{design}|{label}"] = entry

    return {
        "cells": summary,
        "errors": errors,
        "excluded_for_divergences": {"|".join(k): v for k, v in diverged.items()},
        "attempted": {"|".join(k): v for k, v in attempted.items()},
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


# With no analytical arm to compare against, a bias rate still has an absolute
# bar: under a calibrated posterior |z| > 2 happens about 5% of the time, so a
# quarter of the datasets flagged is far outside sampling noise.
UNREFERENCED_MAX_BIAS_RATE = 0.25


def _richer_rungs(design_name: str) -> list[str]:
    """Ladder rungs with at least as much data AND design, and more of one.

    This is the second reference in the recipe. It is what stands in for the
    analytical arm on the models that have none, and it works because the two
    failure modes respond differently to it: an identifiability limit is
    relieved by more trials or more conditions, and a wrong likelihood is not.
    """
    here = rd.DESIGNS.get(design_name)
    if here is None:
        return []
    return [
        name
        for name, other in rd.DESIGNS.items()
        if name != design_name
        and other.n_trials >= here.n_trials
        and other.n_conditions >= here.n_conditions
    ]


def verdict(summary: dict) -> tuple[bool, list[str]]:
    """Judge each network arm against nominal coverage and a no-network reference.

    Two ways to fail, both about calibration rather than sharpness:
      * coverage below the binomial band around nominal, and
      * a bias rate worse than the reference on the same cell.

    A cell the analytical arm also fails is reported, not charged to the
    network. Where there is no analytical arm the ladder answers instead, and
    the resulting failure text says so — a ladder-only attribution is weaker
    evidence and should read as weaker.
    """
    failures: list[str] = []
    cells = summary["cells"]

    for key, entry in cells.items():
        model, likelihood, design, label = key.split("|")
        if likelihood == "analytical" or entry["coverage"] is None:
            continue
        reference = cells.get(f"{model}|analytical|{design}|{label}")

        low, _ = entry["coverage_band"]
        if entry["coverage"] < low:
            failures.extend(
                _explain_coverage(
                    cells, key, entry, reference, low, model, likelihood, label
                )
            )

        if reference and reference["bias_rate"] is not None:
            if entry["bias_rate"] > reference["bias_rate"] + 0.15:
                failures.append(
                    f"{model}/{design}/{label}: bias rate {entry['bias_rate']:.2f} vs "
                    f"analytical {reference['bias_rate']:.2f}"
                )
        elif entry["bias_rate"] is not None:
            if entry["bias_rate"] > UNREFERENCED_MAX_BIAS_RATE:
                failures.append(
                    f"{model}/{design}/{label}: bias rate {entry['bias_rate']:.2f} "
                    f"above the {UNREFERENCED_MAX_BIAS_RATE:.2f} bar for a model "
                    "with no analytical arm"
                )

    return not failures, failures


def _explain_coverage(
    cells, key, entry, reference, low, model, likelihood, label
) -> list[str]:
    """Decide whether a coverage shortfall is the network's fault, and say why."""
    design = key.split("|")[2]
    if reference is not None and reference["coverage"] is not None:
        if reference["coverage"] < low:
            return []  # the analytical arm misses it too: not the network
        return [
            f"{model}/{design}/{label}: coverage {entry['coverage']:.2f} below the "
            f"{low:.2f} floor while analytical reaches {reference['coverage']:.2f}"
        ]

    # No no-network reference for this cell. Ask the ladder instead: if a
    # richer rung recovers, the shortfall tracks the design.
    for rung in _richer_rungs(design):
        higher = cells.get(f"{model}|{likelihood}|{rung}|{label}")
        if higher and higher["coverage"] is not None:
            rung_low, _ = higher["coverage_band"]
            if higher["coverage"] >= rung_low:
                return []
    return [
        f"{model}/{design}/{label}: coverage {entry['coverage']:.2f} below the "
        f"{low:.2f} floor, flat across the ladder and with no analytical arm to "
        "compare against — likely the network, but unconfirmed"
    ]


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
