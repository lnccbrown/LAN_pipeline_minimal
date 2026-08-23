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

# ...but exclusion cannot be unlimited, or attrition becomes a way to pass. A
# cell with a handful of survivors gets a coverage band so wide that anything
# clears it, so below this many usable fits the cell is INCONCLUSIVE: it is
# neither passed nor quietly skipped, and it cannot serve as a reference or as
# a ladder rescue for anything else.
MIN_CONVERGED_FITS = 10

# |z| > 2: the posterior mean sits more than two posterior sds from the truth.
# Same rule as HSSM's own scripts/addm_parameter_recovery.py.
MAX_ABS_Z = 2.0

# With no reference arm a bias rate still has an absolute bar: under a
# calibrated posterior |z| > 2 happens about 5% of the time, so a quarter of
# the datasets flagged is far outside sampling noise.
UNREFERENCED_MAX_BIAS_RATE = 0.25

# Coverage only means something when the likelihood actually moved the prior.
# Truths are drawn from a box inset 10% per side while the prior spans the full
# box, so the PRIOR's own 94% interval contains every possible truth: a network
# whose likelihood is constant scores coverage 1.00 on every parameter. That is
# not the "wide but honest" case the recipe protects — an honest posterior is
# wide because the data are uninformative, a degenerate one is wide because the
# likelihood contributed nothing, and only the second makes coverage vacuous.
# So contraction is still never gated as a *quality* bar; it is gated only at
# the degenerate end.
MAX_CONTRACTION = 0.95
# And against a reference: a network whose posterior is much wider than the
# exact likelihood's on the same data is discarding information that is there.
MAX_CONTRACTION_RATIO = 1.5

# A dataset in which a response category never occurred cannot identify the
# parameters only that category speaks to. That is a fact about the draw, not
# about the likelihood, so such fits are excluded like divergent ones.
MIN_CHOICE_SHARE = 0.02

# Likelihoods that contain no network. These are the reference arms: they are
# never judged as the thing under test, and either may anchor an attribution.
REFERENCE_LIKELIHOODS = ("analytical", "blackbox")

# Family-wise error rate for the whole report. A full ladder produces dozens of
# (model, likelihood, design, parameter) cells and the run fails if ANY of them
# fails, so an uncorrected 2-SE band would fail a perfectly calibrated network
# most of the time. The floor is a Sidak-corrected exact binomial quantile.
FAMILY_ALPHA = 0.05


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact, stdlib only."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _binomial_band(n: int, p: float = NOMINAL_COVERAGE, n_tests: int = 1):
    """One-sided exact floor on coverage, and the symmetric upper bound.

    Exact rather than normal-approximate because the approximation is invalid
    where it is used: at n=20 and p=0.94, n*p*(1-p) = 1.13.

    `n_tests` applies a Sidak correction. Without it the gate is a coin flip on
    a good network -- each cell has its own false-alarm probability and the run
    fails if any single cell fails, so the family-wise rate compounds with the
    number of parameters times the number of ladder rungs.
    """
    if n == 0:
        return (0.0, 1.0)
    alpha = 1 - (1 - FAMILY_ALPHA) ** (1 / max(n_tests, 1))
    # Largest k whose lower tail is still within alpha; anything at or below it
    # is implausible under nominal coverage.
    reject_at = -1
    for k in range(n + 1):
        if _binomial_cdf(k, n, p) <= alpha:
            reject_at = k
        else:
            break
    low = (reject_at + 1) / n
    # Upper end is reported, never gated -- coverage above nominal is not a
    # defect. Kept so the report shows the interval, not a bare floor.
    high = min(1.0, p + 2 * math.sqrt(p * (1 - p) / n))
    return (low, high)


def load_shards(shard_dir: Path) -> list[dict]:
    shards = []
    for path in sorted(shard_dir.glob("recovery_*.json")):
        try:
            shards.append(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            logger.warning(f"skipping unreadable shard {path.name}: {e}")
    return shards


def _key(shard: dict) -> tuple[str, str, str]:
    """(model, arm, design) -- the cell prefix a shard belongs to.

    `model` is part of the key so one shard directory can hold a whole
    catalogue without cells from different models colliding. `arm` is the
    likelihood kind plus, for a network, which ONNX was fitted: two candidate
    networks for the same model and design are different arms and must not be
    pooled into one coverage number. Shards written before those fields existed
    all came from a single ddm_sdv network.
    """
    return (
        shard.get("model", "ddm_sdv"),
        shard.get("arm") or shard.get("likelihood", "?"),
        # The design identity, which carries WHICH parameter varies across
        # conditions: `L1_n500@v` and `L1_n500@sv` are different designs and
        # pooling their shared-parameter cells would average two different
        # experiments into one coverage number.
        shard.get("design_id") or shard.get("design", "?"),
    )


def _base_param(label: str) -> str:
    """`v[2]` -> `v`. The parameter behind a per-condition label.

    Needed because the condition parameter is a scalar at L0 and a vector at
    L1, so it is labelled `v` on one rung and `v[0]`..`v[3]` on the next. The
    ladder has to cross exactly that boundary -- it is the parameter the extra
    conditions exist to rescue -- so rung lookups go by base name.
    """
    return label.split("[", 1)[0]


def summarise(shards: list[dict]) -> dict:
    """Per (model, arm, design, parameter) recovery statistics."""
    cells: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    errors: list[dict] = []
    diverged: dict[tuple[str, str, str], int] = defaultdict(int)
    degenerate: dict[tuple[str, str, str], int] = defaultdict(int)
    attempted: dict[tuple[str, str, str], int] = defaultdict(int)
    likelihood_of: dict[tuple[str, str, str], str] = {}

    for shard in shards:
        key2 = _key(shard)
        attempted[key2] += 1
        likelihood_of[key2] = shard.get("likelihood", "?")
        if "error" in shard or "parameters" not in shard:
            errors.append(
                {
                    "model": key2[0],
                    "arm": key2[1],
                    "design": shard.get("design"),
                    "likelihood": shard.get("likelihood"),
                    "dataset_index": shard.get("dataset_index"),
                    "error": shard.get("error", "shard has no parameters block"),
                }
            )
            continue
        if shard.get("sampler", {}).get("divergence_rate", 0.0) > MAX_DIVERGENCE_RATE:
            diverged[key2] += 1
            continue
        # The data checks _sanity records are read here -- that is what
        # "recorded rather than raised, the aggregator decides" has to mean.
        share = shard.get("data", {}).get("min_choice_share")
        if share is not None and share < MIN_CHOICE_SHARE:
            degenerate[key2] += 1
            continue
        for label, rec in shard["parameters"].items():
            if "error" in rec:
                continue
            cells[(*key2, label)].append(rec)

    # Every gated cell is one test; the floor is corrected for how many there
    # are. Counted before scoring so each cell sees the same correction.
    n_tests = sum(
        1
        for (_, arm, _, _) in cells
        if arm.split("@", 1)[0] not in REFERENCE_LIKELIHOODS
    )

    summary = {}
    for (model, arm, design, label), records in sorted(cells.items()):
        usable = [
            r
            for r in records
            if r["rhat"] == r["rhat"]  # NaN fails this: single-chain fits go out
            and r["rhat"] <= MAX_RHAT
            and r["ess_bulk"] >= MIN_ESS
        ]
        n = len(usable)
        entry = {
            "n_fits": len(records),
            "n_converged": n,
            "eligible": n >= MIN_CONVERGED_FITS,
            "base_param": _base_param(label),
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
            entry["coverage_band"] = list(_binomial_band(n, n_tests=n_tests))
            entry["bias_rate"] = sum(abs(r["z"]) > MAX_ABS_Z for r in usable) / n
            entry["mean_abs_z"] = sum(abs(r["z"]) for r in usable) / n
            contractions = sorted(r["contraction"] for r in usable)
            entry["median_contraction"] = contractions[n // 2]
            entry["truth_recovered_corr"] = _corr(
                [r["truth"] for r in usable], [r["mean"] for r in usable]
            )
        summary[f"{model}|{arm}|{design}|{label}"] = entry

    return {
        "cells": summary,
        "errors": errors,
        "n_gated_tests": n_tests,
        "excluded_for_divergences": {"|".join(k): v for k, v in diverged.items()},
        "excluded_for_degenerate_data": {"|".join(k): v for k, v in degenerate.items()},
        "attempted": {"|".join(k): v for k, v in attempted.items()},
        "likelihood_by_arm": {"|".join(k): v for k, v in likelihood_of.items()},
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


def _is_reference(arm: str) -> bool:
    """Whether this arm contains no network and can anchor an attribution."""
    return arm.split("@", 1)[0] in REFERENCE_LIKELIHOODS


def _rung_of(design_id: str) -> str:
    """`L1_n500@sv` -> `L1_n500`. The ladder position, without the variant."""
    return design_id.split("@", 1)[0]


def _richer_rungs(design_name: str) -> list[str]:
    """Ladder rungs with at least as much data AND design, and more of one.

    This is the second reference in the recipe. It is what stands in for an
    exact likelihood on the models that have none, and it works because the two
    failure modes respond differently to it: an identifiability limit is
    relieved by more trials or more conditions, and a wrong likelihood is not.
    """
    here = rd.DESIGNS.get(_rung_of(design_name))
    if here is None:
        return []
    return [
        name
        for name, other in rd.DESIGNS.items()
        if name != _rung_of(design_name)
        and other.n_trials >= here.n_trials
        and other.n_conditions >= here.n_conditions
    ]


def _reference_cell(cells: dict, model: str, design: str, label: str):
    """The exact-likelihood cell for this parameter, if any arm supplied one."""
    for key, entry in cells.items():
        k_model, k_arm, k_design, k_label = key.split("|")
        if (k_model, k_design, k_label) == (model, design, label) and _is_reference(
            k_arm
        ):
            return entry
    return None


def _rung_recovers(cells: dict, model: str, arm: str, rung: str, label: str) -> bool:
    """Did this parameter clear its floor at a richer rung?

    Looked up by BASE name, because the condition parameter is `v` at L0 and
    `v[0]`..`v[3]` at L1 and the L0->L1 crossing is precisely the one the ladder
    exists to make.

    A rung may hold several *variants* — `L1_n500@v` and `L1_n500@sv` are both
    at L1_n500 — and they are judged separately then combined asymmetrically,
    because the two quantifiers answer different questions. WITHIN one variant
    every condition cell must clear its own floor and be eligible, so that one
    lucky condition (or one variant with two surviving fits and a band wide
    enough to admit anything) cannot excuse the shortfall. ACROSS variants any
    one suffices: the question the ladder asks is whether SOME richer design
    recovers this parameter, and one that does is an answer.
    """
    base = _base_param(label)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for key, entry in cells.items():
        k_model, k_arm, k_design, k_label = key.split("|")
        if (k_model, k_arm) == (model, arm) and _rung_of(k_design) == rung:
            if _base_param(k_label) == base:
                by_variant[k_design].append(entry)
    return any(
        all(
            e["eligible"]
            and e["coverage"] is not None
            and e["coverage"] >= e["coverage_band"][0]
            for e in entries
        )
        for entries in by_variant.values()
    )


def verdict(summary: dict) -> tuple[bool, list[str]]:
    """Judge each network arm against nominal coverage and a no-network reference.

    Three ways to fail, all about calibration rather than sharpness:
      * coverage below the corrected floor, unattributable to the design,
      * a bias rate worse than the reference on the same cell, and
      * a posterior so close to the prior that coverage carries no information.

    A cell an exact-likelihood arm also fails is reported, not charged to the
    network. Where there is no such arm the ladder answers instead, and the
    resulting failure text says so -- a ladder-only attribution is weaker
    evidence and should read as weaker.

    A run also fails when it produced nothing to judge. Silence is not a pass:
    the previous version returned green for a sweep in which every fit errored,
    because it only ever iterated over cells that survived the filters.
    """
    failures: list[str] = []
    cells = summary["cells"]

    judged = 0
    for key, entry in cells.items():
        model, arm, design, label = key.split("|")
        if _is_reference(arm):
            continue
        where = f"{model}/{arm}/{design}/{label}"
        if not entry["eligible"]:
            failures.append(
                f"{where}: only {entry['n_converged']} converged fits, below the "
                f"{MIN_CONVERGED_FITS} needed to judge -- inconclusive, not a pass"
            )
            continue
        judged += 1
        reference = _reference_cell(cells, model, design, label)

        low = entry["coverage_band"][0]
        if entry["coverage"] < low:
            failures.extend(
                _explain_coverage(
                    cells, entry, reference, low, model, arm, design, label
                )
            )

        contraction = entry["median_contraction"]
        if contraction is not None and contraction > MAX_CONTRACTION:
            failures.append(
                f"{where}: posterior is {contraction:.2f} of the prior width -- the "
                "likelihood moved it almost not at all, so its coverage says nothing"
            )
        elif (
            reference
            and reference["median_contraction"]
            and contraction is not None
            and contraction / reference["median_contraction"] > MAX_CONTRACTION_RATIO
        ):
            failures.append(
                f"{where}: posterior {contraction / reference['median_contraction']:.1f}x "
                f"wider than the exact likelihood's on the same data -- information "
                "the data contain is being lost"
            )

        if reference and reference["eligible"] and reference["bias_rate"] is not None:
            if entry["bias_rate"] > reference["bias_rate"] + 0.15:
                failures.append(
                    f"{where}: bias rate {entry['bias_rate']:.2f} vs reference "
                    f"{reference['bias_rate']:.2f}"
                )
        elif entry["bias_rate"] is not None and entry["bias_rate"] > (
            UNREFERENCED_MAX_BIAS_RATE
        ):
            failures.append(
                f"{where}: bias rate {entry['bias_rate']:.2f} above the "
                f"{UNREFERENCED_MAX_BIAS_RATE:.2f} bar for a cell with no usable "
                "exact-likelihood reference"
            )

    # An arm that was attempted but yielded no eligible cell produced no
    # evidence, and no evidence is not evidence of calibration.
    for arm_key, n_attempted in summary["attempted"].items():
        model, arm, design = arm_key.split("|")
        if _is_reference(arm):
            continue
        if any(
            key.startswith(f"{model}|{arm}|{design}|") and cells[key]["eligible"]
            for key in cells
        ):
            continue
        failures.append(
            f"{model}/{arm}/{design}: {n_attempted} fits attempted, none usable "
            f"({len([e for e in summary['errors'] if e.get('arm') == arm])} errored) "
            "-- nothing to judge"
        )

    if not judged and not failures:
        failures.append("no network cells at all: there is nothing here to pass")

    return not failures, failures


def _explain_coverage(
    cells, entry, reference, low, model, arm, design, label
) -> list[str]:
    """Decide whether a coverage shortfall is the network's fault, and say why."""
    where = f"{model}/{arm}/{design}/{label}"
    if reference is not None and reference["coverage"] is not None:
        if not reference["eligible"]:
            # Do NOT fall through to the ladder here. The old code did, and its
            # message then claimed there was no exact arm for a model that has
            # one.
            return [
                f"{where}: coverage {entry['coverage']:.2f} below the {low:.2f} floor, "
                f"and the exact-likelihood arm has only {reference['n_converged']} "
                "converged fits -- inconclusive, fix the reference arm first"
            ]
        if reference["coverage"] < low:
            return []  # the exact arm misses it too: not the network
        return [
            f"{where}: coverage {entry['coverage']:.2f} below the {low:.2f} floor "
            f"while the exact likelihood reaches {reference['coverage']:.2f}"
        ]

    # No no-network reference for this cell. Ask the ladder instead: if a
    # richer rung recovers, the shortfall tracks the design.
    for rung in _richer_rungs(design):
        if _rung_recovers(cells, model, arm, rung, label):
            return []
    return [
        f"{where}: coverage {entry['coverage']:.2f} below the {low:.2f} floor, flat "
        "across the ladder and with no exact likelihood to compare against -- likely "
        "the network, but unconfirmed"
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
        "schema_version": 2,
        "n_shards": len(shards),
        "nominal_coverage": NOMINAL_COVERAGE,
        "thresholds": {
            "max_rhat": MAX_RHAT,
            "min_ess": MIN_ESS,
            "min_converged_fits": MIN_CONVERGED_FITS,
            "max_divergence_rate": MAX_DIVERGENCE_RATE,
            "min_choice_share": MIN_CHOICE_SHARE,
            "max_abs_z": MAX_ABS_Z,
            "unreferenced_max_bias_rate": UNREFERENCED_MAX_BIAS_RATE,
            "max_contraction": MAX_CONTRACTION,
            "max_contraction_ratio": MAX_CONTRACTION_RATIO,
            "family_alpha": FAMILY_ALPHA,
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
                # A driver reading only this line must be able to tell a clean
                # sweep from one where most tasks died: n_shards counts errored
                # shards too, so the usable count has to travel with it.
                "n_usable_fits": sum(
                    c["n_converged"] for c in summary["cells"].values()
                ),
                "n_errored_shards": len(summary["errors"]),
                "failures": failures,
            }
        )
    )
    # Non-zero on failure, matching validate_network.py's contract so a shell
    # driver can gate on the exit code rather than parsing the JSON.
    if not passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
