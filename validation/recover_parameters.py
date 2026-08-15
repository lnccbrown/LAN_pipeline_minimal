#!/usr/bin/env python3
"""Fit one synthetic dataset and record how well the parameters came back.

One invocation = one fit = one SLURM array task. Coverage is a property of an
*ensemble* of fits, so this script deliberately computes none of it; it writes
a per-fit shard and `aggregate_recovery.py` does the statistics. That split is
what lets the sweep fan out and fan back in.

Two arms exist for every dataset:

    --likelihood analytical              the ceiling. No network involved.
    --likelihood approx_differentiable   the network under test.

Both are handed identical data and identical priors, so they differ only in
the likelihood. That pairing is the whole point: a failure the analytical arm
shares is the design's identifiability limit, and a failure only the network
shows is the network's.

Usage (one shard):

    uv run --group validate python validation/recover_parameters.py \\
        --design L1_n4000 --dataset-index 7 \\
        --likelihood approx_differentiable --onnx-path /path/to/model.onnx \\
        --out-dir results/
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent))
import recovery_designs as rd  # noqa: E402

logger = logging.getLogger("recover_parameters")
app = typer.Typer(add_completion=False)

# Requested explicitly: arviz >= 1.0 defaults to an 89% *equal-tailed*
# interval, a different statistic that would silently change every coverage
# verdict. Same constant and same reason as the spine's rlssm recovery skill.
HDI_PROB = 0.94


def _sanity(data) -> dict:
    """Cheap checks on the simulated data itself.

    A dataset where trials hit the simulator's ceiling is censored, and a
    recovery failure on it says nothing about the likelihood. Recorded rather
    than raised: the aggregator decides what to do with it.
    """
    rt = data["rt"].to_numpy()
    return {
        "n_trials": int(rt.size),
        "rt_min": float(rt.min()),
        "rt_max": float(rt.max()),
        "n_rt_nonpositive": int((rt <= 0).sum()),
        "n_rt_at_ceiling": int((rt >= 19.9).sum()),
        "response_balance": float((data["response"].to_numpy() > 0).mean()),
    }


def _summarise(idata, design, truth) -> dict:
    """Per-parameter posterior summary against the known truth.

    A condition-varying parameter is one vector-valued posterior variable, so
    it expands into one record per condition — `v[0]`, `v[1]`, ... — each
    scored against its own truth.
    """
    import arviz as az
    import numpy as np

    posterior = idata["posterior"] if "posterior" in idata else idata.posterior
    names = rd.posterior_names(design)
    out = {}
    for param, variables in names.items():
        var = variables[0]
        if var not in posterior:
            out[param] = {"error": f"{var} absent from posterior"}
            continue
        arr = posterior[var]
        truth_value = truth[param]
        is_vector = isinstance(truth_value, list)
        n_entries = len(truth_value) if is_vector else 1

        for i in range(n_entries):
            sub = arr.isel({arr.dims[-1]: i}) if is_vector else arr
            label = f"{param}[{i}]" if is_vector else param
            true_i = float(truth_value[i]) if is_vector else float(truth_value)
            draws = np.asarray(sub).reshape(-1)
            mean, sd = float(draws.mean()), float(draws.std(ddof=1))
            # arviz 1.x spells the argument `prob`, and its own default is an
            # 89% equal-tailed interval — a different statistic. Always pass it.
            lo, hi = (float(x) for x in az.hdi(draws, prob=HDI_PROB))
            # Prior is Uniform over HSSM's bound for this parameter; its sd is
            # the yardstick contraction is measured against.
            p_lo, p_hi = rd.ONNX_BOUNDS[param]
            prior_sd = (p_hi - p_lo) / (12**0.5)
            out[label] = {
                "truth": true_i,
                "mean": mean,
                "sd": sd,
                "hdi_lo": lo,
                "hdi_hi": hi,
                "covered": bool(lo <= true_i <= hi),
                # Signed, in posterior-sd units. |z| > 2 is the bias flag,
                # matching HSSM's own addm_parameter_recovery script.
                "z": float((mean - true_i) / sd) if sd > 0 else float("inf"),
                "contraction": float(sd / prior_sd),
                "rhat": float(az.rhat(sub)),
                "ess_bulk": float(az.ess(sub)),
            }
    return out


def _posterior_corr(posterior, design) -> dict[str, float]:
    """Pairwise posterior correlations between the scalar parameters.

    This is the identifiability diagnostic proper. A flat direction in the
    likelihood shows up as a near-|1| correlation between two parameters: the
    data pin their combination but not either one, so the posterior lies along
    a ridge. Bias in the marginals then comes in pairs — one parameter pulled
    down exactly as far as its partner is pulled up — which is invisible if you
    only look at each marginal on its own.

    Condition-varying parameters are skipped; they are vectors, and the pair
    that matters for ddm_sdv (sv against t) is scalar in every design.
    """
    import numpy as np

    scalars = [
        p
        for p in rd.PARAM_ORDER
        if p not in design.varies_by_condition and p in posterior
    ]
    flat = {p: np.asarray(posterior[p]).reshape(-1) for p in scalars}
    out = {}
    for i, a in enumerate(scalars):
        for b in scalars[i + 1 :]:
            x, y = flat[a], flat[b]
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            out[f"{a}~{b}"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
    return out


def run_one(
    design_name: str,
    dataset_index: int,
    likelihood: str,
    onnx_path: Path | None,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
) -> dict:
    """Simulate, fit, score. Returns the shard record."""
    import hssm
    import numpy as np

    design = rd.DESIGNS[design_name]
    # The dataset index is the only source of randomness, so the analytical
    # and network arms of the same index see byte-identical data.
    seed = 10_000 + dataset_index
    data, truth = rd.build_dataset(design, seed=seed)

    hssm.set_floatX("float32", update_jax=True)
    kwargs = {}
    if likelihood == "approx_differentiable":
        if onnx_path is None:
            raise typer.BadParameter("--onnx-path is required for the network arm.")
        # A local path, never a bare filename: HSSM would otherwise download
        # ddm_sdv.onnx from franklab/HSSM at construction time, where it does
        # not exist, and compute nodes may have no egress at all.
        kwargs["loglik"] = str(onnx_path)

    model = hssm.HSSM(
        data=data,
        model="ddm_sdv",
        loglik_kind=likelihood,
        **kwargs,
        **rd.model_spec(design),
    )

    started = time.time()
    idata = model.sample(
        sampler="numpyro",
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        progressbar=False,
        # Nothing downstream reads the pointwise log-likelihood, and it is
        # (chains, draws, n_trials) floats per fit.
        idata_kwargs={"log_likelihood": False},
    )
    elapsed = time.time() - started

    sample_stats = (
        idata["sample_stats"] if "sample_stats" in idata else idata.sample_stats
    )
    posterior = idata["posterior"] if "posterior" in idata else idata.posterior
    diverging = np.asarray(sample_stats["diverging"])
    return {
        "schema_version": 1,
        "design": design_name,
        "dataset_index": dataset_index,
        "likelihood": likelihood,
        "onnx": str(onnx_path) if onnx_path else None,
        "seed": seed,
        "data": _sanity(data),
        "sampler": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "divergences": int(diverging.sum()),
            "divergence_rate": float(diverging.mean()),
            "wall_seconds": round(elapsed, 1),
        },
        "parameters": _summarise(idata, design, truth),
        "posterior_corr": _posterior_corr(posterior, design),
        "env": {
            "python": platform.python_version(),
            "hssm": getattr(hssm, "__version__", "unknown"),
        },
    }


@app.command()
def main(
    design: str = typer.Option(..., help=f"One of: {', '.join(rd.DESIGNS)}"),
    dataset_index: int = typer.Option(
        ..., help="Seeds the dataset. Both arms of one index see the same data."
    ),
    likelihood: str = typer.Option(
        "approx_differentiable", help="analytical | approx_differentiable"
    ),
    onnx_path: Path | None = typer.Option(None, help="Required for the network arm."),
    out_dir: Path = typer.Option(Path("."), help="Where the shard JSON is written."),
    draws: int = typer.Option(500),
    tune: int = typer.Option(1000),
    chains: int = typer.Option(2),
    target_accept: float = typer.Option(0.9),
    log_level: str = typer.Option("WARNING"),
):
    """Fit one synthetic dataset and write a recovery shard."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.WARNING))
    if design not in rd.DESIGNS:
        raise typer.BadParameter(f"Unknown design {design!r}. Have: {list(rd.DESIGNS)}")
    if likelihood not in ("analytical", "approx_differentiable"):
        raise typer.BadParameter(f"Unknown likelihood {likelihood!r}.")

    try:
        record = run_one(
            design_name=design,
            dataset_index=dataset_index,
            likelihood=likelihood,
            onnx_path=onnx_path,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
        )
    except Exception as e:  # noqa: BLE001 - a dead shard must not kill the array
        record = {
            "schema_version": 1,
            "design": design,
            "dataset_index": dataset_index,
            "likelihood": likelihood,
            "error": f"{type(e).__name__}: {e}",
        }
        logger.error(f"shard failed: {record['error']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "analytical" if likelihood == "analytical" else "net"
    destination = out_dir / f"recovery_{design}_{tag}_{dataset_index:04d}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")

    # One JSON line on stdout regardless of log level — the same driver
    # contract gen_sbatch and validate_network already use.
    print(json.dumps({"shard": str(destination), "error": record.get("error")}))
    if "error" in record:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
