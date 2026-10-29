# Validate and inspect a candidate network

Validation asks a narrow operational question: will HSSM load this ONNX
likelihood and obtain numerically and scientifically plausible values? Run it
on artifacts you trust before any publication attempt.

## Install the validation profile

```bash
uv sync --locked --group validate
```

This adds HSSM and ONNX Runtime to the normal pipeline environment. The docs
environment is intentionally separate and cannot run these gates.

## Run the LAN gates

```bash
uv run python validation/validate_network.py \
  --onnx-path /path/to/run_uuid_lan_ddm__network.onnx \
  --model-name ddm \
  --network-type lan
```

The command writes `validation_report.json` next to the ONNX unless
`--report-path` is given. It prints one compact JSON line and exits non-zero
when a gate fails.

| Gate | Evidence |
| --- | --- |
| G1 `structure` | ONNX loads; all input dimensions are concrete; there is one input, one scalar output, and the input width matches the model |
| G2 `parity` | The exported ONNX matches the JAX trainer state when the required siblings exist |
| G3 `hssm_load` | HSSM accepts the likelihood and obtains a finite initial log probability on 100 simulated trials |
| G4 `density` | Integrated mass is near one and Hellinger error is acceptable relative to a measured simulator sampling floor, at five draws from the 10 %-shrunk box |
| G5 `mass_survey` | Total mass is near one across the *whole* training box: 20 000 draws on LANfactory's onset-refined grid, judged on the tails of `|total − 1|` (advisory; see below) |

G3 and G4 draw parameters from the training box shrunk by 10% on each side.
HSSM rejects any response time at or below zero, and models whose box allows
`st` larger than `t` (`ddm_st`, and `ddm_normal_st` with its untruncated
non-decision-time kernel) simulate such trials near the edge. G3 therefore
redraws its parameters, up to eight times, until every simulated response time
is positive. It never drops trials. The report records the draw it used
(`theta`, `seed`, `attempts`) and every rejected draw (`rejected_draws`), and
if the budget runs out the gate skips with a `reason` that starts
`no usable dataset` and names the box. A skip, not a failure, because the
network was never loaded and so nothing was learned about it. The rejected
draws stay in the report, and because G3 is required, the publisher refuses a
report whose G3 skipped: an exhausted G3 blocks publication just as a failed
one does.

G2 legitimately skips for Torch artifacts and for a bare ONNX without the JAX
state/config pair. G1, G3, and G4 are required for publication; G5 is
advisory and skips itself in an environment whose LANfactory predates
`lanfactory.derive`. A skipped gate
may carry `passed: true` in the detailed report to mean it did not itself fail;
inspect the compact `gates` states rather than treating that as evidence the
check ran.

## The mass survey (G5)

A LAN is a density, so its mass summed over choices and integrated over
reaction time should be one at every θ in the training box. G4 checks that at
five θ drawn from the box shrunk by 10 % a side, which is where most fits
live — and it is blind to the edge. A survey of the published `ddm` LAN at
20 000 θ over the full box found a median `|total − 1|` of 0.0037 but a p99 of
0.080, a range of 0.877–1.219, and 1.9 % of the box beyond 0.05, all of it in
two regions at the box edge (a > 2.2 with |v| < 0.5, where the tail is
over-estimated by up to +0.22; and v > 2.5, z > 0.8, a > 2.2, where the peak
is under-estimated by −0.12). G4 would fail that network with probability
0.06 %: five draws from the shrunk box almost never land there, and the
0.9–1.1 band is wider than the deviation anyway.

G5 runs LANfactory's `lanfactory.derive.survey` — uniform θ over the full box,
total mass on the *onset-refined* grid — and judges the tails of the
`|total − 1|` distribution. The grid matters: a uniform 1000-point grid carries
15–25 % quadrature error at a < 0.5, which the survey would otherwise blame on
the network. The whole survey dict (per-parameter bins and any leak of mass
below the onset) is recorded in the gate so a deviation can be located,
together with the `seconds` it took.

| Verdict | Rule | Effect |
| --- | --- | --- |
| fail | `total.mean` is not finite, `total.p99_abs_dev > 0.10`, or `total.frac_gt_0.10 > 0.02` | The gate fails and the report fails |
| warn | `total.p99_abs_dev > 0.05` or `total.frac_gt_0.05 > 0.01` | The gate passes with a `warning` detail |
| pass | none of these | — |

A non-finite mean fails on its own: the survey counts a NaN total as an
infinite deviation, but if under 1 % of the box is NaN, the p99 and the 0.10
fraction can both stay clean.

The published `ddm` LAN, which is in production and demonstrably usable, lands
in *warn*: the warn line sits under its p99 and the fail line above it.

The thresholds are **provisional**. They are set from that one survey and are
to be frozen after the four modern LANs (`ddm_sdv`, `gamma_drift`,
`gamma_drift_angle`, `angle_extended`) have been surveyed. Until then
`--mass-survey-p99-max` (default `0.10`) and
`--mass-survey-frac-gt-0.10-max` (default `0.02`) let the freezing run change
them without a code change; record a changed threshold in the report, and do
not loosen one to pass a candidate.

!!! note "G5 needs `lanfactory.derive`"

    The survey lives in LANfactory's `derive` package, which the lock pins. An
    environment synced to an older LANfactory reports G5 as `skipped` with the
    reason `lanfactory.derive not available; this environment's LANfactory
    predates it`. The publisher does not require G5, so the skip never blocks
    a publish; a failing G5 refuses one like any other failed gate.
    `--skip-mass-survey` shortens a diagnostic run the way `--skip-density`
    does.

!!! danger "Only validate trusted artifact folders"

    The parity gate unpickles the sibling `*_network_config.pickle`. Unpickling
    can execute code. Validate folders you produced or fetched from a repository
    you control; do not point the command at arbitrary downloads.

## Use skips only for diagnosis

`--skip-hssm` and `--skip-density` shorten a diagnostic run, but do not create
publishable evidence. In particular, the publisher refuses any report where a
required gate is skipped or missing.

If the density comparison needs a reviewed tolerance change, pass
`--hellinger-ratio-max` and preserve the resulting threshold in the report.
Do not loosen it merely to turn one candidate green.

## Validate an auxiliary network (cpn, opn)

Auxiliary networks are not densities, so the density gate does not apply to
them. They share G1 and G2 with a LAN and then run their own pair:

```bash
uv run python validation/validate_network.py \
  --onnx-path /path/to/run_uuid_opn_ddm__network.onnx \
  --model-name ddm \
  --network-type opn
```

`--model-name` is always the base model. Pass `ddm`, never `ddm_deadline`: the
deadline variant is derived internally wherever a simulation needs it, and the
command rejects a `_deadline` name outright.

| Gate | Evidence |
| --- | --- |
| G1 `structure` | As for a LAN, with the width contract `n_params + 1`: `[θ…, choice]` for a cpn, `[θ…, deadline]` for an opn |
| G2 `parity` | As for a LAN |
| A3 `hssm_missing_load` | HSSM accepts the network as `loglik_missing_data` next to the base LAN and obtains a finite initial log probability on simulated data with missing rows, at `p_outlier = 0` and at HSSM's default lapse |
| A4 `accuracy` | The network's probability tracks a Monte-Carlo truth at 20 parameter draws stratified over the full box: `P(choice | θ)` for a cpn, `P(rt > deadline | θ)` for an opn |

The network's output must already be a log-probability (log-sigmoid baked into
the graph, every value ≤ 0). A raw-logit export fails A4 before any truth is
simulated.

A4's 20 draws are two strata of 10: the `core` stratum comes from the box
shrunk by 10 % a side, as before; the `edge` stratum is drawn uniformly from
the full box and kept only when it falls *outside* the shrunk box. The shrunk
box misses exactly the regions where the published `ddm` LAN's mass is off by
10–20 % (see the mass survey above), and an auxiliary network derived from
that LAN inherits the error there. Every draw records its `stratum`; when
`--lan-onnx` is given and `lanfactory.derive` is importable, it also records
the base LAN's `total_mass` at that θ (integrated on the onset-refined grid),
so a failing draw can be attributed to the LAN rather than to the derived net.
Without either, `total_mass` is `null`. The pass rule and its thresholds are
unchanged.

`--aux-category` names what the network's output is the probability of:
`choice` for a cpn, `omission` for an opn. That is the
vocabulary LANfactory's derived corpora and the publisher's provenance carry,
and each type has exactly one value, so the flag defaults to it; an explicit
value that does not match (say `deadline` for an opn) is rejected naming the
expected one. The report carries the resolved value as top-level
`aux_category`.

A3 resolves the base LAN by model name through HSSM, which downloads it for
registry models. Pass `--lan-onnx` to use a local LAN instead; models outside
HSSM's registry require it. Either way the assembly is LAN + auxiliary net
(`loglik_kind="approx_differentiable"`), never HSSM's analytical likelihood.
A3 simulates its base data as G3 does, from the 10 %-inset box and redrawn
until every response time is positive, so it records `attempts` and
`rejected_draws` too, and skips with G3's `no usable dataset` reason when no
draw is usable. A skipped A3 blocks a publish like any skipped required gate.

!!! warning "Auxiliary reports do not promote yet"

    The publisher still requires the LAN gate set and does not pass
    `--aux-category`, so it refuses every auxiliary report, and
    `lan-publish --network-type cpn` cannot validate a cpn at all. Use this
    command to validate an auxiliary artifact; publishing it is a follow-up.

One skip is built in and reported rather than silent: the cpn A3 gate needs
an HSSM that passes `response` to the network on missing rows (HSSM ≥ 0.6.0).
Under an older HSSM it is `skipped` with the reason `HSSM < 0.6.0 ignores
response on missing rows`, and activates by itself once the locked HSSM
moves.

`--skip-accuracy` shortens a diagnostic run the way `--skip-density` does for a
LAN. The A4 thresholds (`--accuracy-mean-abs-max`, default `0.01`, and
`--accuracy-max-abs-max`, default `0.03`) are provisional: they are to be
calibrated once on the first `ddm_sdv` gate run and then frozen. Record a
changed threshold in the report; do not loosen it to pass one candidate.

!!! note "The cpn truth counts every trial"

    ssm-simulators does not censor a base model at `max_t`: a trial that has
    not terminated by then comes back at `rt ≈ max_t + t` with its
    sign-implied choice, while a derived cpn integrates only up to 20 s. A4
    judges the all-trials proportion and records the windowed one as
    `truth_rt_lt_max_t` on every draw: the share of all trials that gave the
    choice and finished inside the window (a joint probability, not
    conditional on finishing), so the two can be compared where a
    visible share of trials outlasts the window.

## Inspect the result visually

The marimo inspector compares the network's implied likelihood with simulator
KDEs and shows a parameter manifold. It explains a gate result; it does not
replace one.

```bash
export INSPECT_ONNX="/path/to/run_uuid_lan_ddm__network.onnx"
export INSPECT_MODEL="ddm"
uv run --group inspect marimo edit validation/inspect_network.py
```

To produce a static local report without opening an editor:

```bash
uv run --group inspect marimo export html validation/inspect_network.py \
  -o inspection.html
```

If `validation_report.json` is present, the inspector displays it only when its
recorded ONNX filename matches the artifact being viewed.
