# Parameter recovery

Structure, parity, smoke-loading and density sanity all ask whether a network
is a well-formed likelihood. None of them asks whether it supports *inference*.
Users fit models; they do not evaluate densities. Recovery is the acceptance
test that closes that gap: simulate datasets from known parameters, fit them
back, and ask how often the truth came back.

The difficulty is that a recovery failure has two causes that look identical
from the outside — the network is wrong, or **the model is not identifiable
under that design** — and only the first is a defect in the artifact. The
harness in `validation/` exists to tell them apart, and it is written as a
recipe that applies to any model rather than a script for one.

For the operator procedure, see
[Run a parameter-recovery sweep](../how-to/run-recovery-sweep.md).

## 1. Read coverage and contraction together

Two statistics, and neither is interpretable alone.

**Coverage** — how often the 94% HDI contains the truth — tests the
*likelihood*. A calibrated likelihood covers at the nominal rate however wide
its intervals are. **Contraction** — posterior sd over prior sd — tests the
*design*: how much the data narrowed the prior.

|                  | narrow posterior            | wide posterior              |
| ---------------- | --------------------------- | --------------------------- |
| **covers truth** | identifiable, correct       | unidentifiable, but honest  |
| **misses truth** | **the likelihood is wrong** | wrong and vague             |

Only the bottom-left cell blocks a release. Coverage is gated; contraction is
reported. Failing a network for a wide-but-covering posterior would punish it
for being honest about what a hard dataset supports.

There is one exception, and it is not a quality bar. Truths are drawn from a
box inset 10% per side while the prior spans the full box, so the *prior's* own
94% interval contains every possible truth: a network whose likelihood is
constant scores coverage 1.00 on every parameter. An honest posterior is wide
because the data are uninformative; a degenerate one is wide because the
likelihood contributed nothing, and only the second makes coverage vacuous. So
contraction is gated at the degenerate end only, and against the exact
likelihood's width wherever one exists.

## 2. Hold a reference that contains no network

Without one, a failure cannot be attributed. Two are used, in order.

**The analytical arm**, where the model has one. It is fit on byte-identical
data with identical priors, so a shortfall both arms show is the design's
identifiability limit while one only the network shows is the network's. This
is the strong reference, and most models in the catalogue do not have it —
`angle`, `levy` and `ornstein` declare only `approx_differentiable`.

**The ladder**, otherwise. A design limit relaxes when you add trials or
conditions; a broken likelihood does not. A shortfall that a richer rung
repairs is charged to the design; one flat across the whole ladder is charged
to the network, and the failure text says the evidence is weaker, because it
is.

The pairing is what makes either work: the dataset index is the only source of
randomness, so every arm of one index sees the same data, and `--bounds-from`
pins one likelihood's bounds as the priors for *all* arms. Arms that differ in
their priors as well as their likelihoods are not paired.

## 3. Walk a ladder of design complexity

Weak identifiability is usually a property of the design, and the standard
remedy is structure: several conditions in which one parameter varies while the
rest are shared. Each **row** holds the trial budget fixed and changes only the
design — that is the comparison the ladder exists to make. Each **column** holds
the design fixed and raises the budget. Keeping both axes is what stops "not
enough data" and "not enough design" being confused.

| total trials | 1 condition | several conditions   |
| ------------ | ----------- | -------------------- |
| 250          | `L0_n250`   | `L1_n250`  (2 × 125) |
| 500          | `L0_n500`   | `L1_n500`  (4 × 125) |
| 1000         | `L0_n1000`  | `L1_n1000` (4 × 250) |
| *2000, opt.* | `L0_n2000`  | `L1_n2000` (4 × 500) |

Those are the budgets a real experiment has; 1000 trials is already a long
session. The 2000 rung stays addressable for a deliberate "is this recoverable
at all" question but is not in `DEFAULT_LADDER`.

The condition count is not fixed at 4. 250 split four ways is 62 per condition,
which is useless, and 250 is not divisible by 4 anyway — a fixed 4 would
silently drop two trials and break the constant-total guarantee. At that budget
2 × 125 is the better design. So the condition count varies *down* the L1
column, meaning two L1 rungs differ in structure as well as budget; *across* a
row, where the comparison actually lives, the budget is identical and only the
structure changes.
125/condition sits below Ratcliff and McKoon's ~200 floor deliberately — a
ladder that never fails is not measuring anything.

### Which parameter varies is the interesting knob

Drift is only the default, because it is what experiments usually manipulate.
Any parameter is a legitimate choice, and each asks a different question, in
two ways at once:

- the varying parameter gets **direct experimental leverage** — is it
  recoverable when something actually moves it?
- every *other* parameter is **pooled across all conditions**, so it is
  constrained by the whole dataset rather than one cell. This is how a
  multi-condition design rescues a parameter it never manipulates.

If `sv` comes back badly, `L1@v` asks *"does pooling fix `sv`?"* and `L1@sv`
asks *"is `sv` recoverable when we manipulate it?"* Running both against one
L0 is the design working as intended. Each variant is its own rung, identified
as `L1_n500@sv`, scored separately, and never pooled with another — a shortfall
at L0 is excused if **any** variant at a richer rung recovers the parameter,
while **within** a variant every condition must clear its own floor.

## What the verdict will and will not say

A cell needs at least ten converged fits to be judged; below that it is
**inconclusive**, which is not a pass. An arm that was attempted and produced
nothing usable fails — a sweep in which every task crashed must not come back
green, and partial attrition must not quietly widen the band until anything
clears it.

Because the run fails if any single cell fails, the coverage floor carries a
Šidák correction across the number of cells. Uncorrected, each cell's own
~2.9% false-alarm rate compounds over a full ladder into **53%**: a perfectly
calibrated network would fail more often than not. Corrected, the family-wise
rate is 2.2%, at the cost of a floor of 0.75 rather than 0.85 at twenty
datasets. That is the real information content of twenty datasets, and it is
the argument for running more.
