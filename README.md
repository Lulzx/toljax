# `toljax`

**Headless, Differentiable 3D GD&T Variation Analysis & Inverse Cost-Tolerance Synthesis in JAX**

`toljax` predicts how 3D manufacturing tolerances on individual features accumulate across mechanical assemblies, and works backwards via automatic differentiation to synthesize the **lowest-cost tolerance allocation** that satisfies target process capability ($C_{pk}$) requirements.

It replaces GUI-locked desktop plugins (Siemens 3DCS, Sigmetrix CETOL $6\sigma$) with a declarative, Git-native YAML schema (`assembly.yaml`), CLI, and Python API built for CI pipelines.

---

## What toljax Does, in Plain Terms

### 1. Real parts are never exactly the size on the drawing

Every factory makes parts a little off from the drawing. A tolerance is the amount of "off" that's still allowed.

```text
   Drawing says:        Factory actually makes:
   ┌──────────┐         ┌──────────┐   ┌───────────┐   ┌─────────┐
   │  6.000mm │         │  5.995   │   │  6.000    │   │  5.989  │
   └──────────┘         └──────────┘   └───────────┘   └─────────┘
                         part #1         part #2         part #3
```

That's fine for one part. The trouble starts when you bolt several parts together.

### 2. Small errors add up when parts are stacked

Each part's error carries into the next one, and any tilt gets magnified with distance:

```text
          tiny tilt at the base...
                                              ...big miss at the top
   housing ──┐                                    ▲
   pin  ─────┼─ +0.02                             │  0.4mm off!
   carrier ──┼─ +0.03  ╲                          │
   busbar ───┼─ +0.01    ╲                        │
   weld spot ┘            ╲_______________________●  ← laser weld misses
```

The example in this repo, [`examples/ev_busbar.yaml`](examples/ev_busbar.yaml), is an EV battery busbar stack. If those small errors pile up the wrong way, the welding laser misses its spot and the battery pack is scrap.

### 3. The usual way to check this

Engineers use expensive desktop programs (Siemens 3DCS, CETOL) to answer one question: "if I build 100,000 of these, how many fail?"

```text
   Engineer ──click──click──click──► [ $$$ GUI software ] ──► "0.3% fail"
                                            │
                                     can't put it in git,
                                     can't automate it
```

### 4. First trick: describe the assembly in a text file

You describe parts, tolerances and how they fit together in a YAML file, and the tool runs from the command line:

```text
   assembly.yaml ──► $ toljax run ──► results.json / report.html
        │
        │            laser_focal_z  (spec 25.00 ± 0.15 mm)
     git diff,         worst case      ±0.127 mm   PASS
     code review,      linear RSS      Cpk 1.48
     CI pipeline       Monte Carlo     68.4% of builds assemble
                                       (31.6% jam on the pins)
```

These are the real numbers for `examples/ev_busbar.yaml` with `--mc-samples 100000`. This puts mechanical tolerance analysis into the same git and CI workflow software teams use.

### 5. Second trick: work backwards

The usual tools only answer forward questions: "here are my tolerances, will the assembly work?" toljax can also answer the backward question: "what's the **cheapest** set of tolerances that will work?"

```text
   FORWARD (normal tools):
     tolerances ─────────────────────► does it work?

   BACKWARD (toljax):
     "I need it to work 99.99% of the time" ─────► cheapest tolerances
```

This matters because tight tolerances cost money:

```text
   cost
    $$$│█
       │█
       │ █
     $$│  ██
       │    ███
      $│       █████████______
       └─────────────────────── tolerance
        tight             loose
     (slow machining,    (cheap, fast)
      lots of rejects)
```

The optimizer loosens the features that barely affect the result, which saves money, and holds the ones that matter. Real output of `toljax optimize examples/ev_busbar.yaml --target-cpk 1.33`:

```text
   feature              BEFORE     AFTER
   housing.datum_A      0.050  →   0.250    loosened (hit upper bound)
   housing.pin_1        0.080  →   0.300    loosened (hit upper bound)
   housing.pin_2        0.080  →   0.300    loosened (hit upper bound)
   carrier.hole_1       0.100  →   0.400    loosened (hit upper bound)
   carrier.slot_2       0.100  →   0.400    loosened (hit upper bound)
   carrier.weld_pad     0.060  →   0.120    loosened
   carrier.bottom_pad   0.100  →   0.098    held ← drives 91% of the variance
   ──────────────────────────────────────
   cost                 142.5  →    28.9    (−79.7%, solved in ~4 ms)
   Cpk                   1.48  →    1.33    (meets the target exactly)
```

> **Caveat:** the optimizer currently constrains only the measurement's $C_{pk}$, not whether the parts physically fit. Re-running `toljax run` on the optimized YAML shows the worst-case band growing to ±0.156 mm (past the ±0.15 spec) and the fraction of Monte Carlo builds that assemble falling from 68.4% to 17.3%, because the loosened pins and holes jam. Check `jam_dpmo` on the result before using an optimized allocation.

### 6. How it finds the answer quickly

It's built on **JAX**, a library that can compute derivatives automatically: how much the final result moves when you nudge each input. That gives the optimizer a direction to move in, so it doesn't have to guess and check.

```text
   Guess-and-check:              With derivatives:
     try ● ● ● ● ● ●               ●──►──►──►★
     ● ● ● ● ● ● ●                 (follows the downhill slope
     ● ● ● ★ ● ● ●                  straight to the answer)
     thousands of runs             a few steps, milliseconds
```

### 7. Why it's hard to build

- **3D rotations are awkward.** Rotation math tends to break (divide by zero) exactly when a part is perfectly aligned, which is the most common case. `core/se3.py` is written to avoid that.
- **Parts settle into place.** A pin in a hole has slop, so the tool has to solve where each part actually ends up and still get derivatives through that solve. `core/solver.py` does this with the implicit function theorem.
- **It follows the real standard.** The tolerance zones follow ASME Y14.5, the rulebook used on real engineering drawings, including bonus tolerance when a pin is undersized (MMC).
- **It gives four answers.** Worst case, a fast statistical estimate (RSS), a more accurate second-order estimate (SOTA), and a full simulation of many virtual builds (Monte Carlo). They should agree with each other.
- **It's checked against known results.** The tests in `tests/` compare it to published academic benchmarks and to closed-form answers.

### In one line

```text
   ┌─────────────────────────────────────────────────────────────┐
   │  It does the job of expensive desktop GD&T tools as code,   │
   │  and also tells you the cheapest tolerances that will still │
   │  hit your target Cpk.                                       │
   └─────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```text
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ assembly.yaml│     │  part.step   │     │   CLI / API  │
   │ parts, GD&T, │     │  CAD faces   │     │ run/optimize │
   │ mates, specs │     │  (optional)  │     │   /inspect   │
   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
          └─────────┬──────────┘                    │
                    ▼                               │
          ┌───────────────────┐                     │
          │ Pydantic schema   │  validate + convert │
          │ (assembly_spec)   │  units + SVD rank   │
          └─────────┬─────────┘                     │
                    ▼                               │
          ┌───────────────────┐                     │
          │ graph.py          │  YAML → flat JAX    │
          │ CompiledAssembly  │  arrays (no Python  │
          └─────────┬─────────┘  loops under jit)   │
                    ▼                               │
   ┌────────────────────────────────────┐           │
   │  CORE MATH ENGINE (JAX)            │◄──────────┘
   │                                    │
   │  se3.py ─► torsor.py ─► mates.py   │
   │                │                   │
   │                ▼                   │
   │           solver.py (LM + IFT)     │
   │                │                   │
   │      ┌─────────┴────────┐          │
   │      ▼                  ▼          │
   │  metrics.py        optimizer.py    │
   │  (4-tier + jam)    (inverse design)│
   └──────┬──────────────────┬──────────┘
          ▼                  ▼
   results.json /     optimized_assembly.yaml
   report.html / .glb
```

### Key Technical Features

1. **Manifold-Exact $SE(3)$ Kinematics with `expm1` Perturbation Arithmetic (`toljax.core.se3`):**
   - Double-`jnp.where` 8th-order Taylor-safe exponential (`se3_exp`, `se3_expm1`) and logarithmic (`se3_log`) maps with zero `NaN` gradients at $\boldsymbol{\theta} = \mathbf{0}$ to arbitrary differentiation order.
   - Incremental rotation formulation $\Delta R = R - I_3$ prevents floating-point lever-arm cancellation in both `float64` and `float32`.
2. **Direct Geometric Mate Residuals & Engagement-Depth Tilt Binding (`toljax.core.mates`):**
   - Partial-DOF mates (`planar_coincident`, `parallel_pin_hole`, `clocking_pin_slot`, `line_contact`, `point_contact`, etc.) are formulated directly on $(\Delta R, \mathbf{p})$ at the physical engagement plane—avoiding logarithmic twist distortion ($\boldsymbol{\delta} = V^{-1}(\boldsymbol{\theta})\mathbf{p}$) that would pollute second-order Hessians.
   - Dimensionless two-sided SVD rank check refuses both under-constrained mechanisms and redundant rigid mates at compile time, while primary pin angular cocking and secondary fastener patterns are evaluated via exact radial fit margins ($M_{\text{fit}} \ge 0$, reporting `jam_dpmo`).
3. **ASME Y14.5-2018 / Y14.5.1-2019 Coupled Tolerance Zones (`toljax.core.torsor`):**
   - Symmetric 3-disk endpoint decomposition for cylindrical features refined by orientation callouts, plus an exact **3-cone Second-Order Cone Program (SOCP)** solver for Worst-Case Tier 1.
   - Unilateral ISO size midpoint centering ($r_{\text{mid}} = (D_{\max} + D_{\min})/4$) for kink-free affine MMC/LMC bonus evaluation in both `"acceptance_truncated"` (process-driven) and `"fully_consumed"` modes.
4. **Loop-Free Higher-Order Implicit Differentiation (`toljax.core.solver`):**
   - Recursive `@jax.custom_jvp` rule on the equilibrium solver extracts exact first-order Jacobians ($\mathbf{S}$) via `jax.jacrev` and second-order Hessians ($\mathbf{H}$) via `jax.hessian` at the root $\mathbf{F}(\mathbf{q}^*, \mathbf{z}) = \mathbf{0}$ without unrolling Newton/Levenberg-Marquardt iterations.
5. **4-Tier Variation Analysis & Inverse Cost-Tolerance Synthesis (`toljax.core.metrics`, `optimizer`):**
   - Simultaneously computes **Worst-Case (3-cone SOCP)**, **First-Order Linear RSS**, **Second-Order Tolerance Analysis (SOTA)** (with exact fourth-cumulant kurtosis corrections for 1D uniform and 2D unit-disk clearance floats), and **Vectorized Monte Carlo** (conditioned on assembleable non-jammed builds).
   - Synthesizes minimum-cost tolerance bands $\mathbf{t}^*$ subject to smooth second-order $C_{pk} \ge C_{pk,\text{target}}$ constraints in milliseconds.

---

## Quickstart

### Installation

```bash
uv sync --extra dev
```

### CLI Usage

```bash
# 1. Run 4-tier tolerance analysis (WC, Linear RSS, 2nd-Order SOTA, Monte Carlo)
uv run toljax run examples/ev_busbar.yaml \
  --mc-samples 100000 \
  --json-out results.json \
  --html-out report.html

# 2. Run inverse cost-tolerance optimization for target Cpk >= 1.33
uv run toljax optimize examples/ev_busbar.yaml \
  --target-cpk 1.33 \
  --out-yaml optimized_assembly.yaml

# 3. Extract nominal feature frames from a STEP file into YAML boilerplate
uv run toljax inspect part.step --out-yaml part_features.yaml
```

### Python API

```python
import toljax

spec = toljax.load("examples/ev_busbar.yaml")
compiled = toljax.compile_assembly(spec)

# 4-tier variation analysis
results = toljax.analyze(compiled, mc_samples=100_000)
print(results["measurements"][0]["second_order_sota"])

# Inverse cost-tolerance optimization
opt_spec, opt_report = toljax.optimize(compiled, target_cpk=1.33)
print(f"Cost reduced by {opt_report['cost_reduction_pct']}% in {opt_report['solve_time_ms']} ms")
```

---

## Verification & Benchmark Suite

Run all 17 verification and benchmark tests across the 5 numerical falsification gates:

```bash
uv run --extra dev pytest
```

* **Gate 1 (`test_se3_autodiff.py`, `test_plane_pin_slot_ref.py`):** Zero-angle $SE(3)$ autodiff (`jacrev` & `hessian` at $\boldsymbol{\theta} = \mathbf{0}$ with zero `NaN`s) and exact closed-form Plane-Pin-Slot trigonometric parity ($< 10^{-12}$).
* **Gate 2 (`test_implicit_vjp.py`):** Recursive `@jax.custom_jvp` Implicit Function Theorem Jacobian & Hessian vs. central finite differences of an unrolled Newton solver.
* **Gate 3 (`bench_optical_tilt.py`):** Second-order cosine foreshortening ($\Delta \mu_z = -L\sigma_\theta^2, \; \sigma_z = L\sigma_\theta^2$) and 2D uniform-disk fourth-cumulant variance & skewness parity against $10^6$-sample Monte Carlo.
* **Gate 4 (`bench_perf_6part.py`):** 6-part, 24-DOF assembly performance ($< 25\text{ ms}$ $\mathbf{S} + \mathbf{H}$) and `expm1` `float32` vs. `float64` Monte Carlo parity ($< 10^{-4}\sigma_y$).
* **Gate 5 (`bench_sota_clevis.py`):** Glancy & Chase (1999) / Gao et al. (1998) SOTA nonlinear benchmark and end-to-end YAML/CLI/Optimizer pipeline verification.

See [`spec.md`](spec.md) for the complete mathematical and architectural specification.

## License

MIT
