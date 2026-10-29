# `toljax`

**Headless, Differentiable 3D GD&T Variation Analysis & Inverse Cost-Tolerance Synthesis in JAX**

`toljax` predicts how 3D manufacturing tolerances on individual features accumulate across mechanical assemblies, and works backwards via automatic differentiation to synthesize the **lowest-cost tolerance allocation** that satisfies target process capability ($C_{pk}$) requirements.

It replaces GUI-locked desktop plugins (Siemens 3DCS, Sigmetrix CETOL $6\sigma$) with a declarative, Git-native YAML schema (`assembly.yaml`), CLI, and Python API built for CI pipelines.

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
