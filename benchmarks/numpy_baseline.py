"""Speed and accuracy of toljax against the conventional per-sample approach.

The baseline does what a black-box tolerance tool does: for every Monte Carlo sample it calls
SciPy's root finder (finite-difference Jacobian) to settle the assembly, and it gets
sensitivities S and Hessians H by central finite differences of that solve. It calls the
assembly's own compiled residual and measurement functions as black boxes, so both methods
share identical physics and the comparison measures method, not model.

Usage:
    uv run python benchmarks/numpy_baseline.py examples/ev_busbar.yaml \
        --baseline-samples 2000 --mc-samples 100000

Not collected by pytest (CI runs only tests/).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

import toljax
from toljax.core.metrics import _sample_latent_batch


def _baseline(compiled, z_mc: np.ndarray, h_s: float, h_h: float) -> dict:
    n_vars = len(compiled.latent_vars)
    q0 = np.zeros(compiled.n_dofs)

    res_jit = jax.jit(compiled.residual_fn)
    meas_jit = jax.jit(compiled.eval_measurements_dy_at)
    # Warm both so the baseline is not charged for compilation either
    res_jit(jnp.zeros(compiled.n_dofs), jnp.zeros(n_vars)).block_until_ready()
    meas_jit(jnp.zeros(compiled.n_dofs), jnp.zeros(n_vars)).block_until_ready()

    n_solves = 0

    def solve_y(z: np.ndarray) -> np.ndarray:
        nonlocal n_solves
        n_solves += 1
        z_j = jnp.asarray(z)
        sol = root(
            lambda q: np.asarray(res_jit(jnp.asarray(q), z_j)),
            q0,
            method="hybr",
            options={"xtol": 1e-13},
        )
        return np.asarray(meas_jit(jnp.asarray(sol.x), z_j))

    z0 = np.zeros(n_vars)
    y0 = solve_y(z0)

    t = time.perf_counter()
    cols = []
    for i in range(n_vars):
        e = np.zeros(n_vars)
        e[i] = h_s
        cols.append((solve_y(z0 + e) - solve_y(z0 - e)) / (2.0 * h_s))
    S = np.stack(cols, axis=-1)
    s_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    H = np.zeros((y0.shape[0], n_vars, n_vars))
    for i in range(n_vars):
        ei = np.zeros(n_vars)
        ei[i] = h_h
        H[:, i, i] = (solve_y(z0 + ei) - 2.0 * y0 + solve_y(z0 - ei)) / (h_h * h_h)
        for j in range(i + 1, n_vars):
            ej = np.zeros(n_vars)
            ej[j] = h_h
            v = (
                solve_y(z0 + ei + ej)
                - solve_y(z0 + ei - ej)
                - solve_y(z0 - ei + ej)
                + solve_y(z0 - ei - ej)
            ) / (4.0 * h_h * h_h)
            H[:, i, j] = v
            H[:, j, i] = v
    h_ms = (time.perf_counter() - t) * 1000.0
    sh_solves = n_solves

    t = time.perf_counter()
    y_mc = np.stack([solve_y(z) for z in z_mc], axis=0)
    mc_ms = (time.perf_counter() - t) * 1000.0

    return {
        "S": S,
        "H": H,
        "y_mc": y_mc,
        "s_ms": s_ms,
        "h_ms": h_ms,
        "sh_solves": sh_solves,
        "mc_ms": mc_ms,
        "mc_per_sample_ms": mc_ms / len(z_mc),
    }


def _toljax(compiled, z_mc: np.ndarray, z_big: np.ndarray) -> dict:
    n_vars = len(compiled.latent_vars)
    z0 = jnp.zeros(n_vars)

    t = time.perf_counter()
    jac_fn = jax.jit(jax.jacrev(compiled.eval_measurements_dy))
    hess_fn = jax.jit(jax.hessian(compiled.eval_measurements_dy))
    jac_fn(z0).block_until_ready()
    hess_fn(z0).block_until_ready()
    compile_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    S = np.asarray(jac_fn(z0).block_until_ready())
    H = np.asarray(hess_fn(z0).block_until_ready())
    sh_ms = (time.perf_counter() - t) * 1000.0

    batch_fp64 = jax.jit(jax.vmap(compiled.eval_measurements_dy))
    y_mc = np.asarray(batch_fp64(jnp.asarray(z_mc)))

    batch_fp32 = jax.jit(jax.vmap(compiled.eval_measurements_dy))
    batch_fp32(jnp.zeros((64, n_vars), dtype=jnp.float32)).block_until_ready()
    t = time.perf_counter()
    for chunk in np.array_split(z_big, max(1, len(z_big) // 50_000)):
        batch_fp32(jnp.asarray(chunk, dtype=jnp.float32)).block_until_ready()
    big_ms = (time.perf_counter() - t) * 1000.0

    return {"S": S, "H": H, "y_mc": y_mc, "compile_ms": compile_ms, "sh_ms": sh_ms, "big_ms": big_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("assembly_yaml")
    ap.add_argument("--baseline-samples", type=int, default=2000)
    ap.add_argument("--mc-samples", type=int, default=100_000)
    ap.add_argument("--h-s", type=float, default=1e-6, help="FD step for S (latent units)")
    ap.add_argument("--h-h", type=float, default=1e-4, help="FD step for H (latent units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    compiled = toljax.compile_assembly(toljax.load(args.assembly_yaml))
    rng = np.random.default_rng(args.seed)
    z_mc = _sample_latent_batch(rng, args.baseline_samples, compiled)
    z_big = _sample_latent_batch(rng, args.mc_samples, compiled)

    base = _baseline(compiled, z_mc, args.h_s, args.h_h)
    tj = _toljax(compiled, z_mc, z_big)

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))

    projected_ms = base["mc_per_sample_ms"] * args.mc_samples
    out = {
        "assembly": compiled.spec.assembly.name,
        "latent_vars": len(compiled.latent_vars),
        "dofs": compiled.n_dofs,
        "sensitivities": {
            "baseline_S_ms": round(base["s_ms"], 1),
            "baseline_H_ms": round(base["h_ms"], 1),
            "baseline_solves": base["sh_solves"],
            "toljax_S_H_ms": round(tj["sh_ms"], 2),
            "toljax_compile_ms": round(tj["compile_ms"], 1),
            "S_rel_diff": rel(base["S"], tj["S"]),
            "H_rel_diff": rel(base["H"], tj["H"]),
        },
        "monte_carlo": {
            "baseline_samples": args.baseline_samples,
            "baseline_ms": round(base["mc_ms"], 1),
            "baseline_per_sample_ms": round(base["mc_per_sample_ms"], 4),
            "baseline_projected_ms": round(projected_ms, 1),
            "toljax_samples": args.mc_samples,
            "toljax_fp32_ms": round(tj["big_ms"], 1),
            "speedup_vs_projected": round(projected_ms / tj["big_ms"], 1),
            "same_samples_max_abs_diff_mm": float(np.max(np.abs(base["y_mc"] - tj["y_mc"]))),
        },
    }
    text = json.dumps(out, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
