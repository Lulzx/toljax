"""Differentiable inverse cost-tolerance optimization using cached nominal S and H tensors."""

from __future__ import annotations

import copy
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from toljax.core.graph import CompiledAssembly
from toljax.core.metrics import _sample_latent_batch
from toljax.schema.assembly_spec import AssemblySpec, UNIT_TO_MM


def _has_fit_checks(compiled: CompiledAssembly) -> bool:
    spec = compiled.spec
    return bool(spec.jam_checks) or any(
        m.type in ("parallel_pin_hole", "cylindrical_clearance") for m in spec.mates
    )


def optimize_tolerances(
    compiled: CompiledAssembly,
    target_cpk: float = 1.33,
    max_jam_dpmo: float | None = 1350.0,
    fit_samples: int = 4096,
    verify_samples: int = 100_000,
    seed: int = 0,
) -> tuple[AssemblySpec, dict[str, Any]]:
    """Minimize total manufacturing cost C(t) = sum_j (a_j + b_j / t_j^alpha_j)
    subject to smooth second-order capability constraints:
      g_upper,k(t) = USL_k - mu_k(t) - 3 * target_cpk * sigma_k(t) >= 0
      g_lower,k(t) = mu_k(t) - LSL_k - 3 * target_cpk * sigma_k(t) >= 0
    and, when the assembly has clearance fits and `max_jam_dpmo` is not None, a fit constraint:
      g_fit(t) = mean of the n_tail smallest M_fit(z_i; t) over fixed samples z_i >= 0
    with n_tail = floor(max_jam_dpmo * 1e-6 * fit_samples). This CVaR bound implies at most
    n_tail - 1 sampled builds jam, so the empirical jam rate stays below `max_jam_dpmo`. The
    result is re-checked on `verify_samples` fresh builds and reported as `fit.jam_dpmo_after`.
    """
    n_opt = len(compiled.opt_feature_keys)
    if n_opt == 0:
        raise ValueError("No features with cost_model found in AssemblySpec for optimization")

    n_vars = len(compiled.latent_vars)
    z0 = jnp.zeros(n_vars, dtype=jnp.float64)

    # Evaluate S and H once at nominal z = 0 in float64
    t_start = time.perf_counter()
    S_nom = jnp.asarray(jax.jacrev(compiled.eval_measurements_dy)(z0), dtype=jnp.float64)
    H_nom = jnp.asarray(jax.hessian(compiled.eval_measurements_dy)(z0), dtype=jnp.float64)

    t_nom = jnp.asarray(compiled.opt_nominal_tols, dtype=jnp.float64)
    cost_a = jnp.asarray(compiled.opt_cost_a, dtype=jnp.float64)
    cost_b = jnp.asarray(compiled.opt_cost_b, dtype=jnp.float64)
    cost_alpha = jnp.asarray(compiled.opt_cost_alpha, dtype=jnp.float64)

    base_sigmas = jnp.array([lv.sigma for lv in compiled.latent_vars], dtype=jnp.float64)
    opt_indices = [
        lv.opt_param_idx if lv.opt_param_idx is not None else -1
        for lv in compiled.latent_vars
    ]
    opt_idx_arr = jnp.array(opt_indices, dtype=jnp.int32)
    has_opt_mask = opt_idx_arr >= 0
    safe_opt_idx = jnp.maximum(opt_idx_arr, 0)

    y_nominals = jnp.asarray(compiled.meas_nominals, dtype=jnp.float64)
    lsl_arr = jnp.asarray(compiled.meas_lsl, dtype=jnp.float64)
    usl_arr = jnp.asarray(compiled.meas_usl, dtype=jnp.float64)

    def _sigma_vec(t_vec: jax.Array) -> jax.Array:
        ratios = t_vec / t_nom
        gathered = ratios[safe_opt_idx]
        mult = jnp.where(has_opt_mask, gathered, jnp.ones_like(base_sigmas))
        return base_sigmas * mult

    def _moments_of_t(t_vec: jax.Array) -> tuple[jax.Array, jax.Array]:
        sig = _sigma_vec(t_vec)
        var_z = sig * sig
        # Mean shift: 0.5 * sum_i H_{k,ii} * var_z_i
        diag_H = jnp.diagonal(H_nom, axis1=1, axis2=2)  # [K, N]
        mean_shift = 0.5 * jnp.sum(diag_H * var_z[None, :], axis=-1)
        mu_y = y_nominals + mean_shift

        # Second-order variance: sum_i s_{k,i}^2 var_z_i + 0.5 * sum_{i,j} H_{k,ij}^2 var_z_i var_z_j
        var_lin = jnp.sum((S_nom * sig[None, :]) ** 2, axis=-1)
        H_scaled = H_nom * var_z[None, :, None]
        var_quad = 0.5 * jnp.sum(H_scaled * jnp.swapaxes(H_scaled, 1, 2), axis=(1, 2))
        sigma_y = jnp.sqrt(var_lin + var_quad + 1e-24)
        return mu_y, sigma_y

    @jax.jit
    def cost_and_grad(t_vec: jax.Array) -> tuple[jax.Array, jax.Array]:
        def _cost(t: jax.Array) -> jax.Array:
            return jnp.sum(cost_a + cost_b / (t**cost_alpha))

        return jax.value_and_grad(_cost)(t_vec)

    @jax.jit
    def constraints_fn(t_vec: jax.Array) -> jax.Array:
        mu_y, sigma_y = _moments_of_t(t_vec)
        g_upper = usl_arr - mu_y - 3.0 * target_cpk * sigma_y
        g_lower = mu_y - lsl_arr - 3.0 * target_cpk * sigma_y
        return jnp.concatenate([g_upper, g_lower], axis=-1)

    constraints_jac_fn = jax.jit(jax.jacrev(constraints_fn))

    use_fit = max_jam_dpmo is not None and _has_fit_checks(compiled)

    def _jam_margins_of_t(t_vec: jax.Array, z_batch: jax.Array) -> jax.Array:
        scale = t_vec / t_nom
        return jax.vmap(
            lambda z: jnp.min(compiled.eval_jam_margin_vector_scaled(z, scale))
        )(z_batch)

    jam_margins_jit = jax.jit(_jam_margins_of_t)

    if use_fit:
        n_tail = max(1, int(np.floor(max_jam_dpmo * 1e-6 * fit_samples)))
        z_fit = jnp.asarray(
            _sample_latent_batch(np.random.default_rng(seed), fit_samples, compiled),
            dtype=jnp.float64,
        )

        @jax.jit
        def fit_constraint_fn(t_vec: jax.Array) -> jax.Array:
            margins = _jam_margins_of_t(t_vec, z_fit)
            worst = -jax.lax.top_k(-margins, n_tail)[0]
            return jnp.mean(worst)[None]

        fit_constraint_jac_fn = jax.jit(jax.jacrev(fit_constraint_fn))

    # Warm-up JIT compilation
    _ = cost_and_grad(t_nom)
    _ = constraints_fn(t_nom)
    _ = constraints_jac_fn(t_nom)
    if use_fit:
        _ = fit_constraint_jac_fn(t_nom)

    t_opt_start = time.perf_counter()
    init_cost = float(cost_and_grad(t_nom)[0])

    def _scipy_obj(t_np: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = cost_and_grad(jnp.asarray(t_np, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    def _scipy_con(t_np: np.ndarray) -> np.ndarray:
        return np.asarray(
            constraints_fn(jnp.asarray(t_np, dtype=jnp.float64)), dtype=np.float64
        )

    def _scipy_con_jac(t_np: np.ndarray) -> np.ndarray:
        return np.asarray(
            constraints_jac_fn(jnp.asarray(t_np, dtype=jnp.float64)), dtype=np.float64
        )

    scipy_constraints = [{"type": "ineq", "fun": _scipy_con, "jac": _scipy_con_jac}]
    if use_fit:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda t_np: np.asarray(
                    fit_constraint_fn(jnp.asarray(t_np, dtype=jnp.float64)), dtype=np.float64
                ),
                "jac": lambda t_np: np.asarray(
                    fit_constraint_jac_fn(jnp.asarray(t_np, dtype=jnp.float64)),
                    dtype=np.float64,
                ),
            }
        )

    res = minimize(
        _scipy_obj,
        x0=np.asarray(t_nom, dtype=np.float64),
        method="SLSQP",
        jac=True,
        bounds=compiled.opt_bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    opt_solve_ms = (time.perf_counter() - t_opt_start) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    t_star = np.asarray(res.x, dtype=np.float64)
    final_cost = float(_scipy_obj(t_star)[0])
    mu_star, sig_star = _moments_of_t(jnp.asarray(t_star, dtype=jnp.float64))

    # Build updated AssemblySpec with optimized tolerances
    new_spec = copy.deepcopy(compiled.spec)
    unit_scale = UNIT_TO_MM[new_spec.assembly.units]
    key_to_t_star = {
        k: float(t_star[i] / unit_scale) for i, k in enumerate(compiled.opt_feature_keys)
    }

    feature_summary = []
    for i, key in enumerate(compiled.opt_feature_keys):
        t_before = float(compiled.opt_nominal_tols[i] / unit_scale)
        t_after = key_to_t_star[key]
        c_before = float(
            compiled.opt_cost_a[i]
            + compiled.opt_cost_b[i] / (compiled.opt_nominal_tols[i] ** compiled.opt_cost_alpha[i])
        )
        c_after = float(
            compiled.opt_cost_a[i]
            + compiled.opt_cost_b[i] / (t_star[i] ** compiled.opt_cost_alpha[i])
        )
        feature_summary.append(
            {
                "feature": key,
                "tolerance_before": round(t_before, 5),
                "tolerance_after": round(t_after, 5),
                "cost_before": round(c_before, 4),
                "cost_after": round(c_after, 4),
            }
        )

    for p in new_spec.parts:
        for f in p.features:
            k = f"{p.id}.{f.id}"
            if k in key_to_t_star:
                new_t = round(key_to_t_star[k], 6)
                ratio = new_t / max(
                    next(
                        (
                            v
                            for v in (f.gdt.position, f.gdt.profile, f.gdt.flatness)
                            if v is not None
                        ),
                        new_t,
                    ),
                    1e-12,
                )
                if f.gdt.position is not None:
                    f.gdt.position = new_t
                elif f.gdt.profile is not None:
                    f.gdt.profile = new_t
                elif f.gdt.flatness is not None:
                    f.gdt.flatness = new_t
                if f.gdt.perpendicularity is not None:
                    f.gdt.perpendicularity = round(f.gdt.perpendicularity * ratio, 6)
                if f.gdt.parallelism is not None:
                    f.gdt.parallelism = round(f.gdt.parallelism * ratio, 6)

    meas_cpk_after = []
    for k, m_id in enumerate(compiled.meas_ids):
        mu_k = float(mu_star[k])
        sig_k = float(sig_star[k])
        cpk_k = min(compiled.meas_usl[k] - mu_k, mu_k - compiled.meas_lsl[k]) / (3.0 * sig_k)
        meas_cpk_after.append(
            {
                "id": m_id,
                "shifted_mean": round(mu_k, 6),
                "sigma": round(sig_k, 6),
                "cpk": round(cpk_k, 4),
            }
        )

    fit_report: dict[str, Any] | None = None
    if use_fit:
        # Re-check jam rate on fresh builds the optimizer never saw
        rng_v = np.random.default_rng(seed + 1)
        jams_before = 0
        jams_after = 0
        t_star_j = jnp.asarray(t_star, dtype=jnp.float64)
        rem = verify_samples
        while rem > 0:
            cur = min(20_000, rem)
            z_v = jnp.asarray(_sample_latent_batch(rng_v, cur, compiled), dtype=jnp.float64)
            jams_before += int(jnp.sum(jam_margins_jit(t_nom, z_v) < 0.0))
            jams_after += int(jnp.sum(jam_margins_jit(t_star_j, z_v) < 0.0))
            rem -= cur
        fit_report = {
            "max_jam_dpmo": max_jam_dpmo,
            "fit_samples": fit_samples,
            "verify_samples": verify_samples,
            "jam_dpmo_before": round(jams_before / verify_samples * 1_000_000.0, 2),
            "jam_dpmo_after": round(jams_after / verify_samples * 1_000_000.0, 2),
        }

    report = {
        "assembly_name": compiled.spec.assembly.name,
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "solve_time_ms": round(opt_solve_ms, 3),
        "total_time_ms": round(total_ms, 3),
        "target_cpk": target_cpk,
        "cost_before": round(init_cost, 4),
        "cost_after": round(final_cost, 4),
        "cost_reduction_pct": round((1.0 - final_cost / max(init_cost, 1e-12)) * 100.0, 2),
        "features": feature_summary,
        "measurements": meas_cpk_after,
        "fit": fit_report,
    }
    return new_spec, report
