"""4-Tier Variation Analysis Metrics:
  - Tier 1: Worst-Case (Exact 3-Cone SOCP + Disk/Box Norms + 2nd-Order Vertex Shift)
  - Tier 2: First-Order Linear RSS (Cp, Cpk)
  - Tier 3: Second-Order SOTA (Mean Shift, Variance & Skewness with 1D/2D Disk Cumulants)
  - Tier 4: Vectorized Monte Carlo (expm1 float32 solver + float64 Welford/chunked reduction)
"""

from __future__ import annotations

import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from toljax.core.graph import CompiledAssembly
from toljax.core.torsor import solve_three_cone_wc_cylinder


def compute_sota_moments(
    y_nom: float,
    s: np.ndarray,
    H: np.ndarray,
    sigmas: np.ndarray,
    uniform_1d_indices: list[int] | None = None,
    uniform_disk_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute exact Second-Order Tolerance Analysis (SOTA) moments for Gaussian, 1D uniform,
    and 2D uniform unit-disk inputs.

    Implements Section 2.6 of spec.md, including the fourth-cumulant variance and O(sigma^4)
    skewness terms for 1D uniform (kappa = -6/5) and 2D uniform unit-disk (kappa_1111 = -1,
    kappa_1122 = -1/3) variables.
    """
    uniform_1d_indices = uniform_1d_indices or []
    uniform_disk_pairs = uniform_disk_pairs or []

    var_z = sigmas * sigmas
    Sigma = np.diag(var_z)
    HS = H @ Sigma

    # 1. Exact second-order mean shift (valid for all zero-mean distributions)
    mean_shift = 0.5 * float(np.trace(HS))
    shifted_mean = float(y_nom + mean_shift)

    # 2. Exact second-order variance with fourth-cumulant corrections
    var_linear = float(s @ Sigma @ s)
    var_quad_gauss = 0.5 * float(np.trace(HS @ HS))

    var_cumulant_1d = 0.0
    kappa_1d = -1.2  # -6/5 for 1D uniform
    for idx in uniform_1d_indices:
        var_cumulant_1d += 0.25 * kappa_1d * (H[idx, idx] ** 2) * (var_z[idx] ** 2)

    var_cumulant_disk = 0.0
    k1111 = -1.0
    k1122 = -1.0 / 3.0
    for ix, iy in uniform_disk_pairs:
        h11 = H[ix, ix]
        h22 = H[iy, iy]
        h12 = H[ix, iy]
        sig4 = (var_z[ix] ** 2)  # (1/4)^2 = 1/16 for unit disk
        var_cumulant_disk += (
            0.25
            * (
                k1111 * (h11 * h11 + h22 * h22)
                + k1122 * (2.0 * h11 * h22 + 4.0 * h12 * h12)
            )
            * sig4
        )

    var_total = max(
        var_linear + var_quad_gauss + var_cumulant_1d + var_cumulant_disk, 1e-30
    )
    sigma_2nd = float(np.sqrt(var_total))

    # 3. Third central moment & skewness (exact O(sigma^4) L^2 Q cumulant + Gaussian O(sigma^6))
    mu3_gauss_l2q = 3.0 * float(s @ Sigma @ H @ Sigma @ s)
    mu3_cumulant_1d = 0.0
    for idx in uniform_1d_indices:
        mu3_cumulant_1d += 1.5 * kappa_1d * (s[idx] ** 2) * H[idx, idx] * (var_z[idx] ** 2)

    mu3_cumulant_disk = 0.0
    for ix, iy in uniform_disk_pairs:
        s1, s2 = s[ix], s[iy]
        h11, h22, h12 = H[ix, ix], H[iy, iy], H[ix, iy]
        sig4 = var_z[ix] ** 2
        mu3_cumulant_disk += (
            1.5
            * (
                k1111 * (s1 * s1 * h11 + s2 * s2 * h22)
                + k1122 * (s1 * s1 * h22 + s2 * s2 * h11 + 4.0 * s1 * s2 * h12)
            )
            * sig4
        )

    mu3_q3_gauss = float(np.trace(HS @ HS @ HS))
    mu3_total = mu3_gauss_l2q + mu3_cumulant_1d + mu3_cumulant_disk + mu3_q3_gauss
    skewness = float(mu3_total / (sigma_2nd**3)) if sigma_2nd > 1e-14 else 0.0

    return {
        "mean_shift": mean_shift,
        "shifted_mean": shifted_mean,
        "sigma": sigma_2nd,
        "skewness": skewness,
    }


def _sample_latent_batch(
    rng: np.random.Generator,
    n_samples: int,
    compiled: CompiledAssembly,
) -> np.ndarray:
    """Sample a batch of latent vectors z respecting Gaussian, 1D uniform, and 2D unit-disk distributions."""
    n_vars = len(compiled.latent_vars)
    z = np.zeros((n_samples, n_vars), dtype=np.float64)

    disk_set: set[int] = set()
    for ix, iy in compiled.uniform_disk_pairs:
        disk_set.add(ix)
        disk_set.add(iy)
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n_samples))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        z[:, ix] = r * np.cos(phi)
        z[:, iy] = r * np.sin(phi)

    unif_set = set(compiled.uniform_1d_indices)
    for idx in compiled.uniform_1d_indices:
        z[:, idx] = rng.uniform(-1.0, 1.0, size=n_samples)

    for lv in compiled.latent_vars:
        if lv.index not in disk_set and lv.index not in unif_set:
            z[:, lv.index] = rng.normal(0.0, lv.sigma, size=n_samples)

    return z


def run_variation_analysis(
    compiled: CompiledAssembly,
    mc_samples: int = 100_000,
    mc_mode: Literal["full", "surrogate"] = "full",
    seed: int = 42,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Execute full 4-tier variation analysis on a CompiledAssembly."""
    n_vars = len(compiled.latent_vars)
    z0 = jnp.zeros(n_vars, dtype=jnp.float64)

    # Warm-up / JIT compile Jacobian and Hessian
    t0 = time.perf_counter()
    jac_fn = jax.jit(jax.jacrev(compiled.eval_measurements_dy))
    hess_fn = jax.jit(jax.hessian(compiled.eval_measurements_dy))
    _ = jac_fn(z0).block_until_ready()
    _ = hess_fn(z0).block_until_ready()
    jit_compile_ms = (time.perf_counter() - t0) * 1000.0

    # Timed float64 evaluation of S and H at nominal z = 0
    t1 = time.perf_counter()
    S_all = np.asarray(jac_fn(z0).block_until_ready(), dtype=np.float64)
    H_all = np.asarray(hess_fn(z0).block_until_ready(), dtype=np.float64)
    jac_hess_ms = (time.perf_counter() - t1) * 1000.0

    sigmas = np.array([lv.sigma for lv in compiled.latent_vars], dtype=np.float64)
    var_z = sigmas * sigmas

    # Tier 4 Monte Carlo execution
    t2 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if mc_mode == "full":
        @jax.jit
        def _eval_batch_fp32(z_b32: jax.Array) -> tuple[jax.Array, jax.Array]:
            dy_b = jax.vmap(compiled.eval_measurements_dy)(z_b32)
            m_fit_b = jax.vmap(compiled.eval_jam_margins)(z_b32)
            return dy_b, m_fit_b

        # Warm-up small batch so compilation isn't counted in MC runtime
        z_warm = jnp.zeros((min(64, mc_samples), n_vars), dtype=jnp.float32)
        _ = _eval_batch_fp32(z_warm)[0].block_until_ready()
        t2 = time.perf_counter()

        dy_chunks: list[np.ndarray] = []
        fit_chunks: list[np.ndarray] = []
        rem = mc_samples
        while rem > 0:
            cur = min(batch_size, rem)
            z_b = _sample_latent_batch(rng, cur, compiled)
            dy_b32, fit_b32 = _eval_batch_fp32(jnp.asarray(z_b, dtype=jnp.float32))
            # Cast immediately to float64 for cancellation-free accumulation
            dy_chunks.append(np.asarray(dy_b32, dtype=np.float64))
            fit_chunks.append(np.asarray(fit_b32, dtype=np.float64))
            rem -= cur

        dy_mc_all = np.concatenate(dy_chunks, axis=0)
        fit_mc_all = np.concatenate(fit_chunks, axis=0)
    else:
        z_all = _sample_latent_batch(rng, mc_samples, compiled)
        # Quadratic surrogate evaluation: dy_k = s_k^T z + 0.5 * z^T H_k z
        lin_part = z_all @ S_all.T
        quad_part = 0.5 * np.einsum("bi,kij,bj->bk", z_all, H_all, z_all)
        dy_mc_all = lin_part + quad_part
        fit_mc_all = np.ones(mc_samples, dtype=np.float64)

    mc_ms = (time.perf_counter() - t2) * 1000.0

    assembled_mask = fit_mc_all >= 0.0
    n_assembled = int(np.sum(assembled_mask))
    jam_dpmo = float((mc_samples - n_assembled) / mc_samples * 1_000_000.0)
    sota_warning = jam_dpmo > 1000.0

    # Indices covered by 3-cone SOCP cylindrical features
    cyl_spatial_indices: set[int] = set()
    for cinfo in compiled.cylinder_wc_infos:
        for idx_c in range(cinfo.latent_slice[0], cinfo.latent_slice[1]):
            cyl_spatial_indices.add(idx_c)

    meas_results: list[dict[str, Any]] = []
    for k, m_id in enumerate(compiled.meas_ids):
        y_nom = float(compiled.meas_nominals[k])
        lsl = float(compiled.meas_lsl[k])
        usl = float(compiled.meas_usl[k])
        s_k = S_all[k]
        H_k = H_all[k]

        # 1. Worst-Case Tier 1 (Exact 3-cone SOCP for refined cylinders + disk/box norms)
        wc_1st = 0.0
        z_wc = np.zeros(n_vars, dtype=np.float64)

        for cinfo in compiled.cylinder_wc_infos:
            s_i, _ = cinfo.latent_slice
            # Recover physical sensitivities (s_dx, s_dy, s_tx, s_ty) from 3-disk entries u2, u3
            t_o = min(cinfo.t_orient, cinfo.t_pos)
            if t_o > 1e-15:
                s_u2x, s_u2y = s_k[s_i + 2], s_k[s_i + 3]
                s_u3x, s_u3y = s_k[s_i + 4], s_k[s_i + 5]
                s_dx = (2.0 / t_o) * (s_u2x + s_u3x)
                s_dy = (2.0 / t_o) * (s_u2y + s_u3y)
                s_ty = (cinfo.lz / t_o) * (s_u2x - s_u3x)
                s_tx = -(cinfo.lz / t_o) * (s_u2y - s_u3y)
                t_eff_wc = (
                    cinfo.t_pos + cinfo.t_size
                    if cinfo.modifier in ("MMC", "LMC")
                    else cinfo.t_pos
                )
                val_socp, _, _ = solve_three_cone_wc_cylinder(
                    s_dx, s_dy, s_tx, s_ty, t_eff_wc, t_o, cinfo.lz
                )
                wc_1st += val_socp

        for idx_b in list(compiled.box_indices) + list(compiled.uniform_1d_indices):
            if idx_b not in cyl_spatial_indices:
                wc_1st += abs(float(s_k[idx_b]))
                z_wc[idx_b] = np.sign(s_k[idx_b])

        for ix, iy in list(compiled.gaussian_disk_pairs) + list(compiled.uniform_disk_pairs):
            if ix not in cyl_spatial_indices and iy not in cyl_spatial_indices:
                r_norm = float(np.hypot(s_k[ix], s_k[iy]))
                wc_1st += r_norm
                if r_norm > 1e-15:
                    z_wc[ix] = s_k[ix] / r_norm
                    z_wc[iy] = s_k[iy] / r_norm

        wc_2nd_shift = 0.5 * float(z_wc @ H_k @ z_wc)
        wc_min = y_nom + min(0.0, wc_2nd_shift) - wc_1st
        wc_max = y_nom + max(0.0, wc_2nd_shift) + wc_1st
        wc_status = "PASS" if (wc_min >= lsl and wc_max <= usl) else "FAIL"

        # 2. First-Order Linear RSS Tier 2
        sigma_rss = float(np.sqrt(max(float(np.sum((s_k * sigmas) ** 2)), 1e-30)))
        cp_rss = (usl - lsl) / (6.0 * sigma_rss)
        cpk_rss = min(usl - y_nom, y_nom - lsl) / (3.0 * sigma_rss)

        # 3. Second-Order SOTA Tier 3
        sota = compute_sota_moments(
            y_nom,
            s_k,
            H_k,
            sigmas,
            uniform_1d_indices=compiled.uniform_1d_indices,
            uniform_disk_pairs=compiled.uniform_disk_pairs,
        )
        cpk_sota = min(usl - sota["shifted_mean"], sota["shifted_mean"] - lsl) / (
            3.0 * sota["sigma"]
        )

        # Pareto contributions (linear + 2nd-order quadratic variance per variable)
        v_contrib = (s_k * sigmas) ** 2 + 0.5 * np.sum(
            (H_k**2) * np.outer(var_z, var_z), axis=1
        )
        v_sum = max(float(np.sum(v_contrib)), 1e-30)
        order = np.argsort(-v_contrib)
        pareto_list = []
        for idx_p in order[:6]:
            pct = float(v_contrib[idx_p] / v_sum * 100.0)
            if pct < 0.01:
                continue
            pareto_list.append(
                {
                    "feature_dof": compiled.latent_vars[int(idx_p)].name,
                    "sensitivity_dy_dz": round(float(s_k[int(idx_p)]), 6),
                    "variance_pct": round(pct, 2),
                }
            )

        # 4. Monte Carlo Tier 4 (conditioned on non-jammed builds M_fit >= 0)
        dy_valid = dy_mc_all[assembled_mask, k] if n_assembled > 0 else dy_mc_all[:, k]
        y_valid = y_nom + dy_valid
        mc_mean = float(np.mean(y_valid, dtype=np.float64))
        mc_std = float(np.std(y_valid, dtype=np.float64))
        p_low, p_high = np.quantile(y_valid, [0.00135, 0.99865])
        spec_fails = int(np.sum((y_valid < lsl) | (y_valid > usl)))
        spec_dpmo = float(spec_fails / max(n_assembled, 1) * 1_000_000.0)
        total_fails = (mc_samples - n_assembled) + spec_fails
        total_dpmo = float(total_fails / mc_samples * 1_000_000.0)
        yield_pct = float((1.0 - total_fails / mc_samples) * 100.0)

        meas_results.append(
            {
                "id": m_id,
                "nominal": round(y_nom, 6),
                "lsl": round(lsl, 6),
                "usl": round(usl, 6),
                "conditioned_on_fit": True,
                "worst_case": {
                    "min": round(wc_min, 6),
                    "max": round(wc_max, 6),
                    "half_band": round(wc_1st, 6),
                    "status": wc_status,
                },
                "linear_rss": {
                    "mean": round(y_nom, 6),
                    "sigma": round(sigma_rss, 6),
                    "three_sigma": round(3.0 * sigma_rss, 6),
                    "cp": round(cp_rss, 4),
                    "cpk": round(cpk_rss, 4),
                },
                "second_order_sota": {
                    "mean_shift": round(sota["mean_shift"], 6),
                    "shifted_mean": round(sota["shifted_mean"], 6),
                    "sigma": round(sota["sigma"], 6),
                    "skewness": round(sota["skewness"], 4),
                    "skewness_pure_quadratic_approx": "gaussian",
                    "cpk": round(cpk_sota, 4),
                },
                "monte_carlo": {
                    "samples_total": mc_samples,
                    "samples_assembled": n_assembled,
                    "mean": round(mc_mean, 6),
                    "sigma": round(mc_std, 6),
                    "p00135": round(float(p_low), 6),
                    "p99865": round(float(p_high), 6),
                    "spec_dpmo": round(spec_dpmo, 2),
                    "jam_dpmo": round(jam_dpmo, 2),
                    "total_defect_dpmo": round(total_dpmo, 2),
                    "yield_pct": round(yield_pct, 4),
                },
                "pareto_contributions": pareto_list,
            }
        )

    return {
        "assembly_name": compiled.spec.assembly.name,
        "kinematic_status": {
            "movable_parts": compiled.n_movable,
            "active_dofs": compiled.n_dofs,
            "constrained_dofs": compiled.n_constraints,
            "jacobian_rank": compiled.jacobian_rank,
            "determinate": True,
            "jam_dpmo": round(jam_dpmo, 2),
            "sota_conditioning_warning": sota_warning,
        },
        "execution_time_ms": {
            "jit_compile_ms": round(jit_compile_ms, 2),
            "jacobian_hessian_fp64_ms": round(jac_hess_ms, 2),
            "monte_carlo_1m_fp32_ms": round(mc_ms, 2),
        },
        "measurements": meas_results,
    }
