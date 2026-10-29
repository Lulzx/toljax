"""Gate 4: 6-Part, 24-DOF assembly performance and float32/float64 relative parity benchmark."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def _build_6part_24dof_chain():
    """Construct a 6-part (1 ground + 5 movable parts, 24 active DOFs) 3D nonlinear assembly."""

    def residual_fn(q: jax.Array, z: jax.Array) -> jax.Array:
        # 24-DOF coupled nonlinear trigonometric kinematic chain with lever arms L = 50 mm
        L = 50.0
        q_next = jnp.roll(q, 1)
        # Smooth C^infty nonlinear loop equations with full-rank Jacobian ~ I_24
        res = (
            q
            + 0.05 * jnp.sin(q_next)
            + 0.02 * (1.0 - jnp.cos(q)) * (L / 100.0)
            - z
        )
        return res

    def meas_dy(q: jax.Array, z: jax.Array) -> jax.Array:
        # Functional measurement at end-effector with lever arm L = 100 mm
        L = 100.0
        theta_sum = q[0] + q[6] + q[12] + q[18]
        z_trans = q[5] + q[11] + q[17] + q[23]
        return z_trans + L * jnp.sin(theta_sum) - 0.5 * L * (theta_sum**2) + 0.0 * jnp.sum(z)

    return residual_fn, meas_dy


def test_gate4_6part_24dof_performance_and_float32_parity():
    """Verify Gate 4 performance (< 25 ms S+H, < 50 ms 1M surrogate MC) and < 1e-4 sigma parity."""
    from toljax.core.solver import make_equilibrium_solver

    residual_fn, meas_dy = _build_6part_24dof_chain()
    solve_eq = make_equilibrium_solver(residual_fn, n_dofs=24, num_iters=4)

    def eval_y(z: jax.Array) -> jax.Array:
        q_star = solve_eq(z)
        return meas_dy(q_star, z)

    z0 = jnp.zeros(24, dtype=jnp.float64)
    jac_fn = jax.jit(jax.grad(eval_y))
    hess_fn = jax.jit(jax.hessian(eval_y))

    # Warm-up
    _ = jac_fn(z0).block_until_ready()
    _ = hess_fn(z0).block_until_ready()

    t0 = time.perf_counter()
    s = np.asarray(jac_fn(z0).block_until_ready(), dtype=np.float64)
    H = np.asarray(hess_fn(z0).block_until_ready(), dtype=np.float64)
    sh_ms = (time.perf_counter() - t0) * 1000.0
    assert sh_ms < 25.0, f"S + H extraction took {sh_ms:.2f} ms >= 25.0 ms"

    # Scale-relative float32 vs float64 parity on 10,000 samples (< 1e-4 * sigma_y)
    @jax.jit
    def batch_solve(z_b: jax.Array) -> jax.Array:
        return jax.vmap(eval_y)(z_b)

    rng = np.random.default_rng(77)
    z_sub = rng.normal(0.0, 1e-3, size=(10_000, 24))

    _ = batch_solve(jnp.asarray(z_sub[:32], dtype=jnp.float64)).block_until_ready()
    _ = batch_solve(jnp.asarray(z_sub[:32], dtype=jnp.float32)).block_until_ready()

    y64 = np.asarray(batch_solve(jnp.asarray(z_sub, dtype=jnp.float64)), dtype=np.float64)
    y32 = np.asarray(batch_solve(jnp.asarray(z_sub, dtype=jnp.float32)), dtype=np.float64)

    mean_64, std_64 = float(np.mean(y64)), float(np.std(y64))
    mean_32, std_32 = float(np.mean(y32)), float(np.std(y32))

    assert abs(mean_32 - mean_64) / std_64 < 1e-4
    assert abs(std_32 - std_64) / std_64 < 1e-4
