"""Gate 1A: Zero-angle singularity autodiff verification for SE(3) exp, expm1, and log."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toljax.core.se3 import expm1_compose, se3_exp, se3_expm1, se3_inv, se3_log


def _central_fd_jacobian(fn, x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Compute central finite-difference Jacobian of fn at x."""
    x_flat = x.reshape(-1)
    n = x_flat.size
    cols = []
    for i in range(n):
        e = jnp.zeros_like(x_flat).at[i].set(eps)
        dx = e.reshape(x.shape)
        f_plus = fn(x + dx)
        f_minus = fn(x - dx)
        cols.append(((f_plus - f_minus) / (2.0 * eps)).reshape(-1))
    jac_flat = jnp.stack(cols, axis=-1)
    out_shape = fn(x).shape
    return jac_flat.reshape(out_shape + x.shape)


def _central_fd_hessian(fn, x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Compute central finite-difference Hessian of fn at x via FD of jacrev."""
    jac_fn = jax.jacrev(fn)
    return _central_fd_jacobian(jac_fn, x, eps=eps)


def test_se3_exp_zero_angle_jacrev_and_hessian():
    """Verify jacrev and hessian of se3_exp at xi = 0 contain no NaNs and match FD to < 1e-10."""
    xi0 = jnp.zeros(6, dtype=jnp.float64)

    J_ad = jax.jacrev(se3_exp)(xi0)
    H_ad = jax.hessian(se3_exp)(xi0)

    assert not jnp.any(jnp.isnan(J_ad)), "NaN detected in jacrev(se3_exp) at zero!"
    assert not jnp.any(jnp.isnan(H_ad)), "NaN detected in hessian(se3_exp) at zero!"

    J_fd = _central_fd_jacobian(se3_exp, xi0, eps=1e-6)
    H_fd = _central_fd_hessian(se3_exp, xi0, eps=1e-5)

    j_err = float(jnp.linalg.norm(J_ad - J_fd))
    h_err = float(jnp.linalg.norm(H_ad - H_fd))
    assert j_err < 1e-11, f"se3_exp Jacobian error {j_err} >= 1e-11"
    assert h_err < 1e-10, f"se3_exp Hessian error {h_err} >= 1e-10"


def test_se3_log_identity_jacrev_and_hessian():
    """Verify jacrev and hessian of se3_log at T = I_4 contain no NaNs and match FD to < 1e-10."""
    T0 = jnp.eye(4, dtype=jnp.float64)

    J_ad = jax.jacrev(se3_log)(T0)
    H_ad = jax.hessian(se3_log)(T0)

    assert not jnp.any(jnp.isnan(J_ad)), "NaN detected in jacrev(se3_log) at identity!"
    assert not jnp.any(jnp.isnan(H_ad)), "NaN detected in hessian(se3_log) at identity!"

    J_fd = _central_fd_jacobian(se3_log, T0, eps=1e-6)
    H_fd = _central_fd_hessian(se3_log, T0, eps=1e-5)

    j_err = float(jnp.linalg.norm(J_ad - J_fd))
    h_err = float(jnp.linalg.norm(H_ad - H_fd))
    assert j_err < 1e-11, f"se3_log Jacobian error {j_err} >= 1e-11"
    assert h_err < 1e-10, f"se3_log Hessian error {h_err} >= 1e-10"


def test_se3_exp_log_roundtrip_and_expm1_composition():
    """Verify exp/log roundtrip across small and moderate angles and expm1_compose accuracy."""
    rng = np.random.default_rng(42)
    for scale in [0.0, 1e-9, 1e-5, 1e-2, 0.5]:
        xi = jnp.asarray(rng.standard_normal(6) * scale, dtype=jnp.float64)
        T = se3_exp(xi)
        xi_rec = se3_log(T)
        np.testing.assert_allclose(xi_rec, xi, atol=1e-14, rtol=1e-12)

        T_inv = se3_inv(T)
        np.testing.assert_allclose(T @ T_inv, jnp.eye(4), atol=1e-14)

    xi1 = jnp.asarray(rng.standard_normal(6) * 1e-3, dtype=jnp.float64)
    xi2 = jnp.asarray(rng.standard_normal(6) * 1e-3, dtype=jnp.float64)
    dR1, dp1 = se3_expm1(xi1)
    dR2, dp2 = se3_expm1(xi2)
    dR12, dp12 = expm1_compose(dR1, dp1, dR2, dp2)

    T12 = se3_exp(xi1) @ se3_exp(xi2)
    np.testing.assert_allclose(dR12, T12[:3, :3] - jnp.eye(3), atol=1e-15)
    np.testing.assert_allclose(dp12, T12[:3, 3], atol=1e-15)
