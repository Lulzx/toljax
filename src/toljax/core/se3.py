"""Differentiable SO(3) and SE(3) Lie group primitives with double-where Taylor safety
and cancellation-free expm1 perturbation arithmetic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def _theta_cut_sq(dtype: jnp.dtype) -> float:
    """Precision-aware squared angle threshold for Taylor series switching."""
    return 1e-6 if dtype == jnp.float64 else 1e-1


def skew(v: Array) -> Array:
    """Map a 3-vector v = [vx, vy, vz] to its 3x3 skew-symmetric matrix [v]_x in so(3)."""
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    z = jnp.zeros_like(vx)
    row0 = jnp.stack([z, -vz, vy], axis=-1)
    row1 = jnp.stack([vz, z, -vx], axis=-1)
    row2 = jnp.stack([-vy, vx, z], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def unskew(M: Array) -> Array:
    """Extract axial 3-vector v = 0.5 * [M_32 - M_23, M_13 - M_31, M_21 - M_12] from a 3x3 matrix."""
    return 0.5 * jnp.stack(
        [
            M[..., 2, 1] - M[..., 1, 2],
            M[..., 0, 2] - M[..., 2, 0],
            M[..., 1, 0] - M[..., 0, 1],
        ],
        axis=-1,
    )


def _se3_taylor_coeffs(theta_sq: Array) -> tuple[Array, Array, Array]:
    """Compute Taylor-safe Rodrigues and left-Jacobian coefficients (c1, c2, c3).

    Uses the JAX double-jnp.where idiom so that neither primal evaluation nor
    any order of forward/reverse differentiation at theta = 0 ever evaluates
    sqrt(0) or 0/0 in the unselected branch.
    """
    cut_sq = _theta_cut_sq(theta_sq.dtype)
    is_small = theta_sq < cut_sq

    # Double-where guard: replace theta_sq with 1.0 before sqrt in the exact branch.
    theta_sq_safe = jnp.where(is_small, jnp.ones_like(theta_sq), theta_sq)
    theta_safe = jnp.sqrt(theta_sq_safe)

    t2 = theta_sq
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4

    # 8th-order even-power Taylor series
    c1_taylor = 1.0 - t2 / 6.0 + t4 / 120.0 - t6 / 5040.0 + t8 / 362880.0
    c2_taylor = 0.5 - t2 / 24.0 + t4 / 720.0 - t6 / 40320.0 + t8 / 3628800.0
    c3_taylor = (
        1.0 / 6.0
        - t2 / 120.0
        + t4 / 5040.0
        - t6 / 362880.0
        + t8 / 39916800.0
    )

    sin_t = jnp.sin(theta_safe)
    cos_t = jnp.cos(theta_safe)
    c1_exact = sin_t / theta_safe
    c2_exact = (1.0 - cos_t) / theta_sq_safe
    c3_exact = (theta_safe - sin_t) / (theta_sq_safe * theta_safe)

    c1 = jnp.where(is_small, c1_taylor, c1_exact)
    c2 = jnp.where(is_small, c2_taylor, c2_exact)
    c3 = jnp.where(is_small, c3_taylor, c3_exact)
    return c1, c2, c3


def _c4_coeff(theta_sq: Array) -> Array:
    """Compute left-Jacobian inverse coefficient c4(theta^2) with double-where safety."""
    cut_sq = _theta_cut_sq(theta_sq.dtype)
    is_small = theta_sq < cut_sq
    theta_sq_safe = jnp.where(is_small, jnp.ones_like(theta_sq), theta_sq)
    theta_safe = jnp.sqrt(theta_sq_safe)

    t2 = theta_sq
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4

    c4_taylor = (
        1.0 / 12.0
        + t2 / 720.0
        + t4 / 30240.0
        + t6 / 1209600.0
        + t8 / 47900160.0
    )
    sin_t = jnp.sin(theta_safe)
    cos_t = jnp.cos(theta_safe)
    c4_exact = (1.0 / theta_sq_safe) - (1.0 + cos_t) / (2.0 * theta_safe * sin_t)
    return jnp.where(is_small, c4_taylor, c4_exact)


def so3_expm1(theta: Array) -> Array:
    """Compute incremental rotation matrix dR = R(theta) - I_3 without +I/-I cancellation."""
    theta_sq = jnp.sum(theta * theta, axis=-1, keepdims=True)[..., None]
    c1, c2, _ = _se3_taylor_coeffs(theta_sq)
    W = skew(theta)
    W2 = jnp.matmul(W, W)
    return c1 * W + c2 * W2


def so3_exp(theta: Array) -> Array:
    """Compute rotation matrix R(theta) = I_3 + dR(theta) in SO(3)."""
    dR = so3_expm1(theta)
    eye = jnp.eye(3, dtype=theta.dtype)
    return eye + dR


def se3_expm1(xi: Array) -> tuple[Array, Array]:
    """Map twist xi = [theta, delta] in R^6 to incremental (dR = R - I_3, dp = V(theta)*delta)."""
    theta = xi[..., :3]
    delta = xi[..., 3:6]
    theta_sq = jnp.sum(theta * theta, axis=-1, keepdims=True)[..., None]
    c1, c2, c3 = _se3_taylor_coeffs(theta_sq)

    W = skew(theta)
    W2 = jnp.matmul(W, W)
    dR = c1 * W + c2 * W2

    eye = jnp.eye(3, dtype=xi.dtype)
    V = eye + c2 * W + c3 * W2
    dp = jnp.matmul(V, delta[..., None])[..., 0]
    return dR, dp


def se3_exp(xi: Array) -> Array:
    """Map twist xi = [theta, delta] in R^6 to homogeneous 4x4 matrix T in SE(3)."""
    dR, dp = se3_expm1(xi)
    R = jnp.eye(3, dtype=xi.dtype) + dR
    top = jnp.concatenate([R, dp[..., None]], axis=-1)
    bottom_row = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 0.0, 1.0], dtype=xi.dtype),
        xi.shape[:-1] + (1, 4),
    )
    return jnp.concatenate([top, bottom_row], axis=-2)


def se3_log(T: Array) -> Array:
    """Map homogeneous 4x4 matrix T in SE(3) to twist xi = [theta, delta] in R^6.

    Avoids arccos singularities at T = I_4 by expanding atan2(s, c) / s in powers of
    u = s^2 / c^2 near s^2 -> 0 (with c > 0), ensuring C^infty smoothness both on SE(3)
    and across ambient 4x4 perturbations.
    """
    R = T[..., :3, :3]
    p = T[..., :3, 3]

    v = unskew(R)
    s_sq = jnp.sum(v * v, axis=-1, keepdims=True)
    trace_R = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = 0.5 * (trace_R[..., None] - 1.0)

    cut_sq = _theta_cut_sq(T.dtype)
    is_small = (s_sq < cut_sq) & (c > 0.0)

    s_sq_safe = jnp.where(is_small, jnp.ones_like(s_sq), s_sq)
    s_safe = jnp.sqrt(s_sq_safe)
    c_safe = jnp.where(is_small, c, jnp.ones_like(c))

    # Exact Taylor series of atan2(s, c) / s = (1/c) * arctan(s/c) / (s/c)
    u = s_sq / (c_safe * c_safe)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    scale_taylor = (1.0 / c_safe) * (
        1.0 - u / 3.0 + u2 / 5.0 - u3 / 7.0 + u4 / 9.0
    )
    scale_exact = jnp.arctan2(s_safe, c) / s_safe
    scale = jnp.where(is_small, scale_taylor, scale_exact)

    theta = scale * v
    theta_sq = jnp.sum(theta * theta, axis=-1, keepdims=True)[..., None]
    c4 = _c4_coeff(theta_sq)

    W = skew(theta)
    W2 = jnp.matmul(W, W)
    V_inv = jnp.eye(3, dtype=T.dtype) - 0.5 * W + c4 * W2
    delta = jnp.matmul(V_inv, p[..., None])[..., 0]
    return jnp.concatenate([theta, delta], axis=-1)


def se3_inv(T: Array) -> Array:
    """Invert homogeneous 4x4 transformation matrix T in SE(3)."""
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    R_T = jnp.swapaxes(R, -1, -2)
    p_inv = -jnp.matmul(R_T, p[..., None])[..., 0]
    top = jnp.concatenate([R_T, p_inv[..., None]], axis=-1)
    bottom_row = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 0.0, 1.0], dtype=T.dtype),
        T.shape[:-2] + (1, 4),
    )
    return jnp.concatenate([top, bottom_row], axis=-2)


def expm1_compose(
    dR1: Array, dp1: Array, dR2: Array, dp2: Array
) -> tuple[Array, Array]:
    """Compose two incremental transforms (I + dR1, dp1) * (I + dR2, dp2) without +I/-I cancellation."""
    dR12 = dR1 + dR2 + jnp.matmul(dR1, dR2)
    dp12 = dp1 + dp2 + jnp.matmul(dR1, dp2[..., None])[..., 0]
    return dR12, dp12
