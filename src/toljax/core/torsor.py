"""GD&T feature torsor parameterizations, symmetric 3-disk coupled zones,
kink-free affine MMC/LMC bonus modifiers, and exact 3-cone SOCP worst-case solver."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array


def mmc_bonus_diameter(
    z_size: Array,
    t_size: float | Array,
    modifier: Literal["RFS", "MMC", "LMC"] = "RFS",
    is_external: bool = True,
) -> Array:
    """Compute kink-free affine MMC/LMC bonus diameter on the interior domain z_size in [-1, 1].

    Because z_size = 0 corresponds to the midpoint radius r_mid = (D_max + D_min)/4,
    the bonus diameter is strictly linear across the entire manufacturing size tolerance band.
    """
    if modifier == "RFS":
        return jnp.zeros_like(z_size)
    if (modifier == "MMC" and is_external) or (modifier == "LMC" and not is_external):
        return 0.5 * t_size * (1.0 - z_size)
    return 0.5 * t_size * (1.0 + z_size)


def map_planar_torsor(
    z_feat: Array,
    t_pos: float | Array,
    t_orient: float | Array,
    lx: float | Array,
    ly: float | Array,
) -> Array:
    """Map 3D latent vector z_feat = [z1, z2, z3] to planar feature torsor [tx, ty, 0, 0, 0, dz]."""
    t_pos_arr = jnp.asarray(t_pos, dtype=z_feat.dtype)
    t_orient_arr = jnp.asarray(t_orient, dtype=z_feat.dtype)
    t_o = jnp.where(t_pos_arr > 0.0, jnp.minimum(t_orient_arr, t_pos_arr), t_orient_arr)
    dz = 0.5 * jnp.maximum(0.0, t_pos_arr - t_o) * z_feat[..., 0]
    tx = (t_o / ly) * z_feat[..., 1]
    ty = (t_o / lx) * z_feat[..., 2]
    zero = jnp.zeros_like(dz)
    return jnp.stack([tx, ty, zero, zero, zero, dz], axis=-1)


def map_cylinder_torsor_symmetric(
    z_feat: Array,
    r_mid: float | Array,
    t_size: float | Array,
    t_pos: float | Array,
    t_orient: float | Array,
    lz: float | Array,
    modifier: Literal["RFS", "MMC", "LMC"] = "RFS",
    is_external: bool = True,
    mmc_mode: Literal["acceptance_truncated", "fully_consumed"] = "acceptance_truncated",
) -> tuple[Array, Array]:
    """Map latent vector z_feat to cylindrical feature torsor xi in R^6 and actual radius r_i.

    Supports both:
      - 5-DOF unrefined cylinder zone (z_feat has shape [..., 5]: u_top(2), u_bot(2), z_size(1))
      - 7-DOF symmetric 3-disk orientation-refined cylinder zone (z_feat has shape [..., 7]:
        u_1(2) pure translation, u_2(2) top tilt, u_3(2) bottom tilt, z_size(1)).

    For both cases:
      p_top = ((t_eff - t_o) / 2) * u_1 + (t_o / 2) * u_2
      p_bot = ((t_eff - t_o) / 2) * u_1 + (t_o / 2) * u_3
    """
    z_size = z_feat[..., -1]
    dr = 0.25 * t_size * z_size
    r_actual = r_mid + dr

    if mmc_mode == "fully_consumed":
        bonus = mmc_bonus_diameter(z_size, t_size, modifier=modifier, is_external=is_external)
    else:
        bonus = jnp.zeros_like(z_size)

    t_pos_arr = jnp.asarray(t_pos, dtype=z_feat.dtype)
    t_orient_arr = jnp.asarray(t_orient, dtype=z_feat.dtype)
    t_eff = t_pos_arr + bonus
    t_o_base = jnp.where(
        t_pos_arr > 0.0, jnp.minimum(t_orient_arr, t_pos_arr), t_orient_arr
    )
    t_o = jnp.minimum(t_o_base, t_eff)

    n_latent = z_feat.shape[-1]
    if n_latent == 5:
        # Unrefined 2-disk endpoint parameterization (t_o == t_eff)
        u_top = z_feat[..., 0:2]
        u_bot = z_feat[..., 2:4]
        p_top = 0.5 * t_eff[..., None] * u_top
        p_bot = 0.5 * t_eff[..., None] * u_bot
    elif n_latent == 7:
        # Symmetric 3-disk orientation-refined endpoint parameterization
        u_1 = z_feat[..., 0:2]
        u_2 = z_feat[..., 2:4]
        u_3 = z_feat[..., 4:6]
        w_trans = 0.5 * (t_eff - t_o)[..., None]
        w_tilt = 0.5 * t_o[..., None]
        p_top = w_trans * u_1 + w_tilt * u_2
        p_bot = w_trans * u_1 + w_tilt * u_3
    else:
        raise ValueError(f"Expected 5 or 7 latent DOFs for cylinder feature, got {n_latent}")

    delta_xy = 0.5 * (p_top + p_bot)
    diff_xy = p_top - p_bot
    tx = -diff_xy[..., 1] / lz
    ty = diff_xy[..., 0] / lz
    zero = jnp.zeros_like(tx)

    xi = jnp.stack([tx, ty, zero, delta_xy[..., 0], delta_xy[..., 1], zero], axis=-1)
    return xi, r_actual


def map_slot_torsor(
    z_feat: Array,
    r_mid: float | Array,
    t_size: float | Array,
    t_pos: float | Array,
    t_orient: float | Array,
    lx: float | Array,
    lz: float | Array,
    modifier: Literal["RFS", "MMC", "LMC"] = "RFS",
    mmc_mode: Literal["acceptance_truncated", "fully_consumed"] = "acceptance_truncated",
) -> tuple[Array, Array]:
    """Map 4D latent vector z_feat = [z_pos, z_tx, z_tz, z_size] to slot torsor and half-width."""
    z_size = z_feat[..., 3]
    dr = 0.25 * t_size * z_size
    r_actual = r_mid + dr

    bonus = (
        mmc_bonus_diameter(z_size, t_size, modifier=modifier, is_external=False)
        if mmc_mode == "fully_consumed"
        else jnp.zeros_like(z_size)
    )
    t_pos_arr = jnp.asarray(t_pos, dtype=z_feat.dtype)
    t_orient_arr = jnp.asarray(t_orient, dtype=z_feat.dtype)
    t_eff = t_pos_arr + bonus
    t_o = jnp.where(t_orient_arr > 0.0, jnp.minimum(t_orient_arr, t_eff), t_eff)

    dy = 0.5 * (t_eff - 0.5 * t_o) * z_feat[..., 0]
    tx = (0.5 * t_o / lz) * z_feat[..., 1]
    tz = (0.5 * t_o / lx) * z_feat[..., 2]
    zero = jnp.zeros_like(dy)

    xi = jnp.stack([tx, zero, tz, zero, dy, zero], axis=-1)
    return xi, r_actual


def map_point_torsor(z_feat: Array, t_pos: float | Array) -> Array:
    """Map 3D latent vector z_feat = [z1, z2, z3] to spherical/point feature torsor."""
    half_t = 0.5 * t_pos
    dx = half_t * z_feat[..., 0]
    dy = half_t * z_feat[..., 1]
    dz = half_t * z_feat[..., 2]
    zero = jnp.zeros_like(dx)
    return jnp.stack([zero, zero, zero, dx, dy, dz], axis=-1)


def solve_three_cone_wc_cylinder(
    s_dx: float,
    s_dy: float,
    s_tx: float,
    s_ty: float,
    t_eff_wc: float,
    t_orient_wc: float,
    lz: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve the exact ASME Y14.5.1 3-cone SOCP for a cylindrical feature's worst-case contribution:

      max_{p_top, p_bot in R^2}  g_top^T p_top + g_bot^T p_bot
      subject to:
        ||p_top||_2 <= R,  ||p_bot||_2 <= R,  ||p_top - p_bot||_2 <= t_o

    where R = t_eff_wc / 2, t_o = min(t_orient_wc, t_eff_wc),
    g_top = 0.5 * [s_dx, s_dy] + (1 / lz) * [s_ty, -s_tx],
    g_bot = 0.5 * [s_dx, s_dy] - (1 / lz) * [s_ty, -s_tx].

    By strong conic duality, the primal maximum equals the 2D Fermat-Weber dual minimum:
      min_{w in R^2}  R * ||g_top - w||_2 + R * ||g_bot + w||_2 + t_o * ||w||_2
    and includes orthogonal translation + tilt extremes where ||p_mid||^2 + ||dp||^2 = R^2.
    """
    R = 0.5 * float(t_eff_wc)
    t_o = min(float(t_orient_wc), float(t_eff_wc))

    g_top = np.array([0.5 * s_dx + s_ty / lz, 0.5 * s_dy - s_tx / lz], dtype=np.float64)
    g_bot = np.array([0.5 * s_dx - s_ty / lz, 0.5 * s_dy + s_tx / lz], dtype=np.float64)

    if R <= 0.0:
        return 0.0, np.zeros(2), np.zeros(2)

    # Case 1: Unconstrained tilt (t_o >= 2*R) -> independent disks ||p_top|| <= R, ||p_bot|| <= R
    n_top = float(np.linalg.norm(g_top))
    n_bot = float(np.linalg.norm(g_bot))
    p_top_free = R * g_top / n_top if n_top > 1e-15 else np.zeros(2)
    p_bot_free = R * g_bot / n_bot if n_bot > 1e-15 else np.zeros(2)
    if np.linalg.norm(p_top_free - p_bot_free) <= t_o + 1e-15:
        val = float(np.dot(g_top, p_top_free) + np.dot(g_bot, p_bot_free))
        return val, p_top_free, p_bot_free

    # Case 2: Active tilt constraint ||p_top - p_bot||_2 = t_o.
    # Let p_top = p_mid + d, p_bot = p_mid - d with ||d||_2 = r_d in [0, t_o / 2].
    # Notice objective is s_delta^T p_mid + (2/lz) (s_theta_perp)^T d.
    # Solve the 2D convex Fermat-Weber dual min_w f(w) via smoothed Newton / Weiszfeld + primal projection:
    w = 0.25 * (g_top - g_bot)
    eps = 1e-15
    for _ in range(60):
        v1 = w - g_top
        v2 = w + g_bot
        v3 = w
        d1 = max(float(np.linalg.norm(v1)), eps)
        d2 = max(float(np.linalg.norm(v2)), eps)
        d3 = max(float(np.linalg.norm(v3)), eps)

        grad = (R / d1) * v1 + (R / d2) * v2 + (t_o / d3) * v3
        if float(np.linalg.norm(grad)) < 1e-14 * (n_top + n_bot + 1e-15):
            break

        I2 = np.eye(2)
        H = (
            (R / d1) * (I2 - np.outer(v1, v1) / (d1 * d1))
            + (R / d2) * (I2 - np.outer(v2, v2) / (d2 * d2))
            + (t_o / d3) * (I2 - np.outer(v3, v3) / (d3 * d3))
            + 1e-12 * I2
        )
        step = np.linalg.solve(H, grad)
        w = w - step

    # Recover optimal primal (p_top, p_bot) from dual stationarity directions:
    u1 = (g_top - w) / max(float(np.linalg.norm(g_top - w)), eps)
    u2 = (g_bot + w) / max(float(np.linalg.norm(g_bot + w)), eps)
    p_top = R * u1
    p_bot = R * u2
    diff = p_top - p_bot
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm > t_o and diff_norm > eps:
        # Project onto intersection of ||p_top|| <= R, ||p_bot|| <= R, ||p_top - p_bot|| = t_o
        s_delta = g_top + g_bot
        s_tilt = g_top - g_bot
        n_d = float(np.linalg.norm(s_delta))
        n_t = float(np.linalg.norm(s_tilt))
        e_d = s_delta / n_d if n_d > eps else np.array([1.0, 0.0])
        e_t = s_tilt / n_t if n_t > eps else np.array([0.0, 1.0])
        # When ||p_top - p_bot|| = t_o, d = 0.5 * t_o * dir_t, and for orthogonal component
        # max ||p_mid|| = sqrt(R^2 - (t_o/2)^2) along e_d when e_d perp e_t:
        r_d = 0.5 * t_o
        # Alternate projection onto the 3 balls to machine precision
        for _ in range(40):
            d_vec = 0.5 * (p_top - p_bot)
            d_len = float(np.linalg.norm(d_vec))
            if d_len > r_d:
                d_vec = d_vec * (r_d / d_len)
            m_vec = 0.5 * (p_top + p_bot)
            p_top = m_vec + d_vec
            p_bot = m_vec - d_vec
            l1 = float(np.linalg.norm(p_top))
            if l1 > R:
                p_top = p_top * (R / l1)
            l2 = float(np.linalg.norm(p_bot))
            if l2 > R:
                p_bot = p_bot * (R / l2)

    dual_val = float(
        R * np.linalg.norm(g_top - w)
        + R * np.linalg.norm(g_bot + w)
        + t_o * np.linalg.norm(w)
    )
    return dual_val, p_top, p_bot
