"""Verification of ASME Y14.5.1 symmetric 3-disk coupled zones, 3-cone SOCP worst-case
extremes (including orthogonal translation + tilt), and MMC statistical modes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toljax.core.torsor import (
    map_cylinder_torsor_symmetric,
    mmc_bonus_diameter,
    solve_three_cone_wc_cylinder,
)


def test_symmetric_3disk_cylinder_containment_and_extrapolation_symmetry():
    """Verify the 7-DOF symmetric 3-disk map stays inside both t_eff and t_orient
    and reaches symmetric extremes at both +z and -z extrapolations.
    """
    r_mid = 2.997
    t_size = 0.012
    t_pos = 0.10
    t_orient = 0.04
    lz = 10.0

    rng = np.random.default_rng(7)
    for _ in range(200):
        # Sample u1, u2, u3 inside unit disks and z_size in [-1, 1]
        u = []
        for _ in range(3):
            r = np.sqrt(rng.uniform(0.0, 1.0))
            phi = rng.uniform(0.0, 2.0 * np.pi)
            u.extend([r * np.cos(phi), r * np.sin(phi)])
        z_size = rng.uniform(-1.0, 1.0)
        z_feat = jnp.array(u + [z_size], dtype=jnp.float64)

        xi, _ = map_cylinder_torsor_symmetric(
            z_feat,
            r_mid=r_mid,
            t_size=t_size,
            t_pos=t_pos,
            t_orient=t_orient,
            lz=lz,
            modifier="RFS",
            is_external=True,
            mmc_mode="acceptance_truncated",
        )
        tx, ty, _, dx, dy, _ = [float(v) for v in xi]
        p_top = np.array([dx + 0.5 * lz * ty, dy - 0.5 * lz * tx])
        p_bot = np.array([dx - 0.5 * lz * ty, dy + 0.5 * lz * tx])

        assert np.linalg.norm(p_top) <= 0.5 * t_pos + 1e-14
        assert np.linalg.norm(p_bot) <= 0.5 * t_pos + 1e-14
        assert np.linalg.norm(p_top - p_bot) <= t_orient + 1e-14

    # Verify symmetric upward (+L_z) and downward (-L_z) extrapolation extremes:
    # At s = +1.5 * L_z (mu = 1.0 beyond top) and s = -1.5 * L_z (mu = 1.0 beyond bottom),
    # max displacement is t_pos / 2 + 1.0 * t_orient = 0.05 + 0.04 = 0.09 mm for both!
    z_top_ext = jnp.array([1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=jnp.float64)
    xi_top, _ = map_cylinder_torsor_symmetric(
        z_top_ext, r_mid, t_size, t_pos, t_orient, lz
    )
    x_at_plus_1p5 = float(xi_top[3] + 1.5 * lz * xi_top[1])

    z_bot_ext = jnp.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    xi_bot, _ = map_cylinder_torsor_symmetric(
        z_bot_ext, r_mid, t_size, t_pos, t_orient, lz
    )
    x_at_minus_1p5 = float(xi_bot[3] - 1.5 * lz * xi_bot[1])

    np.testing.assert_allclose(x_at_plus_1p5, 0.5 * t_pos + t_orient, rtol=1e-14)
    np.testing.assert_allclose(x_at_minus_1p5, 0.5 * t_pos + t_orient, rtol=1e-14)


def test_three_cone_socp_captures_orthogonal_translation_and_tilt():
    """Verify 3-cone SOCP captures perpendicular translation + tilt where
    ||p_mid||^2 + (t_o/2)^2 = (t_eff/2)^2, which linear disk sums underestimate.
    """
    t_eff = 0.10  # R = 0.05
    t_o = 0.06    # r_d = 0.03 => orthogonal ||p_mid||_max = sqrt(0.05^2 - 0.03^2) = 0.04
    lz = 10.0

    # Objective: 1.0 * delta_x + (lz / 2) * theta_x (orthogonal directions: delta_x along X, theta_x tilts Y!)
    # With p_mid = [0.04, 0] and d = [0, -0.03]:
    #   p_top = [0.04, -0.03] (norm = 0.05 = R)
    #   p_bot = [0.04, +0.03] (norm = 0.05 = R)
    #   ||p_top - p_bot|| = 0.06 = t_o
    # Exact maximum = 1.0 * 0.04 + 1.0 * 0.03 = 0.07!
    # (Whereas a collinear budget split only gets (0.05 - 0.03)*1 + 0.03*sqrt(2) = 0.062426!)
    val_socp, p_top, p_bot = solve_three_cone_wc_cylinder(
        s_dx=1.0,
        s_dy=0.0,
        s_tx=0.5 * lz,
        s_ty=0.0,
        t_eff_wc=t_eff,
        t_orient_wc=t_o,
        lz=lz,
    )
    np.testing.assert_allclose(val_socp, 0.07, rtol=1e-6)
    assert np.linalg.norm(p_top) <= 0.05 + 1e-7
    assert np.linalg.norm(p_bot) <= 0.05 + 1e-7
    assert np.linalg.norm(p_top - p_bot) <= 0.06 + 1e-7


def test_mmc_acceptance_truncated_vs_fully_consumed_hessian_coupling():
    """Verify size-position cross-derivative in H is zero in 'acceptance_truncated' mode
    and exact (+/- t_size / 4) in 'fully_consumed' mode, with zero curvature spike at z=0.
    """
    r_mid = 2.997
    t_size = 0.012
    t_pos = 0.08
    t_orient = 0.08
    lz = 12.0
    z0 = jnp.zeros(5, dtype=jnp.float64)

    def dx_trunc(z):
        xi, _ = map_cylinder_torsor_symmetric(
            z, r_mid, t_size, t_pos, t_orient, lz, "MMC", True, "acceptance_truncated"
        )
        return xi[3]

    def dx_consumed(z):
        xi, _ = map_cylinder_torsor_symmetric(
            z, r_mid, t_size, t_pos, t_orient, lz, "MMC", True, "fully_consumed"
        )
        return xi[3]

    H_trunc = jax.hessian(dx_trunc)(z0)
    H_cons = jax.hessian(dx_consumed)(z0)

    # In acceptance_truncated mode, H is identically zero
    np.testing.assert_allclose(H_trunc, 0.0, atol=1e-16)
    # In fully_consumed mode, d^2(dx) / (d(u_top_x) d(z_size)) = -t_size / 8 = -0.0015 mm
    # and d^2(dx) / dz_size^2 = 0.0 (no softplus kink spike!)
    np.testing.assert_allclose(H_cons[0, 4], -t_size / 8.0, rtol=1e-14)
    np.testing.assert_allclose(H_cons[4, 4], 0.0, atol=1e-16)
