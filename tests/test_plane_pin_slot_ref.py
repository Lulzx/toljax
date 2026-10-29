"""Gate 1B, Gate 2, and Gate 4 vertical slice verification:
Closed-form trigonometric Plane-Pin-Slot reference vs recursive @jax.custom_jvp implicit solver,
unrolled Newton finite differences, SVD determinacy checks, and float32/float64 expm1 parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from toljax.core.mates import (
    FeatureFrame,
    eval_mate_frame_expm1,
    pin_hole_tilt_jam_margin,
    relative_mate_transform_expm1,
    residual_clocking_pin_slot,
    residual_parallel_pin_hole,
    residual_planar_coincident,
)
from toljax.core.solver import (
    RedundantMateError,
    UnderConstrainedMechanismError,
    check_kinematic_determinacy,
    make_equilibrium_solver,
)


def _build_plane_pin_slot_system():
    """Construct the 3D Plane-Pin-Slot assembly from Section 3 of spec.md.

    Latent vector z (9 variables):
      z[0]: datum_A tilt theta_x (rad)
      z[1]: datum_A tilt theta_y (rad)
      z[2]: bottom_pad shift delta_z (mm)
      z[3]: pin_1 shift delta_x at engagement plane (mm)
      z[4]: pin_1 shift delta_y at engagement plane (mm)
      z[5]: pin_2 shift delta_y at engagement plane (mm)
      z[6]: primary_pin float eta_x in [-1, 1]
      z[7]: primary_pin float eta_y in [-1, 1]
      z[8]: clocking_slot float eta_y in [-1, 1]
    """
    I3 = jnp.eye(3, dtype=jnp.float64)
    z3 = jnp.zeros(3, dtype=jnp.float64)

    # Housing is fixed at origin (T_0 = I_4); Carrier has nominal pose at [0, 0, 10]
    p_housing = z3
    p_carrier = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)

    # Mating feature frames (engagement plane at global z = 14.0 for pins/holes)
    datum_a_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_housing,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64),
        z_contact=jnp.array(0.0, dtype=jnp.float64),
    )
    bottom_pad_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_carrier,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        z_contact=jnp.array(0.0, dtype=jnp.float64),
    )

    pin1_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_housing,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([20.0, 20.0, 16.0], dtype=jnp.float64),
        z_contact=jnp.array(-2.0, dtype=jnp.float64),  # z = 14.0 global
    )
    hole1_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_carrier,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([20.0, 20.0, 4.0], dtype=jnp.float64),
        z_contact=jnp.array(0.0, dtype=jnp.float64),  # z = 14.0 global
    )

    pin2_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_housing,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([100.0, 20.0, 16.0], dtype=jnp.float64),
        z_contact=jnp.array(-2.0, dtype=jnp.float64),  # z = 14.0 global
    )
    slot2_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_carrier,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([100.0, 20.0, 4.0], dtype=jnp.float64),
        z_contact=jnp.array(0.0, dtype=jnp.float64),  # z = 14.0 global
    )

    weld_pad_frame = FeatureFrame(
        R_part_nom=I3,
        p_part_nom=p_carrier,
        R_feat_nom=I3,
        p_feat_nom=jnp.array([95.0, 45.0, 25.0], dtype=jnp.float64),
        z_contact=jnp.array(0.0, dtype=jnp.float64),
    )

    r_pin = jnp.array(5.994 / 2.0, dtype=jnp.float64)
    r_hole = jnp.array(6.0625 / 2.0, dtype=jnp.float64)
    L_e = jnp.array(8.0, dtype=jnp.float64)

    def _cast_frame(f: FeatureFrame, dtype) -> FeatureFrame:
        return FeatureFrame(
            R_part_nom=f.R_part_nom.astype(dtype),
            p_part_nom=f.p_part_nom.astype(dtype),
            R_feat_nom=f.R_feat_nom.astype(dtype),
            p_feat_nom=f.p_feat_nom.astype(dtype),
            z_contact=f.z_contact.astype(dtype),
        )

    def residual_fn(q: jax.Array, z: jax.Array) -> jax.Array:
        dtype = q.dtype
        q0_fixed = jnp.zeros(6, dtype=dtype)
        zero3 = jnp.zeros(3, dtype=dtype)

        xi_datum_a = jnp.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0], dtype=dtype)
        xi_bottom_pad = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, z[2]], dtype=dtype)
        xi_pin1 = jnp.array([0.0, 0.0, 0.0, z[3], z[4], 0.0], dtype=dtype)
        xi_hole1 = jnp.zeros(6, dtype=dtype)
        xi_pin2 = jnp.array([0.0, 0.0, 0.0, 0.0, z[5], 0.0], dtype=dtype)
        xi_slot2 = jnp.zeros(6, dtype=dtype)

        # Mate 1: Primary plane (3 DOF)
        dR1, p1 = relative_mate_transform_expm1(
            _cast_frame(datum_a_frame, dtype),
            q0_fixed,
            xi_datum_a,
            _cast_frame(bottom_pad_frame, dtype),
            q,
            xi_bottom_pad,
            dp_nom_precomputed=zero3,
        )
        r1 = residual_planar_coincident(dR1, p1)

        # Mate 2: Primary locating pin in hole (2 DOF)
        dR2, p2 = relative_mate_transform_expm1(
            _cast_frame(pin1_frame, dtype),
            q0_fixed,
            xi_pin1,
            _cast_frame(hole1_frame, dtype),
            q,
            xi_hole1,
            dp_nom_precomputed=zero3,
        )
        r2 = residual_parallel_pin_hole(
            p2,
            r_hole.astype(dtype),
            r_pin.astype(dtype),
            jnp.stack([z[6], z[7]]),
        )

        # Mate 3: Secondary clocking pin in slot (1 DOF)
        dR3, p3 = relative_mate_transform_expm1(
            _cast_frame(pin2_frame, dtype),
            q0_fixed,
            xi_pin2,
            _cast_frame(slot2_frame, dtype),
            q,
            xi_slot2,
            dp_nom_precomputed=zero3,
        )
        r3 = residual_clocking_pin_slot(
            p3,
            r_hole.astype(dtype),
            r_pin.astype(dtype),
            z[8],
        )

        return jnp.concatenate([r1, r2, r3], axis=-1)

    def weld_pad_displacement(q: jax.Array, z: jax.Array) -> jax.Array:
        """Return incremental [dx, dy, dz] displacement of carrier.weld_pad in global coordinates."""
        dtype = q.dtype
        xi_weld = jnp.zeros(6, dtype=dtype)
        _, _, _, dp_weld = eval_mate_frame_expm1(
            _cast_frame(weld_pad_frame, dtype), q, xi_weld
        )
        return dp_weld

    return residual_fn, weld_pad_displacement, float(r_hole - r_pin)


def test_closed_form_plane_pin_slot_jacobian_and_hessian_parity():
    """Gate 1B: Compare solver implicit S and H against exact closed-form trigonometry."""
    residual_fn, weld_pad_displacement, c_r = _build_plane_pin_slot_system()

    # Verify dimensionless two-sided SVD rank check passes with rank = 6
    q0 = jnp.zeros(6, dtype=jnp.float64)
    z0 = jnp.zeros(9, dtype=jnp.float64)
    rank, _ = check_kinematic_determinacy(residual_fn, q0, z0, l_char=100.0)
    assert rank == 6

    solve_eq = make_equilibrium_solver(residual_fn, n_dofs=6, num_iters=5)

    def meas_xy_solver(z_planar: jax.Array) -> jax.Array:
        """Map planar inputs [dx_pin1, dy_pin1, dy_pin2, eta_x1, eta_y1, eta_y2] -> weld_pad [dx, dy]."""
        z_full = jnp.array(
            [
                0.0,
                0.0,
                0.0,
                z_planar[0],
                z_planar[1],
                z_planar[2],
                z_planar[3],
                z_planar[4],
                z_planar[5],
            ],
            dtype=z_planar.dtype,
        )
        q_star = solve_eq(z_full)
        return weld_pad_displacement(q_star, z_full)[:2]

    def meas_xy_exact_trig(z_planar: jax.Array) -> jax.Array:
        """Exact closed-form planar trigonometry for Plane-Pin-Slot mechanism.

        Pin 1 center is at (20 + dx_pin1, 20 + dy_pin1).
        Hole 1 center (20, 20 on carrier) sits at Pin 1 + c_r * [eta_x1, eta_y1].
        Pin 2 is at (100, 20 + dy_pin2), distance D = 80 mm from Pin 1 along X.
        Slot 2 center (100, 20 on carrier) has Y = 20 + dy_pin2 + c_r * eta_y2.
        Since Slot 2 is at distance D = 80 along carrier +X from Hole 1:
          Y_slot2 - Y_hole1 = D * sin(theta_z)
          => sin(theta_z) = ((dy_pin2 + c_r * eta_y2) - (dy_pin1 + c_r * eta_y1)) / 80.0
          => cos(theta_z) = sqrt(1 - sin^2(theta_z))
        And weld_pad at (95, 45) on carrier (lever arm [75, 25] from Hole 1 at (20, 20)) moves by:
          dx = (dx_pin1 + c_r * eta_x1) + 75 * (cos(theta_z) - 1) - 25 * sin(theta_z)
          dy = (dy_pin1 + c_r * eta_y1) + 75 * sin(theta_z) + 25 * (cos(theta_z) - 1)
        """
        dx_p1, dy_p1, dy_p2, eta_x1, eta_y1, eta_y2 = z_planar
        ux1 = dx_p1 + c_r * eta_x1
        uy1 = dy_p1 + c_r * eta_y1
        uy2 = dy_p2 + c_r * eta_y2

        sin_tz = (uy2 - uy1) / 80.0
        cos_tz_m1 = jnp.sqrt(1.0 - sin_tz * sin_tz) - 1.0

        dx = ux1 + 75.0 * cos_tz_m1 - 25.0 * sin_tz
        dy = uy1 + 75.0 * sin_tz + 25.0 * cos_tz_m1
        return jnp.stack([dx, dy])

    z_p0 = jnp.zeros(6, dtype=jnp.float64)

    S_solver = jax.jacrev(meas_xy_solver)(z_p0)
    S_exact = jax.jacrev(meas_xy_exact_trig)(z_p0)

    H_solver = jax.hessian(meas_xy_solver)(z_p0)
    H_exact = jax.hessian(meas_xy_exact_trig)(z_p0)

    s_rel_err = float(jnp.max(jnp.abs(S_solver - S_exact)) / jnp.max(jnp.abs(S_exact)))
    h_rel_err = float(jnp.max(jnp.abs(H_solver - H_exact)) / jnp.max(jnp.abs(H_exact)))

    assert s_rel_err < 1e-12, f"Jacobian S relative error vs exact trig: {s_rel_err}"
    assert h_rel_err < 1e-11, f"Hessian H relative error vs exact trig: {h_rel_err}"

    # Verify physical second-order cosine foreshortening cross-term between uy1 and uy2 is captured:
    # d^2(dx) / (d(dy_p1) d(dy_p2)) = +75 / 80^2 = +0.01171875 mm^-1
    np.testing.assert_allclose(H_solver[0, 1, 2], 75.0 / (80.0**2), rtol=1e-12)
    # While cross-derivative between x-float (eta_x1) and y-slot-float (eta_y2) is zero:
    np.testing.assert_allclose(H_solver[0, 3, 5], 0.0, atol=1e-15)


def test_ift_vs_unrolled_newton_finite_differences_3d():
    """Gate 2: Compare analytical S and H from recursive @jax.custom_jvp against FD of 25-step Newton."""
    residual_fn, weld_pad_displacement, _ = _build_plane_pin_slot_system()
    solve_eq = make_equilibrium_solver(residual_fn, n_dofs=6, num_iters=5)

    def meas_3d_ift(z: jax.Array) -> jax.Array:
        q_star = solve_eq(z)
        return weld_pad_displacement(q_star, z)

    def meas_3d_unrolled(z: jax.Array) -> jax.Array:
        q_init = jnp.zeros(6, dtype=z.dtype)

        def body(_, q):
            f_val = residual_fn(q, z)
            J_q = jax.jacfwd(residual_fn, argnums=0)(q, z)
            return q - jnp.linalg.solve(J_q, f_val)

        q = jax.lax.fori_loop(0, 8, body, q_init)
        return weld_pad_displacement(q, z)

    z0 = jnp.zeros(9, dtype=jnp.float64)
    S_ad = jax.jacrev(meas_3d_ift)(z0)
    H_ad = jax.hessian(meas_3d_ift)(z0)

    # Central finite differences of unrolled solver
    meas_jit = jax.jit(meas_3d_unrolled)
    eps_s = 1e-6
    cols_s = []
    for i in range(9):
        dz = jnp.zeros(9, dtype=jnp.float64).at[i].set(eps_s)
        cols_s.append((meas_jit(z0 + dz) - meas_jit(z0 - dz)) / (2.0 * eps_s))
    S_fd = jnp.stack(cols_s, axis=-1)

    eps_h = 1e-5
    jac_unrolled = jax.jit(jax.jacrev(meas_3d_unrolled))
    cols_h = []
    for i in range(9):
        dz = jnp.zeros(9, dtype=jnp.float64).at[i].set(eps_h)
        cols_h.append((jac_unrolled(z0 + dz) - jac_unrolled(z0 - dz)) / (2.0 * eps_h))
    H_fd = jnp.stack(cols_h, axis=-1)

    s_err = float(jnp.max(jnp.abs(S_ad - S_fd)) / jnp.max(jnp.abs(S_fd)))
    h_err = float(jnp.max(jnp.abs(H_ad - H_fd)) / jnp.max(jnp.abs(H_fd)))

    assert s_err < 1e-10, f"IFT vs FD Jacobian relative error {s_err} >= 1e-10"
    assert h_err < 1e-7, f"IFT vs FD Hessian relative error {h_err} >= 1e-7"


def test_primary_pin_engagement_depth_tilt_jam_margin():
    """Verify engagement-depth tilt binding detects when angular cocking jams the pin in its hole."""
    r_hole = jnp.array(3.03125, dtype=jnp.float64)
    r_pin = jnp.array(2.99700, dtype=jnp.float64)
    c_r = float(r_hole - r_pin)  # 0.03425 mm radial clearance
    L_e = jnp.array(8.0, dtype=jnp.float64)

    # Zero tilt -> margin equals full radial clearance C_r
    dR_zero = jnp.zeros((3, 3), dtype=jnp.float64)
    m0 = float(pin_hole_tilt_jam_margin(r_hole, r_pin, dR_zero, L_e))
    np.testing.assert_allclose(m0, c_r, rtol=1e-14)

    # Tilt exceeding 2 * C_r / L_e = 2 * 0.03425 / 8.0 = 0.0085625 rad -> jam (M_fit < 0)
    dR_jam = jnp.zeros((3, 3), dtype=jnp.float64).at[0, 2].set(0.010)
    m_jam = float(pin_hole_tilt_jam_margin(r_hole, r_pin, dR_jam, L_e))
    assert m_jam < 0.0
    np.testing.assert_allclose(m_jam, c_r - 0.5 * 8.0 * 0.010, rtol=1e-14)

    # Tilt below the binding limit always leaves a feasible float position -> no jam
    dR_ok = jnp.zeros((3, 3), dtype=jnp.float64).at[0, 2].set(0.005).at[1, 2].set(-0.004)
    m_ok = float(pin_hole_tilt_jam_margin(r_hole, r_pin, dR_ok, L_e))
    np.testing.assert_allclose(m_ok, c_r - 0.5 * 8.0 * np.hypot(0.005, 0.004), rtol=1e-14)
    assert m_ok > 0.0


def test_svd_rank_detects_under_and_over_constrained_mates():
    """Verify dimensionless SVD rank check catches both missing and redundant mates."""
    residual_fn, _, _ = _build_plane_pin_slot_system()
    q0 = jnp.zeros(6, dtype=jnp.float64)
    z0 = jnp.zeros(9, dtype=jnp.float64)

    # Under-constrained: drop the clocking slot (only 5 constraints for 6 DOFs)
    def under_constrained_fn(q, z):
        return residual_fn(q, z)[:5]

    with pytest.raises(UnderConstrainedMechanismError):
        check_kinematic_determinacy(under_constrained_fn, q0, z0)

    # Over-constrained: duplicate the primary pin constraint (8 constraints for 6 DOFs)
    def over_constrained_fn(q, z):
        r = residual_fn(q, z)
        return jnp.concatenate([r, r[3:5]], axis=-1)

    with pytest.raises(RedundantMateError):
        check_kinematic_determinacy(over_constrained_fn, q0, z0)


def test_expm1_float32_vs_float64_monte_carlo_parity():
    """Gate 4 parity check: 10,000 Monte Carlo samples in float32 vs float64 agree to < 1e-4 sigma_y."""
    residual_fn, weld_pad_displacement, _ = _build_plane_pin_slot_system()
    solve_eq = make_equilibrium_solver(residual_fn, n_dofs=6, num_iters=4)

    @jax.jit
    def batch_eval(z_batch: jax.Array) -> jax.Array:
        def single(z):
            q_star = solve_eq(z)
            return weld_pad_displacement(q_star, z)[2]  # vertical Z deviation

        return jax.vmap(single)(z_batch)

    rng = np.random.default_rng(123)
    n_samples = 10000
    scales = np.array([2e-4, 2e-4, 0.015, 0.015, 0.015, 0.015, 0.5, 0.5, 0.5])
    z_np = rng.standard_normal((n_samples, 9)) * scales[None, :]

    # Evaluate in float64
    dy_fp64 = batch_eval(jnp.asarray(z_np, dtype=jnp.float64))
    mean_64 = float(jnp.mean(dy_fp64))
    std_64 = float(jnp.std(dy_fp64))

    # Evaluate in float32 and accumulate in float64
    dy_fp32 = batch_eval(jnp.asarray(z_np, dtype=jnp.float32))
    mean_32 = float(jnp.mean(dy_fp32.astype(jnp.float64)))
    std_32 = float(jnp.std(dy_fp32.astype(jnp.float64)))

    mean_rel_err = abs(mean_32 - mean_64) / std_64
    std_rel_err = abs(std_32 - std_64) / std_64

    assert mean_rel_err < 1e-4, f"float32 MC mean drift {mean_rel_err} * sigma >= 1e-4 * sigma"
    assert std_rel_err < 1e-4, f"float32 MC std drift {std_rel_err} * sigma >= 1e-4 * sigma"
