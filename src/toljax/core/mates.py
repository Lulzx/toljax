"""Direct geometric mate residuals (dR, p), engagement-plane offsets, tilt-binding jam margins,
and cancellation-free functional measurement evaluators."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from toljax.core.se3 import expm1_compose, se3_expm1


class FeatureFrame(NamedTuple):
    """Nominal frame and engagement parameters for a single mating feature."""

    R_part_nom: Array       # [3, 3] nominal rotation of parent part
    p_part_nom: Array       # [3] nominal origin of parent part
    R_feat_nom: Array       # [3, 3] nominal local rotation of feature on part
    p_feat_nom: Array       # [3] nominal local origin of feature on part
    z_contact: Array        # scalar axial offset from feature midpoint to engagement plane


def eval_mate_frame_expm1(
    frame: FeatureFrame,
    q_part: Array,
    xi_feat: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate a feature's mate frame at its engagement plane using cancellation-free expm1 arithmetic.

    Returns:
        R_nom: [3, 3] nominal global rotation R_part_nom @ R_feat_nom
        p_nom: [3] nominal global engagement point
        dR_local: [3, 3] incremental rotation in R_nom frame such that R_global = R_nom @ (I + dR_local)
        dp_global: [3] exact incremental displacement p_global - p_nom in global coordinates
    """
    dR_q, dp_q = se3_expm1(q_part)
    dR_xi, dp_xi = se3_expm1(xi_feat)

    z_vec = jnp.stack(
        [
            jnp.zeros_like(frame.z_contact),
            jnp.zeros_like(frame.z_contact),
            frame.z_contact,
        ],
        axis=-1,
    )

    # Nominal mate point in part frame and global frame
    p_local_nom = frame.p_feat_nom + jnp.matmul(frame.R_feat_nom, z_vec[..., None])[..., 0]
    p_global_nom = frame.p_part_nom + jnp.matmul(frame.R_part_nom, p_local_nom[..., None])[..., 0]
    R_global_nom = jnp.matmul(frame.R_part_nom, frame.R_feat_nom)

    # Incremental displacement inside feature frame at z_contact
    dp_feat = dp_xi + jnp.matmul(dR_xi, z_vec[..., None])[..., 0]
    R_feat_dp = jnp.matmul(frame.R_feat_nom, dp_feat[..., None])[..., 0]

    # Incremental displacement in part frame (lever arm p_local_nom only multiplies dR_q!)
    dp_in_part = (
        dp_q
        + jnp.matmul(dR_q, p_local_nom[..., None])[..., 0]
        + R_feat_dp
        + jnp.matmul(dR_q, R_feat_dp[..., None])[..., 0]
    )
    dp_global = jnp.matmul(frame.R_part_nom, dp_in_part[..., None])[..., 0]

    # Incremental rotation expressed in R_global_nom frame:
    # (I + R_feat^T dR_q R_feat) (I + dR_xi) - I
    R_feat_T = jnp.swapaxes(frame.R_feat_nom, -1, -2)
    dR_q_in_feat = jnp.matmul(R_feat_T, jnp.matmul(dR_q, frame.R_feat_nom))
    dR_local = dR_q_in_feat + dR_xi + jnp.matmul(dR_q_in_feat, dR_xi)

    return R_global_nom, p_global_nom, dR_local, dp_global


def relative_mate_transform_expm1(
    src_frame: FeatureFrame,
    q_src: Array,
    xi_src: Array,
    tgt_frame: FeatureFrame,
    q_tgt: Array,
    xi_tgt: Array,
    dp_nom_precomputed: Array | None = None,
) -> tuple[Array, Array]:
    """Compute relative incremental rotation dR = R_src^T R_tgt - I and position p in source mate frame.

    When nominal mate frames coincide (the standard case for aligned assembly mates),
    dp_nom_precomputed = 0, and neither rotations nor lever-arm translations suffer
    any +I/-I or +r/-r floating-point cancellation in float32 or float64.
    """
    R_src_nom, p_src_nom, dR_src_local, dp_src_global = eval_mate_frame_expm1(
        src_frame, q_src, xi_src
    )
    R_tgt_nom, p_tgt_nom, dR_tgt_local, dp_tgt_global = eval_mate_frame_expm1(
        tgt_frame, q_tgt, xi_tgt
    )

    if dp_nom_precomputed is None:
        dp_nom = p_tgt_nom - p_src_nom
    else:
        dp_nom = dp_nom_precomputed

    diff_p_global = dp_nom + (dp_tgt_global - dp_src_global)

    # Rotate diff_p_global into perturbed source frame: R_src^T = (I + dR_src_local^T) R_src_nom^T
    R_src_nom_T = jnp.swapaxes(R_src_nom, -1, -2)
    diff_in_src_nom = jnp.matmul(R_src_nom_T, diff_p_global[..., None])[..., 0]
    dR_src_local_T = jnp.swapaxes(dR_src_local, -1, -2)
    p_rel = (
        diff_in_src_nom
        + jnp.matmul(dR_src_local_T, diff_in_src_nom[..., None])[..., 0]
    )

    # Relative rotation: R_src^T R_tgt = (I + dR_src_local^T) (R_src_nom^T R_tgt_nom) (I + dR_tgt_local)
    R_rel_nom = jnp.matmul(R_src_nom_T, R_tgt_nom)
    dR_nom = R_rel_nom - jnp.eye(3, dtype=R_rel_nom.dtype)
    dR_step1, _ = expm1_compose(
        dR_src_local_T,
        jnp.zeros_like(p_rel),
        dR_nom,
        jnp.zeros_like(p_rel),
    )
    dR_rel, _ = expm1_compose(
        dR_step1,
        jnp.zeros_like(p_rel),
        dR_tgt_local,
        jnp.zeros_like(p_rel),
    )
    return dR_rel, p_rel


def pin_hole_tilt_jam_margin(
    r_hole: Array,
    r_pin: Array,
    dR_rel: Array,
    engagement_length: Array,
) -> Array:
    """Compute the engagement-depth tilt binding margin M_fit = C_r,eff.

    A pin tilted by dtheta_perp = [dR_13, -dR_23]^T relative to its hole sweeps a lateral
    span of (L_e / 2) * ||dtheta_perp||_2 about the engagement midpoint. Some float position
    p_xy,mid fits at both ends (+/- L_e/2) if and only if that span is within the radial
    clearance C_r = r_hole - r_pin:
      M_fit = C_r - (L_e / 2) * ||dtheta_perp||_2

    The sampled float eta_xy is deliberately excluded: it is drawn over the full clearance
    disk independently of tilt, so including it would flag assembleable builds as jammed.
    """
    c_r = r_hole - r_pin
    tilt_vec = jnp.stack([dR_rel[..., 0, 2], -dR_rel[..., 1, 2]], axis=-1)
    return c_r - 0.5 * engagement_length * jnp.linalg.norm(tilt_vec, axis=-1)


def residual_planar_coincident(dR_rel: Array, p_rel: Array) -> Array:
    """3-DOF primary datum plane residual: [-dR_23, dR_13, p_z] = 0."""
    return jnp.stack([-dR_rel[..., 1, 2], dR_rel[..., 0, 2], p_rel[..., 2]], axis=-1)


def residual_parallel_pin_hole(
    p_rel: Array,
    r_hole: Array,
    r_pin: Array,
    eta_xy: Array,
) -> Array:
    """2-DOF primary locating pin-in-hole residual at the engagement plane:
    [p_x - C_r * eta_x, p_y - C_r * eta_y] = 0.
    """
    c_r = r_hole - r_pin
    u_xy = c_r[..., None] * eta_xy
    return p_rel[..., :2] - u_xy


def residual_clocking_pin_slot(
    p_rel: Array,
    r_slot_halfwidth: Array,
    r_pin: Array,
    eta_y: Array,
) -> Array:
    """1-DOF clocking pin-in-slot residual along local Y normal to slot walls:
    [p_y - C_w * eta_y] = 0.
    """
    c_w = r_slot_halfwidth - r_pin
    return (p_rel[..., 1] - c_w * eta_y)[..., None]


def residual_line_contact(dR_rel: Array, p_rel: Array) -> Array:
    """2-DOF secondary datum edge contact residual: [-dR_12, p_y] = 0."""
    return jnp.stack([-dR_rel[..., 0, 1], p_rel[..., 1]], axis=-1)


def residual_point_contact(p_rel: Array) -> Array:
    """1-DOF tertiary datum stop residual: [p_x] = 0."""
    return p_rel[..., 0:1]


def residual_coaxial_rigid(dR_rel: Array, p_rel: Array) -> Array:
    """4-DOF press-fit / centered coaxial residual: [-dR_23, dR_13, p_x, p_y] = 0."""
    return jnp.stack(
        [-dR_rel[..., 1, 2], dR_rel[..., 0, 2], p_rel[..., 0], p_rel[..., 1]],
        axis=-1,
    )


def residual_cylindrical_clearance(
    dR_rel: Array,
    p_rel: Array,
    r_hole: Array,
    r_pin: Array,
    eta_4: Array,
    engagement_length: Array,
) -> Array:
    """4-DOF standalone floating pin-in-hole residual:
    [-dR_23 - (2*C_r/L_e)*eta_tx, dR_13 - (2*C_r/L_e)*eta_ty, p_x - C_r*eta_x, p_y - C_r*eta_y] = 0.
    """
    c_r = r_hole - r_pin
    tilt_scale = 2.0 * c_r / engagement_length
    r_tx = -dR_rel[..., 1, 2] - tilt_scale * eta_4[..., 0]
    r_ty = dR_rel[..., 0, 2] - tilt_scale * eta_4[..., 1]
    r_px = p_rel[..., 0] - c_r * eta_4[..., 2]
    r_py = p_rel[..., 1] - c_r * eta_4[..., 3]
    return jnp.stack([r_tx, r_ty, r_px, r_py], axis=-1)


def residual_spherical_joint(p_rel: Array) -> Array:
    """3-DOF spherical ball-in-socket residual: [p_x, p_y, p_z] = 0."""
    return p_rel


def residual_rigid_weld(dR_rel: Array, p_rel: Array) -> Array:
    """6-DOF full rigid weld residual: [-dR_23, dR_13, -dR_12, p_x, p_y, p_z] = 0."""
    return jnp.stack(
        [
            -dR_rel[..., 1, 2],
            dR_rel[..., 0, 2],
            -dR_rel[..., 0, 1],
            p_rel[..., 0],
            p_rel[..., 1],
            p_rel[..., 2],
        ],
        axis=-1,
    )


def eval_measurement_deviation(
    meas_type: str,
    axis: Array,
    src_frame: FeatureFrame,
    q_src: Array,
    xi_src: Array,
    tgt_frame: FeatureFrame,
    q_tgt: Array,
    xi_tgt: Array,
    d_nom_precomputed: Array,
    eps_reg: float = 1e-12,
) -> tuple[Array, Array]:
    """Evaluate (y_nom, delta_y) for any of the 4 functional measurement types without cancellation."""
    a_unit = axis / jnp.linalg.norm(axis)
    R_src_nom, _, dR_src_local, dp_src_global = eval_mate_frame_expm1(
        src_frame, q_src, xi_src
    )
    _, _, _, dp_tgt_global = eval_mate_frame_expm1(tgt_frame, q_tgt, xi_tgt)
    dd_global = dp_tgt_global - dp_src_global

    if meas_type == "distance_xyz":
        y_nom = jnp.dot(a_unit, d_nom_precomputed)
        dy = jnp.dot(a_unit, dd_global)
        return y_nom, dy

    if meas_type == "point_to_plane_distance":
        n_nom = jnp.matmul(R_src_nom, a_unit[..., None])[..., 0]
        dn = jnp.matmul(R_src_nom, jnp.matmul(dR_src_local, a_unit[..., None]))[..., 0]
        y_nom = jnp.dot(n_nom, d_nom_precomputed)
        dy = jnp.dot(n_nom, dd_global) + jnp.dot(dn, d_nom_precomputed + dd_global)
        return y_nom, dy

    if meas_type == "coaxiality":
        e_z = jnp.array([0.0, 0.0, 1.0], dtype=q_src.dtype)
        n_src = jnp.matmul(
            R_src_nom,
            (e_z + jnp.matmul(dR_src_local, e_z[..., None])[..., 0])[..., None],
        )[..., 0]
        d_total = d_nom_precomputed + dd_global
        d_perp = d_total - jnp.dot(d_total, n_src) * n_src
        val = jnp.sqrt(jnp.sum(d_perp * d_perp) + eps_reg * eps_reg) - eps_reg
        return jnp.zeros_like(val), val

    if meas_type == "angular_tilt":
        dR_rel, _ = relative_mate_transform_expm1(
            src_frame, q_src, xi_src, tgt_frame, q_tgt, xi_tgt, d_nom_precomputed
        )
        theta_rel = jnp.stack(
            [-dR_rel[..., 1, 2], dR_rel[..., 0, 2], -dR_rel[..., 0, 1]], axis=-1
        )
        dy = jnp.dot(a_unit, theta_rel)
        return jnp.zeros_like(dy), dy

    raise ValueError(f"Unsupported measurement type '{meas_type}'")
