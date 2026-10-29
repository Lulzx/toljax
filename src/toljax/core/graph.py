"""Static compilation of AssemblySpec into differentiable JAX residual, measurement,
and jam-margin functions with dimensionless SVD determinacy verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from toljax.core.mates import (
    FeatureFrame,
    eval_measurement_deviation,
    pin_hole_tilt_jam_margin,
    relative_mate_transform_expm1,
    residual_clocking_pin_slot,
    residual_coaxial_rigid,
    residual_cylindrical_clearance,
    residual_line_contact,
    residual_parallel_pin_hole,
    residual_planar_coincident,
    residual_point_contact,
    residual_rigid_weld,
    residual_spherical_joint,
)
from toljax.core.se3 import so3_exp
from toljax.core.solver import check_kinematic_determinacy, make_equilibrium_solver
from toljax.core.torsor import (
    map_cylinder_torsor_symmetric,
    map_planar_torsor,
    map_point_torsor,
    map_slot_torsor,
)
from toljax.schema.assembly_spec import AssemblySpec, FeatureSpec, UNIT_TO_MM


def _rpy_deg_to_rot(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """Convert intrinsic XYZ Roll-Pitch-Yaw angles (degrees) to a 3x3 rotation matrix."""
    rx, ry, rz = np.deg2rad(rpy_deg)
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]]
    )
    return Rx @ Ry @ Rz


@dataclass(frozen=True)
class LatentVarMeta:
    index: int
    name: str
    feature_key: str
    dist_kind: str  # "gaussian", "uniform_1d", "uniform_disk"
    sigma: float
    opt_param_idx: int | None = None


@dataclass(frozen=True)
class CylinderWCInfo:
    feature_key: str
    latent_slice: tuple[int, int]  # spatial latent indices (4 or 6)
    size_idx: int
    t_pos: float
    t_orient: float
    t_size: float
    lz: float
    modifier: str


@dataclass
class CompiledAssembly:
    spec: AssemblySpec
    movable_part_ids: list[str]
    n_movable: int
    n_dofs: int
    n_constraints: int
    jacobian_rank: int
    l_char: float
    latent_vars: list[LatentVarMeta]
    box_indices: list[int]
    uniform_1d_indices: list[int]
    gaussian_disk_pairs: list[tuple[int, int]]
    uniform_disk_pairs: list[tuple[int, int]]
    cylinder_wc_infos: list[CylinderWCInfo]
    opt_feature_keys: list[str]
    opt_nominal_tols: np.ndarray
    opt_cost_a: np.ndarray
    opt_cost_b: np.ndarray
    opt_cost_alpha: np.ndarray
    opt_bounds: list[tuple[float, float]]
    meas_ids: list[str]
    meas_nominals: np.ndarray
    meas_lsl: np.ndarray
    meas_usl: np.ndarray
    residual_fn: Callable[[Array, Array], Array]
    solve_equilibrium: Callable[[Array], Array]
    eval_measurements_dy: Callable[[Array], Array]
    eval_measurements_dy_scaled: Callable[[Array, Array], Array]
    eval_jam_margins: Callable[[Array], Array]


def compile_assembly(spec: AssemblySpec) -> CompiledAssembly:
    """Compile a validated AssemblySpec into a differentiable CompiledAssembly."""
    unit_scale = UNIT_TO_MM[spec.assembly.units]
    proc_sigma = float(spec.assembly.default_process_sigma)
    sigma_gauss = 1.0 / proc_sigma
    mmc_mode = spec.assembly.mmc_statistical_mode

    part_map: dict[str, int] = {}
    movable_part_ids: list[str] = []
    movable_idx_map: dict[str, int] = {}
    part_R_nom: dict[str, np.ndarray] = {}
    part_p_nom: dict[str, np.ndarray] = {}

    for idx, p in enumerate(spec.parts):
        part_map[p.id] = idx
        part_R_nom[p.id] = _rpy_deg_to_rot(p.nominal_pose.rpy_deg)
        part_p_nom[p.id] = np.array(p.nominal_pose.pos, dtype=np.float64) * unit_scale
        if not p.fixed:
            movable_idx_map[p.id] = len(movable_part_ids)
            movable_part_ids.append(p.id)

    # Identify primary datum features
    primary_datum_keys: set[str] = set()
    for m in spec.mates:
        if m.type == "planar_coincident":
            primary_datum_keys.add(m.source)
            primary_datum_keys.add(m.target)

    # Map features and allocate latent variables
    feat_specs: dict[str, FeatureSpec] = {}
    feat_part: dict[str, str] = {}
    feat_R_nom: dict[str, np.ndarray] = {}
    feat_p_nom: dict[str, np.ndarray] = {}
    feat_tol_info: dict[str, tuple[float, float, float, float]] = {}  # (t_pos, t_orient, t_size, r_mid)
    feat_latent_slices: dict[str, tuple[int, int]] = {}
    feat_opt_idx: dict[str, int | None] = {}

    latent_vars: list[LatentVarMeta] = []
    box_indices: list[int] = []
    uniform_1d_indices: list[int] = []
    gaussian_disk_pairs: list[tuple[int, int]] = []
    uniform_disk_pairs: list[tuple[int, int]] = []
    cylinder_wc_infos: list[CylinderWCInfo] = []

    opt_feature_keys: list[str] = []
    opt_nominal_tols_list: list[float] = []
    opt_cost_a_list: list[float] = []
    opt_cost_b_list: list[float] = []
    opt_cost_alpha_list: list[float] = []
    opt_bounds_list: list[tuple[float, float]] = []

    all_global_pts: list[np.ndarray] = []

    for p in spec.parts:
        for f in p.features:
            key = f"{p.id}.{f.id}"
            feat_specs[key] = f
            feat_part[key] = p.id
            R_f = _rpy_deg_to_rot(f.nominal_frame.rpy_deg)
            p_f = np.array(f.nominal_frame.pos, dtype=np.float64) * unit_scale
            feat_R_nom[key] = R_f
            feat_p_nom[key] = p_f
            all_global_pts.append(part_p_nom[p.id] + part_R_nom[p.id] @ p_f)

            is_prim = key in primary_datum_keys
            t_pos_raw, t_orient_raw, t_size_raw = f.resolve_tolerances(is_primary_datum=is_prim)
            t_pos = t_pos_raw * unit_scale
            t_orient = t_orient_raw * unit_scale
            t_size = t_size_raw * unit_scale
            r_mid = f.size.r_mid * unit_scale if f.size is not None else 0.0
            feat_tol_info[key] = (t_pos, t_orient, t_size, r_mid)

            opt_idx: int | None = None
            primary_tol = t_pos if t_pos > 0.0 else t_orient
            if f.cost_model is not None and primary_tol > 0.0:
                opt_idx = len(opt_feature_keys)
                opt_feature_keys.append(key)
                opt_nominal_tols_list.append(primary_tol)
                opt_cost_a_list.append(f.cost_model.base_a)
                opt_cost_b_list.append(f.cost_model.scale_b)
                opt_cost_alpha_list.append(f.cost_model.exponent)
                opt_bounds_list.append(
                    (f.cost_model.bounds[0] * unit_scale, f.cost_model.bounds[1] * unit_scale)
                )
            feat_opt_idx[key] = opt_idx

            start_idx = len(latent_vars)
            if f.type == "planar":
                names = ["delta_z", "theta_x", "theta_y"]
                for nm in names:
                    idx_v = len(latent_vars)
                    latent_vars.append(
                        LatentVarMeta(idx_v, f"{key}.{nm}", key, "gaussian", sigma_gauss, opt_idx)
                    )
                box_indices.append(start_idx)
                gaussian_disk_pairs.append((start_idx + 1, start_idx + 2))
            elif f.type in ("cylinder_pin", "cylinder_hole"):
                is_refined = (0.0 < t_orient < t_pos)
                if is_refined:
                    names = [
                        "u1_x",
                        "u1_y",
                        "u2_x",
                        "u2_y",
                        "u3_x",
                        "u3_y",
                        "delta_r",
                    ]
                    for i_n, nm in enumerate(names):
                        idx_v = len(latent_vars)
                        latent_vars.append(
                            LatentVarMeta(
                                idx_v,
                                f"{key}.{nm}",
                                key,
                                "gaussian",
                                sigma_gauss,
                                opt_idx if i_n < 6 else None,
                            )
                        )
                    cylinder_wc_infos.append(
                        CylinderWCInfo(
                            key,
                            (start_idx, start_idx + 6),
                            start_idx + 6,
                            t_pos,
                            t_orient,
                            t_size,
                            f.length * unit_scale,
                            f.gdt.modifier,
                        )
                    )
                    box_indices.append(start_idx + 6)
                else:
                    names = ["delta_x", "delta_y", " bot_x", "bot_y", "delta_r"]
                    clean_names = ["delta_x", "delta_y", "theta_x", "theta_y", "delta_r"]
                    for i_n, nm in enumerate(clean_names):
                        idx_v = len(latent_vars)
                        latent_vars.append(
                            LatentVarMeta(
                                idx_v,
                                f"{key}.{nm}",
                                key,
                                "gaussian",
                                sigma_gauss,
                                opt_idx if i_n < 4 else None,
                            )
                        )
                    gaussian_disk_pairs.append((start_idx, start_idx + 1))
                    gaussian_disk_pairs.append((start_idx + 2, start_idx + 3))
                    box_indices.append(start_idx + 4)
            elif f.type == "slot_width":
                names = ["delta_y", "theta_x", "theta_z", "delta_r"]
                for i_n, nm in enumerate(names):
                    idx_v = len(latent_vars)
                    latent_vars.append(
                        LatentVarMeta(
                            idx_v,
                            f"{key}.{nm}",
                            key,
                            "gaussian",
                            sigma_gauss,
                            opt_idx if i_n < 3 else None,
                        )
                    )
                for k_i in range(4):
                    box_indices.append(start_idx + k_i)
            elif f.type in ("point", "sphere"):
                for nm in ("delta_x", "delta_y", "delta_z"):
                    idx_v = len(latent_vars)
                    latent_vars.append(
                        LatentVarMeta(idx_v, f"{key}.{nm}", key, "gaussian", sigma_gauss, opt_idx)
                    )
                    box_indices.append(idx_v)
            feat_latent_slices[key] = (start_idx, len(latent_vars))

    # Allocate mate clearance float latent variables
    mate_float_slices: dict[str, tuple[int, int]] = {}
    for m in spec.mates:
        s_idx = len(latent_vars)
        if m.type == "parallel_pin_hole" and m.kinematic_mode == "float_random":
            ix = len(latent_vars)
            latent_vars.append(
                LatentVarMeta(ix, f"mate.{m.id}.eta_x", m.id, "uniform_disk", 0.5, None)
            )
            iy = len(latent_vars)
            latent_vars.append(
                LatentVarMeta(iy, f"mate.{m.id}.eta_y", m.id, "uniform_disk", 0.5, None)
            )
            uniform_disk_pairs.append((ix, iy))
        elif m.type == "clocking_pin_slot" and m.kinematic_mode == "float_random":
            iy = len(latent_vars)
            latent_vars.append(
                LatentVarMeta(
                    iy, f"mate.{m.id}.eta_y", m.id, "uniform_1d", float(1.0 / np.sqrt(3.0)), None
                )
            )
            uniform_1d_indices.append(iy)
        elif m.type == "cylindrical_clearance" and m.kinematic_mode == "float_random":
            for nm in ("eta_tx", "eta_ty", "eta_x", "eta_y"):
                iv = len(latent_vars)
                latent_vars.append(
                    LatentVarMeta(iv, f"mate.{m.id}.{nm}", m.id, "uniform_disk", 0.5, None)
                )
            uniform_disk_pairs.append((s_idx, s_idx + 1))
            uniform_disk_pairs.append((s_idx + 2, s_idx + 3))
        mate_float_slices[m.id] = (s_idx, len(latent_vars))

    # Characteristic length L_char from bounding box diagonal
    if len(all_global_pts) >= 2:
        pts_arr = np.stack(all_global_pts, axis=0)
        bbox_diag = float(np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0)))
        l_char = max(bbox_diag, 10.0)
    else:
        l_char = 100.0

    def _make_feature_frame(key: str, z_off: float, dtype) -> FeatureFrame:
        p_id = feat_part[key]
        return FeatureFrame(
            R_part_nom=jnp.asarray(part_R_nom[p_id], dtype=dtype),
            p_part_nom=jnp.asarray(part_p_nom[p_id], dtype=dtype),
            R_feat_nom=jnp.asarray(feat_R_nom[key], dtype=dtype),
            p_feat_nom=jnp.asarray(feat_p_nom[key], dtype=dtype),
            z_contact=jnp.asarray(z_off * unit_scale, dtype=dtype),
        )

    def _eval_all_features(
        z: Array, tol_scale: Array | None = None
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        dtype = z.dtype
        xi_map: dict[str, Array] = {}
        r_map: dict[str, Array] = {}

        for key, f in feat_specs.items():
            s_i, e_i = feat_latent_slices[key]
            z_f = z[s_i:e_i]
            t_pos, t_orient, t_size, r_mid = feat_tol_info[key]
            opt_idx = feat_opt_idx[key]
            scale = (
                tol_scale[opt_idx]
                if (tol_scale is not None and opt_idx is not None)
                else jnp.ones((), dtype=dtype)
            )
            t_pos_s = jnp.asarray(t_pos, dtype=dtype) * scale
            t_orient_s = jnp.asarray(t_orient, dtype=dtype) * scale
            t_size_s = jnp.asarray(t_size, dtype=dtype)
            r_mid_s = jnp.asarray(r_mid, dtype=dtype)

            if f.type == "planar":
                xi = map_planar_torsor(
                    z_f,
                    t_pos_s,
                    t_orient_s,
                    f.bounds.lx * unit_scale,
                    f.bounds.ly * unit_scale,
                )
                r_act = jnp.zeros((), dtype=dtype)
            elif f.type in ("cylinder_pin", "cylinder_hole"):
                xi, r_act = map_cylinder_torsor_symmetric(
                    z_f,
                    r_mid_s,
                    t_size_s,
                    t_pos_s,
                    t_orient_s,
                    f.length * unit_scale,
                    modifier=f.gdt.modifier,
                    is_external=(f.type == "cylinder_pin"),
                    mmc_mode=mmc_mode,
                )
            elif f.type == "slot_width":
                xi, r_act = map_slot_torsor(
                    z_f,
                    r_mid_s,
                    t_size_s,
                    t_pos_s,
                    t_orient_s,
                    f.bounds.lx * unit_scale,
                    f.bounds.lz * unit_scale,
                    modifier=f.gdt.modifier,
                    mmc_mode=mmc_mode,
                )
            else:
                xi = map_point_torsor(z_f, t_pos_s)
                r_act = jnp.zeros((), dtype=dtype)

            xi_map[key] = xi
            r_map[key] = r_act

        return xi_map, r_map

    def _get_part_q(q: Array, p_id: str) -> Array:
        if p_id not in movable_idx_map:
            return jnp.zeros(6, dtype=q.dtype)
        m_idx = movable_idx_map[p_id]
        return q[6 * m_idx : 6 * (m_idx + 1)]

    # Precompute nominal mate difference vectors in float64
    mate_dp_nom: dict[str, np.ndarray] = {}
    for m in spec.mates:
        f_src = _make_feature_frame(m.source, m.contact_offset_z.source, jnp.float64)
        f_tgt = _make_feature_frame(m.target, m.contact_offset_z.target, jnp.float64)
        p_src_g = np.asarray(f_src.p_part_nom) + np.asarray(f_src.R_part_nom) @ (
            np.asarray(f_src.p_feat_nom)
            + np.asarray(f_src.R_feat_nom) @ np.array([0.0, 0.0, float(f_src.z_contact)])
        )
        p_tgt_g = np.asarray(f_tgt.p_part_nom) + np.asarray(f_tgt.R_part_nom) @ (
            np.asarray(f_tgt.p_feat_nom)
            + np.asarray(f_tgt.R_feat_nom) @ np.array([0.0, 0.0, float(f_tgt.z_contact)])
        )
        diff = p_tgt_g - p_src_g
        if np.linalg.norm(diff) < 1e-12:
            diff = np.zeros(3, dtype=np.float64)
        mate_dp_nom[m.id] = diff

    def residual_fn_scaled(
        q: Array, z: Array, tol_scale: Array | None = None
    ) -> Array:
        dtype = q.dtype
        xi_map, r_map = _eval_all_features(z, tol_scale=tol_scale)
        res_list: list[Array] = []

        for m in spec.mates:
            q_src = _get_part_q(q, feat_part[m.source])
            q_tgt = _get_part_q(q, feat_part[m.target])
            f_src = _make_feature_frame(m.source, m.contact_offset_z.source, dtype)
            f_tgt = _make_feature_frame(m.target, m.contact_offset_z.target, dtype)
            dp_nom = jnp.asarray(mate_dp_nom[m.id], dtype=dtype)

            dR_rel, p_rel = relative_mate_transform_expm1(
                f_src,
                q_src,
                xi_map[m.source],
                f_tgt,
                q_tgt,
                xi_map[m.target],
                dp_nom_precomputed=dp_nom,
            )

            s_m, e_m = mate_float_slices[m.id]
            if m.type == "planar_coincident":
                r_m = residual_planar_coincident(dR_rel, p_rel)
            elif m.type == "parallel_pin_hole":
                if m.kinematic_mode == "float_random":
                    eta_xy = z[s_m:e_m]
                elif m.kinematic_mode == "gravity_bias":
                    b = jnp.asarray(m.bias_vector, dtype=dtype)
                    eta_xy = b / jnp.linalg.norm(b)
                else:
                    eta_xy = jnp.zeros(2, dtype=dtype)
                r_src = r_map[m.source]
                r_tgt = r_map[m.target]
                r_hole = jnp.maximum(r_src, r_tgt)
                r_pin = jnp.minimum(r_src, r_tgt)
                r_m = residual_parallel_pin_hole(p_rel, r_hole, r_pin, eta_xy)
            elif m.type == "clocking_pin_slot":
                if m.kinematic_mode == "float_random":
                    eta_y = z[s_m]
                elif m.kinematic_mode == "gravity_bias":
                    eta_y = jnp.asarray(m.bias_vector[1], dtype=dtype)
                else:
                    eta_y = jnp.zeros((), dtype=dtype)
                r_src = r_map[m.source]
                r_tgt = r_map[m.target]
                r_slot = jnp.maximum(r_src, r_tgt)
                r_pin = jnp.minimum(r_src, r_tgt)
                r_m = residual_clocking_pin_slot(p_rel, r_slot, r_pin, eta_y)
            elif m.type == "line_contact":
                r_m = residual_line_contact(dR_rel, p_rel)
            elif m.type == "point_contact":
                r_m = residual_point_contact(p_rel)
            elif m.type == "coaxial_rigid":
                r_m = residual_coaxial_rigid(dR_rel, p_rel)
            elif m.type == "cylindrical_clearance":
                eta_4 = (
                    z[s_m:e_m]
                    if m.kinematic_mode == "float_random"
                    else jnp.zeros(4, dtype=dtype)
                )
                r_src = r_map[m.source]
                r_tgt = r_map[m.target]
                r_hole = jnp.maximum(r_src, r_tgt)
                r_pin = jnp.minimum(r_src, r_tgt)
                l_e = jnp.asarray(
                    min(feat_specs[m.source].length, feat_specs[m.target].length) * unit_scale,
                    dtype=dtype,
                )
                r_m = residual_cylindrical_clearance(
                    dR_rel, p_rel, r_hole, r_pin, eta_4, l_e
                )
            elif m.type == "spherical_joint":
                r_m = residual_spherical_joint(p_rel)
            elif m.type == "rigid_weld":
                r_m = residual_rigid_weld(dR_rel, p_rel)
            else:
                raise ValueError(f"Unsupported mate type '{m.type}'")

            res_list.append(r_m)

        return jnp.concatenate(res_list, axis=-1)

    def residual_fn(q: Array, z: Array) -> Array:
        return residual_fn_scaled(q, z, tol_scale=None)

    n_movable = len(movable_part_ids)
    n_dofs = 6 * n_movable
    n_vars = len(latent_vars)

    q0 = jnp.zeros(n_dofs, dtype=jnp.float64)
    z0 = jnp.zeros(n_vars, dtype=jnp.float64)
    rank, _ = check_kinematic_determinacy(residual_fn, q0, z0, l_char=l_char)
    n_constraints = int(residual_fn(q0, z0).shape[0])

    solve_equilibrium = make_equilibrium_solver(residual_fn, n_dofs=n_dofs, num_iters=5)

    # Precompute nominal measurement vectors in float64
    meas_d_nom: list[np.ndarray] = []
    meas_nominals_list: list[float] = []
    meas_ids: list[str] = []
    meas_lsl_list: list[float] = []
    meas_usl_list: list[float] = []

    for meas in spec.measurements:
        meas_ids.append(meas.id)
        meas_lsl_list.append(meas.limits.lsl * unit_scale)
        meas_usl_list.append(meas.limits.usl * unit_scale)
        f_src = _make_feature_frame(meas.source, 0.0, jnp.float64)
        f_tgt = _make_feature_frame(meas.target, 0.0, jnp.float64)
        p_s = np.asarray(f_src.p_part_nom) + np.asarray(f_src.R_part_nom) @ np.asarray(
            f_src.p_feat_nom
        )
        p_t = np.asarray(f_tgt.p_part_nom) + np.asarray(f_tgt.R_part_nom) @ np.asarray(
            f_tgt.p_feat_nom
        )
        d_nom = p_t - p_s
        meas_d_nom.append(d_nom)
        y_n, _ = eval_measurement_deviation(
            meas.type,
            jnp.asarray(meas.axis, dtype=jnp.float64),
            f_src,
            jnp.zeros(6, dtype=jnp.float64),
            jnp.zeros(6, dtype=jnp.float64),
            f_tgt,
            jnp.zeros(6, dtype=jnp.float64),
            jnp.zeros(6, dtype=jnp.float64),
            jnp.asarray(d_nom, dtype=jnp.float64),
        )
        meas_nominals_list.append(float(y_n))

    def eval_measurements_dy_scaled(z: Array, tol_scale: Array) -> Array:
        dtype = z.dtype

        def res_scaled(q_v: Array, z_v: Array) -> Array:
            return residual_fn_scaled(q_v, z_v, tol_scale=tol_scale)

        solver_scaled = make_equilibrium_solver(res_scaled, n_dofs=n_dofs, num_iters=5)
        q_star = solver_scaled(z)
        xi_map, _ = _eval_all_features(z, tol_scale=tol_scale)
        dys: list[Array] = []
        for idx_m, meas in enumerate(spec.measurements):
            q_src = _get_part_q(q_star, feat_part[meas.source])
            q_tgt = _get_part_q(q_star, feat_part[meas.target])
            f_src = _make_feature_frame(meas.source, 0.0, dtype)
            f_tgt = _make_feature_frame(meas.target, 0.0, dtype)
            _, dy = eval_measurement_deviation(
                meas.type,
                jnp.asarray(meas.axis, dtype=dtype),
                f_src,
                q_src,
                xi_map[meas.source],
                f_tgt,
                q_tgt,
                xi_map[meas.target],
                jnp.asarray(meas_d_nom[idx_m], dtype=dtype),
            )
            dys.append(dy)
        return jnp.stack(dys, axis=-1)

    def eval_measurements_dy(z: Array) -> Array:
        dtype = z.dtype
        q_star = solve_equilibrium(z)
        xi_map, _ = _eval_all_features(z, tol_scale=None)
        dys: list[Array] = []
        for idx_m, meas in enumerate(spec.measurements):
            q_src = _get_part_q(q_star, feat_part[meas.source])
            q_tgt = _get_part_q(q_star, feat_part[meas.target])
            f_src = _make_feature_frame(meas.source, 0.0, dtype)
            f_tgt = _make_feature_frame(meas.target, 0.0, dtype)
            _, dy = eval_measurement_deviation(
                meas.type,
                jnp.asarray(meas.axis, dtype=dtype),
                f_src,
                q_src,
                xi_map[meas.source],
                f_tgt,
                q_tgt,
                xi_map[meas.target],
                jnp.asarray(meas_d_nom[idx_m], dtype=dtype),
            )
            dys.append(dy)
        return jnp.stack(dys, axis=-1)

    def eval_jam_margins(z: Array) -> Array:
        """Evaluate scalar fit margin M_fit(z) across primary pin tilt-binding and jam_checks."""
        dtype = z.dtype
        q_star = solve_equilibrium(z)
        xi_map, r_map = _eval_all_features(z, tol_scale=None)
        margins: list[Array] = []

        for m in spec.mates:
            if m.type in ("parallel_pin_hole", "cylindrical_clearance"):
                q_src = _get_part_q(q_star, feat_part[m.source])
                q_tgt = _get_part_q(q_star, feat_part[m.target])
                f_src = _make_feature_frame(m.source, m.contact_offset_z.source, dtype)
                f_tgt = _make_feature_frame(m.target, m.contact_offset_z.target, dtype)
                dR_rel, p_rel = relative_mate_transform_expm1(
                    f_src,
                    q_src,
                    xi_map[m.source],
                    f_tgt,
                    q_tgt,
                    xi_map[m.target],
                    dp_nom_precomputed=jnp.asarray(mate_dp_nom[m.id], dtype=dtype),
                )
                r_src = r_map[m.source]
                r_tgt = r_map[m.target]
                r_hole = jnp.maximum(r_src, r_tgt)
                r_pin = jnp.minimum(r_src, r_tgt)
                l_e = jnp.asarray(
                    min(feat_specs[m.source].length, feat_specs[m.target].length) * unit_scale,
                    dtype=dtype,
                )
                margins.append(
                    pin_hole_tilt_jam_margin(r_hole, r_pin, dR_rel, p_rel, l_e)
                )

        for jc in spec.jam_checks:
            q_src = _get_part_q(q_star, feat_part[jc.source])
            q_tgt = _get_part_q(q_star, feat_part[jc.target])
            f_src = _make_feature_frame(jc.source, 0.0, dtype)
            f_tgt = _make_feature_frame(jc.target, 0.0, dtype)
            _, p_rel = relative_mate_transform_expm1(
                f_src,
                q_src,
                xi_map[jc.source],
                f_tgt,
                q_tgt,
                xi_map[jc.target],
            )
            r_src = r_map[jc.source]
            r_tgt = r_map[jc.target]
            c_avail = jnp.abs(r_tgt - r_src)
            if jc.axis == "y":
                misalign = jnp.abs(p_rel[1])
            elif jc.axis == "x":
                misalign = jnp.abs(p_rel[0])
            else:
                misalign = jnp.linalg.norm(p_rel[:2])
            margins.append(c_avail - misalign)

        if not margins:
            return jnp.ones((), dtype=dtype)
        return jnp.min(jnp.stack(margins, axis=-1))

    return CompiledAssembly(
        spec=spec,
        movable_part_ids=movable_part_ids,
        n_movable=n_movable,
        n_dofs=n_dofs,
        n_constraints=n_constraints,
        jacobian_rank=rank,
        l_char=l_char,
        latent_vars=latent_vars,
        box_indices=box_indices,
        uniform_1d_indices=uniform_1d_indices,
        gaussian_disk_pairs=gaussian_disk_pairs,
        uniform_disk_pairs=uniform_disk_pairs,
        cylinder_wc_infos=cylinder_wc_infos,
        opt_feature_keys=opt_feature_keys,
        opt_nominal_tols=np.array(opt_nominal_tols_list, dtype=np.float64),
        opt_cost_a=np.array(opt_cost_a_list, dtype=np.float64),
        opt_cost_b=np.array(opt_cost_b_list, dtype=np.float64),
        opt_cost_alpha=np.array(opt_cost_alpha_list, dtype=np.float64),
        opt_bounds=opt_bounds_list,
        meas_ids=meas_ids,
        meas_nominals=np.array(meas_nominals_list, dtype=np.float64),
        meas_lsl=np.array(meas_lsl_list, dtype=np.float64),
        meas_usl=np.array(meas_usl_list, dtype=np.float64),
        residual_fn=residual_fn,
        solve_equilibrium=solve_equilibrium,
        eval_measurements_dy=eval_measurements_dy,
        eval_measurements_dy_scaled=eval_measurements_dy_scaled,
        eval_jam_margins=eval_jam_margins,
    )
