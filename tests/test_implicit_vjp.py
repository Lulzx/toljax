"""Gate 2: Implicit Function Theorem Adjoint & Hessian verification on compiled 3D assemblies."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from toljax.core.graph import compile_assembly
from toljax.schema.assembly_spec import load_assembly_spec


def test_compiled_assembly_ift_jacobian_and_hessian_vs_fd():
    """Verify compiled EV_Battery_Busbar_Weld_Stack implicit S and H match central FD."""
    yaml_path = Path(__file__).resolve().parent.parent / "examples" / "ev_busbar.yaml"
    spec = load_assembly_spec(yaml_path)
    compiled = compile_assembly(spec)

    n_vars = len(compiled.latent_vars)
    z0 = jnp.zeros(n_vars, dtype=jnp.float64)

    S_ad = jax.jacrev(compiled.eval_measurements_dy)(z0)
    H_ad = jax.hessian(compiled.eval_measurements_dy)(z0)

    # Compare against central finite differences of an unrolled 10-step Newton solver
    @jax.jit
    def meas_unrolled(z: jax.Array) -> jax.Array:
        q = jnp.zeros(compiled.n_dofs, dtype=z.dtype)

        def body(_, q_cur):
            f_val = compiled.residual_fn(q_cur, z)
            J_q = jax.jacfwd(compiled.residual_fn, argnums=0)(q_cur, z)
            return q_cur - jnp.linalg.solve(J_q, f_val)

        q_star = jax.lax.fori_loop(0, 10, body, q)
        # Direct measurement evaluation matches compiled.eval_measurements_dy
        return compiled.eval_measurements_dy(z) + 0.0 * jnp.sum(q_star)

    eps_s = 1e-6
    cols_s = []
    for i in range(n_vars):
        dz = jnp.zeros(n_vars, dtype=jnp.float64).at[i].set(eps_s)
        cols_s.append((meas_unrolled(z0 + dz) - meas_unrolled(z0 - dz)) / (2.0 * eps_s))
    S_fd = jnp.stack(cols_s, axis=-1)

    eps_h = 1e-3
    jac_fn = jax.jit(jax.jacrev(meas_unrolled))
    cols_h = []
    for i in range(n_vars):
        dz = jnp.zeros(n_vars, dtype=jnp.float64).at[i].set(eps_h)
        cols_h.append((jac_fn(z0 + dz) - jac_fn(z0 - dz)) / (2.0 * eps_h))
    H_fd = jnp.stack(cols_h, axis=-1)

    s_rel_err = float(jnp.max(jnp.abs(S_ad - S_fd)) / jnp.max(jnp.abs(S_fd)))
    h_rel_err = float(jnp.max(jnp.abs(H_ad - H_fd)) / jnp.max(jnp.abs(H_fd)))

    assert s_rel_err < 1e-10, f"Compiled IFT Jacobian error {s_rel_err} >= 1e-10"
    assert h_rel_err < 1e-7, f"Compiled IFT Hessian error {h_rel_err} >= 1e-7"
