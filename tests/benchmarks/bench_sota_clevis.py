"""Gate 5: Published SOTA / DLM nonlinear stackup benchmark (Glancy & Chase 1999;
Gao, Chase & Magleby 1998) and full end-to-end YAML -> Analysis -> Optimization -> CLI test."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from typer.testing import CliRunner

import toljax
from toljax.cad.glb_exporter import export_assembly_glb
from toljax.cli.app import app
from toljax.core.metrics import compute_sota_moments


def test_gate5_glancy_chase_nonlinear_cylinder_lever_stackup():
    """Reproduce the classic Glancy & Chase (1999) / Gao et al. (1998) nonlinear 2-link / cylinder-in-groove
    stackup where gap y = sqrt((R + r)^2 - (x_1 - x_2)^2) - (R + r) + z_1 exhibits a pure
    second-order downward mean shift and negative skewness.
    """
    R_nom = 25.0
    r_nom = 15.0
    D = R_nom + r_nom  # 40.0 mm contact center distance

    # Inputs x = [x1, x2, dR, dr, z1] with standard deviations from Glancy & Chase (1999):
    sigmas = np.array([0.50, 0.50, 0.05, 0.05, 0.05], dtype=np.float64)

    def gap_fn(x: jax.Array) -> jax.Array:
        x1, x2, dR, dr, z1 = x
        rad_sum = (R_nom + dR) + (r_nom + dr)
        dx = x1 - x2
        return jnp.sqrt(rad_sum * rad_sum - dx * dx) + z1

    x0 = jnp.zeros(5, dtype=jnp.float64)
    y0 = float(gap_fn(x0))
    s = np.asarray(jax.grad(gap_fn)(x0), dtype=np.float64)
    H = np.asarray(jax.hessian(gap_fn)(x0), dtype=np.float64)

    sota = compute_sota_moments(y_nom=y0, s=s, H=H, sigmas=sigmas)

    # Analytical verification:
    # dy/dx1 = 0, d^2y/dx1^2 = -1/D = -1/40, d^2y/dx2^2 = -1/40
    # => exact mean shift = 0.5 * (-1/40) * (0.5^2 + 0.5^2) = -0.00625 mm
    expected_shift = -0.5 * (sigmas[0] ** 2 + sigmas[1] ** 2) / D
    np.testing.assert_allclose(sota["mean_shift"], expected_shift, rtol=1e-12)

    # Compare against 1,000,000-sample Monte Carlo (with antithetic pairing for zero linear MC noise)
    rng = np.random.default_rng(1999)
    X_half = rng.normal(0.0, 1.0, size=(500_000, 5)) * sigmas[None, :]
    X_mc = np.concatenate([X_half, -X_half], axis=0)
    rad_mc = (R_nom + X_mc[:, 2]) + (r_nom + X_mc[:, 3])
    dx_mc = X_mc[:, 0] - X_mc[:, 1]
    y_mc = np.sqrt(rad_mc**2 - dx_mc**2) + X_mc[:, 4]

    mc_mean = float(np.mean(y_mc))
    mc_std = float(np.std(y_mc))
    assert abs(sota["shifted_mean"] - mc_mean) / abs(sota["mean_shift"]) < 0.005
    assert abs(sota["sigma"] - mc_std) / mc_std < 0.005


def test_end_to_end_ev_busbar_yaml_analyze_optimize_and_cli(tmp_path: Path):
    """Verify full end-to-end pipeline on examples/ev_busbar.yaml:
    load -> compile -> analyze -> optimize -> CLI run/optimize/inspect -> HTML & GLB export.
    """
    yaml_path = Path(__file__).resolve().parent.parent.parent / "examples" / "ev_busbar.yaml"
    spec = toljax.load(yaml_path)
    compiled = toljax.compile_assembly(spec)

    results = toljax.analyze(compiled, mc_samples=20_000, mc_mode="full", seed=42)
    assert results["kinematic_status"]["determinate"] is True
    assert results["kinematic_status"]["jacobian_rank"] == 6

    m0 = results["measurements"][0]
    assert m0["id"] == "laser_focal_z"
    assert abs(m0["nominal"] - 25.0) < 1e-6
    assert abs(m0["second_order_sota"]["sigma"] - m0["monte_carlo"]["sigma"]) < 0.002

    # Run inverse cost-tolerance optimization for target Cpk = 1.33
    opt_spec, opt_report = toljax.optimize(compiled, target_cpk=1.33)
    assert opt_report["converged"] is True
    assert opt_report["measurements"][0]["cpk"] >= 1.33 - 1e-3

    # Export GLB
    glb_path = tmp_path / "busbar.glb"
    export_assembly_glb(compiled, glb_path, results=results)
    assert glb_path.exists() and glb_path.stat().st_size > 20

    # Test Typer CLI commands (`run`, `optimize`, `inspect`)
    runner = CliRunner()
    json_out = tmp_path / "results.json"
    html_out = tmp_path / "report.html"
    res_run = runner.invoke(
        app,
        [
            "run",
            str(yaml_path),
            "--mc-samples",
            "5000",
            "--json-out",
            str(json_out),
            "--html-out",
            str(html_out),
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    assert json_out.exists()
    assert html_out.exists()

    opt_yaml = tmp_path / "optimized.yaml"
    res_opt = runner.invoke(
        app,
        ["optimize", str(yaml_path), "--target-cpk", "1.33", "--out-yaml", str(opt_yaml)],
    )
    assert res_opt.exit_code == 0, res_opt.output
    assert opt_yaml.exists()

    # Create a minimal synthetic STEP file and test `toljax inspect`
    step_file = tmp_path / "housing.step"
    step_file.write_text(
        """ISO-10303-21;
DATA;
#10 = CARTESIAN_POINT('Origin', (0.0, 0.0, 10.0));
#11 = AXIS2_PLACEMENT_3D('FACE_DATUM_A', #10, #12, #13);
#14 = PLANE('FACE_DATUM_A', #11);
#20 = CARTESIAN_POINT('Pin1', (20.0, 20.0, 16.0));
#21 = AXIS2_PLACEMENT_3D('PIN_1', #20, #12, #13);
#22 = CYLINDRICAL_SURFACE('PIN_1', #21, 3.0);
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    insp_yaml = tmp_path / "inspected.yaml"
    res_insp = runner.invoke(app, ["inspect", str(step_file), "--out-yaml", str(insp_yaml)])
    assert res_insp.exit_code == 0, res_insp.output
    assert insp_yaml.exists()
