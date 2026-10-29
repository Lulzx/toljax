"""Inverse optimization must respect the jam-rate constraint, not only the Cpk target."""

from __future__ import annotations

from pathlib import Path

import toljax

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ev_busbar.yaml"


def test_optimizer_keeps_jam_rate_below_limit():
    compiled = toljax.compile_assembly(toljax.load(EXAMPLE))
    _, report = toljax.optimize(
        compiled, target_cpk=1.33, max_jam_dpmo=1350.0, verify_samples=20_000
    )

    assert report["converged"]
    assert report["cost_after"] < report["cost_before"]
    assert report["measurements"][0]["cpk"] >= 1.33 - 1e-3
    # Verified on fresh builds the optimizer never saw
    assert report["fit"]["jam_dpmo_after"] <= 1350.0


def test_original_busbar_design_assembles():
    """The tilt margin is a feasibility test, so the example's tolerances do not jam."""
    compiled = toljax.compile_assembly(toljax.load(EXAMPLE))
    results = toljax.analyze(compiled, mc_samples=20_000)
    assert results["kinematic_status"]["jam_dpmo"] < 1000.0
