"""Typer CLI entrypoints (`toljax run`, `toljax optimize`, `toljax inspect`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from toljax.cad.step_inspector import export_step_boilerplate_yaml, inspect_step_file
from toljax.cli.html_report import generate_html_report
from toljax.core.graph import compile_assembly
from toljax.core.metrics import run_variation_analysis
from toljax.core.optimizer import optimize_tolerances
from toljax.schema.assembly_spec import load_assembly_spec, save_assembly_spec

app = typer.Typer(
    name="toljax",
    help="Headless, differentiable 3D GD&T variation analysis and inverse cost-tolerance synthesis.",
    add_completion=False,
)


@app.command("run")
def cli_run(
    assembly_yaml: Path = typer.Argument(..., help="Path to assembly.yaml"),
    mc_samples: int = typer.Option(100_000, "--mc-samples", help="Number of Monte Carlo samples"),
    mc_mode: str = typer.Option("full", "--mc-mode", help="Monte Carlo mode: full or surrogate"),
    seed: int = typer.Option(42, "--seed", help="PRNG seed for Monte Carlo"),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Output path for results.json"),
    html_out: Optional[Path] = typer.Option(None, "--html-out", help="Output path for report.html"),
) -> None:
    """Execute full 4-tier tolerance analysis (WC, Linear RSS, 2nd-Order SOTA, Monte Carlo)."""
    spec = load_assembly_spec(assembly_yaml)
    compiled = compile_assembly(spec)
    results = run_variation_analysis(
        compiled,
        mc_samples=mc_samples,
        mc_mode="surrogate" if mc_mode == "surrogate" else "full",
        seed=seed,
    )
    formatted = json.dumps(results, indent=2)
    if json_out is not None:
        json_out.write_text(formatted, encoding="utf-8")
    if html_out is not None:
        generate_html_report(results, html_out)
    typer.echo(formatted)


@app.command("optimize")
def cli_optimize(
    assembly_yaml: Path = typer.Argument(..., help="Path to assembly.yaml"),
    target_cpk: float = typer.Option(1.33, "--target-cpk", help="Target minimum Cpk"),
    max_jam_dpmo: float = typer.Option(
        1350.0, "--max-jam-dpmo", help="Maximum allowed jam rate (defects per million builds)"
    ),
    no_fit_check: bool = typer.Option(
        False, "--no-fit-check", help="Optimize Cpk only, without the jam-rate constraint"
    ),
    out_yaml: Optional[Path] = typer.Option(
        None, "--out-yaml", help="Output path for optimized assembly YAML"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Output path for optimization summary JSON"
    ),
) -> None:
    """Run inverse cost-tolerance optimization."""
    spec = load_assembly_spec(assembly_yaml)
    compiled = compile_assembly(spec)
    new_spec, report = optimize_tolerances(
        compiled,
        target_cpk=target_cpk,
        max_jam_dpmo=None if no_fit_check else max_jam_dpmo,
    )
    if out_yaml is not None:
        save_assembly_spec(new_spec, out_yaml)
    formatted = json.dumps(report, indent=2)
    if json_out is not None:
        json_out.write_text(formatted, encoding="utf-8")
    typer.echo(formatted)


@app.command("inspect")
def cli_inspect(
    step_file: Path = typer.Argument(..., help="Path to part STEP file"),
    out_yaml: Optional[Path] = typer.Option(
        None, "--out-yaml", help="Output path for extracted feature boilerplate YAML"
    ),
) -> None:
    """Extract nominal feature frames and face tags from a STEP file into YAML boilerplate."""
    if out_yaml is not None:
        info = export_step_boilerplate_yaml(step_file, out_yaml)
    else:
        info = inspect_step_file(step_file)
    typer.echo(json.dumps(info, indent=2))
