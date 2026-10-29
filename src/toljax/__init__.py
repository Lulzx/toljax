"""toljax: Differentiable 3D GD&T variation analysis and inverse cost-tolerance synthesis."""

import jax

# Enforce float64 precision by default for exact analytical Jacobians (S) and Hessians (H).
jax.config.update("jax_enable_x64", True)

from toljax.core.graph import CompiledAssembly, compile_assembly
from toljax.core.metrics import compute_sota_moments, run_variation_analysis
from toljax.core.optimizer import optimize_tolerances
from toljax.schema.assembly_spec import AssemblySpec, load_assembly_spec, save_assembly_spec

__version__ = "0.5.0"

load = load_assembly_spec
analyze = run_variation_analysis
optimize = optimize_tolerances

__all__ = [
    "AssemblySpec",
    "CompiledAssembly",
    "__version__",
    "analyze",
    "compile_assembly",
    "compute_sota_moments",
    "load",
    "load_assembly_spec",
    "optimize",
    "optimize_tolerances",
    "run_variation_analysis",
    "save_assembly_spec",
]
