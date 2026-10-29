"""toljax: Differentiable 3D GD&T variation analysis and inverse cost-tolerance synthesis."""

import os

import jax

# Enforce float64 precision by default for exact analytical Jacobians (S) and Hessians (H).
jax.config.update("jax_enable_x64", True)

# Persist compiled XLA programs across processes so repeat CLI runs skip recompilation.
# Respects a cache dir the caller already configured; TOLJAX_NO_CACHE=1 disables it.
if os.environ.get("TOLJAX_NO_CACHE") != "1" and not jax.config.jax_compilation_cache_dir:
    _cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("TOLJAX_CACHE_DIR") or os.path.join(_cache_root, "toljax", "jax"),
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

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
