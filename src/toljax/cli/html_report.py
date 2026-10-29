"""Self-contained HTML report builder for 4-tier tolerance analysis and Pareto breakdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_html_report(results: dict[str, Any], out_path: str | Path) -> Path:
    """Write a self-contained interactive HTML report with 4-tier tables and Pareto bars."""
    asm_name = results.get("assembly_name", "Assembly")
    kin = results.get("kinematic_status", {})
    timing = results.get("execution_time_ms", {})

    rows_html = []
    for m in results.get("measurements", []):
        wc = m["worst_case"]
        rss = m["linear_rss"]
        sota = m["second_order_sota"]
        mc = m["monte_carlo"]
        pareto_items = "".join(
            f"<li><code>{p['feature_dof']}</code>: {p['variance_pct']}% (S = {p['sensitivity_dy_dz']})</li>"
            for p in m.get("pareto_contributions", [])
        )
        rows_html.append(
            f"""
            <div class="card">
              <h2>Measurement: <code>{m['id']}</code></h2>
              <p>Nominal: <b>{m['nominal']}</b> | LSL: <b>{m['lsl']}</b> | USL: <b>{m['usl']}</b></p>
              <table>
                <thead>
                  <tr><th>Tier</th><th>Mean</th><th>Sigma</th><th>Cpk / Status</th><th>Details</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td>1. Worst-Case (3-Cone SOCP)</td>
                    <td>{m['nominal']}</td>
                    <td>—</td>
                    <td><b>{wc['status']}</b></td>
                    <td>[{wc['min']}, {wc['max']}] (±{wc['half_band']})</td>
                  </tr>
                  <tr>
                    <td>2. First-Order Linear RSS</td>
                    <td>{rss['mean']}</td>
                    <td>{rss['sigma']}</td>
                    <td>{rss['cpk']}</td>
                    <td>Cp = {rss['cp']}, 3σ = ±{rss['three_sigma']}</td>
                  </tr>
                  <tr>
                    <td>3. Second-Order SOTA</td>
                    <td>{sota['shifted_mean']}</td>
                    <td>{sota['sigma']}</td>
                    <td>{sota['cpk']}</td>
                    <td>Δμ = {sota['mean_shift']}, skew = {sota['skewness']}</td>
                  </tr>
                  <tr>
                    <td>4. Monte Carlo ({mc['samples_total']:,})</td>
                    <td>{mc['mean']}</td>
                    <td>{mc['sigma']}</td>
                    <td>{mc['yield_pct']}% yield</td>
                    <td>[{mc['p00135']}, {mc['p99865']}], DPMO = {mc['total_defect_dpmo']}</td>
                  </tr>
                </tbody>
              </table>
              <h3>Pareto Variance Contributions</h3>
              <ul>{pareto_items}</ul>
            </div>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>toljax Report — {asm_name}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto; max-width: 1000px; color: #1a202c; }}
    .card {{ border: 1px solid #cbd5e0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.75rem; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f7fafc; }}
    code {{ background: #edf2f7; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1><code>toljax</code> Variation Analysis — {asm_name}</h1>
  <p>Determinate: <b>{kin.get('determinate')}</b> | Rank: <b>{kin.get('jacobian_rank')}/{kin.get('active_dofs')}</b> |
     Jam DPMO: <b>{kin.get('jam_dpmo')}</b> | S+H Time: <b>{timing.get('jacobian_hessian_fp64_ms')} ms</b> |
     MC Time: <b>{timing.get('monte_carlo_fp32_ms')} ms</b></p>
  {''.join(rows_html)}
  <script id="toljax-json" type="application/json">{json.dumps(results)}</script>
</body>
</html>
"""
    path = Path(out_path)
    path.write_text(html, encoding="utf-8")
    return path
