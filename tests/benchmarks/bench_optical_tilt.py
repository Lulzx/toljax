"""Gate 3: Second-Order Geometric Shift (Cosine Foreshortening) & Uniform Disk Cumulants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toljax.core.metrics import compute_sota_moments
from toljax.core.se3 import se3_exp


def test_gate3a_optical_tilt_cosine_foreshortening():
    """Gate 3A: Rigid rod of length L = 100 mm subject to Gaussian angular tilt
    theta_x, theta_y ~ N(0, sigma_theta^2).

    Linear RSS predicts mean = 100.0 and sigma = 0.0 (complete failure).
    Second-order SOTA must predict:
      Delta_mu_z = -L * sigma_theta^2
      sigma_z    =  L * sigma_theta^2
      Pareto     =  50.0% theta_x, 50.0% theta_y
    matching 1,000,000-sample Monte Carlo within 0.2%.
    """
    L = 100.0
    sigma_theta = 0.01  # 10 mrad

    def rod_tip_z(theta_xy: jax.Array) -> jax.Array:
        xi = jnp.array([theta_xy[0], theta_xy[1], 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        T = se3_exp(xi)
        tip = T @ jnp.array([0.0, 0.0, L, 1.0], dtype=jnp.float64)
        return tip[2]

    theta0 = jnp.zeros(2, dtype=jnp.float64)
    s = np.asarray(jax.grad(rod_tip_z)(theta0), dtype=np.float64)
    H = np.asarray(jax.hessian(rod_tip_z)(theta0), dtype=jnp.float64)
    sigmas = np.array([sigma_theta, sigma_theta], dtype=np.float64)

    # First-order sensitivity is identically zero
    np.testing.assert_allclose(s, [0.0, 0.0], atol=1e-15)
    # Exact Hessian at zero is diag(-L, -L) = diag(-100, -100)
    np.testing.assert_allclose(H, np.diag([-L, -L]), rtol=1e-14, atol=1e-14)

    sota = compute_sota_moments(y_nom=L, s=s, H=H, sigmas=sigmas)
    expected_shift = -L * (sigma_theta**2)  # -0.01 mm
    expected_sigma = L * (sigma_theta**2)   #  0.01 mm

    np.testing.assert_allclose(sota["mean_shift"], expected_shift, rtol=1e-14)
    np.testing.assert_allclose(sota["sigma"], expected_sigma, rtol=1e-14)

    # Verify 50% / 50% second-order Pareto variance contribution
    var_z = sigmas**2
    v_contrib = (s * sigmas) ** 2 + 0.5 * np.sum((H**2) * np.outer(var_z, var_z), axis=1)
    pct = v_contrib / np.sum(v_contrib) * 100.0
    np.testing.assert_allclose(pct, [50.0, 50.0], rtol=1e-14)

    # Compare against 1,000,000-sample Monte Carlo
    rng = np.random.default_rng(2026)
    thetas = rng.normal(0.0, sigma_theta, size=(1_000_000, 2))
    theta_norm = np.hypot(thetas[:, 0], thetas[:, 1])
    z_mc = L * np.cos(theta_norm)

    mc_mean_shift = float(np.mean(z_mc) - L)
    mc_sigma = float(np.std(z_mc))

    assert abs(sota["mean_shift"] - mc_mean_shift) / abs(mc_mean_shift) < 0.002
    assert abs(sota["sigma"] - mc_sigma) / mc_sigma < 0.002


def test_gate3b_uniform_disk_fourth_cumulant_variance_and_skewness():
    """Gate 3B: Verify fourth-cumulant variance and O(sigma^4) skewness on a quadratic
    function with off-diagonal coupling H_12 != 0 driven by uniform disk float eta ~ U(B_2(1)).
    """
    s = np.array([0.8, -0.5], dtype=np.float64)
    H = np.array([[0.40, 0.25], [0.25, -0.30]], dtype=np.float64)
    sigmas = np.array([0.5, 0.5], dtype=np.float64)  # sigma^2 = 1/4 per component

    sota = compute_sota_moments(
        y_nom=10.0,
        s=s,
        H=H,
        sigmas=sigmas,
        uniform_disk_pairs=[(0, 1)],
    )

    # 1,000,000-sample uniform unit disk Monte Carlo
    rng = np.random.default_rng(999)
    n_mc = 1_000_000
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n_mc))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_mc)
    eta = np.column_stack([r * np.cos(phi), r * np.sin(phi)])

    y_mc = 10.0 + eta @ s + 0.5 * np.einsum("bi,ij,bj->b", eta, H, eta)
    mc_mean = float(np.mean(y_mc))
    mc_std = float(np.std(y_mc))
    mc_skew = float(np.mean(((y_mc - mc_mean) / mc_std) ** 3))

    assert abs(sota["shifted_mean"] - mc_mean) < 1e-3
    assert abs(sota["sigma"] - mc_std) / mc_std < 0.005
    assert abs(sota["skewness"] - mc_skew) < 0.015
