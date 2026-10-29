# `toljax` — Technical Specification (`spec.md`)

* **Version:** 0.5.0
* **License:** Apache-2.0 / MIT
* **Target Runtime:** Python $\ge 3.11$, JAX $\ge 0.4.30$
* **Precision Model:** Dual-Precision (`float64` on CPU/CUDA for exact analytical $\mathbf{S}$ and $\mathbf{H}$ extraction and inverse optimization; `expm1`-formulated `float32` perturbation arithmetic with `float64` accumulation for GPU/Metal batched Monte Carlo)
* **Core Dependencies:** `jax`, `jaxlie` ($SE(3)$ manifold ops), `optimistix` / `equinox` (implicit root-finding & PyTrees), `pydantic>=2.0`, `typer`
* **Hardware Targets:** Apple Silicon (CPU AMX `float64` + Metal `float32`), x86_64 AVX-512, NVIDIA CUDA

---

## 1. Executive Summary & Visual Architecture

`toljax` is a headless, differentiable 3D Geometric Dimensioning and Tolerancing (GD&T) variation analysis and inverse cost-tolerance synthesis engine written in JAX. It replaces legacy desktop GUI plugins (Siemens 3DCS, Sigmetrix CETOL $6\sigma$, Ansys RD&T) with a declarative, Git-native CLI and Python API designed for automated CI workflows.

```text
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ assembly.yaml│     │  part.step   │     │   CLI / API  │
   │ parts, GD&T, │     │  CAD faces   │     │ run/optimize │
   │ mates, specs │     │  (optional)  │     │   /inspect   │
   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
          └─────────┬──────────┘                    │
                    ▼                               │
          ┌───────────────────┐                     │
          │ Pydantic schema   │  validate + convert │
          │ (assembly_spec)   │  units + SVD rank   │
          └─────────┬─────────┘                     │
                    ▼                               │
          ┌───────────────────┐                     │
          │ graph.py          │  YAML → flat JAX    │
          │ CompiledAssembly  │  arrays (no Python  │
          └─────────┬─────────┘  loops under jit)   │
                    ▼                               │
   ┌────────────────────────────────────┐           │
   │  CORE MATH ENGINE (JAX)            │◄──────────┘
   │                                    │
   │  se3.py ─► torsor.py ─► mates.py   │
   │                │                   │
   │                ▼                   │
   │           solver.py (LM + IFT)     │
   │                │                   │
   │      ┌─────────┴────────┐          │
   │      ▼                  ▼          │
   │  metrics.py        optimizer.py    │
   │  (4-tier + jam)    (inverse design)│
   └──────┬──────────────────┬──────────┘
          ▼                  ▼
   results.json /     optimized_assembly.yaml
   report.html / .glb
```

### Core Architectural Tenets
1. **Manifold-Exact Kinematics & Direct Geometric Mate Residuals:** All nominal frames and feature deviations live on the Special Euclidean group $SE(3)$ and its Lie algebra $\mathfrak{se}(3)$. Partial-DOF kinematic mates are formulated directly on relative position $\mathbf{p}$ and direction-cosine entries of $\Delta R = R - I_3$ at the physical engagement plane—avoiding logarithmic map twist distortion ($\boldsymbol{\delta} = V^{-1}(\boldsymbol{\theta})\mathbf{p}$) that would otherwise inject spurious second-order rotation-translation cross-terms into the Hessian.
2. **Dimensionless SVD Determinacy Check & Engagement-Depth Jam Semantics:** Real mechanical contact is an inequality, not a least-squares compromise. `toljax` scales rotational Jacobian columns by the assembly's characteristic length $L_{\text{char}}$ before checking both row rank ($C$) and column rank ($6 N_{\text{movable}}$) via SVD at compile time—refusing both under-constrained mechanisms and redundant rigid mates. Both primary locating pins (via engagement-depth tilt binding $C_{r,\text{eff}} = C_r - \frac{L_e}{2}\lVert\Delta\boldsymbol{\theta}_\perp\rVert_2$) and secondary fastener patterns are evaluated for **Jam Probability ($P_{\text{jam}}$ in DPMO)**, with Monte Carlo measurement statistics conditioned on assembleable builds ($M_{\text{fit}} \ge 0$).
3. **Higher-Order Implicit Differentiation (`@jax.custom_jvp`):** First-order Jacobians ($\mathbf{S}$) and second-order Hessians ($\mathbf{H}$) are extracted analytically at the equilibrium root $\mathbf{F}(\mathbf{q}^*, \mathbf{z}) = \mathbf{0}$ via the Implicit Function Theorem—never by unrolling solver iterations through autodiff.
4. **End-to-End Differentiability for Inverse Design:** Because functional assembly yields ($C_p$, $C_{pk}$) are differentiable functions of the input tolerance bands $\mathbf{t}$, `toljax` solves the inverse problem—minimizing manufacturing cost subject to statistical yield constraints—in milliseconds using gradient-based optimization.

### Prior Art & Ecosystem Positioning
* **Academic Lineage:** `toljax` builds upon Small Displacement Torsor (SDT) representation (Desrochers & Rivière, 1997), Lie-group matrix tolerance analysis (Whitney, *Mechanical Assemblies*, 2004), the **Direct Linearization Method (DLM)** and its nonlinear comparison against Monte Carlo (Gao, Chase & Magleby, 1998), and **Second-Order Tolerance Analysis (SOTA)** (Glancy & Chase, 1999). Zone containment and virtual condition semantics follow **ASME Y14.5-2018** and **ASME Y14.5.1-2019**, while assembly stack-up verification targets published cases from Fischer (*Mechanical Tolerance Stackup and Analysis*) and Drake (*Dimensioning and Tolerancing Handbook*).
* **Open-Source Landscape:** Existing open-source Python tools (`dimstack`, `pytolerance`, `Tol-Ninja`) only model 1D scalar chains, while 3D tools (`stackcore`, FreeCAD GD&T workbench) lack kinematic mate solvers, analytical derivatives, or inverse tolerance optimization.
* **JAX Building Blocks:** `toljax` interoperates with `jaxlie` ($SE(3)$ group operations) and `optimistix` / `equinox` (implicit root-finding and static PyTree structures), focusing custom code on GD&T zone coupling, geometric mate mechanics, non-Gaussian SOTA moments, and cost-tolerance synthesis.

---

## 2. Mathematical Specification

### 2.1 Lie Group $SE(3)$, Lie Algebra $\mathfrak{se}(3)$ & `expm1` Perturbation Arithmetic

A rigid coordinate frame $F_a$ relative to $F_b$ is represented by $T_{ba} = (R_{ba}, \mathbf{p}_{ba}) \in SE(3)$. Small Displacement Torsors (SDTs) and part pose perturbations are parameterized in the tangent space $\mathfrak{se}(3)$ as a 6-vector $\boldsymbol{\xi} = [\boldsymbol{\theta}^T, \boldsymbol{\delta}^T]^T \in \mathbb{R}^6$, where $\boldsymbol{\theta} = [\theta_x, \theta_y, \theta_z]^T$ is the rotation vector (rad) and $\boldsymbol{\delta} = [\delta_x, \delta_y, \delta_z]^T$ is the translation vector (mm).

#### Exponential Map & `se3_expm1` ($\mathfrak{se}(3) \to \Delta R = R - I_3, \, \Delta \mathbf{p}$)
To ensure $C^\infty$ smoothness, avoid `NaN` gradients at $\lVert\boldsymbol{\theta}\rVert \to 0$, and prevent catastrophic cancellation when multiplying $\sim 100\text{ mm}$ lever arms by small rotations in `float32`, `se3.py` computes the **incremental rotation matrix $\Delta R(\boldsymbol{\theta}) = R(\boldsymbol{\theta}) - I_3$** directly (`se3_expm1`) using 8th-order even-power Taylor expansions in $\theta^2 = \boldsymbol{\theta}^T \boldsymbol{\theta}$ guarded by the **JAX Double-`jnp.where` Idiom**:

$$\theta_{\text{safe}} = \sqrt{\text{where}(\theta^2 < \theta_{\text{cut}}^2, \, 1.0, \, \theta^2)}$$

where $\theta_{\text{cut}}^2 = 10^{-6}$ in `float64` and $\theta_{\text{cut}}^2 = 10^{-1}$ in `float32`:

$$\Delta R(\boldsymbol{\theta}) = R(\boldsymbol{\theta}) - I_3 = c_1(\theta^2) [\boldsymbol{\theta}]_\times + c_2(\theta^2) [\boldsymbol{\theta}]_\times^2, \qquad \Delta \mathbf{p}(\boldsymbol{\theta}, \boldsymbol{\delta}) = \left( I_3 + c_2(\theta^2) [\boldsymbol{\theta}]_\times + c_3(\theta^2) [\boldsymbol{\theta}]_\times^2 \right) \boldsymbol{\delta}$$

$$c_1(\theta^2) = \begin{cases} 1 - \frac{\theta^2}{6} + \frac{\theta^4}{120} - \frac{\theta^6}{5040} + \frac{\theta^8}{362880} & \theta^2 < \theta_{\text{cut}}^2 \\ \frac{\sin\theta_{\text{safe}}}{\theta_{\text{safe}}} & \text{otherwise} \end{cases}$$

$$c_2(\theta^2) = \begin{cases} \frac{1}{2} - \frac{\theta^2}{24} + \frac{\theta^4}{720} - \frac{\theta^6}{40320} + \frac{\theta^8}{3628800} & \theta^2 < \theta_{\text{cut}}^2 \\ \frac{1 - \cos\theta_{\text{safe}}}{\theta_{\text{safe}}^2} & \text{otherwise} \end{cases}$$

$$c_3(\theta^2) = \begin{cases} \frac{1}{6} - \frac{\theta^2}{120} + \frac{\theta^4}{5040} - \frac{\theta^6}{362880} + \frac{\theta^8}{39916800} & \theta^2 < \theta_{\text{cut}}^2 \\ \frac{\theta_{\text{safe}} - \sin\theta_{\text{safe}}}{\theta_{\text{safe}}^3} & \text{otherwise} \end{cases}$$

> **Why `expm1` Eliminates `float32` Lever-Arm Cancellation:** If a part rotates by $\theta \sim 10^{-3}\text{ rad}$ at a lever arm $\mathbf{r} = [100, 0, 0]^T\text{ mm}$, computing $(I_3 + \Delta R)\mathbf{r} - \mathbf{r}$ in `float32` loses 5 digits of precision ($\sim 6 \times 10^{-6}\text{ mm}$ roundoff error) because $\mathbf{r}$ is added and subtracted. By storing nominal transforms in `float64` at compile time (`graph.py`) where nominal mate frames coincide ($\mathbf{p}_{\text{tgt,nom}} - \mathbf{p}_{\text{src,nom}} = \mathbf{0}$) and evaluating $\Delta \mathbf{p}_{\text{global}} = R_{\text{nom}}(\Delta R_{\text{part}} \, \mathbf{r}_{\text{feat}} + \Delta \mathbf{p}_{\text{part}} + \dots)$, large lever arms $\mathbf{r}_{\text{feat}}$ **only ever multiply $\mathcal{O}(10^{-3})$ matrices $\Delta R$**, preserving `< 10^{-9}\text{ mm}` accuracy even in `float32`!

#### Logarithmic Map ($\log^\vee: SE(3) \to \mathbb{R}^6$)
Given $\Delta T = (R, \mathbf{p})$, let $\mathbf{v} = \frac{1}{2}[R_{32}-R_{23}, \; R_{13}-R_{31}, \; R_{21}-R_{12}]^T$, $s^2 = \mathbf{v}^T \mathbf{v} = \sin^2\theta$, and $c = \frac{\text{Tr}(R) - 1}{2} = \cos\theta$. Using $s_{\text{safe}} = \sqrt{\text{where}(s^2 < s_{\text{cut}}^2, 1.0, s^2)}$ ($s_{\text{cut}}^2 = 10^{-6}$ in `float64`, $10^{-1}$ in `float32`):

$$\boldsymbol{\theta} = \begin{cases} \left(1 + \frac{s^2}{6} + \frac{3 s^4}{40} + \frac{5 s^6}{112} + \frac{35 s^8}{1152}\right) \mathbf{v} & s^2 < s_{\text{cut}}^2 \text{ and } c > 0 \\ \frac{\operatorname{atan2}(s_{\text{safe}}, c)}{s_{\text{safe}}} \mathbf{v} & \text{otherwise} \end{cases}$$

$$\boldsymbol{\delta} = V^{-1}(\boldsymbol{\theta}) \mathbf{p} = \left( I_3 - \frac{1}{2}[\boldsymbol{\theta}]_\times + c_4(\theta^2) [\boldsymbol{\theta}]_\times^2 \right) \mathbf{p}, \qquad c_4(\theta^2) = \begin{cases} \frac{1}{12} + \frac{\theta^2}{720} + \frac{\theta^4}{30240} + \frac{\theta^6}{1209600} & \theta^2 < \theta_{\text{cut}}^2 \\ \frac{1}{\theta_{\text{safe}}^2} - \frac{1 + \cos\theta_{\text{safe}}}{2\theta_{\text{safe}}\sin\theta_{\text{safe}}} & \text{otherwise} \end{cases}$$

---

### 2.2 GD&T Feature Torsors, Coupled Tolerance Zones & Material Modifiers

Let feature $i$ on part $p$ have a nominal frame $T_{p,i}^{\text{nom}} \in SE(3)$:
* **Planar Surface (`planar`):** $+Z$ axis is the outward surface normal; origin at face area centroid; $X, Y$ aligned with principal inertia axes of boundary $\{L_x, L_y\}$.
* **Cylindrical Feature (`cylinder_pin` / `cylinder_hole`):** $+Z$ axis is the cylinder axis; origin at the axial midpoint of feature length $L_z$; nominal diameter $D_{\text{nom}}$ with upper/lower deviations $(\Delta_{\text{upper}}, \Delta_{\text{lower}})$.
* **Slot Feature (`slot_width`):** $+Z$ axis along slot depth $L_z$; $+Y$ axis normal to the opposing slot walls of width $W_{\text{nom}}$; $+X$ axis along slot length $L_x$.
* **Spherical / Point Feature (`sphere` / `point`):** Origin at sphere center or functional target point.

#### Unilateral / Asymmetrical Size Limits (ISO Fits)
Because engineering size tolerances (e.g., ISO `h6` pin `0.000 / -0.012` or `H7` hole `+0.025 / 0.000`) are often unilateral relative to nominal $D_{\text{nom}}$, `toljax` separates the **mean manufactured radius** $r_{\text{mid},i}$ from the **radial size half-tolerance** $\frac{t_{\text{size},i}}{4}$:

$$D_{\max,i} = D_{\text{nom},i} + \Delta_{\text{upper},i}, \qquad D_{\min,i} = D_{\text{nom},i} + \Delta_{\text{lower},i}$$
$$r_{\text{mid},i} = \frac{D_{\max,i} + D_{\min,i}}{4}, \qquad t_{\text{size},i} = D_{\max,i} - D_{\min,i}, \qquad r_i(z_{\text{size}}) = r_{\text{mid},i} + \underbrace{\frac{t_{\text{size},i}}{4} z_{\text{size}}}_{\delta r_i}$$

#### Symmetric Coupled Zone Parameterization (Statistical Tiers) & Exact SOCP (Worst-Case Tier)
In ASME Y14.5-2018 / Y14.5.1-2019, a positional or profile tolerance zone $t_{\text{eff}}$ constrains the entire feature of length $L_z$ to lie inside a cylinder/slab of width $t_{\text{eff}}$ ($\lVert \mathbf{p}_{\text{top}} \rVert_2 \le t_{\text{eff}}/2, \; \lVert \mathbf{p}_{\text{bot}} \rVert_2 \le t_{\text{eff}}/2$), while an optional orientation refinement $t_{\text{o}} = \min(t_{\text{orient}}, t_{\text{eff}})$ further restricts the end-to-end tilt difference $\lVert \mathbf{p}_{\text{top}} - \mathbf{p}_{\text{bot}} \rVert_2 \le t_{\text{o}}$:

1. **Statistical Tiers 2–4 (Symmetric Endpoint Decomposition):**
   * **Unrefined Cylindrical Zone ($t_{\text{o}} = t_{\text{eff}}$, 4 spatial DOFs + 1 size DOF):**
     Parameterized by top and bottom axis endpoints $\mathbf{u}_{\text{top}} = [z_1, z_2]^T \in \mathcal{B}_2(1)$ and $\mathbf{u}_{\text{bot}} = [z_3, z_4]^T \in \mathcal{B}_2(1)$:
     $$\mathbf{p}_{\text{top}} = \frac{t_{\text{eff}}}{2}\mathbf{u}_{\text{top}}, \qquad \mathbf{p}_{\text{bot}} = \frac{t_{\text{eff}}}{2}\mathbf{u}_{\text{bot}}$$
   * **Orientation-Refined Cylindrical Zone ($t_{\text{o}} < t_{\text{eff}}$, 6 spatial DOFs + 1 size DOF):**
     To treat top and bottom endpoints completely symmetrically without rejection sampling, introduce pure-translation disk $\mathbf{u}_1 = [z_1, z_2]^T \in \mathcal{B}_2(1)$ and symmetric endpoint tilt disks $\mathbf{u}_2 = [z_3, z_4]^T \in \mathcal{B}_2(1)$, $\mathbf{u}_3 = [z_5, z_6]^T \in \mathcal{B}_2(1)$:
     $$\mathbf{p}_{\text{top}} = \frac{t_{\text{eff}} - t_{\text{o}}}{2}\mathbf{u}_1 + \frac{t_{\text{o}}}{2}\mathbf{u}_2, \qquad \mathbf{p}_{\text{bot}} = \frac{t_{\text{eff}} - t_{\text{o}}}{2}\mathbf{u}_1 + \frac{t_{\text{o}}}{2}\mathbf{u}_3$$
     which maps linearly to the feature midpoint translation and axial tilt:
     $$\begin{bmatrix} \delta_x \\ \delta_y \end{bmatrix} = \frac{\mathbf{p}_{\text{top}} + \mathbf{p}_{\text{bot}}}{2} = \frac{t_{\text{eff}} - t_{\text{o}}}{2}\mathbf{u}_1 + \frac{t_{\text{o}}}{4}(\mathbf{u}_2 + \mathbf{u}_3), \qquad \begin{bmatrix} \theta_x \\ \theta_y \end{bmatrix} = \frac{t_{\text{o}}}{2 L_z}\begin{bmatrix} -(z_4 - z_6) \\ z_3 - z_5 \end{bmatrix}$$
     Notice that $\lVert\mathbf{p}_{\text{top}}\rVert_2 \le \frac{t_{\text{eff}}}{2}$, $\lVert\mathbf{p}_{\text{bot}}\rVert_2 \le \frac{t_{\text{eff}}}{2}$, $\lVert\mathbf{p}_{\text{top}} - \mathbf{p}_{\text{bot}}\rVert_2 \le t_{\text{o}}$, and both upward and downward axial extrapolations reach their symmetric bounds $\pm\left(\frac{t_{\text{eff}}}{2} + \mu t_{\text{o}}\right)$!
   * **Planar Surface Zone (`planar`):**
     Symmetric top/bottom/left/right decomposition with midpoint shift $\delta_z = \frac{t_{\text{pos}} - t_{\text{o}}}{2} z_1$ and tilt $\theta_x = \frac{t_{\text{o}}}{L_y} z_2, \; \theta_y = \frac{t_{\text{o}}}{L_x} z_3$. Form tolerance (`flatness`) is converted to seating rock uncertainty ($\theta_x, \theta_y$ with $\delta_z = 0$) only when the planar face acts as a primary datum contact surface.

2. **Worst-Case Tier 1 (Exact Three-Cone SOCP Boundary Optimization):**
   Because linear decompositions of $\mathcal{B}_2(1)$ disks do not include orthogonal translation-plus-tilt combinations ($\mathbf{p}_{\text{mid}} \perp \Delta \mathbf{p}_{\text{tilt}}$ where $\lVert \mathbf{p}_{\text{mid}} \pm \frac{1}{2}\Delta \mathbf{p}_{\text{tilt}} \rVert_2 = \sqrt{\lVert\mathbf{p}_{\text{mid}}\rVert_2^2 + \frac{1}{4}\lVert\Delta\mathbf{p}_{\text{tilt}}\rVert_2^2} \le \frac{t_{\text{eff}}}{2}$), Worst-Case Tier 1 maximizes each feature's contribution over the **exact ASME Y14.5.1 convex intersection of three Euclidean cones**:
   $$\max_{\mathbf{p}_{\text{top}}, \mathbf{p}_{\text{bot}} \in \mathbb{R}^2} \quad \mathbf{g}_{\text{top}}^T \mathbf{p}_{\text{top}} + \mathbf{g}_{\text{bot}}^T \mathbf{p}_{\text{bot}} \qquad \text{s.t.} \quad \lVert\mathbf{p}_{\text{top}}\rVert_2 \le \frac{t_{\text{eff,WC}}}{2}, \quad \lVert\mathbf{p}_{\text{bot}}\rVert_2 \le \frac{t_{\text{eff,WC}}}{2}, \quad \lVert\mathbf{p}_{\text{top}} - \mathbf{p}_{\text{bot}}\rVert_2 \le t_{\text{o}}$$
   where $\mathbf{g}_{\text{top}} = \frac{1}{2}[s_{\delta_x}, s_{\delta_y}]^T + \frac{1}{L_z}[s_{\theta_y}, -s_{\theta_x}]^T$ and $\mathbf{g}_{\text{bot}} = \frac{1}{2}[s_{\delta_x}, s_{\delta_y}]^T - \frac{1}{L_z}[s_{\theta_y}, -s_{\theta_x}]^T$ are the endpoint sensitivity vectors. This 4-variable 3-cone SOCP is solved exactly per feature via projected Lagrangian dual / closed-form KKT cases in microseconds.

#### ASME Y14.5 MMC / LMC Bonus Modes (`mmc_statistical_mode`)
For a feature of size with modifier $m \in \{\text{RFS}, \text{MMC}, \text{LMC}\}$, the available diameter bonus on the physical interval $z_{\text{size}} \in [-1, 1]$ around the midpoint radius $r_{\text{mid},i}$ is the kink-free affine function:

$$\Delta_{\text{bonus}}(\delta r_i) = \begin{cases} 0 & \text{if } m = \text{RFS} \\ \frac{t_{\text{size},i}}{2}(1 - z_{\text{size}}) & \text{if } m = \text{MMC (External Pin)} \text{ or } \text{LMC (Internal Hole)} \\ \frac{t_{\text{size},i}}{2}(1 + z_{\text{size}}) & \text{if } m = \text{MMC (Internal Hole)} \text{ or } \text{LMC (External Pin)} \end{cases}$$

`toljax` provides two selectable physical interpretations via `assembly.mmc_statistical_mode`:
1. **`"acceptance_truncated"` (Default for Statistical Tiers 2–4 — Process-Driven):**
   * **Tiers 2 & 3 (Linear RSS & SOTA):** Endpoint variables $\mathbf{u}_k$ are independent 2D Gaussians with component standard deviation $\sigma_u = \frac{1}{C_{p,\text{proc}}} = \frac{1}{3}$ scaled by the nominal process capability $t_{\text{eff,stat}} = t_{\text{pos}}$ (independent of individual part size $\delta r_i$). Consequently, the size-position cross-derivative $\frac{\partial^2 y}{\partial z_{\text{pos}} \partial z_{\text{size}}}$ is zero in Tiers 2 and 3.
   * **Tier 4 (Monte Carlo):** Draws $\mathbf{u}_k \sim \mathcal{N}(\mathbf{0}, \sigma_u^2 I_2)$ scaled by $t_{\text{pos}}/2$, and truncates/accepts parts against the actual size-dependent virtual condition boundary $\lVert\mathbf{p}_{\text{top/bot}}\rVert_2 \le \frac{t_{\text{pos}} + \Delta_{\text{bonus}}(\delta r_i)}{2}$.
   * **Tier 1 (Worst-Case):** Uses the full maximum boundary $t_{\text{eff,WC}} = t_{\text{pos}} + t_{\text{size},i}$.
2. **`"fully_consumed"` (Conservative Gauge-Limit Statistical Mode):**
   * Scales $\mathbf{u}_k$ by $t_{\text{eff}}(\delta r_i) = t_{\text{pos}} + \Delta_{\text{bonus}}(\delta r_i)$ across all tiers, activating the bilinear size-position cross-derivative in $\mathbf{H}_k$ for Tiers 3 and 4.

---

### 2.3 Direct Geometric Mate Residuals, Engagement-Depth Tilt Binding & Jam Mechanics

#### Mating Engagement Frame (`contact_offset_z`)
Because a cylindrical pin or hole has its GD&T tolerance zone centered at its axial midpoint ($z = 0$, spanning $[-L_z/2, +L_z/2]$), a perpendicularity tilt $\theta_x, \theta_y$ causes the axis offset at the **seating/engagement plane** to differ from the midpoint offset by $\theta \cdot z_{\text{contact}}$. Each mate $m$ specifies local axial engagement coordinates $z_{\text{contact,src}}, z_{\text{contact,tgt}}$ (defaulting to the mating datum face intersection, e.g., $z_{\text{contact,src}} = -L_{z,\text{pin}}/2$ at the housing seating face, with engagement depth $L_e = \min(L_{z,\text{src}}, L_{z,\text{tgt}})$):

$$T_{p,i}^{\text{mate}}(\mathbf{q}_p, \mathbf{z}) = T_p(\mathbf{q}_p) \, T_{p,i}^{\text{nom}} \exp(\boldsymbol{\xi}_i(\mathbf{z})^\wedge) \, \text{Trans}_z(z_{\text{contact,src}})$$

#### Direct Geometric Mate Residuals (Eliminating Second-Order Twist Distortion)
Let the relative transform between the source and target mate frames at the engagement plane be:

$$\Delta T_m(\mathbf{q}, \mathbf{z}) = \left( T_{p,i}^{\text{mate}}(\mathbf{q}_p, \mathbf{z}) \right)^{-1} T_{p',j}^{\text{mate}}(\mathbf{q}_{p'}, \mathbf{z}) = \begin{bmatrix} I_3 + \Delta R & \mathbf{p} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

where $\mathbf{p} = [p_x, p_y, p_z]^T = R_{\text{src}}^T (\mathbf{p}_{\text{tgt}} - \mathbf{p}_{\text{src}})$ is the exact position of the target mate origin expressed in the source mate frame (computed via `expm1` perturbation differences), and $\Delta R = R_{\text{src}}^T R_{\text{tgt}} - I_3$ is the incremental relative rotation matrix. Each mate $m$ evaluates its residual $\mathbf{r}_m(\mathbf{q}, \mathbf{z}) \in \mathbb{R}^{c_m}$ directly from $(\Delta R, \mathbf{p})$:

| Mate Type | DOFs ($c_m$) | Exact Geometric Residual $\mathbf{r}_m(\mathbf{q}, \mathbf{z}) = \mathbf{0}$ | Physical Meaning |
| :--- | :---: | :--- | :--- |
| **`planar_coincident`** | 3 | $[-\Delta R_{23}, \; \Delta R_{13}, \; p_z]^T = \mathbf{0}$ | Target normal $\hat{\mathbf{e}}_{z,\text{tgt}}$ parallel to $\hat{\mathbf{e}}_{z,\text{src}}$; target origin on source $XY$ plane |
| **`parallel_pin_hole`** | 2 | $[p_x - C_r \eta_x, \; p_y - C_r \eta_y]^T = \mathbf{0}$ | Hole center at engagement mid-depth offset by radial clearance $C_r \boldsymbol{\eta}$ |
| **`clocking_pin_slot`** | 1 | $[p_y - C_w \eta_y] = 0$ | Pin center at engagement mid-depth offset by half-width clearance $C_w \eta_y$ |
| **`line_contact`** | 2 | $[-\Delta R_{12}, \; p_y]^T = \mathbf{0}$ | Secondary datum edge aligned with $X$ axis ($p_y = 0, \theta_z = 0$) |
| **`point_contact`** | 1 | $[p_x] = 0$ | Tertiary datum stop at $p_x = 0$ |
| **`coaxial_rigid`** | 4 | $[-\Delta R_{23}, \; \Delta R_{13}, \; p_x, \; p_y]^T = \mathbf{0}$ | Coaxial axes ($\hat{\mathbf{e}}_{z,\text{tgt}} \parallel \hat{\mathbf{e}}_{z,\text{src}}$) and coincident $XY$ centers |
| **`cylindrical_clearance`** | 4 | $[-\Delta R_{23} - \frac{2 C_r}{L_e}\eta_{\theta_x}, \; \Delta R_{13} - \frac{2 C_r}{L_e}\eta_{\theta_y}, \; p_x - C_r \eta_x, \; p_y - C_r \eta_y]^T = \mathbf{0}$ | Standalone floating pin/hole (4-DOF) |
| **`spherical_joint`** | 3 | $[p_x, \; p_y, \; p_z]^T = \mathbf{0}$ | Coincident sphere centers |
| **`rigid_weld`** | 6 | $[-\Delta R_{23}, \; \Delta R_{13}, \; -\Delta R_{12}, \; p_x, \; p_y, \; p_z]^T = \mathbf{0}$ | Full 6-DOF rigid lock |

#### Engagement-Depth Tilt Binding Margin (`pin_hole_tilt_jam_margin`)
When a pin mates inside a hole of engagement depth $L_e$ while their axes have a relative perpendicularity tilt $\Delta \boldsymbol{\theta}_\perp = [\Delta R_{13}, \; -\Delta R_{23}]^T$ (determined by the primary datum plane contact and the features' perpendicularity errors), the lateral offset at the top ($z = +L_e/2$) and bottom ($z = -L_e/2$) of the engagement cylinder is $\mathbf{p}_{xy,\text{mid}} \pm \frac{L_e}{2}\Delta \boldsymbol{\theta}_\perp$.

> **Keeping Equality Residuals $C^\infty$ at Nominal:** Subtracting a regularized cone norm $\frac{L_e}{2}\sqrt{\Delta R_{13}^2 + \Delta R_{23}^2 + \epsilon_{\text{reg}}^2}$ directly inside the equality residual $\mathbf{F}(\mathbf{q}, \mathbf{z}) = \mathbf{0}$ would place a cone apex at $\Delta R = 0$, shifting the analytical Jacobian by $\frac{L_e}{2}\epsilon_{\text{reg}}$ and injecting a $1/\epsilon_{\text{reg}}$ curvature spike into the Hessian at nominal. Instead, the equality residual uses the smooth bilinear product $C_r(\mathbf{z})\boldsymbol{\eta}_{xy}$, while engagement-depth tilt binding at $z = \pm L_e/2$ is evaluated as an inequality fit margin:

$$M_{\text{fit},m}^{\text{tilt}}(\mathbf{z}) = C_r(\mathbf{z}) - \max\!\left(\left\lVert \mathbf{p}_{xy,\text{mid}} + \frac{L_e}{2}\begin{bmatrix} \Delta R_{13} \\ -\Delta R_{23} \end{bmatrix} \right\rVert_2, \; \left\lVert \mathbf{p}_{xy,\text{mid}} - \frac{L_e}{2}\begin{bmatrix} \Delta R_{13} \\ -\Delta R_{23} \end{bmatrix} \right\rVert_2\right)$$

Whenever $\frac{L_e}{2}\lVert\Delta\boldsymbol{\theta}_\perp\rVert_2 > C_r(\mathbf{z})$ (or the combined float + tilt exceeds $C_r(\mathbf{z})$ at either end of the hole), $M_{\text{fit},m}^{\text{tilt}}(\mathbf{z}) < 0$ and **the primary pin itself binds/jams in the hole due to angular cocking**, even without a second pin!

#### Dimensionless Two-Sided SVD Rank Check & Conditional Jam Analysis
Because the columns of $\mathbf{J}_{\mathbf{q}}(\mathbf{0}, \mathbf{0}) \in \mathbb{R}^{C \times 6 N_{\text{movable}}}$ mix dimensionless angular derivatives (scaled by lever arms $\sim L_{\text{char}}$ in translational rows) and translational derivatives ($\sim 1$), `graph.py` first computes the assembly's characteristic length $L_{\text{char}} = \max_{i,j} \lVert \mathbf{p}_{i}^{\text{nom}} - \mathbf{p}_{j}^{\text{nom}} \rVert_2$ (or $100.0\text{ mm}$ minimum), right-multiplies $\mathbf{J}_{\mathbf{q}}$ by the column-scaling matrix $\mathbf{D}_{\text{scale}} = \text{diag}(L_{\text{char}}^{-1} I_3, I_3)$ and left-multiplies translational rows by $L_{\text{char}}^{-1}$ to form the **dimensionless Jacobian** $\tilde{\mathbf{J}}_{\mathbf{q}}$, and computes its SVD with threshold $\epsilon_{\text{svd}} = 10^{-8} \sigma_{\max}(\tilde{\mathbf{J}}_{\mathbf{q}})$:
1. **Under-Constrained Check ($\text{rank}(\tilde{\mathbf{J}}_{\mathbf{q}}) < 6 N_{\text{movable}}$):** Raises `UnderConstrainedMechanismError` and reports the nullspace twist eigenvector (identifying which part and DOF is free to drift).
2. **Over-Constrained Check ($C > \text{rank}(\tilde{\mathbf{J}}_{\mathbf{q}})$):** Raises `RedundantMateError` identifying the conflicting mates.
3. **Primary Tilt-Binding & Secondary Pattern Jam Evaluation (`M_fit`):**
   Every clearance mate $m$ automatically contributes a primary tilt-binding margin $M_{\text{fit},m}^{\text{tilt}}(\mathbf{z}) = C_{r,\text{eff},m}(\mathbf{z})$, and every secondary fastener check $k$ in `jam_checks` contributes a pattern alignment margin $M_{\text{fit},k}^{\text{pat}}(\mathbf{z}) = C_{r,\text{eff},k}(\mathbf{z}) - \lVert \mathbf{p}_{xy,k}(\mathbf{q}^*(\mathbf{z}), \mathbf{z}) \rVert_2$.
   * An assembly realization is **assembleable** if and only if $M_{\text{fit}}(\mathbf{z}) = \min\!\left(\min_m M_{\text{fit},m}^{\text{tilt}}(\mathbf{z}), \; \min_k M_{\text{fit},k}^{\text{pat}}(\mathbf{z})\right) \ge 0$.
   * All Tier 4 Monte Carlo measurement moments ($\mu, \sigma, P_{0.00135}, P_{0.99865}, \text{spec\_dpmo}$) are conditioned on $M_{\text{fit}}(\mathbf{z}) \ge 0$. Whenever `jam_dpmo > 1000.0`, `results.json` emits `"sota_conditioning_warning": true` noting that Tier 3 SOTA moments are unconditional while Tier 4 MC moments are conditioned on non-jammed builds.

---

### 2.4 Functional Measurement Functions ($\Delta y_k = h_k(\mathbf{q}^*, \mathbf{z}) - y_{k,\text{nom}}$)
To avoid `float32` subtraction cancellation on nominal dimensions, each measurement $k$ evaluates its **deviation from nominal** $\Delta y_k(\mathbf{q}^*, \mathbf{z})$ in solver precision (`float64` or `float32`) and adds $y_{k,\text{nom}}$ in `float64`:

1. **`distance_xyz` (Projected Global Axis Distance):**
   $$\Delta y_{\text{dist}}(\mathbf{q}^*, \mathbf{z}) = \hat{\mathbf{a}}^T \left( \Delta \mathbf{p}_{\text{tgt,global}} - \Delta \mathbf{p}_{\text{src,global}} \right), \qquad y_{\text{dist}} = y_{\text{dist,nom}} + \Delta y_{\text{dist}}$$
2. **`point_to_plane_distance` (Signed Distance Along Source Datum Normal):**
   Let $\mathbf{d}_{\text{nom}} = \mathbf{p}_{\text{tgt,nom}} - \mathbf{p}_{\text{src,nom}}$ and $\Delta \mathbf{d} = \Delta \mathbf{p}_{\text{tgt,global}} - \Delta \mathbf{p}_{\text{src,global}}$:
   $$\Delta y_{\text{p2p}}(\mathbf{q}^*, \mathbf{z}) = \left( R_{\text{src,nom}} \hat{\mathbf{a}} \right)^T \Delta \mathbf{d} + \left( R_{\text{src,nom}} \Delta R_{\text{src}} \hat{\mathbf{a}} \right)^T (\mathbf{d}_{\text{nom}} + \Delta \mathbf{d})$$
3. **`coaxiality` (Radial Axis Offset):**
   $$\Delta \mathbf{p}_\perp = \left( I_3 - \hat{\mathbf{n}}_{\text{src}} \hat{\mathbf{n}}_{\text{src}}^T \right) (\mathbf{p}_{\text{tgt}} - \mathbf{p}_{\text{src}}), \qquad h_{\text{coax}}(\mathbf{q}^*, \mathbf{z}) = \sqrt{\lVert \Delta \mathbf{p}_\perp \rVert_2^2 + \epsilon_{\text{reg}}^2}$$
4. **`angular_tilt` (Signed or Unsigned Relative Angular Deviation):**
   Let $\Delta R_{\text{rel}} = R_{\text{src}}^T R_{\text{tgt}} - I_3$; returns either signed $\hat{\mathbf{a}}^T [-\Delta R_{\text{rel},23}, \; \Delta R_{\text{rel},13}, \; -\Delta R_{\text{rel},12}]^T$ or unsigned cone tilt $\sqrt{\Delta R_{\text{rel},13}^2 + \Delta R_{\text{rel},23}^2 + \epsilon_{\text{reg}}^2}$.

---

### 2.5 Forward Solver & Higher-Order Implicit Differentiation (`@jax.custom_jvp`)

Stacking all $C = 6 N_{\text{movable}}$ active geometric mate residuals yields $\mathbf{F}(\mathbf{q}, \mathbf{z}) \in \mathbb{R}^{C}$.

1. **Forward Pass (Batched Newton-Raphson / Damped Levenberg-Marquardt):**
   Starting from $\mathbf{q}^{(0)} = \mathbf{0}$, iterate for $k = 0, \dots, K_{\max}-1$ ($K_{\max} = 4$ via `jax.lax.fori_loop`):
   $$\mathbf{J}_{\mathbf{q}}^{(k)} = \frac{\partial \mathbf{F}(\mathbf{q}^{(k)}, \mathbf{z})}{\partial \mathbf{q}}, \qquad \Delta \mathbf{q}^{(k)} = -\left( {\mathbf{J}_{\mathbf{q}}^{(k)}}^T \mathbf{J}_{\mathbf{q}}^{(k)} + \lambda I \right)^{-1} {\mathbf{J}_{\mathbf{q}}^{(k)}}^T \mathbf{F}(\mathbf{q}^{(k)}, \mathbf{z})$$
   where $\lambda = 10^{-12}$ in `float64` ($10^{-7}$ in `float32`). Because $\mathbf{F}(\mathbf{0}, \mathbf{0}) = \mathbf{0}$ and $\mathbf{J}_{\mathbf{q}}$ is square and full-rank for every validated determinate assembly, quadratic convergence reaches $\lVert\mathbf{F}\rVert_\infty < 10^{-14}$ (`float64`) in 3 iterations.

2. **First- and Second-Order Implicit Differentiation via Recursive `@jax.custom_jvp`:**
   Because $\mathbf{F}(\mathbf{q}^*(\mathbf{z}), \mathbf{z}) = \mathbf{0}$ identically on the entire tolerance neighborhood:
   $$\mathbf{J}_{\mathbf{q}} \frac{\partial \mathbf{q}^*}{\partial \mathbf{z}} + \mathbf{J}_{\mathbf{z}} = \mathbf{0} \implies \dot{\mathbf{q}}^* = -\mathbf{J}_{\mathbf{q}}^{-1} \left( \mathbf{J}_{\mathbf{z}} \dot{\mathbf{z}} \right)$$
   By decorating `solve_equilibrium(z)` with `@jax.custom_jvp` and calling `q_star = solve_equilibrium(z)` recursively inside the JVP rule:
   * **Reverse Mode (`jax.jacrev` / `jax.grad`):** JAX automatically transposes `jnp.linalg.solve` and `jax.jvp` into the exact adjoint solve $\mathbf{J}_{\mathbf{q}}^T \boldsymbol{\nu} = \bar{\mathbf{q}}^*$ and $\bar{\mathbf{z}} = -\boldsymbol{\nu}^T \mathbf{J}_{\mathbf{z}}$.
   * **Second-Order Hessian (`jax.hessian = jax.jacfwd(jax.jacrev)`):** The outer `jacfwd` differentiates through `q_star = solve_equilibrium(z)` via the `@jax.custom_jvp` rule again, capturing the exact implicit Hessian $\mathbf{H}_k = \nabla_{\mathbf{z}}^2 y_k(\mathbf{0})$ without ever unrolling the `fori_loop`.

---

### 2.6 Statistical Moments (Gaussian & Non-Gaussian Cumulants), Worst-Case & Inverse Synthesis

For each functional measurement $y_k(\mathbf{z})$ with limits $[\text{LSL}_k, \text{USL}_k]$, let $\mathbf{s}_k = \nabla_{\mathbf{z}} y_k(\mathbf{0}) \in \mathbb{R}^{N_{\text{vars}}}$ and $\mathbf{H}_k = \nabla_{\mathbf{z}}^2 y_k(\mathbf{0}) \in \mathbb{R}^{N_{\text{vars}} \times N_{\text{vars}}}$ be evaluated at $\mathbf{z} = \mathbf{0}$ in `float64`:

1. **Tier 1 — Worst-Case (WC) Boundary Analysis:**
   Evaluates the exact 3-cone SOCP maximum for orientation-refined features plus $\sum_{i \in \mathcal{I}_{\text{box}}} \lvert s_{k,i} \rvert + \sum_{(i_x, i_y) \in \mathcal{I}_{\text{disk}}} \sqrt{s_{k,i_x}^2 + s_{k,i_y}^2}$ for unrefined intervals and disks, plus the second-order directional shift $\frac{1}{2}\mathbf{z}_{\text{WC}}^T \mathbf{H}_k \mathbf{z}_{\text{WC}}$.

2. **Tier 2 — First-Order Linear RSS:**
   With diagonal covariance $\boldsymbol{\Sigma}_{\mathbf{z}} = \text{diag}(\sigma_{z_1}^2, \dots, \sigma_{z_N}^2)$ where:
   * $\sigma_{z_i} = \frac{1}{C_{p,\text{proc}}} = \frac{1}{3}$ ($\sigma_{z_i}^2 = \frac{1}{9}$) for Gaussian manufacturing tolerances,
   * $\sigma_{\eta} = \frac{1}{2}$ ($\sigma_{\eta}^2 = \frac{1}{4}$) per component for 2D uniform unit-disk clearance float $\boldsymbol{\eta} \sim \mathcal{U}(\mathcal{B}_2(1))$,
   * $\sigma_{\eta} = \frac{1}{\sqrt{3}}$ ($\sigma_{\eta}^2 = \frac{1}{3}$) for 1D uniform slot clearance float $\eta \sim \mathcal{U}([-1, 1])$:
   $$\mu_{k,\text{RSS}} = y_k(\mathbf{0}), \qquad \sigma_{k,\text{RSS}} = \sqrt{\mathbf{s}_k^T \boldsymbol{\Sigma}_{\mathbf{z}} \mathbf{s}_k}$$

3. **Tier 3 — Second-Order Tolerance Analysis (SOTA with Exact Non-Gaussian Cumulants):**
   For the quadratic expansion $y_k(\mathbf{z}) \approx y_k(\mathbf{0}) + \mathbf{s}_k^T \mathbf{z} + \frac{1}{2}\mathbf{z}^T \mathbf{H}_k \mathbf{z}$:
   * **Exact Mean Shift (valid for any zero-mean distribution with covariance $\boldsymbol{\Sigma}_{\mathbf{z}}$):**
     $$\Delta \mu_k = \frac{1}{2}\text{Tr}(\mathbf{H}_k \boldsymbol{\Sigma}_{\mathbf{z}}), \qquad \mu_{k,\text{2nd}} = y_k(\mathbf{0}) + \Delta \mu_k$$
   * **Exact Second-Order Variance (with 1D and 2D Uniform Disk Fourth-Cumulant Terms):**
     $$\sigma_{k,\text{2nd}}^2 = \mathbf{s}_k^T \boldsymbol{\Sigma}_{\mathbf{z}} \mathbf{s}_k + \frac{1}{2}\text{Tr}\!\left((\mathbf{H}_k \boldsymbol{\Sigma}_{\mathbf{z}})^2\right) + \frac{1}{4} \sum_{i \in \mathcal{I}_{\text{1D-unif}}} \kappa_i H_{k,ii}^2 \sigma_{z_i}^4 + \frac{1}{4} \sum_{(i_x, i_y) \in \mathcal{I}_{\text{disk}}^{\text{unif}}} \left[ \kappa_{1111}\!\left(H_{k,i_x i_x}^2 + H_{k,i_y i_y}^2\right) + \kappa_{1122}\!\left(2 H_{k,i_x i_x} H_{k,i_y i_y} + 4 H_{k,i_x i_y}^2\right) \right] \sigma_{\eta}^4$$
     where $\kappa_i = -\frac{6}{5}$ for 1D uniform variables, and for a 2D uniform unit disk ($\sigma_\eta^2 = \frac{1}{4}$, $\mathbb{E}[\eta_x^4] = \frac{1}{8}$, $\mathbb{E}[\eta_x^2 \eta_y^2] = \frac{1}{24}$), the normalized fourth cumulants are:
     $$\kappa_{1111} = \kappa_{2222} = \frac{\mathbb{E}[\eta_x^4]}{\sigma_\eta^4} - 3 = -1, \qquad \kappa_{1122} = \frac{\mathbb{E}[\eta_x^2 \eta_y^2]}{\sigma_\eta^4} - 1 = -\frac{1}{3}$$
   * **Third Central Moment & Skewness (Exact $\mathcal{O}(\sigma^4)$ Linear-Quadratic Term + Gaussian-Approximate $\mathcal{O}(\sigma^6)$ Pure-Quadratic Term):**
     $$\mu_{3,k} = 3 \, \mathbf{s}_k^T \boldsymbol{\Sigma}_{\mathbf{z}} \mathbf{H}_k \boldsymbol{\Sigma}_{\mathbf{z}} \mathbf{s}_k + \frac{3}{2} \sum_{i \in \mathcal{I}_{\text{1D-unif}}} \kappa_i s_{k,i}^2 H_{k,ii} \sigma_{z_i}^4 + \frac{3}{2} \sum_{(i_x, i_y) \in \mathcal{I}_{\text{disk}}^{\text{unif}}} \left[ \kappa_{1111}\!\left(s_{k,i_x}^2 H_{k,i_x i_x} + s_{k,i_y}^2 H_{k,i_y i_y}\right) + \kappa_{1122}\!\left(s_{k,i_x}^2 H_{k,i_y i_y} + s_{k,i_y}^2 H_{k,i_x i_x} + 4 s_{k,i_x} s_{k,i_y} H_{k,i_x i_y}\right) \right] \sigma_\eta^4 + \text{Tr}\!\left((\mathbf{H}_k \boldsymbol{\Sigma}_{\mathbf{z}})^3\right)$$
     $$\text{Skewness } \gamma_{1,k} = \frac{\mu_{3,k}}{\sigma_{k,\text{2nd}}^3}$$
     *(Note: The $\mathcal{O}(\sigma^4)$ $L^2 Q$ skewness term is exact for all Gaussian, 1D-uniform, and 2D-disk variables. The $\mathcal{O}(H^3 \sigma^6)$ $\mathbb{E}[\tilde{Q}^3] \approx \text{Tr}((\mathbf{H}_k \boldsymbol{\Sigma}_{\mathbf{z}})^3)$ term is exact for Gaussian variables and marked as Gaussian-approximate when non-Gaussian variables dominate a zero-first-derivative measurement.)*

4. **Tier 4 — Vectorized Monte Carlo (`expm1` `float32` Perturbation Solve + `float64` Accumulation):**
   Evaluates $N_{\text{MC}}$ samples (default $10^6$) in chunked batches of $B = 50,000$ samples. Forward solver iterations run in `float32` using `expm1` incremental rotations $\Delta R$ and relative deviations $\Delta y_k^{(s)}$, which are then cast to `float64` and added to $y_{k,\text{nom}}$ for exact Welford/pairwise reduction.

#### Inverse Cost-Tolerance Optimization Formulation (`src/toljax/core/optimizer.py`)
Given $M$ manufacturable tolerance bands $\mathbf{t} = [t_1, \dots, t_M]^T \in [\mathbf{t}_{\min}, \mathbf{t}_{\max}]$, using the cached `float64` nominal sensitivity tensors $(\tilde{\mathbf{s}}_k, \tilde{\mathbf{H}}_k)$, we solve:

$$\min_{\mathbf{t} \in [\mathbf{t}_{\min}, \mathbf{t}_{\max}]} \quad C_{\text{total}}(\mathbf{t}) = \sum_{j=1}^M \left( a_j + \frac{b_j}{t_j^{\alpha_j}} \right)$$

$$\text{subject to:} \quad \begin{cases} g_{k,\text{upper}}(\mathbf{t}) = \text{USL}_k - \mu_{y_k}(\mathbf{t}) - 3 \, C_{pk,\text{target}} \, \sigma_{y_k}(\mathbf{t}) \ge 0 \\ g_{k,\text{lower}}(\mathbf{t}) = \mu_{y_k}(\mathbf{t}) - \text{LSL}_k - 3 \, C_{pk,\text{target}} \, \sigma_{y_k}(\mathbf{t}) \ge 0 \end{cases} \quad \forall k \in \{1, \dots, K\}$$

---

## 3. Declarative Input Schema (`assembly.yaml`)

All input files are validated via **Pydantic v2** (`src/toljax/schema/assembly_spec.py`). Below is the complete normative schema for a **kinematically determinate 6-DOF Plane-Pin-Slot assembly** with engagement-plane offsets and secondary fastener `jam_checks`:

```yaml
schema_version: "1.0"
assembly:
  name: "EV_Battery_Busbar_Weld_Stack"
  units: "mm"                                    # Allowed: "mm", "in", "um"
  default_process_sigma: 3.0                     # Maps tolerance half-bandwidth (t/2) to 3*sigma
  mmc_statistical_mode: "acceptance_truncated"   # Allowed: "acceptance_truncated", "fully_consumed"

parts:
  - id: "housing"
    fixed: true                                  # Ground reference part (q_0 = 0, T_0 = T_0^nom)
    step_file: "cad/housing.step"
    nominal_pose:
      pos: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 0.0]
    features:
      - id: "datum_A"
        type: "planar"
        step_face_tag: "FACE_DATUM_A"
        nominal_frame:
          pos: [0.0, 0.0, 10.0]
          rpy_deg: [0.0, 0.0, 0.0]
        bounds: {lx: 120.0, ly: 80.0}
        gdt:
          flatness: 0.05
        cost_model: {base_a: 2.0, scale_b: 0.15, exponent: 2.0, bounds: [0.01, 0.25]}

      - id: "pin_1"
        type: "cylinder_pin"
        nominal_frame:
          pos: [20.0, 20.0, 16.0]                # Midpoint of 12mm pin rising from z=10 to z=22
          rpy_deg: [0.0, 0.0, 0.0]
        size: {nom: 6.000, upper_dev: 0.000, lower_dev: -0.012} # ISO h6 pin (D_mid = 5.994)
        length: 12.0
        gdt:
          position: 0.08
          perpendicularity: 0.04                 # Orientation refinement t_orient < t_pos
          modifier: "MMC"
          datum_refs: ["datum_A"]
        cost_model: {base_a: 1.5, scale_b: 0.08, exponent: 2.0, bounds: [0.02, 0.30]}

      - id: "pin_2"
        type: "cylinder_pin"
        nominal_frame:
          pos: [100.0, 20.0, 16.0]
          rpy_deg: [0.0, 0.0, 0.0]
        size: {nom: 6.000, upper_dev: 0.000, lower_dev: -0.012}
        length: 12.0
        gdt:
          position: 0.08
          perpendicularity: 0.04
          modifier: "MMC"
          datum_refs: ["datum_A", "pin_1"]
        cost_model: {base_a: 1.5, scale_b: 0.08, exponent: 2.0, bounds: [0.02, 0.30]}

  - id: "carrier"
    fixed: false
    nominal_pose:
      pos: [0.0, 0.0, 10.0]
      rpy_deg: [0.0, 0.0, 0.0]
    features:
      - id: "bottom_pad"
        type: "planar"
        nominal_frame: {pos: [0.0, 0.0, 0.0], rpy_deg: [0.0, 0.0, 0.0]}
        bounds: {lx: 120.0, ly: 80.0}
        gdt: {profile: 0.10, flatness: 0.04}
        cost_model: {base_a: 1.0, scale_b: 0.10, exponent: 2.0, bounds: [0.02, 0.40]}

      - id: "hole_1"
        type: "cylinder_hole"
        nominal_frame: {pos: [20.0, 20.0, 4.0], rpy_deg: [0.0, 0.0, 0.0]} # Midpoint of 8mm hole
        size: {nom: 6.050, upper_dev: 0.025, lower_dev: 0.000}            # ISO H7 hole (D_mid = 6.0625)
        length: 8.0
        gdt:
          position: 0.10
          perpendicularity: 0.05
          modifier: "MMC"
          datum_refs: ["bottom_pad"]
        cost_model: {base_a: 1.0, scale_b: 0.12, exponent: 2.0, bounds: [0.02, 0.40]}

      - id: "slot_2"
        type: "slot_width"
        nominal_frame: {pos: [100.0, 20.0, 4.0], rpy_deg: [0.0, 0.0, 0.0]}
        size: {nom: 6.050, upper_dev: 0.025, lower_dev: 0.000}
        bounds: {lx: 10.0, lz: 8.0}
        gdt:
          position: 0.10
          modifier: "MMC"
          datum_refs: ["bottom_pad", "hole_1"]
        cost_model: {base_a: 1.0, scale_b: 0.10, exponent: 2.0, bounds: [0.02, 0.40]}

      - id: "weld_pad"
        type: "point"
        nominal_frame: {pos: [95.0, 45.0, 25.0], rpy_deg: [0.0, 0.0, 0.0]}
        gdt: {profile: 0.06}
        cost_model: {base_a: 0.8, scale_b: 0.06, exponent: 2.0, bounds: [0.02, 0.25]}

mates:
  - id: "primary_seat"
    type: "planar_coincident"      # Locks 3 DOFs: [-dR_23, dR_13, p_z] = 0
    source: "housing.datum_A"
    target: "carrier.bottom_pad"

  - id: "primary_pin"
    type: "parallel_pin_hole"      # Locks 2 DOFs: [p_x - C_r,eff*eta_x, p_y - C_r,eff*eta_y] = 0
    source: "housing.pin_1"
    target: "carrier.hole_1"
    contact_offset_z: {source: -2.0, target: 0.0} # Mid-engagement plane z = 14.0 (L_e = 8.0 mm)
    kinematic_mode: "float_random"

  - id: "clocking_slot"
    type: "clocking_pin_slot"      # Locks 1 DOF: [p_y - C_w,eff*eta_y] = 0 at z = 14.0
    source: "housing.pin_2"
    target: "carrier.slot_2"
    contact_offset_z: {source: -2.0, target: 0.0}
    kinematic_mode: "float_random"

jam_checks:
  - id: "pin_2_slot_clearance_fit"
    source: "housing.pin_2"
    target: "carrier.slot_2"
    axis: "y"

measurements:
  - id: "laser_focal_z"
    description: "Vertical Z distance from housing datum A plane to carrier weld target"
    type: "point_to_plane_distance"
    axis: [0.0, 0.0, 1.0]
    source: "housing.datum_A"
    target: "carrier.weld_pad"
    limits: {lsl: 24.850, usl: 25.150}
```

---

## 4. Module-by-Module Software Architecture & Hardware Execution Model

```text
toljax/
├── pyproject.toml
├── README.md
├── spec.md
├── src/
│   └── toljax/
│       ├── __init__.py            # Public Python API exports (load, analyze, optimize)
│       ├── core/
│       │   ├── se3.py             # Precision-aware Taylor-safe SO(3)/SE(3) & expm1 ops
│       │   ├── torsor.py          # Symmetric 3-disk zones, 3-cone WC SOCP, ISO midpoints & MMC
│       │   ├── mates.py           # Direct geometric (dR, p) residuals, tilt-binding & jam margins
│       │   ├── graph.py           # Compiles AssemblySpec + dimensionless SVD rank check into PyTree
│       │   ├── solver.py          # Batched Newton/LM solver + recursive @jax.custom_jvp IFT
│       │   ├── metrics.py         # WC SOCP, Linear RSS, SOTA (+disk cumulants), float32/float64 MC
│       │   └── optimizer.py       # Differentiable inverse cost-tolerance synthesis
│       ├── schema/
│       │   └── assembly_spec.py   # Pydantic v2 validation, unit normalization & mate rank check
│       ├── cad/
│       │   ├── step_inspector.py  # OCP/build123d STEP face centroid/normal extractor
│       │   └── glb_exporter.py    # Exports deflected worst-case mode shapes to glTF
│       └── cli/
│           ├── app.py             # Typer CLI entrypoints (`run`, `optimize`, `inspect`)
│           └── html_report.py     # Self-contained HTML/Three.js/Chart.js report builder
└── tests/
    ├── test_se3_autodiff.py       # Gate 1A: jacrev & hessian at theta = 0.0 (no NaNs, < 1e-10)
    ├── test_plane_pin_slot_ref.py # Gate 1B: Closed-form trigonometric Plane-Pin-Slot S & H parity
    ├── test_implicit_vjp.py       # Gate 2: IFT Jacobian & Hessian vs FD (< 1e-10 / < 1e-7)
    ├── test_mmc_coupled_zones.py  # Verifies ASME Y14.5.1 3-cone SOCP & symmetric 3-disk bounds
    └── benchmarks/
        ├── bench_optical_tilt.py  # Gate 3: Second-order cosine foreshortening & disk cumulants
        ├── bench_perf_6part.py    # Gate 4: Hardware timing & float32/float64 relative parity (< 1e-4 sigma)
        └── bench_sota_clevis.py   # Gate 5: Glancy & Chase SOTA / Fischer / Drake parity
```

---

## 5. CLI & Output Specification

### 5.1 CLI Commands

```bash
# 1. Execute full 4-tier tolerance analysis (WC, Linear RSS, 2nd-Order SOTA, Monte Carlo)
toljax run <assembly.yaml> \
  [--mc-samples 1000000] \
  [--mc-mode full|surrogate] \
  [--seed 42] \
  [--json-out results.json] \
  [--html-out report.html]

# 2. Run inverse cost-tolerance optimization
toljax optimize <assembly.yaml> \
  --target-cpk 1.33 \
  [--out-yaml optimized_assembly.yaml] \
  [--json-out opt_results.json]

# 3. Extract nominal feature frames & face tags from a STEP file into YAML boilerplate
toljax inspect <part.step> [--out-yaml part_features.yaml]
```

### 5.2 Structured JSON Output Contract (`results.json`)

```json
{
  "assembly_name": "EV_Battery_Busbar_Weld_Stack",
  "kinematic_status": {
    "movable_parts": 1,
    "active_dofs": 6,
    "constrained_dofs": 6,
    "jacobian_rank": 6,
    "determinate": true,
    "jam_dpmo": 0.0,
    "sota_conditioning_warning": false
  },
  "execution_time_ms": {
    "jit_compile_ms": 185.0,
    "jacobian_hessian_fp64_ms": 2.4,
    "monte_carlo_1m_fp32_ms": 148.2
  },
  "measurements": [
    {
      "id": "laser_focal_z",
      "nominal": 25.0000,
      "lsl": 24.8500,
      "usl": 25.1500,
      "conditioned_on_fit": true,
      "worst_case": {
        "min": 24.8392,
        "max": 25.1608,
        "half_band": 0.1608,
        "status": "FAIL"
      },
      "linear_rss": {
        "mean": 25.0000,
        "sigma": 0.0324,
        "three_sigma": 0.0972,
        "cp": 1.543,
        "cpk": 1.543
      },
      "second_order_sota": {
        "mean_shift": -0.0004,
        "shifted_mean": 24.9996,
        "sigma": 0.0325,
        "skewness": -0.038,
        "skewness_ pure_quadratic_approx": "gaussian",
        "cpk": 1.534
      },
      "monte_carlo": {
        "samples_total": 1000000,
        "samples_assembled": 1000000,
        "mean": 24.9996,
        "sigma": 0.0325,
        "p00135": 24.9018,
        "p99865": 25.0970,
        "spec_dpmo": 4.0,
        "jam_dpmo": 0.0,
        "total_defect_dpmo": 4.0,
        "yield_pct": 99.9996
      },
      "pareto_contributions": [
        {
          "feature_dof": "housing.datum_A.theta_y",
          "sensitivity_dy_dz": -0.0396,
          "variance_pct": 55.28
        },
        {
          "feature_dof": "carrier.bottom_pad.delta_z",
          "sensitivity_dy_dz": 0.0300,
          "variance_pct": 31.76
        },
        {
          "feature_dof": "carrier.weld_pad.delta_z",
          "sensitivity_dy_dz": 0.0300,
          "variance_pct": 12.96
        }
      ]
    }
  ]
}
```

---

## 6. Verification, Falsification & Benchmark Suite

Every pull request must pass five numerical falsification gates before merge:

1. **Gate 1 — Zero-Angle Autodiff & Closed-Form Plane-Pin-Slot Analytic Reference Gate (`test_se3_autodiff.py`, `test_plane_pin_slot_ref.py`):**
   * Assert that `jax.jacrev(se3_exp)(jnp.zeros(6))`, `jax.hessian(se3_exp)(jnp.zeros(6))`, `jax.jacrev(se3_log)(jnp.eye(4))`, and `jax.hessian(se3_log)(jnp.eye(4))` in `float64` contain zero `NaN`s and match central finite differences to $< 10^{-10}$ Frobenius norm error.
   * Compare the solver's implicit Jacobian $\mathbf{S}$ and Hessian $\mathbf{H}$ on a 3D Plane-Pin-Slot assembly against the **exact closed-form trigonometric solution** ($\theta_z = \arcsin\frac{\Delta y_{\text{slot}} - \Delta y_{\text{pin}}}{D_{12}}$), asserting relative error $\lVert \mathbf{S}_{\text{AD}} - \mathbf{S}_{\text{exact}} \rVert_\infty \le 10^{-12} \lVert \mathbf{S}_{\text{exact}} \rVert_\infty$ and $\lVert \mathbf{H}_{\text{AD}} - \mathbf{H}_{\text{exact}} \rVert_\infty \le 10^{-11} \lVert \mathbf{H}_{\text{exact}} \rVert_\infty$.

2. **Gate 2 — Implicit Function Theorem Adjoint & Hessian Gate (`test_implicit_vjp.py`):**
   * On a closed-loop 3D determinate assembly, compare the analytical Jacobian $\mathbf{S}$ and Hessian $\mathbf{H}$ from recursive `@jax.custom_jvp` against central finite differences of an unrolled 25-step Newton-Raphson solver.
   * **Pass Criterion:** $\lVert \mathbf{S}_{\text{AD}} - \mathbf{S}_{\text{FD}} \rVert_\infty < 10^{-10} \lVert \mathbf{S}_{\text{FD}} \rVert_\infty$ and $\lVert \mathbf{H}_{\text{AD}} - \mathbf{H}_{\text{FD}} \rVert_\infty < 10^{-7} \lVert \mathbf{H}_{\text{FD}} \rVert_\infty$.

3. **Gate 3 — Second-Order Geometric Shift & Disk Cumulant Gate (`bench_optical_tilt.py`):**
   * **Part A (Gaussian Tilt Foreshortening):** For a rigid rod of length $L = 100\text{ mm}$ subject to angular tilt $\theta_x, \theta_y \sim \mathcal{N}(0, \sigma_\theta^2)$, verify that SOTA predicts $\Delta \mu_z = -L \sigma_\theta^2$ and $\sigma_z = L \sigma_\theta^2$ analytically, matching $10^6$-sample Monte Carlo within $0.2\%$.
   * **Part B (Uniform Disk Clearance Float):** On a quadratic measurement with off-diagonal coupling $H_{12} \ne 0$ driven by $\boldsymbol{\eta} \sim \mathcal{U}(\mathcal{B}_2(1))$, verify that the fourth-cumulant variance formula (including the $(2 H_{11} H_{22} + 4 H_{12}^2)\kappa_{1122}$ cross-term) and $\mathcal{O}(\sigma^4)$ skewness formula match $10^6$-sample Monte Carlo within $0.5\%$.

4. **Gate 4 — Dual-Precision Performance & `expm1` `float32`/`float64` Parity Gate (`bench_perf_6part.py`):**
   * **Scale-Relative Precision Parity:** On a $10^4$-sample subset of the 6-part assembly, assert that the `expm1` `float32` perturbation solver with `float64` Welford/chunked accumulation agrees with full `float64` Monte Carlo to within $< 10^{-4} \sigma_y$ on both mean and standard deviation.
   * **Throughput:** On a 6-part, 24-DOF determinate assembly, warm-started `float64` CPU extraction of $\mathbf{S} + \mathbf{H}$ + inverse optimization must complete in **$< 25\text{ ms}$**, and $10^6$-sample Monte Carlo must complete in **$< 50\text{ ms}$** (surrogate) and **$< 250\text{ ms}$** (`float32` GPU/Metal nonlinear solver).

5. **Gate 5 — Published Literature Parity Gate (`bench_sota_clevis.py`):**
   * Reproduce published 3D assembly stack-up benchmarks from **Glancy & Chase (1999)** (*SOTA*), **Gao, Chase & Magleby (1998)** (*DLM vs. MC*), and worked 3D cases from **Fischer (*Mechanical Tolerance Stackup and Analysis*)** and **Drake (*Dimensioning and Tolerancing Handbook*)**, matching published means, standard deviations, and defect rates within $0.5\%$.
