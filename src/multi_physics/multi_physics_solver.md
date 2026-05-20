# Electromechanical MEMS beam–electrostatics coupling (2D air domain + 4-mode mechanics)

This note documents the mathematical model implemented in the coupled solver:

- **Electrostatics**: FEM on a **2D air domain** with Dirichlet boundary conditions on the electrodes.
- **Mechanics**: **4-mode reduced-order (modal) model** for the moving cantilever electrode.
- **Coupling**: Maxwell-stress traction computed from the electrostatic field and **projected onto the modal basis** to obtain generalized forces.
- **Geometry update**: at each time step the deformed moving electrode boundary is regenerated from modal coordinates and **the air mesh is remeshed** (Gmsh).

## 1. Geometry and kinematics

### 1.1 Reference geometry (planar 2D model)
Two parallel electrodes (beams/plates) are embedded in a circular outer boundary representing air. In the solver, only the **air region** is meshed and used for the electrostatic PDE.

Typical physical tags in the Gmsh `.geo` template:

- **Line tags**
  - `10`: `force_segment` — portion of moving electrode boundary where forces are evaluated
  - `11`: `upper_plate` — remaining moving electrode boundary
  - `12`: `lower_plate` — fixed electrode boundary
  - `20`: `boundary` — outer circular boundary
- **Surface tags**
  - air: a single surface after Boolean difference (air = outer circle minus electrode holes)

> The electrodes are treated as **perfect conductors**: they do not belong to the electrostatic computational domain; their boundaries impose Dirichlet conditions.

### 1.2 Modal parametrization of the moving electrode

Let $x$ denote the coordinate along the beam length and $y$ the transverse direction (gap direction). The moving electrode bottom boundary is parametrized by the first four cantilever-like mode shapes:

$$
y_b(x,t) = y_0 + \sum_{i=1}^{4} q_i(t)\,\varphi_i(x),
$$

where:
- $y_0$ is the undeformed bottom boundary level (from the template parameters),
- $q_i(t)$ are the **modal coordinates** (units of length),
- $\varphi_i(x)$ are mode shapes (dimensionless in this implementation).

The `.geo` template uses Euler–Bernoulli cantilever mode formulas. A common representation is:

$$
\varphi_i(\xi) = \cosh(\beta_i\xi)-\cos(\beta_i\xi)
- C_i\left[\sinh(\beta_i\xi)-\sin(\beta_i\xi)\right],
\quad \xi\in[0,L],
$$

with:

$$
C_i=\frac{\cosh(\beta_iL)+\cos(\beta_iL)}{\sinh(\beta_iL)+\sin(\beta_iL)}.
$$

For a cantilever, the wavenumbers are $\beta_i = \lambda_i/L$ with $\lambda_i$ the classical roots:

$$
\lambda_1\approx 1.8751,\quad
\lambda_2\approx 4.6941,\quad
\lambda_3\approx 7.8548,\quad
\lambda_4\approx 10.9955.
$$

The deformed top boundary is then:

$$
y_t(x,t) = y_b(x,t) + h,
$$

with $h$ the electrode thickness in the 2D section.


## 2. Electrostatics in the air domain

### 2.1 Governing equation

In the air domain $\Omega(t)\subset\mathbb{R}^2$ (time-dependent due to remeshing), the electrostatic potential $\phi$ satisfies:

$$
-\nabla\cdot\left(\varepsilon\,\nabla \phi\right) = 0 \quad \text{in } \Omega(t),
$$

where $\varepsilon=\varepsilon_0\varepsilon_r$ is the permittivity. In the implementation:
- geometry is converted to SI (meters),
- $\varepsilon_0 = 8.8541878128\times 10^{-12}\,\mathrm{F/m}$,
- $\varepsilon_r$ is user-specified (default $1$ for air).

### 2.2 Boundary conditions

Let the electrode boundaries be $\Gamma_u(t)$ (moving/upper conductor) and $\Gamma_\ell$ (fixed/lower conductor), with outer boundary $\Gamma_o$.

Dirichlet conditions are imposed on the conductors:

$$
\phi = V_u(t)\ \text{on }\Gamma_u(t),\qquad
\phi = V_\ell(t)\ \text{on }\Gamma_\ell.
$$

Optional outer boundary condition:
- Dirichlet: $\phi=V_o$ on $\Gamma_o$, or
- Natural Neumann (default if disabled): $\varepsilon\nabla\phi\cdot n = 0$ on $\Gamma_o$.

A typical driving voltage uses:

$$
V_\ell(t)=V_{\mathrm{dc}} + V_{\mathrm{ac}}\sin(2\pi f t),\qquad V_u(t)=0.
$$

### 2.3 Weak form (FEM)

For $V_h$ (continuous Lagrange $P_1$), find $\phi_h\in V_h$ satisfying Dirichlet constraints such that:

$$
\int_{\Omega(t)} \varepsilon\,\nabla \phi_h\cdot\nabla v_h\,dx = 0
\quad \forall v_h\in V_h^0.
$$


## 3. Electrostatic traction via Maxwell stress

### 3.1 Electric field

$$
\mathbf{E} = -\nabla \phi.
$$

In post-processing, $\mathbf{E}$ is projected to a DG0 vector space for robust diagnostics.

### 3.2 Maxwell stress tensor (electrostatics)

In a linear dielectric:

$$
\mathbf{T} = \varepsilon\left(\mathbf{E}\otimes\mathbf{E} - \frac{1}{2}|\mathbf{E}|^2\mathbf{I}\right).
$$

### 3.3 Traction on the conductor

Let $n$ be the **outward unit normal of the air domain** on the conductor boundary (for a hole boundary this points into the hole). The traction acting on the air is:

$$
\mathbf{t}_{\mathrm{air}} = \mathbf{T}\,n.
$$

By action–reaction, the force on the conductor is:

$$
\mathbf{t}_{\mathrm{beam}} = -\mathbf{T}\,n.
$$

This sign convention is essential when the conductor is represented as a hole in the air mesh.

### 3.4 Resultant force scaling for 2D model

The 2D computation yields traction per unit out-of-plane thickness. To obtain physical force (Newtons), the traction integral is multiplied by the **out-of-plane thickness** $b$ (meters):

$$
\mathbf{F} = b\int_{\Gamma}\mathbf{t}_{\mathrm{beam}}\,ds.
$$

## 4. Modal reduction and generalized forces

### 4.1 Modal mechanical model

The moving beam is represented by modal coordinates $q_i(t)$, $i=1,\dots,4$. A diagonal modal model is assumed:

$$
m_i\ddot q_i(t) + c_i \dot q_i(t) + k_i q_i(t) = F_i(t),
$$

where:
- $m_i$ is modal mass [kg],
- $c_i$ is modal damping [kg/s],
- $k_i$ is modal stiffness [N/m],
- $F_i$ is generalized force [N].

Often the parameters are specified via natural frequencies $\omega_i$ and damping ratios $\zeta_i$:

$$
k_i = m_i\omega_i^2,\qquad c_i = 2\zeta_i\omega_i m_i.
$$

### 4.2 Generalized forces from Maxwell traction

The modal shape vector is taken as transverse-only:

$$
\boldsymbol{\psi}_i(x) =
\begin{bmatrix}
0\\
\varphi_i(x)
\end{bmatrix}.
$$

Generalized force is computed by virtual work:

$$
F_i(t) = b\int_{\Gamma_{10}(t)} \mathbf{t}_{\mathrm{beam}}(s,t)\cdot \boldsymbol{\psi}_i(s)\,ds,
$$

where $\Gamma_{10}(t)$ is the boundary segment marked by physical tag 10 (`force_segment`). The same analytic $\varphi_i$ used for geometry is used for force projection, avoiding mesh-to-mesh mapping.


## 5. Time integration (Newmark average acceleration)

A Newmark scheme with $\beta=1/4$, $\gamma=1/2$ is applied independently to each modal DOF.

Given $q^n, \dot q^n, \ddot q^n$, define predictors:

$$
q_{\mathrm{pred}} = q^n + \Delta t\,\dot q^n + \frac{\Delta t^2}{2}(1-2\beta)\ddot q^n,
$$

$$
\dot q_{\mathrm{pred}} = \dot q^n + \Delta t(1-\gamma)\ddot q^n.
$$

Solve for $\ddot q^{n+1}$:

$$
\ddot q^{n+1} =
\frac{
F^{n+1} - c\,\dot q_{\mathrm{pred}} - k\,q_{\mathrm{pred}}
}{
m + \gamma\Delta t\,c + \beta\Delta t^2\,k
}.
$$

Correct:

$$
q^{n+1} = q_{\mathrm{pred}} + \beta\Delta t^2 \ddot q^{n+1},\qquad
\dot q^{n+1} = \dot q_{\mathrm{pred}} + \gamma\Delta t \ddot q^{n+1}.
$$

In the coupled implementation, $F^{n+1}$ is approximated by the force computed on the mesh generated by the current modal coordinates (explicit/partitioned coupling). More accurate options include fixed-point iterations per time step.


## 6. Coupling algorithm (partitioned, remeshing each step)

For time steps $n=0,\dots,N-1$:

1. **Geometry update**: construct moving electrode boundary from $\{q_i^n\}$ and write `.geo`.
2. **Remesh**: run Gmsh to generate air mesh $\Omega(t_n)$.
3. **Electrostatics solve**: compute $\phi_h^n$ with Dirichlet BCs on electrode boundaries.
4. **Maxwell stress**: compute traction $\mathbf{t}_{\mathrm{beam}}^n$ on $\Gamma_{10}$.
5. **Modal projection**: compute generalized forces $F_i^n$.
6. **Time integration**: update $q_i^{n+1}, \dot q_i^{n+1}, \ddot q_i^{n+1}$ with Newmark.
7. **Output**: write $\phi_h^n$ (and optionally $\mathbf{E}^n$) to a ParaView time series; store modal histories.


## 7. Units and scaling

- The `.geo` template is written in microns. The solver converts mesh coordinates to meters immediately after reading.
- Modal coordinates $q_i$ are stored internally in meters and converted to microns only when substituting `__COEFFi__` placeholders.
- The force integral produces Newtons only if:
  - coordinates are meters,
  - $\varepsilon_0$ is in F/m,
  - the out-of-plane thickness $b$ is in meters.


## 8. Diagnostic quantities (recommended)

At each step:

- Potential bounds: $\min\phi_h$, $\max\phi_h$
- Field scale: $\max|\mathbf{E}|$
- Electrostatic energy:

$$
W = \frac12 \varepsilon \int_{\Omega}|\nabla\phi|^2\,dx
$$

- Capacitance-like estimate (two-conductor):

$$
C \approx \frac{2W}{(V_\ell - V_u)^2}
$$

- Mesh sanity: number of nodes/cells; presence of facet tags 10/12.


## 9. Modelling assumptions

- Quasi-static electrostatics (no displacement currents in time).
- Perfectly conducting electrodes with prescribed potentials.
- Air treated as linear dielectric (constant $\varepsilon_r$).
- Modal mechanics assumes a linear basis and diagonal modal dynamics (no geometric nonlinearity, no contact/pull-in).
- Coupling is partitioned with remeshing; stability near pull-in may require smaller $\Delta t$ and/or iterative coupling.


## To launch the code
Case of cantilever with small deformations
```bash
python -m src.multi_physics.solver --nmodes 2 --template-geo geometries/cantilever1.geo --dt 1e-5 --nsteps 40 --Vdc 0 --Vac 5 --freq 2.5e3 --Vupper 0 --Vouter 0 --omega 6.3e5 3.9e6 1.1e7 2.1e7 --mass 1e-12 1e-12 1e-12 1e-12 --zeta 0.01 0.01 0.01 0.01 --print-every 1 --fail-fast --derivative-nn-path models/derivative1.keras --postprocessing-step 5 --potential-nn-path models/potential1.keras --no-outer-bc
```
Case of cantilever with big deformations
```bash
python -m src.multi_physics.solver --nmodes 2 --template-geo geometries/cantilever2.geo --dt 5e-6 --nsteps 80 --Vdc 0 --Vac 230 --freq 2.5e3 --Vupper 0 --Vouter 0 --omega 6.3e5 3.9e6 1.1e7 2.1e7 --mass 1e-12 1e-12 1e-12 1e-12 --zeta 0.01 0.01 0.01 0.01 --print-every 1 --fail-fast
```

### References:

1. Forces: https://www.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcfxsg&utm_source=chatgpt.com
2. Modal Projection:
    - https://lib.physcon.ru/file?id=c3839b099d0c
    - https://amsdottorato.unibo.it/id/eprint/461/1/2007_03_13_Tesi_Dottorato_Laura_Del_Tin.pdf