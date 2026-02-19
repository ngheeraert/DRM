# MEDRM — Master-Equation integration for a driven/damped Rabi model

This repository contains a Python codebase to **time-integrate a Lindblad master equation**
for a cavity–qubit (Rabi-type) model with:

- a **coherent drive** applied via a time-dependent Hamiltonian term, and
- **cavity decay** (single Lindblad jump operator).

An example Jupyter notebook is included to show the typical workflow: build the `system`, choose
parameters, evolve `ρ(t)`, then post-process observables and save outputs.

---

## Contents

- `system_class.py` — main simulation engine (`system` class): operator construction, Hamiltonian,
  initial states, Lindblad RHS, and time integration.
- `functions.py` — analysis helpers (partial trace, coherent states, Wigner/Q-function helpers).
- `qfunction_plot.py` — standalone script to plot a previously saved Husimi-Q array from `data/`.
- `*.ipynb` — example run notebook (imports the code as a package and demonstrates usage).

> In the example notebook, the package is imported as `import medrm`.  
> If you are using these modules directly (without a package wrapper), see **Packaging** below.

---

## Physics model (as implemented)

The code integrates a density matrix `ρ(t)` using the RHS implemented in `system.ode_RHS()`:

\[
\dot\rho = -i[H,\rho] \,\;\; -i\, A_d\sin(\omega_d t)\,[H_{drive},\rho]
\; + \; \gamma\left(a\rho a^\dagger - \tfrac{1}{2}\{a^\dagger a,\rho\}\right).
\]

- `H = H_qb + H_cav + H_coupling`
- drive term uses `H_drive = i(a - a†)` (a cavity quadrature) and a sinusoidal modulation
- decay uses jump operator `a` with rate `gamma` (often called `κ` in cQED notation)

**Units:** all frequencies are treated as angular frequencies (rad/time-unit).  
In the example notebook, parameters are set as `(value_in_GHz) * 2π`, and time is in the
corresponding reciprocal unit.

---

## Requirements

Core dependencies:

- Python 3.x
- `numpy`
- `scipy`
- `matplotlib` (for plotting)

Optional (only needed for certain phase-space routines):

- `qutip` (used in the Wigner/displacement-based methods)

Jupyter usage:

- `jupyter` / `notebook` / `jupyterlab`

Install quickly (typical):

```bash
pip install numpy scipy matplotlib qutip jupyter
```

---

## Quick start (from the example notebook)

The example notebook does roughly:

1) import the package  
2) choose parameters  
3) build a `system`  
4) set initial state  
5) call `time_evolve(times)`  
6) compute observables and save to `data/`

Minimal script equivalent:

```python
import numpy as np
import medrm  # or import system from system_class if not packaged

# Example parameters (angular frequencies)
g     = 0.25 * 2*np.pi
wc    = 7.5  * 2*np.pi
Ad    = 0.2  * 2*np.pi
wd    = 7.5  * 2*np.pi
kappa = 0.1  * 2*np.pi
tmax  = 13.0

s = medrm.system(
    g=g, w=wc, Ad=Ad, wd=wd, gamma=kappa,
    coupling_type="11",
    cavity_dim=20, qubit_dim=2,
    dvice="TRSM3",
    atol=1e-8, rtol=1e-6, max_step=1e-2,
    verbose=True,
)

# Initial state: qubit excited, cavity vacuum coherent state
s.set_initial_qb_state([1])
s.set_initial_cs_state(alpha=0)
s.initialise_density_matrix()

nsteps = 3000
times  = np.linspace(0.0, tmax, nsteps)
rhos   = s.time_evolve(times, verbose=False)

# Example observable: ⟨a†a⟩
n_arr = np.array([np.real(np.trace(s.na @ rho)) for rho in rhos])
np.savetxt("data/PHOTONS_" + s.paramchar(tmax) + ".txt", n_arr)
```

---

## Outputs and file naming

The `system.paramchar(tmax)` helper builds a **parameter string** used as an identifier in filenames.
It encodes (among others) `tmax`, Hilbert space sizes (`Nq`, `Nc`), drive amplitude (`amp`),
decay rate (`kappa`), device frequencies, coupling `g`, drive `wd`, integrator max step (`ms`),
initial qubit state (`qb[...]`), coupling type, and device name.

Typical convention used in the notebook/scripts:

- `data/` — saved arrays (time grids, observables, phase-space distributions)
- `figures/` — plots exported as PDF

Create these folders if they don’t exist:

```bash
mkdir -p data figures
```

---

## Key API (system class)

### Construction

```python
s = system(
    w01=-1.0,        # qubit frequency; may be overridden by device files depending on dvice/coupling_type
    w=1.0,           # cavity frequency
    g=0.1,           # coupling strength
    wd=0.0,          # drive frequency
    gamma=0.0,       # cavity decay rate (κ)
    Ad=0.0,          # drive amplitude
    cavity_dim=3,    # cavity Hilbert dimension
    qubit_dim=2,     # qubit Hilbert dimension (2 = TLS, >2 = multi-level)
    dvice="TRSM1",   # device label used to select parameter files for some couplings
    coupling_type="00",
    atol=1e-8, rtol=1e-6, max_step=1e-2,
    verbose=True
)
```

### Initial state helpers

- `set_initial_qb_state([level_index])` — pick a bare qubit level (e.g., `[0]` ground, `[1]` excited).
- `set_initial_photon_state(n)` — set a Fock state `|n⟩` in the cavity.
- `set_initial_cs_state(alpha)` — set a cavity coherent state `|α⟩`.
- `initialise_density_matrix()` — builds `ρ(0)` from the chosen initial state components.

### Time evolution

- `time_evolve(times, verbose=True)`  
  Integrates from `times[0]` to `times[-1]` and returns a list of density matrices `[ρ(t0), ρ(t1), ...]`.

Integrator notes:
- Uses SciPy’s `complex_ode` with `dopri5` (tolerances controlled by `atol`, `rtol`).
- `max_step` is forwarded to the integrator to control internal step size.

### Diagnostics / analysis

- `expect(rho, op)` — expectation value helper (if used in your version).
- `renyi_entropy(...)`, `calc_fidelity(...)` — utilities for state diagnostics.
- `my_wigner(...)` and helpers — phase-space visualizations (requires `qutip`).

---

## Phase-space / Q-function plotting

- `functions.py` contains utilities like:
  - `partial_trace(rho, rho_dims, keep, optimize=False)`
  - `coherent_state(alpha, dim)`
  - `q_function(...)`, `characteristic_function(...)`

- `qfunction_plot.py` is a **standalone plotting script** that expects something like:

  - `data/qfunction_<filename>.d`

  and will save figures to `figures/`.

If you change the save naming scheme, update the `filename = ...` inside `qfunction_plot.py`
or refactor it to accept a CLI argument.

---

## Device parameter files (important for some couplings)

For some `dvice` / `coupling_type` combinations, `system_class.py` loads external text files such as:

- `../MPOL_DCT_FOR/qubit_params/...`

These are **not part of this repository** (in the files provided here). If you run with
`coupling_type="11"` and `dvice="TRSM2"/"TRSM3"/"QUTR2"`, you must ensure those files exist at the
expected relative paths, or modify the code to point to your local parameter directory.

If you want a purely self-contained “toy” run, use `dvice="QUBIT"` and a coupling type that does not
require external level tables.

---

## Packaging (if you want `import medrm`)

If your notebook uses `import medrm`, the simplest layout is:

```
medrm/
  __init__.py
  system_class.py
  functions.py
  qfunction_plot.py
notebooks/
  example.ipynb
```

Example `medrm/__init__.py`:

```python
from .system_class import system
from .functions import (
    partial_trace, coherent_state, my_wigner, q_function,
    characteristic_function, kronecker_delta,
    factorial_approx, sqrt_factorial_approx,
)
```

Then from the repository root you can do:

```bash
export PYTHONPATH=.
# or install editable with a minimal pyproject / setup.cfg if you prefer
```

---

## Troubleshooting

- **`ModuleNotFoundError: qutip`**  
  Install `qutip` or avoid calling Wigner/Q-function utilities that require it.

- **Missing device parameter files**  
  If you use `coupling_type="11"` with TRSM/QUTR devices, make sure the external parameter files
  exist at the hard-coded relative paths, or update the path logic.

- **Integration is slow / memory-heavy**  
  Returning and storing every `ρ(t)` can be expensive for large Hilbert spaces. If you only need a few
  observables, consider post-processing on the fly (or thinning the `times` array).
