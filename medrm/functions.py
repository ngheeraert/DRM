"""
functions.py
============

Small numerical/linear-algebra utilities used by the driven, lossy Rabi-model
master-equation integrator.

This module is intentionally "plain NumPy/SciPy" so it can be imported from
scripts, notebooks, or a lightweight package layout without pulling in heavy
dependencies.  A few routines optionally use QuTiP (only when called).

Main physics-facing helpers
---------------------------
- partial_trace: Reduce a composite density matrix to a subsystem.
- coherent_state: Build a truncated coherent-state ket |alpha> in the Fock basis.
- q_function: Husimi-Q function Q(alpha) = <alpha| rho |alpha> on a phase-space grid.
- my_wigner: Wigner function via displaced-parity expectation (requires QuTiP).

Notes on conventions
--------------------
- Matrices are dense NumPy arrays of dtype complex128 unless otherwise stated.
- Tensor-product ordering is not enforced here; callers must be consistent about
  their basis ordering when using partial_trace.
"""

import sys  # used for error exits in some helpers
from scipy.integrate import quad  # (currently only used in commented-out code)
from scipy.linalg import expm
import numpy as np
from math import factorial
import sys  # duplicate import kept to preserve original structure
from decimal import Decimal


def partial_trace(rho, rho_dims, keep, optimize=False):
	"""
	Compute a partial trace of a density matrix over selected subsystems.

	Given a composite Hilbert space H = H_0 ⊗ H_1 ⊗ ... ⊗ H_{N-1} with dimensions
	`rho_dims = [dim_0, dim_1, ..., dim_{N-1}]`, this returns the reduced density
	matrix on the subspace indexed by `keep`:

		ho_keep = Tr_{traced}(ho)

	Parameters
	----------
	rho : (D, D) complex array
		Density matrix in the full space (D = prod(rho_dims)).
	rho_dims : sequence[int]
		Dimensions of each tensor factor.
	keep : sequence[int]
		Subsystem indices to keep (0-based).  Example: if the full space is
		A ⊗ B ⊗ C ⊗ D and you want Tr_{B,D}, then keep = [0, 2].
	optimize : bool
		Forwarded to `np.einsum(..., optimize=...)` for potential speedups.

	Returns
	-------
	rho_a : (D_keep, D_keep) complex array
		Reduced density matrix on the kept subsystems.

	Implementation details
	----------------------
	We reshape rho into a 2N-dimensional tensor (one index per subsystem for the
	"bra" and one for the "ket") and use einsum to contract the traced indices.
	This is a standard, efficient approach for dense arrays.
	"""
	# Normalize/validate indices and dimensions
	keep = np.asarray(keep)
	dims = np.array(rho_dims)
	Ndim = dims.size
	Nkeep = np.prod(dims[keep])

	# Build einsum index lists:
	# - idx1 labels the "bra" indices (0..Ndim-1)
	# - idx2 labels the "ket" indices; for kept subsystems we shift their label
	#   by +Ndim so they remain uncontracted, while traced subsystems share the
	#   same label as in idx1 to trigger contraction.
	idx1 = [i for i in range(Ndim)]
	idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]

	# Reshape rho into tensor with shape (dim0, dim1, ..., dimN-1, dim0, ..., dimN-1)
	rho_a = rho.reshape(np.tile(dims, 2))
	rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)

	# Flatten back to a matrix on the kept subsystem(s)
	return rho_a.reshape(Nkeep, Nkeep)


def coherent_state(alpha, dim):
	"""
	Return a truncated coherent state |alpha> in the Fock basis.

	|alpha> = exp(-|alpha|^2/2) * sum_{n=0}^{∞} alpha^n / sqrt(n!) |n>

	Parameters
	----------
	alpha : complex
		Coherent amplitude.
	dim : int
		Truncation dimension of the oscillator Hilbert space (max photon number = dim-1).

	Returns
	-------
	ket : (dim,) complex array
		State vector in the number basis.

	Numerical note
	--------------
	We compute sqrt(n!) with Decimal to reduce overflow/rounding issues for larger n.
	"""
	array = np.zeros(dim, dtype='complex128')
	for i in range(dim):
		array[i] = np.exp(-np.abs(alpha) ** 2 / 2) * alpha ** i / float(Decimal(factorial(i)).sqrt())

	return array


def my_wigner(rho, re_lambda_list, im_lambda_list):
	"""
	Compute the Wigner function W(λ) on a grid using displaced parity.

	This helper uses QuTiP's displacement operator. For each phase-space point λ:
		W(λ) = (2/π) Tr[ D(-λ) ρ D(λ) Π ]
	where Π is the photon-number parity operator.

	Parameters
	----------
	rho : (dim, dim) complex array
		Density matrix of a *single* bosonic mode (already traced to the cavity subspace).
	re_lambda_list, im_lambda_list : 1D arrays
		Grid coordinates for Re(λ) and Im(λ).

	Returns
	-------
	wigner : (len(im_lambda_list), len(re_lambda_list)) float array
		Wigner function sampled on the grid.

	Dependency
	----------
	Requires `qutip` at call time.
	"""
	from qutip import destroy, displace

	# NOTE: This print is kept from the original code; remove if you want silence.
	print('here')

	list_len = len(re_lambda_list)
	dim = np.shape(rho)[0]
	a = destroy(dim)  # unused but kept (can be useful for debugging / future refactors)
	wigner = np.zeros((list_len, list_len), dtype='float')

	for i in range(list_len):
		for j in range(list_len):
			lambda_val = re_lambda_list[i] + 1j * im_lambda_list[j]

			# Displace the state to the origin
			rho_disp = displace(dim, -lambda_val) * rho * displace(dim, -lambda_val).dag()

			# Parity expectation: sum_n (-1)^n <n|rho_disp|n>
			tmp = 0.0
			for k in range(dim):
				tmp += abs(rho_disp[k, k]) * (-1) ** k

			# Optional cutoff outside a disk (helps avoid plotting artifacts)
			if abs(lambda_val) <= re_lambda_list.max() * (1.1):
				wigner[j, i] = (2.0 / np.pi) * tmp
			else:
				wigner[j, i] = 0

	return wigner


def q_function(rho, re_lambda_list, im_lambda_list, dim_exp, min_val=0):
	"""
	Compute the Husimi-Q function Q(α) = <α|ρ|α> on a rectangular grid.

	In cavity QED, Q(α) is often used as a "smoothed" phase-space distribution.
	This routine evaluates Q on points α = Re + i Im using truncated coherent states.

	Parameters
	----------
	rho : (dim, dim) complex array
		Density matrix of the cavity mode (already traced to the cavity subspace).
	re_lambda_list, im_lambda_list : 1D arrays
		Grid coordinates for Re(α) and Im(α). The returned array is indexed such that
		re_lambda_list runs along x and im_lambda_list along y.
	dim_exp : int
		Expansion dimension used for coherent states. If dim_exp >= dim, rho is embedded
		into a larger matrix (zero-padded) before evaluation. This can reduce truncation
		artifacts when |α| approaches the cutoff.
	min_val : float
		Floor value enforced to avoid zeros (useful for log-scale plots).

	Returns
	-------
	Q : (len(im_lambda_list), len(re_lambda_list)) float array
		Husimi Q-function on the grid.
	"""
	# "random" imported in original code but not used; retained to avoid behavior changes.
	from random import random  # noqa: F401

	list_len = len(re_lambda_list)
	dim = np.shape(rho)[0]

	# Zero-pad rho into a larger Hilbert space if requested
	rho_exp = np.zeros((dim_exp, dim_exp), dtype='complex128')
	rho_exp[0:dim, 0:dim] = rho

	array_out = np.zeros((list_len, list_len), dtype='float64')

	for i in range(list_len):
		for j in range(list_len):
			lambda_val = re_lambda_list[i] + 1j * im_lambda_list[j]

			# Outside the requested grid radius: fill with a small floor value.
			if np.abs(lambda_val) > re_lambda_list[-1]:
				array_out[-j, i] = min_val * 1.001
			else:
				cs = coherent_state(lambda_val, dim_exp)
				# Q(α) = <α|ρ|α> = cs† (rho_exp cs)
				array_out[-j, i] = np.real(np.dot(np.conj(cs), np.dot(rho_exp, cs)))

				# Enforce a floor to make log-plots well-defined
				if array_out[-j, i] < min_val:
					array_out[-j, i] = min_val * 1.001

	return array_out


def characteristic_function(lmbd_re, lmbd_im, x, p, s):
	"""
	Characteristic function used for phase-space reconstructions (Wigner, etc.).

	This evaluates:
		χ(λ) = Tr[ ρ D(λ) ]
	and returns the integrand used in Fourier transforms to phase-space distributions.

	Parameters
	----------
	lmbd_re, lmbd_im : float
		Real/imaginary parts of λ.
	x, p : float
		Phase-space coordinates of the target point (x, p) for the Fourier kernel.
	s : object
		System-like object providing:
		- s.rho : density matrix
		- s.a, s.a_dag : annihilation/creation operators

	Returns
	-------
	integrand : complex
		Complex integrand value for the given (λ, x, p).
	"""
	lmbd = lmbd_re + 1j * lmbd_im

	# Displacement operator D(λ) = exp( λ a† - λ* a )
	displ_mat = expm(lmbd * s.a_dag - np.conj(lmbd) * s.a)

	characteristic_function = np.trace(s.rho.dot(displ_mat))

	# Fourier kernel convention (matches the original implementation)
	integrand = characteristic_function * np.exp(2 * 1j * (p * lmbd_re - x * lmbd_im))

	return integrand


# The code below was part of an alternate numerical Wigner implementation and is kept
# commented-out as in the original source.
# def int_real( lmd_im, x, y, s ):
#     return quad( characteristic_function, -np.inf, np.inf, args=(lmd_im,x,y,s) )[0]
# def wigner_function( x, y, s ):
#     return quad( int_real, -np.inf, np.inf, args=(x,y,s) )[0]


def kronecker_delta(a, b):
	"""
	Kronecker delta δ_{ab} with basic type checking.

	Returns 1 if a == b else 0.

	Parameters
	----------
	a, b : int
		Integer indices.

	Returns
	-------
	int
		1 if equal, 0 otherwise.

	Raises
	------
	SystemExit
		If inputs are not integers (preserves original behavior).
	"""
	q1 = isinstance(a, int)
	q2 = isinstance(b, int)

	if q1 == False or q2 == False:
		print("ERROR in kronecker delta: input not integer")
		sys.exit()

	output = None
	if a == b:
		output = 1
	else:
		output = 0

	return output


# Legacy/experimental code block retained as comments (original file preserved).


def factorial_approx(n):
	"""
	Stirling approximation for n! (returns a float).

	Useful for rough scaling estimates; not used in the master-equation solver.
	"""
	return np.sqrt(2 * np.pi * n) * (n / np.e) ** n


def sqrt_factorial_approx(n):
	"""
	Approximation to sqrt(n!) based on Stirling's formula.

	Useful for coherent-state amplitudes at large n.
	"""
	return (2 * np.pi * n) ** (0.25) * (n / np.e) ** (n / 2)
