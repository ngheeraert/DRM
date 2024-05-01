import sys
from scipy.integrate import quad
from scipy.linalg import expm
import numpy as np
from math import factorial
import sys
from decimal import Decimal


def partial_trace( rho, rho_dims, keep, optimize=False):

	"""Calculate the partial trace

	ρ_a = Tr_b(ρ)

	Parameters
	----------
	ρ : 2D array
		Matrix to trace
	keep : array
		An array of indices of the spaces to keep after
		being traced. For instance, if the space is
		A x B x C x D and we want to trace out B and D,
		keep = [0,2]
	dims : array
		An array of the dimensions of each space.
		For instance, if the space is A x B x C x D,
		dims = [dim_A, dim_B, dim_C, dim_D]

	Returns
	-------
	ρ_a : 2D array
		Traced matrix
	"""
	keep = np.asarray(keep)
	dims = np.array( rho_dims )
	Ndim = dims.size
	Nkeep = np.prod(dims[keep])

	idx1 = [i for i in range(Ndim)]
	idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
	rho_a = rho.reshape(np.tile(dims,2))
	rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)

	return rho_a.reshape(Nkeep, Nkeep)

def coherent_state( alpha, dim):

	array = np.zeros( dim, dtype='complex128' )
	for i in range(dim):
		array[i] = np.exp( -np.abs(alpha)**2/2 ) * alpha**i / float(Decimal( factorial(i) ).sqrt())

	return array

def my_wigner( rho, re_lambda_list, im_lambda_list):

	from qutip import destroy, displace

	#cav_rho = partial_trace(s.rho, [s.qubit_dim, s.cavity_dim], 1, optimize=False)
	print('here')

	list_len = len( re_lambda_list )
	dim = np.shape(rho)[0]
	a = destroy(dim)
	wigner = np.zeros((list_len,list_len),dtype='float')

	for i in range(list_len):
		for j in range(list_len):

			lambda_val = re_lambda_list[i] + 1j*im_lambda_list[j]
			rho_disp = displace(dim,-lambda_val) * rho * displace(dim,-lambda_val).dag()
#
			tmp = 0.0
			for k in range(dim):
				tmp += abs(rho_disp[k,k])*(-1)**k
#
			if  abs(lambda_val) <= re_lambda_list.max()*(1.1 ):
				wigner[j,i] = (2.0/np.pi)*tmp
			else:
				wigner[j,i] = 0

	return wigner

def q_function( rho, re_lambda_list, im_lambda_list, dim_exp, min_val=0):

	from random import random
	list_len = len( re_lambda_list )
	dim = np.shape(rho)[0]
	rho_exp = np.zeros( (dim_exp, dim_exp),dtype='complex128')
	rho_exp[0:dim,0:dim] = rho
	array_out = np.zeros( (list_len,list_len),dtype='float64')

	for i in range(list_len):
		for j in range(list_len):

			lambda_val = re_lambda_list[i] + 1j*im_lambda_list[j]
			if np.abs(lambda_val) > re_lambda_list[-1]:
				array_out[-j,i] = min_val*1.001
			else:
				cs = coherent_state( lambda_val, dim_exp )
				array_out[-j,i] = np.real( np.dot( np.conj(cs), np.dot( rho_exp, cs ) ) )
				if array_out[-j,i] < min_val:
					array_out[-j,i] = min_val*1.001

	return array_out

def characteristic_function( lmbd_re, lmbd_im, x, p, s  ):

	lmbd = lmbd_re + 1j*lmbd_im
	displ_mat = expm( lmbd*s.a_dag - np.conj(lmbd)*s.a )

	characteristic_function = np.trace( s.rho.dot(displ_mat) ) 

	integrand = characteristic_function * np.exp( 2*1j*(p*lmbd_re-x*lmbd_im) )

	return integrand


#def int_real( lmd_im, x, y, s ):
#   
#    return quad( characteristic_function, -np.inf, np.inf, args=(lmd_im,x,y,s) )[0]
#
#def wigner_function( x, y, s ):
#    
#    return quad( int_real, -np.inf, np.inf, args=(x,y,s) )[0]

def kronecker_delta(a,b):

	q1 = isinstance(a, int)
	q2 = isinstance(b, int)

	if q1==False or q2==False:
		print("ERROR in kronecker delta: input not integer")
		sys.exit()

	output = None
	if a==b:
		output = 1
	else:
		output = 0

	return output




#list_len = len( re_lambda_list )
	#dims = rho.dims[0][0]
	#a = destroy(dims)
	#wigner = np.zeros((list_len,list_len),dtype='float')
	#for i in range(list_len):
	#	for j in range(list_len):
#
	#		lambda_val = re_lambda_list[i] + 1j*im_lambda_list[j]
	#		rho_disp = displace(dims,-lambda_val) * rho * displace(dims,-lambda_val).dag()
#
	#		tmp = 0.0
	#		for k in range(dims):
	#			tmp += abs(rho_disp[k,k])*(-1)**k
#
	#		if  abs(lambda_val) <= re_lambda_list.max()*(1.1 ):
	#			wigner[j,i] = (2.0/np.pi)*tmp
	#		else:
	#			wigner[j,i] = 1

def factorial_approx(n):

	return np.sqrt(2*np.pi*n)*( n/np.e )**n

def sqrt_factorial_approx(n):

	return (2*np.pi*n)**(0.25)*( n/np.e )**(n/2)

