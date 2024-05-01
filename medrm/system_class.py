import numpy as np
from scipy.integrate import complex_ode, ode, quad
from scipy.linalg import expm
import sys
from time import time
from math import factorial
from scipy.linalg import expm

hbar = 1.05457162e-34
pi = np.pi
ide = np.identity(2)
sig_x = np.array([[0.,1.],[1.,0.]])
sig_y = np.array([[0.,-1j],[1j,0.]])
sig_z = np.array([[1.,0.],[0.,-1.]])

class system(object):

	def __init__( self, w01=-1.0, w=1.0, g=0.1, wd= 0, gamma=0, Ad=0.0, cavity_dim=3, qubit_dim=2, tint = 10, dvice='TRSM1',atol=1e-8,rtol=1e-6, max_step=1e-2,qb_ini=[0], dim_exp=20,  coupling_type='00', verbose=True ):

		self.w = w
		self.g = g
		self.wd = wd
		self.t = 0.0
		self.gamma = gamma
		self.Ad = Ad
		self.tint = tint
		self.dvice = dvice
		self.atol = atol
		self.rtol = rtol
		self.max_step = max_step
		self.coupling_type = coupling_type

		self.qubit_dim = qubit_dim
		self.cavity_dim = cavity_dim
		self.sys_dim = qubit_dim*cavity_dim
		self.dim_exp=dim_exp

		self.initial_qb_state = np.zeros( qubit_dim, dtype='complex128' )
		self.initial_cavity_state = np.zeros( cavity_dim, dtype='complex128' )

		self.qb_ini = None
		self.initialise_density_matrix()

		self.ide_qb = np.identity( self.qubit_dim )
		self.ide_cav = np.identity( self.cavity_dim )
		self.ide = np.identity( self.sys_dim )

		a_temp = np.zeros( (cavity_dim,cavity_dim) )
		for i in range(cavity_dim-1):
			a_temp[i,i+1] = np.sqrt(i+1) 

		b_temp = np.zeros( (qubit_dim,qubit_dim) )
		for i in range(qubit_dim-1):
			b_temp[i,i+1] = np.sqrt(i+1) 

		Pg_temp = np.zeros( (qubit_dim,qubit_dim) )
		Pe_temp = np.zeros( (qubit_dim,qubit_dim) )
		Pg_temp[0,0] = 1
		Pe_temp[1,1] = 1

		P_plus_temp = np.zeros( (qubit_dim,qubit_dim) )
		P_minus_temp = np.zeros( (qubit_dim,qubit_dim) )
		P_plus_temp[:,:]  = 1/2
		P_minus_temp[:,:] = 1/2
		P_minus_temp[0,1] = -1/2
		P_minus_temp[1,0] = -1/2

		sigmaz_temp = np.zeros( (qubit_dim,qubit_dim) )
		sigmaz_temp[:,:]  = 0
		sigmaz_temp[0,0] = 1
		sigmaz_temp[1,1] = 1

		self.na_red = (a_temp.T).dot( a_temp )

		self.a = np.kron( self.ide_qb, a_temp )
		self.a_dag = np.transpose( self.a )
		self.b = np.kron( b_temp, self.ide_cav )
		self.b_dag = np.transpose( self.b )
		self.na = self.a_dag.dot( self.a )
		self.nb = self.b_dag.dot( self.b )
		self.sigmaz = np.kron( sigmaz_temp, self.ide_cav  )
		self.Pg = np.kron( Pg_temp, self.ide_cav  )
		self.Pe = np.kron( Pe_temp, self.ide_cav  )
		self.P_plus = np.kron( P_plus_temp, self.ide_cav  )
		self.P_minus = np.kron( P_minus_temp, self.ide_cav  )

		self.sigma_z = self.b_dag.dot( self.b )

		self.Xa = ( self.a + self.a_dag )
		self.Xb = ( self.b + self.b_dag )

		if ( self.dvice == 'QUBIT' ):
			self.w01 = w01
			self.anh = 0*-1 * 2*np.pi

		else:

			if ( self.dvice == 'TRSM2' ):
				energy_filename = "../MPOL_DCT_FOR/qubit_params/FOR_E_lvl_Ec0.190_Ej14.368.txt"
				coupling_op_filename = "../MPOL_DCT_FOR/qubit_params/FOR_Charge_Mat_Ec0.190_Ej14.368.txt"
			elif ( self.dvice == 'TRSM3' ):
				energy_filename = "../MPOL_DCT_FOR/qubit_params/T_FOR_E_lvl_Ec0.280_Ej14.000.txt"
				coupling_op_filename = "../MPOL_DCT_FOR/qubit_params/T_FOR_Charge_Mat_Ec0.280_Ej14.000.txt"
			elif ( self.dvice == 'QUTR2' ):
				energy_filename = "../MPOL_DCT_FOR/qubit_params/FOR_E_lvl_Ec0.190_Ej14.368.txt"
				coupling_op_filename = "../MPOL_DCT_FOR/qubit_params/FOR_Cos_Phi_q_Mat_Ec0.190_Ej14.368.txt"

			energy_levels = np.loadtxt( energy_filename )*2*np.pi
			coupling_op = np.loadtxt( coupling_op_filename )

			H_qb_IS = np.zeros( (qubit_dim,qubit_dim), dtype='float64' )
			for i in range(qubit_dim):
				H_qb_IS[i,i] = energy_levels[i]
			self.w01 = H_qb_IS[1,1] - H_qb_IS[0,0]
			if qubit_dim>2:
				self.anh = H_qb_IS[2,2] - 2*H_qb_IS[1,1]
			else:
				self.anh = 99


		#===========================
		#== cavity and drive Hamiltonian
		#===========================
		self.H_cav = self.w*self.na
		self.H_drive = self.a + self.a_dag

		#===========================
		#== qubit and coupling Hamiltonian
		#===========================
		if ( coupling_type == '00' ):

			#self.H_qb = self.w01*self.nb + (self.anh/2)*( self.nb ).dot( self.ide - self.nb )
			self.H_qb = self.sigmaz*self.nb# + (self.anh/2)*( self.nb ).dot( self.ide - self.nb )
			self.H_coupling = g*( self.Xa ).dot( self.Xb )

		elif ( coupling_type == '11' ):

			self.H_qb = np.kron( H_qb_IS, self.ide_cav )
			#-- renormalisation was removed to match benchmarking with Angela
			g_qc = self.g*coupling_op[ :qubit_dim , :qubit_dim ]# / coupling_op[ 0,1 ]
			if ( self.dvice[0:4] == 'TRSM' ):
				self.H_coupling = np.kron( g_qc, a_temp + np.transpose(a_temp)  )
			elif ( self.dvice[0:4] == 'QUTR' ):
				self.H_coupling = np.kron( g_qc, (a_temp + np.transpose(a_temp)).dot(a_temp + np.transpose(a_temp)) )

		elif ( coupling_type == '01' ):

			self.H_qb = self.w01*self.nb + (self.anh/2)*( self.nb ).dot( self.ide - self.nb )
			g_qc = - self.g*coupling_op[ :qubit_dim , :qubit_dim ] / coupling_op[ 0,1 ]
			self.H_coupling = np.kron( g_qc, a_temp + np.transpose(a_temp)  )

		elif ( coupling_type == '10' ):

			self.H_qb = np.kron( H_qb_IS, self.ide_cav )
			self.H_coupling = -g*( self.Xa ).dot( self.Xb )

		self.H = self.H_qb + self.H_cav + self.H_coupling

		evalue, evec = np.linalg.eig( self.H )  
		sorted_indices = np.argsort( evalue )
		self.eig_vec = evec[ :, sorted_indices ]
		self.eig_val = evalue[ sorted_indices ]

		if verbose:
			print("===============================================" )
			print("Nq, Nc  = {:d}, {:d}".format( self.qubit_dim, self.cavity_dim ) )
			print("w01, wc, wd  = {:7.4f}, {:7.4f}, {:7.4f} "\
					.format( self.w01/(2.0*np.pi),self.w/(2.0*np.pi),self.wd/(2.0*np.pi) ) )
			print("Ad  = {:7.4f} ".format( self.Ad/(2.0*np.pi) ) )
			print("ah  = {:7.4f} ".format( self.anh/(2.0*np.pi) ) )
			print("g  = {:7.4f} ".format( g/(2.0*np.pi) ) )
			print("kappa  = {:7.4f} ".format( self.gamma/(2.0*np.pi) ) )
			print("atol  = {:.1e} ".format( self.atol ) )
			print("rtol  = {:.1e} ".format( self.rtol ) )
			print("max_step  = {:.0e} ".format( self.max_step ) )
			print("couling_type  = {:} ".format( self.coupling_type ) )
			print("device  = {:} ".format( self.dvice ) )
			print("===============================================" )

	def expect( self, op ):

		return np.trace( op.dot(self.rho) )

	def renyi_entropy( self ):

		return 1.0 - np.trace( self.rho.dot(self.rho) )

	def renyi_entropy_2( self, rho_in ):

		return 1.0 - np.trace( rho_in.dot(rho_in) )

	def set_initial_qb_state( self, qb_state ):

		self.initial_qb_state[ : ] = 0.0

		for i in range(len(qb_state)):
			self.initial_qb_state[ qb_state[i] ] = 1.0

		self.qb_ini = qb_state
		self.qb_ini.append('B')

		norm = np.sum( np.abs(self.initial_qb_state)**2 )
		self.initial_qb_state /= np.sqrt(norm)

	def set_initial_qb_cav_dm( self, qb_state ):

		qb_cav_state = np.zeros( self.sys_dim )

		for qubit_level in qb_state:
			qb_cav_state += self.eig_vec[:, qubit_level ]

		self.qb_ini = qb_state
		self.qb_ini.append('D')

		norm = np.sum( np.abs(qb_cav_state)**2 )
		qb_cav_state /= np.sqrt(norm)

		self.st_ini = qb_cav_state
		self.rho_ini = np.outer( qb_cav_state, np.conj(qb_cav_state) )
		self.rho = np.outer( qb_cav_state, np.conj(qb_cav_state) )

	def set_initial_photon_state( self, n ):

		self.initial_cavity_state[:] = 0.0
		self.initial_cavity_state[n] = 1.0

	def set_initial_cs_state( self, alpha ):

		from decimal import Decimal
		from math import factorial

		for n in range(self.cavity_dim):
			denom = float( Decimal( factorial(n) ).sqrt() )
			self.initial_cavity_state[n] = np.exp( -np.abs(alpha)**2/2 ) * alpha**n / denom

	def initialise_density_matrix( self ):

		qubit_rho0 = np.outer( self.initial_qb_state, np.conj(self.initial_qb_state) )
		cavity_rho0 = np.outer( self.initial_cavity_state, np.conj(self.initial_cavity_state) )

		self.rho = np.kron( qubit_rho0, cavity_rho0 )

	def calc_fidelity( self, rho ):

		partial_rho = partial_trace( rho, [self.qubit_dim,self.cavity_dim], [0] )

		return np.real( np.conj((self.initial_qb_state.T)).dot( partial_rho ).dot( self.initial_qb_state ) )

	def ode_RHS( self, t, dm_1D ):

		dm = dm_1D.reshape( self.sys_dim, self.sys_dim )

		unitary = - 1j * ( self.H.dot(dm) - dm.dot(self.H) )
		drive = - 1j * ( self.H_drive.dot(dm) - dm.dot(self.H_drive) ) * self.Ad * np.cos( self.wd*t )
		decay = self.gamma * ( self.a.dot(dm).dot(self.a_dag) \
				-0.5*( self.na.dot(dm) + dm.dot(self.na) ) ) 

		dm_result = ( unitary + decay + drive ).reshape(self.sys_dim**2)
		#dm_result = unitary.reshape(self.sys_dim**2)

		return dm_result

	def time_evolve( self, times, verbose=True ):

		t_int = times[1]-times[0]

		print_times = list( np.linspace( -1e-7, times[-1]*1.001 , 11 ) )
		last_print_t = -1e10
		last_cpu_t = time()

		dm_integrator = complex_ode( self.ode_RHS )
		dm_integrator.set_integrator('dopri5', atol=self.atol, rtol=self.rtol, method='adams',nsteps=t_int*1e5, max_step=self.max_step)
		#dm_integrator = ode( self.ode_RHS )
		#dm_integrator.set_integrator('zvode', atol=self.atol, rtol=self.rtol, method='adams',nsteps=t_int*1e5, order=self.order )
		rhos_out = []
		rhos_1D = []

		rho_1D = self.rho.reshape( self.sys_dim**2 )
		dm_integrator.set_initial_value(rho_1D, times[0])
		rhos_1D.append(rho_1D)

		for i in range(1,len(times)):

			if print_times[0] < times[i]:
				new_cpu_t = time()
				print_times.pop(0)
				if verbose:
					print( 't=', "{:.1f}".format( times[i-1] )+' || '\
						+"{:7d}".format( int( new_cpu_t-last_cpu_t ) ) )
				last_cpu_t = new_cpu_t

			sol_rho = dm_integrator.integrate( times[i] )
			if dm_integrator.successful() == False:
				print("ERROR: ", dm_integrator.get_return_code())
			rhos_1D.append(sol_rho)

		for i in range(len(times)):
			rhos_out.append(rhos_1D[i].reshape( self.sys_dim, self.sys_dim ))

		self.rho = rhos_out[-1]

		return rhos_out

	def my_wigner(self, re_lambda_list, im_lambda_list):

		from qutip import displace
		cav_rho = partial_trace(self.rho, [self.qubit_dim, self.cavity_dim], 1, optimize=False)

		list_len = len( re_lambda_list )
		dim = np.shape(cav_rho)[0]
		wigner = np.zeros((list_len,list_len),dtype='float')


		for i in range(list_len):
			for j in range(list_len):

				lambda_val = re_lambda_list[i] + 1j*im_lambda_list[j]
				disp_mat1 = displace(dim,-lambda_val).full()
				disp_mat2 = displace(dim,+lambda_val).full()
				rho_disp = disp_mat1.dot(cav_rho).dot(disp_mat2)

				tmp = 0.0
				for k in range(dim):
					tmp += rho_disp[k,k]*(-1)**k

				#if  abs(lambda_val) <= re_lambda_list.max()*1.1:
				wigner[j,i] = np.real( (2.0/np.pi)*tmp )
				#else:
				#    wigner[j,i] = 0

		return wigner

	def paramchar(self, tmax):

		return ('tmax{:4d}_Nq{:2d}_Nc{:2d}_amp{:7.4f}_kappa{:7.4f}_wq{:7.4f}_anh{:7.4f}_wc{:7.4f}_g{:7.4f}_wd{:7.4f}_ms{:.0e}_dimexp{:}_qb'+str(self.qb_ini)+'_{:}_'+self.dvice)\
				.format( int(tmax), self.qubit_dim, self.cavity_dim, self.Ad/(2.0*pi), self.gamma/(2.0*pi), self.w01/(2.0*pi), self.anh/(2.0*pi),self.w/(2.0*pi), self.g/(2.0*pi), self.wd/(2.0*pi), self.max_step, self.dim_exp, self.coupling_type ).replace(" ","")

	def save_and_plot(self, times, rhos, lvl_plot=[1], max_lambda=7 ):

		import matplotlib.pyplot as plt
		from matplotlib.colors import LinearSegmentedColormap
		from matplotlib.colors import LogNorm


		nsteps = len(times)

		pop_arr = np.zeros( (nsteps, self.qubit_dim+1), dtype='float64' )
		pop_arr_dressed = np.zeros( (nsteps, self.qubit_dim+1), dtype='float64' )
		n_arr = np.zeros( (nsteps,2), dtype='float64' )
		entropy_arr = np.zeros( (nsteps), dtype='float64' )

		pplt_buffer = np.zeros( (self.qubit_dim,int(1/(times[1]-times[0]))), dtype='float64' )

		vac_qb = np.zeros( (self.qubit_dim, self.qubit_dim), dtype='complex128' )
		vac_qb[0,0] = 1
		vac = np.kron( vac_qb, self.ide_cav )  

		re_lambda_list = np.linspace( -max_lambda, max_lambda, 70 )
		im_lambda_list = np.linspace( -max_lambda, max_lambda, 70 )

		#rho_cav = partial_trace( rhos[-1], [self.qubit_dim,self.cavity_dim], [1] )
		#min_val = 0.01
		##qfunction_arr = q_function( rho_cav, re_lambda_list, im_lambda_list, self.dim_exp, min_val )
		#filename = 'data/Qfunction_' + self.paramchar(times[-1])  + '_ML'+str(max_lambda)+ '.d'
		#np.savetxt( filename, qfunction_arr )

		#-- entropy_plot
		plt.plot( times, n_arr[:,1] )
		plt.xlabel(r'$t$')
		plt.ylabel(r'$n_{cav}$')
		plt.savefig( "figures/PHOTONS_" + self.paramchar(times[-1])+ '.pdf', format='pdf'  )
		plt.show()

		#-- entropy_plot
		#plt.plot( times, entropy_arr )
		#plt.xlabel(r'$t$')
		#plt.ylabel('Entropy')
		#plt.savefig( "figures/Entropy_" + self.paramchar(times[-1])+ '.pdf', format='pdf'  )
		#plt.show()
		#plt.close()

		#-- non log plot
		#ext = (-max_lambda, max_lambda, -max_lambda, max_lambda)
		#interval = np.linspace(0.00, 1)
		#colors = plt.cm.magma(interval)
		#my_cmap = LinearSegmentedColormap.from_list('name', colors)

		#zmin, zmax = min_val, qfunction_arr.max()
		#plt.imshow( qfunction_arr, extent=ext, cmap=my_cmap  )
		#plt.colorbar()
		#plt.xlabel(r'$\varphi$')
		#plt.ylabel(r'$Q$')
		#plt.savefig( "figures/Qfunction_" + self.paramchar(times[-1]) + '_ML'+str(max_lambda) + '.pdf', format='pdf'  )
		#plt.close()

		#-- log plot
		#ext = (-max_lambda, max_lambda, -max_lambda, max_lambda)
		#interval = np.linspace(0.00, 1)
		#colors = plt.cm.magma(interval)
		#my_cmap = LinearSegmentedColormap.from_list('name', colors)

		#zmin, zmax = min_val, qfunction_arr.max()
		#norm=LogNorm(vmin=zmin, vmax=zmax)
		#plt.imshow( qfunction_arr, extent=ext, norm=norm, cmap=my_cmap )
		#plt.colorbar()
		#plt.xlabel(r'$\varphi$')
		#plt.ylabel(r'$Q$')
		#plt.savefig( "figures/LOG_Qfunction_" + self.paramchar(times[-1]) + '_ML'+str(max_lambda)+'.pdf', format='pdf'  )

		#plt.close()

	def save_husimi_for_gif(self, times, rhos, max_lambda, frame_nb, log ):

		import matplotlib.pyplot as plt
		from matplotlib.colors import LinearSegmentedColormap
		from matplotlib.colors import LogNorm

		nsteps = len(times)

		re_lambda_list = np.linspace( -max_lambda, max_lambda, 70 )
		im_lambda_list = np.linspace( -max_lambda, max_lambda, 70 )

		color_map_defined = False

		n_arr = np.zeros( (frame_nb,2), dtype='float64' )
		for i in range( frame_nb ):
			ind = int( (nsteps/frame_nb)*i )
			n_arr[i,0] = times[ind]
			n_arr[i,1] = np.real( np.trace( self.na.dot( rhos[ind] ) ) )

		print("-- frame_nb=",frame_nb)
		for i in range( frame_nb ):

			print( i, end=" " )
			ind = int( (nsteps/frame_nb)*i )

			fig, (ax1,ax2) = plt.subplots(1,2)
			fig.set_size_inches(10, 4)

			ax2.plot( n_arr[:i+1,0], n_arr[:i+1,1] )
			ax2.set_xlim(0,n_arr[-1,0]*1.1)
			ax2.set_ylim(0, max(n_arr[:,1])*1.1)
			ax2.set_xlabel('Time')
			ax2.set_ylabel('PHOTON NUMBER')
			ax2.scatter( n_arr[i,0], n_arr[i,1], s=20, color='red'  )

			rho_rot = expm( 1j*self.wd*self.na*times[ind] ).dot( rhos[ind].dot( expm( -1j*self.wd*self.na*times[ind] ) ) )
			rho_rot_cav = partial_trace( rho_rot, [self.qubit_dim,self.cavity_dim], [1] )

			#qfunction_arr = q_function( rho_cav, re_lambda_list, im_lambda_list, self.dim_exp )
			min_val = 0.001
			qfunction_arr = q_function( rho_rot_cav, re_lambda_list, im_lambda_list, self.dim_exp, min_val )
			filename = 'figures/husimi/data/qfunction_' + self.paramchar(times[-1]) \
					+ '_ML'+str(max_lambda)+'_'+str(i)+'_'+str(frame_nb)+'.d'
			np.savetxt( filename, qfunction_arr )

			ax1.set_xlabel(r'$\hat \varphi$')
			ax1.set_ylabel(r'$\hat Q$')

			ext = (-max_lambda, max_lambda, -max_lambda, max_lambda)
			interval = np.linspace(0.00, 1)
			colors = plt.cm.magma(interval)
			my_cmap = LinearSegmentedColormap.from_list('name', colors)
			color_map_defined = True

			if not log:
				zmin, zmax = 0.00, qfunction_arr.max()
				s = ax1.imshow( qfunction_arr, extent=(-max_lambda, max_lambda, -max_lambda, max_lambda), cmap=my_cmap , vmin=zmin, vmax=zmax )
				fig.colorbar(s, ax=ax1)
				plt.tight_layout()
				plt.savefig( "figures/husimi/"+str(i)+'_'+str(frame_nb)+"_Qfunction_" + self.paramchar(times[-1])\
						+ '_ML'+str(max_lambda)+'.jpg', format='jpg', dpi=120 )
			else:
				zmin, zmax = min_val, qfunction_arr.max()
				norm=LogNorm(vmin=zmin, vmax=zmax)
				s = ax1.imshow( qfunction_arr, extent=ext, norm=norm, cmap=my_cmap )
				fig.colorbar(s, ax=ax1)
				plt.tight_layout()
				plt.savefig( "figures/husimi/"+str(i)+'_'+str(frame_nb)+"_LOG_Qfunction_" + self.paramchar(times[-1])\
						+ '_ML'+str(max_lambda)+'.jpg', format='jpg', dpi=120)

			plt.close()

	def photon_distribution( self, rho_in ):

		out = np.zeros( (self.qubit_dim, self.cavity_dim) )

		for i in range(self.qubit_dim):
			for n in range(self.cavity_dim):
				out[i,n] = np.abs( rho_in[ i*self.cavity_dim + n,i*self.cavity_dim + n ] )
		
		return out
