"""
system_class.py
===============

Dense-matrix simulator for a driven, lossy qubit–cavity (Rabi-model-like) system.

This module defines a `system` class that:
- builds the composite Hilbert space (qubit ⊗ cavity),
- constructs Hamiltonian and dissipator operators,
- integrates a Lindblad master equation using SciPy ODE solvers,
- provides a few convenience diagnostics (expectation values, Wigner/Q functions).

Conventions
-----------
- All frequencies are treated as angular frequencies (rad/time). The time unit is
  whatever is consistent with your chosen parameters (e.g. nanoseconds if ω is in rad/ns).
"""

import numpy as np
from scipy.integrate import complex_ode, ode, quad
from scipy.linalg import expm
import sys
from time import time
from math import factorial
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Basic constants / small helpers (mostly unused in the current solver)
# ---------------------------------------------------------------------------
hbar = 1.05457162e-34
pi = np.pi
ide = np.identity(2)
sig_x = np.array([[0.,1.],[1.,0.]])
sig_y = np.array([[0.,-1j],[1j,0.]])
sig_z = np.array([[1.,0.],[0.,-1.]])

class system(object):

    """Composite qubit–cavity system with Lindblad master-equation time evolution."""

    def __init__( self, w01=-1.0, w=1.0, g=0.1, wd= 0, gamma=0, Ad=0.0, cavity_dim=3, qubit_dim=2, tint = 10, dvice='TRSM1',atol=1e-8,rtol=1e-6, max_step=1e-2,qb_ini=[0], dim_exp=20,  coupling_type='00', verbose=True ):
        """Create a driven-dissipative qubit–cavity system and precompute operators.
        
        Parameters (most important)
        --------------------------
        w01 : float
            Qubit |0>→|1> transition angular frequency (rad/time). Used when dvice=='QUBIT'
            or as a fallback if device parameters are not loaded from file.
        w : float
            Cavity angular frequency ω_c (rad/time).
        g : float
            Qubit–cavity coupling strength (rad/time). Interpretation depends on coupling_type.
        wd : float
            Drive angular frequency ω_d (rad/time). Used in the time-dependent drive term.
        gamma : float
            Cavity energy decay rate κ (rad/time) for the Lindblad dissipator with collapse operator `a`.
        Ad : float
            Drive amplitude (rad/time) multiplying the chosen cavity quadrature operator `H_drive`.
        cavity_dim, qubit_dim : int
            Hilbert-space truncation sizes for cavity and qubit degrees of freedom.
        coupling_type : str
            Selects how to build the qubit Hamiltonian and coupling operator:
              - '00': simple Rabi-like coupling g (a+a†)(b+b†) with a toy qubit Hamiltonian
              - '11', '01', '10': variants using device-specific level structure and coupling operators
        dvice : str
            Device model selector. For 'TRSM*' / 'QUTR*', this code loads precomputed transmon
            parameters from text files on disk (see paths below).
        
        Physics model
        -------------
        The density matrix ρ evolves according to a (time-local) Lindblad master equation:
        
            dρ/dt = -i [H, ρ]
                    -i Ad sin(ω_d t) [H_drive, ρ]
                    + γ ( a ρ a† - 1/2 {a†a, ρ} )
        
        where:
        - H = H_qb + H_cav + H_coupling is time-independent.
        - H_drive is a cavity quadrature operator (here chosen as i(a - a†), i.e. proportional to P).
        - The dissipator models single-photon loss from the cavity.
        
        Implementation notes
        --------------------
        - Basis ordering is **qubit ⊗ cavity** (qubit index varies slowest). In matrix indexing,
          the composite basis state |q, n> corresponds to linear index q*cavity_dim + n.
        - Operators are stored as dense NumPy arrays; this is fine for moderate truncations but
          scales as O((Nq*Nc)^2) memory and O((Nq*Nc)^3) for dense linear algebra.
        - The solver uses SciPy's `complex_ode` with explicit RK (dopri5). For strong dissipation
          or large Hilbert spaces, a stiff solver or sparse representation may be preferable.
        """

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
        # Total Hilbert-space dimension D = Nq * Nc
        self.sys_dim = qubit_dim*cavity_dim

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

        # Cavity annihilation operator a acts on the cavity factor: I_qb ⊗ a_cav
        self.a = np.kron( self.ide_qb, a_temp )
        self.a_dag = np.transpose( self.a )
        # Qubit lowering operator b acts on the qubit factor: b_qb ⊗ I_cav
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
                energy_filename = '../MPOL_DCT_FOR/qubit_params/FOR_E_lvl_Ec0.190_Ej14.368.txti'
                coupling_op_filename = '../MPOL_DCT_FOR/qubit_params/FOR_Charge_Mat_Ec0.190_Ej14.368.txt'
            elif ( self.dvice == 'TRSM3' ):
                energy_filename = '../MPOL_DCT_FOR/qubit_params/T_FOR_E_lvl_Ec0.280_Ej14.000.txt'
                coupling_op_filename = '../MPOL_DCT_FOR/qubit_params/T_FOR_Charge_Mat_Ec0.280_Ej14.000.txt'
            elif ( self.dvice == 'QUTR2' ):
                energy_filename = '../MPOL_DCT_FOR/qubit_params/FOR_E_lvl_Ec0.190_Ej14.368.txt'
                coupling_op_filename = '../MPOL_DCT_FOR/qubit_params/FOR_Cos_Phi_q_Mat_Ec0.190_Ej14.368.txt'

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
        #self.H_drive = self.a + self.a_dag
        self.H_drive = 1j*(self.a - self.a_dag)

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
            print("Nq, Nc = {:d}, {:d}".format( self.qubit_dim, self.cavity_dim ) )
            print("w01, wc, wd = {:7.4f}, {:7.4f}, {:7.4f} "\
                    .format( self.w01/(2.0*np.pi),self.w/(2.0*np.pi),self.wd/(2.0*np.pi) ) )
            print("Ad = {:7.4f} ".format( self.Ad/(2.0*np.pi) ) )
            print("ah = {:7.4f} ".format( self.anh/(2.0*np.pi) ) )
            print("g = {:7.4f} ".format( g/(2.0*np.pi) ) )
            print("kappa = {:7.4f} ".format( self.gamma/(2.0*np.pi) ) )
            print("atol = {:.1e} ".format( self.atol ) )
            print("rtol = {:.1e} ".format( self.rtol ) )
            print("max_step = {:.0e} ".format( self.max_step ) )
            print("couling_type = {:} ".format( self.coupling_type ) )
            print("device = {:} ".format( self.dvice ) )
            print("===============================================" )

    def expect( self, op ):
        """Return expectation value Tr[op ρ] for the current state."""

        return np.trace( op.dot(self.rho) )

    def renyi_entropy( self ):
        """Convenience: 1 - Tr[ρ²] (linear/Rényi-2 entropy proxy)."""

        return 1.0 - np.trace( self.rho.dot(self.rho) )

    def renyi_entropy_2( self, rho_in ):
        """Same as renyi_entropy but for an explicit density matrix argument."""

        return 1.0 - np.trace( rho_in.dot(rho_in) )

    def set_initial_qb_state( self, qb_state ):
        """Set the initial qubit state as a superposition of computational levels.
        
        `qb_state` is a list of level indices. For example:
        - [0] prepares |0>
        - [1] prepares |1>
        - [0,1] prepares (|0> + |1>)/sqrt(2)
        
        The cavity state is not modified here.
        """

        self.initial_qb_state[ : ] = 0.0

        for i in range(len(qb_state)):
            self.initial_qb_state[ qb_state[i] ] = 1.0

        self.qb_ini = qb_state
        self.qb_ini.append('B')

        norm = np.sum( np.abs(self.initial_qb_state)**2 )
        self.initial_qb_state /= np.sqrt(norm)

    def set_initial_qb_cav_dressed_state_density_matrix( self, qb_state ):
        """Initialize ρ to a *dressed* eigenstate superposition.
        
        The state is constructed from eigenvectors of the full Hamiltonian H (computed in __init__):
            |ψ> = sum_{k in qb_state} |eig_k>
        then ρ = |ψ><ψ|.
        
        This is useful for working in the interacting (dressed) basis rather than the bare basis.
        """

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

        print('-- state installed in: ', np.round( self.eig_vec[:, qubit_level ], 6 ) )

    def set_initial_photon_state( self, n ):
        """Set the cavity to a Fock state |n> (qubit left unchanged)."""

        self.initial_cavity_state[:] = 0.0
        self.initial_cavity_state[n] = 1.0

    def set_initial_cs_state( self, alpha ):
        """Set the cavity to a coherent state |α> (truncated to cavity_dim)."""

        from decimal import Decimal
        from math import factorial

        for n in range(self.cavity_dim):
            denom = float( Decimal( factorial(n) ).sqrt() )
            self.initial_cavity_state[n] = np.exp( -np.abs(alpha)**2/2 ) * alpha**n / denom

    def initialise_density_matrix( self ):
        """Build the full initial density matrix ρ = ρ_qubit ⊗ ρ_cavity from the stored kets."""

        qubit_rho0 = np.outer( self.initial_qb_state, np.conj(self.initial_qb_state) )
        cavity_rho0 = np.outer( self.initial_cavity_state, np.conj(self.initial_cavity_state) )

        self.rho = np.kron( qubit_rho0, cavity_rho0 )

    def calc_fidelity( self, rho ):
        """Compute fidelity with the initial qubit state after tracing out the cavity.
        
        This measures F = <ψ_qb| Tr_cav(ρ) |ψ_qb> using the currently configured `initial_qb_state`.
        """

        partial_rho = partial_trace( rho, [self.qubit_dim,self.cavity_dim], [0] )

        return np.real( np.conj((self.initial_qb_state.T)).dot( partial_rho ).dot( self.initial_qb_state ) )

    def ode_RHS( self, t, dm_1D ):
        """Right-hand side of the master equation for SciPy's ODE integrator.
        
        The integrator state is a flattened 1D vector containing the density matrix entries.
        We reshape it, compute the Lindblad RHS, then flatten again.
        """

        dm = dm_1D.reshape( self.sys_dim, self.sys_dim )

        # Coherent (Hamiltonian) part: -i[H, ρ]
        unitary = - 1j * ( self.H.dot(dm) - dm.dot(self.H) )
        #drive = - 1j * ( self.H_drive.dot(dm) - dm.dot(self.H_drive) ) * self.Ad * np.cos( self.wd*t )
        # Time-dependent cavity drive: -i Ad sin(ω_d t) [H_drive, ρ]
        drive = - 1j * ( self.H_drive.dot(dm) - dm.dot(self.H_drive) ) * self.Ad * np.sin( self.wd*t )
        # Lindblad cavity decay (single-photon loss) with rate γ and collapse op a
        decay = self.gamma * ( self.a.dot(dm).dot(self.a_dag) \
                -0.5*( self.na.dot(dm) + dm.dot(self.na) ) ) 

        dm_result = ( unitary + decay + drive ).reshape(self.sys_dim**2)
        #dm_result = unitary.reshape(self.sys_dim**2)

        return dm_result

    def time_evolve( self, times, verbose=True ):
        """Integrate the master equation over an array of time points.
        
        Parameters
        ----------
        times : 1D array
            Monotonically increasing time grid.
        verbose : bool
            If True, prints progress updates with rough CPU time.
        
        Returns
        -------
        rhos_out : list[np.ndarray]
            List of density matrices ρ(t_i) with shape (sys_dim, sys_dim).
        """

        t_int = times[1]-times[0]

        print_times = list( np.linspace( -1e-7, times[-1]*1.001 , 11 ) )
        last_print_t = -1e10
        last_cpu_t = time()

        # ODE integrator over the flattened density matrix (complex-valued).
        dm_integrator = complex_ode( self.ode_RHS )
        # Use explicit Runge–Kutta (dopri5). Parameters control accuracy and step size.
        dm_integrator.set_integrator('dopri5', atol=self.atol, rtol=self.rtol, method='adams',nsteps=t_int*1e5, max_step=self.max_step)
        #dm_integrator = ode( self.ode_RHS )
        #dm_integrator.set_integrator('zvode', atol=self.atol, rtol=self.rtol, method='adams',nsteps=t_int*1e5, order=self.order )
        rhos_out = []

        #rhos_1D = []
        rhos_out=[]

        rho_1D = self.rho.reshape( self.sys_dim**2 )
        dm_integrator.set_initial_value(rho_1D, times[0])
        rhos_out.append(rho_1D.reshape(self.sys_dim, self.sys_dim).copy())

        for i in range(1, len(times)):
            sol_rho_1D = dm_integrator.integrate(times[i])
            if not dm_integrator.successful():
                print("ERROR: ", dm_integrator.get_return_code())
            rhos_out.append(sol_rho_1D.reshape(self.sys_dim, self.sys_dim).copy())

        self.rho = rhos_out[-1]

        return rhos_out

    def my_wigner(self, re_lambda_list, im_lambda_list):
        """Compute cavity Wigner function of the *final* state (requires QuTiP)."""

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
        """Build a human-readable parameter string used to name output files."""

        return ('tmax{:4d}_Nq{:2d}_Nc{:2d}_amp{:7.4f}_kappa{:7.4f}_wq{:7.4f}_anh{:7.4f}_wc{:7.4f}_g{:7.4f}_wd{:7.4f}_ms{:.0e}_qb'+str(self.qb_ini)+'_{:}_'+self.dvice)\
                .format( int(tmax), self.qubit_dim, self.cavity_dim, self.Ad/(2.0*pi), self.gamma/(2.0*pi), self.w01/(2.0*pi), self.anh/(2.0*pi),self.w/(2.0*pi), self.g/(2.0*pi), self.wd/(2.0*pi), self.max_step, self.coupling_type ).replace(" ","")

    def save_and_plot(self, times, rhos, lvl_plot=[1], max_lambda=7 ):
        """(Partially implemented) Save diagnostic arrays and produce basic plots."""

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
        """Generate and save a sequence of Husimi-Q frames for GIF/video rendering."""

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

            # Rotate cavity frame at ω_d to produce a phase-space distribution in the rotating frame.
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
        """Return bare-basis populations P(q, n) = <q,n|ρ|q,n| for a given density matrix."""

        out = np.zeros( (self.qubit_dim, self.cavity_dim) )

        for i in range(self.qubit_dim):
            for n in range(self.cavity_dim):
                out[i,n] = np.abs( rho_in[ i*self.cavity_dim + n,i*self.cavity_dim + n ] )
        
        return out
