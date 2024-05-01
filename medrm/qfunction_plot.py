import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm



filename= "tmax1199_Nq2_Nc30_amp0.0060_kappa0.0014_wq4.4619_anh-0.1592_wc7.4150_wd7.4215_ms1e-03_dimexp120_qb[0,1]_11"
qfunction_arr= np.loadtxt( 'data/qfunction_'+filename+'.d' )

max_lambda = 9

#-- normal scale
interval = np.linspace(0.00, 1.0)
colors = plt.cm.magma(interval)
my_cmap = LinearSegmentedColormap.from_list('name', colors)

#plt.cbrange( 0.5 )
plt.imshow( qfunction_arr, extent=(-max_lambda, max_lambda, -max_lambda, max_lambda), cmap=my_cmap, vmin=0, vmax=0.5 )
plt.colorbar()
plt.savefig( "figures/qfunction_" + filename + '.pdf', format='pdf'  )
plt.show()

#-- log scale
interval = np.linspace(0.00, qfunction_arr.max())
colors = plt.cm.magma(interval)
zmin, zmax = 0.005, qfunction_arr.max()
norm=LogNorm(vmin=zmin, vmax=zmax)

my_cmap = LinearSegmentedColormap.from_list('name', colors)
ext = (-max_lambda, max_lambda, -max_lambda, max_lambda)
plt.imshow( qfunction_arr, extent=ext, norm=norm, cmap=my_cmap )
plt.colorbar()
plt.savefig( "figures/LOG_qfunction_" + filename + '.pdf', format='pdf'  )
plt.show()

