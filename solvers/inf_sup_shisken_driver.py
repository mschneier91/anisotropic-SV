from dolfin import *
from inf_sup.calculate_inf_sup import *
from meshing.gen_shishkin import *
from meshing.gen_barycenter_incenter_mesh_aspect import *

import numpy as np

tot = 6

#Hold values for barycenter reinfement
inf_sup_bc = np.zeros((tot))
aspect_bc = np.zeros((tot))

#Hold values for incenter reinfement
inf_sup_ic = np.zeros((tot))
aspect_ic = np.zeros((tot))

#Stokes Shisken input values
N = 8
eps_val = .01
tau = min(.5,3*eps_val*abs(np.log(eps_val)))

for i in range(0,tot):
    numRefines = i+1
    mesh = gen_shishkin(N, tau)
    mesh_bc, aspect_bc[i] = gen_barycenter_incenter_mesh_aspect(mesh,numRefines,'bc')
    inf_sup_bc[i] = calculate_inf_sup(mesh_bc)
    mesh = gen_shishkin(N, tau)
    mesh_ic, aspect_ic[i] = gen_barycenter_incenter_mesh_aspect(mesh,numRefines,'ic')
    inf_sup_ic[i] = calculate_inf_sup(mesh_ic)

#Calculate the dependence of the inf-sup constant on the aspect ratio
# for i in range(0,tot-1):
#     rate_hold_bc[i] = np.log(inf_sup_bc[i]/inf_sup_bc[i+1])/np.log(aspect_bc[i]/aspect_bc[i+1])
#     rate_hold_ic[i] = np.log(inf_sup_ic[i]/inf_sup_ic[i+1])/np.log(aspect_ic[i]/aspect_ic[i+1])


# xax = np.arange(1,tot_iters+1)
# fig1, ax1 = plt.subplots()
# ax1.plot(xax,unormL2_bc,'o-', label='Barycenter')
# ax1.plot(xax,unormL2_ic,'ro-',label='Incenter')
# ax1.set_xlabel("iteration numbetr")
# ax1.set_ylabel("L2 error velocity")
# ax1.legend()
# fig1.savefig("./results/ipm_velocity_ic_v_bc")
# plt.show()

# fig2, ax2 = plt.subplots()
# ax2.plot(xax,pnorm_bc,'o-',label='Barycenter')
# ax2.plot(xax,pnorm_ic,'ro-',label='Incenter')
# ax2.set_xlabel("iteration number")
# ax2.set_ylabel("L2 error pressure")
# ax2.legend()
# fig2.savefig("./results/ipm_pressure_ic_v_bc")
# plt.show()

print("The inf-sup constants for barycenter refinement:")
print(inf_sup_bc)

print("The inf-sup constants for incenter refinement:")
print(inf_sup_ic)

# print("The aspect ratios for barycenter refinement:")
# print(aspect_bc)

# print("The aspect ratios for incenter refinement:")
# print(aspect_ic)

# print("The rate dependence on the aspect ratio for barycenter refinement:")
# print(rate_hold_bc)

# print("The rate dependence on the aspect ratio for incenter refinement:")
# print(rate_hold_ic)

# np.savetxt(fname1,aspect_hold)
# np.savetxt(fname2,inf_sup_hold)
# np.savetxt(fname3,rate_hold)