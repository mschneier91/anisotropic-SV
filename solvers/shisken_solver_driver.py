from dolfin import *
from inf_sup.calculate_inf_sup import *
from meshing.gen_shishkin import *
from meshing.gen_barycenter_incenter_mesh_aspect import *
from stokes_shishken import *


import matplotlib.pyplot as plt
import numpy as np

tot = 5

#Hold error values for barycenter reinfement
unormL2_bc = np.zeros((tot))
unormH1_bc = np.zeros((tot))
pnorm_bc = np.zeros((tot))
inf_sup_bc = np.zeros((tot))
aspect_bc = np.zeros((tot))

#Hold error values for incenter reinfement
unormL2_ic = np.zeros((tot))
unormH1_ic = np.zeros((tot))
pnorm_ic = np.zeros((tot))
inf_sup_ic = np.zeros((tot))
aspect_ic = np.zeros((tot))

#Stokes Shisken input values
eps_val = .01
tau = min(.5,3*eps_val*abs(np.log(eps_val)))
numRefines = 1

for i in range(0,tot):
    N=8*(1+i)
    unormL2_bc[i], unormH1_bc[i],pnorm_bc[i], aspect_bc[i], inf_sup_bc[i] = stokes_shishken(N,eps_val,numRefines,'bc')
    unormL2_ic[i], unormH1_ic[i],pnorm_ic[i], aspect_ic[i], inf_sup_ic[i] = stokes_shishken(N,eps_val,numRefines,'ic')


xax = np.arange(8,8*(tot+1),8)

fig1, ax1 = plt.subplots()
ax1.semilogy(xax,unormL2_bc,'o-', label='Barycenter')
ax1.semilogy(xax,unormL2_ic,'ro-',label='Incenter')
ax1.set_xlabel("N")
ax1.set_ylabel("L2 error velocity")
ax1.legend()
fig1.savefig("./results/shisken_L2_velocity_ic_v_bc")
plt.show()

fig2, ax2 = plt.subplots()
ax2.semilogy(xax,unormH1_bc,'o-', label='Barycenter')
ax2.semilogy(xax,unormH1_ic,'ro-',label='Incenter')
ax2.set_xlabel("N")
ax2.set_ylabel("H1 error velocity")
ax2.legend()
fig2.savefig("./results/shisken_H1_velocity_ic_v_bc")
plt.show()

fig3, ax3 = plt.subplots()
ax3.semilogy(xax,pnorm_bc,'o-', label='Barycenter')
ax3.semilogy(xax,pnorm_ic,'ro-',label='Incenter')
ax3.set_xlabel("N")
ax3.set_ylabel("L2 error pressure")
ax3.legend()
fig3.savefig("./results/shisken_L2_pressure_ic_v_bc")
plt.show()

print("The inf-sup constants for barycenter refinement:")
print(inf_sup_bc)

print("The inf-sup constants for incenter refinement:")
print(inf_sup_ic)

print("The aspect ratios for barycenter refinement:")
print(aspect_bc)

print("The aspect ratios for incenter refinement:")
print(aspect_ic)