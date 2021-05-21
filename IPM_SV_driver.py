from stokes_IPM import *
import numpy as np
import matplotlib.pyplot as plt

tot = 3

#Hold error values for barycenter reinfement
unormL2_bc = np.zeros((tot))
unormH1_bc = np.zeros((tot))
pnorm_bc = np.zeros((tot))
inf_sup_bc = np.zeros((tot))
iterhold_bc = np.zeros((tot))

#Hold error values for incenter reinfement
unormL2_ic = np.zeros((tot))
unormH1_ic = np.zeros((tot))
pnorm_ic = np.zeros((tot))
inf_sup_ic = np.zeros((tot))
iterhold_ic = np.zeros((tot))

h_vals = np.zeros((tot))


tot_iters = 20
tol = 1e-7
numRefines = 2

h = 2
for i in range(0,tot):
    h = 2**(i+1)
    refine_type = 'bc'
    unormL2_bc[i],unormH1_bc[i],pnorm_bc[i],inf_sup_bc[i],iterhold_bc[i]= stokes_IPM(h, tot_iters, tol, numRefines, refine_type)

h = 2
for i in range(0,tot):
    h = 2**(i+1)
    refine_type = 'ic'
    unormL2_ic[i],unormH1_ic[i],pnorm_ic[i],inf_sup_ic[i],iterhold_ic[i] = stokes_IPM(h, tot_iters, tol, numRefines, refine_type)


fname1 = './results/ipm_unormL2_bc.txt'
fname2 = './results/ipm_unormL2_ic.txt'
fname3 = './results/ipm_pnormL2_bc.txt'
fname4 = './results/ipm_pnormL2_ic.txt'
fname5 = './results/ipm_inf_sup_bc.txt'
fname6 = './results/ipm_inf_sup_ic.txt'

np.savetxt(fname1,unormL2_bc)
np.savetxt(fname2,unormL2_ic)
np.savetxt(fname3,pnorm_bc)
np.savetxt(fname4,pnorm_ic)
np.savetxt(fname5,inf_sup_bc)
np.savetxt(fname6,inf_sup_ic)


fig1, ax1 = plt.subplots()
ax1.plot(inf_sup_bc,unormL2_bc,'o-', label='Barycenter')
ax1.plot(inf_sup_ic,unormL2_ic,'ro-',label='Incenter')
ax1.set_xlabel("inf-sup constant")
ax1.set_ylabel("L2 error velocity")
ax1.legend()
fig1.savefig("./results/ipm_velocity_ic_v_bc")
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(inf_sup_bc,pnorm_bc,'o-',label='Barycenter')
ax2.plot(inf_sup_ic,pnorm_ic,'ro-',label='Incenter')
ax2.set_xlabel("inf-sup constant")
ax2.set_ylabel("L2 error pressure")
ax2.legend()
fig2.savefig("./results/ipm_pressure_ic_v_bc")
plt.show()

print(iterhold_bc)
print(iterhold_ic)

