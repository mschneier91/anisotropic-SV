from NSE_ACT import *
import numpy as np
import matplotlib.pyplot as plt

tot = 6

#Hold error values for barycenter reinfement
unormL2_bc = np.zeros((tot))
unormH1_bc = np.zeros((tot))
pnorm_bc = np.zeros((tot))
inf_sup_bc = np.zeros((tot))

#Hold error values for incenter reinfement
unormL2_ic = np.zeros((tot))
unormH1_ic = np.zeros((tot))
pnorm_ic = np.zeros((tot))
inf_sup_ic = np.zeros((tot))

h_vals = np.zeros((tot))

num_steps = 4
dt = .0001
h = 2

for i in range(0,tot):
    numRefines = i+1
    refine_type = 'bc'
    unormL2_bc[i],unormH1_bc[i],pnorm_bc[i],inf_sup_bc[i] = NSE_ACT(h, num_steps, dt, numRefines, refine_type)

for i in range(0,tot):
    numRefines = i+1
    refine_type = 'ic'
    unormL2_ic[i],unormH1_ic[i],pnorm_ic[i],inf_sup_ic[i] = NSE_ACT(h, num_steps, dt, numRefines, refine_type)


fname1 = './results/unormL2_bc.txt'
fname2 = './results/unormL2_ic.txt'
fname3 = './results/pnormL2_bc.txt'
fname4 = './results/pnormL2_ic.txt'
fname5 = './results/inf_sup_bc.txt'
fname6 = './results/inf_sup_ic.txt'

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
fig1.savefig("./results/velocity_ic_v_bc")
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(inf_sup_bc,pnorm_bc,'o-',label='Barycenter')
ax2.plot(inf_sup_ic,pnorm_ic,'ro-',label='Incenter')
ax2.set_xlabel("inf-sup constant")
ax2.set_ylabel("L2 error pressure")
ax2.legend()
fig2.savefig("./results/pressure_ic_v_bc")
plt.show()
# plt.plot(inf_sup_bc,pnorm_bc,"barycenter")
# plt.plot(inf_sup_ic,pnorm_ic,"incenter")
# plt.xlabel("h")
# plt.ylabel("L2 error velocity")
# plt.legend()
# plt.savefig("./results/pressure_ic_v_bc")
# plt.close()
