from dolfin import *
from inf_sup.calculate_inf_sup import *
from meshing.gen_barycenter_incenter_mesh_aspect import *

import numpy as np

tot = 6

#Hold values for barycenter reinfement
inf_sup_bc = np.zeros((tot))
aspect_bc = np.zeros((tot))
rate_hold_bc = np.zeros((tot-1))

#Hold values for incenter reinfement
inf_sup_ic = np.zeros((tot))
aspect_ic = np.zeros((tot))
rate_hold_ic = np.zeros((tot-1))

h = 2
mesh = RectangleMesh(Point(0,0),Point(1.0,1.0), h, h)

for i in range(0,tot):
    numRefines = i+1
    mesh_bc, aspect_bc[i] = gen_barycenter_incenter_mesh_aspect(mesh,numRefines,'bc')
    inf_sup_bc[i] = calculate_inf_sup(mesh_bc)
    mesh_ic, aspect_ic[i] = gen_barycenter_incenter_mesh_aspect(mesh,numRefines,'ic')
    inf_sup_ic[i] = calculate_inf_sup(mesh_ic)

#Calculate the dependence of the inf-sup constant on the aspect ratio
for i in range(0,tot-1):
    rate_hold_bc[i] = np.log(inf_sup_bc[i]/inf_sup_bc[i+1])/np.log(aspect_bc[i]/aspect_bc[i+1])
    rate_hold_ic[i] = np.log(inf_sup_ic[i]/inf_sup_ic[i+1])/np.log(aspect_ic[i]/aspect_ic[i+1])


print("The inf-sup constants for barycenter refinement:")
print(inf_sup_bc)

print("The inf-sup constants for incenter refinement:")
print(inf_sup_ic)

print("The aspect ratios for barycenter refinement:")
print(aspect_bc)

print("The aspect ratios for incenter refinement:")
print(aspect_ic)

print("The rate dependence on the aspect ratio for barycenter refinement:")
print(rate_hold_bc)

print("The rate dependence on the aspect ratio for incenter refinement:")
print(rate_hold_ic)

# np.savetxt(fname1,aspect_hold)
# np.savetxt(fname2,inf_sup_hold)
# np.savetxt(fname3,rate_hold)