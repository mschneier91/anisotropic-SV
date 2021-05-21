from dolfin import *

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import math
import numpy as np
from mshr import *

import WorseyFarin as WF

# Define mesh size
ms = 8

# Define auxiliary mesh
m = UnitCubeMesh(ms,ms,ms)

# Define WF refined mesh
mesh,interiorSplitPoints,boundarySplitPoints = WF.WorseyFarin12(m)

mesh.init(3,0)
mesh.init(0,3)
nISV = len(interiorSplitPoints) #number of interior singular vertices in mesh
nBSV = len(boundarySplitPoints) #number of boundary singular vertices in mesh

print('Number of interior singular vertices', nISV)
print('Number of boundary singular vertices', nBSV)

#exact data
g = 100
r = 100
gx0 = Expression('pow(x[0] - pow(x[0],2.),2.)', degree=3, domain = mesh)
gx1 = Expression('2.*x[0] - 6.*pow(x[0],2.) + 4.*pow(x[0],3.)', degree=3, domain = mesh)
gx2 = Expression('2. - 12.*x[0] + 12.*pow(x[0],2.)', degree=2, domain = mesh)
gx3 = Expression('-12. + 24.*x[0]', degree=1, domain = mesh)

gy0 = Expression('pow(x[1] - pow(x[1],2.),2.)', degree=3, domain = mesh)
gy1 = Expression('2.*x[1] - 6.*pow(x[1],2.) + 4.*pow(x[1],3.)', degree=3, domain = mesh)
gy2 = Expression('2. - 12.*x[1] + 12.*pow(x[1],2.)', degree=2, domain = mesh)
gy3 = Expression('-12. + 24.*x[1]', degree=1, domain = mesh)

gz0 = Expression('pow(x[2] - pow(x[2],2.),2.)', degree=3, domain = mesh)
gz1 = Expression('2.*x[2] - 6.*pow(x[2],2.) + 4.*pow(x[2],3.)', degree=3, domain = mesh)
gz2 = Expression('2. - 12.*x[2] + 12.*pow(x[2],2.)', degree=2, domain = mesh)
gz3 = Expression('-12. + 24.*x[2]', degree=1, domain = mesh)

dgxxy = Expression('dgx2*dgy1*dgz0', dgx2 = gx2, dgy1 = gy1, dgz0 = gz0, degree=3, domain = mesh)
dgyyy = Expression('dgx0*dgz0*dgy3', dgx0 = gx0, dgz0 = gz0, dgy3 = gy3, degree=3, domain = mesh)
dgyzz = Expression('dgx0*dgy1*dgz2', dgx0 = gx0, dgy1 = gy1, dgz2 = gz2, degree=3, domain = mesh)
dgxxz = Expression('dgy0*dgx2*dgz1', dgx2 = gx2, dgy0 = gy0, dgz1 = gz1, degree=3, domain = mesh)
dgyyz = Expression('dgx0*dgz1*dgy2', dgx0 = gx0, dgz1 = gz1, dgy2 = gy2, degree=3, domain = mesh)
dgzzz = Expression('dgx0*dgy0*dgz3', dgx0 = gx0, dgy0 = gy0, dgz3 = gz3, degree=3, domain = mesh)
dgxxx = Expression('dgy0*dgz0*dgx3', dgx3 = gx3, dgy0 = gy0, dgz0 = gz0, degree=3, domain = mesh)
dgxyy = Expression('dgz0*dgx1*dgy2', dgx1 = gx1, dgy2 = gy2, dgz0 = gz0, degree=3, domain = mesh)
dgxzz = Expression('dgy0*dgx1*dgz2', dgx1 = gx1, dgy0 = gy0, dgz2 = gz2, degree=3, domain = mesh)
dgxyz = Expression('dgx1*dgy1*dgz1', dgx1 = gx1, dgy1 = gy1, dgz1 = gz1, degree=3, domain = mesh)
dgxy  = Expression('dgz0*dgx1*dgy1', dgx1 = gx1, dgy1 = gy1, dgz0 = gz0, degree=3, domain = mesh)
dgx   = Expression('dgy0*dgz0*dgx1', dgx1 = gx1, dgy0 = gy0, dgz0 = gz0, degree=3, domain = mesh)
dgy   = Expression('dgx0*dgz0*dgy1', dgx0 = gx0, dgy1 = gy1, dgz0 = gz0, degree=3, domain = mesh)
dgz   = Expression('dgx0*dgy0*dgz1', dgx0 = gx0, dgy0 = gy0, dgz1 = gz1, degree=3, domain = mesh)

f1 = Expression('-(8./9.)*gxxy - gyyy - gyzz + gxxz + gyyz + gzzz', gxxy = dgxxy, gyyy = dgyyy, gyzz = dgyyz, gxxz = dgxxz, gyyz = dgyyz, gzzz = dgzzz, degree=3, domain = mesh)
f2 = Expression('gxxx + (10./9.)*gxyy + gxzz', gxxx = dgxxx, gxyy = dgxyy, gxzz = dgxzz, degree=3, domain = mesh)
f3 = Expression('-gxxx - gxyy - gxzz + (1./9.)*gxyz', gxxx = dgxxx, gxyy = dgxyy, gxzz = dgxzz, gxyz = dgxyz, degree=3, domain = mesh)
f = Expression((('4096.*fx','4096.*fy','4096.*fz')), fx = f1, fy = f2,fz = f3,degree=3, domain = mesh)
Exactu = Expression((('4096.*(gy - gz)','-4096.*gx','4096.*gx')), gy = dgy, gz = dgz, gx = dgx, degree=3, domain = mesh)
Exactp = Expression('(4096./9.)*gxy', gxy = dgxy, degree=3, domain = mesh)
tol = 1E-7

# define spaces
X = VectorElement("Lagrange",mesh.ufl_cell(),1) #velocity space

W = FunctionSpace(mesh,X)

def boundary(x, on_boundary):
	return on_boundary

noslip = Constant((0,0,0))
bc = DirichletBC(W,noslip,boundary)

u = TrialFunction(W)
v = TestFunction(W)

a = (inner(grad(u),grad(v)) + g*div(u)*div(v))*dx
L = dot(f,v)*dx
w = Function(W)
#solve(a == L, w, bc, solver_parameters={"linear_solver": "cg"})
solve(a == L, w, bc)
wi = w
dvg = sqrt(assemble((div(w)*div(w))*dx))
print(dvg)
i = 0

while dvg > tol and i<20:
	L = (dot(f,v) - div(wi)*div(v))*dx
	w = Function(W)
	solve(a == L, w, bc)
	#solve(a == L, w, bc, solver_parameters={"linear_solver": "cg"})
	wi = wi + r*w
	dvg = sqrt(assemble((div(w)*div(w))*dx))
	print(dvg)
	i = i + 1
	

# Comput error
eu  = (w - Exactu)
print('Velocity L2 error:',sqrt(assemble(dot(eu,eu)*dx)))
print('Velocity H1 error:',sqrt(assemble((inner(grad(eu),grad(eu)))*dx)))
print('Pressure L2 error:',sqrt(assemble((div(wi)-Exactp)*(div(wi)-Exactp)*dx)))
