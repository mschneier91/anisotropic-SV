from dolfin import *
from calculate_inf_sup import *
from gen_barycenter_incenter_mesh import *
import sys
import numpy as np
import pdb
import sympy as sym




def stokes_IPM(h, tot_iters, tol, numRefines, refine_type):

    err_uL2 = np.zeros((tot_iters))
    err_uH1 = np.zeros((tot_iters))
    err_p = np.zeros((tot_iters))
    
    mesh0 = RectangleMesh(Point(0,0),Point(1.0,1.0), h, h)


    #Generate the barcenter/incetner refined mesh and calculate the inf-sup constant
    mesh = gen_barycenter_incenter_mesh(mesh0, numRefines, refine_type)
    inf_sup_val = calculate_inf_sup(mesh)

    x, y = sym.symbols('x[0], x[1]')
    # u1 = 2*x*y*(x-1)*(y-1)*-1*x*(x-1)*(2*y-1)
    # u2 = 2*x*y*(x-1)*(y-1)*y*(y-1)*(2*x-1)
    # f1 = -1*(sym.diff(sym.diff(u1, x), x) + sym.diff(sym.diff(u1, y), y))
    # f2 = -1*(sym.diff(sym.diff(u2, x), x) + sym.diff(sym.diff(u2, y), y))
    #u1 = 0
    #u2 = 0
    u1 = sym.cos(x)*sym.sin(y)
    u2 = -1*sym.sin(x)*sym.cos(y)
    p_var = -1*.25*(sym.cos(2.0*x) + sym.cos(2.0*y)-sym.sin(2.0))
    f1 = -1*(sym.diff(sym.diff(u1, x), x) + sym.diff(sym.diff(u1, y), y)) + sym.diff(p_var,x)
    f2 = -1*(sym.diff(sym.diff(u2, x), x) + sym.diff(sym.diff(u2, y), y)) + sym.diff(p_var,y)
    u1_code = sym.printing.ccode(u1)
    u2_code = sym.printing.ccode(u2)
    f1_code = sym.printing.ccode(f1)
    f2_code = sym.printing.ccode(f2)
    p_code = sym.printing.ccode(p_var)


    #Define the exact values of the velocity and pressure
    exact_u = Expression((u1_code,u2_code),degree=5)
    exact_p = Expression(p_code,degree=5)

    #Define the righthand side
    f = Expression((f1_code,f2_code), degree=5)

    #IPM parameters
    g = 100
    r = 100

    # Define Function Spaces
    X = VectorElement("Lagrange",mesh.ufl_cell(),2) #velocity space
    W = FunctionSpace(mesh,X)

    def boundary(x, on_boundary):
        return on_boundary

    noslip = exact_u #Constant((0,0))
    bc = DirichletBC(W,noslip,boundary)

    u = TrialFunction(W)
    v = TestFunction(W)

    a = (inner(grad(u),grad(v)) + g*div(u)*div(v))*dx
    L = dot(f,v)*dx
    w = Function(W)
    solve(a == L, w, bc)
    wi = w
    dvg = sqrt(assemble((div(w)*div(w))*dx))
    print("Initial residual: ", dvg)
    i = 0

    # while dvg > tol and i<tot_iters:
    while i<tot_iters:
        i = i + 1
        L = (dot(f,v) - div(wi)*div(v))*dx
        w = Function(W)
        solve(a == L, w, bc)
        wi = wi + r*w
        dvg = sqrt(assemble((div(w)*div(w))*dx))
        print("\nresidual", dvg)
        print("iteration number:" + str(i))
        err_uL2[i-1] = errornorm(exact_u, w, norm_type='L2', degree_rise=3)
        print("L2 velocity error: ", err_uL2[i-1])
        err_uH1[i-1] = errornorm(exact_u, w, norm_type='H1', degree_rise=3)
        print("H1 velocity error: ", err_uH1[i-1])
        err_p[i-1] = sqrt(assemble((-div(wi)-exact_p)*(-div(wi)-exact_p)*dx))
        print("L2 pressure error: ", err_p[i-1])

    # Compute error
    # err_uL2 = errornorm(exact_u, w, norm_type='L2', degree_rise=3)
    # err_uH1 = errornorm(exact_u, w, norm_type='H1', degree_rise=3)
    # err_p = sqrt(assemble((div(wi)-exact_p)*(div(wi)-exact_p)*dx))
    #div_err = norm(un, norm_type='Hdiv0')

    # eu  = (w - exact_u )
    # err_uL2 = sqrt(assemble(dot(eu,eu)*dx))
    # err_uH1 = sqrt(assemble((inner(grad(eu),grad(eu)))*dx))
    # err_p = sqrt(assemble((div(wi)-exact_p)*(div(wi)-exact_p)*dx))

    return err_uL2, err_uH1, err_p, inf_sup_val, i

if __name__ == "__main__":
    h = int(sys.argv[1])
    tot_iters = int(sys.argv[2])
    num_refines = int(sys.argv[3])
    refine_type = sys.argv[4]
    print("The mesh refinement level is:" + str(1/h))
    #NSE_ACT(h, tot_iters, dt, num_refines, refine_type)
