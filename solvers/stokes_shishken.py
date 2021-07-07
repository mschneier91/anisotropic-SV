#import matplotlib.pyplot as plt
from dolfin import *
from meshing.gen_barycenter_incenter_mesh_aspect import *
from meshing.gen_shishkin import *
from inf_sup.calculate_inf_sup import *

import sympy as sym

def stokes_shishken(N,eps_val,numRefines,refine_type):
    tau = min(.5,3*eps_val*abs(np.log(eps_val)))
    mesh = gen_shishkin(N, tau)
    mesh, aspect_ratio = gen_barycenter_incenter_mesh_aspect(mesh,numRefines,refine_type)
    inf_sup_val = calculate_inf_sup(mesh)


    x, y, epsilon = sym.symbols('x[0], x[1], epsilon')
    xi = x*x*(1-x)*(1-x)*y*y*(1-y)*(1-y)*sym.exp(-1*x/epsilon)
    u1 = sym.diff(xi,y)
    u2 = -1*sym.diff(xi,x)
    p = sym.exp(-1*x/epsilon) - epsilon + sym.exp(-1*x/epsilon)*epsilon
    f1 = -1*(sym.diff(sym.diff(u1, x), x) + sym.diff(sym.diff(u1, y), y)) + sym.diff(p,x)
    f2 = -1*(sym.diff(sym.diff(u2, x), x) + sym.diff(sym.diff(u2, y), y)) + sym.diff(p,y)
    u1 = sym.simplify(u1)
    u2 = sym.simplify(u2)
    p = sym.simplify(p)
    f1 = sym.simplify(f1)
    f2 = sym.simplify(f2)
    u1_code = sym.printing.ccode(u1)
    u2_code = sym.printing.ccode(u2)
    p_code = sym.printing.ccode(p)
    f1_code = sym.printing.ccode(f1)
    f2_code = sym.printing.ccode(f2)

    # print('u1 =', u1_code)
    # print('u2 =', u2_code)
    # print('f1 =', f1_code)
    # print('f2 =', f2_code)

    #Define the exact values of the velocity and pressure
    exact_u = Expression((u1_code,u2_code), epsilon = eps_val, degree=3)
    exact_p = Expression(p_code,epsilon = eps_val, degree=3)
    f = Expression((f1_code,f2_code), epsilon = eps_val, degree=3)
    
    # Define function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("DG", mesh.ufl_cell(), 1)
    R = FiniteElement("Real",mesh.ufl_cell(),0)
    TH = P2 * (P1*R)
    W = FunctionSpace(mesh, TH)
    tdof = W.dim()
    print("The total velocity plus pressure degrees of freedom is:" + str(tdof))
    ###Declare Boundary Conditions##################################################
    def u0_boundary(x, on_boundary):
        return on_boundary
    
    bcu = DirichletBC(V,exact_u, u0_boundary)
    bcu_s = DirichletBC(W.sub(0),exact_u, u0_boundary)
    bcs = [bcu_s]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Functions for holding the solutions
    un = Function(V)
    pn = Function(Q)

    #Define the weak formulation
    a = inner(nabla_grad(u), nabla_grad(v))*dx - div(v)*p[0]*dx + q[0]*div(u)*dx+-p[0]*q[1]*dx+p[1]*q[0]*dx
    L = inner(f,v)*dx 

    # Compute solution
    w = Function(W)
    solve(a == L, w, bcs)
    
    # Assign the solutions
    assign(un,w.sub(0))
    assign(pn,w.sub(1).sub(0))
    
    velocity_paraview_file = File("para_plotting/shisken_velocity_solution.pvd")
    pressure_paraview_file = File("para_plotting/shisken_pressure_solution.pvd")
    
    velocity_paraview_file << un
    pressure_paraview_file << pn

    div_err = norm(un, norm_type='Hdiv0')
    print("The divergence error for u is:" + str(div_err))

    err_uL2 = errornorm(exact_u, un, norm_type='L2', degree_rise=3)
    err_uH1 = errornorm(exact_u, un, norm_type='H1', degree_rise=3)
    err_p = errornorm(exact_p, pn, norm_type='L2', degree_rise=3)
    
    print("The L2 error in the velocity:" + str(err_uL2))
    print("The H1 error in the velocity:" + str(err_uH1))
    print("The L2 error in the pressure:" + str(err_p))
    print("The aspect ratio is:" + str(aspect_ratio))
    print("The inf-sup constant is:" + str(inf_sup_val))
    return err_uL2,err_uH1,err_p, aspect_ratio, inf_sup_val

if __name__ == "__main__":
    N = 16
    eps_val = .01
    numRefines = 3
    refine_type = 'ic'
    err_uL2,err_uH1,err_p, aspect_ratio, inf_sup_val = stokes_shishken(N,eps_val,numRefines,refine_type)    

