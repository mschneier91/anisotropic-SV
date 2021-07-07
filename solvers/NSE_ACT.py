from dolfin import *
from calculate_inf_sup import *
from gen_barycenter_incenter_mesh import *
import sys
import numpy as np
import pdb
import sympy as sym




def NSE_ACT(h, tot_iters, dt, numRefines, refine_type):
    
    mesh0 = RectangleMesh(Point(0,0),Point(1.0,1.0), h, h)
    nu = .001
    t = 0.0

    #Generate the barcenter/incetner refined mesh and calculate the inf-sup constant
    cur_mesh = gen_barycenter_incenter_mesh(mesh0, numRefines, refine_type):
    inf_sup_val = calculate_inf_sup(cur_mesh)

    #Define Expressions for exact functions
    x, y, s = sym.symbols('x[0], x[1], s')
    u1 = sym.cos(y)*(s + sym.exp(s))
    u2 = sym.sin(x)*(s + sym.exp(s))
    #p_var = sym.sin(x)
    p_var = sym.exp(2*s)*(sym.sin(x+y)-2*sin(1)+sin(2))
    f1 = sym.diff(u1,s) - nu*(sym.diff(sym.diff(u1, x), x) + sym.diff(sym.diff(u1, y), y)) + sym.diff(p_var,x) + u1*sym.diff(u1,x) + u2*sym.diff(u1,y)
    f2 = sym.diff(u2,s) - nu*(sym.diff(sym.diff(u2, x), x) + sym.diff(sym.diff(u2, y), y)) + sym.diff(p_var,y) + u1*sym.diff(u2,x) + u2*sym.diff(u2,y)
    u1 = sym.simplify(u1)
    u2 = sym.simplify(u2)
    f1 = sym.simplify(f1)
    f2 = sym.simplify(f2)
    p_var = sym.simplify(p_var)
    u1_code = sym.printing.ccode(u1)
    u2_code = sym.printing.ccode(u2)
    f1_code = sym.printing.ccode(f1)
    f2_code = sym.printing.ccode(f2)
    p_code = sym.printing.ccode(p_var)


    exact_u = Expression((u1_code,u2_code),s=0.0,degree=3)
    exact_p = Expression(p_code,s = 0.0, degree=3)
    f = Expression((f1_code,f2_code),s =0.0, degree=3)


    # Define function spaces
    V = VectorFunctionSpace(cur_mesh, "Lagrange", 2)
    Q = FunctionSpace(cur_mesh, "DG", 1)
    P2 = VectorElement("Lagrange", cur_mesh.ufl_cell(), 2)
    P1 = FiniteElement("DG", cur_mesh.ufl_cell(), 1)
    R = FiniteElement("Real",cur_mesh.ufl_cell(),0)
    TH = P2 * (P1*R)
    W = FunctionSpace(cur_mesh, TH)
    tdof = W.dim()
    print("The total degrees of freedom is:" + str(tdof))
    ###Declare Boundary Conditions##################################################
    def u0_boundary(x, on_boundary):
        return on_boundary

    class OriginPoint(SubDomain):
        def inside(self, x, on_boundary):
            tol = .001
            return (near(x[0], 0.0)  and  near(x[1], 0.0))
    originpoint = OriginPoint()
    
    bcu = DirichletBC(V,exact_u, u0_boundary)
    bcu_s = DirichletBC(W.sub(0),exact_u, u0_boundary)
    bcs = [bcu_s]

    def b(u,v,w):
        return (inner(dot(u,nabla_grad(v)),w))

    # Define functions
    z = TrialFunction(V)
    v = TestFunction(V)
    
    un = Function(V)
    un_1 = Function(V)
    pn = Function(Q)
    
    zn = Function(V)
    w = Function(W)
    
    (chi, p_2) = TrialFunctions(W)
    (v_2, q_2) = TestFunctions(W)

    alpha = 3.0/(2.0*dt)

    # Define variational problems
    a1 = alpha*inner(z,v)*dx + nu*inner(grad(z), grad(v))*dx + b(2*un - un_1,z,v)*dx
    L1 = inner(f,v)*dx + 1.0/(2.0*dt)*inner(4.0*un - un_1,v)*dx 

    a2 = alpha*inner(chi,v_2)*dx + nu*inner(nabla_grad(chi), nabla_grad(v_2))*dx - div(v_2)*p_2[0]*dx + q_2[0]*div(chi)*dx+-p_2[0]*q_2[1]*dx+p_2[1]*q_2[0]*dx
    L2 = alpha*inner(zn,v_2)*dx + (nu*inner(grad(zn), grad(v_2)))*dx

    #Project initial conditions
    un_1.assign(interpolate(exact_u,V))
    t = t + dt
    exact_u.s = t
    un.assign(interpolate(exact_u,V))

    for jj in range(0,tot_iters):
        t = t + dt
        exact_u.s = t
        exact_p.s = t
        f.s = t

        #Solve first linear system
        A1, b1 = assemble_system(a1, L1, bcu)
        solve(A1,zn.vector(),b1)
 
        #Solve second linear system
        A2, b2 = assemble_system(a2, L2, bcs)
        solve(A2,w.vector(),b2)
        un_1.assign(un)
        assign(un,w.sub(0))
        assign(pn,w.sub(1).sub(0))
        
        err_zL2 = errornorm(exact_u, zn, norm_type='L2', degree_rise=3)
        err_uL2 = errornorm(exact_u, un, norm_type='L2', degree_rise=3)
        err_uH1 = errornorm(exact_u, un, norm_type='H1', degree_rise=3)
        err_p = errornorm(exact_p, pn, norm_type='L2', degree_rise=3)
        div_err = norm(un, norm_type='Hdiv0')


        print("The divergence error for u is:" + str(div_err))
        print("The L2 error in the first equation z:" + str(err_zL2))
        print("The L2 error in the velocity u:" + str(err_uL2))
        print("The H1 error in the velocity u:" + str(err_uH1))
        print("The L2 error in the pressure p:" + str(err_p))

    return err_uL2, err_uH1, err_p, inf_sup_val

if __name__ == "__main__":
    h = int(sys.argv[1])
    tot_iters = int(sys.argv[2])
    dt = float(sys.argv[3])
    num_refines = int(sys.argv[4])
    refine_type = sys.argv[5]
    print("The mesh refinement level is:" + str(1/h))
    NSE_ACT(h, tot_iters, dt, num_refines, refine_type)