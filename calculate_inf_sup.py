from dolfin import *
import numpy as np

def calculate_inf_sup(mesh):

    Vel = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Press= FiniteElement("DG", mesh.ufl_cell(), 1)
    SV = Vel * Press
    W = FunctionSpace(mesh, SV)
    print("Dofs:" + str(W.dim()))


    ###Declare Boundary Conditions##################################################
    # No-slip boundary condition for velocity
    def u0_boundary(x, on_boundary):
        return on_boundary
    bcu = DirichletBC(W.sub(0), Constant((0, 0)), u0_boundary)
    bcs = [bcu]

    #Define variational problem for the inf-sup constant. This is equivalent to [A -B^T;B 0] = [0, 0;0, M]
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    A_form = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
    B_form = inner(p,q)*dx

    #Dummy variable for assemble system call
    dummy = v[0]*dx


    A = PETScMatrix()
    B = PETScMatrix()


    #Assemble form using assemble system.
    assemble_system(A_form, dummy, bcs, A_tensor=A)
    assemble_system(B_form, dummy, A_tensor=B)



    #Establish solver
    solver = SLEPcEigenSolver(A, B)

    solver.parameters["solver"] = "arpack"
    solver.parameters["tolerance"] = 1e-8
    solver.parameters["spectrum"] = "target magnitude"
    solver.parameters["spectral_transform"] = "shift-and-invert"
    solver.parameters["spectral_shift"] = .000001
    neigs = 3
    solver.solve(neigs)

    print("The total number of eigenvalues convereged is:" + str(solver.get_number_converged()))
    computed_eigenvalues = []
    for i in range(0,solver.get_number_converged()):
        r,_ = solver.get_eigenvalue(i)
        computed_eigenvalues.append(r)
    computed_eigenvalues = np.sort(np.array(computed_eigenvalues))

    #Note since we did not perscribe a condition for the pressure we expect the first eigenvalue to be 0. The infsup constant will be the
    #square root of the second eigenvalue
    for i in range(0,neigs):
        print("square root of eigenvalue #" + str(i+1) + ":" + str(np.sqrt(computed_eigenvalues[i])))

    return np.sqrt(computed_eigenvalues[1])
