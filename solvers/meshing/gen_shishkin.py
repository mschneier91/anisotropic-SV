from dolfin import *
import numpy as np
#import matplotlib.pyplot as plt


def gen_shishkin(N, tau):

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    numv = (N+1)**2 #number of verticies
    numc = 2*N**2 #number of cells
    editor.init_vertices(numv)
    editor.init_cells(numc)

    ct = 0 #counter for vertex
    for j in range(0,N+1):
        for i in range(0,N+1):
            if(i <= N/2):
                editor.add_vertex(ct, np.array([i*2*tau/N,j/N]))
                #print(np.array([i*2*tau/N,j/N]))
                ct = ct + 1
            else:
                editor.add_vertex(ct, np.array([tau + (i - N/2)*2*(1-tau)/N,j/N]))
                #print(np.array([tau + (i - N/2)*2*(1-tau)/N,j/N]))
                ct = ct + 1


    # Adding cells
    ct = 0 #counter for cells
    for i in range(0, N):
        for k in range(0,2):
            for j in range(0,N):
                editor.add_cell(ct, np.array([N+1+i + j*(N+1), i+k + j*(N+1) , i+1 + k +k*N + j*(N+1)], dtype=np.uintp))
                #print(np.array([N+1+i + j*(N+1), i+k + j*(N+1) , i+1 + k + k*N + j*(N+1)]))
                ct = ct + 1

    editor.close()
    #plot(mesh)
    #plt.show()

    return mesh

if __name__ == "__main__":
    N = 16
    tau = .005
    gen_shishkin(N,tau)
