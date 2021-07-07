from dolfin import *
import numpy as np
import pdb

def gen_barycenter_incenter_mesh_aspect(mesh,numRefines,refine_type):

	#We do a barycentric refinement of the mesh so that the Scott-Vogelius element pair is stable
    for iteration in range(numRefines):
        if(iteration == 0):
            a = mesh.cells()
            b = mesh.coordinates()
            numv = mesh.num_vertices()
            numc = mesh.num_cells()
        else:
            a = prev_mesh.cells()
            b = prev_mesh.coordinates()
            numv = prev_mesh.num_vertices()
            numc = prev_mesh.num_cells()

        cur_mesh = Mesh()
        editor = MeshEditor()
        editor.open(cur_mesh, "triangle", 2, 2)
        editor.init_vertices(numc + numv)
        editor.init_cells(3*numc)

        # Adding original vertices
        for i in range(numv):
            editor.add_vertex(i, np.array([b[i,0], b[i,1]]))

        if(refine_type == 'bc'):
        # Adding the barycenter vertices
            for i in range(numv, numv + numc):
                k = i - numv
                editor.add_vertex(i, np.array([(b[a[k,0],0]+b[a[k,1],0]+b[a[k,2],0])/3., (b[a[k,0],1]+b[a[k,1],1]+b[a[k,2],1])/3.]))
        if(refine_type == 'ic'):
        # Adding the incenter vertices
            for i in range(numv, numv + numc):
                k = i - numv
                #calculate the length of the sides
                e1 = np.linalg.norm(b[a[k,1],:] - b[a[k,2],:])
                e2 = np.linalg.norm(b[a[k,2],:] - b[a[k,0],:])
                e3 = np.linalg.norm(b[a[k,0],:] - b[a[k,1],:])
                P = e1 + e2 + e3 #calculate the perimeter 
                editor.add_vertex(i, np.array([(e1*b[a[k,0],0]+e2*b[a[k,1],0]+e3*b[a[k,2],0])/P, (e1*b[a[k,0],1]+e2*b[a[k,1],1]+e3*b[a[k,2],1])/P]))

        # Adding cells
        for i in range(numv, numv + numc):
            k = i - numv
            editor.add_cell(3*k, np.array([i, a[k,1], a[k,2]], dtype=np.uintp))
            editor.add_cell(3*k + 1, np.array([i, a[k,0], a[k,2]], dtype=np.uintp))
            editor.add_cell(3*k + 2, np.array([i, a[k,0], a[k,1]], dtype=np.uintp))

        editor.close()
        prev_mesh = cur_mesh

    #Find the aspect ratio on cur_mesh
    a_cur = cur_mesh.cells()
    b_cur = cur_mesh.coordinates()
    numv_cur = cur_mesh.num_vertices()
    numc_cur = cur_mesh.num_cells()


    aspect_hold = np.zeros(numc_cur)

    for i in range(0,numc_cur): #Loop over all the cells
        #Calculate maxium edge length (this will be local h_T)
        e1 = np.linalg.norm(b_cur[a_cur[i,0],:] - b_cur[a_cur[i,1],:])
        e2 = np.linalg.norm(b_cur[a_cur[i,1],:] - b_cur[a_cur[i,2],:])
        e3 = np.linalg.norm(b_cur[a_cur[i,2],:] - b_cur[a_cur[i,0],:])
        #Calculate local h_T
        h_loc = max(e1,e2,e3)
        #Calculate inradius using Herons formula
        s = (e1+e2+e3)/2.0
        inradius = np.sqrt((s-e1)*(s-e2)*(s-e3)/s) #Herons formula (https://en.wikipedia.org/wiki/Incircle_and_excircles_of_a_triangle)
        #calculate local aspect ratio
        rho_t = inradius/h_loc
        aspect_hold[i] = rho_t

    rho = np.min(aspect_hold) #minimum aspect ratio over all cells

    return cur_mesh, rho

if __name__ == "__main__":
    h = 2
    mesh = RectangleMesh(Point(0,0),Point(1.0,1.0), h, h)
    numRefines = 1
    gen_barycenter_incenter_mesh_aspect(mesh,numRefines,'bc')

