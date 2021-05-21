from fenics import *
import numpy as np


def WorseyFarin12(m):

	a = m.cells()
	b = m.coordinates()
	numv = m.num_vertices()
	numc = m.num_cells()
	nume = m.init(2)
	auxver = np.zeros(shape=(numv + numc,3))

	# initiate the mapping between cells and edges

	m.init(2,3)
	# Define WF refined mesh

	mesh = Mesh()
	editor = MeshEditor()
	editor.open(mesh,'tetrahedron',3,3)
	editor.init_vertices(numv + numc + nume)
	editor.init_cells(12*numc)

	# Adding original vertices

	for i in range(numv):
    		auxver[i] = [b[i,0], b[i,1],b[i,2]]
    		editor.add_vertex(i, np.array([b[i,0], b[i,1],b[i,2]]))

	# Create and Store Incenters
	# Loop over all elements in OG mesh
	faceAreas = np.zeros(4)
	for i in range(numc):
    
    		faceAreas = [Face(m,Cell(m,i).entities(2)[j]).area() for j in range(4)]
    
    		X = b[a[i,:],:].transpose()

    		in_center = X.dot(faceAreas)/np.sum(faceAreas)
    
    		auxver[i+numv] = in_center
    		editor.add_vertex(i+numv, in_center)



# Finding the intersection point between the face and the line connecting the incenters
# and start adding 6 tetrahedra around that point on the fly in order
# c is a counter to count the number of intersection points
	c=0
	interiorSplitPoints = []
	# Loop over each internal faces in the m
	for face in faces(m):
    		cell_idx = face.entities(3)
    		if len(cell_idx) == 2: #interior face

        		verts = face.entities(0)
        		E1 = cell_idx[0]; E2 = cell_idx[1]
        		v_E1 = auxver[numv + E1,:]; v_E2 = auxver[numv + E2,:]
        
        		baryF = [face.midpoint().x(),face.midpoint().y(),face.midpoint().z()] #barycenter of face

        		n    = [face.normal().x(),face.normal().y(),face.normal().z()] # normal vector for current face
        
        		t = np.dot(n,baryF-v_E1)/np.dot(n,v_E2-v_E1)
        
        	#point where line intersects the face
        	#split_point_face = np.array(baryF)
        		split_point_face = v_E1 + t * (v_E2-v_E1) # equation of a line
        		editor.add_vertex(numv + numc + c, split_point_face)

        	# Adding Tetrahedra in order
        		editor.add_cell(6*c, np.array([verts[1], verts[2], numv + E1, numv + numc + c], dtype=np.uintp))
        		editor.add_cell(6*c + 1, np.array([verts[2], verts[0], numv + E1, numv + numc + c], dtype=np.uintp))
        		editor.add_cell(6*c + 2, np.array([verts[0], verts[1], numv + E1, numv + numc + c], dtype=np.uintp))
        		editor.add_cell(6*c + 3, np.array([verts[2], verts[1], numv + E2, numv + numc + c], dtype=np.uintp))
        		editor.add_cell(6*c + 4, np.array([verts[0], verts[2], numv + E2, numv + numc + c], dtype=np.uintp))
        		editor.add_cell(6*c + 5, np.array([verts[1], verts[0], numv + E2, numv + numc + c], dtype=np.uintp))
        		interiorSplitPoints.append(numv+numc+c)

        		c = c+1

	d = 0
	boundarySplitPoints = []
# Loop over each external faces in the m and ading 3 tetrahedra on the fly
	for face in faces(m):
    		cell_idx = face.entities(3)
    		if len(cell_idx) == 1:
        		verts = face.entities(0)
        
        		baryF = [face.midpoint().x(),face.midpoint().y(),face.midpoint().z()] #barycenter of face
        
        		editor.add_vertex(numv + numc + c + d, np.array(baryF))
        		editor.add_cell(6*c + 3*d, np.array([verts[0], verts[1], numv + cell_idx[0], numv + numc + c + d], dtype=np.uintp))
        		editor.add_cell(6*c + 3*d + 1, np.array([verts[1], verts[2], numv + cell_idx[0], numv + numc + c + d], dtype=np.uintp))
        		editor.add_cell(6*c + 3*d + 2, np.array([verts[2], verts[0], numv + cell_idx[0], numv + numc + c + d], dtype=np.uintp))
        		boundarySplitPoints.append(numv+numc+c+d)
        		d = d + 1

	editor.close()

	return mesh,interiorSplitPoints,boundarySplitPoints
