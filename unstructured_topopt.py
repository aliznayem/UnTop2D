'''
UnTop2D: Calculation Module
'''


import copy, time, meshio
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# If not found in input file
DEFULT_MAX_ITER = 250
DEFAULT_MAX_X_CHANGE_THRESH = 0.01


class Node:

    def __init__(self, node_id, x, y):
        '''
        Parameters:
            node_id: int
                Node index
            x: float
                x-coordinate
            y: float
                y-coordinate
        '''

        self.id = node_id
        self.x = x
        self.y = y

        # Displacements
        self.u = None
        self.v = None

        # DOF
        self.dof_x = int(self.id*2 -1)  # 1 indexing
        self.dof_y = int(self.id*2)  # 1 indexing

        # DOF constraint; Boundary condition
        self.dof_x_constr = False
        self.dof_y_constr = False

        # DOF load
        self.dof_x_load = 0.0
        self.dof_y_load = 0.0

        # Parent elements lst
        self.parent_elem_id_lst = []

    def assign_id(self, _id):
        self.id = _id

        self.dof_x = int(self.id*2 -1)
        self.dof_y = int(self.id*2)

    def assign_parent_elem_id(self, element):
        if not element.id in self.parent_elem_id_lst:
            self.parent_elem_id_lst.append(element.id)

    def remove_parent_elem_id(self, element):
        if element.id in self.parent_elem_id_lst:
            self.parent_elem_id_lst.remove(element.id)

    def clear_parent_elem_id(self):
        self.parent_elem_id_lst = []


class Element:

    def __init__(self, elem_id, node1, node2, node3, node4, E, Nu, t, Xe, penal):
        '''
        Parameters:
            elem_id: int
                Element index
            node1: Node class instance
                First node
            node2: Node class instance
                Second node
            node3: Node class instance
                Third node
            node4: Node class instance
                Fourth node
            E: float
                Modulus of elasticity
            Nu: float
                Poisson's ratio
            t: float
                Thickness of element
            Xe: float
                Vector of design variables
            penal: float
                Penalization power
        '''

        self.id = elem_id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.node_lst = [node1, node2, node3, node4]
        self.assign_elem_id_to_nodes()
        self.E = E
        self.Nu = Nu
        self.t = t
        self.Xe = Xe
        self.penal = penal
        # Find Center coordinate (xg, yg)
        self.find_elem_center()

        # Edge lengths
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.e4 = None
        # Compliance
        self.Ce = None
        # Gradient of compliance
        self.dCe = None
        # on/off state
        self.exist = True
        # Element stiffness matrix
        self.Ke = None
        # adjacent element list; list of elements which are inside rmin radius
        self.adjacent_elem_lst = []
        # if the element is on boundary
        self.is_boundary = True

    def assign_id(self, _id):
        self.id = _id

    def set_elem_on_off(self, exist=True):
        self.exist = exist
        if self.exist:
            self.node1.assign_parent_elem_id(self)
            self.node2.assign_parent_elem_id(self)
            self.node3.assign_parent_elem_id(self)
            self.node4.assign_parent_elem_id(self)
        else:
            self.node1.remove_parent_elem_id(self)
            self.node2.remove_parent_elem_id(self)
            self.node3.remove_parent_elem_id(self)
            self.node4.remove_parent_elem_id(self)

    def assign_elem_id_to_nodes(self):
        self.node1.assign_parent_elem_id(self)
        self.node2.assign_parent_elem_id(self)
        self.node3.assign_parent_elem_id(self)
        self.node4.assign_parent_elem_id(self)

    def find_elem_center(self):
        self.xg = (self.node1.x+self.node2.x+self.node3.x+self.node4.x)/4.0
        self.yg = (self.node1.y+self.node2.y+self.node3.y+self.node4.y)/4.0

    def distance(self, elem):
        return np.sqrt((self.xg-elem.xg)**2 + (self.yg-elem.yg)**2)

    def append_adjacent_elem(self, elem):
        self.adjacent_elem_lst.append(elem)
        if len(self.adjacent_elem_lst) > 3:
            self.is_boundary = False

    def remove_adjacent_elem(self, elem):
        self.adjacent_elem_lst.remove(elem)

    def clear_adjacent_elem_lst(self):
        self.adjacent_elem_lst = []

    def calc_edge_lengths(self):
        self.e1 = np.sqrt((self.node2.x-self.node1.x)**2 + (self.node2.y-self.node1.y)**2)
        self.e2 = np.sqrt((self.node3.x-self.node2.x)**2 + (self.node3.y-self.node2.y)**2)
        self.e3 = np.sqrt((self.node4.x-self.node3.x)**2 + (self.node4.y-self.node3.y)**2)
        self.e4 = np.sqrt((self.node1.x-self.node4.x)**2 + (self.node1.y-self.node4.y)**2)

    def calc_area(self):
        self.area = 1/2 * ((self.node1.x*self.node2.y + self.node2.x*self.node3.y + self.node3.x*self.node4.y + self.node4.x*self.node1.y) -
                           (self.node2.x*self.node1.y + self.node3.x*self.node2.y + self.node4.x*self.node3.y + self.node1.x*self.node4.y))

    def calc_aspect_ratio(self):
        self.calc_edge_lengths()
        e_max = max(self.e1, self.e2, self.e3, self.e4)
        e_min = min(self.e1, self.e2, self.e3, self.e4)

        self.aspect_ratio = e_max / e_min

    def calc_skewness(self):
        midx1 = (self.node1.x + self.node2.x)/2
        midy1 = (self.node1.y + self.node2.y)/2
        midx2 = (self.node2.x + self.node3.x)/2
        midy2 = (self.node2.y + self.node3.y)/2
        midx3 = (self.node3.x + self.node4.x)/2
        midy3 = (self.node3.y + self.node4.y)/2
        midx4 = (self.node4.x + self.node1.x)/2
        midy4 = (self.node4.y + self.node1.y)/2

        d1 = np.array([midx3-midx1, midy3-midy1])
        d2 = np.array([midx4-midx2, midy4-midy2])

        d1_mod = np.sqrt(d1[0]**2 + d1[1]**2)
        d2_mod = np.sqrt(d2[0]**2 + d2[1]**2)

        theta_rad1 = np.arccos(sum(d1*d2)/(d1_mod*d2_mod))
        theta_deg1 = abs(np.degrees(theta_rad1))
        theta_deg2 = 180 - theta_deg1
        theta_deg_max = max([theta_deg1, theta_deg2])
        theta_deg_min = min([theta_deg1, theta_deg2])

        self.skew = max([(theta_deg_max - 90)/90, (90 - theta_deg_min)/90])

    def gen_Ke(self):
        '''
        Finite Element Linear Membrane (Quad 2-D, 2x2 Gauss Quadrature rule) Analysis.
        '''

        if isinstance(self.Ke, np.ndarray):
            return self.Ke
        else:
            # Present configuration; same as topopt 99 line matlab code
            # (0.5773, 0.5773)         (0.5773, -0.5773)
            #         _________________________
            #        | 1                   2  |
            #        |                        |
            #        |                        |
            #        |                        |
            #        |                        |
            #        |                        |
            #        |                        |
            #        |                        |
            #        | 4                   3  |
            #         ________________________
            # (-0.5773, -0.5773)       (-0.5773, 0.5773)

            J1 = self.gen_jacobian(0.5773, 0.5773)
            J2 = self.gen_jacobian(0.5773, -0.5773)
            J3 = self.gen_jacobian(-0.5773, 0.5773)
            J4 = self.gen_jacobian(-0.5773, -0.5773)

            # Determinents
            det_J1 = np.linalg.det(J1)
            det_J2 = np.linalg.det(J2)
            det_J3 = np.linalg.det(J3)
            det_J4 = np.linalg.det(J4)

            A1 = self.gen_A(J1, det_J1)
            A2 = self.gen_A(J2, det_J2)
            A3 = self.gen_A(J3, det_J3)
            A4 = self.gen_A(J4, det_J4)

            G1 = self.gen_G(0.5773, 0.5773)
            G2 = self.gen_G(0.5773, -0.5773)
            G3 = self.gen_G(-0.5773, 0.5773)
            G4 = self.gen_G(-0.5773, -0.5773)

            self.B1 = A1.dot(G1)  # Will be needed in stress calculation; so assigned as instance variable
            self.B2 = A2.dot(G2)
            self.B3 = A3.dot(G3)
            self.B4 = A4.dot(G4)

            self.D = self.gen_D()

            Ke1 = (self.B1.T.dot(self.D).dot(self.B1))*det_J1
            Ke2 = (self.B2.T.dot(self.D).dot(self.B2))*det_J2
            Ke3 = (self.B3.T.dot(self.D).dot(self.B3))*det_J3
            Ke4 = (self.B4.T.dot(self.D).dot(self.B4))*det_J4

            self.Ke = (Ke1 + Ke2 + Ke3 + Ke4) * self.t

            return self.Ke

    def solve_element_stress(self):
        assert (self.node1.u!=None or self.node1.v!=None or
                self.node2.u!=None or self.node2.v!=None or
                self.node3.u!=None or self.node3.v!=None or
                self.node4.u!=None or self.node4.v!=None), 'Solve displacement first.'

        D_disp = self.gen_D_disp()

        stress1 = self.D.dot(self.B1).dot(D_disp)
        self.gp1sx = stress1[0]
        self.gp1sy = stress1[1]
        self.gp1sxy = stress1[2]
        self.gp1i1 = stress1[0] + stress1[1]
        self.gp1i2 = stress1[0]*stress1[1] - (stress1[2])**2
        self.gp1vm = np.sqrt(self.gp1i1**2 - 3*self.gp1i2)

        stress2 = self.D.dot(self.B2).dot(D_disp)
        self.gp2sx = stress2[0]
        self.gp2sy = stress2[1]
        self.gp2sxy = stress2[2]
        self.gp2i1 = stress2[0] + stress2[1]
        self.gp2i2 = stress2[0]*stress2[1] - (stress2[2])**2
        self.gp2vm = np.sqrt(self.gp2i1**2 - 3*self.gp2i2)

        stress3 = self.D.dot(self.B3).dot(D_disp)
        self.gp3sx = stress3[0]
        self.gp3sy = stress3[1]
        self.gp3sxy = stress3[2]
        self.gp3i1 = stress3[0] + stress3[1]
        self.gp3i2 = stress3[0]*stress3[1] - (stress3[2])**2
        self.gp3vm = np.sqrt(self.gp3i1**2 - 3*self.gp3i2)

        stress4 = self.D.dot(self.B4).dot(D_disp)
        self.gp4sx = stress4[0]
        self.gp4sy = stress4[1]
        self.gp4sxy = stress4[2]
        self.gp4i1 = stress4[0] + stress4[1]
        self.gp4i2 = stress4[0]*stress4[1] - (stress4[2])**2
        self.gp4vm = np.sqrt(self.gp4i1**2 - 3*self.gp4i2)

        # stress at nodes from chapter 28 of IFEM: page 28-6, equation 28-7

        sqrt_3 = np.sqrt(3)

        self.n1sx = (1+sqrt_3/2)*self.gp1sx - 0.5*self.gp2sx + (1-sqrt_3/2)*self.gp3sx - 0.5*self.gp4sx
        self.n2sx = (1+sqrt_3/2)*self.gp2sx - 0.5*self.gp1sx + (1-sqrt_3/2)*self.gp4sx - 0.5*self.gp3sx
        self.n3sx = (1-sqrt_3/2)*self.gp1sx - 0.5*self.gp2sx + (1+sqrt_3/2)*self.gp3sx - 0.5*self.gp4sx
        self.n4sx = (1-sqrt_3/2)*self.gp2sx - 0.5*self.gp1sx + (1+sqrt_3/2)*self.gp4sx - 0.5*self.gp3sx
        self.node1.sx = self.n1sx
        self.node2.sx = self.n2sx
        self.node3.sx = self.n3sx
        self.node4.sx = self.n4sx

        self.n1sy = (1+sqrt_3/2)*self.gp1sy - 0.5*self.gp2sy + (1-sqrt_3/2)*self.gp3sy - 0.5*self.gp4sy
        self.n2sy = (1+sqrt_3/2)*self.gp2sy - 0.5*self.gp1sy + (1-sqrt_3/2)*self.gp4sy - 0.5*self.gp3sy
        self.n3sy = (1-sqrt_3/2)*self.gp1sy - 0.5*self.gp2sy + (1+sqrt_3/2)*self.gp3sy - 0.5*self.gp4sy
        self.n4sy = (1-sqrt_3/2)*self.gp2sy - 0.5*self.gp1sy + (1+sqrt_3/2)*self.gp4sy - 0.5*self.gp3sy
        self.node1.sy = self.n1sy
        self.node2.sy = self.n2sy
        self.node3.sy = self.n3sy
        self.node4.sy = self.n4sy

        self.n1sxy = (1+sqrt_3/2)*self.gp1sxy - 0.5*self.gp2sxy + (1-sqrt_3/2)*self.gp3sxy - 0.5*self.gp4sxy
        self.n2sxy = (1+sqrt_3/2)*self.gp2sxy - 0.5*self.gp1sxy + (1-sqrt_3/2)*self.gp4sxy - 0.5*self.gp3sxy
        self.n3sxy = (1-sqrt_3/2)*self.gp1sxy - 0.5*self.gp2sxy + (1+sqrt_3/2)*self.gp3sxy - 0.5*self.gp4sxy
        self.n4sxy = (1-sqrt_3/2)*self.gp2sxy - 0.5*self.gp1sxy + (1+sqrt_3/2)*self.gp4sxy - 0.5*self.gp3sxy
        self.node1.sxy = self.n1sxy
        self.node2.sxy = self.n2sxy
        self.node3.sxy = self.n3sxy
        self.node4.sxy = self.n4sxy

        self.n1vm = (1+sqrt_3/2)*self.gp1vm - 0.5*self.gp2vm + (1-sqrt_3/2)*self.gp3vm - 0.5*self.gp4vm
        self.n2vm = (1+sqrt_3/2)*self.gp2vm - 0.5*self.gp1vm + (1-sqrt_3/2)*self.gp4vm - 0.5*self.gp3vm
        self.n3vm = (1-sqrt_3/2)*self.gp1vm - 0.5*self.gp2vm + (1+sqrt_3/2)*self.gp3vm - 0.5*self.gp4vm
        self.n4vm = (1-sqrt_3/2)*self.gp2vm - 0.5*self.gp1vm + (1+sqrt_3/2)*self.gp4vm - 0.5*self.gp3vm
        self.node1.vm = self.n1vm
        self.node2.vm = self.n2vm
        self.node3.vm = self.n3vm
        self.node4.vm = self.n4vm

    def gen_jacobian(self, s, t):
        J = np.zeros((2, 2), dtype=np.float64)

        J[0, 0] = (-(1-t)*self.node1.x + (1-t)*self.node2.x + (1+t)*self.node3.x - (1+t)*self.node4.x)/4
        J[0, 1] = (-(1-t)*self.node1.y + (1-t)*self.node2.y + (1+t)*self.node3.y - (1+t)*self.node4.y)/4
        J[1, 0] = (-(1-s)*self.node1.x - (1+s)*self.node2.x + (1+s)*self.node3.x + (1-s)*self.node4.x)/4
        J[1, 1] = (-(1-s)*self.node1.y - (1+s)*self.node2.y + (1+s)*self.node3.y + (1-s)*self.node4.y)/4

        return J

    def gen_A(self, J, det_J):
        A = np.zeros((3, 4), dtype=np.float64)

        A[0, 0] = J[1, 1]/det_J
        A[0, 1] = -J[0, 1]/det_J
        A[1, 2] = -J[1, 0]/det_J
        A[1, 3] = J[0, 0]/det_J
        A[2, 0] = -J[1, 0]/det_J
        A[2, 1] = J[0, 0]/det_J
        A[2, 2] = J[1, 1]/det_J
        A[2, 3] = -J[0, 1]/det_J

        return A

    def gen_G(self, s, t):
        G = np.zeros((4, 8), dtype=np.float64)

        G[0, 0]=-(1-t)/4; G[0, 1]=0; G[0, 2]=(1-t)/4; G[0, 3]=0; G[0, 4]=(1+t)/4; G[0, 5]=0; G[0, 6]=-(1+t)/4; G[0, 7]=0
        G[1, 0]=-(1-s)/4; G[1, 1]=0; G[1, 2]=-(1+s)/4; G[1, 3]=0; G[1, 4]=(1+s)/4; G[1, 5]=0; G[1, 6]=(1-s)/4; G[0, 7]=0
        G[2, 0]=0; G[2, 1]=-(1-t)/4; G[2, 2]=0; G[2, 3]=(1-t)/4; G[2, 4]=0; G[2, 5]=(1+t)/4; G[2, 6]=0; G[2, 7]=-(1+t)/4
        G[3, 0]=0; G[3, 1]=-(1-s)/4; G[3, 2]=0; G[3, 3]=-(1+s)/4; G[3, 4]=0; G[3, 5]=(1+s)/4; G[3, 6]=0; G[3, 7]=(1-s)/4

        return G

    def gen_D(self):
        D = np.zeros((3, 3), dtype=np.float64)

        # For plain stress
        coeff = self.E / (1 - self.Nu**2)
        D[0, 0] = coeff
        D[0, 1] = coeff * self.Nu
        D[1, 0] = coeff * self.Nu
        D[1, 1] = coeff
        D[2, 2] = coeff * (1 - self.Nu) / 2

        return D

    def gen_D_disp(self):
        '''
        Element displacement matrix.
        '''

        D_disp = np.zeros(8, dtype=np.float64)

        D_disp[0] = self.node1.u
        D_disp[1] = self.node1.v
        D_disp[2] = self.node2.u
        D_disp[3] = self.node2.v
        D_disp[4] = self.node3.u
        D_disp[5] = self.node3.v
        D_disp[6] = self.node4.u
        D_disp[7] = self.node4.v

        return D_disp


class Structure:

    def __init__(self):
        self.mult_penalty_const = 1000000

        # Compliance
        self.C = None
        # Volume fraction
        self.volfrac = None
        # penalization power
        self.penal = None
        # Filter size divided by element size
        self.rmin = None
        # Freeze mesh walls
        self.freeze_walls = False
        # Factor used in mesh independency filter
        self.fac = 0.1
        self.optimize_mode = True  # To or not to consider x and penality factor while generating K
        # Force stop optimization
        self.stop_opt_iter = False
        # Maximum opt iteration number
        self.max_iter = DEFULT_MAX_ITER
        # Maximum relative density change threshold for stopping optimization loop
        self.max_x_change_thresh = DEFAULT_MAX_X_CHANGE_THRESH

    def parse_input_file(self, filename='input_file.txt'):
        f = open(filename, 'r')
        lines_lst = f.readlines()
        f.close()

        # Omit extra white spaces
        lines_lst = [line.strip() for line in lines_lst]
        # Omit blank lines
        lines_lst = [line for line in lines_lst if line!='']

        # Init data accumulation flangs
        in_node = False
        in_element = False
        in_bc = False
        in_load = False
        in_freeze = False
        # Init dictionaries
        self.node_dict = {}
        self.element_dict = {}
        self.bc_dict = {}
        self.load_dict = {}
        self.freeze_elem_lst = []

        for line in lines_lst:
            # Ignore comments
            if line.startswith('%'):
                continue

            # Set start accumulate data flag
            if line.startswith('NODE') and line.endswith('START'):
                in_node = True
                continue
            elif line.startswith('ELEMENT') and line.endswith('START'):
                in_element = True
                continue
            elif line.startswith('BC') and line.endswith('START'):
                in_bc = True
                continue
            elif line.startswith('LOAD') and line.endswith('START'):
                in_load = True
                continue
            elif line.startswith('FREEZE') and line.endswith('START'):
                in_freeze = True
                continue
            elif line.startswith('VOLFRAC'):
                self.volfrac = float(line.split()[1])
                continue
            elif line.startswith('PENAL'):
                self.penal = float(line.split()[1])
                continue
            elif line.startswith('RMIN'):
                self.rmin = float(line.split()[1])
                continue
            elif line.startswith('FREEZE_WALLS'):
                self.freeze_walls = True if line.split()[1].strip()=='TRUE' else False
                continue
            elif line.startswith('MAX_X_CHANGE_THRESH'):
                self.max_x_change_thresh = float(line.split()[1].strip())
                continue
            elif line.startswith('MAX_ITER'):
                self.max_iter = int(line.split()[1].strip())
                continue

            # Set end accumulate data flag
            if line.startswith('NODE') and line.endswith('END'):
                in_node = False
                continue
            elif line.startswith('ELEMENT') and line.endswith('END'):
                in_element = False
                continue
            elif line.startswith('BC') and line.endswith('END'):
                in_bc = False
                continue
            elif line.startswith('LOAD') and line.endswith('END'):
                in_load = False
                continue
            elif line.startswith('FREEZE') and line.endswith('END'):
                in_freeze = False
                continue

            if in_node:
                node_num, x, y = line.split()
                self.node_dict[int(node_num)] = {'x':float(x), 'y':float(y)}
            elif in_element:
                element_num, n1, n2, n3, n4, E, Nu, t = line.split()
                self.element_dict[int(element_num)] = {'n1':int(n1), 'n2':int(n2),
                                                  'n3':int(n3), 'n4':int(n4),
                                                  'E':float(E), 'Nu':float(Nu),
                                                  't':float(t)}
            elif in_bc:
                dof, displacement = line.split()  # Displacement is not been used currently. By default value is 0
                self.bc_dict[int(dof)] = float(displacement)
            elif in_load:
                dof, load = line.split()
                self.load_dict[int(dof)] = float(load)
            elif in_freeze:
                self.freeze_elem_lst.append(int(line))

    def create_object_tables(self):
        # Init tables
        self.node_table = []
        self.element_table = []

        node_id_sorted = sorted(self.node_dict.keys())

        for node_id in node_id_sorted:
            x = self.node_dict[node_id]['x']
            y = self.node_dict[node_id]['y']

            # Node instance
            node = Node(node_id, x, y)

            # Add object to node table
            self.node_table.append(node)

        elem_id_sorted = sorted(self.element_dict.keys())

        for elem_id in elem_id_sorted:
            n1 = self.element_dict[elem_id]['n1']
            n2 = self.element_dict[elem_id]['n2']
            n3 = self.element_dict[elem_id]['n3']
            n4 = self.element_dict[elem_id]['n4']
            E = self.element_dict[elem_id]['E']
            Nu = self.element_dict[elem_id]['Nu']
            t = self.element_dict[elem_id]['t']

            # Element instance
            element = Element(elem_id, self.node_table[n1-1],
                              self.node_table[n2-1],
                              self.node_table[n3-1],
                              self.node_table[n4-1],
                              E, Nu, t, self.volfrac, self.penal)

            # Add object to element table
            self.element_table.append(element)

        self.node_num = len(self.node_table)
        self.elem_num = len(self.element_table)

        # Set Boundary condition to nodes
        for dof in self.bc_dict.keys():
            if  dof % 2 == 1:
                self.node_table[int((dof+1)/2 - 1)].dof_x_constr = True
            else:
                self.node_table[int(dof/2 - 1)].dof_y_constr = True

        # Set loads to nodes
        for dof in self.load_dict.keys():
            if dof % 2 == 1:
                self.node_table[int((dof+1)/2 - 1)].dof_x_load = self.load_dict[dof]
            else:
                self.node_table[int(dof/2 -1)].dof_y_load = self.load_dict[dof]

    def check_mesh(self):
        area_lst = []
        aspect_lst = []
        skew_lst = []

        for elem in self.element_table:
            elem.calc_area()
            elem.calc_aspect_ratio()
            elem.calc_skewness()

            area_lst.append(elem.area)
            aspect_lst.append(elem.aspect_ratio)
            skew_lst.append(elem.skew)

        elem_count = len(self.element_table)
        min_area = min(area_lst)
        max_area = max(area_lst)
        avg_area = sum(area_lst)/elem_count
        min_aspect = min(aspect_lst)
        max_aspect = max(aspect_lst)
        avg_aspet = sum(aspect_lst)/elem_count
        min_skew = min(skew_lst)
        max_skew = max(skew_lst)
        avg_skew = sum(skew_lst)/elem_count

        parent_num_dict = {}
        for node in self.node_table:
            parent_num = len(node.parent_elem_id_lst)
            try:
                parent_num_dict[parent_num] += 1
            except:
                parent_num_dict[parent_num] = 1

        parent_num_keys = sorted(parent_num_dict.keys())

        mesh_check_txt = ''

        mesh_check_txt += '------------- Mesh check completed -------------\n'
        mesh_check_txt += 'Total elements: {}, total nodes: {}\n'.format(len(self.element_table), len(self.node_table))
        mesh_check_txt += 'Area of elements: min={:.4e}, max={:.4e}, avg={:.4e}\n'.format(min_area, max_area, avg_area)
        mesh_check_txt += 'Aspect ratio: min={:.4f}, max={:.4f}, avg={:.4f}\n'.format(min_aspect, max_aspect, avg_aspet)
        mesh_check_txt += 'Skewness: min={:.2f}, max={:.2f}, avg={:.2f}\n'.format(min_skew, max_skew, avg_skew)
        mesh_check_txt += 'Number of nodes who has 1 or more parents:\n'
        for parent_num in parent_num_keys:
            if parent_num == 0:
                mesh_check_txt += '  {} node(s) have no parent element\n'.format(parent_num_dict[parent_num])
            else:
                mesh_check_txt += '  {} node(s) have {} parent element(s)\n'.format(parent_num_dict[parent_num], parent_num)

        return mesh_check_txt

    def find_adjacent_elem_1(self):
        '''
        Finds adjacent elements based on filter radius.
        '''

        print('Start finding adjacent element')
        t1 = time.time()

        for elem in self.element_table:
            for elem_comp in self.element_table:
                if elem != elem_comp:
                    if elem.distance(elem_comp) <= self.rmin:
                        elem.append_adjacent_elem(elem_comp)

        print('Finding adjacent element ended. Time required {:.2f} sec'.format(time.time()-t1))

    def find_adjacent_elem(self):
        '''
        Finds adjacent elements based on edge sharing.
        '''

        print('Start finding adjacent element')
        t1 = time.time()

        for elem in self.element_table:
            common_elem_count = 0
            for elem_comp in self.element_table:
                if common_elem_count >= 4:
                    break
                if elem != elem_comp:
                    common_node_count = 0
                    for node in elem.node_lst:
                        for node_comp in elem_comp.node_lst:
                            if node == node_comp:
                                common_node_count += 1
                    if common_node_count >= 2:  # Share an edge
                        elem.append_adjacent_elem(elem_comp)
                        common_elem_count += 1

        print('Finding adjacent element ended. Time required {:.2f} sec'.format(time.time()-t1))

    def gen_K(self):
        '''
        Generate global stiffness matrix.
        '''

        node_num_x2 = self.node_num * 2
        self.K = lil_matrix((node_num_x2, node_num_x2), dtype=np.float64)

        for elem in self.element_table:
            # Do not Place local stiffness into global if element exist flag set to False
            if not elem.exist:
                continue

            #dof1 = (self.element_table[i_elem].node1.id-1)*2  # 1 indexing to 0 indexing
            #dof2 = (self.element_table[i_elem].node1.id-1)*2+1
            #dof3 = (self.element_table[i_elem].node2.id-1)*2
            #dof4 = (self.element_table[i_elem].node2.id-1)*2+1
            #dof5 = (self.element_table[i_elem].node3.id-1)*2
            #dof6 = (self.element_table[i_elem].node3.id-1)*2+1
            #dof7 = (self.element_table[i_elem].node4.id-1)*2
            #dof8 = (self.element_table[i_elem].node4.id-1)*2+1

            dof1 = elem.node1.dof_x - 1  # 1 indexing to 0 indexing
            dof2 = elem.node1.dof_y - 1
            dof3 = elem.node2.dof_x - 1
            dof4 = elem.node2.dof_y - 1
            dof5 = elem.node3.dof_x - 1
            dof6 = elem.node3.dof_y - 1
            dof7 = elem.node4.dof_x - 1
            dof8 = elem.node4.dof_y - 1

            gdof_lst = [dof1, dof2, dof3, dof4, dof5, dof6, dof7, dof8]
            #ldof_lst = [0, 1, 2, 3, 4, 5, 6, 7]

            # Generate local stiffness matrix
            Ke = elem.gen_Ke()

            if self.optimize_mode:
                self.K[np.ix_(gdof_lst, gdof_lst)] += Ke * elem.Xe**self.penal
            else:
                self.K[np.ix_(gdof_lst, gdof_lst)] += Ke

    def prepare_bc_load(self):
        # Force array
        self.F = lil_matrix((self.node_num*2, 1), dtype=np.float64)

        # Extract BC and load from nodes
        self.con_dof_lst = []
        for node in self.node_table:
            if node.dof_x_constr:
                self.con_dof_lst.append(node.dof_x-1)  # 1 to 0 indexing
            if node.dof_y_constr:
                self.con_dof_lst.append(node.dof_y-1)
            if node.dof_x_load != 0:
                self.F[node.dof_x-1] = node.dof_x_load
            if node.dof_y_load != 0:
                self.F[node.dof_y-1] = node.dof_y_load

        # dofs list
        self.dof_lst = list(range(self.node_num*2))
        # Free dofs list
        self.free_dof_lst = np.setdiff1d(self.dof_lst, self.con_dof_lst)

    def optimize(self, write_x=True):
        self.stop_opt_iter = False
        change = 1.0
        count = 1

        compl_f = open('compl_plot.txt', 'w')
        compl_f.close()
        max_xchange_f = open('max_xchange_plot.txt', 'w')
        max_xchange_f.close()

        print('Optimization started.')
        t1 = time.time()
        while (change > self.max_x_change_thresh) and (count <= self.max_iter) and (not self.stop_opt_iter):
            X_old = self.get_X_array()
            self.gen_K()
            self.solve_displacement()
            self.solve_compliance()
            self.mesh_independency_filter()
            self.optimality_criteria_update()
            X_new = self.get_X_array()

            # Write to file
            if write_x:
                fname = str(count)+'.dat'
                X_new.tofile('X/'+fname)

            change = np.max(np.abs(X_new - X_old))
            if write_x:
                print('Iter: {} Max. change of X: {:.3f} Compl: {:.6f}'.format(count, change, self.C))

            # Write compliance vs iteration
            with open('compl_plot.txt', 'a') as compl_f:
                compl_f.write(f'{count} {self.C}\n')
            # Write max. relative density change vs iteration
            with open('max_xchange_plot.txt', 'a') as max_xchange_f:
                max_xchange_f.write(f'{count} {change}\n')

            count += 1

        if count > self.max_iter:
            print('Max. iteration limit exceeded.')
        print('Optimization finished.')
        print('Time required {:.2f} sec'.format(time.time()-t1))

    def solve_displacement(self, method='elimination'):
        '''
        Finds displacemnt by solving KD = F

        Parameters:
            method: str
                Incorporate boundary condition in solving.
                Choose between penalty and elimination.
        '''
        #ta = time.time()
        no_parent_node_lst = self.find_nodes_no_parent()
        # If any dof is off after setting element exist flag False
        self.off_dof_lst = []
        for node in no_parent_node_lst:
            self.off_dof_lst.append((node.id-1)*2)
            self.off_dof_lst.append((node.id-1)*2+1)

        if method == 'penalty': # Not tested, may need to edit before use
            self.solve_displacement_penalty()
        elif method == 'elimination':
            self.solve_displacement_elimination()
        else:
            raise Exception('Solve displacement method must be chosen between penalty and elimination.')

        # Set diaplacements as node attributes
        for i_node in range(self.node_num):
            self.node_table[i_node].u = self.D[2*i_node]
            self.node_table[i_node].v = self.D[2*i_node+1]

    def find_nodes_no_parent(self):
        '''
        Finds if any node has not parent element after setting element exist flag False.
        '''

        no_parent_node_lst = []
        for node in self.node_table:
            if node.parent_elem_id_lst == []:
                no_parent_node_lst.append(node)

        return no_parent_node_lst

    # Not tested, may need to modify before use
    def solve_displacement_penalty(self):
        # Penalty constant
        K_coo = self.K.tocoo()
        i_max = K_coo.data.argmax()
        i_min = K_coo.data.argmin()
        penalty_const = abs(max(K_coo.data[i_max], K_coo.data[i_min], key=abs))
        penalty_const *= self.mult_penalty_const

        # Put penalty constant on K
        self.K[np.ix_(self.con_dof_lst, self.con_dof_lst)] += penalty_const

        # Remove off dofs after setting element exist flag False
        calc_dof_lst = np.setdiff1d(self.dof_lst, self.off_dof_lst)

        # Accumulating data removing off dofs
        K_calc = self.K[np.ix_(calc_dof_lst, calc_dof_lst)]
        F_calc = self.K[calc_dof_lst]

        # Initialize Displacement array
        self.D = lil_matrix((self.node_num*2, 1), dtype=np.float64)

        # Solve for displacement
        D_calc = spsolve(K_calc.tocsr(), F_calc.tocsr())

        self.D = D_calc[calc_dof_lst]

    def solve_displacement_elimination(self):
        # Eliminate constraint and off dofs
        calc_dof_lst = np.setdiff1d(self.free_dof_lst, self.off_dof_lst)

        # Removed constraint and off dofs
        K_calc = self.K[np.ix_(calc_dof_lst, calc_dof_lst)]
        F_calc = self.F[calc_dof_lst]

        # Initialize global Displacement array
        self.D = np.zeros(self.node_num*2, dtype=np.float64)

        # Solve for displacement
        #D_calc = cg(K_calc, F_calc)[0]
        #D_calc = np.linalg.solve(K_calc, F_calc)
        #D_calc = solve(K_calc, F_calc)
        D_calc = spsolve(K_calc.tocsr(), F_calc.tocsr())

        self.D[calc_dof_lst] = D_calc

    def solve_stress(self):
        for element in self.element_table:
            if not element.exist:
                continue

            element.solve_element_stress()

        print('Stresses calculated.')

    def solve_compliance(self):
        self.C = 0.0
        for elem in self.element_table:
            dof1 = elem.node1.dof_x - 1  # 1 indexing to 0 indexing
            dof2 = elem.node1.dof_y - 1
            dof3 = elem.node2.dof_x - 1
            dof4 = elem.node2.dof_y - 1
            dof5 = elem.node3.dof_x - 1
            dof6 = elem.node3.dof_y - 1
            dof7 = elem.node4.dof_x - 1
            dof8 = elem.node4.dof_y - 1
            gdof_lst = [dof1, dof2, dof3, dof4, dof5, dof6, dof7, dof8]

            Ue = self.D[gdof_lst]
            Ke = elem.gen_Ke()

            mult = (Ue.T).dot(Ke).dot(Ue)

            # Element compliance
            elem.Ce = elem.Xe**self.penal * mult
            # Gradient of compliance
            elem.dCe = -self.penal * elem.Xe**(self.penal-1) * mult

            self.C += elem.Ce

    def mesh_independency_filter_1(self):
        '''
        Filters based on filter radius.
        '''

        for elem in self.element_table:
            elem.dCe_new = 0.0

        for elem in self.element_table:
            if len(elem.adjacent_elem_lst) > 0:
                s = 0.0
                for elem_adj in elem.adjacent_elem_lst:
                    fac = max(0, self.rmin - elem.distance(elem_adj))
                    s += fac
                    elem.dCe_new += fac * elem_adj.Xe * elem_adj.dCe
                # Element itself
                s += self.rmin
                elem.dCe_new += self.rmin * elem.Xe * elem.dCe

                elem.dCe_new /= elem.Xe * s
            else:
                elem.dCe_new = elem.dCe

        for elem in self.element_table:
            elem.dCe = copy.deepcopy(elem.dCe_new)

        # Freeze elements. Assigning minimum dC to elements.
        dC = self.get_dC_array()
        mindCe = np.min(dC)
        for elem_num in self.freeze_elem_lst:
            self.element_table[elem_num-1].dCe = mindCe
        # Freeze boundary elements
        if self.freeze_walls:
            for elem in self.element_table:
                if elem.is_boundary:
                    elem.dCe = mindCe

    def mesh_independency_filter(self):
        '''
        Filters based on edge sharing elements.
        '''

        for elem in self.element_table:
            elem.dCe_new = 0.0

        for elem in self.element_table:
            if len(elem.adjacent_elem_lst) > 0:
                s = 0.0
                for elem_adj in elem.adjacent_elem_lst:
                    #fac = max(0, self.rmin - elem.distance(elem_adj))
                    s += self.fac
                    elem.dCe_new += self.fac * elem_adj.Xe * elem_adj.dCe
                # Element itself
                s += (1.0 + self.fac)
                elem.dCe_new += (1 + self.fac) * elem.Xe * elem.dCe

                elem.dCe_new /= elem.Xe * s
            else:
                elem.dCe_new = elem.dCe

        for elem in self.element_table:
            elem.dCe = copy.deepcopy(elem.dCe_new)

        # Freeze elements. Assigning minimum dC to elements.
        dC = self.get_dC_array()
        mindCe = np.min(dC)
        for elem_num in self.freeze_elem_lst:
            self.element_table[elem_num-1].dCe = mindCe
        # Freeze boundary elements
        if self.freeze_walls:
            for elem in self.element_table:
                if elem.is_boundary:
                    elem.dCe = mindCe

    def optimality_criteria_update(self):
        X = self.get_X_array()
        dC = self.get_dC_array()
        a_001 = np.full(self.elem_num, 0.001)
        a_1 = np.full(self.elem_num, 1.0)

        l1 = 0
        l2 = 100000
        move = 0.15

        while l2-l1 > 1e-14:
            lmid = (l2+l1)/2.0
            X_new = np.maximum(a_001, np.maximum(X-move, np.minimum(a_1, np.minimum(X+move, X*np.sqrt(np.abs(-dC/lmid))))))

            if np.sum(X_new) - self.volfrac*self.elem_num > 0:
              l1 = lmid
            else:
              l2 = lmid

        for i_elem, elem in enumerate(self.element_table):
            elem.Xe = X_new[i_elem]

    def get_X_array(self):
        X = np.zeros(self.elem_num)

        for i_elem, elem in enumerate(self.element_table):
            X[i_elem] = elem.Xe

        return X

    def set_X(self, X):
        for i_elem, Xe in enumerate(X):
            self.element_table[i_elem].Xe = Xe

    def set_volfrac(self, volfrac):
        self.volfrac = volfrac

        for elem in self.element_table:
            elem.Xe = volfrac

    def get_dC_array(self):
        dC = np.zeros(self.elem_num)

        for i_elem, elem in enumerate(self.element_table):
            dC[i_elem] = elem.dCe

        return dC

    def filter_out_elements(self, min_Xe=0.9):
        for elem in self.element_table:
            if elem.Xe < min_Xe:
                elem.set_elem_on_off(exist=False)
            else:
                elem.set_elem_on_off(exist=True)

        self.reorganize_nodes_and_elements()

    def reorganize_nodes_and_elements(self):
        node_table_new = []
        node_count = 0
        for node in self.node_table:
            if node.parent_elem_id_lst == []:
                continue

            node_count += 1
            node.clear_parent_elem_id()
            node.assign_id(node_count)
            node_table_new.append(node)

        element_table_new = []
        elem_count = 0
        for elem in self.element_table:
            if not elem.exist:
                continue

            elem_count += 1
            elem.assign_id(elem_count)
            elem.assign_elem_id_to_nodes()
            element_table_new.append(elem)

        self.node_table = node_table_new
        self.element_table = element_table_new

        self.node_num = len(self.node_table)
        self.elem_num = len(self.element_table)

        self.prepare_bc_load()

    # Currently not in use in the GUI
    def write_results(self, filename='results.txt'):
        f = open(filename, 'w')

        f.write('Displacements:\n\n')
        for node in self.node_table:
            if node.parent_elem_id_lst == []:
                continue

            f.write('x displacement of node '+str(node.id)+' is '+str(node.u)+'\n')
            f.write('y displacement of node '+str(node.id)+' is '+str(node.v)+'\n')
        f.write('\n')

        f.write('Stresses:\n\n')
        for element in self.element_table:
            if not element.exist:
                continue

            f.write('Element Number: '+str(element.id)+'\n')
            f.write('Stress at Gauss point 1: sx = '+str(element.gp1sx)+' sy = '+str(element.gp1sy)+' sxy = '+str(element.gp1sxy)+' svm = '+str(element.gp1vm)+'\n')
            f.write('Stress at Gauss point 2: sx = '+str(element.gp2sx)+' sy = '+str(element.gp2sy)+' sxy = '+str(element.gp2sxy)+' svm = '+str(element.gp2vm)+'\n')
            f.write('Stress at Gauss point 3: sx = '+str(element.gp3sx)+' sy = '+str(element.gp3sy)+' sxy = '+str(element.gp3sxy)+' svm = '+str(element.gp3vm)+'\n')
            f.write('Stress at Gauss point 4: sx = '+str(element.gp4sx)+' sy = '+str(element.gp4sy)+' sxy = '+str(element.gp4sxy)+' svm = '+str(element.gp4vm)+'\n')
            f.write('Stress at LN 1 and GN = '+str(element.node1.id)+': sx = '+str(element.n1sx)+' sy = '+str(element.n1sy)+' sxy = '+str(element.n1sxy)+' svm = '+str(element.n1vm)+'\n')
            f.write('Stress at LN 2 and GN = '+str(element.node2.id)+': sx = '+str(element.n2sx)+' sy = '+str(element.n2sy)+' sxy = '+str(element.n2sxy)+' svm = '+str(element.n2vm)+'\n')
            f.write('Stress at LN 3 and GN = '+str(element.node3.id)+': sx = '+str(element.n3sx)+' sy = '+str(element.n3sy)+' sxy = '+str(element.n3sxy)+' svm = '+str(element.n3vm)+'\n')
            f.write('Stress at LN 4 and GN = '+str(element.node4.id)+': sx = '+str(element.n4sx)+' sy = '+str(element.n4sy)+' sxy = '+str(element.n4sxy)+' svm = '+str(element.n4vm)+'\n')
        f.write('\n')

        f.close()

        print('Wrote results to ' + filename + ' successfully.')

    # Currently not in use. Previously used for QBrush mesh plot.
    def get_bounding_box(self):
        x_lst = [node.x for node in self.node_table]
        y_lst = [node.y for node in self.node_table]

        w = max(x_lst) - min(x_lst)
        h = max(y_lst) - min(y_lst)

        return w, h

    def write_mesh(self, output_filename='mesh.vtk', mesh_type='quad', write_format='vtk-ascii'):
        '''
        Parameters:
            filename: str
                Output filename
            mesh_type: str
            write_format: str
                Choose among followings-
                    'abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml', 'exodus',
                    'flac3d', 'gmsh', 'gmsh22', 'h5m', 'hmf', 'mdpa', 'med',
                    'medit', 'nastran', 'netgen', 'neuroglancer', 'obj', 'off',
                    'permas', 'ply', 'stl', 'su2', 'svg', 'tecplot', 'tetgen',
                    'ugrid', 'vtk', 'vtu', 'wkt', 'xdmf'
        '''

        # **************** Note for abaqus exporting *******************
        # 1. Last line (*end) of inp file must be deleted. -> fixed in latest meshio
        # 2. Replace element type CAX4P with CPS4R  -> implemented in abacus_export_file_correction()
        # **************************************************************

        points = np.array([(node.x, node.y) for node in self.node_table])

        if mesh_type == "quad":
            cells = {mesh_type: np.array([(elem.node1.id-1, elem.node2.id-1, elem.node3.id-1, elem.node4.id-1)
                                          for elem in self.element_table])}
        elif mesh_type == "triangle":
            cell_lst = []
            for elem in self.element_table:
                cell_lst.append((elem.node1.id-1, elem.node2.id-1, elem.node4.id-1))
                cell_lst.append((elem.node2.id-1, elem.node3.id-1, elem.node4.id-1))
            cells = {mesh_type: cell_lst}
        else:
            raise Exception("Please choose mesh type between quad and triangle.")

        mesh = meshio.Mesh(points,
                           cells,
                           # Optionally provide extra data on points, cells, etc.
                           #point_data={'ux': np.array([[node.u] for node in self.node_table]),
                           #            'uy': np.array([[node.v] for node in self.node_table])},
                                       #'sx': np.array([[node.sx] for node in self.node_table]),
                                       #'sy': np.array([[node.sy] for node in self.node_table]),
                                       #'sxy': np.array([[node.sxy] for node in self.node_table]),
                                       #'vm': np.array([[node.vm] for node in self.node_table])},
                           #cell_data={0: {'volfrac': np.array([[elem.Xe] for elem in self.element_table]),
                           #               'compliance': np.array([[elem.Ce] for elem in self.element_table])}}
                           # field_data=field_data
                           )

        # Write to file
        meshio.write(output_filename, mesh, write_format)
        if write_format == 'abaqus':
            self.abacus_export_file_correction(output_filename)

        print('Wrote {} successfully.'.format(output_filename))

    def stop_optimization(self):
        self.stop_opt_iter = True

    def abacus_export_file_correction(self, output_filename):
        f = open(output_filename, 'r')
        lines = f.readlines()
        f.close()

        f = open(output_filename, 'w')
        for line in lines:
            f.write(line.replace('CAX4P', 'CPS4R'))
        f.close()
