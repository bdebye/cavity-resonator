# Copyright (C) 2025 Wen Wang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import math
import collections
from mesh_info import gl_mesh

# Speed of Light in Vacuum (c)
c0 = 299792458  # m/s

# Permittivity of Free Space (ε0)
epsilon0 = 8.854187817e-12  # F/m (farads per meter)

# Permeability of Free Space (μ0)
mu0 = 4 * 3.141592653589793 * 1e-7  # N/A² (newtons per ampere squared)

# Impedance of Free Space (Z0)
Z0 = 376.730313461  # ohms

# Intrinsic Impedance of Free Space (η0)
eta0 = 376.730313461  # ohms

class element_tet(object):

    def __init__(self, label):
        self.node_list = sorted(gl_mesh.elems[label])

        self.coord_matrix = np.ones((4, 4))
        self.coord_matrix[:, 1:] = gl_mesh.nodes[self.node_list]
        self.coef_matrix = np.linalg.inv(self.coord_matrix)
        self.volume = np.abs(np.linalg.det(self.coord_matrix)) / 6.0

        # Edge-based functions: 6 edges, each with B_e1 and B_e2
        self.edge_node_pairs = [
            (0, 1),  # edge 0
            (0, 2),  # edge 1
            (0, 3),  # edge 2
            (1, 2),  # edge 3
            (1, 3),  # edge 4
            (2, 3)   # edge 5
        ]
        self.face_node_triples = [
            (0, 1, 2), # face 0
            (0, 1, 3), # face 1
            (0, 2, 3), # face 2
            (1, 2, 3)  # face 3
        ]

        self.gl_label = [0] * 20
        self.gl_pec_nonzero = [True] * 20
        # Block ordering: e1 (0-5), e2 (6-11), f1 (12-15), f2 (16-19)
        for i in range(6):
            global_edge = (self.node_list[self.edge_node_pairs[i][0]],
                self.node_list[self.edge_node_pairs[i][1]])
            self.gl_label[i] = gl_mesh.find_global_label(global_edge, 0)
            self.gl_label[6 + i] = gl_mesh.find_global_label(global_edge, 1)
            if gl_mesh.is_pec(global_edge):
                self.gl_pec_nonzero[i] = False
                self.gl_pec_nonzero[6 + i] = False
                
        for i in range(4):
            global_face = (self.node_list[self.face_node_triples[i][0]],
                self.node_list[self.face_node_triples[i][1]],
                self.node_list[self.face_node_triples[i][2]])
            self.gl_label[12 + i] = gl_mesh.find_global_label(global_face, 0)
            self.gl_label[16 + i] = gl_mesh.find_global_label(global_face, 1)
            if gl_mesh.is_pec(global_face):
                self.gl_pec_nonzero[12 + i] = False
                self.gl_pec_nonzero[16 + i] = False

        self._compute_stiffness_matrix()
        self._compute_mass_matrix()
        # self._compute_element_matrices_numerical()

    def a(self, i):
        return self.coef_matrix[0, i]

    def b(self, i):
        return self.coef_matrix[1, i]

    def c(self, i):
        return self.coef_matrix[2, i]

    def d(self, i):
        return self.coef_matrix[3, i]

    def l(self, i1, i2):
        p1 = self.coord_matrix[i1, :]
        p2 = self.coord_matrix[i2, :]
        return np.linalg.norm(p1 - p2)

    def phi(self, i, j):
        return np.dot(self.grad_nodal(i), self.grad_nodal(j))

    def tet_vertices(self):
        return gl_mesh.nodes[self.node_list]

    def grad_nodal(self, i):
        return self.coef_matrix[1: 4, i]

    def nodal_basis(self, i, coord):
        coord_vec = np.ones(4)
        coord_vec[1: 4] = coord
        return np.dot(coord_vec, self.coef_matrix[:, i])
    
    def v_bar(self, i, j):
        grad_L_i = self.grad_nodal(i)
        grad_L_j = self.grad_nodal(j)
        return np.cross(grad_L_i, grad_L_j)

    def tet_center(self):
        return np.sum(self.coord_matrix, 0)[1: 4] / 4.0

    # ========================================================================
    # Second-Order Nédélec Basis Functions (LT/QN) - 20 functions total
    # ========================================================================
        
    def B_e1(self, i, j, coord):
        L_i = self.nodal_basis(i, coord)
        grad_L_j = self.grad_nodal(j)
        return self.l(i, j) * L_i * grad_L_j
    
    def B_e2(self, i, j, coord):
        L_j = self.nodal_basis(j, coord)
        grad_L_i = self.grad_nodal(i)
        return self.l(i, j) * L_j * grad_L_i
    
    def B_f1(self, i, j, k, coord):
        L_i = self.nodal_basis(i, coord)
        L_j = self.nodal_basis(j, coord)
        L_k = self.nodal_basis(k, coord)
        grad_L_j = self.grad_nodal(j)
        grad_L_k = self.grad_nodal(k)
        
        return L_i * L_j * grad_L_k - L_i * L_k * grad_L_j
    
    def B_f2(self, i, j, k, coord):
        L_i = self.nodal_basis(i, coord)
        L_j = self.nodal_basis(j, coord)
        L_k = self.nodal_basis(k, coord)
        grad_L_i = self.grad_nodal(i)
        grad_L_k = self.grad_nodal(k)
        
        return L_i * L_j * grad_L_k - L_j * L_k * grad_L_i

    def B(self, i, coord):
        # Block ordering: e1 (0-5), e2 (6-11), f1 (12-15), f2 (16-19)
        if 0 <= i < 6:
            # e1 type edge functions
            edge_index = i
            a, b = self.edge_node_pairs[edge_index]
            return self.B_e1(a, b, coord)
        elif 6 <= i < 12:
            # e2 type edge functions
            edge_index = i - 6
            a, b = self.edge_node_pairs[edge_index]
            return self.B_e2(a, b, coord)
        elif 12 <= i < 16:
            # f1 type face functions
            face_index = i - 12
            a, b, c = self.face_node_triples[face_index]
            return self.B_f1(a, b, c, coord)
        elif 16 <= i < 20:
            # f2 type face functions
            face_index = i - 16
            a, b, c = self.face_node_triples[face_index]
            return self.B_f2(a, b, c, coord)
        else:
            raise IndexError("Basis function index out of range (must be 0 <= i < 20)")
        
    def curl_B_e1(self, i, j):
        return self.l(i, j) * self.v_bar(i, j)
    
    def curl_B_e2(self, i, j):
        return -self.l(i, j) * self.v_bar(i, j)
    
    def curl_B_f1(self, i, j, k, coord):
        L_i = self.nodal_basis(i, coord)
        L_j = self.nodal_basis(j, coord)
        L_k = self.nodal_basis(k, coord)
        
        v_jk = self.v_bar(j, k)
        v_ik = self.v_bar(i, k)
        v_ij = self.v_bar(i, j)
        
        return 2 * L_i * v_jk + L_j * v_ik - L_k * v_ij
    
    def curl_B_f2(self, i, j, k, coord):
        L_i = self.nodal_basis(i, coord)
        L_j = self.nodal_basis(j, coord)
        L_k = self.nodal_basis(k, coord)
        
        v_jk = self.v_bar(j, k)
        v_ik = self.v_bar(i, k)
        v_ij = self.v_bar(i, j)
        
        return L_i * v_jk + 2 * L_j * v_ik + L_k * v_ij

    def curl_B(self, i, coord):
        # Block ordering: e1 (0-5), e2 (6-11), f1 (12-15), f2 (16-19)
        if 0 <= i < 6:
            # e1 type edge functions
            edge_index = i
            a, b = self.edge_node_pairs[edge_index]
            return self.curl_B_e1(a, b)
        elif 6 <= i < 12:
            # e2 type edge functions
            edge_index = i - 6
            a, b = self.edge_node_pairs[edge_index]
            return self.curl_B_e2(a, b)
        elif 12 <= i < 16:
            # f1 type face functions
            face_index = i - 12
            a, b, c = self.face_node_triples[face_index]
            return self.curl_B_f1(a, b, c, coord)
        elif 16 <= i < 20:
            # f2 type face functions
            face_index = i - 16
            a, b, c = self.face_node_triples[face_index]
            return self.curl_B_f2(a, b, c, coord)
        else:
            raise IndexError("Curl basis function index out of range (must be 0 <= i < 20)")
    
    def _compute_N_matrix(self):
        alpha = math.factorial(3) / float(math.factorial(6))
        N = np.ones((4, 4, 4)) * alpha
        for i in range(4):
            for j in range(4):
                for k in range(4):
                        counter = collections.Counter((i, j, k))
                        for c in counter.values():
                            N[i, j, k] *= math.factorial(c) 
        return N
    
    def _compute_P_matrix(self):
        alpha = math.factorial(3) / float(math.factorial(7))
        P = np.ones((4, 4, 4, 4)) * alpha
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        counter = collections.Counter((i, j, k, l))
                        for c in counter.values():
                            P[i, j, k, l] *= math.factorial(c)            
        return P
    
    def _compute_stiffness_matrix(self):

        self.Ke = np.zeros((20, 20), dtype=np.float64)
        
        V = self.volume
        
        # Precompute v_bar vectors for efficiency
        v_bar = np.zeros((4, 4, 3))  # v_bar[i,j] = grad_L_i × grad_L_j
        
        for i in range(4):
            for j in range(4):
                v_bar[i, j] = self.v_bar(i, j)
        
        # Compute L_i * L_j integration matrix: ∫ L_i L_j dV = V * M_ij
        M = (np.ones((4, 4)) + np.eye(4)) / 20.0
        
        # Helper function to get edge length and node indices
        def get_edge_info(edge_idx):
            i1, i2 = self.edge_node_pairs[edge_idx]
            return i1, i2, self.l(i1, i2)
        
        # Helper function to get face node indices
        def get_face_info(face_idx):
            return self.face_node_triples[face_idx]
        
        # ===== STIFFNESS MATRIX (E matrix in the paper) =====
        
        # E^{e1,e1} block: equations (26)
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                self.Ke[i, j] = V * l_i * l_j * np.dot(v_bar[i1, i2], v_bar[j1, j2])
        
        # E^{e1,e2} block: coupling between e1 and e2
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                # curl(B_e1) = l_i v̄_{i1,i2}, curl(B_e2) = -l_j v̄_{j1,j2}
                self.Ke[i, 6 + j] = -V * l_i * l_j * np.dot(v_bar[i1, i2], v_bar[j1, j2])
        
        # E^{e2,e1} block: transpose of E^{e1,e2} (for symmetry)
        for i in range(6):
            for j in range(6):
                self.Ke[6 + i, j] = self.Ke[j, 6 + i]
        
        # E^{e2,e2} block: equation (28) - same as E^{e1,e1}
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                self.Ke[6 + i, 6 + j] = V * l_i * l_j * np.dot(v_bar[i2, i1], v_bar[j2, j1])
        
        # E^{e1,f1} block: equation (29)
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                self.Ke[i, 12 + j] = V * l_i / 4.0 * np.dot(v_bar[i1, i2], 
                    (2 * v_bar[j2, j3] + v_bar[j1, j3] - v_bar[j1, j2]))
        
        # E^{e2,f1} block: equation (30)
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                self.Ke[6 + i, 12 + j] = V * l_i / 4.0 * np.dot(v_bar[i2, i1], 
                    (2 * v_bar[j2, j3] + v_bar[j1, j3] - v_bar[j1, j2]))
        
        # E^{f1,e1} block: transpose of E^{e1,f1} (for symmetry)
        for i in range(4):
            for j in range(6):
                self.Ke[12 + i, j] = self.Ke[j, 12 + i]
        
        # E^{f1,e2} block: transpose of E^{e2,f1} (for symmetry)
        for i in range(4):
            for j in range(6):
                self.Ke[12 + i, 6 + j] = self.Ke[6 + j, 12 + i]
        
        # E^{f1,f1} block: analytical integration
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # curl(B_f1^i) = 2 L_i1 v̄_i2,i3 + L_i2 v̄_i1,i3 - L_i3 v̄_i1,i2
                # Expand dot product and integrate term by term
                self.Ke[12 + i, 12 + j] = V * (
                    4 * np.dot(v_bar[i2, i3], v_bar[j2, j3]) * M[i1, j1] +
                    2 * np.dot(v_bar[i2, i3], v_bar[j1, j3]) * M[i1, j2] -
                    2 * np.dot(v_bar[i2, i3], v_bar[j1, j2]) * M[i1, j3] +
                    2 * np.dot(v_bar[i1, i3], v_bar[j2, j3]) * M[i2, j1] +
                        np.dot(v_bar[i1, i3], v_bar[j1, j3]) * M[i2, j2] -
                        np.dot(v_bar[i1, i3], v_bar[j1, j2]) * M[i2, j3] -
                    2 * np.dot(v_bar[i1, i2], v_bar[j2, j3]) * M[i3, j1] -
                        np.dot(v_bar[i1, i2], v_bar[j1, j3]) * M[i3, j2] +
                        np.dot(v_bar[i1, i2], v_bar[j1, j2]) * M[i3, j3]
                )
        
        # E^{e1,f2} block: equation (32)
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                self.Ke[i, 16 + j] = V * l_i / 4.0 * np.dot(v_bar[i1, i2],
                    (v_bar[j2, j3] + 2 * v_bar[j1, j3] + v_bar[j1, j2]))
        
        # E^{e2,f2} block: equation (33)
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                self.Ke[6 + i, 16 + j] = V * l_i / 4.0 * np.dot(v_bar[i2, i1],
                    (v_bar[j2, j3] + 2 * v_bar[j1, j3] + v_bar[j1, j2]))
        
        # E^{f2,e1} block: transpose (for symmetry)
        for i in range(4):
            for j in range(6):
                self.Ke[16 + i, j] = self.Ke[j, 16 + i]
        
        # E^{f2,e2} block: transpose (for symmetry)
        for i in range(4):
            for j in range(6):
                self.Ke[16 + i, 6 + j] = self.Ke[6 + j, 16 + i]
        
        # E^{f1,f2} block: analytical integration
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # curl(B_f1^i) = 2 L_i1 v̄_i2,i3 + L_i2 v̄_i1,i3 - L_i3 v̄_i1,i2
                # curl(B_f2^j) = L_j1 v̄_j2,j3 + 2 L_j2 v̄_j1,j3 + L_j3 v̄_j1,j2
                self.Ke[12 + i, 16 + j] = V * (
                    2 * np.dot(v_bar[i2, i3], v_bar[j2, j3]) * M[i1, j1] +
                    4 * np.dot(v_bar[i2, i3], v_bar[j1, j3]) * M[i1, j2] +
                    2 * np.dot(v_bar[i2, i3], v_bar[j1, j2]) * M[i1, j3] +
                        np.dot(v_bar[i1, i3], v_bar[j2, j3]) * M[i2, j1] +
                    2 * np.dot(v_bar[i1, i3], v_bar[j1, j3]) * M[i2, j2] +
                        np.dot(v_bar[i1, i3], v_bar[j1, j2]) * M[i2, j3] -
                        np.dot(v_bar[i1, i2], v_bar[j2, j3]) * M[i3, j1] -
                    2 * np.dot(v_bar[i1, i2], v_bar[j1, j3]) * M[i3, j2] -
                        np.dot(v_bar[i1, i2], v_bar[j1, j2]) * M[i3, j3]
                )
        
        # E^{f2,f1} block: analytical integration
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # curl(B_f2^i) = L_i1 v̄_i2,i3 + 2 L_i2 v̄_i1,i3 + L_i3 v̄_i1,i2
                # curl(B_f1^j) = 2 L_j1 v̄_j2,j3 + L_j2 v̄_j1,j3 - L_j3 v̄_j1,j2
                self.Ke[16 + i, 12 + j] = V * (
                    2 * np.dot(v_bar[i2, i3], v_bar[j2, j3]) * M[i1, j1] +
                        np.dot(v_bar[i2, i3], v_bar[j1, j3]) * M[i1, j2] -
                        np.dot(v_bar[i2, i3], v_bar[j1, j2]) * M[i1, j3] +
                    4 * np.dot(v_bar[i1, i3], v_bar[j2, j3]) * M[i2, j1] +
                    2 * np.dot(v_bar[i1, i3], v_bar[j1, j3]) * M[i2, j2] -
                    2 * np.dot(v_bar[i1, i3], v_bar[j1, j2]) * M[i2, j3] +
                    2 * np.dot(v_bar[i1, i2], v_bar[j2, j3]) * M[i3, j1] +
                        np.dot(v_bar[i1, i2], v_bar[j1, j3]) * M[i3, j2] -
                        np.dot(v_bar[i1, i2], v_bar[j1, j2]) * M[i3, j3]
                )
        
        # E^{f2,f2} block: analytical integration
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # curl(B_f2^i) = L_i1 v̄_i2,i3 + 2 L_i2 v̄_i1,i3 + L_i3 v̄_i1,i2
                # curl(B_f2^j) = L_j1 v̄_j2,j3 + 2 L_j2 v̄_j1,j3 + L_j3 v̄_j1,j2
                self.Ke[16 + i, 16 + j] = V * (
                        np.dot(v_bar[i2, i3], v_bar[j2, j3]) * M[i1, j1] +
                    2 * np.dot(v_bar[i2, i3], v_bar[j1, j3]) * M[i1, j2] +
                        np.dot(v_bar[i2, i3], v_bar[j1, j2]) * M[i1, j3] +
                    2 * np.dot(v_bar[i1, i3], v_bar[j2, j3]) * M[i2, j1] +
                    4 * np.dot(v_bar[i1, i3], v_bar[j1, j3]) * M[i2, j2] +
                    2 * np.dot(v_bar[i1, i3], v_bar[j1, j2]) * M[i2, j3] +
                        np.dot(v_bar[i1, i2], v_bar[j2, j3]) * M[i3, j1] +
                    2 * np.dot(v_bar[i1, i2], v_bar[j1, j3]) * M[i3, j2] +
                        np.dot(v_bar[i1, i2], v_bar[j1, j2]) * M[i3, j3]
                )

    def _compute_mass_matrix(self):

        self.Me = np.zeros((20, 20), dtype=np.float64)

        # Compute integration matrices
        N = self._compute_N_matrix()  # N_ijk = (1/V) ∫ L_i L_j L_k dV
        P = self._compute_P_matrix()  # P_ijkl = (1/V) ∫ L_i L_j L_k L_l dV
        
        V = self.volume
        
        # Precompute phi values for efficiency
        phi = np.zeros((4, 4))  # phi[i,j] = ∇L_i · ∇L_j
        for i in range(4):
            for j in range(4):
                phi[i, j] = self.phi(i, j)
        
        # Compute L_i * L_j integration matrix: ∫ L_i L_j dV = V * M_ij
        M = (np.ones((4, 4)) + np.eye(4)) / 20.0
        
        # Helper function to get edge length and node indices
        def get_edge_info(edge_idx):
            i1, i2 = self.edge_node_pairs[edge_idx]
            return i1, i2, self.l(i1, i2)
        
        # Helper function to get face node indices
        def get_face_info(face_idx):
            return self.face_node_triples[face_idx]
        
        
        # ===== F^{e1,e1} block: equation (36) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                # F_ij^{e1e1} = V * l_i * l_j * φ_{i2,j2} * M_{i1,j1}
                self.Me[i, j] = V * l_i * l_j * phi[i2, j2] * M[i1, j1]
        
        # ===== F^{e1,e2} block: equation (37) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                # F_ij^{e1e2} = V * l_i * l_j * φ_{i2,j1} * M_{i1,j2}
                self.Me[i, 6 + j] = V * l_i * l_j * phi[i2, j1] * M[i1, j2]
        
        # ===== F^{e2,e1} block: equation (38) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                # F_ij^{e2e1} = V * l_i * l_j * φ_{i1,j2} * M_{i2,j1}
                self.Me[6 + i, j] = V * l_i * l_j * phi[i1, j2] * M[i2, j1]
        
        # ===== F^{e2,e2} block: equation (39) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(6):
                j1, j2, l_j = get_edge_info(j)
                # F_ij^{e2e2} = V * l_i * l_j * φ_{i1,j1} * M_{i2,j2}
                self.Me[6 + i, 6 + j] = V * l_i * l_j * phi[i1, j1] * M[i2, j2]
        
        # ===== F^{e1,f1} block: equation (40) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # F_ij^{e1f1} = V * l_i * (φ_{i2,j3} * N_{i1,j1,j2} - φ_{i2,j2} * N_{i1,j1,j3})
                self.Me[i, 12 + j] = V * l_i * (
                    phi[i2, j3] * N[i1, j1, j2] - 
                    phi[i2, j2] * N[i1, j1, j3]
                )
        
        # ===== F^{e2,f1} block: equation (41) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # F_ij^{e2f1} = V * l_i * (φ_{i1,j3} * N_{i2,j1,j2} - φ_{i1,j2} * N_{i2,j1,j3})
                self.Me[6 + i, 12 + j] = V * l_i * (
                    phi[i1, j3] * N[i2, j1, j2] - 
                    phi[i1, j2] * N[i2, j1, j3]
                )
        
        # ===== F^{f1,e1} block: transpose of F^{e1,f1} =====
        for i in range(4):
            for j in range(6):
                self.Me[12 + i, j] = self.Me[j, 12 + i]
        
        # ===== F^{f1,e2} block: transpose of F^{e2,f1} =====
        for i in range(4):
            for j in range(6):
                self.Me[12 + i, 6 + j] = self.Me[6 + j, 12 + i]
        
        # ===== F^{e1,f2} block: equation (42) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # F_ij^{e1f2} = V * l_i * (φ_{i2,j3} * N_{i1,j1,j2} - φ_{i2,j1} * N_{i1,j2,j3})
                self.Me[i, 16 + j] = V * l_i * (
                    phi[i2, j3] * N[i1, j1, j2] - 
                    phi[i2, j1] * N[i1, j2, j3]
                )
        
        # ===== F^{e2,f2} block: equation (43) =====
        for i in range(6):
            i1, i2, l_i = get_edge_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # F_ij^{e2f2} = V * l_i * (φ_{i1,j3} * N_{i2,j1,j2} - φ_{i1,j1} * N_{i2,j2,j3})
                self.Me[6 + i, 16 + j] = V * l_i * (
                    phi[i1, j3] * N[i2, j1, j2] - 
                    phi[i1, j1] * N[i2, j2, j3]
                )
        
        # ===== F^{f2,e1} block: transpose of F^{e1,f2} =====
        for i in range(4):
            for j in range(6):
                self.Me[16 + i, j] = self.Me[j, 16 + i]
        
        # ===== F^{f2,e2} block: transpose of F^{e2,f2} =====
        for i in range(4):
            for j in range(6):
                self.Me[16 + i, 6 + j] = self.Me[6 + j, 16 + i]
        
        # ===== F^{f1,f1} block: equation (44) =====
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # B_f1^i · B_f1^j with φ terms from ∇L·∇L products
                self.Me[12 + i, 12 + j] = V * (
                    phi[i3, j3] * P[i1, i2, j1, j2] - 
                    phi[i3, j2] * P[i1, i2, j1, j3] - 
                    phi[i2, j3] * P[i1, i3, j1, j2] +
                    phi[i2, j2] * P[i1, i3, j1, j3]
                )
        
        # ===== F^{f1,f2} block: equation (45) =====
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # B_f1^i · B_f2^j with φ terms
                self.Me[12 + i, 16 + j] = V * (
                    phi[i3, j3] * P[i1, i2, j1, j2] - 
                    phi[i3, j1] * P[i1, i2, j2, j3] - 
                    phi[i2, j3] * P[i1, i3, j1, j2] +
                    phi[i2, j1] * P[i1, i3, j2, j3]
                )
        
        # ===== F^{f2,f1} block: transpose of F^{f1,f2} =====
        for i in range(4):
            for j in range(4):
                self.Me[16 + i, 12 + j] = self.Me[12 + j, 16 + i]
        
        # ===== F^{f2,f2} block: similar to F^{f1,f1} =====
        for i in range(4):
            i1, i2, i3 = get_face_info(i)
            for j in range(4):
                j1, j2, j3 = get_face_info(j)
                # B_f2^i · B_f2^j with φ terms
                self.Me[16 + i, 16 + j] = V * (
                    phi[i3, j3] * P[i1, i2, j1, j2] - 
                    phi[i3, j1] * P[i1, i2, j2, j3] - 
                    phi[i1, j3] * P[i2, i3, j1, j2] +
                    phi[i1, j1] * P[i2, i3, j2, j3]
                )
    
    def _compute_element_matrices_numerical(self):
        """
        Optimized assembly of stiffness (Ke) and mass (Me) matrices.
        
        Key optimizations:
        1. Exploits symmetry: only compute upper triangle
        2. Vectorized quadrature: evaluates all basis functions at each quadrature point
        3. Uses Einstein summation for efficient matrix assembly
        """
        self.Ke = np.zeros((20, 20), dtype=np.float64)
        self.Me = np.zeros((20, 20), dtype=np.float64)

        # 4th order quadrature points and weights for tetrahedron
        ref_points = np.array([
            (0.2500000000000000, 0.2500000000000000, 0.2500000000000000),
            (0.7857142857142857, 0.0714285714285714, 0.0714285714285714),
            (0.0714285714285714, 0.0714285714285714, 0.0714285714285714),
            (0.0714285714285714, 0.0714285714285714, 0.7857142857142857),
            (0.0714285714285714, 0.7857142857142857, 0.0714285714285714),
            (0.1005964238332008, 0.3994035761667992, 0.3994035761667992),
            (0.3994035761667992, 0.1005964238332008, 0.3994035761667992),
            (0.3994035761667992, 0.3994035761667992, 0.1005964238332008),
            (0.3994035761667992, 0.1005964238332008, 0.1005964238332008),
            (0.1005964238332008, 0.3994035761667992, 0.1005964238332008),
            (0.1005964238332008, 0.1005964238332008, 0.3994035761667992)
        ])
        
        ref_weights = np.array([
            -0.0789333333333333, 0.0457333333333333, 0.0457333333333333,
            0.0457333333333333, 0.0457333333333333, 0.1493333333333333,
            0.1493333333333333, 0.1493333333333333, 0.1493333333333333,
            0.1493333333333333, 0.1493333333333333
        ])
        
        # Convert to physical coordinates
        vertices = self.tet_vertices()
        V = np.asarray(vertices)
        edges = V[1:] - V[0]
        phys_points = V[0] + ref_points @ edges  # Shape: (11, 3)
        
        n_quad = len(ref_weights)
        
        # Pre-compute all basis function values at quadrature points
        # B_values[q, i, d] = component d of basis function i at quadrature point q
        B_values = np.zeros((n_quad, 20, 3))
        curl_B_values = np.zeros((n_quad, 20, 3))
        
        for q in range(n_quad):
            coord = phys_points[q]
            for i in range(20):
                B_values[q, i] = self.B(i, coord)
                curl_B_values[q, i] = self.curl_B(i, coord)
        
        # Vectorized assembly using Einstein summation
        # Ke[i,j] = sum_q w[q] * sum_d curl_B[q,i,d] * curl_B[q,j,d]
        # Me[i,j] = sum_q w[q] * sum_d B[q,i,d] * B[q,j,d]
        
        # Shape: (n_quad, 20, 20) - dot product over spatial dimension
        K_integrands = np.einsum('qid,qjd->qij', curl_B_values, curl_B_values)
        M_integrands = np.einsum('qid,qjd->qij', B_values, B_values)
        
        # Weighted sum over quadrature points: (20, 20)
        self.Ke[:] = np.einsum('q,qij->ij', ref_weights, K_integrands) * self.volume
        self.Me[:] = np.einsum('q,qij->ij', ref_weights, M_integrands) * self.volume
    
    def interp_electric(self, weight, coord):
        e_field = np.zeros(3)
        for i in range(20):
            e_field += weight[i] * self.B(i, coord)
        return e_field

    def interp_magnetic(self, weight, coord, k0):
        curl_e = np.zeros(3, dtype=np.complex128)
        for i in range(20):
            curl_e += weight[i] * self.curl_B(i, coord)
        return -curl_e / (1j * k0 * c0 * mu0)


if __name__ == "__main__":
    element = element_tet(0)
    element._compute_mass_matrix()
    nan_matrix = element.Me.copy()
    element._compute_element_matrices_numerical()
    num_matrix = element.Me.copy()
    print(np.abs(nan_matrix - num_matrix))

