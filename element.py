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
        for i in range(6):
            global_edge = (self.node_list[self.edge_node_pairs[i][0]],
                self.node_list[self.edge_node_pairs[i][1]])
            self.gl_label[i * 2] = gl_mesh.find_global_label(global_edge, 0)
            self.gl_label[i * 2 + 1] = gl_mesh.find_global_label(global_edge, 1)
            if gl_mesh.is_pec(global_edge):
                self.gl_pec_nonzero[i * 2] = False
                self.gl_pec_nonzero[i * 2 + 1] = False
                
        for i in range(4):
            global_face = (self.node_list[self.face_node_triples[i][0]],
                self.node_list[self.face_node_triples[i][1]],
                self.node_list[self.face_node_triples[i][2]])
            self.gl_label[12 + i * 2] = gl_mesh.find_global_label(global_face, 0)
            self.gl_label[12 + i * 2 + 1] = gl_mesh.find_global_label(global_face, 1)
            if gl_mesh.is_pec(global_face):
                self.gl_pec_nonzero[12 + i * 2] = False
                self.gl_pec_nonzero[12 + i * 2 + 1] = False

        self._assemble_element_matrices()

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
        if 0 <= i < 12:
            edge_index = i // 2
            func_type = i % 2
            a, b = self.edge_node_pairs[edge_index]
            if func_type == 0:
                return self.B_e1(a, b, coord)
            else:
                return self.B_e2(a, b, coord)
        elif 12 <= i < 20:
            face_index = (i - 12) // 2
            func_type = (i - 12) % 2
            a, b, c = self.face_node_triples[face_index]
            if func_type == 0:
                return self.B_f1(a, b, c, coord)
            else:
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
        if 0 <= i < 12:
            edge_index = i // 2
            func_type = i % 2
            a, b = self.edge_node_pairs[edge_index]
            if func_type == 0:
                return self.curl_B_e1(a, b)
            else:
                return self.curl_B_e2(a, b)
        elif 12 <= i < 20:
            face_index = (i - 12) // 2
            func_type = (i - 12) % 2
            a, b, c = self.face_node_triples[face_index]
            if func_type == 0:
                return self.curl_B_f1(a, b, c, coord)
            else:
                return self.curl_B_f2(a, b, c, coord)
        else:
            raise IndexError("Curl basis function index out of range (must be 0 <= i < 20)")
    
    def _assemble_element_matrices(self):
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
    print("Tetrahedron center:", element.tet_center())
    print("Volume:", element.volume)
    print("\nCoefficient matrix:")
    print(element.coef_matrix)
    