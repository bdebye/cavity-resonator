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


from element import element_tet
from mesh_info import gl_mesh

import scipy
import time
import multiprocessing
import numpy as np

NC = 10

start = time.time()
def time_elapse():
    elapsed_time = time.time() - start
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time elapsed: {hours} hours {minutes} minutes {seconds} seconds.")

class resonator(object):

    def __init__(self):
        print("Number of DOFs:", gl_mesh.DoF)
        print("Number of elements:", gl_mesh.number_of_elements())
        print("Initialize the elements.")

        self.dof = gl_mesh.DoF
        # self.Nd_elem = list(multiprocessing.Pool(NC).map(element_tet, range(gl_mesh.number_of_elements())))
        self.Nd_elem = list(map(element_tet, range(gl_mesh.number_of_elements())))

    def assemble(self):
        print("Assemble the global matrices.")

        # Initialize empty coordinate lists for sparse matrices
        K_rows, K_cols, K_data = [], [], []
        M_rows, M_cols, M_data = [], [], []

        for el in self.Nd_elem:
            K_elem = el.Ke
            M_elem = el.Me
            for i in range(20):
                for j in range(20):
                    # The PEC entries are actually imposed in the following line.
                    if el.gl_pec_nonzero[i] and el.gl_pec_nonzero[j]:
                        K_rows.append(el.gl_label[i])
                        K_cols.append(el.gl_label[j])
                        K_data.append(K_elem[i, j])
                        
                        M_rows.append(el.gl_label[i])
                        M_cols.append(el.gl_label[j])
                        M_data.append(M_elem[i, j])

        # Impose dirichlet boundary condition for PEC walls.
        for label in gl_mesh.pec_label:
            K_rows.append(label)
            K_cols.append(label)
            K_data.append(1.0)

            M_rows.append(label)
            M_cols.append(label)
            M_data.append(1.0)
        
        # Initialize sparse matrices directly in CSR format
        self.K = scipy.sparse.csr_matrix((K_data, (K_rows, K_cols)), shape=(self.dof, self.dof))
        self.M = scipy.sparse.csr_matrix((M_data, (M_rows, M_cols)), shape=(self.dof, self.dof))

    def solve_generaleigen(self):
        print("Solving general eigensystem...")
        self.eigvals, self.eigvecs = scipy.sparse.linalg.eigs(self.K, k=50, M=self.M, which='LR', sigma=2.0, OPpart='r')

    def post_field(self, eigen_vals, eigen_vecs):
        from post_proc import gmsh_post
        print("Postprocessing...")
        assert eigen_vecs.shape[1] == len(eigen_vals)
        n_mode = eigen_vecs.shape[1]
        post = gmsh_post("cavity_field.pos")
        for i in range(n_mode):
            post.open_view(f"Electric field {i}")
            for el in self.Nd_elem:
                center_field = el.interp_electric(eigen_vecs[el.gl_label, i], el.tet_center()).real
                post.add_vector_field(el.tet_center(), center_field)
            post.close_vew()

        for i in range(n_mode):
            k0 = np.sqrt(eigen_vals[i].real)
            post.open_view(f"Magnetic field {i}")
            for el in self.Nd_elem:
                center_field = el.interp_magnetic(eigen_vecs[el.gl_label, i], el.tet_center(), k0).imag
                post.add_vector_field(el.tet_center(), center_field)
            post.close_vew()
        post.close()


if __name__ == "__main__":
    task = resonator()
    task.assemble()
    task.solve_generaleigen()

    eigen_pairs = []
    for i in range(len(task.eigvals.real)):
        if task.eigvals[i].real > 1.0:
            eigen_pairs.append((i, float(np.sqrt(float(task.eigvals[i].real)))))

    eigen_pairs.sort(key=lambda a: a[1])
    eigen_pairs = eigen_pairs[: 30]

    print("Computed wavenumbers:")
    for pair in eigen_pairs:
        print(pair)

    if len(eigen_pairs) < 8:
        print("ERROR! Not enough eigenvalues are found!")

    exact = np.array([5.23599, 7.02481, 7.55145, 7.55145, 8.17887, 8.17887, 8.88577, 8.94726])
    computed = np.array(eigen_pairs)[:8, 1]
    error_rate = (computed - exact) / exact

    print("ERROR RATE:")
    print(error_rate)

    eigen_vecs = np.array(task.eigvecs[:, [a[0] for a in eigen_pairs]]).real
    eigen_vals = np.array([task.eigvals[a[0]] for a in eigen_pairs])
    task.post_field(eigen_vals, eigen_vecs)
    time_elapse()