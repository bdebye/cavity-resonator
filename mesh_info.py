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


import gmsh
import bisect
import numpy as np

def generate_rectangular_cavity_mesh(lx, ly, lz, element_size):
    """
    Generate a mesh for a rectangular cavity using gmsh.

    Args:
        lx (float): Length in x-direction.
        ly (float): Length in y-direction.
        lz (float): Length in z-direction.
        element_size (float): Target size of mesh elements.

    Returns:
        node_coords: (N_nodes, 3) array of node coordinates.
        elements: (N_elements, 4) array of tetrahedral element connectivity (node ids).
        outer_faces: (N_faces, 3) array of triangular face connectivities (node ids).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
    gmsh.option.setNumber("General.Verbosity", 0)  # Set verbosity to silent
    gmsh.model.add("rectangular_cavity")

    # Add box (volume)
    box = gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
    gmsh.model.occ.synchronize()

    # Set mesh size - apply to each corner point of the box
    for p in gmsh.model.getEntities(0):
        gmsh.model.mesh.setSize([p], element_size)

    # Mesh cavity
    print("Meshing cavity...")
    gmsh.model.mesh.generate(3)

    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)

    # Get tetrahedral (volume) elements
    element_types, element_tags, node_tags_elem = gmsh.model.mesh.getElements(dim=3)
    if not node_tags_elem:
        raise RuntimeError("No 3D elements found in mesh.")
    elements = np.array(node_tags_elem[0]).reshape(-1, 4)

    # Get outer surface faces (triangles only, quads are split into triangles)
    outer_faces = []
    for surf_tag in gmsh.model.getEntities(2):
        surf_elem_types, surf_elem_tags, surf_node_tags = gmsh.model.mesh.getElements(dim=2, tag=surf_tag[1])
        for e_type, node_tags_surf in zip(surf_elem_types, surf_node_tags):
            if e_type == 2:  # 3-node triangle
                faces_local = np.array(node_tags_surf, dtype=np.int64).reshape(-1, 3)
                outer_faces.append(faces_local)
            elif e_type == 3:  # 4-node quad - split into two triangles
                quads = np.array(node_tags_surf, dtype=np.int64).reshape(-1, 4)
                # Split each quad [n0, n1, n2, n3] into triangles [n0, n1, n2] and [n0, n2, n3]
                tri1 = quads[:, [0, 1, 2]]
                tri2 = quads[:, [0, 2, 3]]
                outer_faces.append(tri1)
                outer_faces.append(tri2)
    if outer_faces:
        outer_faces = np.vstack(outer_faces)
    else:
        outer_faces = np.zeros((0, 3), dtype=np.int64)

    gmsh.write("rectangular_cavity.msh")
    gmsh.finalize()

    elements = elements - 1
    outer_faces = outer_faces - 1
    return node_coords, elements, outer_faces


class mesh_info(object):
    def __init__(self, nodes, elems, boundary_faces):
        self.nodes = nodes
        self.elems = elems
        self.boundary_faces = boundary_faces

        self._parse_face_and_edge()
        self._parse_boundary_pec()

    def number_of_elements(self):
        return self.elems.shape[0]
    
    def number_of_nodes(self):
        return self.nodes.shape[0]
    
    def tet_center(self, label):
        return self.nodes[self.elems[label, :], :].mean(axis=0)
    
    def tet_volume(self, label):
        return np.linalg.det(self.nodes[self.elems[label, :], :]) / 6.0

    def _parse_face_and_edge(self):
        all_faces = []
        all_edges = []
        for tet in self.elems:
            # Tetrahedron faces (4 faces per tet)
            faces = [
                [tet[0], tet[1], tet[2]],
                [tet[0], tet[1], tet[3]],
                [tet[0], tet[2], tet[3]],
                [tet[1], tet[2], tet[3]],
            ]
            all_faces.extend([sorted(f) for f in faces])
            # Tetrahedron edges (6 edges per tet)
            edges = [
                [tet[0], tet[1]],
                [tet[0], tet[2]],
                [tet[0], tet[3]],
                [tet[1], tet[2]],
                [tet[1], tet[3]],
                [tet[2], tet[3]],
            ]
            all_edges.extend([sorted(e) for e in edges])
        # Deduplicate faces and edges using map to tuple for hashability
        unique_faces = list({tuple(f) for f in all_faces})
        unique_edges = list({tuple(e) for e in all_edges})
        self.DoF = (len(unique_faces) + len(unique_edges)) * 2

        # Merge unique_edges and unique_faces together and sort in dictionary order
        merged_edges_faces = unique_edges + unique_faces
        # Each row is a list-like, i.e., [i, j] or [i, j, k]; need to sort as tuples for lexicographical order
        # Convert each row to a tuple (shorter tuples come before longer ones with same leading entries)
        self.v_table = sorted(merged_edges_faces)

    def _parse_boundary_pec(self):
        all_boundary_faces = []
        for face in self.boundary_faces:
            all_boundary_faces.append(sorted(face))
        unique_boundary_faces = list({tuple(f) for f in all_boundary_faces})

        all_boundary_edges = []
        for face in self.boundary_faces:
            edges = [
                [face[0], face[1]],
                [face[1], face[2]],
                [face[2], face[0]],
            ]
            all_boundary_edges.extend([sorted(e) for e in edges])
        unique_boundary_edges = list({tuple(e) for e in all_boundary_edges})
        merged_edges_faces = unique_boundary_edges + unique_boundary_faces
        self.v_table_pec = sorted(merged_edges_faces)
        self.pec_label = [self.find_global_label(face, 0) for face in self.v_table_pec] \
            + [self.find_global_label(face, 1) for face in self.v_table_pec]

    def is_pec(self, node_list):
        key = tuple(node_list)
        idx = bisect.bisect_left(self.v_table_pec, key)
        return idx < len(self.v_table_pec) and self.v_table_pec[idx] == key

    def find_global_label(self, node_list, type_id):
        key = tuple(node_list)
        idx = bisect.bisect_left(self.v_table, key)
        if idx >= len(self.v_table) or self.v_table[idx] != key:
            raise ValueError("Node list {} not found in v_table".format(node_list))
        
        return idx * 2 + type_id

nodes, elems, boundary_faces = generate_rectangular_cavity_mesh(1.0, 0.5, 0.75, 0.3)
gl_mesh = mesh_info(nodes, elems, boundary_faces)

# Example usage (remove/comment out for library use):
if __name__ == "__main__":
    # nodes, elems, boundary_faces = generate_rectangular_cavity_mesh(1.0, 0.5, 0.75, 1.0)
    print("Number of nodes:", nodes.shape[0])
    print("Number of tetrahedral elements:", elems.shape[0])
    print("Number of outer faces:", boundary_faces.shape[0])
    print("First 5 node coordinates:\n", nodes[:5])
    print("First 5 volume elements:\n", elems[:5])
    print("First 5 boundary faces:\n", boundary_faces[:5])

    print("Number of DOFs:", gl_mesh.DoF)
    print("Number of elements:", gl_mesh.number_of_elements())


