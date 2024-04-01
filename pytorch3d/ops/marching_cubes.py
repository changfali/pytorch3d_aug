# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, List, Optional, Tuple

import torch
from pytorch3d import _C
from pytorch3d.ops.marching_cubes_data import EDGE_TO_VERTICES, FACE_TABLE, INDEX, EDGE_TABLE
from pytorch3d.transforms import Translate
from torch.autograd import Function


EPS = 0.00001


class Cube:
    def __init__(
        self,
        bfl_v: Tuple[int, int, int],
        volume: torch.Tensor,
        isolevel: float,
    ) -> None:
        """
        Initializes a cube given the bottom front left vertex coordinate
        and computes the cube configuration given vertex values and isolevel.

        Edge and vertex convention:

                    v4_______e4____________v5
                    /|                    /|
                   / |                   / |
                e7/  |                e5/  |
                 /___|______e6_________/   |
              v7|    |                 |v6 |e9
                |    |                 |   |
                |    |e8               |e10|
             e11|    |                 |   |
                |    |______e0_________|___|
                |   / v0(bfl_v)        |   |v1
                |  /                   |  /
                | /e3                  | /e1
                |/_____________________|/
                v3         e2          v2

        Args:
            bfl_vertex: a tuple of size 3 corresponding to the bottom front left vertex
                of the cube in (x, y, z) format
            volume: the 3D scalar data
            isolevel: the isosurface value used as a threshold for determining whether a point
                is inside/outside the volume
        """
        x, y, z = bfl_v
        self.x, self.y, self.z = x, y, z
        self.bfl_v = bfl_v
        self.verts = [
            [x + (v & 1), y + (v >> 1 & 1), z + (v >> 2 & 1)] for v in range(8)
        ]  # vertex position (x, y, z) for v0-v1-v4-v5-v3-v2-v7-v6

        # Calculates cube configuration index given values of the cube vertices
        self.cube_index = 0
        for i in range(8):
            v = self.verts[INDEX[i]]
            value = volume[v[2]][v[1]][v[0]]
            if value < isolevel:
                self.cube_index |= 1 << i

    def get_vpair_from_edge(self, edge: int, W: int, H: int) -> Tuple[int, int]:
        """
        Get a tuple of global vertex ID from a local edge ID
        Global vertex ID is calculated as (x + dx) + (y + dy) * W + (z + dz) * W * H

        Args:
            edge: local edge ID in the cube
            bfl_vertex: bottom-front-left coordinate of the cube

        Returns:
            a pair of global vertex ID
        """
        v1, v2 = EDGE_TO_VERTICES[edge]  # two end-points on the edge
        v1_id = self.verts[v1][0] + self.verts[v1][1] * W + self.verts[v1][2] * W * H
        v2_id = self.verts[v2][0] + self.verts[v2][1] * W + self.verts[v2][2] * W * H
        return (v1_id, v2_id)

    def vert_interp(
        self,
        isolevel: float,
        edge: int,
        vol: torch.Tensor,
    ) -> List:
        """
        Linearly interpolate a vertex where an isosurface cuts an edge
        between the two endpoint vertices, based on their values

        Args:
            isolevel: the isosurface value to use as the threshold to determine
                whether points are within a volume.
            edge: edge (ID) to interpolate
            cube: current cube vertices
            vol: 3D scalar field

        Returns:
            interpolated vertex: position of the interpolated vertex on the edge
        """
        v1, v2 = EDGE_TO_VERTICES[edge]
        p1, p2 = self.verts[v1], self.verts[v2]
        val1, val2 = (
            vol[p1[2]][p1[1]][p1[0]],
            vol[p2[2]][p2[1]][p2[0]],
        )
        point = None
        if abs(isolevel - val1) < EPS:
            point = p1
        elif abs(isolevel - val2) < EPS:
            point = p2
        elif abs(val1 - val2) < EPS:
            point = p1

        if point is None:
            mu = (isolevel - val1) / (val2 - val1)
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x = x1 + mu * (x2 - x1)
            y = y1 + mu * (y2 - y1)
            z = z1 + mu * (z2 - z1)
        else:
            x, y, z = point
        return [x, y, z]


def marching_cubes_naive(
    vol_batch: torch.Tensor,
    isolevel: Optional[float] = None,
    return_local_coords: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs the classic marching cubes algorithm, iterating over
    the coordinates of the volume and using a given isolevel
    for determining intersected edges of cubes.
    Returns vertices and faces of the obtained mesh.
    This operation is non-differentiable.

    Args:
        vol_batch: a Tensor of size (N, D, H, W) corresponding to
            a batch of 3D scalar fields
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume. If None, then the average of the
            maximum and minimum value of the scalar field will be used.
        return_local_coords: bool. If True the output vertices will be in local coordinates in
        the range [-1, 1] x [-1, 1] x [-1, 1]. If False they will be in the range
        [0, W-1] x [0, H-1] x [0, D-1]
    Returns:
        verts: [{V_0}, {V_1}, ...] List of N sets of vertices of shape (|V_i|, 3) in FloatTensor
        faces: [{F_0}, {F_1}, ...] List of N sets of faces of shape (|F_i|, 3) in LongTensors
    """
    batched_verts, batched_faces = [], []
    D, H, W = vol_batch.shape[1:]

    # each edge is represented with its two endpoints (represented with global id)
    for i in range(len(vol_batch)):
        vol = vol_batch[i]
        thresh = ((vol.max() + vol.min()) / 2).item() if isolevel is None else isolevel
        vpair_to_edge = {}  # maps from tuple of edge endpoints to edge_id
        edge_id_to_v = {}  # maps from edge ID to vertex position
        uniq_edge_id = {}  # unique edge IDs
        verts = []  # store vertex positions
        faces = []  # store face indices
        # enumerate each cell in the 3d grid
        for z in range(0, D - 1):
            for y in range(0, H - 1):
                for x in range(0, W - 1):
                    cube = Cube((x, y, z), vol, thresh)
                    edge_indices = FACE_TABLE[cube.cube_index]
                    # cube is entirely in/out of the surface
                    if len(edge_indices) == 0:
                        continue

                    # gather mesh vertices/faces by processing each cube
                    interp_points = [[0.0, 0.0, 0.0]] * 12
                    # triangle vertex IDs and positions
                    tri = []
                    ps = []
                    for i, edge in enumerate(edge_indices):
                        interp_points[edge] = cube.vert_interp(thresh, edge, vol)

                        # Bind interpolated vertex with a global edge_id, which
                        # is represented by a pair of vertex ids (v1_id, v2_id)
                        # corresponding to a local edge.
                        (v1_id, v2_id) = cube.get_vpair_from_edge(edge, W, H)
                        edge_id = vpair_to_edge.setdefault(
                            (v1_id, v2_id), len(vpair_to_edge)
                        )
                        tri.append(edge_id)
                        ps.append(interp_points[edge])
                        # when the isolevel are the same as the edge endpoints, the interploated
                        # vertices can share the same values, and lead to degenerate triangles.
                        if (
                            (i + 1) % 3 == 0
                            and ps[0] != ps[1]
                            and ps[1] != ps[2]
                            and ps[2] != ps[0]
                        ):
                            for j, edge_id in enumerate(tri):
                                edge_id_to_v[edge_id] = ps[j]
                                if edge_id not in uniq_edge_id:
                                    uniq_edge_id[edge_id] = len(verts)
                                    verts.append(edge_id_to_v[edge_id])
                            faces.append([uniq_edge_id[tri[j]] for j in range(3)])
                            tri = []
                            ps = []

        if len(faces) > 0 and len(verts) > 0:
            verts = torch.tensor(verts, dtype=vol.dtype)
            # Convert from world coordinates ([0, D-1], [0, H-1], [0, W-1]) to
            # local coordinates in the range [-1, 1]
            if return_local_coords:
                verts = (
                    Translate(x=+1.0, y=+1.0, z=+1.0, device=vol_batch.device)
                    .scale((vol_batch.new_tensor([W, H, D])[None] - 1) * 0.5)
                    .inverse()
                ).transform_points(verts[None])[0]
            batched_verts.append(verts)
            batched_faces.append(torch.tensor(faces, dtype=torch.int64))
        else:
            batched_verts.append([])
            batched_faces.append([])
    return batched_verts, batched_faces


########################################
# Marching Cubes Implementation in C++/Cuda
########################################
class _marching_cubes(Function):
    """
    Torch Function wrapper for marching_cubes implementation.
    This function is not differentiable. An autograd wrapper is used
    to ensure an error if user tries to get gradients.
    """

    @staticmethod
    def forward(ctx, vol, isolevel):
        verts, faces, ids = _C.marching_cubes(vol, isolevel)
        return verts, faces, ids

    @staticmethod
    def backward(ctx, grad_verts, grad_faces):
        raise ValueError("marching_cubes backward is not supported")


def marching_cubes(
    vol_batch: torch.Tensor,
    isolevel: Optional[float] = None,
    return_local_coords: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run marching cubes over a volume scalar field with a designated isolevel.
    Returns vertices and faces of the obtained mesh.
    This operation is non-differentiable.

    Args:
        vol_batch: a Tensor of size (N, D, H, W) corresponding to
            a batch of 3D scalar fields
        isolevel: float used as threshold to determine if a point is inside/outside
            the volume.  If None, then the average of the maximum and minimum value
            of the scalar field is used.
        return_local_coords: bool. If True the output vertices will be in local coordinates in
            the range [-1, 1] x [-1, 1] x [-1, 1]. If False they will be in the range
            [0, W-1] x [0, H-1] x [0, D-1]

    Returns:
        verts: [{V_0}, {V_1}, ...] List of N sets of vertices of shape (|V_i|, 3) in FloatTensor
        faces: [{F_0}, {F_1}, ...] List of N sets of faces of shape (|F_i|, 3) in LongTensors
    """
    batched_verts, batched_faces = [], []
    D, H, W = vol_batch.shape[1:]
    for i in range(len(vol_batch)):
        vol = vol_batch[i]
        thresh = ((vol.max() + vol.min()) / 2).item() if isolevel is None else isolevel
        verts, faces, ids = _marching_cubes.apply(vol, thresh)
        if len(faces) > 0 and len(verts) > 0:
            # Convert from world coordinates ([0, D-1], [0, H-1], [0, W-1]) to
            # local coordinates in the range [-1, 1]
            if return_local_coords:
                verts = (
                    Translate(x=+1.0, y=+1.0, z=+1.0, device=vol.device)
                    .scale((vol.new_tensor([W, H, D])[None] - 1) * 0.5)
                    .inverse()
                ).transform_points(verts[None])[0]
            # deduplication for cuda
            if vol.is_cuda:
                unique_ids, inverse_idx = torch.unique(ids, return_inverse=True)
                verts_ = verts.new_zeros(unique_ids.shape[0], 3)
                verts_[inverse_idx] = verts
                verts = verts_
                faces = inverse_idx[faces]
            batched_verts.append(verts)
            batched_faces.append(faces)
        else:
            batched_verts.append([])
            batched_faces.append([])
    return batched_verts, batched_faces


def polygonise(
    cube: Cube,
    isolevel: float,
    volume_data: torch.Tensor,
    edge_vertices_to_index: Dict[Tuple[Tuple, Tuple], int],
    vertex_coords_to_index: Dict[Tuple[float, float, float], int],
) -> Tuple[list, list]:
    """
    Runs the classic marching cubes algorithm for one Cube in the volume.
    Returns the vertices and faces for the given cube.

    Args:
        cube: a Cube indicating the cube being examined for edges that intersect
            the volume data.
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume.
        volume_data: a Tensor of shape (D, H, W) corresponding to
            a 3D scalar field
        edge_vertices_to_index: A dictionary which maps an edge's two coordinates
            to the index of its interpolated point, if that interpolated point
            has already been used by a previous point
        vertex_coords_to_index: A dictionary mapping a point (x, y, z) to the corresponding
            index of that vertex, if that point has already been marked as a vertex.
    Returns:
        verts: List of triangle vertices for the given cube in the volume
        faces: List of triangle faces for the given cube in the volume
    """
    num_existing_verts = max(edge_vertices_to_index.values(), default=-1) + 1
    verts, faces = [], []
    cube_index = cube.get_index(volume_data, isolevel)
    edges = EDGE_TABLE[cube_index]
    edge_indices = _get_edge_indices(edges)
    if len(edge_indices) == 0:
        return [], []

    new_verts, edge_index_to_point_index = _calculate_interp_vertices(
        edge_indices,
        volume_data,
        cube,
        isolevel,
        edge_vertices_to_index,
        vertex_coords_to_index,
        num_existing_verts,
    )

    # Create faces
    face_triangles = FACE_TABLE[cube_index]
    for i in range(0, len(face_triangles), 3):
        tri1 = edge_index_to_point_index[face_triangles[i]]
        tri2 = edge_index_to_point_index[face_triangles[i + 1]]
        tri3 = edge_index_to_point_index[face_triangles[i + 2]]
        if tri1 != tri2 and tri2 != tri3 and tri1 != tri3:
            faces.append([tri1, tri2, tri3])

    verts += new_verts
    return verts, faces


def _get_edge_indices(edges: int) -> List[int]:
    """
    Finds which edge numbers are intersected given the bit representation
    detailed in marching_cubes_data.EDGE_TABLE.

    Args:
        edges: an integer corresponding to the value at cube_index
            from the EDGE_TABLE in marching_cubes_data.py

    Returns:
        edge_indices: A list of edge indices
    """
    if edges == 0:
        return []

    edge_indices = []
    for i in range(12):
        if edges & (2 ** i):
            edge_indices.append(i)
    return edge_indices


def _calculate_interp_vertices(
    edge_indices: List[int],
    volume_data: torch.Tensor,
    cube: Cube,
    isolevel: float,
    edge_vertices_to_index: Dict[Tuple[Tuple, Tuple], int],
    vertex_coords_to_index: Dict[Tuple[float, float, float], int],
    num_existing_verts: int,
) -> Tuple[List, Dict[int, int]]:
    """
    Finds the interpolated vertices for the intersected edges, either referencing
    previous calculations or newly calculating and storing the new interpolated
    points.

    Args:
        edge_indices: the numbers of the edges which are intersected. See the
            Cube class for more detail on the edge numbering convention.
        volume_data: a Tensor of size (D, H, W) corresponding to
            a 3D scalar field
        cube: a Cube indicating the cube being examined for edges that intersect
            the volume
        isolevel: the isosurface value to use as the threshold to determine
            whether points are within a volume.
        edge_vertices_to_index: A dictionary which maps an edge's two coordinates
            to the index of its interpolated point, if that interpolated point
            has already been used by a previous point
        vertex_coords_to_index: A dictionary mapping a point (x, y, z) to the corresponding
            index of that vertex, if that point has already been marked as a vertex.
        num_existing_verts: the number of vertices that have been found in previous
            calls to polygonise for the given volume_data in the above function, marching_cubes.
            This is equal to the 1 + the maximum value in edge_vertices_to_index.
    Returns:
        interp_points: a list of new interpolated points
        edge_index_to_point_index: a dictionary mapping an edge number to the index in the
            marching cubes' vertices list of the interpolated point on that edge. To be precise,
            it refers to the index within the vertices list after interp_points
            has been appended to the verts list constructed in the marching_cubes_naive
            function.
    """
    interp_points = []
    edge_index_to_point_index = {}
    for edge_index in edge_indices:
        v1, v2 = EDGE_TO_VERTICES[edge_index]
        point1, point2 = cube.vertices[v1], cube.vertices[v2]
        p_tuple1, p_tuple2 = tuple(point1.tolist()), tuple(point2.tolist())
        if (p_tuple1, p_tuple2) in edge_vertices_to_index:
            edge_index_to_point_index[edge_index] = edge_vertices_to_index[
                (p_tuple1, p_tuple2)
            ]
        else:
            val1, val2 = _get_value(point1, volume_data), _get_value(
                point2, volume_data
            )

            point = None
            if abs(isolevel - val1) < EPS:
                point = point1

            if abs(isolevel - val2) < EPS:
                point = point2

            if abs(val1 - val2) < EPS:
                point = point1

            if point is None:
                mu = (isolevel - val1) / (val2 - val1)
                x1, y1, z1 = point1
                x2, y2, z2 = point2
                x = x1 + mu * (x2 - x1)
                y = y1 + mu * (y2 - y1)
                z = z1 + mu * (z2 - z1)
            else:
                x, y, z = point

            x, y, z = x.item(), y.item(), z.item()  # for dictionary keys

            vert_index = None
            if (x, y, z) in vertex_coords_to_index:
                vert_index = vertex_coords_to_index[(x, y, z)]
            else:
                vert_index = num_existing_verts + len(interp_points)
                interp_points.append([x, y, z])
                vertex_coords_to_index[(x, y, z)] = vert_index

            edge_vertices_to_index[(p_tuple1, p_tuple2)] = vert_index
            edge_index_to_point_index[edge_index] = vert_index

    return interp_points, edge_index_to_point_index


def _get_value(point: Tuple[int, int, int], volume_data: torch.Tensor) -> float:
    """
    Gets the value at a given coordinate point in the scalar field.

    Args:
        point: data of shape (3) corresponding to an xyz coordinate.
        volume_data: a Tensor of size (D, H, W) corresponding to
            a 3D scalar field
    Returns:
        data: scalar value in the volume at the given point
    """
    x, y, z = point
    return volume_data[z][y][x]
