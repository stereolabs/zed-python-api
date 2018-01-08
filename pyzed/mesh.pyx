########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

# Source file of the mesh Python module.

from cython.operator cimport dereference as deref
from cpython cimport bool

import enum
import numpy as np
cimport numpy as np

cimport pyzed.types as types


class PyMESH_FILE_FORMAT(enum.Enum):
    PyMESH_FILE_PLY = MESH_FILE_PLY
    PyMESH_FILE_PLY_BIN = MESH_FILE_PLY_BIN
    PyMESH_FILE_OBJ = MESH_FILE_OBJ
    PyMESH_FILE_LAST = MESH_FILE_LAST

class PyMESH_TEXTURE_FORMAT(enum.Enum):
    PyMESH_TEXTURE_RGB = MESH_TEXTURE_RGB
    PyMESH_TEXTURE_RGBA = MESH_TEXTURE_RGBA
    PyMESH_TEXTURE_LAST = MESH_TEXTURE_LAST

class PyFILTER(enum.Enum):
    PyFILTER_LOW = MESH_FILTER_LOW
    PyFILTER_MEDIUM = MESH_FILTER_MEDIUM
    PyFILTER_HIGH = MESH_FILTER_HIGH


cdef class PyMeshFilterParameters:
    cdef MeshFilterParameters* meshFilter
    def __cinit__(self):
        self.meshFilter = new MeshFilterParameters(MESH_FILTER_LOW)

    def set(self, filter=PyFILTER.PyFILTER_LOW):
        if isinstance(filter, PyFILTER):
            self.meshFilter.set(filter.value)
        else:
            raise TypeError("Argument is not of PyFILTER type.")

    def save(self, str filename):
        filename_save = filename.encode()
        return self.meshFilter.save(types.String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.meshFilter.load(types.String(<char*> filename_load))


cdef class PyTexture:
    def __cinit__(self):
        self.texture = Texture()

    @property
    def name(self):
        if not self.texture.name.empty():
            return self.texture.name.get().decode()
        else:
            return ""

    def get_data(self, core.PyMat py_mat):
       py_mat.mat = self.texture.data
       return py_mat

    @property
    def indice_gl(self):
        return self.texture.indice_gl

    def clear(self):
        self.texture.clear()


cdef class PyChunk:
    def __cinit__(self):
        self.chunk = Chunk()

    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 3))
        for i in range(self.chunk.vertices.size()):
            for j in range(3):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    @property
    def triangles(self):
        cdef np.ndarray arr = np.zeros((self.chunk.triangles.size(), 3))
        for i in range(self.chunk.triangles.size()):
            for j in range(3):
                arr[i,j] = self.chunk.triangles[i].ptr()[j]+1
        return arr

    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3))
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    @property
    def uv(self):
        cdef np.ndarray arr = np.zeros((self.chunk.uv.size(), 2))
        for i in range(self.chunk.uv.size()):
            for j in range(2):
                arr[i,j] = self.chunk.uv[i].ptr()[j]
        return arr

    @property
    def timestamp(self):
        return self.chunk.timestamp

    @property
    def barycenter(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    @property
    def has_been_updated(self):
        return self.chunk.has_been_updated

    def clear(self):
        self.chunk.clear()

cdef class PyMesh:
    def __cinit__(self):
        self.mesh = new Mesh()

    @property
    def chunks(self):
        list = []
        for i in range(self.mesh.chunks.size()):
            py_chunk = PyChunk()
            py_chunk.chunk = self.mesh.chunks[i]
            list.append(py_chunk)
        return list

    def __getitem__(self, x):
        return self.chunks[x]

    def filter(self, PyMeshFilterParameters params, update_mesh=True):
        if isinstance(update_mesh, bool):
            return self.mesh.filter(deref(params.meshFilter), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def apply_texture(self, texture_format=PyMESH_TEXTURE_FORMAT.PyMESH_TEXTURE_RGB):
        if isinstance(texture_format, PyMESH_TEXTURE_FORMAT):
            return self.mesh.applyTexture(texture_format.value)
        else:
            raise TypeError("Argument is not of PyMESH_TEXTURE_FORMAT type.")

    def save(self, str filename, typeMesh=PyMESH_FILE_FORMAT.PyMESH_FILE_OBJ, id=[]):
        if isinstance(typeMesh, PyMESH_FILE_FORMAT):
            return self.mesh.save(types.String(filename.encode()), typeMesh.value, id)
        else:
            raise TypeError("Argument is not of PyMESH_FILE_FORMAT type.")

    def load(self, str filename, update_mesh=True):
        if isinstance(update_mesh, bool):
            return self.mesh.load(types.String(filename.encode()), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def clear(self):
        self.mesh.clear()

    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.mesh.vertices.size(), 3))
        for i in range(self.mesh.vertices.size()):
            for j in range(3):
                arr[i,j] = self.mesh.vertices[i].ptr()[j]
        return arr

    @property
    def triangles(self):
        cdef np.ndarray arr = np.zeros((self.mesh.triangles.size(), 3))
        for i in range(self.mesh.triangles.size()):
            for j in range(3):
                arr[i,j] = self.mesh.triangles[i].ptr()[j]+1
        return arr

    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.mesh.normals.size(), 3))
        for i in range(self.mesh.normals.size()):
            for j in range(3):
                arr[i,j] = self.mesh.normals[i].ptr()[j]
        return arr

    @property
    def uv(self):
        cdef np.ndarray arr = np.zeros((self.mesh.uv.size(), 2))
        for i in range(self.mesh.uv.size()):
            for j in range(2):
                arr[i,j] = self.mesh.uv[i].ptr()[j]
        return arr

    @property
    def texture(self):
        py_texture = PyTexture()
        py_texture.texture = self.mesh.texture
        return py_texture

    def get_number_of_triangles(self):
        return self.mesh.getNumberOfTriangles()

    def merge_chunks(self, faces_per_chunk):
        self.mesh.mergeChunks(faces_per_chunk)

    def get_gravity_estimate(self):
        gravity = self.mesh.getGravityEstimate()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = gravity[i]
        return arr

    def get_visible_list(self, core.PyTransform camera_pose):
        return self.mesh.getVisibleList(camera_pose.transform)

    def get_surrounding_list(self, core.PyTransform camera_pose, float radius):
        return self.mesh.getSurroundingList(camera_pose.transform, radius)

    def update_mesh_from_chunklist(self, id=[]):
        self.mesh.updateMeshFromChunkList(id)
