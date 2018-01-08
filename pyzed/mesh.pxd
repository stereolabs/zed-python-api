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

# File containing the Cython declarations to use the Mesh.hpp functions.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport pyzed.types as types
cimport pyzed.core as core


ctypedef unsigned int uint

cdef extern from "sl/Mesh.hpp" namespace "sl":

    ctypedef enum MESH_FILE_FORMAT:
        MESH_FILE_PLY
        MESH_FILE_PLY_BIN
        MESH_FILE_OBJ
        MESH_FILE_LAST


    ctypedef enum MESH_TEXTURE_FORMAT:
        MESH_TEXTURE_RGB
        MESH_TEXTURE_RGBA
        MESH_TEXTURE_LAST


    ctypedef enum MESH_FILTER 'sl::MeshFilterParameters::MESH_FILTER':
        MESH_FILTER_LOW 'sl::MeshFilterParameters::MESH_FILTER::MESH_FILTER_LOW'
        MESH_FILTER_MEDIUM 'sl::MeshFilterParameters::MESH_FILTER::MESH_FILTER_MEDIUM'
        MESH_FILTER_HIGH 'sl::MeshFilterParameters::MESH_FILTER::MESH_FILTER_HIGH'


    cdef cppclass MeshFilterParameters 'sl::MeshFilterParameters':
        MeshFilterParameters(MESH_FILTER filtering_)
        void set(MESH_FILTER filtering_)
        bool save(types.String filename)
        bool load(types.String filename)


    cdef cppclass Texture 'sl::Texture':
        Texture()
        types.String name
        core.Mat data
        unsigned int indice_gl
        void clear()

    cdef cppclass Chunk 'sl::Chunk':
        Chunk()
        vector[types.Vector3[float]] vertices
        vector[types.Vector3[uint]] triangles
        vector[types.Vector3[float]] normals
        vector[types.Vector2[float]] uv
        unsigned long long timestamp
        types.Vector3[float] barycenter
        bool has_been_updated
        void clear()

    cdef cppclass Mesh 'sl::Mesh':
        ctypedef vector[size_t] chunkList
        Mesh()
        vector[Chunk] chunks
        Chunk &operator[](int index)
        vector[types.Vector3[float]] vertices
        vector[types.Vector3[uint]] triangles
        vector[types.Vector3[float]] normals
        vector[types.Vector2[float]] uv
        Texture texture
        size_t getNumberOfTriangles()
        void mergeChunks(int faces_per_chunk)
        types.Vector3[float] getGravityEstimate()
        chunkList getVisibleList(core.Transform camera_pose)
        chunkList getSurroundingList(core.Transform camera_pose, float radius)
        void updateMeshFromChunkList(chunkList IDs)
        bool filter(MeshFilterParameters params, bool updateMesh)
        bool applyTexture(MESH_TEXTURE_FORMAT texture_format)
        bool save(types.String filename, MESH_FILE_FORMAT type, chunkList IDs)
        bool load(const types.String filename, bool updateMesh)
        void clear()


cdef class PyTexture:
    cdef Texture texture

cdef class PyChunk:
    cdef Chunk chunk

cdef class PyMesh:
    cdef Mesh* mesh
