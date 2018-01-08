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

# Source file of the core Python module.

from cython.operator cimport dereference as deref
from libc.string cimport memcpy
from cpython cimport bool

import enum
import numpy as np
cimport numpy as np
import pyzed.types as types
cimport pyzed.camera as camera
import pyzed.camera as camera


class PyMEM(enum.Enum):
    PyMEM_CPU = MEM_CPU
    PyMEM_GPU = MEM_GPU


class PyCOPY_TYPE(enum.Enum):
    PyCOPY_TYPE_CPU_CPU = COPY_TYPE_CPU_CPU
    PyCOPY_TYPE_CPU_GPU = COPY_TYPE_CPU_GPU
    PyCOPY_TYPE_GPU_GPU = COPY_TYPE_GPU_GPU
    PyCOPY_TYPE_GPU_CPU = COPY_TYPE_GPU_CPU


class PyMAT_TYPE(enum.Enum):
    PyMAT_TYPE_32F_C1 = MAT_TYPE_32F_C1
    PyMAT_TYPE_32F_C2 = MAT_TYPE_32F_C2
    PyMAT_TYPE_32F_C3 = MAT_TYPE_32F_C3
    PyMAT_TYPE_32F_C4 = MAT_TYPE_32F_C4
    PyMAT_TYPE_8U_C1 = MAT_TYPE_8U_C1
    PyMAT_TYPE_8U_C2 = MAT_TYPE_8U_C2
    PyMAT_TYPE_8U_C3 = MAT_TYPE_8U_C3
    PyMAT_TYPE_8U_C4 = MAT_TYPE_8U_C4


def get_current_timestamp():
    return getCurrentTimeStamp()


cdef class PyResolution:
    cdef size_t width
    cdef size_t height
    def __cinit__(self, width=0, height=0):
        self.width = width
        self.height = height

    def py_area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.width

    @property
    def height(self):
        return self.height

    def __richcmp__(PyResolution left, PyResolution right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height
        if op == 3:
            return left.width!=right.width or left.height!=right.height
        else:
            raise NotImplementedError()


cdef class PyCameraParameters:
    @property
    def fx(self):
        return self.camera_params.fx

    @property
    def fy(self):
        return self.camera_params.fy

    @property
    def cx(self):
        return self.camera_params.cx

    @property
    def cy(self):
        return self.camera_params.cy

    @property
    def disto(self):
        return self.camera_params.disto

    @property
    def v_fov(self):
        return self.camera_params.v_fov

    @property
    def h_fov(self):
        return self.camera_params.h_fov

    @property
    def d_fov(self):
        return self.camera_params.d_fov

    @property
    def image_size(self):
        return self.camera_params.image_size


cdef class PyCalibrationParameters:
    def __cinit__(self):
        self.py_left_cam = PyCameraParameters()
        self.py_right_cam = PyCameraParameters()

    def set(self):
        self.py_left_cam.camera_params = self.calibration.left_cam
        self.py_right_cam.camera_params = self.calibration.right_cam
        self.R = self.calibration.R
        self.T = self.calibration.T

    @property
    def R(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.calibration.R[i]
        return arr

    @property
    def T(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.calibration.T[i]
        return arr

    @property
    def left_cam(self):
        return self.py_left_cam

    @property
    def right_cam(self):
        return self.py_right_cam


cdef class PyCameraInformation:
    def __cinit__(self, camera.PyZEDCamera py_camera, PyResolution resizer):
        res = Resolution(resizer.width, resizer.height)
        self.py_calib = PyCalibrationParameters()
        self.py_calib.calibration = py_camera.camera.getCameraInformation(res).calibration_parameters
        self.py_calib_raw = PyCalibrationParameters()
        self.py_calib_raw.calibration = py_camera.camera.getCameraInformation(res).calibration_parameters_raw
        self.py_calib.set()
        self.py_calib_raw.set()
        self.serial_number = py_camera.camera.getCameraInformation(res).serial_number
        self.firmware_version = py_camera.camera.getCameraInformation(res).firmware_version
        self.camera_model = py_camera.camera.getCameraInformation(res).camera_model

    @property
    def camera_model(self):
        return self.camera_model

    @property  
    def calibration_parameters(self):
        return self.py_calib

    @property
    def calibration_parameters_raw(self):
        return self.py_calib_raw

    @property
    def serial_number(self):
        return self.serial_number

    @property
    def firmware_version(self):
        return self.firmware_version


cdef class PyMat:
    def __cinit__(self):
        self.mat = Mat()

    def init_mat_type(self, width, height, mat_type, memory_type=PyMEM.PyMEM_CPU):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat = Mat(width, height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Argument are not of PyMAT_TYPE or PyMEM type.")

    def init_mat_cpu(self, width, height, mat_type, ptr, step, memory_type=PyMEM.PyMEM_CPU):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat = Mat(width, height, mat_type.value, ptr.encode(), step, memory_type.value)
        else:
            raise TypeError("Argument are not of PyMAT_TYPE or PyMEM type.")

    def init_mat_cpu_gpu(self, width, height, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        if isinstance(mat_type, PyMAT_TYPE):
             self.mat = Mat(width, height, mat_type.value, ptr_cpu.encode(), step_cpu, ptr_gpu.encode(), step_gpu)
        else:
            raise TypeError("Argument is not of PyMAT_TYPE type.")

    def init_mat_resolution(self, PyResolution resolution, mat_type, memory_type):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat = Mat(Resolution(resolution.width, resolution.height), mat_type.value, memory_type.value)
        else:
            raise TypeError("Argument are not of PyMAT_TYPE or PyMEM type.")

    def init_mat_resolution_cpu(self, PyResolution resolution, mat_type, ptr, step, memory_type=PyMEM.PyMEM_CPU):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat = Mat(Resolution(resolution.width, resolution.height), mat_type.value, ptr.encode(),
                            step, memory_type.value)
        else:
            raise TypeError("Argument are not of PyMAT_TYPE or PyMEM type.")

    def init_mat_resolution_cpu_gpu(self, PyResolution resolution, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        if isinstance(mat_type, PyMAT_TYPE):
             self.mat = matResolution(Resolution(resolution.width, resolution.height), mat_type.value, ptr_cpu.encode(),
                             step_cpu, ptr_gpu.encode(), step_gpu)
        else:
            raise TypeError("Argument is not of PyMAT_TYPE type.")

    def init_mat(self, PyMat matrix):
        self.mat = Mat(matrix.mat)

    def alloc_size(self, width, height, mat_type, memory_type=PyMEM.PyMEM_CPU):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat.alloc(<size_t> width, <size_t> height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of PyMat and PyMEM types.")

    def alloc_resolution(self, PyResolution resolution, mat_type, memory_type=PyMEM.PyMEM_CPU):
        if isinstance(mat_type, PyMAT_TYPE) and isinstance(memory_type, PyMEM):
            self.mat.alloc(resolution.width, resolution.height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of PyMat and PyMEM types.")

    def free(self, memory_type=None):
        if isinstance(memory_type, PyMEM):
            self.mat.free(memory_type.value)
        elif memory_type is None:
            self.mat.free(MEM_CPU or MEM_GPU)
        else:
            raise TypeError("Argument is not of PyMEM type.")

    def update_cpu_from_gpu(self):
        return types.PyERROR_CODE(self.mat.updateCPUfromGPU())

    def update_gpu_from_cpu(self):
        return types.PyERROR_CODE(self.mat.updateGPUfromCPU())

    def copy_to(self, cpy_type=PyCOPY_TYPE.PyCOPY_TYPE_CPU_CPU):
        dst = PyMat()
        print(types.PyERROR_CODE(self.mat.copyTo(dst.mat, cpy_type.value)))
        return dst

    def set_from(self, cpy_type=PyCOPY_TYPE.PyCOPY_TYPE_CPU_CPU):
        dst = PyMat()
        print(self.mat.setFrom(dst.mat, cpy_type.value))
        return dst

    def read(self, str filepath):
        return types.PyERROR_CODE(self.mat.read(filepath.encode()))

    def write(self, str filepath):
        return types.PyERROR_CODE(self.mat.write(filepath.encode()))

    def set_to(self, value, memory_type=PyMEM.PyMEM_CPU):
        if self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C1:
            return types.PyERROR_CODE(setToUchar1(self.mat, value, memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C2:
            return types.PyERROR_CODE(setToUchar2(self.mat, types.Vector2[uchar1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C3:
            return types.PyERROR_CODE(setToUchar3(self.mat, types.Vector3[uchar1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C4:
            return types.PyERROR_CODE(setToUchar4(self.mat, types.Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C1:
            return types.PyERROR_CODE(setToFloat1(self.mat, value, memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C2:
            return types.PyERROR_CODE(setToFloat2(self.mat, types.Vector2[float1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C3:
            return types.PyERROR_CODE(setToFloat3(self.mat, types.Vector3[float1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C4:
            return types.PyERROR_CODE(setToFloat4(self.mat, types.Vector4[float1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))

    def set_value(self, x, y, value, memory_type=PyMEM.PyMEM_CPU):
        if self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C1:
            return types.PyERROR_CODE(setValueUchar1(self.mat, x, y, value, memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C2:
            return types.PyERROR_CODE(setValueUchar2(self.mat, x, y, types.Vector2[uchar1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C3:
            return types.PyERROR_CODE(setValueUchar3(self.mat, x, y, types.Vector3[uchar1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C4:
            return types.PyERROR_CODE(setValueUchar4(self.mat, x, y, types.Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C1:
            return types.PyERROR_CODE(setValueFloat1(self.mat, x, y, value, memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C2:
            return types.PyERROR_CODE(setValueFloat2(self.mat, x, y, types.Vector2[float1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C3:
            return types.PyERROR_CODE(setValueFloat3(self.mat, x, y, types.Vector3[float1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C4:
            return types.PyERROR_CODE(setValueFloat4(self.mat, x, y, types.Vector4[float1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))

    def get_value(self, x, y, memory_type=PyMEM.PyMEM_CPU):
        cdef uchar1 value1u
        cdef types.Vector2[uchar1]* value2u = new types.Vector2[uchar1]()
        cdef types.Vector3[uchar1]* value3u = new types.Vector3[uchar1]()
        cdef types.Vector4[uchar1]* value4u = new types.Vector4[uchar1]()

        cdef float1 value1f
        cdef types.Vector2[float1]* value2f = new types.Vector2[float1]()
        cdef types.Vector3[float1]* value3f = new types.Vector3[float1]()
        cdef types.Vector4[float1]* value4f = new types.Vector4[float1]()

        if self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C1:
            status = getValueUchar1(self.mat, x, y, &value1u, memory_type.value)
            return types.PyERROR_CODE(status), self.get_data()[x, y]
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C2:
            status = getValueUchar2(self.mat, x, y, value2u, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value2u.ptr()[0], value2u.ptr()[1]])
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C3:
            status = getValueUchar3(self.mat, x, y, value3u, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value3u.ptr()[0], value3u.ptr()[1], value3u.ptr()[2]])
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_8U_C4:
            status = getValueUchar4(self.mat, x, y, value4u, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value4u.ptr()[0], value4u.ptr()[1], value4u.ptr()[2],
                                                         value4u.ptr()[3]])

        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C1:
            status = getValueFloat1(self.mat, x, y, &value1f, memory_type.value)
            return types.PyERROR_CODE(status), self.get_data()[x, y]
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C2:
            status = getValueFloat2(self.mat, x, y, value2f, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value2f.ptr()[0], value2f.ptr()[1]])
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C3:
            status = getValueFloat3(self.mat, x, y, value3f, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value3f.ptr()[0], value3f.ptr()[1], value3f.ptr()[2]])
        elif self.get_data_type() == PyMAT_TYPE.PyMAT_TYPE_32F_C4:
            status = getValueFloat4(self.mat, x, y, value4f, memory_type.value)
            return types.PyERROR_CODE(status), np.array([value4f.ptr()[0], value4f.ptr()[1], value4f.ptr()[2],
                                                         value4f.ptr()[3]])

    def get_width(self):
        return self.mat.getWidth()

    def get_height(self):
        return self.mat.getHeight()

    def get_resolution(self):
        return PyResolution(self.mat.getResolution().width, self.mat.getResolution().height)

    def get_channels(self):
        return self.mat.getChannels()

    def get_data_type(self):
        return PyMAT_TYPE(self.mat.getDataType())

    def get_memory_type(self):
        return PyMEM(self.mat.getMemoryType())

    def get_data(self, memory_type=PyMEM.PyMEM_CPU):
        shape = None
        if self.mat.getChannels() == 1:
            shape = (self.mat.getHeight(), self.mat.getWidth())
        else:
            shape = (self.mat.getHeight(), self.mat.getWidth(), self.mat.getChannels())

        cdef size_t size = 0
        dtype = None
        if self.mat.getDataType() in (MAT_TYPE_8U_C1, MAT_TYPE_8U_C2, MAT_TYPE_8U_C3, MAT_TYPE_8U_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()
            dtype = np.uint8
        elif self.mat.getDataType() in (MAT_TYPE_32F_C1, MAT_TYPE_32F_C2, MAT_TYPE_32F_C3, MAT_TYPE_32F_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()*sizeof(float)
            dtype = np.float32
        else:
            raise RuntimeError("Unknown Mat data_type value: {0}".format(self.mat.getDataType()))

        cdef np.ndarray arr = np.zeros(shape, dtype=dtype)
        if isinstance(memory_type, PyMEM):
            if self.mat.getDataType() == MAT_TYPE_8U_C1:
                memcpy(<void*>arr.data, <void*>getPointerUchar1(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_8U_C2:
                memcpy(<void*>arr.data, <void*>getPointerUchar2(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_8U_C3:
                memcpy(<void*>arr.data, <void*>getPointerUchar3(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_8U_C4:
                memcpy(<void*>arr.data, <void*>getPointerUchar4(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_32F_C1:
                memcpy(<void*>arr.data, <void*>getPointerFloat1(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_32F_C2:
                memcpy(<void*>arr.data, <void*>getPointerFloat2(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_32F_C3:
                memcpy(<void*>arr.data, <void*>getPointerFloat3(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == MAT_TYPE_32F_C4:
                memcpy(<void*>arr.data, <void*>getPointerFloat4(self.mat, memory_type.value), size)
        else:
            raise TypeError("Argument is not of PyMEM type.")

        return arr

    def get_step_bytes(self, memory_type=PyMEM.PyMEM_CPU):
        if type(memory_type) == PyMEM:
            return self.mat.getStepBytes(memory_type.value)
        else:
            raise TypeError("Argument is not of PyMEM type.")

    def get_step(self, memory_type=PyMEM.PyMEM_CPU):
        if type(memory_type) == PyMEM:
            return self.mat.getStep(memory_type.value)
        else:
            raise TypeError("Argument is not of PyMEM type.")

    def get_pixel_bytes(self):
        return self.mat.getPixelBytes()

    def get_width_bytes(self):
        return self.mat.getWidthBytes()

    def get_infos(self):
        return self.mat.getInfos().get().decode()

    def is_init(self):
        return self.mat.isInit()

    def is_memory_owner(self):
        return self.mat.isMemoryOwner()

    def clone(self, PyMat py_mat):
        return types.PyERROR_CODE(self.mat.clone(py_mat.mat))

    def move(self, PyMat py_mat):
        return types.PyERROR_CODE(self.mat.move(py_mat.mat))

    @staticmethod
    def swap(self, PyMat mat1, PyMat mat2):
        self.mat.swap(mat1, mat2)

    @property
    def name(self):
        if not self.mat.name.empty():
            return self.mat.name.get().decode()
        else:
            return ""

    @property
    def verbose(self):
        return self.mat.verbose

    def __repr__(self):
        return self.get_infos()


cdef class PyRotation(types.PyMatrix3f):
    def __cinit__(self):
        self.rotation = Rotation()

    def init_rotation(self, PyRotation rot):
        self.rotation = Rotation(rot.rotation)
        self.mat = rot.mat

    def init_matrix(self, types.PyMatrix3f matrix):
        self.rotation = Rotation(matrix.mat)
        self.mat = matrix.mat

    def init_orientation(self, PyOrientation orient):
        self.rotation = Rotation(orient.orientation)
        self.mat = types.Matrix3f(self.rotation.r)

    def init_angle_translation(self, float angle, PyTranslation axis):
        self.rotation = Rotation(angle, axis.translation)
        self.mat = types.Matrix3f(self.rotation.r)

    def set_orientation(self, PyOrientation py_orientation):
        self.rotation.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        py_orientation = PyOrientation()
        py_orientation.orientation = self.rotation.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.rotation.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        self.rotation.setRotationVector(types.Vector3[float](input0, input1, input2))

    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.rotation.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    def set_euler_angles(self, float input0, float input1, float input2, bool radian=True):
        if isinstance(radian, bool):
            self.rotation.setEulerAngles(types.Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")

cdef class PyTranslation:
    def __cinit__(self):
        self.translation = Translation()

    def init_translation(self, PyTranslation tr):
        self.translation = Translation(tr.translation)

    def init_vector(self, float t1, float t2, float t3):
        self.translation = Translation(t1, t2, t3)

    def normalize(self):
        self.translation.normalize()

    def normalize_translation(self, PyTranslation tr):
        py_translation = PyTranslation()
        py_translation.translation = self.translation.normalize(tr.translation)
        return py_translation

    def size(self):
        return self.translation.size()

    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.translation(i)
        return arr

    def __mul__(PyTranslation self, PyOrientation other):
        tr = PyTranslation()
        tr.translation = self.translation * other.orientation
        return tr


cdef class PyOrientation:
    def __cinit__(self):
        self.orientation = Orientation()

    def init_orientation(self, PyOrientation orient):
        self.orientation = Orientation(orient.orientation)

    def init_vector(self, float v0, float v1, float v2, float v3):
        self.orientation = Orientation(types.Vector4[float](v0, v1, v2, v3))

    def init_rotation(self, PyRotation rotation):
        self.orientation = Orientation(rotation.rotation)

    def init_translation(self, PyTranslation tr1, PyTranslation tr2):
        self.orientation = Orientation(tr1.translation, tr2.translation)

    def set_rotation_matrix(self, PyRotation py_rotation):
       self.orientation.setRotationMatrix(py_rotation.rotation)

    def get_rotation_matrix(self):
        py_rotation = PyRotation()
        py_rotation.mat = self.orientation.getRotationMatrix()
        return py_rotation

    def set_identity(self):
        self.orientation.setIdentity()
        return self

    def identity(self):
        self.orientation.identity()
        return self

    def set_zeros(self):
        self.orientation.setZeros()

    def zeros(self):
        self.orientation.zeros()
        return self

    def normalize(self):
        self.orientation.normalise()

    @staticmethod
    def normalize_orientation(PyOrientation orient):
        orient.orientation.normalise()
        return orient

    def size(self):
        return self.orientation.size()

    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.orientation(i)
        return arr

    def __mul__(PyOrientation self, PyOrientation other):
        orient = PyOrientation()
        orient.orientation = self.orientation * other.orientation
        return orient


cdef class PyTransform(types.PyMatrix4f):
    def __cinit__(self):
        self.transform = Transform()

    def init_transform(self, PyTransform motion):
        self.transform = Transform(motion.transform)
        self.mat = motion.mat

    def init_matrix(self, types.PyMatrix4f matrix):
        self.transform = Transform(matrix.mat)
        self.mat = matrix.mat

    def init_rotation_translation(self, PyRotation rot, PyTranslation tr):
        self.transform = Transform(rot.rotation, tr.translation)
        self.mat = types.Matrix4f(self.transform.m)

    def init_orientation_translation(self, PyOrientation orient, PyTranslation tr):
        self.transform = Transform(orient.orientation, tr.translation)
        self.mat = types.Matrix4f(self.transform.m)

    def set_rotation_matrix(self, PyRotation py_rotation):
        self.transform.setRotationMatrix(<Rotation>py_rotation.mat)

    def get_rotation_matrix(self):
        py_rotation = PyRotation()
        py_rotation.mat = self.transform.getRotationMatrix()
        return py_rotation

    def set_translation(self, PyTranslation py_translation):
        self.transform.setTranslation(py_translation.translation)

    def get_translation(self):
        py_translation = PyTranslation()
        py_translation.translation = self.transform.getTranslation()
        return py_translation

    def set_orientation(self, PyOrientation py_orientation):
        self.transform.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        py_orientation = PyOrientation()
        py_orientation.orientation = self.transform.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.transform.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        self.transform.setRotationVector(types.Vector3[float](input0, input1, input2))

    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.transform.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    def set_euler_angles(self, float input0, float input1, float input2, radian=True):
        if isinstance(radian, bool):
            self.transform.setEulerAngles(types.Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")