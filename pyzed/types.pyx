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

# Source file of the types Python module.

import enum

import numpy as np
cimport numpy as np
from math import sqrt


class PyERROR_CODE(enum.Enum):
    PySUCCESS = SUCCESS
    PyERROR_CODE_FAILURE = ERROR_CODE_FAILURE
    PyERROR_CODE_NO_GPU_COMPATIBLE = ERROR_CODE_NO_GPU_COMPATIBLE
    PyERROR_CODE_NOT_ENOUGH_GPUMEM = ERROR_CODE_NOT_ENOUGH_GPUMEM
    PyERROR_CODE_CAMERA_NOT_DETECTED = ERROR_CODE_CAMERA_NOT_DETECTED
    PyERROR_CODE_SENSOR_NOT_DETECTED = ERROR_CODE_SENSOR_NOT_DETECTED
    PyERROR_CODE_INVALID_RESOLUTION = ERROR_CODE_INVALID_RESOLUTION
    PyERROR_CODE_LOW_USB_BANDWIDTH = ERROR_CODE_LOW_USB_BANDWIDTH
    PyERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE = ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE
    PyERROR_CODE_INVALID_SVO_FILE = ERROR_CODE_INVALID_SVO_FILE
    PyERROR_CODE_SVO_RECORDING_ERROR = ERROR_CODE_SVO_RECORDING_ERROR
    PyERROR_CODE_INVALID_COORDINATE_SYSTEM = ERROR_CODE_INVALID_COORDINATE_SYSTEM
    PyERROR_CODE_INVALID_FIRMWARE = ERROR_CODE_INVALID_FIRMWARE
    PyERROR_CODE_INVALID_FUNCTION_PARAMETERS = ERROR_CODE_INVALID_FUNCTION_PARAMETERS
    PyERROR_CODE_NOT_A_NEW_FRAME = ERROR_CODE_NOT_A_NEW_FRAME
    PyERROR_CODE_CUDA_ERROR = ERROR_CODE_CUDA_ERROR
    PyERROR_CODE_CAMERA_NOT_INITIALIZED = ERROR_CODE_CAMERA_NOT_INITIALIZED
    PyERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE = ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE
    PyERROR_CODE_INVALID_FUNCTION_CALL = ERROR_CODE_INVALID_FUNCTION_CALL
    PyERROR_CODE_CORRUPTED_SDK_INSTALLATION = ERROR_CODE_CORRUPTED_SDK_INSTALLATION
    PyERROR_CODE_INCOMPATIBLE_SDK_VERSION = ERROR_CODE_INCOMPATIBLE_SDK_VERSION
    PyERROR_CODE_INVALID_AREA_FILE = ERROR_CODE_INVALID_AREA_FILE
    PyERROR_CODE_INCOMPATIBLE_AREA_FILE = ERROR_CODE_INCOMPATIBLE_AREA_FILE 
    PyERROR_CODE_CAMERA_DETECTION_ISSUE = ERROR_CODE_CAMERA_DETECTION_ISSUE
    PyERROR_CODE_CAMERA_ALREADY_IN_USE = ERROR_CODE_CAMERA_ALREADY_IN_USE
    PyERROR_CODE_NO_GPU_DETECTED = ERROR_CODE_NO_GPU_DETECTED
    PyERROR_CODE_LAST = ERROR_CODE_LAST

    def __str__(self):
        return to_str(errorCode2str(self.value)).decode()

    def __repr__(self):
        return to_str(errorCode2str(self.value)).decode()

class PyMODEL(enum.Enum):
    PyMODEL_ZED = MODEL_ZED
    PyMODEL_ZED_M = MODEL_ZED_M
    PyMODEL_LAST = MODEL_LAST
    
    def __str__(self):
        return to_str(model2str(self.value)).decode()

    def __repr__(self):
        return to_str(model2str(self.value)).decode()


def c_sleep_ms(int time):
    sleep_ms(time)


cdef class PyMatrix3f:
    def __cinit__(self):
        self.mat = Matrix3f()

    def init_matrix(self, PyMatrix3f matrix):
        self.mat = Matrix3f(matrix.mat)

    def inverse(self):
        self.mat.inverse()

    def inverse_mat(self, PyMatrix3f rotation):
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        self.mat.transpose()

    def transpose(self, PyMatrix3f rotation):
        rotation.mat.transpose(rotation.mat)
        return rotation

    def set_identity(self):
        self.mat.setIdentity()
        return self

    def identity(self):
        self.mat.identity()
        return self

    def set_zeros(self):
        self.mat.setZeros()
 
    def zeros(self):
        self.mat.zeros()
        return self

    def get_infos(self):
        return to_str(self.mat.getInfos()).decode()

    @property
    def matrix_name(self):
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @property
    def nbElem(self):
        return self.mat.nbElem

    @property
    def r(self):
        nbElem = self.nbElem
        sqrt_nbElem = int(sqrt(nbElem))
        cdef np.ndarray arr = np.zeros(nbElem)
        for i in range(nbElem):
            arr[i] = self.mat.r[i]
        return arr.reshape(sqrt_nbElem, sqrt_nbElem)

    @r.setter
    def r(self, value):
        if isinstance(value, list):
            if len(value) <= self.nbElem:
                for i in range(len(value)):
                    self.mat.r[i] = value[i]
            else:
                raise IndexError("Value list must be of length 9 max.")
        elif isinstance(value, np.ndarray):
            if value.size <= self.nbElem:
                for i in range(value.size):
                    self.mat.r[i] = value[i]
            else:
                raise IndexError("Numpy array must be of size 9.")
        else:
            raise TypeError("Argument must be numpy array or list type.")

    def __mul__(self, other):
        matrix = PyMatrix3f()
        if isinstance(other, PyMatrix3f):
            matrix.r = (self.r * other.r).reshape(self.nbElem)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.r = (other * self.r).reshape(self.nbElem)
        else:
            raise TypeError("Argument must be PyMatrix3f or scalar type.")
        return matrix

    def __richcmp__(PyMatrix3f left, PyMatrix3f right, int op):
        if op == 2:
            return left.mat == right.mat
        if op == 3:
            return left.mat != right.mat
        else:
            raise NotImplementedError()

    def __getitem__(self, x):
        return self.r[x]

    def __setitem__(self, key, value):
        if key == (0,0):
            self.mat.r00 = value
        elif key == (0,1):
            self.mat.r01 = value
        elif key == (0,2):
            self.mat.r02 = value
        elif key == (1,0):
            self.mat.r10 = value
        elif key == (1,1):
            self.mat.r11 = value
        elif key == (1,2):
            self.mat.r12 = value
        elif key == (2,0):
            self.mat.r20 = value
        elif key == (2,1):
            self.mat.r21 = value
        elif key == (2,2):
            self.mat.r22 = value

    def __repr__(self):
        return self.get_infos()


cdef class PyMatrix4f:
    def __cinit__(self):
        self.mat = Matrix4f()

    def init_matrix(self, PyMatrix4f matrix):
        self.mat = Matrix4f(matrix.mat)

    def inverse(self):
        return PyERROR_CODE(self.mat.inverse())

    def inverse_mat(self, PyMatrix4f rotation):
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        self.mat.transpose()

    def transpose(self, PyMatrix4f rotation):
        rotation.mat.transpose(rotation.mat)
        return rotation

    def set_identity(self):
        self.mat.setIdentity()
        return self

    def identity(self):
        self.mat.identity()
        return self

    def set_zeros(self):
        self.mat.setZeros()

    def zeros(self):
        self.mat.zeros()
        return self

    def get_infos(self):
        return to_str(self.mat.getInfos()).decode()

    def set_sub_matrix3f(self, PyMatrix3f input, row=0, column=0):
        if row != 0 and row != 1 or column != 0 and column != 1:
            raise TypeError("Arguments row and column must be 0 or 1.")
        else:
            return PyERROR_CODE(self.mat.setSubMatrix3f(input.mat, row, column))

    def set_sub_vector3f(self, float input0, float input1, float input2, column=3):
        return PyERROR_CODE(self.mat.setSubVector3f(Vector3[float](input0, input1, input2), column))

    def set_sub_vector4f(self, float input0, float input1, float input2, float input3, column=3):
        return PyERROR_CODE(self.mat.setSubVector4f(Vector4[float](input0, input1, input2, input3), column))

    @property
    def nbElem(self):
        return self.mat.nbElem

    @property
    def matrix_name(self):
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @property
    def m(self):
        nbElem = self.nbElem
        sqrt_nbElem = int(sqrt(nbElem))
        cdef np.ndarray arr = np.zeros(nbElem)
        for i in range(nbElem):
            arr[i] = self.mat.m[i]
        return arr.reshape(sqrt_nbElem, sqrt_nbElem)

    @m.setter
    def m(self, value):
        if isinstance(value, list):
            if len(value) <= self.nbElem:
                for i in range(len(value)):
                    self.mat.m[i] = value[i]
            else:
                raise IndexError("Value list must be of length 16 max.")
        elif isinstance(value, np.ndarray):
            if value.size <= self.nbElem:
                for i in range(value.size):
                    self.mat.m[i] = value[i]
            else:
                raise IndexError("Numpy array must be of size 16.")
        else:
            raise TypeError("Argument must be numpy array or list type.")

    def __mul__(self, other):
        matrix = PyMatrix4f()
        if isinstance(other, PyMatrix4f):
            matrix.m = (self.m * other.m).reshape(self.nbElem)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.m = (other * self.m).reshape(self.nbElem)
        else:
            raise TypeError("Argument must be PyMatrix4f or scalar type.")
        return matrix

    def __richcmp__(PyMatrix4f left, PyMatrix4f right, int op):
        if op == 2:
            return left.mat == right.mat
        if op == 3:
            return left.mat != right.mat
        else:
            raise NotImplementedError()

    def __getitem__(self, x):
        return self.m[x]

    def __setitem__(self, key, value):
        if key == (0,0):
            self.mat.r00 = value
        elif key == (0,1):
            self.mat.r01 = value
        elif key == (0,2):
            self.mat.r02 = value
        elif key == (0,3):
            self.mat.tx = value
        elif key == (1,0):
            self.mat.r10 = value
        elif key == (1,1):
            self.mat.r11 = value
        elif key == (1,2):
            self.mat.r12 = value
        elif key == (1,3):
            self.mat.ty = value
        elif key == (2,0):
            self.mat.r20 = value
        elif key == (2,1):
            self.mat.r21 = value
        elif key == (2,2):
            self.mat.r22 = value
        elif key == (2,3):
            self.mat.tz = value
        elif key == (3,0):
            self.mat.m30 = value
        elif key == (3,1):
            self.mat.m31 = value
        elif key == (3,2):
            self.mat.m32 = value
        elif key == (3,3):
            self.mat.m33 = value

    def __repr__(self):
        return self.get_infos()
