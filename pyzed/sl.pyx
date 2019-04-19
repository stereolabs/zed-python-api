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

from libcpp.vector cimport vector
from libc.string cimport const_char
from libcpp.string cimport string
from libcpp.pair cimport pair
from sl_c cimport to_str, ERROR_CODE as c_ERROR_CODE, toString, sleep_ms, MODEL as c_MODEL, model2str, CAMERA_STATE as c_CAMERA_STATE, String, DeviceProperties as c_DeviceProperties, Vector2, Vector3, Vector4, Matrix3f as c_Matrix3f, Matrix4f as c_Matrix4f, UNIT as c_UNIT, COORDINATE_SYSTEM as c_COORDINATE_SYSTEM, RESOLUTION as c_RESOLUTION, CAMERA_SETTINGS as c_CAMERA_SETTINGS, SELF_CALIBRATION_STATE as c_SELF_CALIBRATION_STATE, DEPTH_MODE as c_DEPTH_MODE, SENSING_MODE as c_SENSING_MODE, MEASURE as c_MEASURE, VIEW as c_VIEW, TIME_REFERENCE as c_TIME_REFERENCE, DEPTH_FORMAT as c_DEPTH_FORMAT, POINT_CLOUD_FORMAT as c_POINT_CLOUD_FORMAT, TRACKING_STATE as c_TRACKING_STATE, AREA_EXPORT_STATE as c_AREA_EXPORT_STATE, REFERENCE_FRAME as c_REFERENCE_FRAME, SPATIAL_MAPPING_STATE as c_SPATIAL_MAPPING_STATE, SVO_COMPRESSION_MODE as c_SVO_COMPRESSION_MODE, RecordingState, cameraResolution, resolution2str, statusCode2str, str2mode, depthMode2str, sensingMode2str, unit2str, str2unit, trackingState2str, spatialMappingState2str, getCurrentTimeStamp, Resolution as c_Resolution, CameraParameters as c_CameraParameters, CalibrationParameters as c_CalibrationParameters, CameraInformation as c_CameraInformation, MEM as c_MEM, COPY_TYPE as c_COPY_TYPE, MAT_TYPE as c_MAT_TYPE, Mat as c_Mat, Rotation as c_Rotation, Translation as c_Translation, Orientation as c_Orientation, Transform as c_Transform, uchar1, uchar2, uchar3, uchar4, float1, float2, float3, float4, matResolution, setToUchar1, setToUchar2, setToUchar3, setToUchar4, setToFloat1, setToFloat2, setToFloat3, setToFloat4, setValueUchar1, setValueUchar2, setValueUchar3, setValueUchar4, setValueFloat1, setValueFloat2, setValueFloat3, setValueFloat4, getValueUchar1, getValueUchar2, getValueUchar3, getValueUchar4, getValueFloat1, getValueFloat2, getValueFloat3, getValueFloat4, getPointerUchar1, getPointerUchar2, getPointerUchar3, getPointerUchar4, getPointerFloat1, getPointerFloat2, getPointerFloat3, getPointerFloat4, uint, MESH_FILE_FORMAT as c_MESH_FILE_FORMAT, MESH_TEXTURE_FORMAT as c_MESH_TEXTURE_FORMAT, MESH_FILTER as c_MESH_FILTER, PLANE_TYPE as c_PLANE_TYPE, MeshFilterParameters as c_MeshFilterParameters, Texture as c_Texture, Chunk as c_Chunk, PointCloudChunk as c_PointCloudChunk, FusedPointCloud as c_FusedPointCloud, Mesh as c_Mesh, Plane as c_Plane, CUctx_st, CUcontext, MAPPING_RESOLUTION as c_MAPPING_RESOLUTION, MAPPING_RANGE as c_MAPPING_RANGE, SPATIAL_MAP_TYPE as c_SPATIAL_MAP_TYPE, InputType as c_InputType, InitParameters as c_InitParameters, RuntimeParameters as c_RuntimeParameters, TrackingParameters as c_TrackingParameters, SpatialMappingParameters as c_SpatialMappingParameters, Pose as c_Pose, IMUData as c_IMUData, Camera as c_Camera, StreamingParameters as c_StreamingParameters, STREAMING_CODEC as c_STREAMING_CODEC, StreamingProperties as c_StreamingProperties, FusedPointCloud as c_FusedPointCloud, SPATIAL_MAP_TYPE as c_SPATIAL_MAP_TYPE, PointCloudChunk as c_PointCloudChunk, saveDepthAs, savePointCloudAs, saveMatDepthAs, saveMatPointCloudAs
from cython.operator cimport dereference as deref
from libc.string cimport memcpy
from cpython cimport bool

import enum

import numpy as np
cimport numpy as np
from math import sqrt

class ERROR_CODE(enum.Enum):
    """
    Lists error codes in the ZED SDK.
    """

    SUCCESS = c_ERROR_CODE.SUCCESS
    ERROR_CODE_FAILURE = c_ERROR_CODE.ERROR_CODE_FAILURE
    ERROR_CODE_NO_GPU_COMPATIBLE = c_ERROR_CODE.ERROR_CODE_NO_GPU_COMPATIBLE
    ERROR_CODE_NOT_ENOUGH_GPUMEM = c_ERROR_CODE.ERROR_CODE_NOT_ENOUGH_GPUMEM
    ERROR_CODE_CAMERA_NOT_DETECTED = c_ERROR_CODE.ERROR_CODE_CAMERA_NOT_DETECTED
    ERROR_CODE_SENSOR_NOT_DETECTED = c_ERROR_CODE.ERROR_CODE_SENSOR_NOT_DETECTED
    ERROR_CODE_INVALID_RESOLUTION = c_ERROR_CODE.ERROR_CODE_INVALID_RESOLUTION
    ERROR_CODE_LOW_USB_BANDWIDTH = c_ERROR_CODE.ERROR_CODE_LOW_USB_BANDWIDTH
    ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE = c_ERROR_CODE.ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE
    ERROR_CODE_INVALID_SVO_FILE = c_ERROR_CODE.ERROR_CODE_INVALID_SVO_FILE
    ERROR_CODE_SVO_RECORDING_ERROR = c_ERROR_CODE.ERROR_CODE_SVO_RECORDING_ERROR
    ERROR_CODE_SVO_UNSUPPORTED_COMPRESSION = c_ERROR_CODE.ERROR_CODE_SVO_UNSUPPORTED_COMPRESSION
    ERROR_CODE_INVALID_COORDINATE_SYSTEM = c_ERROR_CODE.ERROR_CODE_INVALID_COORDINATE_SYSTEM
    ERROR_CODE_INVALID_FIRMWARE = c_ERROR_CODE.ERROR_CODE_INVALID_FIRMWARE
    ERROR_CODE_INVALID_FUNCTION_PARAMETERS = c_ERROR_CODE.ERROR_CODE_INVALID_FUNCTION_PARAMETERS
    ERROR_CODE_NOT_A_NEW_FRAME = c_ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME
    ERROR_CODE_CUDA_ERROR = c_ERROR_CODE.ERROR_CODE_CUDA_ERROR
    ERROR_CODE_CAMERA_NOT_INITIALIZED = c_ERROR_CODE.ERROR_CODE_CAMERA_NOT_INITIALIZED
    ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE = c_ERROR_CODE.ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE
    ERROR_CODE_INVALID_FUNCTION_CALL = c_ERROR_CODE.ERROR_CODE_INVALID_FUNCTION_CALL
    ERROR_CODE_CORRUPTED_SDK_INSTALLATION = c_ERROR_CODE.ERROR_CODE_CORRUPTED_SDK_INSTALLATION
    ERROR_CODE_INCOMPATIBLE_SDK_VERSION = c_ERROR_CODE.ERROR_CODE_INCOMPATIBLE_SDK_VERSION
    ERROR_CODE_INVALID_AREA_FILE = c_ERROR_CODE.ERROR_CODE_INVALID_AREA_FILE
    ERROR_CODE_INCOMPATIBLE_AREA_FILE = c_ERROR_CODE.ERROR_CODE_INCOMPATIBLE_AREA_FILE
    ERROR_CODE_CAMERA_FAILED_TO_SETUP = c_ERROR_CODE.ERROR_CODE_CAMERA_FAILED_TO_SETUP
    ERROR_CODE_CAMERA_DETECTION_ISSUE = c_ERROR_CODE.ERROR_CODE_CAMERA_DETECTION_ISSUE
    ERROR_CODE_CAMERA_ALREADY_IN_USE = c_ERROR_CODE.ERROR_CODE_CAMERA_ALREADY_IN_USE
    ERROR_CODE_NO_GPU_DETECTED = c_ERROR_CODE.ERROR_CODE_NO_GPU_DETECTED
    ERROR_CODE_PLANE_NOT_FOUND = c_ERROR_CODE.ERROR_CODE_PLANE_NOT_FOUND
    ERROR_CODE_LAST = c_ERROR_CODE.ERROR_CODE_LAST

    def __str__(self):
        return to_str(toString(<c_ERROR_CODE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_ERROR_CODE>self.value)).decode()

class MODEL(enum.Enum):
    MODEL_ZED = c_MODEL.MODEL_ZED
    MODEL_ZED_M = c_MODEL.MODEL_ZED_M
    MODEL_LAST = c_MODEL.MODEL_LAST

    def __str__(self):
        return to_str(toString(<c_MODEL>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_MODEL>self.value)).decode()

class CAMERA_STATE(enum.Enum):
    CAMERA_STATE_AVAILABLE = c_CAMERA_STATE.CAMERA_STATE_AVAILABLE
    CAMERA_STATE_NOT_AVAILABLE = c_CAMERA_STATE.CAMERA_STATE_NOT_AVAILABLE
    CAMERA_STATE_LAST = c_CAMERA_STATE.CAMERA_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_CAMERA_STATE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_CAMERA_STATE>self.value)).decode()

def c_sleep_ms(int time):
    """
    Tells the program to wait for x ms.

    **Parameters**
        - **time**    : the number of ms to wait.
    """
    sleep_ms(time)

cdef class DeviceProperties:
    """
    Properties of a camera.

    .. note::
        A camera_model MODEL_ZED_M with an id '-1' can be due to an inverted USB-C cable.

    .. warning::
        Experimental on Windows. 
    """
    cdef c_DeviceProperties c_device_properties

    def __cinit__(self):
        self.c_device_properties = c_DeviceProperties()

    @property
    def camera_state(self):
        """
        the camera state
        """
        return self.c_device_properties.camera_state
    @camera_state.setter
    def camera_state(self, camera_state):
        if isinstance(camera_state, CAMERA_STATE):
            self.c_device_properties.camera_state = (<c_CAMERA_STATE> camera_state.value)
        elif isinstance(camera_state, int):
            self.c_device_properties.camera_state = (<c_CAMERA_STATE> camera_state)
        else:
            raise TypeError("Argument is not of CAMERA_STATE type.")

    @property
    def id(self):
        """
        the camera id (Notice that only the camera with id '0' can be used on Windows)
        """
        return self.c_device_properties.id
    @id.setter
    def id(self, id):
        self.c_device_properties.id = id

    @property
    def path(self):
        """
        the camera system path
        """
        if not self.c_device_properties.path.empty():
            return self.c_device_properties.path.get().decode()
        else:
            return ""
    @path.setter
    def path(self, str path):
        path_ = path.encode()
        self.c_device_properties.path = (String(<char*> path_))

    @property
    def camera_model(self):
        """
        the camera model
        """
        return self.c_device_properties.camera_model
    @camera_model.setter
    def camera_model(self, camera_model):
        if isinstance(camera_model, MODEL):
            self.c_device_properties.camera_model = (<c_MODEL> camera_model.value)
        elif isinstance(camera_model, int):
            self.c_device_properties.camera_model = (<c_MODEL> camera_model)
        else:
            raise TypeError("Argument is not of MODEL type.")

    @property
    def serial_number(self):
        """
        the camera serial number

        Not provided for Windows 
        """
        return self.c_device_properties.serial_number
    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_device_properties.serial_number = serial_number

    def __str__(self):
        return to_str(toString(self.c_device_properties)).decode()

    def __repr__(self):
        return to_str(toString(self.c_device_properties)).decode()


cdef class Matrix3f:
    """
    Represents a generic three-dimensional matrix.

    It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on. You can access the data with the 'r' ptr or by element attribute.
    """
    cdef c_Matrix3f mat
    def __cinit__(self):
        self.mat = c_Matrix3f()

    def init_matrix(self, Matrix3f matrix):
        """
        Creates a :class:`~pyzed.sl.Matrix3f` from another :class:`~pyzed.sl.Matrix3f` (deep copy)

        **Parameters**
            - **matrix** : the :class:`~pyzed.sl.Matrix3f` to copy
        """
        self.mat = c_Matrix3f(matrix.mat)

    def inverse(self):
        """
        Inverses the matrix.
        """
        self.mat.inverse()

    def inverse_mat(self, Matrix3f rotation):
        """
        Inverses the :class:`~pyzed.sl.Matrix3f` passed as a parameter.

        **Parameters**
            - **rotation** : the :class:`~pyzed.sl.Matrix3f` to inverse

        **Returns**
            - the inversed :class:`~pyzed.sl.Matrix3f`
        """
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        """
        Sets the RotationArray to its transpose.
        """
        self.mat.transpose()

    def transpose(self, Matrix3f rotation):
        """
        Returns the transpose of a :class:`~pyzed.sl.Matrix3f`

        **Parameters**
            - **rotation** : the :class:`~pyzed.sl.Matrix3f` to compute the transpose from.

        **Returns**
            - the transposed :class:`~pyzed.sl.Matrix3f`
        """
        rotation.mat.transpose(rotation.mat)
        return rotation

    def set_identity(self):
        """
        Sets the :class:`~pyzed.sl.Matrix3f` to identity.

        **Returns**
            - returns itself
        """
        self.mat.setIdentity()
        return self

    def identity(self):
        """
        Creates an identity :class:`~pyzed.sl.Matrix3f`

        **Returns**
            - A :class:`~pyzed.sl.Matrix3f` set to identity
        """
        new_mat = Matrix3f()
        return new_mat.set_identity()

    def set_zeros(self):
        """
        Sets the :class:`~pyzed.sl.Matrix3f` to zero.
        """
        self.mat.setZeros()

    def zeros(self):
        """
        Creates a :class:`~pyzed.sl.Matrix3f` filled with zeros.

        **Returns**
            - A :class:`~pyzed.sl.Matrix3f` filled with zeros
        """
        self.mat.zeros()
        return self

    def get_infos(self):
        """
        Returns the components of the :class:`~pyzed.sl.Matrix3f` in a string.

        **Returns**
            - A string containing the components of the current :class:`~pyzed.sl.Matrix3f`
        """
        return to_str(self.mat.getInfos()).decode()

    @property
    def matrix_name(self):
        """
        Name of the matrix (optional).
        """
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, str name):
        self.mat.matrix_name.set(name.encode()) 

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
        matrix = Matrix3f()
        if isinstance(other, Matrix3f):
            matrix.r = (self.r * other.r).reshape(self.nbElem)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.r = (other * self.r).reshape(self.nbElem)
        else:
            raise TypeError("Argument must be Matrix3f or scalar type.")
        return matrix

    def __richcmp__(Matrix3f left, Matrix3f right, int op):
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


cdef class Matrix4f:
    """
    Represents a generic fourth-dimensional matrix.

    It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on. 
    """
    cdef c_Matrix4f mat
    def __cinit__(self):
        self.mat = c_Matrix4f()

    def init_matrix(self, Matrix4f matrix):
        """
        Creates a :class:`~pyzed.sl.Matrix4f` from another :class:`~pyzed.sl.Matrix4f` (deep copy)

        **Parameters**
            - **matrix** : the :class:`~pyzed.sl.Matrix3f` to copy
        """
        self.mat = c_Matrix4f(matrix.mat)

    def inverse(self):
        """
        Inverses the matrix.
        """
        return ERROR_CODE(self.mat.inverse())

    def inverse_mat(self, Matrix4f rotation):
        """
        Inverses the :class:`~pyzed.sl.Matrix4f` passed as a parameter.

        **Parameters**
            - **rotation** : the :class:`~pyzed.sl.Matrix4f` to inverse

        **Returns**
            - the inversed :class:`~pyzed.sl.Matrix4f`
        """
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        """
        Sets the RotationArray to its transpose.
        """
        self.mat.transpose()

    def transpose(self, Matrix4f rotation):
        """
        Returns the transpose of a :class:`~pyzed.sl.Matrix4f`

        **Parameters**
            - **rotation** : the :class:`~pyzed.sl.Matrix4f` to compute the transpose from.

        **Returns**
            - the transposed :class:`~pyzed.sl.Matrix4f`
        """
        rotation.mat.transpose(rotation.mat)
        return rotation

    def set_identity(self):
        """
        Sets the :class:`~pyzed.sl.Matrix4f` to identity.

        **Returns**
            - returns itself
        """
        self.mat.setIdentity()
        return self

    def identity(self):
        """
        Creates an identity :class:`~pyzed.sl.Matrix4f`

        **Returns**
            - A :class:`~pyzed.sl.Matrix4f` set to identity
        """
        new_mat = Matrix4f()
        return new_mat.set_identity()

    def set_zeros(self):
        """
        Sets the :class:`~pyzed.sl.Matrix4f` to zero.
        """
        self.mat.setZeros()

    def zeros(self):
        """
        Creates a :class:`~pyzed.sl.Matrix4f` filled with zeros.

        **Returns**
            - A :class:`~pyzed.sl.Matrix4f` filled with zeros
        """
        self.mat.zeros()
        return self

    def get_infos(self):
        """
        Returns the components of the :class:`~pyzed.sl.Matrix4f` in a string.

        **Returns**
            - A string containing the components of the current :class:`~pyzed.sl.Matrix4f`
        """
        return to_str(self.mat.getInfos()).decode()

    def set_sub_matrix3f(self, Matrix3f input, row=0, column=0):
        """
        Sets a 3x3 Matrix inside the :class:`~pyzed.sl.Matrix4f`.

        .. note::
            Can be used to set the rotation matrix when the matrix4f is a pose or an isometric matrix. 

        **Parameters**
            - **input** : sub matrix to put inside the :class:`~pyzed.sl.Matrix4f`
            - **row** : index of the row to start the 3x3 block. Must be 0 or 1.
            - **column** : index of the column to start the 3x3 block. Must be 0 or 1.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.
        """
        if row != 0 and row != 1 or column != 0 and column != 1:
            raise TypeError("Arguments row and column must be 0 or 1.")
        else:
            return ERROR_CODE(self.mat.setSubMatrix3f(input.mat, row, column))

    def set_sub_vector3f(self, float input0, float input1, float input2, column=3):
        """
        Sets a 3x1 Vector inside the :class:`~pyzed.sl.Matrix4f` at the specified column index.

        .. note::
            Can be used to set the Translation/Position matrix when the matrix4f is a pose or an isometry. 

        **Parameters**
            - **input** : sub vector to put inside the :class:`~pyzed.sl.Matrix4f`
            - **column** : index of the column to start the 3x3 block. By default, it is the last column (translation for a :class:`~pyzed.sl.Pose`).

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.
        """
        return ERROR_CODE(self.mat.setSubVector3f(Vector3[float](input0, input1, input2), column))

    def set_sub_vector4f(self, float input0, float input1, float input2, float input3, column=3):
        """
        Sets a 4x1 Vector inside the :class:`~pyzed.sl.Matrix4f` at the specified column index.

        **Parameters**
            - **input** : sub vector to put inside the :class:`~pyzed.sl.Matrix4f`
            - **column** : index of the column to start the 3x3 block. By default, it is the last column (translation for a :class:`~pyzed.sl.Pose`).

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.
        """
        return ERROR_CODE(self.mat.setSubVector4f(Vector4[float](input0, input1, input2, input3), column))

    @property
    def nbElem(self):
        return self.mat.nbElem

    @property
    def matrix_name(self):
        """
        Returns the components of the :class:`~pyzed.sl.Matrix4f` in a string.
        """
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, str name):
        self.mat.matrix_name.set(name.encode())

    @property
    def m(self):
        """
        Access to the content of the :class:`~pyzed.sl.Matrix4f` as a numpy array or list.
        """
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
        matrix = Matrix4f()
        if isinstance(other, Matrix4f):
            matrix.m = (self.m * other.m).reshape(self.nbElem)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.m = (other * self.m).reshape(self.nbElem)
        else:
            raise TypeError("Argument must be Matrix4f or scalar type.")
        return matrix

    def __richcmp__(Matrix4f left, Matrix4f right, int op):
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


class RESOLUTION(enum.Enum):
    """
    Represents the available resolution list.

    .. note::
        The VGA resolution does respect the 640*480 standard to better fit the camera sensor (672*376 is used).

    .. warning::
        NVIDIA Jetson X1 only supports RESOLUTION_HD1080@15, RESOLUTION_HD720@30/15, and RESOLUTION_VGA@60/30/15.

    +----------------------------------------------------------------------+
    |Enumerators                                                           |
    +=================+====================================================+
    |RESOLUTION_HD2K  |2208*1242, available framerates: 15 fps.            |
    +-----------------+----------------------------------------------------+
    |RESOLUTION_HD1080|1920*1080, available framerates: 15, 30 fps.        |
    +-----------------+----------------------------------------------------+
    |RESOLUTION_HD720 |1280*720, available framerates: 15, 30, 60 fps.     |
    +-----------------+----------------------------------------------------+
    |RESOLUTION_VGA   |672*376, available framerates: 15, 30, 60, 100 fps. |
    +-----------------+----------------------------------------------------+
    |RESOLUTION_LAST  |                                                    |
    +-----------------+----------------------------------------------------+
    """
    RESOLUTION_HD2K = c_RESOLUTION.RESOLUTION_HD2K
    RESOLUTION_HD1080 = c_RESOLUTION.RESOLUTION_HD1080
    RESOLUTION_HD720 = c_RESOLUTION.RESOLUTION_HD720
    RESOLUTION_VGA  = c_RESOLUTION.RESOLUTION_VGA
    RESOLUTION_LAST = c_RESOLUTION.RESOLUTION_LAST

    def __str__(self):
        return (<bytes> resolution2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> resolution2str(self.value)).decode()


class CAMERA_SETTINGS(enum.Enum):
    """
    Lists available camera settings for the ZED camera (contrast, hue, saturation, gain...).

    .. warning::
        - CAMERA_SETTINGS_GAIN and CAMERA_SETTINGS_EXPOSURE are linked in auto/default mode (see sl::Camera::setCameraSettings). Each enum defines one of those settings. 

    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                                        |
    +=================================+=================================================================================================================================================+
    |CAMERA_SETTINGS_BRIGHTNESS       |Defines the brightness control. Affected value should be between 0 and 8.                                                                        |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_CONTRAST         |Defines the contrast control. Affected value should be between 0 and 8.                                                                          |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_HUE              |Defines the hue control. Affected value should be between 0 and 11.                                                                              |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_SATURATION       |Defines the saturation control. Affected value should be between 0 and 8.                                                                        |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_GAIN             |Defines the gain control. Affected value should be between 0 and 100 for manual control.                                                         |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_EXPOSURE         |Defines the exposure control. Affected value should be between 0 and 100 for manual control.                                                     |
    |                                 |                                                                                                                                                 |
    |                                 |The exposition is mapped linearly in a percentage of the following max values. Special case for the setExposure(0) that corresponds to 0.17072ms.|
    |                                 |                                                                                                                                                 |
    |                                 |The conversion to milliseconds depends on the framerate:                                                                                         |
    |                                 |                                                                                                                                                 |
    |                                 |    - 15fps setExposure(100) -> 19.97ms                                                                                                          |
    |                                 |    - 30fps setExposure(100) -> 19.97ms                                                                                                          |
    |                                 |    - 60fps setExposure(100) -> 10.84072ms                                                                                                       |
    |                                 |    - 100fps setExposure(100) -> 10.106624ms                                                                                                     |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_WHITEBALANCE     |Defines the color temperature control. Affected value should be between 2800 and 6500 with a step of 100.                                        |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_AUTO_WHITEBALANCE|Defines the status of white balance (automatic or manual). A value of 0 disable the AWB, while 1 activate it.                                    |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    |CAMERA_SETTINGS_LAST             |                                                                                                                                                 |
    +---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------+
    """
    CAMERA_SETTINGS_BRIGHTNESS = c_CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
    CAMERA_SETTINGS_CONTRAST = c_CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST
    CAMERA_SETTINGS_HUE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_HUE
    CAMERA_SETTINGS_SATURATION = c_CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION
    CAMERA_SETTINGS_GAIN = c_CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN
    CAMERA_SETTINGS_EXPOSURE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE
    CAMERA_SETTINGS_WHITEBALANCE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE
    CAMERA_SETTINGS_AUTO_WHITEBALANCE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_AUTO_WHITEBALANCE
    CAMERA_SETTINGS_LED_STATUS = c_CAMERA_SETTINGS.CAMERA_SETTINGS_LED_STATUS
    CAMERA_SETTINGS_LAST = c_CAMERA_SETTINGS.CAMERA_SETTINGS_LAST


class SELF_CALIBRATION_STATE(enum.Enum):
    """
    Status for asynchrnous self-calibration.

    See also
        :meth:`~pyzed.sl.Camera.open()`

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                           |
    +==================================+===================================================================================================================================+
    |SELF_CALIBRATION_STATE_NOT_STARTED|Self calibration has not run yet (no :meth:`~pyzed.sl.Camera.open()` or :meth:`~pyzed.sl.Camera.reset_self_calibration()` called). |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
    |SELF_CALIBRATION_STATE_RUNNING    |Self calibration is currently running.                                                                                             |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
    |SELF_CALIBRATION_STATE_FAILED     |Self calibration has finished running but did not manage to get accurate values. Old parameters are taken instead.                 |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
    |SELF_CALIBRATION_STATE_SUCCESS    |Self calibration has finished running and did manage to get accurate values. New parameters are set.                               |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
    |SELF_CALIBRATION_STATE_LAST       |                                                                                                                                   |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+ 
    """
    SELF_CALIBRATION_STATE_NOT_STARTED = c_SELF_CALIBRATION_STATE.SELF_CALIBRATION_STATE_NOT_STARTED
    SELF_CALIBRATION_STATE_RUNNING = c_SELF_CALIBRATION_STATE.SELF_CALIBRATION_STATE_RUNNING
    SELF_CALIBRATION_STATE_FAILED = c_SELF_CALIBRATION_STATE.SELF_CALIBRATION_STATE_FAILED
    SELF_CALIBRATION_STATE_SUCCESS = c_SELF_CALIBRATION_STATE.SELF_CALIBRATION_STATE_SUCCESS
    SELF_CALIBRATION_STATE_LAST = c_SELF_CALIBRATION_STATE.SELF_CALIBRATION_STATE_LAST

    def __str__(self):
        return (<bytes> statusCode2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> statusCode2str(self.value)).decode()


class DEPTH_MODE(enum.Enum):
    """
    Lists available depth computation modes. 

    +-------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                          |
    +=======================+=============================================================================================================+
    |DEPTH_MODE_NONE        |This mode does not compute any depth map. Only rectified stereo images will be available.                    |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    |DEPTH_MODE_PERFORMANCE |Computation mode optimized for speed.                                                                        |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    |DEPTH_MODE_MEDIUM      |Balanced quality mode. Depth map is robust in any environment and requires medium resources for computation. |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    |DEPTH_MODE_QUALITY     |Computation mode designed for high quality results.                                                          |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    |DEPTH_MODE_ULTRA       |Computation mode favorising edges and sharpness. Requires more GPU memory and computation power.             |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    |DEPTH_MODE_LAST        |                                                                                                             |
    +-----------------------+-------------------------------------------------------------------------------------------------------------+
    """
    DEPTH_MODE_NONE = c_DEPTH_MODE.DEPTH_MODE_NONE
    DEPTH_MODE_PERFORMANCE = c_DEPTH_MODE.DEPTH_MODE_PERFORMANCE
    DEPTH_MODE_MEDIUM = c_DEPTH_MODE.DEPTH_MODE_MEDIUM
    DEPTH_MODE_QUALITY = c_DEPTH_MODE.DEPTH_MODE_QUALITY
    DEPTH_MODE_ULTRA = c_DEPTH_MODE.DEPTH_MODE_ULTRA
    DEPTH_MODE_LAST = c_DEPTH_MODE.DEPTH_MODE_LAST

    def __str__(self):
        return (<bytes> depthMode2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> depthMode2str(self.value)).decode()


class SENSING_MODE(enum.Enum):
    """
    Lists available depth sensing modes.

    +--------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                             |
    +======================+=================================================================================================================================+
    |SENSING_MODE_STANDARD |This mode outputs ZED standard depth map that preserves edges and depth accuracy.                                                |
    |                      |                                                                                                                                 |
    |                      |Applications example: Obstacle detection, Automated navigation, People detection, 3D reconstruction, measurements.               |
    +----------------------+---------------------------------------------------------------------------------------------------------------------------------+
    |SENSING_MODE_FILL     |This mode outputs a smooth and fully dense depth map. Applications example: AR/VR, Mixed-reality capture, Image post-processing. |
    +----------------------+---------------------------------------------------------------------------------------------------------------------------------+
    |SENSING_MODE_LAST     |                                                                                                                                 |
    +----------------------+---------------------------------------------------------------------------------------------------------------------------------+
    """
    SENSING_MODE_STANDARD = c_SENSING_MODE.SENSING_MODE_STANDARD
    SENSING_MODE_FILL = c_SENSING_MODE.SENSING_MODE_FILL
    SENSING_MODE_LAST = c_SENSING_MODE.SENSING_MODE_LAST

    def __str__(self):
        return (<bytes> sensingMode2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> sensingMode2str(self.value)).decode()


class UNIT(enum.Enum):
    UNIT_MILLIMETER = c_UNIT.UNIT_MILLIMETER
    UNIT_CENTIMETER = c_UNIT.UNIT_CENTIMETER
    UNIT_METER = c_UNIT.UNIT_METER
    UNIT_INCH = c_UNIT.UNIT_INCH
    UNIT_FOOT = c_UNIT.UNIT_FOOT
    UNIT_LAST = c_UNIT.UNIT_LAST

    def __str__(self):
        return (<bytes> unit2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> unit2str(self.value)).decode()


class COORDINATE_SYSTEM(enum.Enum):
    COORDINATE_SYSTEM_IMAGE = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_IMAGE
    COORDINATE_SYSTEM_LEFT_HANDED_Y_UP = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_LEFT_HANDED_Y_UP
    COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP
    COORDINATE_SYSTEM_LEFT_HANDED_Z_UP = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_LEFT_HANDED_Z_UP
    COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD
    COORDINATE_SYSTEM_LAST = c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_LAST

    def __str__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>self.value)).decode()

class MEASURE(enum.Enum):
    """
    Lists retrievable measures.

    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                                                                |
    +========================+==================================================================================================================================================================================+
    |MEASURE_DISPARITY       |Disparity map. Each pixel contains 1 float. MAT_TYPE_32F_C1.                                                                                                                      |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_DEPTH           |Depth map. Each pixel contains 1 float. MAT_TYPE_32F_C1.                                                                                                                          |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_CONFIDENCE      |Certainty/confidence of the depth map. Each pixel contains 1 float. MAT_TYPE_32F_C1.                                                                                              |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZ             |Point cloud. Each pixel contains 4 float (X, Y, Z, not used). MAT_TYPE_32F_C4.                                                                                                    |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZRGBA         |Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the RGBA color. MAT_TYPE_32F_C4.                  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZBGRA         |Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the BGRA color. MAT_TYPE_32F_C4.                  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZARGB         |Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the ARGB color. MAT_TYPE_32F_C4.                  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZABGR         |Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the ABGR color. MAT_TYPE_32F_C4.                  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_NORMALS         |Normals vector. Each pixel contains 4 float (X, Y, Z, 0). MAT_TYPE_32F_C4.                                                                                                        |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_DISPARITY_RIGHT |Disparity map for right sensor. Each pixel contains 1 float. MAT_TYPE_32F_C1.                                                                                                     |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_DEPTH_RIGHT     |Depth map for right sensor. Each pixel contains 1 float. MAT_TYPE_32F_C1.                                                                                                         |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZ_RIGHT       |Point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, not used). sl::MAT_TYPE_32F_C4.                                                                               |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZRGBA_RIGHT   |Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the RGBA color. MAT_TYPE_32F_C4. |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZBGRA_RIGHT   |Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the BGRA color. MAT_TYPE_32F_C4. |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZARGB_RIGHT   |Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the ARGB color. MAT_TYPE_32F_C4. |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_XYZABGR_RIGHT   |Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color need to be read as an usigned char[4] representing the ABGR color. MAT_TYPE_32F_C4. |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_NORMALS_RIGHT   |Normals vector for right view. Each pixel contains 4 float (X, Y, Z, 0). MAT_TYPE_32F_C4.                                                                                         |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |MEASURE_LAST            |                                                                                                                                                                                  |
    +------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """
    MEASURE_DISPARITY = c_MEASURE.MEASURE_DISPARITY
    MEASURE_DEPTH = c_MEASURE.MEASURE_DEPTH
    MEASURE_CONFIDENCE = c_MEASURE.MEASURE_CONFIDENCE
    MEASURE_XYZ = c_MEASURE.MEASURE_XYZ
    MEASURE_XYZRGBA = c_MEASURE.MEASURE_XYZRGBA
    MEASURE_XYZBGRA = c_MEASURE.MEASURE_XYZBGRA
    MEASURE_XYZARGB = c_MEASURE.MEASURE_XYZARGB
    MEASURE_XYZABGR = c_MEASURE.MEASURE_XYZABGR
    MEASURE_NORMALS = c_MEASURE.MEASURE_NORMALS
    MEASURE_DISPARITY_RIGHT = c_MEASURE.MEASURE_DISPARITY_RIGHT
    MEASURE_DEPTH_RIGHT = c_MEASURE.MEASURE_DEPTH_RIGHT
    MEASURE_XYZ_RIGHT = c_MEASURE.MEASURE_XYZ_RIGHT
    MEASURE_XYZRGBA_RIGHT = c_MEASURE.MEASURE_XYZRGBA_RIGHT
    MEASURE_XYZBGRA_RIGHT = c_MEASURE.MEASURE_XYZBGRA_RIGHT
    MEASURE_XYZARGB_RIGHT = c_MEASURE.MEASURE_XYZARGB_RIGHT
    MEASURE_XYZABGR_RIGHT = c_MEASURE.MEASURE_XYZABGR_RIGHT
    MEASURE_NORMALS_RIGHT = c_MEASURE.MEASURE_NORMALS_RIGHT
    MEASURE_LAST = c_MEASURE.MEASURE_LAST

    def __str__(self):
        return to_str(toString(<c_MEASURE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_MEASURE>self.value)).decode()

class VIEW(enum.Enum):
    """
    Lists available views.

    +--------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                             |
    +============================+===========================================================================================================================+
    |VIEW_LEFT                   |Left RGBA image. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                                             |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_RIGHT                  |Right RGBA image. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                                            |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_LEFT_GRAY              |Left GRAY image. Each pixel contains 1 usigned char. MAT_TYPE_8U_C1.                                                       |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_RIGHT_GRAY             |Right GRAY image. Each pixel contains 1 usigned char. MAT_TYPE_8U_C1.                                                      |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_LEFT_UNRECTIFIED       |Left RGBA unrectified image. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                                 |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_LEFT_UNRECTIFIED_GRAY  |Left GRAY unrectified image. Each pixel contains 1 usigned char. MAT_TYPE_8U_C1.                                           |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_RIGHT_UNRECTIFIED_GRAY |Right GRAY unrectified image. Each pixel contains 1 usigned char. MAT_TYPE_8U_C1.                                          |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_SIDE_BY_SIDE           |Left and right image (the image width is therefore doubled). Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4. |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_DEPTH                  |Color rendering of the depth. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                                |
    |                            |                                                                                                                           |
    |                            |Use MEASURE_DEPTH with :meth:`~pyzed.sl.Camera.retrieve_measure()` to get depth values.                                    |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_CONFIDENCE             |Color rendering of the depth confidence. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                     |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_NORMALS                |Color rendering of the normals. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.                              |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_DEPTH_RIGHT            |Color rendering of the right depth mapped on right sensor, MAT_TYPE_8U_C4.                                                 |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_NORMALS_RIGHT          |Color rendering of the normals mapped on right sensor. Each pixel contains 4 usigned char (R,G,B,A). MAT_TYPE_8U_C4.       |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    |VIEW_LAST                   |                                                                                                                           |
    +----------------------------+---------------------------------------------------------------------------------------------------------------------------+
    """
    VIEW_LEFT = c_VIEW.VIEW_LEFT
    VIEW_RIGHT = c_VIEW.VIEW_RIGHT
    VIEW_LEFT_GRAY = c_VIEW.VIEW_LEFT_GRAY
    VIEW_RIGHT_GRAY = c_VIEW.VIEW_RIGHT_GRAY
    VIEW_LEFT_UNRECTIFIED = c_VIEW.VIEW_LEFT_UNRECTIFIED
    VIEW_RIGHT_UNRECTIFIED = c_VIEW.VIEW_RIGHT_UNRECTIFIED
    VIEW_LEFT_UNRECTIFIED_GRAY = c_VIEW.VIEW_LEFT_UNRECTIFIED_GRAY
    VIEW_RIGHT_UNRECTIFIED_GRAY = c_VIEW.VIEW_RIGHT_UNRECTIFIED_GRAY
    VIEW_SIDE_BY_SIDE = c_VIEW.VIEW_SIDE_BY_SIDE
    VIEW_DEPTH = c_VIEW.VIEW_DEPTH
    VIEW_CONFIDENCE = c_VIEW.VIEW_CONFIDENCE
    VIEW_NORMALS = c_VIEW.VIEW_NORMALS
    VIEW_DEPTH_RIGHT = c_VIEW.VIEW_DEPTH_RIGHT
    VIEW_NORMALS_RIGHT = c_VIEW.VIEW_NORMALS_RIGHT
    VIEW_LAST = c_VIEW.VIEW_LAST

    def __str__(self):
        return to_str(toString(<c_VIEW>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_VIEW>self.value)).decode()

class DEPTH_FORMAT(enum.Enum):
    """
    Lists available file formats for saving depth maps.

    +-------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                |
    +==================+========================================================================================================================+
    |DEPTH_FORMAT_PNG  |PNG image format in 16bits. 32bits depth is mapped to 16bits color image to preserve the consistency of the data range. |
    +------------------+------------------------------------------------------------------------------------------------------------------------+
    |DEPTH_FORMAT_PFM  |stream of bytes, graphic image file format.                                                                             |
    +------------------+------------------------------------------------------------------------------------------------------------------------+
    |DEPTH_FORMAT_PGM  |gray-scale image format.                                                                                                |
    +------------------+------------------------------------------------------------------------------------------------------------------------+
    |DEPTH_FORMAT_LAST |                                                                                                                        |
    +------------------+------------------------------------------------------------------------------------------------------------------------+
    """
    DEPTH_FORMAT_PNG = c_DEPTH_FORMAT.DEPTH_FORMAT_PNG
    DEPTH_FORMAT_PFM = c_DEPTH_FORMAT.DEPTH_FORMAT_PFM
    DEPTH_FORMAT_PGM = c_DEPTH_FORMAT.DEPTH_FORMAT_PGM
    DEPTH_FORMAT_LAST = c_DEPTH_FORMAT.DEPTH_FORMAT_LAST

    def __str__(self):
        return to_str(toString(<c_DEPTH_FORMAT>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_DEPTH_FORMAT>self.value)).decode()

class POINT_CLOUD_FORMAT(enum.Enum):
    """
    Lists available file formats for saving point clouds. Stores the spatial coordinates (x,y,z) of each pixel and optionally its RGB color. 

    +------------------------------------------------------------------------------------------+
    |Enumerators                                                                               |
    +=============================+============================================================+
    |POINT_CLOUD_FORMAT_XYZ_ASCII |Generic point cloud file format, without color information. |
    +-----------------------------+------------------------------------------------------------+
    |POINT_CLOUD_FORMAT_PCD_ASCII |Point Cloud Data file, with color information.              |
    +-----------------------------+------------------------------------------------------------+
    |POINT_CLOUD_FORMAT_PLY_ASCII |PoLYgon file format, with color information.                |
    +-----------------------------+------------------------------------------------------------+
    |POINT_CLOUD_FORMAT_VTK_ASCII |Visualization ToolKit file, without color information.      |
    +-----------------------------+------------------------------------------------------------+
    |POINT_CLOUD_FORMAT_LAST      |                                                            |
    +-----------------------------+------------------------------------------------------------+
    """
    POINT_CLOUD_FORMAT_XYZ_ASCII = c_POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
    POINT_CLOUD_FORMAT_PCD_ASCII = c_POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_PCD_ASCII
    POINT_CLOUD_FORMAT_PLY_ASCII = c_POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_PLY_ASCII
    POINT_CLOUD_FORMAT_VTK_ASCII = c_POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_VTK_ASCII
    POINT_CLOUD_FORMAT_LAST = c_POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_LAST

    def __str__(self):
        return to_str(toString(<c_POINT_CLOUD_FORMAT>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_POINT_CLOUD_FORMAT>self.value)).decode()

class TRACKING_STATE(enum.Enum):
    """
    Lists the different states of positional tracking. 

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                                                         |
    +===========================+========================================================================================================================================================================+
    |TRACKING_STATE_SEARCHING   |The camera is searching for a previously known position to locate itself.                                                                                               |
    +---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |TRACKING_STATE_OK          |Positional tracking is working normally.                                                                                                                                |
    +---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |TRACKING_STATE_OFF         |Positional tracking is not enabled.                                                                                                                                     |
    +---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |TRACKING_STATE_FPS_TOO_LOW |Effective FPS is too low to give proper results for motion tracking. Consider using PERFORMANCES parameters (DEPTH_MODE_PERFORMANCE, low camera resolution (VGA,HD720)) |
    +---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |TRACKING_STATE_LAST        |                                                                                                                                                                        |
    +---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """
    TRACKING_STATE_SEARCHING = c_TRACKING_STATE.TRACKING_STATE_SEARCHING
    TRACKING_STATE_OK = c_TRACKING_STATE.TRACKING_STATE_OK
    TRACKING_STATE_OFF = c_TRACKING_STATE.TRACKING_STATE_OFF
    TRACKING_STATE_FPS_TOO_LOW = c_TRACKING_STATE.TRACKING_STATE_FPS_TOO_LOW
    TRACKING_STATE_LAST = c_TRACKING_STATE.TRACKING_STATE_LAST

    def __str__(self):
        return (<bytes> trackingState2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> trackingState2str(self.value)).decode()


class AREA_EXPORT_STATE(enum.Enum):
    """
    Lists the different states of spatial memory area export.

    +----------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                           |
    +==========================================+===========================================================================+
    |AREA_EXPORT_STATE_SUCCESS                 |The spatial memory file has been successfully created.                     |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_RUNNING                 |The spatial memory is currently written.                                   |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_NOT_STARTED             |The spatial memory file exportation has not been called.                   |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_FILE_EMPTY              |The spatial memory contains no data, the file is empty.                    |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_FILE_ERROR              |The spatial memory file has not been written because of a wrong file name. |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_SPATIAL_MEMORY_DISABLED |The spatial memory learning is disable, no file can be created.            |
    +------------------------------------------+---------------------------------------------------------------------------+
    |AREA_EXPORT_STATE_LAST                    |                                                                           |
    +------------------------------------------+---------------------------------------------------------------------------+
    """
    AREA_EXPORT_STATE_SUCCESS = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_SUCCESS
    AREA_EXPORT_STATE_RUNNING = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_RUNNING
    AREA_EXPORT_STATE_NOT_STARTED = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_NOT_STARTED
    AREA_EXPORT_STATE_FILE_EMPTY = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_FILE_EMPTY
    AREA_EXPORT_STATE_FILE_ERROR = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_FILE_ERROR
    AREA_EXPORT_STATE_SPATIAL_MEMORY_DISABLED = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_SPATIAL_MEMORY_DISABLED
    AREA_EXPORT_STATE_LAST = c_AREA_EXPORT_STATE.AREA_EXPORT_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_AREA_EXPORT_STATE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_AREA_EXPORT_STATE>self.value)).decode()

class REFERENCE_FRAME(enum.Enum):
    """
    Defines which type of position matrix is used to store camera path and pose. 

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                     |
    +=======================+========================================================================================================================================+
    |REFERENCE_FRAME_WORLD  |The transform of :class:`~pyzed.sl.Pose` will contains the motion with reference to the world frame (previously called PATH).           |
    +-----------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    |REFERENCE_FRAME_CAMERA |The transform of :class:`~pyzed.sl.Pose` will contains the motion with reference to the previous camera frame (previously called POSE). |
    +-----------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    |REFERENCE_FRAME_LAST   |                                                                                                                                        |
    +-----------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    """
    REFERENCE_FRAME_WORLD = c_REFERENCE_FRAME.REFERENCE_FRAME_WORLD
    REFERENCE_FRAME_CAMERA = c_REFERENCE_FRAME.REFERENCE_FRAME_CAMERA
    REFERENCE_FRAME_LAST = c_REFERENCE_FRAME.REFERENCE_FRAME_LAST

    def __str__(self):
        return to_str(toString(<c_REFERENCE_FRAME>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_REFERENCE_FRAME>self.value)).decode()

class TIME_REFERENCE(enum.Enum):
    """
    Lists specific and particular timestamps.

    +--------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                             |
    +=======================+================================================================================+
    |TIME_REFERENCE_IMAGE   |Defines the timestamp at the time the frame has been extracted from USB stream. |
    +-----------------------+--------------------------------------------------------------------------------+
    |TIME_REFERENCE_CURRENT |Defines the timestamp at the time of the function call.                         |
    +-----------------------+--------------------------------------------------------------------------------+
    |TIME_REFERENCE_LAST    |                                                                                |
    +-----------------------+--------------------------------------------------------------------------------+
    """
    TIME_REFERENCE_IMAGE = c_TIME_REFERENCE.TIME_REFERENCE_IMAGE
    TIME_REFERENCE_CURRENT = c_TIME_REFERENCE.TIME_REFERENCE_CURRENT
    TIME_REFERENCE_LAST = c_TIME_REFERENCE.TIME_REFERENCE_LAST

    def __str__(self):
        return to_str(toString(<c_TIME_REFERENCE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_TIME_REFERENCE>self.value)).decode()

class SPATIAL_MAPPING_STATE(enum.Enum):
    """
    Gives the spatial mapping state. 

    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                                                                                                      |
    +========================================+========================================================================================================================================================================================================+
    |SPATIAL_MAPPING_STATE_INITIALIZING      |The spatial mapping is initializing.                                                                                                                                                                    |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |SPATIAL_MAPPING_STATE_OK                |The depth and tracking data were correctly integrated in the fusion algorithm.                                                                                                                          |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY |The maximum memory dedicated to the scanning has been reach, the mesh will no longer be updated.                                                                                                        |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |SPATIAL_MAPPING_STATE_NOT_ENABLED       |:meth:`~pyzed.sl.enable_spatial_mapping()` wasn't called (or the scanning was stopped and not relaunched).                                                                                              |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |SPATIAL_MAPPING_STATE_FPS_TOO_LOW       |Effective FPS is too low to give proper results for spatial mapping. Consider using PERFORMANCES parameters (DEPTH_MODE_PERFORMANCE, low camera resolution (VGA,HD720), spatial mapping low resolution) |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |SPATIAL_MAPPING_STATE_LAST              |                                                                                                                                                                                                        |
    +----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """
    SPATIAL_MAPPING_STATE_INITIALIZING = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_INITIALIZING
    SPATIAL_MAPPING_STATE_OK = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_OK
    SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY
    SPATIAL_MAPPING_STATE_NOT_ENABLED = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_NOT_ENABLED
    SPATIAL_MAPPING_STATE_FPS_TOO_LOW = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_FPS_TOO_LOW
    SPATIAL_MAPPING_STATE_LAST = c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_LAST

    def __str__(self):
        return (<bytes> spatialMappingState2str(self.value)).decode()

    def __repr__(self):
        return (<bytes> spatialMappingState2str(self.value)).decode()


class SVO_COMPRESSION_MODE(enum.Enum):
    """
    Lists available compression modes for SVO recording.

    SVO_COMPRESSION_MODE_LOSSLESS is an improvement of previous lossless compression (used in ZED Explorer), even if size may be bigger, compression time is much faster.

    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                                                                      |
    +==============================+==================================================================================================================================+
    |SVO_COMPRESSION_MODE_RAW      |RAW images, no compression.                                                                                                       |
    |                              |                                                                                                                                  |
    |                              | **Deprecated**: This compresion is deprecated asit doesn't support timestamp and IMU data. Only playback is allowed in the mode. |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |SVO_COMPRESSION_MODE_LOSSLESS |PNG/ZSTD (lossless) CPU based compression : avg size = 42% (of RAW).                                                              |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |SVO_COMPRESSION_MODE_LOSSY    |JPEG (lossy) CPU based compression : avg size = 22% (of RAW). More compressed but can introduce compression artifacts.            |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |SVO_COMPRESSION_MODE_AVCHD    |H264(AVCHD) GPU based compression : avg size = 1% (of RAW). Requires a NVIDIA GPU                                                 |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |SVO_COMPRESSION_MODE_HEVC     |H265(HEVC) GPU based compression : avg size = 1% (of raw). Requires a NVIDIA GPU                                                  |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    |SVO_COMPRESSION_MODE_LAST     |                                                                                                                                  |
    +------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
    """
    SVO_COMPRESSION_MODE_RAW = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_RAW
    SVO_COMPRESSION_MODE_LOSSLESS = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS
    SVO_COMPRESSION_MODE_LOSSY = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSY
    SVO_COMPRESSION_MODE_AVCHD = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_AVCHD
    SVO_COMPRESSION_MODE_HEVC = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_HEVC
    SVO_COMPRESSION_MODE_LAST = c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LAST

    def __str__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>self.value)).decode()

def video_modes():
    return cameraResolution


def str_to_mode(str mode):
    return DEPTH_MODE(str2mode(mode.encode()))


def str_to_unit(str unit):
    return UNIT(str2unit(unit.encode()))

class MEM(enum.Enum):
    """
    List available memory type. 

    +-----------------------------------------+
    |Enumerators                              |
    +========+================================+
    |MEM_CPU |CPU Memory (Processor side).    |
    +--------+--------------------------------+
    |MEM_GPU |GPU Memory (Graphic card side). |
    +--------+--------------------------------+
    """
    MEM_CPU = c_MEM.MEM_CPU
    MEM_GPU = c_MEM.MEM_GPU


class COPY_TYPE(enum.Enum):
    """
    List available copy operation on :class:`~pyzed.sl.Mat`.

    +----------------------------------------------+
    |Enumerators                                   |
    +==================+===========================+
    |COPY_TYPE_CPU_CPU |copy data from CPU to CPU. |
    +------------------+---------------------------+
    |COPY_TYPE_CPU_GPU |copy data from CPU to GPU. |
    +------------------+---------------------------+
    |COPY_TYPE_GPU_GPU |copy data from GPU to GPU. |
    +------------------+---------------------------+
    |COPY_TYPE_GPU_CPU |copy data from GPU to CPU. |
    +------------------+---------------------------+
    """
    COPY_TYPE_CPU_CPU = c_COPY_TYPE.COPY_TYPE_CPU_CPU
    COPY_TYPE_CPU_GPU = c_COPY_TYPE.COPY_TYPE_CPU_GPU
    COPY_TYPE_GPU_GPU = c_COPY_TYPE.COPY_TYPE_GPU_GPU
    COPY_TYPE_GPU_CPU = c_COPY_TYPE.COPY_TYPE_GPU_CPU


class MAT_TYPE(enum.Enum):
    """
    List available :class:`~pyzed.sl.Mat` formats. 

    +-------------------------------------------+
    |Enumerators                                |
    +================+==========================+
    |MAT_TYPE_32F_C1 |float 1 channel.          |
    +----------------+--------------------------+
    |MAT_TYPE_32F_C2 |float 2 channels.         |
    +----------------+--------------------------+
    |MAT_TYPE_32F_C3 |float 3 channels.         |
    +----------------+--------------------------+
    |MAT_TYPE_32F_C4 |float 4 channels.         |
    +----------------+--------------------------+
    |MAT_TYPE_8U_C1  |unsigned char 1 channel.  |
    +----------------+--------------------------+
    |MAT_TYPE_8U_C2  |unsigned char 2 channels. |
    +----------------+--------------------------+
    |MAT_TYPE_8U_C3  |unsigned char 3 channels. |
    +----------------+--------------------------+
    |MAT_TYPE_8U_C4  |unsigned char 4 channels. |
    +----------------+--------------------------+
    """
    MAT_TYPE_32F_C1 = c_MAT_TYPE.MAT_TYPE_32F_C1
    MAT_TYPE_32F_C2 = c_MAT_TYPE.MAT_TYPE_32F_C2
    MAT_TYPE_32F_C3 = c_MAT_TYPE.MAT_TYPE_32F_C3
    MAT_TYPE_32F_C4 = c_MAT_TYPE.MAT_TYPE_32F_C4
    MAT_TYPE_8U_C1 = c_MAT_TYPE.MAT_TYPE_8U_C1
    MAT_TYPE_8U_C2 = c_MAT_TYPE.MAT_TYPE_8U_C2
    MAT_TYPE_8U_C3 = c_MAT_TYPE.MAT_TYPE_8U_C3
    MAT_TYPE_8U_C4 = c_MAT_TYPE.MAT_TYPE_8U_C4


def get_current_timestamp():
    return getCurrentTimeStamp()


cdef class Resolution:
    """
    Width and height of an array.
    """
    cdef size_t width
    cdef size_t height
    def __cinit__(self, width=0, height=0):
        self.width = width
        self.height = height

    def py_area(self):
        """
        Returns the area of the image.

        **Returns**
            - The number of pixels of the array.
        """
        return self.width * self.height

    @property
    def width(self):
        """
        array width in pixels
        """
        return self.width

    @width.setter
    def width(self, value):
        self.width = value

    @property
    def height(self):
        """
        array height in pixels
        """
        return self.height

    @height.setter
    def height(self, value):
        self.height = value

    def __richcmp__(Resolution left, Resolution right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height
        if op == 3:
            return left.width!=right.width or left.height!=right.height
        else:
            raise NotImplementedError()


cdef class CameraParameters:
    """
    Intrinsic parameters of a camera.

    Those information about the camera will be returned by :meth:`~pyzed.sl.Camera.get_camera_information()`.

    .. note::
        Similar to the :class:`~pyzed.sl.CalibrationParameters`, those parameters are taken from the settings file (SNXXX.conf) and are modified during the :meth:`~pyzed.sl.Camera.open` call when running a self-calibration). Those parameters given after :meth:`~pyzed.sl.Camera.open` call, represent the camera matrix corresponding to rectified or unrectified images.

        When filled with rectified parameters, fx,fy,cx,cy must be the same for Left and Right :class:`~pyzed.sl.Camera` once :meth:`~pyzed.sl.Camera.open` has been called. Since distortion is corrected during rectification, distortion should not be considered on rectified images. 
    """
    cdef c_CameraParameters camera_params
    @property
    def fx(self):
        """
        Focal length in pixels along x axis.
        """
        return self.camera_params.fx

    @fx.setter
    def fx(self, float fx_):
        self.camera_params.fx = fx_

    @property
    def fy(self):
        """
        Focal length in pixels along y axis.
        """
        return self.camera_params.fy

    @fy.setter
    def fy(self, float fy_):
        self.camera_params.fy = fy_

    @property
    def cx(self):
        """
        Optical center along x axis, defined in pixels (usually close to width/2).
        """
        return self.camera_params.cx

    @cx.setter
    def cx(self, float cx_):
        self.camera_params.cx = cx_

    @property
    def cy(self):
        """
        Optical center along y axis, defined in pixels (usually close to height/2).
        """
        return self.camera_params.cy

    @cy.setter
    def cy(self, float cy_):
        self.camera_params.cy = cy_

    @property
    def disto(self):

        """
        Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.
        """
        cdef np.ndarray arr = np.zeros(5)
        for i in range(5):
            arr[i] = self.camera_params.disto[i]
        return arr

    def set_disto(self, float value1, float value2, float value3, float value4, float value5):
        """
        Sets the elements of the disto array.

        **Parameters**
            - float **value1** : k1
            - float **value2** : k2
            - float **value3** : p1
            - float **value4** : p2
            - float **value5** : k3
        """
        self.camera_params.disto[0] = value1
        self.camera_params.disto[1] = value2
        self.camera_params.disto[2] = value3
        self.camera_params.disto[3] = value4
        self.camera_params.disto[4] = value5

    @property
    def v_fov(self):
        """
        Vertical field of view, in degrees.
        """
        return self.camera_params.v_fov

    @v_fov.setter
    def v_fov(self, float v_fov_):
        self.camera_params.v_fov = v_fov_

    @property
    def h_fov(self):
        """
        Horizontal field of view, in degrees.
        """
        return self.camera_params.h_fov

    @h_fov.setter
    def h_fov(self, float h_fov_):
        self.camera_params.h_fov = h_fov_

    @property
    def d_fov(self):
        """
        Diagonal field of view, in degrees.
        """
        return self.camera_params.d_fov

    @d_fov.setter
    def d_fov(self, float d_fov_):
        self.camera_params.d_fov = d_fov_

    @property
    def image_size(self):
        return Resolution(self.camera_params.image_size.width, self.camera_params.image_size.height)

    @image_size.setter
    def image_size(self, Resolution size_):
        self.camera_params.image_size.width = size_.width
        self.camera_params.image_size.height = size_.height

    def set_up(self, float fx_, float fy_, float cx_, float cy_) :
        """
        Setups the parameter of a camera.

        size in pixels of the images given by the camera.

        **Parameters**
            - **fx_** : horizontal focal length.
            - **fy_** : vertical focal length.
            - **cx_** : horizontal optical center.
            - **cy_** : vertical optical center.
        """
        self.camera_params.fx = fx_
        self.camera_params.fy = fy_
        self.camera_params.cx = cx_
        self.camera_params.cy = cy_

cdef class CalibrationParameters:
    """
    Intrinsic and Extrinsic parameters of the camera (translation and rotation).

    Those information about the camera will be returned by :meth:`~pyzed.sl.Camera.get_camera_information()`.

    .. note::
        The calibration/rectification process, called during :meth:`~pyzed.sl.Camera.open()`, is using the raw parameters defined in the SNXXX.conf file, where XXX is the ZED Serial Number.

        Those values may be adjusted or not by the Self-Calibration to get a proper image alignment. After :meth:`~pyzed.sl.Camera.open()` is done (with or without Self-Calibration activated) success, most of the stereo parameters (except Baseline of course) should be 0 or very close to 0.

    It means that images after rectification process (given by :meth:`~pyzed.sl.Camera.retrieve_image()`) are aligned as if they were taken by a "perfect" stereo camera, defined by the new :class:`~pyzed.sl.CalibrationParameters`. 
    """
    cdef c_CalibrationParameters calibration
    cdef CameraParameters py_left_cam
    cdef CameraParameters py_right_cam
    cdef Vector3[float] R
    cdef Vector3[float] T

    def __cinit__(self):
        self.py_left_cam = CameraParameters()
        self.py_right_cam = CameraParameters()

    def set(self):
        """
        Update the data of the :class:`~pyzed.sl.CalibrationParameters` with the :data:`~pyzed.sl.CalibrationParameters.py_left_cam` and :data:`~pyzed.sl.CalibrationParameters.py_right_cam` data. Already used in py_left_cam and py_right_cam setters.
        """
        self.py_left_cam.camera_params = self.calibration.left_cam
        self.py_right_cam.camera_params = self.calibration.right_cam
        self.R = self.calibration.R
        self.T = self.calibration.T

    @property
    def R(self):
        """
        :class:`~pyzed.sl.Rotation` on its own (using Rodrigues' transformation) of the right sensor. The left is considered as the reference. Defined as 'tilt', 'convergence' and 'roll'. Using a :class:`~pyzed.sl.Rotation`, you can use Rotation.set_rotation_vector() to convert into other representations.

        Returns a numpy array of float.
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.calibration.R[i]
        return arr

    def set_R(self, float value1, float value2, float value3) :
        """
        Set :data:`~pyzed.sl.CalibrationParameters.R`'s data.
        """
        self.calibration.R[0] = value1
        self.calibration.R[1] = value2
        self.calibration.R[3] = value3
        self.set()

    @property
    def T(self):
        """
        Translation between the two sensors. T[0] is the distance between the two cameras (baseline) in the :class:`~pyzed.sl.UNIT` chosen during :meth:`~pyzed.sl.Camera.open` (mm, cm, meters, inches...).

        Returns a numpy array of float.
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.calibration.T[i]
        return arr

    def set_T(self, float value1, float value2, float value3) :
        """
        Set :data:`~pyzed.sl.CalibrationParameters.T`'s data.
        """
        self.calibration.T[0] = value1
        self.calibration.T[1] = value2
        self.calibration.T[2] = value3
        self.set()

    @property
    def left_cam(self):
        """
        Intrinsic parameters of the left camera
        """
        return self.py_left_cam

    @left_cam.setter
    def left_cam(self, CameraParameters left_cam_) :
        self.calibration.left_cam = left_cam_.camera_params
        self.set()

    @property
    def right_cam(self):
        """
        Intrinsic parameters of the right camera
        """
        return self.py_right_cam

    @right_cam.setter
    def right_cam(self, CameraParameters right_cam_) :
        self.calibration.right_cam = right_cam_.camera_params
        self.set()


cdef class CameraInformation:
    """
    Structure containing information of a signle camera (serial number, model, calibration, etc.)

    Those information about the camera will be returned by :meth:`~pyzed.sl.Camera.get_camera_information()`.

    .. note::
        This object is meant to be used as a read-only container, editing any of its field won't impact the SDK.
    """
    cdef CalibrationParameters py_calib
    cdef CalibrationParameters py_calib_raw
    cdef unsigned int serial_number
    cdef unsigned int firmware_version
    cdef c_MODEL camera_model
    cdef Transform py_camera_imu_transform

    def __cinit__(self, Camera py_camera, Resolution resizer):
        res = c_Resolution(resizer.width, resizer.height)
        self.py_calib = CalibrationParameters()
        self.py_calib.calibration = py_camera.camera.getCameraInformation(res).calibration_parameters
        self.py_calib_raw = CalibrationParameters()
        self.py_calib_raw.calibration = py_camera.camera.getCameraInformation(res).calibration_parameters_raw
        self.py_calib.set()
        self.py_calib_raw.set()
        self.py_camera_imu_transform = Transform()
        self.py_camera_imu_transform.transform = py_camera.camera.getCameraInformation(res).camera_imu_transform
        self.serial_number = py_camera.camera.getCameraInformation(res).serial_number
        self.firmware_version = py_camera.camera.getCameraInformation(res).firmware_version
        self.camera_model = py_camera.camera.getCameraInformation(res).camera_model

    @property
    def camera_model(self):
        """
        The model of the camera (ZED or ZED-M).
        """
        return MODEL(self.camera_model)

    @property
    def calibration_parameters(self):
        """
        Intrinsic and Extrinsic stereo parameters for rectified/undistorded images (default).
        """
        return self.py_calib

    @property
    def calibration_parameters_raw(self):
        """
        Intrinsic and Extrinsic stereo parameters for original images (unrectified/distorded).
        """
        return self.py_calib_raw

    @property
    def camera_imu_transform(self):
        """
        IMU to Left camera transform matrix, that contains rotation and translation between IMU frame and camera frame. Note that this transform was applied to the fused quaternion provided in get_imu_ata() in v2.4 but not anymore starting from v2.5. See :meth:`~pyzed.sl.Camera.get_imu_data()` for more info.
        """
        return self.py_camera_imu_transform

    @property
    def serial_number(self):
        """
        The serial number of the camera.
        """
        return self.serial_number

    @property
    def firmware_version(self):
        """
        The internal firmware version of the camera.
        """
        return self.firmware_version


cdef class Mat:
    """
    The :class:`~pyzed.sl.Mat` class can handle multiple matrix format from 1 to 4 channels, with different value types (float or uchar), and can be stored CPU and/or GPU side.

    :class:`~pyzed.sl.Mat` is defined in a row-major order, it means that, for an image buffer, the entire first row is stored first, followed by the entire second row, and so on.

    The CPU and GPU buffer aren't automatically synchronized for performance reasons, you can use updateCPUfromGPU / updateGPUfromCPU to do it. If you are using the GPU side of the :class:`~pyzed.sl.Mat` object, you need to make sure to call free before destroying the :class:`~pyzed.sl.Camera` object. The destruction of the :class:`~pyzed.sl.Camera` object delete the CUDA context needed to free the GPU :class:`~pyzed.sl.Mat` memory.
    """
    cdef c_Mat mat
    def __cinit__(self, width=0, height=0, mat_type=MAT_TYPE.MAT_TYPE_32F_C1, memory_type=MEM.MEM_CPU):
        c_Mat(width, height, <c_MAT_TYPE>(mat_type.value), <c_MEM>(memory_type.value)).move(self.mat)

    def init_mat_type(self, width, height, mat_type, memory_type=MEM.MEM_CPU):
        """
        Inits a new :class:`~pyzed.sl.Mat`.

        This function directly allocates the requested memory. It calls :meth:`~pyzed.sl.Mat.alloc_size`.

        **Parameters**
            - **width**   : width of the matrix in pixels.
            - **height**  : height of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **memory_type** : defines where the buffer will be stored. (:data:`~pyzed.sl.MEM.MEM_CPU` and/or :data:`~pyzed.sl.MEM.MEM_GPU`).
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(mat_type.value), <c_MEM>(memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_cpu(self, width, height, mat_type, ptr, step, memory_type=MEM.MEM_CPU):
        """
        Inits a new :class:`~pyzed.sl.Mat` from an existing data pointer.

        This function doesn't allocate the memory.

        **Parameters**
            - **width**   : width of the matrix in pixels.
            - **height**  : height of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **ptr** : pointer to the data array. (CPU or GPU).
            - **step**    : step of the data array. (the Bytes size of one pixel row).
            - **memory_type** : defines where the buffer will be stored. (:data:`~pyzed.sl.MEM.MEM_CPU` and/or (:data:`~pyzed.sl.MEM.MEM_GPU`). 
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(mat_type.value), ptr.encode(), step, <c_MEM>(memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_cpu_gpu(self, width, height, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        """
        Inits a new :class:`~pyzed.sl.Mat` from two existing data pointers, CPU and GPU.

        This function doesn't allocate the memory.

        **Parameters**
            - **width**   : width of the matrix in pixels.
            - **height**  : height of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **ptr_cpu** : CPU pointer to the data array.
            - **step_cpu**    : step of the CPU data array (the Bytes size of one pixel row).
            - **ptr_gpu** : GPU pointer to the data array.
            - **step_gpu**    : step of the GPU data array (the Bytes size of one pixel row).
        """
        if isinstance(mat_type, MAT_TYPE):
             c_Mat(width, height, mat_type.value, ptr_cpu.encode(), step_cpu, ptr_gpu.encode(), step_gpu).move(self.mat)
        else:
            raise TypeError("Argument is not of MAT_TYPE type.")

    def init_mat_resolution(self, Resolution resolution, mat_type, memory_type):
        """
        Inits a new :class:`~pyzed.sl.Mat`.

        This function directly allocates the requested memory. It calls :meth:`~pyzed.sl.Mat.alloc_resolution`.

        **Parameters**
            - **resolution**  : the size of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **memory_type** : defines where the buffer will be stored (:data:`~pyzed.sl.MEM.MEM_CPU` and/or :data:`~pyzed.sl.MEM.MEM_GPU`).
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), mat_type.value, memory_type.value).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_resolution_cpu(self, Resolution resolution, mat_type, ptr, step, memory_type=MEM.MEM_CPU):
        """
        Inits a new :class:`~pyzed.sl.Mat` from an existing data pointer.

        This function doesn't allocate the memory.

        **Parameters**
            - **resolution**  : the size of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **ptr** : pointer to the data array. (CPU or GPU).
            - **step**    : step of the data array. (the Bytes size of one pixel row).
            - **memory_type** : defines where the buffer will be stored. (:data:`~pyzed.sl.MEM.MEM_CPU` and/or (:data:`~pyzed.sl.MEM.MEM_GPU`). 
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), mat_type.value, ptr.encode(), step, memory_type.value).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_resolution_cpu_gpu(self, Resolution resolution, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        """
        Inits a new :class:`~pyzed.sl.Mat` from two existing data pointers, CPU and GPU.

        This function doesn't allocate the memory.

        **Parameters**
            - **resolution**  : the size of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **ptr_cpu** : CPU pointer to the data array.
            - **step_cpu**    : step of the CPU data array (the Bytes size of one pixel row).
            - **ptr_gpu** : GPU pointer to the data array.
            - **step_gpu**    : step of the GPU data array (the Bytes size of one pixel row).
        """
        if isinstance(mat_type, MAT_TYPE):
             matResolution(c_Resolution(resolution.width, resolution.height), mat_type.value, ptr_cpu.encode(), step_cpu, ptr_gpu.encode(), step_gpu).move(self.mat)
        else:
            raise TypeError("Argument is not of MAT_TYPE type.")

    def init_mat(self, Mat matrix):
        """
        Inits a new :class:`~pyzed.sl.Mat` by copy (shallow copy).

        This function doesn't allocate the memory.

        **Parameters**
            - **mat** : the reference to the :class:`~pyzed.sl.Mat` to copy. 
        """
        c_Mat(matrix.mat).move(self.mat)

    def alloc_size(self, width, height, mat_type, memory_type=MEM.MEM_CPU):
        """
        Allocates the :class:`~pyzed.sl.Mat` memory.

        **Parameters**
            - **width**   : width of the matrix in pixels.
            - **height**  : height of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **memory_type** : defines where the buffer will be stored. (:data:`~pyzed.sl.MEM.MEM_CPU` and/or :data:`~pyzed.sl.MEM.MEM_GPU`).

        .. warning::
            It erases previously allocated memory. 
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(<size_t> width, <size_t> height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    def alloc_resolution(self, Resolution resolution, mat_type, memory_type=MEM.MEM_CPU):
        """
        Allocates the :class:`~pyzed.sl.Mat` memory.

        **Parameters**
            - **resolution**  : the size of the matrix in pixels.
            - **mat_type**    : the type of the matrix (:data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_32F_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`...).
            - **memory_type** : defines where the buffer will be stored. (:data:`~pyzed.sl.MEM.MEM_CPU` and/or :data:`~pyzed.sl.MEM.MEM_GPU`).

        .. warning::
            It erases previously allocated memory. 
        """
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(resolution.width, resolution.height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    def free(self, memory_type=None):
        """
        Free the owned memory.

        **Parameters**
            - **memory_type** : specify whether you want to free the :data:`~pyzed.sl.MEM.MEM_CPU` and/or :data:`~pyzed.sl.MEM.MEM_GPU` memory. If None, it frees both CPU and GPU memory.
        """
        if isinstance(memory_type, MEM):
            self.mat.free(memory_type.value)
        elif memory_type is None:
            self.mat.free(MEM.MEM_CPU or MEM.MEM_GPU)
        else:
            raise TypeError("Argument is not of MEM type.")

    def update_cpu_from_gpu(self):
        """
        Downloads data from DEVICE (GPU) to HOST (CPU), if possible.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. note::
            If no CPU or GPU memory are available for this :class:`~pyzed.sl.Mat`, some are directly allocated.
            If verbose sets, you have informations in case of failure. 
        """
        return ERROR_CODE(self.mat.updateCPUfromGPU())

    def update_gpu_from_cpu(self):
        """
        Downloads data from HOST (CPU) to DEVICE (GPU), if possible.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. note::
            If no CPU or GPU memory are available for this :class:`~pyzed.sl.Mat`, some are directly allocated.
            If verbose sets, you have informations in case of failure.
        """
        return ERROR_CODE(self.mat.updateGPUfromCPU())

    def copy_to(self, Mat dst=Mat(), cpy_type=COPY_TYPE.COPY_TYPE_CPU_CPU):
        """
        Copies data to another :class:`~pyzed.sl.Mat` (deep copy).

        **Parameters**
            - **dst** : the :class:`~pyzed.sl.Mat` where the data will be copied.
            - **cpy_type** : specify the memories that will be used for the copy.

        **Returns**
            - The **dst** :class:`~pyzed.sl.Mat`
        .. note::
            If the destination is not allocated or has a not a compatible :class:`~pyzed.sl.MAT_TYPE` or :class:`~pyzed.sl.Resolution`, current memory is freed and new memory is directly allocated. 
        """
        self.mat.copyTo(dst.mat, cpy_type.value)
        return dst

    def set_from(self, Mat src=Mat(), cpy_type=COPY_TYPE.COPY_TYPE_CPU_CPU):
        """
        Copies data from an other :class:`~pyzed.sl.Mat` (deep copy).

        **Parameters**
            - **src** : the :class:`~pyzed.sl.Mat` where the data will be copied from.
            - **cpy_type** : specify the memories that will be used for the update.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. note::
            If the current :class:`~pyzed.sl.Mat` is not allocated or has a not a compatible :class:`~pyzed.sl.MAT_TYPE` or :class:`~pyzed.sl.Resolution` with the source, current memory is freed and new memory is directly allocated. 
        """
        return ERROR_CODE(self.mat.setFrom(<const c_Mat>src.mat, cpy_type.value))

    def read(self, str filepath):
        """
        Reads an image from a file (only if :data:`~pyzed.sl.MEM.MEM_CPU` is available on the current :class:`~pyzed.sl.Mat`).

        Supported input files format are PNG and JPEG.

        **Parameters**
            - **filepath** : file path including the name and extension.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. note::
            Supported :class:`~pyzed.sl.MAT_TYPE` are : :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C3` and :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`
        """
        return ERROR_CODE(self.mat.read(filepath.encode()))

    def write(self, str filepath):
        """
        Writes the :class:`~pyzed.sl.Mat` (only if :data:`~pyzed.sl.MEM.MEM_CPU` is available) into a file as an image.

        Supported output files format are PNG and JPEG.

        **Parameters**
            - **filepath** : file path including the name and extension.

        **Returns**
            :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. note::
            Supported :class:`~pyzed.sl.MAT_TYPE` are : :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C1`, :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C3` and :data:`~pyzed.sl.MAT_TYPE.MAT_TYPE_8U_C4`
        """
        return ERROR_CODE(self.mat.write(filepath.encode()))

    def set_to(self, value, memory_type=MEM.MEM_CPU):
        """
        Fills the :class:`~pyzed.sl.Mat` with the given value.

        This function overwrite all the matrix.

        **Parameters**
            - **value**   : the value to be copied all over the matrix.
            - **memory_type** : defines which buffer to fill, CPU and/or GPU.
        """
        if self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C1:
            return ERROR_CODE(setToUchar1(self.mat, value, memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C2:
            return ERROR_CODE(setToUchar2(self.mat, Vector2[uchar1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C3:
            return ERROR_CODE(setToUchar3(self.mat, Vector3[uchar1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C4:
            return ERROR_CODE(setToUchar4(self.mat, Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C1:
            return ERROR_CODE(setToFloat1(self.mat, value, memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C2:
            return ERROR_CODE(setToFloat2(self.mat, Vector2[float1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C3:
            return ERROR_CODE(setToFloat3(self.mat, Vector3[float1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C4:
            return ERROR_CODE(setToFloat4(self.mat, Vector4[float1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))

    def set_value(self, x, y, value, memory_type=MEM.MEM_CPU):
        """
        Sets a value to a specific point in the matrix.

        **Parameters**
            - **x**   : specify the column.
            - **y**   : specify the row.
            - **value**   : the value to be set.
            - **memory_type** : defines which memory will be updated.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. warning::
            Not efficient for MEM_GPU, use it on sparse data.
        """
        if self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C1:
            return ERROR_CODE(setValueUchar1(self.mat, x, y, value, memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C2:
            return ERROR_CODE(setValueUchar2(self.mat, x, y, Vector2[uchar1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C3:
            return ERROR_CODE(setValueUchar3(self.mat, x, y, Vector3[uchar1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C4:
            return ERROR_CODE(setValueUchar4(self.mat, x, y, Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C1:
            return ERROR_CODE(setValueFloat1(self.mat, x, y, value, memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C2:
            return ERROR_CODE(setValueFloat2(self.mat, x, y, Vector2[float1](value[0], value[1]),
                                      memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C3:
            return ERROR_CODE(setValueFloat3(self.mat, x, y, Vector3[float1](value[0], value[1],
                                      value[2]), memory_type.value))
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C4:
            return ERROR_CODE(setValueFloat4(self.mat, x, y, Vector4[float1](value[0], value[1], value[2],
                                      value[3]), memory_type.value))

    def get_value(self, x, y, memory_type=MEM.MEM_CPU):
        """
        Returns the value of a specific point in the matrix.

        **Parameters**
            - **x**   : specify the column
            - **y**   : specify the row
            - **memory_type** : defines which memory should be read.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went well, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. warning::
            Not efficient for MEM_GPU, use it on sparse data.
        """
        cdef uchar1 value1u
        cdef Vector2[uchar1] value2u = Vector2[uchar1]()
        cdef Vector3[uchar1] value3u = Vector3[uchar1]()
        cdef Vector4[uchar1] value4u = Vector4[uchar1]()

        cdef float1 value1f
        cdef Vector2[float1] value2f = Vector2[float1]()
        cdef Vector3[float1] value3f = Vector3[float1]()
        cdef Vector4[float1] value4f = Vector4[float1]()

        if self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C1:
            status = getValueUchar1(self.mat, x, y, &value1u, memory_type.value)
            return ERROR_CODE(status), self.get_data()[x, y]
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C2:
            status = getValueUchar2(self.mat, x, y, &value2u, memory_type.value)
            return ERROR_CODE(status), np.array([value2u.ptr()[0], value2u.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C3:
            status = getValueUchar3(self.mat, x, y, &value3u, memory_type.value)
            return ERROR_CODE(status), np.array([value3u.ptr()[0], value3u.ptr()[1], value3u.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_8U_C4:
            status = getValueUchar4(self.mat, x, y, &value4u, memory_type.value)
            return ERROR_CODE(status), np.array([value4u.ptr()[0], value4u.ptr()[1], value4u.ptr()[2],
                                                         value4u.ptr()[3]])

        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C1:
            status = getValueFloat1(self.mat, x, y, &value1f, memory_type.value)
            return ERROR_CODE(status), self.get_data()[x, y]
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C2:
            status = getValueFloat2(self.mat, x, y, &value2f, memory_type.value)
            return ERROR_CODE(status), np.array([value2f.ptr()[0], value2f.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C3:
            status = getValueFloat3(self.mat, x, y, &value3f, memory_type.value)
            return ERROR_CODE(status), np.array([value3f.ptr()[0], value3f.ptr()[1], value3f.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.MAT_TYPE_32F_C4:
            status = getValueFloat4(self.mat, x, y, &value4f, memory_type.value)
            return ERROR_CODE(status), np.array([value4f.ptr()[0], value4f.ptr()[1], value4f.ptr()[2],
                                                         value4f.ptr()[3]])

    def get_width(self):
        """
        Returns the width of the matrix.

        **Returns**
            - The width of the matrix in pixels.
        """
        return self.mat.getWidth()

    def get_height(self):
        """
        Returns the height of the matrix.

        **Returns**
            - The height of the matrix in pixels.
        """
        return self.mat.getHeight()

    def get_resolution(self):
        """
        Returns the resolution of the matrix.

        **Returns**
            - The resolution of the matrix in pixels.
        """
        return Resolution(self.mat.getResolution().width, self.mat.getResolution().height)

    def get_channels(self):
        """
        Returns the number of values stored in one pixel.

        **Returns**
            - The number of values in a pixel. 
        """
        return self.mat.getChannels()

    def get_data_type(self):
        """
        Returns the format of the matrix.

        **Returns**
            - The format of the current :class:`~pyzed.sl.Mat`.
        """
        return MAT_TYPE(self.mat.getDataType())

    def get_memory_type(self):
        """
        Returns the format of the matrix.

        **Returns**
            - The format of the current :class:`~pyzed.sl.Mat`. 
        """
        return MEM(self.mat.getMemoryType())

    def get_data(self, memory_type=MEM.MEM_CPU):
        """
        Copies the data of the :class:`~pyzed.sl.Mat` in a numpy array.

        **Parameters**
            - **memory_type** : defines which memory should be read.

        **Returns**
            - A numpy array containing the :class:`~pyzed.sl.Mat` data.
        """
        shape = None
        if self.mat.getChannels() == 1:
            shape = (self.mat.getHeight(), self.mat.getWidth())
        else:
            shape = (self.mat.getHeight(), self.mat.getWidth(), self.mat.getChannels())

        cdef size_t size = 0
        dtype = None
        if self.mat.getDataType() in (c_MAT_TYPE.MAT_TYPE_8U_C1, c_MAT_TYPE.MAT_TYPE_8U_C2, c_MAT_TYPE.MAT_TYPE_8U_C3, c_MAT_TYPE.MAT_TYPE_8U_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()
            dtype = np.uint8
        elif self.mat.getDataType() in (c_MAT_TYPE.MAT_TYPE_32F_C1, c_MAT_TYPE.MAT_TYPE_32F_C2, c_MAT_TYPE.MAT_TYPE_32F_C3, c_MAT_TYPE.MAT_TYPE_32F_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()*sizeof(float)
            dtype = np.float32
        else:
            raise RuntimeError("Unknown Mat data_type value: {0}".format(self.mat.getDataType()))

        cdef np.ndarray arr = np.zeros(shape, dtype=dtype)
        if isinstance(memory_type, MEM):
            if self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_8U_C1:
                memcpy(<void*>arr.data, <void*>getPointerUchar1(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_8U_C2:
                memcpy(<void*>arr.data, <void*>getPointerUchar2(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_8U_C3:
                memcpy(<void*>arr.data, <void*>getPointerUchar3(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_8U_C4:
                memcpy(<void*>arr.data, <void*>getPointerUchar4(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_32F_C1:
                memcpy(<void*>arr.data, <void*>getPointerFloat1(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_32F_C2:
                memcpy(<void*>arr.data, <void*>getPointerFloat2(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_32F_C3:
                memcpy(<void*>arr.data, <void*>getPointerFloat3(self.mat, memory_type.value), size)
            elif self.mat.getDataType() == c_MAT_TYPE.MAT_TYPE_32F_C4:
                memcpy(<void*>arr.data, <void*>getPointerFloat4(self.mat, memory_type.value), size)
        else:
            raise TypeError("Argument is not of MEM type.")

        return arr

    def get_step_bytes(self, memory_type=MEM.MEM_CPU):
        """
        Returns the memory step in Bytes (the Bytes size of one pixel row).

        **Parameters**
            - **memory_type** : specify whether you want :data:`~pyzed.sl.MEM.MEM_CPU` or :data:`~pyzed.sl.MEM.MEM_GPU` step.

        **Returns**
            - The step in bytes of the specified memory.
        """
        if type(memory_type) == MEM:
            return self.mat.getStepBytes(memory_type.value)
        else:
            raise TypeError("Argument is not of MEM type.")

    def get_step(self, memory_type=MEM.MEM_CPU):
        """
        Returns the memory step in number of elements (the number of values in one pixel row).

        **Parameters**
            - **memory_type** : specify whether you want :data:`~pyzed.sl.MEM.MEM_CPU` or :data:`~pyzed.sl.MEM.MEM_GPU` step.

        **Returns**
            - The step in number of elements.
        """
        if type(memory_type) == MEM:
            return self.mat.getStep(memory_type.value)
        else:
            raise TypeError("Argument is not of MEM type.")

    def get_pixel_bytes(self):
        """
        Returns the size in bytes of one pixel.

        **Returns**
            - The size in bytes of a pixel. 
        """
        return self.mat.getPixelBytes()

    def get_width_bytes(self):
        """
        Returns the size in bytes of a row.

        **Returns**
            - The size in bytes of a row. 
        """
        return self.mat.getWidthBytes()

    def get_infos(self):
        """
        Returns the informations about the :class:`~pyzed.sl.Mat` into a string.

        **Returns**
            - A string containing the :class:`~pyzed.sl.Mat` informations.
        """
        return self.mat.getInfos().get().decode()

    def is_init(self):
        """
        Defines whether the :class:`~pyzed.sl.Mat` is initialized or not.

        **Returns**
            - True if current :class:`~pyzed.sl.Mat` has been allocated (by the constructor or therefore). 
        """
        return self.mat.isInit()

    def is_memory_owner(self):
        """
        Returns whether the :class:`~pyzed.sl.Mat` is the owner of the memory it access.

        If not, the memory won't be freed if the Mat is destroyed.

        **Returns**
            - True if the :class:`~pyzed.sl.Mat` is owning its memory, else false. 
        """
        return self.mat.isMemoryOwner()

    def clone(self, Mat py_mat):
        """
        Duplicates :class:`~pyzed.sl.Mat` by copy (deep copy).

        **Parameters**
            - **py_mat** : the reference to the :class:`~pyzed.sl.Mat` to copy. This function copies the data array(s), it mark the new :class:`~pyzed.sl.Mat` as the memory owner.
        """
        return ERROR_CODE(self.mat.clone(py_mat.mat))

    def move(self, Mat py_mat):
        """
        Moves Mat data to another :class:`~pyzed.sl.Mat`.

        This function gives the attribute of the current :class:`~pyzed.sl.Mat` to the specified one. (No copy).

        **Parameters**
            - **py_mat** : the reference to the :class:`~pyzed.sl.Mat` to move.

        .. note::
            : the current :class:`~pyzed.sl.Mat` is then no more usable since its loose its attributes. 
        """
        return ERROR_CODE(self.mat.move(py_mat.mat))

    @staticmethod
    def swap(self, Mat mat1, Mat mat2):
        """
        Swaps the content of the provided :class:`~pyzed.sl.Mat` (only swaps the pointers, no data copy).

        This function swaps the pointers of the given :class:`~pyzed.sl.Mat`.

        **Parameters**
            - **mat1**    : the first mat.
            - **mat2**    : the second mat.
        """
        self.mat.swap(mat1, mat2)

    @property
    def name(self):
        """
        The name of the :class:`~pyzed.sl.Mat` (optional)
        """
        if not self.mat.name.empty():
            return self.mat.name.get().decode()
        else:
            return ""

    @name.setter
    def name(self, str name_):
        self.mat.name.set(name_.encode())

    @property
    def verbose(self):
        return self.mat.verbose

    @verbose.setter
    def verbose(self, bool verbose_):
        self.mat.verbose = verbose_

    def __repr__(self):
        return self.get_infos()


cdef class Rotation(Matrix3f):
    """
    Designed to contain rotation data of the positional tracking. It inherits from the generic :class:`~pyzed.sl.Matrix3f`.
    """
    cdef c_Rotation rotation
    def __cinit__(self):
        self.rotation = c_Rotation()

    def init_rotation(self, Rotation rot):
        """
        Deep copy from another :class:`~pyzed.sl.Rotation`

        **Parameters**
            - **orient** : :class:`~pyzed.sl.Rotation` to be copied
        """
        self.rotation = c_Rotation(rot.rotation)
        self.mat = rot.mat

    def init_matrix(self, Matrix3f matrix):
        """
        Inits the :class:`~pyzed.sl.Rotation` from a :class:`~pyzed.sl.Matrix3f`.

        **Parameters**
            - **matrix** : :class:`~pyzed.sl.Matrix3f` to be used
        """
        self.rotation = c_Rotation(matrix.mat)
        self.mat = matrix.mat

    def init_orientation(self, Orientation orient):
        """
        Inits the :class:`~pyzed.sl.Rotation` from a :class:`~pyzed.sl.Orientation`.

        **Parameters**
            - **orient** : :class:`~pyzed.sl.Orientation` to be used
        """
        self.rotation = c_Rotation(orient.orientation)
        self.mat = c_Matrix3f(self.rotation.r)

    def init_angle_translation(self, float angle, Translation axis):
        """
        Inits the :class:`~pyzed.sl.Rotation` from an angle and an arbitrary 3D axis.

        **Parameters**
            - **angle** : 
            - **axis** : :class:`~pyzed.sl.Translation` axis to be used
        """
        self.rotation = c_Rotation(angle, axis.translation)
        self.mat = c_Matrix3f(self.rotation.r)

    def set_orientation(self, Orientation py_orientation):
        """
        Sets the :class:`~pyzed.sl.Rotation` from an :class:`~pyzed.sl.Orientation`.

        **Parameters**
            - **orientation** : the :class:`~pyzed.sl.Orientation` containing the rotation to set.
        """
        self.rotation.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        """
        Returns the :class:`~pyzed.sl.Orientation` corresponding to the current :class:`~pyzed.sl.Rotation`.

        **Returns**
            - The rotation of the current orientation. 
        """
        py_orientation = Orientation()
        py_orientation.orientation = self.rotation.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        """
        Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.

        **Returns**
            - The rotation vector (numpy array)
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.rotation.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        """
        Sets the :class:`~pyzed.sl.Rotation` from a rotation vector (using Rodrigues' transformation).

        **Parameters**
            - **input0** : First float value
            - **input1** : Second float value
            - **input2** : Third float value
        """
        self.rotation.setRotationVector(Vector3[float](input0, input1, input2))

    def get_euler_angles(self, radian=True):
        """
        Convert the :class:`~pyzed.sl.Rotation` as Euler angles.

        Parameters
            **radian**  : Define if the angle in is radian or degree

        **Returns**
            The Euler angles, as a numpy array representing the rotations arround the X, Y and Z axes.
        """
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.rotation.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    def set_euler_angles(self, float input0, float input1, float input2, bool radian=True):
        """
        Sets the :class:`~pyzed.sl.Rotation` from the Euler angles.

        **Parameters**
            - **input0** : Roll value
            - **input1** : Pitch value
            - **input2** : Yaw value
            - **radian**  : Define if the angle in is radian or degree
        """
        if isinstance(radian, bool):
            self.rotation.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")

cdef class Translation:
    """
    Designed to contain translation data of the positional tracking.

    :class:`~pyzed.sl.Translation` is a vector as [tx, ty, tz]. You can access the data with the :meth:`~pyzed.sl.Translation.get()` method that returns a numpy array.
    """
    cdef c_Translation translation
    def __cinit__(self):
        self.translation = c_Translation()

    def init_translation(self, Translation tr):
        """
        Deep copy from another :class:`~pyzed.sl.Translation`

        **Parameters**
            - **tr** : :class:`~pyzed.sl.Translation` to be copied
        """
        self.translation = c_Translation(tr.translation)

    def init_vector(self, float t1, float t2, float t3):
        """
        Inits :class:`~pyzed.sl.Translation` from float values.

        **Parameters**
            - **t1** : First float value
            - **t2** : Second float value
            - **t3** : Third float value
        """
        self.translation = c_Translation(t1, t2, t3)

    def normalize(self):
        """
        Normalizes the current translation.
        """
        self.translation.normalize()

    def normalize_translation(self, Translation tr):
        """
        Get the normalized version of a given :class:`~pyzed.sl.Translation`.

        **Parameters**
            - **tr**  : the :class:`~pyzed.sl.Translation` to be used.

        **Returns**
            - An other :class:`~pyzed.sl.Translation` object, which is equal to tr.normalize. 
        """
        py_translation = Translation()
        py_translation.translation = self.translation.normalize(tr.translation)
        return py_translation

    def size(self):
        return self.translation.size()

    def get(self):
        """
        Gets the :class:`~pyzed.sl.Translation` as a numpy array.

        **Returns**
            - A numpy array of float with the :class:`~pyzed.sl.Translation` values.
        """
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.translation(i)
        return arr

    def __mul__(Translation self, Orientation other):
        tr = Translation()
        tr.translation = self.translation * other.orientation
        return tr


cdef class Orientation:
    """
    Designed to contain orientation (quaternion) data of the positional tracking.

    :class:`~pyzed.sl.Orientation` is a vector defined as [ox, oy, oz, ow].
    """
    cdef c_Orientation orientation
    def __cinit__(self):
        self.orientation = c_Orientation()

    def init_orientation(self, Orientation orient):
        """
        Deep copy from another :class:`~pyzed.sl.Orientation`

        **Parameters**
            - **orient** : :class:`~pyzed.sl.Orientation` to be copied
        """
        self.orientation = c_Orientation(orient.orientation)

    def init_vector(self, float v0, float v1, float v2, float v3):
        """
        Inits :class:`~pyzed.sl.Orientation` from float values.

        **Parameters**
            - **v0** : ox value
            - **v1** : oy value
            - **v2** : oz value
            - **v3** : ow value
        """
        self.orientation = c_Orientation(Vector4[float](v0, v1, v2, v3))

    def init_rotation(self, Rotation rotation):
        """
        Inits :class:`~pyzed.sl.Orientation` from :class:`~pyzed.sl.Rotation`

        It converts the :class:`~pyzed.sl.Rotation` representation to the :class:`~pyzed.sl.Orientation` one.

        **Parameters**
            - **rotation** : :class:`~pyzed.sl.Rotation` to be converted
        """
        self.orientation = c_Orientation(rotation.rotation)

    def init_translation(self, Translation tr1, Translation tr2):
        """
        Inits  :class:`~pyzed.sl.Orientation` from two :class:`~pyzed.sl.Translation`

        **Parameters**
            - **tr1** : First :class:`~pyzed.sl.Translation`
            - **tr2** : Second :class:`~pyzed.sl.Translation`
        """
        self.orientation = c_Orientation(tr1.translation, tr2.translation)

    def set_rotation_matrix(self, Rotation py_rotation):
        """
        Sets the orientation from a :class:`~pyzed.sl.Rotation`.

        **Parameters**
            - **rotation** : the :class:`~pyzed.sl.Rotation` to be used.
        """
        self.orientation.setRotationMatrix(py_rotation.rotation)

    def get_rotation_matrix(self):
        """
        Returns the current orientation as a :class:`~pyzed.sl.Rotation`.

        **Returns**
            - The rotation computed from the orientation data.
        """
        py_rotation = Rotation()
        py_rotation.mat = self.orientation.getRotationMatrix()
        return py_rotation

    def set_identity(self):
        """
        Sets the current :class:`~pyzed.sl.Orientation` to identity.
        """
        self.orientation.setIdentity()
        return self

    def identity(self):
        """
        Creates an :class:`~pyzed.sl.Orientation` initialized to identity.

        **Returns**
            - An identity :class:`~pyzed.sl.Orientation`.
        """
        self.orientation.identity()
        return self

    def set_zeros(self):
        """
        Fills the current :class:`~pyzed.sl.Orientation` with zeros.
        """
        self.orientation.setZeros()

    def zeros(self):
        """
        Creates an :class:`~pyzed.sl.Orientation` filled with zeros.

        **Returns**
            - An :class:`~pyzed.sl.Orientation` filled with zeros.
        """
        self.orientation.zeros()
        return self

    def normalize(self):
        """
        Normalizes the current :class:`~pyzed.sl.Orientation`.
        """
        self.orientation.normalise()

    @staticmethod
    def normalize_orientation(Orientation orient):
        """
        Creates the normalized version of an existing :class:`~pyzed.sl.Orientation`.

        **Parameters**
            - **orient**  : the :class:`~pyzed.sl.Orientation` to be used.

        **Returns**
            - The normalized version of the :class:`~pyzed.sl.Orientation`. 
        """
        orient.orientation.normalise()
        return orient

    def size(self):
        return self.orientation.size()

    def get(self):
        """
        Returns a numpy array of the :class:`~pyzed.sl.Orientation`.

        **Returns**
            - A numpy array of the :class:`~pyzed.sl.Orientation`
        """
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.orientation(i)
        return arr

    def __mul__(Orientation self, Orientation other):
        orient = Orientation()
        orient.orientation = self.orientation * other.orientation
        return orient


cdef class Transform(Matrix4f):
    """
    Designed to contain translation and rotation data of the positional tracking.

    It contains the orientation as well. It can be used to create any type of Matrix4x4 or :class:`~pyzed.sl.Matrix4f` that must be specifically used for handling a rotation and position information (OpenGL, Tracking...). It inherits from the generic :class:`~pyzed.sl.Matrix4f`.
    """
    cdef c_Transform transform
    def __cinit__(self):
        self.transform = c_Transform()

    def init_transform(self, Transform motion):
        """
        Deep copy from another :class:`~pyzed.sl.Transform`

        **Parameters**
            - **motion** : :class:`~pyzed.sl.Transform` to be copied
        """
        self.transform = c_Transform(motion.transform)
        self.mat = motion.mat

    def init_matrix(self, Matrix4f matrix):
        """
        Inits :class:`~pyzed.sl.Transform` from a :class:`~pyzed.sl.Matrix4f`

        **Parameters**
            - **matrix** : :class:`~pyzed.sl.Matrix4f` to be used.
        """
        self.transform = c_Transform(matrix.mat)
        self.mat = matrix.mat

    def init_rotation_translation(self, Rotation rot, Translation tr):
        """
        Inits :class:`~pyzed.sl.Transform` from a :class:`~pyzed.sl.Rotation` and a :class:`~pyzed.sl.Translation`

        **Parameters**
            - **rot** : :class:`~pyzed.sl.Rotation` to be used.
            - **tr**  : :class:`~pyzed.sl.Translation` to be used.
        """
        self.transform = c_Transform(rot.rotation, tr.translation)
        self.mat = c_Matrix4f(self.transform.m)

    def init_orientation_translation(self, Orientation orient, Translation tr):
        """
        Inits :class:`~pyzed.sl.Transform` from a :class:`~pyzed.sl.Orientation` and a :class:`~pyzed.sl.Translation`

        **Parameters**
            - **orient** : :class:`~pyzed.sl.Orientation` to be used.
            - **tr**  : :class:`~pyzed.sl.Translation` to be used.
        """
        self.transform = c_Transform(orient.orientation, tr.translation)
        self.mat = c_Matrix4f(self.transform.m)

    def set_rotation_matrix(self, Rotation py_rotation):
        """
        Sets the rotation of the current :class:`~pyzed.sl.Transform` from an :class:`~pyzed.sl.Rotation`.

        **Parameters**
            - **rotation**    : the :class:`~pyzed.sl.Rotation` to be used. 
        """
        self.transform.setRotationMatrix(<c_Rotation>py_rotation.mat)

    def get_rotation_matrix(self):
        """
        Returns the :class:`~pyzed.sl.Rotation` of the current :class:`~pyzed.sl.Transform`.

        **Returns**
            - The :class:`~pyzed.sl.Rotation` created from the :class:`~pyzed.sl.Transform` values. 
        """
        py_rotation = Rotation()
        py_rotation.mat = self.transform.getRotationMatrix()
        return py_rotation

    def set_translation(self, Translation py_translation):
        """
        Sets the translation of the current :class:`~pyzed.sl.Transform` from an :class:`~pyzed.sl.Translation`.

        **Parameters**
            - **translation** : the :class:`~pyzed.sl.Translation` to be used.
        """
        self.transform.setTranslation(py_translation.translation)

    def get_translation(self):
        """
        Returns the :class:`~pyzed.sl.Translation` of the current :class:`~pyzed.sl.Transform`.

        **Returns**
            - The :class:`~pyzed.sl.Translation` created from the :class:`~pyzed.sl.Transform` values.

        .. warning::
            The given :class:`~pyzed.sl.Translation` contains a copy of the :class:`~pyzed.sl.Transform` values. Not references. 
        """
        py_translation = Translation()
        py_translation.translation = self.transform.getTranslation()
        return py_translation

    def set_orientation(self, Orientation py_orientation):
        """
        Sets the orientation of the current :class:`~pyzed.sl.Transform` from an :class:`~pyzed.sl.Orientation`.

        **Parameters**
            - **orientation** : the :class:`~pyzed.sl.Orientation` to be used.
        """
        self.transform.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        """
        Returns the :class:`~pyzed.sl.Orientation` of the current :class:`~pyzed.sl.Transform`.

        **Returns**
            - The :class:`~pyzed.sl.Orientation` created from the :class:`~pyzed.sl.Transform` values.

        .. warning::
            The given :class:`~pyzed.sl.Orientation` contains a copy of the :class:`~pyzed.sl.Transform` values. Not references. 
        """
        py_orientation = Orientation()
        py_orientation.orientation = self.transform.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        """
        Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.

        **Returns**
            - The rotation vector (numpy array)
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.transform.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        """
        Sets the Rotation 3x3 of the Transform with a 3x1 rotation vector (using Rodrigues' transformation).

        **Parameters**
            - **input0** : First float value
            - **input1** : Second float value
            - **input2** : Third float value
        """
        self.transform.setRotationVector(Vector3[float](input0, input1, input2))

    def get_euler_angles(self, radian=True):
        """
        Converts the :class:`~pyzed.sl.Rotation` of the :class:`~pyzed.sl.Transform` as Euler angles.

        **Parameters**
            - **radian**  : Define if the angle in is radian or degree

        **Returns**
            - The Euler angles, as a float3 representing the rotations arround the X, Y and Z axes. 
        """
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.transform.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    def set_euler_angles(self, float input0, float input1, float input2, radian=True):
        """
        Sets the :class:`~pyzed.sl.Rotation` of the :class:`~pyzed.sl.Transform` from the Euler angles.

        **Parameters**
            - **input0** : First float euler value
            - **input1** : Second float euler value
            - **input2** : Third float euler value
            - **radian**  : Define if the angle in is radian or degree
        """
        if isinstance(radian, bool):
            self.transform.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")



class MESH_FILE_FORMAT(enum.Enum):
    """
    Lists available mesh file formats.

    +---------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                        |
    +=========================+=========================================================================+
    |MESH_FILE_FORMAT_PLY     |Contains only vertices and faces.                                        |
    +-------------------------+-------------------------------------------------------------------------+
    |MESH_FILE_FORMAT_PLY_BIN |Contains only vertices and faces, encoded in binary.                     |
    +-------------------------+-------------------------------------------------------------------------+
    |MESH_FILE_FORMAT_OBJ     |Contains vertices, normals, faces and textures informations if possible. |
    +-------------------------+-------------------------------------------------------------------------+
    |MESH_FILE_FORMAT_LAST    |                                                                         |
    +-------------------------+-------------------------------------------------------------------------+
    """
    MESH_FILE_PLY = c_MESH_FILE_FORMAT.MESH_FILE_PLY
    MESH_FILE_PLY_BIN = c_MESH_FILE_FORMAT.MESH_FILE_PLY_BIN
    MESH_FILE_OBJ = c_MESH_FILE_FORMAT.MESH_FILE_OBJ
    MESH_FILE_LAST = c_MESH_FILE_FORMAT.MESH_FILE_LAST

class MESH_TEXTURE_FORMAT(enum.Enum):
    """
    Lists available mesh texture formats. 

    +-----------------------------------------------+
    |Enumerators                                    |
    +==================+============================+
    |MESH_TEXTURE_RGB  |The texture has 3 channels. |
    +------------------+----------------------------+
    |MESH_TEXTURE_RGBA |The texture has 4 channels. |
    +------------------+----------------------------+
    |MESH_TEXTURE_LAST |                            |
    +------------------+----------------------------+
    """
    MESH_TEXTURE_RGB = c_MESH_TEXTURE_FORMAT.MESH_TEXTURE_RGB
    MESH_TEXTURE_RGBA = c_MESH_TEXTURE_FORMAT.MESH_TEXTURE_RGBA
    MESH_TEXTURE_LAST = c_MESH_TEXTURE_FORMAT.MESH_TEXTURE_LAST

class MESH_FILTER(enum.Enum):
    MESH_FILTER_LOW = c_MESH_FILTER.MESH_FILTER_LOW
    MESH_FILTER_MEDIUM = c_MESH_FILTER.MESH_FILTER_MEDIUM
    MESH_FILTER_HIGH = c_MESH_FILTER.MESH_FILTER_HIGH

class PLANE_TYPE(enum.Enum):
    PLANE_TYPE_HORIZONTAL = c_PLANE_TYPE.PLANE_TYPE_HORIZONTAL
    PLANE_TYPE_VERTICAL = c_PLANE_TYPE.PLANE_TYPE_VERTICAL
    PLANE_TYPE_UNKNOWN = c_PLANE_TYPE.PLANE_TYPE_UNKNOWN
    PLANE_TYPE_LAST = c_PLANE_TYPE.PLANE_TYPE_LAST

cdef class MeshFilterParameters:
    """
    Defines the behavior of the :meth:`~pyzed.sl.Mesh.filter()` function.

    The constructor sets all the default parameters.


    """
    cdef c_MeshFilterParameters* meshFilter
    def __cinit__(self):
        self.meshFilter = new c_MeshFilterParameters(c_MESH_FILTER.MESH_FILTER_LOW)

    def __dealloc__(self):
        del self.meshFilter

    def set(self, filter=MESH_FILTER.MESH_FILTER_LOW):
        """
        Set the filtering intensity.

        **Parameters**
            - **filter**  : the desired :class:`~pyzed.sl.MESH_FILTER`
        """
        if isinstance(filter, MESH_FILTER):
            self.meshFilter.set(filter.value)
        else:
            raise TypeError("Argument is not of MESH_FILTER type.")

    def save(self, str filename):
        """
        Saves the current bunch of parameters into a file.

        **Parameters**
            - **filename** : the path to the file in which the parameters will be stored.

        **Returns**
            - true if the file was successfully saved, otherwise false.
        """
        filename_save = filename.encode()
        return self.meshFilter.save(String(<char*> filename_save))

    def load(self, str filename):
        """
        Loads the values of the parameters contained in a file.

        **Parameters**
            - **filename**  : the path to the file from which the parameters will be loaded.

        **Returns**
            - true if the file was successfully loaded, otherwise false. 
        """
        filename_load = filename.encode()
        return self.meshFilter.load(String(<char*> filename_load))


cdef class Texture:
    """
    Contains information about texture image associated to a :class:`~pyzed.sl.Mesh`.
    """
    cdef c_Texture texture
    def __cinit__(self):
        self.texture = c_Texture()

    @property
    def name(self):
        """
        The name of the file in which the texture is saved.
        """
        if not self.texture.name.empty():
            return self.texture.name.get().decode()
        else:
            return ""

    @name.setter
    def name(self, str name_):
        self.texture.name.set(name_.encode())

    def get_data(self, Mat py_mat):
        """
        Puts the data of a texture in a :class:`~pyzed.sl.Mat`
        """
        py_mat.mat = self.texture.data
        return py_mat

    @property
    def indice_gl(self):
        """
        Useful for OpenGL binding reference (value not set by the SDK).
        """
        return self.texture.indice_gl

    def clear(self):
        """
        Clears data.
        """
        self.texture.clear()

cdef class PointCloudChunk :
    cdef c_PointCloudChunk chunk

    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 3))
        for i in range(self.chunk.vertices.size()):
            for j in range(3):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3))
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
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


cdef class Chunk:
    """
    Represents a sub-mesh, it contains local vertices and triangles.

    Vertices and normals have the same size and are linked by id stored in triangles.

    .. note::
        uv contains data only if your mesh have textures (by loading it or after calling applyTexture).
    """
    cdef c_Chunk chunk
    def __cinit__(self):
        self.chunk = c_Chunk()

    @property
    def vertices(self):
        """
        Vertices are defined by a 3D point (numpy array).
        """
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 3))
        for i in range(self.chunk.vertices.size()):
            for j in range(3):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    @property
    def triangles(self):
        """
        Triangles (or faces) contains the index of its three vertices. It corresponds to the 3 vertices of the triangle (numpy array).
        """
        cdef np.ndarray arr = np.zeros((self.chunk.triangles.size(), 3))
        for i in range(self.chunk.triangles.size()):
            for j in range(3):
                arr[i,j] = self.chunk.triangles[i].ptr()[j]+1
        return arr

    @property
    def normals(self):
        """
        Normals are defined by three components (numpy array). Normals are defined for each vertices.
        """
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3))
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    @property
    def uv(self):
        """
        UVs defines the 2D projection of each vertices onto the :class:`~pyzed.sl.Texture`
        Values are normalized [0;1], starting from the bottom left corner of the texture (as requested by opengl).

        In order to display a textured mesh you need to bind the :class:`~pyzed.sl.Texture` and then draw each triangles by picking its uv values.

        .. note::
            Contains data only if your mesh have textures (by loading it or calling applytexture). 
        """
        cdef np.ndarray arr = np.zeros((self.chunk.uv.size(), 2))
        for i in range(self.chunk.uv.size()):
            for j in range(2):
                arr[i,j] = self.chunk.uv[i].ptr()[j]
        return arr

    @property
    def timestamp(self):
        """
        Timestamp of the latest update.
        """
        return self.chunk.timestamp

    @property
    def barycenter(self):
        """
        3D centroid of the chunk.
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    @property
    def has_been_updated(self):
        """
        True if the chunk has been updated by an inner process.
        """
        return self.chunk.has_been_updated

    def clear(self):
        """
        Clears all chunk data.
        """
        self.chunk.clear()

cdef class FusedPointCloud :
    cdef c_FusedPointCloud* fpc
    def __cinit__(self):
        self.fpc = new c_FusedPointCloud()

    def __dealloc__(self):
        del self.fpc

    @property
    def chunks(self):
        list = []
        for i in range(self.mesh.chunks.size()):
            py_chunk = PointCloudChunk()
            py_chunk.chunk = self.fpc.chunks[i]
            list.append(py_chunk)
        return list

    def __getitem__(self, x):
        return self.chunks[x]

    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.fpc.vertices.size(), 3))
        for i in range(self.fpc.vertices.size()):
            for j in range(3):
                arr[i,j] = self.fpc.vertices[i].ptr()[j]
        return arr

    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.fpc.normals.size(), 3))
        for i in range(self.fpc.normals.size()):
            for j in range(3):
                arr[i,j] = self.fpc.normals[i].ptr()[j]
        return arr

    def save(self, str filename, typeMesh=MESH_FILE_FORMAT.MESH_FILE_OBJ, id=[]):
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.fpc.save(String(filename.encode()), typeMesh.value, id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    def load(self, str filename, update_mesh=True):
        if isinstance(update_mesh, bool):
            return self.fpc.load(String(filename.encode()), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def clear(self):
        self.fpc.clear()

    def update_from_chunklist(self, id=[]):
        self.fpc.updateFromChunkList(id)

    def get_number_of_points(self):
        return self.fpc.getNumberOfPoints()


cdef class Mesh:
    """
    A mesh contains the geometric (and optionally texture) data of the scene captured by spatial mapping.

    By default the mesh is defined as a set of chunks, this way we update only the data that has to be updated avoiding a time consuming remapping process every time a small part of the :class:`~pyzed.sl.Mesh` is updated.
    """
    cdef c_Mesh* mesh
    def __cinit__(self):
        self.mesh = new c_Mesh()

    def __dealloc__(self):
        del self.mesh

    @property
    def chunks(self):
        """
        contains the list of chunks
        """
        list = []
        for i in range(self.mesh.chunks.size()):
            py_chunk = Chunk()
            py_chunk.chunk = self.mesh.chunks[i]
            list.append(py_chunk)
        return list

    def __getitem__(self, x):
        return self.chunks[x]

    def filter(self, MeshFilterParameters params, update_mesh=True):
        """
        Filters the mesh.

        The resulting mesh in smoothed, small holes are filled and small blobs of non connected triangles are deleted.

        **Parameters**
            - **mesh_filter_params** : defines the filtering parameters, for more info checkout the :class:`~pyzed.sl.MeshFilterParameters` documentation. default : preset.
            - **update_chunk_only** : if set to false the mesh data (vertices/normals/triangles) are updated otherwise only the chunk's data are updated. default : false.

        **Returns**
            - True if the filtering was successful, false otherwise.

        .. note::
            The filtering is a costly operation, its not recommended to call it every time you retrieve a mesh but at the end of your spatial mapping process.
        """
        if isinstance(update_mesh, bool):
            return self.mesh.filter(deref(params.meshFilter), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def apply_texture(self, texture_format=MESH_TEXTURE_FORMAT.MESH_TEXTURE_RGB):
        """
        Applies texture to the mesh.

        By using this function you will get access to :data:`~pyzed.sl.Mesh.uv`, and :data:`~pyzed.sl.Mesh.texture`. The number of triangles in the mesh may slightly differ before and after calling this functions due to missing texture information. There is only one texture for the mesh, the uv of each chunks are expressed for it in its globality. Vectors of vertices/normals and uv have now the same size.

        **Parameters**
            - **texture_format** : define the number of channels desired for the computed texture. default : MESH_TEXTURE_RGB. 

        **Returns**
            - True if the texturing was successful, false otherwise.

        .. note::
            This function can be called as long as you do not start a new spatial mapping process, due to shared memory. 
            This function can require a lot of computation time depending on the number of triangles in the mesh. Its recommended to call it once a the end of your spatial mapping process.

        .. warning::
            The save_texture parameter in :class:`~pyzed.sl.SpatialMappingParameters` must be set as true when enabling the spatial mapping to be able to apply the textures. 
            The mesh should be filtered before calling this function since :meth:`~pyzed.sl.Mesh.filter` will erase the textures, the texturing is also significantly slower on non-filtered meshes. 
        """
        if isinstance(texture_format, MESH_TEXTURE_FORMAT):
            return self.mesh.applyTexture(texture_format.value)
        else:
            raise TypeError("Argument is not of MESH_TEXTURE_FORMAT type.")

    def save(self, str filename, typeMesh=MESH_FILE_FORMAT.MESH_FILE_OBJ, id=[]):
        """
        Saves the current :class:`~pyzed.sl.Mesh` into a file.

        True if the file was successfully saved, false otherwise.

        **Parameters**
            - **filename**  : the path and filename of the mesh.
            - **typeMesh**  : defines the file type (extension). default : :data:`~pyzed.sl.MESH_FILE_FORMAT.MESH_FILE_OBJ`.
            - **IDs**       :  (by default empty) Specify a set of chunks to be saved, if none provided alls chunks are saved. default : (empty).

        **Returns**
            - True if the file was successfully saved, false otherwise.

        .. note::
            Only :data:`~pyzed.sl.MESH_FILE.MESH_FILE_OBJ` support textures data.

            This function operate on the :class:`~pyzed.sl.Mesh` not on the chunks. This way you can save different parts of your :class:`~pyzed.sl.Mesh` (update your :class:`~pyzed.sl.Mesh` with :meth:`~pyzed.sl.Mesh.update_mesh_from_chunklist()`). 
        """
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.mesh.save(String(filename.encode()), typeMesh.value, id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    def load(self, str filename, update_mesh=True):
        """
        Loads the mesh from a file.

        **Parameters**
            - **filename** : the path and filename of the mesh (do not forget the extension).
            - **update_chunk_only** : if set to false the mesh data (vertices/normals/triangles) are updated otherwise only the chunk's data are updated. default : false. 

        **Returns**
            - True if the loading was successful, false otherwise.

        .. note::
            Updating the :class:`~pyzed.sl.Mesh` is time consuming, consider using only Chunks for better performances.
        """
        if isinstance(update_mesh, bool):
            return self.mesh.load(String(filename.encode()), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def clear(self):
        """
        Clears all the data.
        """
        self.mesh.clear()

    @property
    def vertices(self):
        """
        Vertices are defined by a 3D point (numpy array)
        """
        cdef np.ndarray arr = np.zeros((self.mesh.vertices.size(), 3))
        for i in range(self.mesh.vertices.size()):
            for j in range(3):
                arr[i,j] = self.mesh.vertices[i].ptr()[j]
        return arr

    @property
    def triangles(self):
        """
        Triangles (or faces) contains the index of its three vertices. It corresponds to the 3 vertices of the triangle (numpy array).
        """
        cdef np.ndarray arr = np.zeros((self.mesh.triangles.size(), 3))
        for i in range(self.mesh.triangles.size()):
            for j in range(3):
                arr[i,j] = self.mesh.triangles[i].ptr()[j]+1
        return arr

    @property
    def normals(self):
        """
        Normals are defined by three components, {nx, ny, nz}. Normals are defined for each vertices. (numpy array)
        """
        cdef np.ndarray arr = np.zeros((self.mesh.normals.size(), 3))
        for i in range(self.mesh.normals.size()):
            for j in range(3):
                arr[i,j] = self.mesh.normals[i].ptr()[j]
        return arr

    @property
    def uv(self):
        """
        UVs defines the 2D projection of each vertices onto the :class:`~pyzed.sl.Texture`. (numpy array)

        Values are normalized [0;1], starting from the bottom left corner of the texture (as requested by opengl).

        In order to display a textured mesh you need to bind the :class:`~pyzed.sl.Texture` and then draw each triangles by picking its uv values.

        .. note::
            Contains data only if your mesh have textures (by loading it or calling :meth:`~pyzed.sl.apply_texture()`). 
        """
        cdef np.ndarray arr = np.zeros((self.mesh.uv.size(), 2))
        for i in range(self.mesh.uv.size()):
            for j in range(2):
                arr[i,j] = self.mesh.uv[i].ptr()[j]
        return arr

    @property
    def texture(self):
        """
        :class:`~pyzed.sl.Texture` of the :class:`~pyzed.sl.Mesh`.

        .. note::
            Contains data only if your mesh have textures (by loading it or calling :meth:`~pyzed.sl.apply_texture()`).
        """
        py_texture = Texture()
        py_texture.texture = self.mesh.texture
        return py_texture

    def get_number_of_triangles(self):
        """
        Computes the total number of triangles stored in all chunks.

        **Returns**
            - The number of triangles stored in all chunks.
        """
        return self.mesh.getNumberOfTriangles()

    def merge_chunks(self, faces_per_chunk):
        """
        Merges currents chunks.

        This can be used to merge chunks into bigger sets to improve rendering process.

        **Parameters**
            - **faces_per_chunk** : define the new number of faces per chunk (useful for Unity that doesn't handle chunks over 65K vertices).

        .. note::
            You should not use this function during spatial mapping process because mesh updates will revert this changes. 
        """
        self.mesh.mergeChunks(faces_per_chunk)

    def get_gravity_estimate(self):
        """
        Estimates the gravity vector.

        This function looks for a dominant plane in the whole mesh considering that it is the floor (or a horizontal plane). This can be used to find the gravity and then create realistic physical interactions.

        **Rerturns**
            - The gravity vector.
        """
        gravity = self.mesh.getGravityEstimate()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = gravity[i]
        return arr

    def get_visible_list(self, Transform camera_pose):
        """
        Computes the list of visible chunk from a specific point of view.

        **Parameters**
            - **world_reference_pose** : the point of view, given in world reference.

        **Returns**
            - The list of visible chunks.
        """
        return self.mesh.getVisibleList(camera_pose.transform)

    def get_surrounding_list(self, Transform camera_pose, float radius):
        """
        Computes the list of chunks which are close to a specific point of view.

        **Parameters**
            - **world_reference_position** : the point of view, given in world reference.
            - **radius** : the radius in defined :class:`~pyzed.sl.UNIT`

        **Returns**
            - The list of chunks close to the given point.
        """
        return self.mesh.getSurroundingList(camera_pose.transform, radius)

    def update_mesh_from_chunklist(self, id=[]):
        """
        Updates :data:`~pyzed.sl.Mesh.vertices` / :data:`~pyzed.sl.Mesh.normals` / :data:`~pyzed.sl.Mesh.triangles` / :data:`~pyzed.sl.Mesh.uv` from chunks' data pointed by the given chunkList.

        .. note::
            If the given chunkList is empty, all chunks will be used to update the current :class:`~pyzed.sl.Mesh`. 
        """
        self.mesh.updateMeshFromChunkList(id)

cdef class Plane:
    """
    A plane defined by a point and a normal, or a plane equation. Other elements can be extracted such as the mesh, the 3D bounds...

    .. note::
        The plane measurement are expressed in REFERENCE_FRAME defined by the :class:`~pyzed.sl.RuntimeParameters` :data:`~pyzed.sl.RuntimeParameters.measure3D_reference_frame`
    """
    cdef c_Plane plane
    def __cinit__(self):
        self.plane = c_Plane()

    @property
    def type(self):
        """
        The plane type define the plane orientation : vertical or horizontal.

        .. warning::
            It is deduced from the gravity vector and is therefore only available with the ZED-M. The ZED will give PLANE_TYPE_UNKNOWN for every planes.
        """
        return PLANE_TYPE(self.plane.type)

    @type.setter
    def type(self, type_):
        if isinstance(type_, PLANE_TYPE) :
            self.plane.type = <c_PLANE_TYPE>(type_.value)
        else :
            raise TypeError("Argument is not of PLANE_TYPE type")

    def get_normal(self):
        """
        Get the plane normal vector.

        **Returns**
            - :class:`~pyzed.sl.Plane` normal vector, with normalized components (numpy array)
        """
        normal = self.plane.getNormal()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = normal[i]
        return arr

    def get_center(self):
        """
        Get the plane center point.

        **Returns**
            - :class:`~pyzed.sl.Plane` center point (numpy array)
        """
        center = self.plane.getCenter()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = center[i]
        return arr

    def get_pose(self, Transform py_pose = Transform()):
        """
        Get the plane pose relative to the global reference frame.

        **Parameters**
            - A :class:`~pyzed.sl.Transform` or it creates one by default.

        **Returns**
            - A transformation matrix (rotation and translation) which give the plane pose. Can be used to transform the global reference frame center (0,0,0) to the plane center.
        """
        py_pose.transform = self.plane.getPose()
        return py_pose

    def get_extents(self):
        """
        Get the width and height of the bounding rectangle around the plane contours.

        **Returns**
            - Width and height of the bounding plane contours (numpy array)

        .. warning::
            This value is expressed in the plane reference frame 
        """
        extents = self.plane.getExtents()
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = extents[i]
        return arr

    def get_plane_equation(self):
        """
        Get the plane equation.

        **Returns**
            - :class:`~pyzed.sl.Plane` equation, in the form : ax+by+cz=d, the returned values are (a,b,c,d) (numpy array)
        """
        plane_eq = self.plane.getPlaneEquation()
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = plane_eq[i]
        return arr

    def get_bounds(self):
        """
        Get the polygon bounds of the plane.

        **Returns**
            - Vector of 3D points forming a polygon bounds corresponding to the current visible limits of the plane (numpy array)
        """
        cdef np.ndarray arr = np.zeros((self.plane.getBounds().size(), 3))
        for i in range(self.plane.getBounds().size()):
            for j in range(3):
                arr[i,j] = self.plane.getBounds()[i].ptr()[j]
        return arr

    def extract_mesh(self):
        """
        Compute and return the mesh of the bounds polygon.

        **Returns**
            - A mesh representing the plane delimited by the visible bounds
        """
        ext_mesh = self.plane.extractMesh()
        pymesh = Mesh()
        pymesh.mesh[0] = ext_mesh
        return pymesh

    def get_closest_distance(self, point=[0,0,0]):
        """
        Get the distance between the input point and the projected point alongside the normal vector onto the plane. This corresponds to the closest point on the plane.

        **Parameters**
            - The point to project into the plane 
        """
        cdef Vector3[float] vec = Vector3[float](point[0], point[1], point[2])
        return self.plane.getClosestDistance(vec)


class MAPPING_RESOLUTION(enum.Enum):
    """
    List the spatial mapping resolution presets.

    +---------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                        |
    +==========================+========================================================================+
    |MAPPING_RESOLUTION_HIGH   |Create a detail geometry, requires lots of memory.                      |
    +--------------------------+------------------------------------------------------------------------+
    |MAPPING_RESOLUTION_MEDIUM |Smalls variations in the geometry will disappear, useful for big object |
    +--------------------------+------------------------------------------------------------------------+
    |MAPPING_RESOLUTION_LOW    |Keeps only huge variations of the geometry , useful outdoor.            |
    +--------------------------+------------------------------------------------------------------------+
    """
    MAPPING_RESOLUTION_HIGH = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH
    MAPPING_RESOLUTION_MEDIUM  = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_MEDIUM
    MAPPING_RESOLUTION_LOW = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_LOW


class MAPPING_RANGE(enum.Enum):
    """
    List the spatial mapping depth range presets.

    **Deprecated**: Since SDK 2.6 range is computed from the requested resolution and inner parameters to best fit the current application 

    +---------------------------------------------------------------------------------------------------------------------+
    |Enumerators                                                                                                          |
    +=====================+===============================================================================================+
    |MAPPING_RANGE_NEAR   |Only depth close to the camera will be used during spatial mapping.                            |
    +---------------------+-----------------------------------------------------------------------------------------------+
    |MAPPING_RANGE_MEDIUM |Medium depth range.                                                                            |
    +---------------------+-----------------------------------------------------------------------------------------------+
    |MAPPING_RANGE_FAR    |Takes into account objects that are far, useful outdoor.                                       |
    +---------------------+-----------------------------------------------------------------------------------------------+
    |MAPPING_RANGE_AUTO   |Depth range will be computed based on current :class:`~pyzed.sl.Camera` states and parameters. |
    +---------------------+-----------------------------------------------------------------------------------------------+
    """
    MAPPING_RANGE_NEAR = c_MAPPING_RANGE.MAPPING_RANGE_NEAR
    MAPPING_RANGE_MEDIUM = c_MAPPING_RANGE.MAPPING_RANGE_MEDIUM
    MAPPING_RANGE_FAR = c_MAPPING_RANGE.MAPPING_RANGE_FAR

class SPATIAL_MAP_TYPE(enum.Enum):
    SPATIAL_MAP_TYPE_MESH = c_SPATIAL_MAP_TYPE.SPATIAL_MAP_TYPE_MESH
    SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD = c_SPATIAL_MAP_TYPE.SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD

cdef class InputType:
    """
    Defines the input type used in the ZED SDK. Can be used to select a specific camera with ID or serial number, or a svo file.

    .. note::
        This replaces the previous :data:`~pyzed.sl.InitParameters.camera_linux_id` and :data:`~pyzed.sl.InitParameters.svo_input_filename` (now deprecated). 
    """
    cdef c_InputType input
    def __cinit__(self, input_type=0):
        if input_type == 0 :
            self.input = c_InputType()
        elif isinstance(input_type, InputType) :
            input_t = <InputType> input_type
            self.input = c_InputType(input_t.input)
        else :
            raise TypeError("Argument is not of right type.")

    def set_from_camera_id(self, id):
        """
        Set the input as the camera with specified id
        """
        self.input.setFromCameraID(id)

    def set_from_serial_number(self, serial_number):
        """
        Set the input as the camera with specified serial number
        """
        self.input.setFromSerialNumber(serial_number)

    def set_from_svo_file(self, str svo_input_filename):
        """
        Set the input as the svo specified with the filename
        """
        filename = svo_input_filename.encode()
        self.input.setFromSVOFile(String(<char*> filename))

    def set_from_stream(self, str sender_ip, port=30000):
        sender_ip_ = sender_ip.encode()
        self.input.setFromStream(String(<char*>sender_ip_), port)
 

cdef class InitParameters:
    """
    Holds the options used to initialize the :class:`~pyzed.sl.Camera` object.

    Once passed to the :meth:`~pyzed.sl.Camera.open()` function, these settings will be set for the entire execution life time of the :class:`~pyzed.sl.Camera`.

    You can get further information in the detailed description bellow.

    This structure allows you to select multiple parameters for the :class:`~pyzed.sl.Camera` such as the selected camera, its resolution, depth mode, coordinate system, and unit, of measurement. Once filled with the desired options, it should be passed to the :meth:`~pyzed.sl.Camera.open()` function.

    .. code-block:: python

        import pyzed.sl as sl

        def main() :
            zed = sl.Camera() # Create a ZED camera object
            init_params = sl.InitParameters() # Set initial parameters
            init_params.sdk_verbose = False  # Disable verbose mode
            init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080 # Use HD1080 video mode
            init_params.camera_fps = 30 # Set fps at 30
            # Other parameters are left to their default values

            # Open the camera
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS :
                exit(-1)

            # Close the camera
            zed.close()
            return 0

        if __name__ == "__main__" :
            main()

    With its default values, it opens the ZED camera in live mode at :data:`~pyzed.sl.RESOLUTION.RESOLUTION_HD720` and sets the depth mode to :data:`~pyzed.sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE`.

    You can customize it to fit your application. The parameters can also be saved and reloaded using its :meth:`~pyzed.sl.InitParameters.save()` and :meth:`~pyzed.sl.InitParameters.load()` functions. 
    """
    cdef c_InitParameters* init
    def __cinit__(self, camera_resolution=RESOLUTION.RESOLUTION_HD720, camera_fps=0,
                  camera_linux_id=0, svo_input_filename="", svo_real_time_mode=False,
                  depth_mode=DEPTH_MODE.DEPTH_MODE_PERFORMANCE,
                  coordinate_units=UNIT.UNIT_MILLIMETER,
                  coordinate_system=COORDINATE_SYSTEM.COORDINATE_SYSTEM_IMAGE,
                  sdk_verbose=False, sdk_gpu_id=-1, depth_minimum_distance=-1.0, camera_disable_self_calib=False,
                  camera_image_flip=False, enable_right_side_measure=False, camera_buffer_count_linux=4,
                  sdk_verbose_log_file="", depth_stabilization=True, InputType input_t=InputType(),
                  optional_settings_path=""):
        if (isinstance(camera_resolution, RESOLUTION) and isinstance(camera_fps, int) and
            isinstance(camera_linux_id, int) and isinstance(svo_input_filename, str) and
            isinstance(svo_real_time_mode, bool) and isinstance(depth_mode, DEPTH_MODE) and
            isinstance(coordinate_units, UNIT) and
            isinstance(coordinate_system, COORDINATE_SYSTEM) and isinstance(sdk_verbose, bool) and
            isinstance(sdk_gpu_id, int) and isinstance(depth_minimum_distance, float) and
            isinstance(camera_disable_self_calib, bool) and isinstance(camera_image_flip, bool) and
            isinstance(enable_right_side_measure, bool) and isinstance(camera_buffer_count_linux, int) and
            isinstance(sdk_verbose_log_file, str) and isinstance(depth_stabilization, bool) and
            isinstance(input_t, InputType) and isinstance(optional_settings_path, str)) :

            filename = svo_input_filename.encode()
            filelog = sdk_verbose_log_file.encode()
            fileoption = optional_settings_path.encode()
            self.init = new c_InitParameters(camera_resolution.value, camera_fps, camera_linux_id,
                                            String(<char*> filename), svo_real_time_mode, depth_mode.value,
                                            coordinate_units.value, coordinate_system.value, sdk_verbose, sdk_gpu_id,
                                            depth_minimum_distance, camera_disable_self_calib, camera_image_flip,
                                            enable_right_side_measure, camera_buffer_count_linux,
                                            String(<char*> filelog), depth_stabilization,
                                            <CUcontext> 0, input_t.input, String(<char*> fileoption))
        else:
            raise TypeError("Argument is not of right type.")

    def __dealloc__(self):
        del self.init

    def save(self, str filename):
        """
        This function saves the current set of parameters into a file to be reloaded with the :meth:`~pyzed.sl.InitParameters.load()` function.

        **Parameters**
            - **filename** : the path to the file in which the parameters will be stored.

        **Returns**
            - True if file was successfully saved, otherwise false.

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            init_params.sdk_verbose = True # Enable verbose mode
            init_params.input.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read
            init_params.save("initParameters.conf") # Export the parameters into a file
        """
        filename_save = filename.encode()
        return self.init.save(String(<char*> filename_save))

    def load(self, str filename):
        """
        This function set the other parameters from the values contained in a previously saved file.

        **Parameters**
            - **filename** : the path to the file from which the parameters will be loaded.

        **Returns**
            - True if the file was successfully loaded, otherwise false.

        .. note::
            As the :class:`~pyzed.sl.InitParameters` files can be easilly modified manually (using a text editor) this functions allows you to test various settings without re-compiling your application. 

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            init_params.load("initParameters.conf") # Load the init_params from a previously exported file
        """
        filename_load = filename.encode()
        return self.init.load(String(<char*> filename_load))

    @property
    def camera_resolution(self):
        """
        Define the chosen camera resolution. Small resolutions offer higher framerate and lower computation time.

        In most situations, the :data:`~pyzed.sl.RESOLUTION.RESOLUTION_HD720` at 60 fps is the best balance between image quality and framerate.

        Available resolutions are listed here: :class:`~pyzed.sl.RESOLUTION`.
        """
        return RESOLUTION(self.init.camera_resolution)

    @camera_resolution.setter
    def camera_resolution(self, value):
        if isinstance(value, RESOLUTION):
            self.init.camera_resolution = value.value
        else:
            raise TypeError("Argument must be of RESOLUTION type.")

    @property
    def camera_fps(self):
        """
        Requested camera frame rate. If set to 0, the highest FPS of the specified :data:`~pyzed.sl.InitParameters.camera_resolution` will be used.

        See :class:`~pyzed.sl.RESOLUTION` for a list of supported framerates.

        default 0

        .. note::
            If the requested camera_fps is unsuported, the closest available FPS will be used. 
        """
        return self.init.camera_fps

    @camera_fps.setter
    def camera_fps(self, int value):
        self.init.camera_fps = value

    @property
    def camera_linux_id(self):
        """
        **Only for Linux** : This parameter allows you to select the ZED device to be opened when multiple cameras are connected. This ID matches the system ID found in /dev/videoX.
        default : 0

        **Deprecated** : Please check :data:`~pyzed.sl.InitParameters.input`
        """
        return self.init.camera_linux_id

    @camera_linux_id.setter
    def camera_linux_id(self, int value):
        self.init.camera_linux_id = value

    @property
    def svo_input_filename(self):
        """
        The :class:`~pyzed.sl.Camera` object can be used with a live ZED or a recorded sequence saved in an SVO file.
        This parameter allows you to specify the path to the recorded sequence to be played back.
        default : (empty)

        .. note::
            If this parameter remains empty, the SDK will attempt to open a live camera. Setting it to any value will disable live mode. 

        **Deprecated** : Please check :data:`~pyzed.sl.InitParameters.input`
        """
        if not self.init.svo_input_filename.empty():
            return self.init.svo_input_filename.get().decode()
        else:
            return ""

    @svo_input_filename.setter
    def svo_input_filename(self, str value):
        value_filename = value.encode()
        self.init.svo_input_filename.set(<char*>value_filename)

    @property
    def svo_real_time_mode(self):
        """
        When playing back an SVO file, each call to :meth:`~pyzed.sl.Camera.grab()` will extract a new frame and use it.

        However, this ignores the real capture rate of the images saved in the SVO file.

        Enabling this parameter will bring the SDK closer to a real simulation when playing back a file by using the images' timestamps. However, calls to :meth:`~pyzed.sl.Camera.grab()` will return an error when trying to play to fast, and frames will be dropped when playing too slowly.
        """
        return self.init.svo_real_time_mode

    @svo_real_time_mode.setter
    def svo_real_time_mode(self, bool value):
        self.init.svo_real_time_mode = value

    @property
    def depth_mode(self):
        """
        The SDK offers several :class:`~pyzed.sl.DEPTH_MODE` options offering various level of performance and accuracy.

        This parameter allows you to set the :class:`~pyzed.sl.DEPTH_MODE` that best matches your needs.

        default : :data:`~pyzed.sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE`
        """
        return DEPTH_MODE(self.init.depth_mode)

    @depth_mode.setter
    def depth_mode(self, value):
        if isinstance(value, DEPTH_MODE):
            self.init.depth_mode = value.value
        else:
            raise TypeError("Argument must be of DEPTH_MODE type.")

    @property
    def coordinate_units(self):
        """
        This parameter allows you to select the unit to be used for all metric values of the SDK. (depth, point cloud, tracking, mesh, and others).

        default : :data:`~pyzed.sl.UNIT.UNIT_MILLIMETER`
        """
        return UNIT(self.init.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value):
        if isinstance(value, UNIT):
            self.init.coordinate_units = value.value
        else:
            raise TypeError("Argument must be of UNIT type.")

    @property
    def coordinate_system(self):
        """
        Positional tracking, point clouds and many other features require a given :class:`~pyzed.sl.COORDINATE_SYSTEM` to be used as reference. This parameter allows you to select the :class:`~pyzed.sl.COORDINATE_SYSTEM` use by the :class:`~pyzed.sl.Camera` to return its measures.

        This defines the order and the direction of the axis of the coordinate system.

        default : :data:`~pyzed.sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_IMAGE`
        """
        return COORDINATE_SYSTEM(self.init.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, COORDINATE_SYSTEM):
            self.init.coordinate_system = value.value
        else:
            raise TypeError("Argument must be of COORDINATE_SYSTEM type.")

    @property
    def sdk_verbose(self):
        """
        This parameters allows you to enable the verbosity of the SDK to get a variety of runtime information in the console. When developing an application, enabling verbose mode can help you understand the current SDK behavior.

        However, this might not be desirable in a shipped version.

        default : false
        """
        return self.init.sdk_verbose

    @sdk_verbose.setter
    def sdk_verbose(self, bool value):
        self.init.sdk_verbose = value

    @property
    def sdk_gpu_id(self):
        """
        By default the SDK will use the most powerful NVIDIA graphics card found. However, when running several applications, or using several cameras at the same time, splitting the load over available GPUs can be useful. This parameter allows you to select the GPU used by the :class:`~pyzed.sl.Camera` using an ID from 0 to n-1 GPUs in your PC.

        default : -1

        .. note::
            A non-positive value will search for all CUDA capable devices and select the most powerful. 
        """
        return self.init.sdk_gpu_id

    @sdk_gpu_id.setter
    def sdk_gpu_id(self, int value):
        self.init.sdk_gpu_id = value

    @property
    def depth_minimum_distance(self):
        """
        This parameter allows you to specify the minimum depth value (from the camera) that will be computed, measured in the :class:`~pyzed.sl.UNIT` you define.

        In stereovision (the depth technology used by the camera), looking for closer depth values can have a slight impact on performance. However, this difference is almost invisible on modern GPUs.

        In cases of limited computation power, increasing this value can provide better performance.

        default : (-1) corresponding to 700 mm for a ZED and 200 mm for ZED Mini.

        .. note::
            With a ZED camera you can decrease this value to 300 mm whereas you can set it to 100 mm using a ZED Mini. In any case this value cannot be greater than 3 meters. 
        """
        return self.init.depth_minimum_distance

    @depth_minimum_distance.setter
    def depth_minimum_distance(self, float value):
        self.init.depth_minimum_distance = value

    @property
    def camera_disable_self_calib(self):
        """
        At initialization, the :class:`~pyzed.sl.Camera` runs a self-calibration process that corrects small offsets from the device's factory calibration.

        A drawback is that calibration parameters will sligtly change from one run to another, which can be an issue for repeatability.

        If set to true, self-calibration will be disabled and calibration parameters won't be optimized.

        default : false

        .. note::
            In most situations, self calibration should remain enabled. 
        """
        return self.init.camera_disable_self_calib

    @camera_disable_self_calib.setter
    def camera_disable_self_calib(self, bool value):
        self.init.camera_disable_self_calib = value

    @property
    def camera_image_flip(self):
        """
        If you are using the camera upside down, setting this parameter to true will cancel its rotation. The images will be horizontally flipped.

        default : false
        """
        return self.init.camera_image_flip

    @camera_image_flip.setter
    def camera_image_flip(self, bool value):
        self.init.camera_image_flip = value

    @property
    def enable_right_side_measure(self):
        """
        By default, the SDK only computes a single depth map, aligned with the left camera image.

        This parameter allows you to enable the :data:̀`~pyzed.sl.MEASURE_DEPTH.MEASURE_DEPTH_RIGHT` and other MEASURE_<XXX>_RIGHT at the cost of additional computation time.

        For example, mixed reality passthrough applications require one depth map per eye, so this parameter can be activated.

        default : false
        """
        return self.init.enable_right_side_measure

    @enable_right_side_measure.setter
    def enable_right_side_measure(self, bool value):
        self.init.enable_right_side_measure = value

    @property
    def camera_buffer_count_linux(self):
        """
        Images coming from the camera will be saved in a FIFO buffer waiting for the program to call the :meth:`~pyzed.sl.Camera.grab()` function.

        **On Linux Desktop** : This parameter sets the buffer size between 2 and 5. Low values will reduce the latency but can also produce more corrupted frames.

        **On Jetson Boards** : This parameter is fixed to 2 for memory and performance optimizations.

        **On Windows Desktop** : The images aren't buffered, so this parameter won't be interpreted.

        default: 4 on Linux Desktop, 2 on Jetson.

        .. warning::
            Linux Desktop Only, changing this parameter has no effect on Windows or Jetson boards. 
        """
        return self.init.camera_buffer_count_linux

    @camera_buffer_count_linux.setter
    def camera_buffer_count_linux(self, int value):
        self.init.camera_buffer_count_linux = value

    @property
    def sdk_verbose_log_file(self):
        """
        When :data:`~pyzed.sl.InitParameters.sdk_verbose` is enabled, this parameter allows you to redirect both the SDK verbose messages and your own application messages to a file.

        default : (empty) Should contain the path to the file to be written. A file will be created if missing.

        .. note::
            Setting this parameter to any value will redirect all standard output print calls of the entire program. This means that your own standard output print calls will be redirected to the log file. 

        .. warning::
            The log file won't be clear after successive executions of the application. This means that it can grow indefinitely if not cleared. 
        """
        if not self.init.sdk_verbose_log_file.empty():
            return self.init.sdk_verbose_log_file.get().decode()
        else:
            return ""

    @sdk_verbose_log_file.setter
    def sdk_verbose_log_file(self, str value):
        value_filename = value.encode()
        self.init.sdk_verbose_log_file.set(<char*>value_filename)

    @property
    def depth_stabilization(self):
        """
        Regions of the generated depth map can oscillate from one frame to another. These oscillations result from a lack of texture (too homogeneous) on an object and by image noise. 

        This parameter enables a stabilization filter that reduces these oscilations.

        default : true

        .. note::
            The stabilization uses the positional tracking to increase its accuracy, so the Tracking module will be enabled automatically when set to true.

            Notice that calling :meth:`~pyzed.sl.Camera.enable_tracking()` with your own parameters afterward is still possible. 
        """
        return self.init.depth_stabilization

    @depth_stabilization.setter
    def depth_stabilization(self, bool value):
        self.init.depth_stabilization = value

    @property
    def input(self):
        """
        The SDK can handle different input types:

        - Select a camera by its ID (/dev/videoX on Linux, and 0 to N cameras connected on Windows)
        - Select a camera by its serial number
        - Open a recorded sequence in the SVO file format This parameter allows you to select to desired input. It should be used like this:

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            init_params.sdk_verbose = True # Enable verbose mode
            init_params.input.set_from_camera_id(0) # Selects the camera with ID = 0

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            init_params.sdk_verbose = True # Enable verbose mode
            init_params.input.set_from_serial_number(1010) # Selects the camera with serial number = 1010

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            init_params.sdk_verbose = True # Enable verbose mode
            init_params.input.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read

        Available cameras and their ID/serial can be listed using :meth:`~pyzed.sl.Camera.get_device_list`.

        Each :class:`~pyzed.sl.Camera` will create its own memory (CPU and GPU), therefore the number of ZED used at the same time can be limited by the configuration of your computer. (GPU/CPU memory and capabilities)

        default : (empty)

        See :class:`~pyzed.sl.InputType` for complementary information.
        """
        input_t = InputType()
        input_t.input = self.init.input
        return input_t

    @input.setter
    def input(self, InputType input_t):
        self.init.input = input_t.input

    @property
    def optional_settings_path(self):
        """
        Set the optional path where the SDK has to search for the settings files (SN<XXXX>.conf files). Those file contains the calibration of the camera.

        default : (empty). The SNXXX.conf file will be searched in the default directory (/usr/local/zed/settings/ for Linux or C:/ProgramData/stereolabs/settings for Windows)

        .. note::
            if a path is specified and no files has been found, the SDK will search on the default path (see default) for the *.conf file.

            Automatic download of conf file (through ZED Explorer or the installer) will still download the files on the default path. If you want to use another path by using this entry, make sure to copy the file in the proper location.

        .. code-block:: python

            init_params = sl.InitParameters() # Set initial parameters
            home = "/path/to/home"
            path= home+"/Documents/settings/" # assuming /path/to/home/Documents/settings/SNXXXX.conf exists. Otherwise, it will be searched in /usr/local/zed/settings/
            init_params.optional_settings_path = path
        """
        if not self.init.optional_settings_path.empty():
            return self.init.optional_settings_path.get().decode()
        else:
            return ""

    @optional_settings_path.setter
    def optional_settings_path(self, str value):
        value_filename = value.encode()
        self.init.optional_settings_path.set(<char*>value_filename)

    def set_from_camera_id(self, id):
        """
        Call of :meth:`~pyzed.sl.InputType.set_from_camera_id` function of :data:`~pyzed.sl.InitParameters.input`
        """
        self.init.input.setFromCameraID(id)

    def set_from_serial_number(self, serial_number):
        """
        Call of :meth:`~pyzed.sl.InputType.set_from_serial_number` function of :data:`~pyzed.sl.InitParameters.input`
        """
        self.init.input.setFromSerialNumber(serial_number)

    def set_from_svo_file(self, str svo_input_filename):
        """
        Call of :meth:`~pyzed.sl.InputType.set_from_svo_file` function of :data:`~pyzed.sl.InitParameters.input`
        """
        filename = svo_input_filename.encode()
        self.init.input.setFromSVOFile(String(<char*> filename))

    def set_from_stream(self, str sender_ip, port=30000):
        sender_ip_ = sender_ip.encode()
        self.init.input.setFromStream(String(<char*>sender_ip_), port)

cdef class RuntimeParameters:
    """
    Parameters that defines the behavior of the grab.

    Default values are enabled.
    You can customize it to fit your application and then save it to create a preset that can be loaded for further executions.
    """
    cdef c_RuntimeParameters* runtime
    def __cinit__(self, sensing_mode=SENSING_MODE.SENSING_MODE_STANDARD, enable_depth=True,
                  enable_point_cloud=True,
                  measure3D_reference_frame=REFERENCE_FRAME.REFERENCE_FRAME_CAMERA):
        if (isinstance(sensing_mode, SENSING_MODE) and isinstance(enable_depth, bool)
            and isinstance(enable_point_cloud, bool) and
            isinstance(measure3D_reference_frame, REFERENCE_FRAME)):

            self.runtime = new c_RuntimeParameters(sensing_mode.value, enable_depth, enable_point_cloud,
                                                 measure3D_reference_frame.value)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.runtime

    def save(self, str filename):
        """
        Saves the current set of parameters into a file.

        **Parameters**
            - **filename**  : the path to the file in which the parameters will be stored.

        **Returns**
            - true if the file was successfully saved, otherwise false. 
        """
        filename_save = filename.encode()
        return self.runtime.save(String(<char*> filename_save))

    def load(self, str filename):
        """
        Loads the values of the parameters contained in a file.

        **Parameters**
            - **filename** : the path to the file from which the parameters will be loaded.

        **Returns**
            - true if the file was successfully loaded, otherwise false.
        """
        filename_load = filename.encode()
        return self.runtime.load(String(<char*> filename_load))

    @property
    def sensing_mode(self):
        """
        Defines the algorithm used for depth map computation, more info : :class:`~pyzed.sl.SENSING_MODE` definition.

        default : :data:`~pyzed.sl.SENSING_MODE.SENSING_MODE_STANDARD`
        """
        return SENSING_MODE(self.runtime.sensing_mode)

    @sensing_mode.setter
    def sensing_mode(self, value):
        if isinstance(value, SENSING_MODE):
            self.runtime.sensing_mode = value.value
        else:
            raise TypeError("Argument must be of SENSING_MODE type.")

    @property
    def enable_depth(self):
        """
        Defines if the depth map should be computed.

        If false, only the images are available.

        default : true
        """
        return self.runtime.enable_depth

    @enable_depth.setter
    def enable_depth(self, bool value):
        self.runtime.enable_depth = value

    @property
    def measure3D_reference_frame(self):
        """
        Provides 3D measures (point cloud and normals) in the desired reference frame (default is :data:`~pyzed.sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA`)

        default : :data:`~pyzed.sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA`

        .. note::
            : replaces previous move_point_cloud_to_world_frame parameter. 
        """
        return REFERENCE_FRAME(self.runtime.measure3D_reference_frame)

    @measure3D_reference_frame.setter
    def measure3D_reference_frame(self, value):
        if isinstance(value, REFERENCE_FRAME):
            self.runtime.measure3D_reference_frame = value.value
        else:
            raise TypeError("Argument must be of REFERENCE type.")


cdef class TrackingParameters:
    """
    Parameters for positional tracking initialization.

    A default constructor is enabled and set to its default parameters.

    You can customize it to fit your application and then save it to create a preset that can be loaded for further executions.

    .. note::
        Parameters can be user adjusted.
    """
    cdef c_TrackingParameters* tracking
    def __cinit__(self, Transform init_pos, _enable_memory=True, _area_path=None):
        if _area_path is None:
            self.tracking = new c_TrackingParameters(init_pos.transform, _enable_memory, String())
        else:
            raise TypeError("Argument init_pos must be initialized first with Transform().")
    
    def __dealloc__(self):
        del self.tracking

    def save(self, str filename):
        """
        Saves the current set of parameters into a file.

        **Parameters**
            - **filename**  : the path to the file in which the parameters will be stored.

        **Returns**
            - true if the file was successfully saved, otherwise false. 
        """
        filename_save = filename.encode()
        return self.tracking.save(String(<char*> filename_save))

    def load(self, str filename):
        """
        Loads the values of the parameters contained in a file.

        **Parameters**
            - **filename** : the path to the file from which the parameters will be loaded.

        **Returns**
            - true if the file was successfully loaded, otherwise false.
        """
        filename_load = filename.encode()
        return self.tracking.load(String(<char*> filename_load))

    def initial_world_transform(self, Transform init_pos = Transform()):
        """
        Get the position of the camera in the world frame when camera is started. By default it should be identity.

        **Parameters**
            - **init_pose** : :class:`~pyzed.sl.Transform` to be returned, by default it creates one

        **Returns**
            - Position of the camera in the world frame when camera is started.

        .. note::
            The camera frame (defines the reference frame for the camera) is by default positioned at the world frame when tracking is started. 
        """
        init_pos.transform = self.tracking.initial_world_transform
        return init_pos

    def set_initial_world_transform(self, Transform value):
        """
        Set the position of the camera in the world frame when camera is started.

        **Parameters**
            - **value** : :class:`~pyzed.sl.Transform` input
        """
        self.tracking.initial_world_transform = value.transform

    @property
    def enable_spatial_memory(self):
        """
        This mode enables the camera to learn and remember its surroundings. This helps correct positional tracking drift and position different cameras relative to each other in space.

        default : true

        .. warning::
            : This mode requires few resources to run and greatly improves tracking accuracy. We recommend to leave it on by default. 
        """
        return self.tracking.enable_spatial_memory

    @enable_spatial_memory.setter
    def enable_spatial_memory(self, bool value):
        self.tracking.enable_spatial_memory = value

    @property
    def enable_pose_smoothing(self):
        """
        This mode enables smooth pose correction for small drift correction.

        default : false
        """
        return self.tracking.enable_pose_smoothing

    @enable_pose_smoothing.setter
    def enable_pose_smoothing(self, bool value):
        self.tracking.enable_pose_smoothing = value

    @property
    def set_floor_as_origin(self):
        """
        This mode initialize the tracking aligned with the floor plane to better position the camera in space

        default : false 

        .. note::
            : The floor plane detection is launched in the background until it is found. The tracking is in TRACKING_STATE_SEARCHING state.

        .. warning::
            : This features work best with the ZED-M since it needs an IMU to classify the floor. The ZED needs to look at the floor during the initialization for optimum results. 
        """
        return self.tracking.set_floor_as_origin

    @set_floor_as_origin.setter
    def set_floor_as_origin(self, bool value):
        self.tracking.set_floor_as_origin = value

    @property
    def enable_imu_fusion(self):
        """
        This setting allows you to enable or disable the IMU fusion. When set to false, only the optical odometry will be used.

        default : true

        .. note::
            This setting has no impact on the tracking of a ZED camera, only the ZED Mini uses a built-in IMU.
        """
        return self.tracking.enable_imu_fusion

    @enable_imu_fusion.setter
    def enable_imu_fusion(self, bool value):
        self.tracking.enable_imu_fusion = value

    @property
    def area_file_path(self):
        """
        Area localization file that describes the surroundings (previously saved).

        default : (empty)

        .. note::
            Loading an area file will start a searching phase during which the camera will try to position itself in the previously learned area.

        .. warning::
            : The area file describes a specific location. If you are using an area file describing a different location, the tracking function will continuously search for a position and may not find a correct one. 
            The '.area' file can only be used with the same depth mode (MODE) as the one used during area recording. 
        """
        if not self.tracking.area_file_path.empty():
            return self.tracking.area_file_path.get().decode()
        else:
            return ""

    @area_file_path.setter
    def area_file_path(self, str value):
        value_area = value.encode()
        self.tracking.area_file_path.set(<char*>value_area)

class STREAMING_CODEC(enum.Enum):
    """
    List of possible camera state.
    """
    STREAMING_CODEC_AVCHD = c_STREAMING_CODEC.STREAMING_CODEC_AVCHD
    STREAMING_CODEC_HEVC = c_STREAMING_CODEC.STREAMING_CODEC_HEVC
    STREAMING_CODEC_LAST = c_STREAMING_CODEC.STREAMING_CODEC_LAST

cdef class StreamingProperties:
    """
    Properties of all streaming devices
    """
    cdef c_StreamingProperties c_streaming_properties

    @property
    def ip(self):
        return to_str(self.c_streaming_properties.ip).decode()

    @ip.setter
    def ip(self, str ip_):
        self.c_streaming_properties.ip = String(ip_.encode())

    @property
    def port(self):
        return self.c_streaming_properties.port

    @port.setter
    def port(self, port_):
         self.c_streaming_properties.port = port_


cdef class StreamingParameters:
    """
    Sets the streaming parameters.

    The default constructor sets all parameters to their default settings.

    .. note::
        Parameters can be user adjusted.
    """
    cdef c_StreamingParameters* streaming
    def __cinit__(self, codec=STREAMING_CODEC.STREAMING_CODEC_HEVC, port=30000, bitrate=2000, gop_size=-1, adaptative_bitrate=False):
            self.streaming = new c_StreamingParameters(codec.value, port, bitrate, gop_size, adaptative_bitrate)

    def __dealloc__(self):
        del self.streaming

    @property
    def codec(self):
        """
        Defines the codec used for streaming.

        .. warning::
            If HEVC is used, make sure the receiving host is compatible with HEVC decoding (basically a pascal NVIDIA card). If not, prefer to use AVCHD since every compatible NVIDIA card supports AVCHD decoding
        """
        return STREAMING_CODEC(self.streaming.codec)

    @codec.setter
    def codec(self, codec):
        self.streaming.codec = codec.value

    @property
    def port(self):
        """
        Defines the port the data will be streamed on.
        .. warning::
            port must be an even number. Any odd number will be rejected.
        """
        return self.streaming.port

    @port.setter
    def port(self, unsigned short value):
        self.streaming.port = value

    @property
    def bitrate(self):
        """
        Defines the port the data will be streamed on.
        """
        return self.streaming.bitrate

    @bitrate.setter
    def bitrate(self, unsigned int value):
        self.streaming.bitrate = value

    @property
    def adaptative_bitrate(self):
        """
        Enable/Disable adaptive bitrate

        .. note::
            Bitrate will be adjusted regarding the number of packet loss during streaming.

        .. note::
            if activated, bitrate can vary between [bitrate/4, bitrate]

        .. warning::
            Bitrate will be adjusted regarding the number of packet loss during streaming. 
        """
        return self.streaming.adaptative_bitrate

    @adaptative_bitrate.setter
    def adaptative_bitrate(self, bool value):
        self.streaming.adaptative_bitrate = value

    @property
    def gop_size(self):
        """
        Defines the gop size in frame unit.

        .. note::
            if value is set to -1, the gop size will match 2 seconds, depending on camera fps.

        .. note::
            The gop size determines the maximum distance between IDR/I-frames. Very high GOP size will result in slightly more efficient compression, especially on static scene. But it can result in more latency if IDR/I-frame packet are lost during streaming.

        .. note::
            Default value is -1. Maximum allowed value is 256 (frames).
        """ 

        return self.streaming.gop_size

    @gop_size.setter
    def gop_size(self, int value):
        self.streaming.gop_size = value


cdef class SpatialMappingParameters:
    """

    A default constructor is enabled and set to its default parameters.

    You can customize it to fit your application and then save it to create a preset that can be loaded for further executions.

    .. note::
        Parameters can be user adjusted.

    """
    cdef c_SpatialMappingParameters* spatial
    def __cinit__(self, resolution=MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH, mapping_range=MAPPING_RANGE.MAPPING_RANGE_MEDIUM,
                  max_memory_usage=2048, save_texture=True, use_chunk_only=True,
                  reverse_vertex_order=False, map_type=SPATIAL_MAP_TYPE.SPATIAL_MAP_TYPE_MESH):
        if (isinstance(resolution, MAPPING_RESOLUTION) and isinstance(mapping_range, MAPPING_RANGE) and
            isinstance(use_chunk_only, bool) and isinstance(reverse_vertex_order, bool) and isinstance(map_type, SPATIAL_MAP_TYPE)):
            self.spatial = new c_SpatialMappingParameters(resolution.value, mapping_range.value, max_memory_usage, save_texture,
                                                        use_chunk_only, reverse_vertex_order, map_type.value)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.spatial

    def set_resolution(self, resolution=MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH):
        """
        Sets the resolution corresponding to the given :class:`~pyzed.sl.MAPPING_RESOLUTION` preset.

        **Parameters**
            - **resolution**  : the desired :class:`~pyzed.sl.MAPPING_RESOLUTION`. default : :data:`~pyzed.sl.MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH`.
        """
        if isinstance(resolution, MAPPING_RESOLUTION):
            self.spatial.set(<c_MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    def set_range(self, mapping_range=MAPPING_RANGE.MAPPING_RANGE_MEDIUM):
        """
        Sets the range corresponding to the given :class:`~pyzed.sl.MAPPING_RANGE` preset.

        **Parameters**
            - **mapping_range** : the desired :class:`~pyzed.sl.MAPPING_RANGE`. default : :data:`~pyzed.sl.MAPPING_RANGE.MAPPING_RESOLUTION_HIGH`
        """
        if isinstance(mapping_range, MAPPING_RANGE):
            self.spatial.set(<c_MAPPING_RANGE> mapping_range.value)
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    def get_range_preset(self, mapping_range=MAPPING_RANGE.MAPPING_RANGE_MEDIUM):
        """
        Returns the maximum value of depth corresponding to the given :class:`~pyzed.sl.MAPPING_RANGE` presets.

        **Parameters**
            - **range** : the desired :class:`~pyzed.sl.MAPPING_RANGE`. default : :data:`~pyzed.sl.MAPPING_RANGE.MAPPING_RANGE_MEDIUM`.

        **Returns**
            - The maximum value of depth.
        """
        if isinstance(mapping_range, MAPPING_RANGE):
            return self.spatial.get(<c_MAPPING_RANGE> mapping_range.value)
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    def get_resolution_preset(self, resolution=MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH):
        """
        Returns the resolution corresponding to the given :class:`~pyzed.sl.MAPPING_RESOLUTION` preset.

        **Parameters**
            - **resolution**  : the desired :class:`~pyzed.sl.MAPPING_RESOLUTION`. default : :data:`~pyzed.sl.MAPPING_RESOLUTION.RESOLUTION_HIGH`.

        **Returns**
            - The resolution in meter.
        """
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.get(<c_MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    def get_recommended_range(self, resolution, Camera py_cam):
        """
        Returns the recommanded maximum depth value for the given :class:`~pyzed.sl.MAPPING_RESOLUTION` preset.

        **Parameters**
            - **mapping_resolution**  : the desired :class:`~pyzed.sl.MAPPING_RESOLUTION`.
            - **camera**  : the Camera object which will run the spatial mapping.

        **Returns**
            - The maximum value of depth in meters.
        """
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.getRecommendedRange(<c_MAPPING_RESOLUTION> resolution.value, py_cam.camera)
        elif isinstance(resolution, float):
            return self.spatial.getRecommendedRange(<float> resolution, py_cam.camera)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION or float type.")

    @property
    def map_type(self):
        return self.spatial.map_type

    @map_type.setter
    def map_type(self, value):
        self.spatial.map_type = <c_SPATIAL_MAP_TYPE> value.value

    @property
    def max_memory_usage(self):
        """
        The maximum CPU memory (in mega bytes) allocated for the meshing process. 
        """
        return self.spatial.max_memory_usage

    @max_memory_usage.setter
    def max_memory_usage(self, int value):
        self.spatial.max_memory_usage = value

    @property
    def save_texture(self):
        """
        Set to true if you want to be able to apply the texture to your mesh after its creation.

        .. note::
            This option will take more memory. 
        """
        return self.spatial.save_texture

    @save_texture.setter
    def save_texture(self, bool value):
        self.spatial.save_texture = value

    @property
    def use_chunk_only(self):
        """
        Set to false if you want to keep consistency between the mesh and its inner chunks data.

        .. note::
            Updating the :class:`~pyzed.sl.Mesh` is time consuming, consider using only Chunks data for better performances. 
        """
        return self.spatial.use_chunk_only

    @use_chunk_only.setter
    def use_chunk_only(self, bool value):
        self.spatial.use_chunk_only = value

    @property
    def reverse_vertex_order(self):
        """
        Specify if the order of the vertices of the triangles needs to be inverted. If your display process does not handle front and back face culling you can use this to set it right.
        """
        return self.spatial.reverse_vertex_order

    @reverse_vertex_order.setter
    def reverse_vertex_order(self, bool value):
        self.spatial.reverse_vertex_order = value

    @property
    def allowed_range(self):
        """
        Gets the range of the minimal/maximal depth value allowed by the spatial mapping. (numpy array)

        The first value of the array is the minimum value allowed.

        The second value of the array is the maximum value allowed.
        """
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_range.first
        arr[1] = self.spatial.allowed_range.second
        return arr

    @property
    def range_meter(self):
        """
        Depth range in meters. Can be different from the value set by :meth:`~pyzed.sl.Camera.set_depth_max_range_value()`.

        Is sets to 0 by default, in this case the range is computed from resolution_meter and from the currents inner parameters to fit your application.

        **Deprecated** : Since SDK 2.6 range is computed from the requested resolution and inner parameters to best fit the current application
        """
        return self.spatial.range_meter

    @range_meter.setter
    def range_meter(self, float value):
        self.spatial.range_meter = value

    @property
    def allowed_resolution(self):
        """
        Gets the range of the maximal depth value allowed by the spatial mapping. (numpy array)

        The first value of the array is the minimum value allowed.

        The second value of the array is the maximum value allowed.
        """
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_resolution.first
        arr[1] = self.spatial.allowed_resolution.second
        return arr

    @property
    def resolution_meter(self):
        """
        Spatial mapping resolution in meters, should fit :data:`~pyzed.sl.SpatialMappingParameters.allowed_resolution`.
        """
        return self.spatial.resolution_meter

    @resolution_meter.setter
    def resolution_meter(self, float value):
        self.spatial.resolution_meter = value

cdef class Pose:
    """
    Contains positional tracking data which gives the position and orientation of the ZED in 3D space.

    Different representations of position and orientation can be retrieved, along with timestamp and pose confidence.
    """
    cdef c_Pose pose
    def __cinit__(self):
        self.pose = c_Pose()

    def init_pose(self, Pose pose):
        """
        Deep copy from another :class:`~pyzed.sl.Pose`.

        **Parameters**
            - **pose** : the :class:`~pyzed.sl.Pose` to copy.
        """
        self.pose = c_Pose(pose.pose)

    def init_transform(self, Transform pose_data, mtimestamp=0, mconfidence=0):
        """
        Inits :class:`~pyzed.sl.Pose` from pose data.

        **Parameters**
            - **pose_data** : class:`~pyzed.sl.Transform` containing pose data to copy.
            - **mtimestamp** : pose timestamp
            - **mconfidence** : pose confidence
        """
        self.pose = c_Pose(pose_data.transform, mtimestamp, mconfidence)

    def get_translation(self, Translation py_translation = Translation()):
        """
        Returns the translation from the pose. 

        **Parameters**
            - **py_translation** : :class:`~pyzed.sl.Translation` to be returned. It creates one by default.

        **Returns**
            - The (3x1) translation vector
        """
        py_translation.translation = self.pose.getTranslation()
        return py_translation

    def get_orientation(self, Orientation py_orientation = Orientation()):
        """
        Returns the orientation from the pose. 

        **Parameters**
            - **py_orientation** : :class:`~pyzed.sl.Orientation` to be returned. It creates one by default.

        **Returns**
            - The (3x1) orientation vector
        """
        py_orientation.orientation = self.pose.getOrientation()
        return py_orientation

    def get_rotation_matrix(self, Rotation py_rotation = Rotation()):
        """
        Returns the rotation (3x3) from the pose. 

        **Parameters**
            - **py_rotation** : :class:`~pyzed.sl.Rotation` to be returned. It creates one by default.

        **Returns**
            - The (3x3) rotation matrix

        .. warning::
            The given :class:`~pyzed.sl.Rotation` contains a copy of the :class:`~pyzed.sl.Transform` values. Not references. 
        """
        py_rotation.rotation = self.pose.getRotationMatrix()
        py_rotation.mat = self.pose.getRotationMatrix()
        return py_rotation

    def get_rotation_vector(self):
        """
        Returns the rotation (3x1) rotation vector obtained from 3x3 rotation matrix using Rodrigues formula) from the pose.

        **Returns**
            - The (3x1) rotation vector (numpy array)
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.pose.getRotationVector()[i]
        return arr

    def get_euler_angles(self, radian=True):
        """
        Convert the :class:`~pyzed.sl.Rotation` of the :class:`~pyzed.sl.Transform` as Euler angles.

        **Parameters**
            - **radian** : Define if the angle in is radian or degree. default : true.

        **Returns**
            - The Euler angles, as a float3 representing the rotations arround the X, Y and Z axes. (numpy array) 
        """
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.pose.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr

    @property
    def valid(self):
        """
        boolean that indicates if tracking is activated or not. You should check that first if something wrong.
        """
        return self.pose.valid

    @valid.setter
    def valid(self, bool valid_):
        self.pose.valid = valid_

    @property
    def timestamp(self):
        """
        Timestamp of the pose. This timestamp should be compared with the camera timestamp for synchronization.
        """
        return self.pose.timestamp

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.pose.timestamp = timestamp

    def pose_data(self, Transform pose_data = Transform()):
        """
        Gets the 4x4 Matrix which contains the rotation (3x3) and the translation. :class:`~pyzed.sl.Orientation` is extracted from this transform as well.

        **Parameters**
            - **pose_data** : :class:`~pyzed.sl.Transform` to be returned. It creates one by default.

        **Returns**
            - the pose data :class:`~pyzed.sl.Transform`
        """ 
        pose_data.transform = self.pose.pose_data
        pose_data.mat = self.pose.pose_data
        return pose_data

    @property
    def pose_confidence(self):
        """
        Confidence/Quality of the pose estimation for the target frame.

        A confidence metric of the tracking [0-100], 0 means that the tracking is lost, 100 means that the tracking can be fully trusted.
        """
        return self.pose.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, int pose_confidence_):
        self.pose.pose_confidence = pose_confidence_

    @property
    def pose_covariance(self):
        """
        6x6 Pose covariance of translation (the first 3 values) and rotation in so3 (the last 3 values) (numpy array)

        .. note::
            Computed only if :data:`~pyzed.sl.TrackingParameters.enable_spatial_memory` is disabled.
        """
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.pose.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.pose.pose_covariance[i] = pose_covariance_[i]

cdef class IMUData:
    """
    Contains inertial positional tracking data which gives the orientation of the ZED-M.

    Different representations of orientation can be retrieved, along with timestamp and pose confidence. Raw data (linear acceleration and angular velocity) are also given along with the calculated orientation. 

    .. note::
        This data will not be filled if you are using a ZED camera.
    """
    cdef c_IMUData imuData

    def __cinit__(self):
        self.imuData = c_IMUData()
        
    def init_imuData(self, IMUData imuData):
        """
        Deep copy from another :class:`~pyzed.sl.IMUData`.

        **Parameters**
            - **imuData** : the :class:`~pyzed.sl.IMUData` to copy.
        """
        self.imuData = c_IMUData(imuData.imuData)

    def init_transform(self, Transform pose_data, mtimestamp=0, mconfidence=0):
        """
        Inits :class:`~pyzed.sl.IMUData` from pose data.

        **Parameters**
            - **pose_data** : class:`~pyzed.sl.Transform` containing pose data to copy.
            - **mtimestamp** : pose timestamp
            - **mconfidence** : pose confidence
        """
        self.imuData = c_IMUData(pose_data.transform, mtimestamp, mconfidence)

    def get_orientation_covariance(self, Matrix3f orientation_covariance = Matrix3f()):
        """
        Gets the (3x3) Covariance matrix for orientation (x,y,z axes)

        **Parameters**
            - **orientation_covariance** : :class:`~pyzed.sl.Matrix3f` to be returned. It creates one by default.

        **Returns**
            - The (3x3) Covariance matrix for orientation
        """
        orientation_covariance.mat = self.imuData.orientation_covariance
        return orientation_covariance

    def get_angular_velocity(self, angular_velocity):
        """
        Gets the (3x1) Vector for angular velocity of the IMU, given in deg/s. In other words, the current velocity at which the sensor is rotating around the x, y, and z axes.

        **Parameters**
            - **angular_velocity** : A numpy array to be returned.

        **Returns**
            - The angular velocity (3x1) vector in a numpy array
        """
        for i in range(3):
            angular_velocity[i] = self.imuData.angular_velocity[i]
        return angular_velocity

    def get_linear_acceleration(self, linear_acceleration):
        """
        Gets the (3x1) Vector for linear acceleration of the IMU, given in m/s^2. In other words, the current acceleration of the sensor, along with the x, y, and z axes.

        **Parameters**
            - **linear_acceleration** : A numpy array to be returned.

        **Returns**
            - The linear acceleration (3x1) vector in a numpy array
        """
        for i in range(3):
            linear_acceleration[i] = self.imuData.linear_acceleration[i]
        return linear_acceleration

    def get_angular_velocity_convariance(self, Matrix3f angular_velocity_convariance = Matrix3f()):
        """
        Gets the (3x3) Covariance matrix for angular velocity (x,y,z axes)

        **Parameters**
            - **angular_velocity_covariance** : :class:`~pyzed.sl.Matrix3f` to be returned. It creates one by default.

        **Returns**
            - The (3x3) Covariance matrix for angular velocity
        """
        angular_velocity_convariance.mat = self.imuData.angular_velocity_convariance
        return angular_velocity_convariance

    def get_linear_acceleration_convariance(self, Matrix3f linear_acceleration_convariance):
        """
        Gets the (3x3) Covariance matrix for angular velocity (x,y,z axes)

        **Parameters**
            - **angular_velocity_covariance** : :class:`~pyzed.sl.Matrix3f` to be returned. It creates one by default.

        **Returns**
            - The (3x3) Covariance matrix for angular velocity
        """
        linear_acceleration_convariance.mat = self.imuData.linear_acceleration_convariance
        return linear_acceleration_convariance

    def get_translation(self, Translation py_translation):
        """
        Returns the translation from the pose. 

        **Parameters**
            - **py_translation** : :class:`~pyzed.sl.Translation` to be returned. It creates one by default.

        **Returns**
            - The (3x1) translation vector
        """
        py_translation.translation = self.imuData.getTranslation()
        return py_translation

    def get_orientation(self, Orientation py_orientation):
        """
        Returns the orientation from the pose. 

        **Parameters**
            - **py_orientation** : :class:`~pyzed.sl.Orientation` to be returned. It creates one by default.

        **Returns**
            - The (3x1) orientation vector
        """
        py_orientation.orientation = self.imuData.getOrientation()
        return py_orientation

    def get_rotation_matrix(self, Rotation py_rotation):
        """
        Returns the rotation (3x3) from the pose. 

        **Parameters**
            - **py_rotation** : :class:`~pyzed.sl.Rotation` to be returned. It creates one by default.

        **Returns**
            - The (3x3) rotation matrix
        """
        py_rotation.rotation = self.imuData.getRotationMatrix()
        py_rotation.mat = self.imuData.getRotationMatrix()
        return py_rotation

    def get_rotation_vector(self):
        """
        Returns the rotation (3x1) rotation vector obtained from 3x3 rotation matrix using Rodrigues formula) from the pose.

        **Returns**
            - The (3x1) rotation vector (numpy array)
        """
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.imuData.getRotationVector()[i]
        return arr

    def get_euler_angles(self, radian=True):
        """
        Convert the :class:`~pyzed.sl.Rotation` of the :class:`~pyzed.sl.Transform` as Euler angles.

        **Parameters**
            - **radian** : Define if the angle in is radian or degree. default : true.

        **Returns**
            - The Euler angles, as a float3 representing the rotations arround the X, Y and Z axes. (numpy array) 
        """
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.imuData.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr
        
    def pose_data(self, Transform pose_data = Transform()):
        """
        Get the 4x4 Matrix which contains the rotation (3x3) and the translation. :class:`~pyzed.sl.Orientation` is extracted from this transform as well.

        **Parameters**
            - **pose_data** : :class:`~pyzed.sl.Transform` to be returned. It creates one by default.

        **Returns**
            - the pose data :class:`~pyzed.sl.Transform`
        """
        pose_data.transform = self.imuData.pose_data
        pose_data.mat = self.imuData.pose_data
        return pose_data

    @property
    def valid(self):
        """
        boolean that indicates if tracking is activated or not. You should check that first if something wrong.
        """
        return self.imuData.valid

    @valid.setter
    def valid(self, bool valid_):
        self.imuData.valid = valid_
 
    @property
    def timestamp(self):
        """
        Timestamp of the pose. This timestamp should be compared with the camera timestamp for synchronization.
        """
        return self.imuData.timestamp

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.imuData.timestamp = timestamp

    @property
    def pose_confidence(self):
        """
        Confidence/Quality of the pose estimation for the target frame.

        A confidence metric of the tracking [0-100], 0 means that the tracking is lost, 100 means that the tracking can be fully trusted.
        """
        return self.imuData.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, int pose_confidence_):
        self.imuData.pose_confidence = pose_confidence_

    @property
    def pose_covariance(self):
        """
        6x6 Pose covariance of translation (the first 3 values) and rotation in so3 (the last 3 values) (numpy array)

        .. note::
            Computed only if :data:`~pyzed.sl.TrackingParameters.enable_spatial_memory` is disabled.
        """
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.imuData.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.imuData.pose_covariance[i] = pose_covariance_[i]


cdef class Camera:
    """
    This class is the main interface with the camera and the SDK features, such as: video, depth, tracking, mapping, and more.

    Find more information in the detailed description below.

    A standard program will use the :class:`~pyzed.sl.Camera` like this:

    .. code-block:: python

        import pyzed.sl as sl

        def main():
            # --- Initialize a Camera object and open the ZED
            # Create a ZED camera object
            zed = sl.Camera()

            # Set configuration parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720 #Use HD720 video mode
            init_params.camera_fps = 60 # Set fps at 60

            # Open the camera
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS :
                print(repr(err))
                exit(-1)

            runtime_param = sl.RuntimeParameters()
            runtime_param.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

            # --- Main loop grabing images and depth values
            # Capture 50 frames and stop
            i = 0
            image = sl.Mat()
            depth = sl.Mat()
            while i < 50 :
                # Grab an image
                if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
                    # Display a pixel color
                    zed.retrieve_image(image, sl.VIEW.VIEW_LEFT) # Get the left image
                    center_rgb = image.get_value(image.get_width() / 2, image.get_height() / 2)
                    print("Image ", i, " center pixel R:", int(center_rgb[0]), " G:", int(center_rgb[1]), " B:", int(center_rgb[2]))

                    # Display a pixel depth
                    zed.retrieve_measure(depth, sl.MEASURE.MEASURE_DEPTH) # Get the depth map
                    center_depth = depth.get_value(depth.get_width() / 2, depth.get_height() /2)
                    print("Image ", i," center depth:", center_depth)

                    i = i+1

            # --- Close the Camera
            zed.close()
            return 0

        if __name__ == "__main__":
            main()

    """
    cdef c_Camera camera
    def __cinit__(self):
        self.camera = c_Camera()

    def close(self):
        """
        If :func:`~pyzed.sl.Camera.open()` has been called, this function will close the connection to the camera (or the SVO file) and free the corresponding memory.

        If :func:`~pyzed.sl.Camera.open()` wasn't called or failed, this function won't have any effects.

        .. note::
            If an asynchronous task is running within the :class:`~pyzed.sl.Camera` object, like :meth:`~pyzed.sl.Camera.save_current_area()`, this function will wait for its completion.

            The :meth:`~pyzed.sl.Camera.open()` function can then be called if needed.

        .. warning::
            If the CUDA context was created by :meth:`~pyzed.sl.Camera.open()`, this function will destroy it. Please make sure to delete your GPU :class:`~pyzed.sl.Mat` objects before the context is destroyed. 
        """
        self.camera.close()

    def open(self, InitParameters py_init):
        """
        Opens the ZED camera from the provided InitParameter.
        This function will also check the hardware requirements and run a self-calibration.

        **Parameters**
            - **py_init** : a structure containing all the initial parameters. default : a preset of :class:`~pyzed.sl.InitParameters`.

        **Returns**
            - An error code giving information about the internal process. If :data:`~pyzed.sl.ERROR_CODE.SUCCESS` is returned, the camera is ready to use. Every other code indicates an error and the program should be stopped.

        Here is the proper way to call this function:

        .. code-block:: python

            zed = sl.Camera() # Create a ZED camera object

            init_params = sl.InitParameters() # Set configuration parameters
            init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720 # Use HD720 video mode
            init_params.camera_fps = 60 # Set fps at 60

            # Open the camera
            err = zed.open(init_params)
            if (err != sl.ERROR_CODE.SUCCESS) :
                print(repr(err)) # Display the error
                exit(-1)

        .. note::
            If you are having issues opening a camera, the diagnostic tool provided in the SDK can help you identify to problems.

            If this function is called on an already opened camera, :meth:`~pyzed.sl.Camera.close()` will be called.

            Windows: C:\\\\Program Files (x86)\\\\ZED SDK\\\\tools\\\\ZED Diagnostic.exe

            Linux: /usr/local/zed/tools/ZED Diagnostic
        """
        if py_init:
            return ERROR_CODE(self.camera.open(deref(py_init.init)))
        else:
            print("InitParameters must be initialized first with InitParameters().")

    def is_opened(self):
        """
        Reports if the camera has been successfully opened. It has the same behavior as checking if :func:`~pyzed.sl.Camera.open()` returns :data:`~pyzed.sl.ERROR_CODE.SUCCESS`.

        **Returns**
            - true if the ZED is already setup, otherwise false. 
        """
        return self.camera.isOpened()

    def grab(self, RuntimeParameters py_runtime):
        """
        This function will grab the latest images from the camera, rectify them, and compute the measurements based on the :class:`~pyzed.sl.RuntimeParameters` provided (depth, point cloud, tracking, etc.)
        As measures are created in this function, its execution can last a few milliseconds, depending on your parameters and your hardware.
        The exact duration will mostly depend on the following parameters:

            - :data:`~pyzed.sl.InitParameters.enable_right_side_measure` : Activating this parameter increases computation time

            - :data:`~pyzed.sl.InitParameters.depth_stabilization` : Stabilizing the depth requires an additional computation load as it enables tracking

            - :data:`~pyzed.sl.InitParameters.camera_resolution` : Lower resolutions are faster to compute

            - :data:`~pyzed.sl.InitParameters.depth_mode` : :data:`~pyed.sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE` will run faster than :data:`~pyzed.sl.DEPTH_MODE.DEPTH_MODE_ULTRA`

            - :meth:`~pyzed.sl.Camera.enable_tracking()` : Activating the tracking is an additional load

            - :data:`~pyzed.sl.RuntimeParameters.sensing_mode` : :data:`~pyzed.sl.SENSING_MODE.SENSING_MODE_STANDARD` mode will run faster than :data:`~pyzed.sl.SENSING_MODE.SENSING_MODE_FILL` mode, which needs to estimate the depth of occluded pixels.

            - :data:`~pyzed.sl.RuntimeParameters.enable_depth` : Avoiding the depth computation must be faster. However, it is required by most SDK features (tracking, spatial mapping, plane estimation, etc.)

        If no images are available yet, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_NOT_A_NEW_FRAME` will be returned.
        This function is meant to be called frequently in the main loop of your application.

        **Parameters**
            - **rt_parameters** : a structure containing all the runtime parameters. default : a preset of :class:`~pyzed.sl.RuntimeParameters`.

        **Returns**
            - Returning :data:`~pyzed.sl.ERROR_CODE.SUCCESS` means that no problem was encountered. Returned errors can be displayed using toString(error)

        .. code-block:: python

            # Set runtime parameters after opening the camera
            runtime_param = sl.RuntimeParameters()
            runtime_param.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD # Use STANDARD sensing mode

            image = sl.Mat()
            while True :
            # Grab an image
            if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image, sl.VIEW.VIEW_LEFT) # Get the left image
        
                # Use the image for your application
        """
        if py_runtime:
            return ERROR_CODE(self.camera.grab(deref(py_runtime.runtime)))
        else:
            print("RuntimeParameters must be initialized first with RuntimeParameters().")

    def retrieve_image(self, Mat py_mat, view=VIEW.VIEW_LEFT, type=MEM.MEM_CPU, width=0,
                       height=0):
        """
        Retrieves images from the camera (or SVO file).

        Multiple images are available along with a view of various measures for display purposes.
        As an example, :data:`~pyzed.sl.VIEW.VIEW_DEPTH` can be used to get a grayscale version of the depth map, but the actual depth values can be retrieved using :meth:`~pyzed.sl.Camera.retrieve_measure()`.

        **Memory**

        By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
        If your application can use GPU images, using the type parameter can increase performance by avoiding this copy.
        If the provided :class:`~pyzed.sl.Mat` object is already allocated and matches the requested image format, memory won't be re-allocated.

        **Image size**

        By default, images are returned in the resolution provided by :meth:`~pyzed.sl.Camera.get_resolution()`.
        However, you can request custom resolutions. For example, requesting a smaller image can help you speed up your application.

        **Parameters**
            - **mat**     : [out] the :class:`~pyzed.sl.Mat` to store the image.
            - **view**    : defines the image you want (see :class:`~pyzed.sl.VIEW`). default : :data:`~pyzed.sl.VIEW.VIEW_LEFT`.
            - **type**    : whether the image should be provided in CPU or GPU memory. default : :data:`~pyzed.sl.MEM.MEM_CPU`.
            - **width**   : if specified, define the width of the output mat. If set to 0, the width of the ZED resolution will be taken. default : 0.
            - **height**  : if specified, define the height of the output mat. If set to 0, the height of the ZED resolution will be taken. default : 0.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the method succeeded, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` if an error occurred.

        .. note::
            As this function retrieves the images grabbed by the :meth:`~pyzed.sl.Camera.grab()` function, it should be called afterward.

        .. code-block:: python

            # create sl.Mat objects to store the images
            left_image = sl.Mat()
            depth_view = sl.Mat()
            while True :
                # Grab an image
                if zed.grab() == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(left_image, sl.VIEW.VIEW_LEFT) # Get the rectified left image
                zed.retrieve_image(depth_view, sl.VIEW.VIEW_DEPTH) # Get a grayscale preview of the depth map

                # Display the center pixel colors
                left_center = left_image.get_value(left_image.get_width() / 2, left_image.get_height() / 2)
                print("left_image center pixel R:", int(left_center[0]), " G:", int(left_center[1]), " B:", int(left_center[2]))

                depth_center = depth_view.get_value(depth_view.get_width() / 2, depth_view.get_height() / 2)
                print("depth_view center pixel R:", int(depth_venter[1]), " G:", int(depth_center[1]), " B:", int(depth_center[2]))
        """
        if (isinstance(view, VIEW) and isinstance(type, MEM) and isinstance(width, int) and
           isinstance(height, int)):
            return ERROR_CODE(self.camera.retrieveImage(py_mat.mat, view.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of VIEW, MEM and integer types.")

    def retrieve_measure(self, Mat py_mat, measure=MEASURE.MEASURE_DEPTH, type=MEM.MEM_CPU,
                         width=0, height=0):
        """
        Computed measures, like depth, point cloud, or normals, can be retrieved using this method.

        Multiple measures are available after a :meth:`~pyzed.sl.Camera.grab()` call. A full list is available here.

        **Memory**

        By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.

        If your application can use GPU images, using the type parameter can increase performance by avoiding this copy.

        If the provided :class:`~pyzed.sl.Mat` object is already allocated and matches the requested image format, memory won't be re-allocated.

        **Measure size**

        By default, measures are returned in the resolution provided by :meth:`~pyzed.sl.get_resolution()`.

        However, custom resolutions can be requested. For example, requesting a smaller measure can help you speed up your application.

        **Parameters**
            - **mat**     : [out] the Mat to store the measures.
            - **measure** : defines the measure you want. (see :class:`~pyzed.sl.MEASURE`), default : :data:`~pyzed.sl.MEASURE.MEASURE_DEPTH`
            - **type**    : the type of the memory of provided mat that should by used. default : :data:`~pyzed.sl.MEM.MEM_CPU`.
            - **width**   : if specified, define the width of the output mat. If set to 0, the width of the ZED resolution will be taken. default : 0
            - **height**  : if specified, define the height of the output mat. If set to 0, the height of the ZED resolution will be taken. default : 0

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the method succeeded, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` if an error occurred.

        .. note::
            As this function retrieves the measures computed by the :meth:`~pyzed.sl.Camera.grab()` function, it should be called after.

            Measures containing "RIGHT" in their names, requires :data:`~pyzed.sl.InitParameters.enable_right_side_measure` to be enabled.

        .. code-block:: python

            depth_map = sl.Mat()
            point_cloud = sl.Mat()
            x = int(zed.getResolution().width / 2) # Center coordinates
            y = int(zed.getResolution().height / 2)

            while True :
                if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image

                zed.retrieve_measure(depth_map, sl.MEASURE.MEASURE_DEPTH, sl.MEM.MEM_CPU) # Get the depth map

                # Read a depth value
                center_depth = depth_map.get_value(x, y sl.MEM.MEM_CPU) # each depth map pixel is a float value
                if isnormal(center_depth) : # + Inf is "too far", -Inf is "too close", Nan is "unknown/occlusion"
                    print("Depth value at center: ", center_depth, " ", init_params.coordinate_units)
                zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA, sl.MEM.MEM_CPU) # Get the point cloud

                # Read a point cloud value
                pc_value = point_cloud.get_value(x, y) # each point cloud pixel contains 4 floats, so we are using a numpy array

                if (isnormal(pc_value[2])) :
                    print("Point cloud coordinates at center: X=", pc_value[0], ", Y=", pc_value[1], ", Z=", pc_value[2])
        """
        if (isinstance(measure, MEASURE) and isinstance(type, MEM) and isinstance(width, int) and
           isinstance(height, int)):
            return ERROR_CODE(self.camera.retrieveMeasure(py_mat.mat, measure.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of MEASURE, MEM and integer types.")

    def set_confidence_threshold(self, int conf_treshold_value):
        """
        Sets a threshold to reject depth values based on their confidence.

        Each depth pixel has a corresponding confidence. (:data:`~pyzed.sl.MEASURE.MEASURE_CONFIDENCE`)
        A lower value means more confidence and precision (but less density). An upper value reduces filtering (more density, less certainty).

        - **set_confidence_threshold(100)** will allow values from **0** to **100**. (no filtering)
        - **setConfidenceThreshold(90)** will allow values from **10** to **100**. (filtering lowest confidence values)
        - **setConfidenceThreshold(30)** will allow values from **70** to **100**. (keeping highest confidence values and lowering the density of the depth map)

        **Parameters**
            - **conf_threshold_value**    : a value in [1,100]. 
        """
        self.camera.setConfidenceThreshold(conf_treshold_value)

    def get_confidence_threshold(self):
        """
        Returns the current confidence threshold value applied to the depth map.

        Each depth pixel has a corresponding confidence. (:data:`~pyzed.sl.MEASURE.MEASURE_CONFIDENCE`)
        This function returns the value currently used to reject unconfident depth pixels.
        By default, the confidence threshold is set at 100, meaning that no depth pixel will be rejected.

        **Returns**
            - The current threshold value between **0** and **100**.

        **See also**
            :meth:`~pyzed.sl.Camera.set_confidence_threshold()`
        """
        return self.camera.getConfidenceThreshold()

    def get_resolution(self):
        """
        Returns the size of the grabbed images.

        In live mode it matches :data:`~pyzed.sl.InitParameters.camera_resolution`.
        In SVO mode the recording resolution will be returned.
        All the default :meth:`~pyzed.sl.Camera.retrieve_image()` and :meth:`~pyzed.sl.InitParameters.retrieve_measure()` calls will generate an image matching this resolution.

        **Returns**
            - The grabbed images resolution. 
        """
        return Resolution(self.camera.getResolution().width, self.camera.getResolution().height)

    def set_depth_max_range_value(self, float depth_max_range):
        """
        Sets the maximum distance of depth estimation (All values beyond this limit will be reported as TOO_FAR).

        This method can be used to extend or reduce the depth perception range. However, the depth accuracy decreases with distance.

        **Parameters**
            - **depth_max_range** : maximum distance in the defined :class:`~pyzed.sl.UNIT`.

        .. note::
            Changing this value has no impact on performance and doesn't affect the positional tracking nor the spatial mapping. (Only the depth, point cloud, normals) 
        """
        self.camera.setDepthMaxRangeValue(depth_max_range)

    def get_depth_max_range_value(self):
        """
        Returns the current maximum distance of depth estimation.

        When estimating the depth, the SDK uses this upper limit to turn higher values into TOO_FAR ones.

        **Returns**
            - The current maximum distance that can be computed in the defined :class:`~pyzed.sl.UNIT`.

        .. note::
            This method doesn't return the highest value of the current depth map, but the highest possible one. 
        """
        return self.camera.getDepthMaxRangeValue()

    def get_depth_min_range_value(self):
        """
        Returns the closest measurable distance by the camera, according to the camera and the depth map parameters.

        When estimating the depth, the SDK uses this lower limit to turn lower values into TOO_CLOSE ones.

        **Returns**
            - The minimum distance that can be computed in the defined :class:`~pyzed.sl.UNIT`.

        .. note::
            This method doesn't return the lowest value of the current depth map, but the lowest possible one. 
        """
        return self.camera.getDepthMinRangeValue()

    def set_svo_position(self, int frame_number):
        """
        Sets the playback cursor to the desired frame number in the SVO file.

        This function allows you to move around within a played-back SVO file. After calling, the next call to :meth:`~pyzed.sl.Camera.grab()` will read the provided frame number.

        **Parameters**
            - **frame_number**    : the number of the desired frame to be decoded.

        .. note::
            Works only if the camera is open in SVO playback mode.

        .. code-block:: python

            import pyzed.sl as sl

            def main() :

                # Create a ZED camera object
                zed = sl.Camera()

                # Set configuration parameters
                init_params = sl.InitParameters()
                init_params.set_from_svo_file("path/to/my/file.svo")

                # Open the camera
                err = zed.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS :
                    print(repr(err))
                    exit(-1)

                # Loop between frame 0 and 50
                i = 0
                left_image = sl.Mat()
                while zed.get_svo_position() < zed.get_svo_number_of_frames()-1 :

                    print("Current frame: ", zed.get_svo_position())

                    # Loop if we reached frame 50
                    if zed.get_svo_position() == 50 :
                    zed.set_svo_position(0)

                    # Grab an image
                    if zed.grab() == sl.ERROR_CODE.SUCCESS :
                        zed.retrieve_image(left_image, sl.VIEW.VIEW_LEFT) # Get the rectified left image

                        # Use the image in your application

                # Close the Camera
                zed.close()
                return 0

            if __name__ == "__main__" :
                main()
        """
        self.camera.setSVOPosition(frame_number)

    def get_svo_position(self):
        """
        Returns the current playback position in the SVO file.

        The position corresponds to the number of frames already read from the SVO file, starting from 0 to n.
        Each :meth:`~pyzed.sl.Camera.grab()` call increases this value by one (except when using :data:`~pyzed.sl.InitParameters.svo_real_time_mode`).

        **Returns**
            The current frame position in the SVO file. Returns -1 if the SDK is not reading an SVO.

        .. note::
            Works only if the camera is open in SVO playback mode.
        """
        return self.camera.getSVOPosition()

    def get_svo_number_of_frames(self):
        """
        Returns the number of frames in the SVO file.

        **Returns**
            - The total number of frames in the SVO file (-1 if the SDK is not reading a SVO).

        .. note::
            Works only if the camera is open in SVO reading mode. 
        """
        return self.camera.getSVONumberOfFrames()

    def set_camera_settings(self, settings, int value, use_default=False):
        """
        Sets the value of the requested :class:`~pyzed.sl.CAMERA_SETTINGS`. (gain, brightness, hue, exposure, etc.)

        Possible values (range) of each setting are available here.

        **Parameters**
            - **settings**    : the setting to be set.
            - **value**   : the value to set.
            - **use_default** : will set default (or automatic) value if set to true. If so, **Value** parameter will be ignored. default: false.

        .. code-block:: python

            # Set the gain to 50
            zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, 50, false)

        .. warning::
            Setting :data:`~pyzed.sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE` or :data:`~pyzed.sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN` to default will automatically sets the other to default.

        .. note::
            Works only if the camera is open in live mode. 
        """
        if isinstance(settings, CAMERA_SETTINGS) and isinstance(use_default, bool):
            self.camera.setCameraSettings(settings.value, value, use_default)
        else:
            raise TypeError("Arguments must be of CAMERA_SETTINGS and boolean types.")

    def get_camera_settings(self, setting):
        """
        Returns the current value of the requested camera setting. (gain, brightness, hue, exposure, etc.)

        Possible values (range) of each setting are available here.

        **Parameters**
            **setting** : the requested setting.

        **Returns**
            The current value for the corresponding setting. Returns -1 if encounters an error.

        .. code-block:: python

            gain = zed.get_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN)
            print("Current gain value: ", gain)

        .. note::
            The returned parameters might vary between two execution due to the self-calibration being ran in the :meth:`~pyzed.sl.Camera.open()` method.
        """
        if isinstance(setting, CAMERA_SETTINGS):
            return self.camera.getCameraSettings(setting.value)
        else:
            raise TypeError("Argument is not of CAMERA_SETTINGS type.")

    def get_camera_fps(self):
        """
        Returns the framerate of the camera.

        In live mode, this value should match :data:`~pyzed.sl.InitParameters.camera_fps`.

        When playing an SVO file, this value matches the requested framerate of the recording camera.

        **Returns**
            - The framerate at wich the ZED is streaming its images (or the corresponding recorded value in SVO mode). Returns -1.f if it encounters an error.

        .. code-block:: python

            camera_fps = zed.get_camera_fps()
            print("Camera framerate: ", camera_fps)

        .. warning::
            The actual framerate (number of images retrieved per second) can be lower if the :meth:`~pyzed.sl.Camera.grab()` function runs slower than the image stream or is called too often.
        """
        return self.camera.getCameraFPS()

    def set_camera_fps(self, int desired_fps):
        """
        Sets a new target frame rate for the camera.

        When a live camera is opened, this function allows you to override the value previously set in :data:`~pyzed.sl.InitParameters.camera_fps`.

        It has no effect when playing back an SVO file.

        **Parameters**
            - **desired_fps** : the new desired frame rate.

        **Deprecated** : This function is no more supported and can cause stability issues.

        .. warning::
            This function isn't thread safe and will be removed in a later version.

            If you want to artificially reduce the camera's framerate, you can lower the frequency at which you call the :meth:`~pyzed.sl.Camera.grab()` method.

            If a not-supported framerate is requested, the closest available setting will be used.

        .. note::
            Works only if the camera is open in live mode. 
        """
        self.camera.setCameraFPS(desired_fps)

    def get_current_fps(self):
        """
        Returns the current framerate at which the :meth:`~pyzed.sl.Camera.grab()` method is successfully called.

        The returned value is based on the difference of camera timestamps between two successful :meth:`~pyzed.sl.Camera.grab()` calls.

        **Returns**
            - The current SDK framerate

        .. warning::
            The returned framerate (number of images grabbed per second) can be lower than :data:`~pyzed.sl.InitParameters.camera_fps` if the :meth:`~pyzed.sl.Camera.grab()` function runs slower than the image stream or is called too often.

        .. code-block:: python

            current_fps = zed.get_current_fps()
            print("Current framerate: ", current_fps)
        """
        return self.camera.getCurrentFPS()

    def get_camera_timestamp(self):
        """
        This function has been deprecated. Please refer to :meth:`~pyzed.sl.Camera.getTimestamp()` which has the exact same behavior.
        """
        return self.camera.getCameraTimestamp()

    def get_current_timestamp(self):
        """
        This function has been deprecated. Please refer to :meth:`~pyzed.sl.Camera.getTimestamp()` which has the exact same behavior. 
        """
        return self.camera.getCurrentTimestamp()

    def get_timestamp(self, time_reference):
        """
        Returns the timestamp in the requested :class:`~pyzed.sl.TIME_REFERENCE`.

            - When requesting the :class:`~pyzed.sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE` timestamp, the UNIX nanosecond timestamp of the latest grabbed image will be returned.
            - This value corresponds to the time at which the entire image was available in the PC memory. As such, it ignores the communication time that corresponds to 2 or 3 frame-time based on the fps (ex: 33.3ms to 50ms at 60fps).
            - When requesting the :data:`~pyzed.sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT` timestamp, the current UNIX nanosecond timestamp is returned.


        This function can also be used when playing back an SVO file.

        **Parameters**
            - reference_time  : The selected :class:`~pyzed.sl.TIME_REFERENCE`.

        **Returns**
            - The timestamp in nanosecond. 0 if not available (SVO file without compression).

        .. note::
            As this function returns UNIX timestamps, the reference it uses is common across several :class:`~pyzed.sl.Camera` instances.

            This can help to organized the grabbed images in a multi-camera application.

        .. code-block:: python

            last_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)
            current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)
            print("Latest image timestamp: ", last_image_timestamp, "ns from Epoch.")
            print("Current timestamp: ", current_timestamp, "ns from Epoch.")
        """
        if isinstance(time_reference, TIME_REFERENCE):
            return self.camera.getTimestamp(time_reference.value)
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")

    def get_frame_dropped_count(self):
        """
        Returns the number of frames dropped since :meth:`~pyzed.sl.Camera.grab()` was called for the first time.

        A dropped frame corresponds to a frame that never made it to the grab function.

        This can happen if two frames were extracted from the camera when :meth:`~pyzed.sl.Camera.grab()` is called. The older frame will be dropped so as to always use the latest (which minimizes latency).

        **Returns**
            - The number of frames dropped since the first :meth:`~pyzed.sl.Camera.grab()` call.
        """
        return self.camera.getFrameDroppedCount()

    def get_camera_information(self, resizer = Resolution(0, 0)):
        """
        Returns the calibration parameters, serial number and other information about the camera being used.

        As calibration parameters depend on the image resolution, you can provide a custom resolution as a parameter to get scaled information.

        When reading an SVO file, the parameters will correspond to the camera used for recording.

        **Parameters**
            - **image_size**  : You can specify a size different from default image size to get the scaled camera information. default = (0,0) meaning original image size (aka :meth:`~pyzed.sl.Camera.get_resolution()` ).

        **Returns**
            - :class:`~pyzed.sl.CameraInformation` containing the calibration parameters of the ZED, as well as serial number and firmware version.
        """
        return CameraInformation(self, resizer)

    def get_self_calibration_state(self):
        """
        Returns the current status of the self-calibration.

        When opening the camera, the ZED will self-calibrate itself to optimize the factory calibration.

        As this process can run slightly slower than :meth:`~pyzed.sl.Camera.open()`, this function allows you to check its status.

        The self-calibration can be disabled using :data:`~pyzed.sl.InitParameters.camera_disable_self_calib`.

        **Returns**
            - A status code giving information about the status of the self calibration. 

        **See also**
            - :class:`~pyzed.sl.SELF_CALIBRATION_STATE`

        .. note::
            The first call to the :meth:`~pyzed.sl.Camera.grab()` function will wait for the self-calibration to finish.
        """
        return SELF_CALIBRATION_STATE(self.camera.getSelfCalibrationState())

    def reset_self_calibration(self):
        """
        Resets the camera's self calibration. This function can be called at any time **after** the :meth:`~pyzed.sl.Camera.open()` function.

        It will reset and optimize the calibration parameters against misalignment, convergence, and color mismatch. It can be called if the calibration file of the current camera has been updated while the application is running.

        If the self-calibration didn't succeed, previous parameters will be used. 
        
        .. note::
            The next call to the :meth:`~pyzed.sl.Camera.grab()` function will wait for the self-calibration to finish.
        """
        self.camera.resetSelfCalibration()

    def enable_tracking(self, TrackingParameters py_tracking):
        """
        Initializes and starts the positional tracking processes.

        This function allows you to enable the position estimation of the SDK. It only has to be called once in the camera's lifetime.

        When enabled, the position will be update at each grab call.

        Tracking-specific parameter can be set by providing :class:`~pyzed.sl.TrackingParameters` to this function.

        **Parameters**
            - **py_tracking** : A structure containing all the TrackingParameters . default : a preset of :class:`~pyzed.sl.TrackingParameters`.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` if the area_file_path file wasn't found, :data:`~pyzed.sl.ERROR_CODE.SUCCESS` otherwise.

        .. warning::
            The positional tracking feature benefits from a high framerate. We found HD720@60fps to be the best compromise between image quality and framerate.

        .. code-block:: python

            import pyzed.sl as sl
            def main() :
                # --- Initialize a Camera object and open the ZED
                # Create a ZED camera object
                zed = sl.Camera()

                # Set configuration parameters
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720 # Use HD720 video mode
                init_params.camera_fps = 60 # Set fps at 60

                # Open the camera
                err = zed.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS :
                    print(repr(err))
                    exit(-1)

                # Set tracking parameters
                track_params = sl.TrackingParameters()
                track_params.enable_spatial_memory = True

                # Enable positional tracking
                err = zed.enable_tracking(track_params)
                if err != sl.ERROR_CODE.SUCCESS :
                    print("Tracking error: ", repr(err))
                    exit(-1)

                # --- Main loop
                while True :
                    if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
                        camera_pose = sl.Pose()
                        zed.get_position(camera_pose, sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD)
                        print("Camera position: X=", camera_pose.get_translation()[0], " Y=", camera_pose.get_translation()[1], " Z=", camera_pose.get_translation()[2])

                # --- Close the Camera
                zed.close()
                return 0
        """
        if py_tracking:
            return ERROR_CODE(self.camera.enableTracking(deref(py_tracking.tracking)))
        else:
            print("TrackingParameters must be initialized first with TrackingParameters().")
   
    def get_imu_data(self, IMUData py_imu_data, time_reference = TIME_REFERENCE.TIME_REFERENCE_CURRENT):
        """
        Retrieves the IMU Data at a specific time reference.

        Calling :meth:`~pyzed.sl.Camera.get_imu_data()` with :data:`~pyzed.sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT` gives you the latest IMU data received. Getting all the data requires to call this function at 800Hz in a thread.

        Calling :meth:`~pyzed.sl.Camera.get_imu_data()` with :data:`~pyzed.sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE` gives you the IMU data at the time of the latest image grabbed.

        :class:`~pyzed.sl.IMUData` object contains 2 category of data:
        Time-fused pose estimation that can be accessed using:

            - :meth:`~pyzed.sl.IMUData.get_orientation()`
            - :meth:`~pyzed.sl.IMUData.get_rotation()`
            - :meth:`~pyzed.sl.IMUData.get_rotation_vector()`
            - :meth:`~pyzed.sl.IMUData.get_euler_angles()`
            - :meth:`~pyzed.sl.IMUData.pose_data()`
            - :meth:`~pyzed.sl.IMUData.get_rotation_matrix()`

        Raw values from the IMU sensor:

            - :data:`~pyzed.sl.IMUData.angular_velocity`, corresponding to the gyroscope
            - :data:`~pyzed.sl.IMUData.linear_acceleration`, corresponding to the accelerometer

        **Parameters**
            - **imu_data**    : [out] the :class:`~pyzed.sl.IMUData` that inherits from :class:`~pyzed.sl.Pose`, containing the orientation of the IMU (pose in world reference frame) and other information (timestamp, raw imu data)
            - **reference_time**  : defines the time reference from when you want the pose to be extracted.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if :class:`~pyzed.sl.IMUData` has been filled


        Extract :class:`~pyzed.sl.Rotation` Matrix : imu_data.get_rotation()

        Extract :class:`~pyzed.sl.Orientation` / Quaternion: imu_data.get_orientation()

        .. note::
            : :class:`~pyzed.sl.Translation` is not provided when using the IMU only. 
            : The quaternion (fused data) is given in the specified :class:`~pyzed.sl.COORDINATE_SYSTEM` of :class:`~pyzed.sl.InitParameters`.

        .. warning::
            : Until v2.4, the IMU rotation (quaternion, rotation matrix, etc.) was expressed relative to the left camera frame. This means that the camera_imu_transform now given was already applied on the fused quaternion. Starting from v2.5, this transformation is given in :meth:`~pyzed.sl.Camera.get_camera_information()`. :data:`~pyzed.sl.CameraInformation.camera_imu_transform` and not applied anymore to the fused quaternion, to keep the data in the IMU frame. Therefore, to get the same values calculated in v2.4 with the ZED SDK v2.5 (and up), you will need to apply the transformation as shown in the example below :

        .. code-block:: python

            # Example to get the same quaternion between v2.4 and v2.5
            # SDK v2.4 : 
            imudata = sl.IMUData()
            zed.get_imu_data(imudata)
            quaternionOnImage = imudata.get_orientation() # quaternion is given in left camera frame

            # SDK v2.5 and up
            imudata = sl.IMUData()
            zed.get_imu_data(imudata) # quaternion ( imudata.get_orientation() ) is given in IMU frame
            cIMuMatrix = zed.get_camera_information().camera_imu_transform
            cIMuMatrix_inv = zed.inverse(cIMuMatrix);
            data_on_image = cIMuMatrix * imudata.pose_data() * cIMuMatrix_inv
            quaternionOnImage = data_on_image.get_orientation() # quaternion is given in left camera frame

        .. warning::
            : In SVO reading mode, the TIME_REFERENCE_CURRENT is currently not available (yielding :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_INVALID_FUNCTION_PARAMETERS`. Only the quaternion data at TIME_REFERENCE_IMAGE is available. Other values will be set to 0.
        """
        if isinstance(time_reference, TIME_REFERENCE):
            return ERROR_CODE(self.camera.getIMUData(py_imu_data.imuData, time_reference.value))
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")
    
    def set_imu_prior(self, Transform transfom):
        """
        Set an optionnal IMU orientation hint that will be used to assist the tracking during the next :meth:`~pyzed.sl.Camera.grab()`.

        This function can be used to assist the positional tracking rotation while using a ZED Mini.

        .. note::
            This function is only effective if a ZED Mini (ZED-M) is used.

            It needs to be called before the :meth:`~pyzed.sl.Camera.grab()` function.
        """
        return ERROR_CODE(self.camera.setIMUPrior(transfom.transform))

    def get_position(self, Pose py_pose, reference_frame = REFERENCE_FRAME.REFERENCE_FRAME_WORLD):
        """
        Retrieves the estimated position and orientation of the camera in the specified reference frame.


        Using :data:`~pyzed.sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD`, the returned pose relates to the initial position of the camera. (:data:`~pyzed.sl.TrackingParameters.initial_world_transform` )

        Using :data:`~pyzed.sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA`, the returned pose relates to the previous position of the camera.

        If the tracking has been initialized with :data:`~pyzed.sl.TrackingParameters.enable_spatial_memory` to true (default), this function can return :data:`~pyzed.sl.TRACKING_STATE.TRACKING_STATE_SEARCHING`.
        This means that the tracking lost its link to the initial referential and is currently trying to relocate the camera. However, it will keep on providing position estimations.

        **Parameters**
            - **camera_pose** [out]: the pose containing the position of the camera and other information (timestamp, confidence)
            - **reference_frame** : defines the reference from which you want the pose to be expressed. Default : :data:`~pyzed.sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD`.

        **Returns**
            - The current :class:`~pyzed.sl.TRACKING_STATE` of the tracking process.


        Extract :class:`~pyzed.sl.Rotation` Matrix : camera_pose.get_rotation();
        Extract :class:`~pyzed.sl.Translation` Vector: camera_pose.get_translation();
        Convert to :class:`~pyzed.sl.Orientation` / quaternion : camera_pose.get_orientation();

        .. note::
            The position is provided in the :data:`~pyzed.sl.InitParameters.coordinate_system` . See :class:`~pyzed.sl.COORDINATE_SYSTEM` for its physical origin. 

        .. warning::
            This function requires the tracking to be enabled. :meth:`~pyzed.sl.Camera.enable_tracking()`.

        .. code-block:: python

            # --- Main loop
            while True :
                if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
                    camera_pose = sl.Pose()
                    zed.get_position(camera_pose, sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD)

                    print("Camera position: X=", cameraPose.get_translation().[0], " Y=", camera_pose.get_translation()[1], " Z=", camera_pose.get_translation()[2])
                    print("Camera Euler rotation: X=", camera_pose.get_euler_angles()[0], " Y=", camera_pose.getEulerAngles()[1], " Z=", camera_pose.get_euler_angles()[2])
                    print("Camera Rodrigues rotation: X=", camera_pose.get_rotation_vector()[0], " Y=", camera_pose.get_rotation_vector()[1], " Z=", camera_pose.get_rotation_vector()[2])
                    print("Camera quaternion orientation: X=", camera_pose.get_orientation()[0], " Y=", camera_pose.get_orientation()[1], " Z=", camera_pose.get_orientation()[2], " W=", camera_pose.get_orientation()[3])
        """
        if isinstance(reference_frame, REFERENCE_FRAME):
            return TRACKING_STATE(self.camera.getPosition(py_pose.pose, reference_frame.value))
        else:
            raise TypeError("Argument is not of REFERENCE_FRAME type.")

    def get_area_export_state(self):
        """
        Returns the state of the spatial memory export process.

        As :meth:`~pyzed.sl.Camera.save_current_area()` only starts the exportation, this function allows you to know when the exportation finished or if it failed.

        **Returns**
            - The current :class:`~pyzed.sl.TRACKING_STATE` of the tracking process.
        """
        return AREA_EXPORT_STATE(self.camera.getAreaExportState())
   
    def save_current_area(self, str area_file_path):
        """
        Saves the current area learning file. The file will contain spatial memory data generated by the tracking.

        If the tracking has been initialized with :data:`~pyzed.sl.TrackingParameters.enable_spatial_memory` to true (default), the function allows you to export the spatial memory.

         Reloading the exported file in a future session with :data:`~pyzed.sl.TrackingParameters.area_file_path` initialize the tracking within the same referential.

        This function is asynchronous, and only triggers the file generation. You can use getAreaExportState() to get the export state. The positional tracking keeps running while exporting.

        **Parameters**
            - **area_file_path**    : save the spatial memory database in an '.area' file.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` if the area_file_path file wasn't found, :data:`~pyzed.sl.ERROR_CODE.SUCCESS` otherwise.

        **See also**
            :meth:`~pyzed.sl.Camera.get_area_export_state()`

        .. note::
            Please note that this function will also flush the area database that was built / loaded. 

        .. warning::
            If the camera wasn't moved during the tracking session, or not enough, the spatial memory won't be usable and the file won't be exported.

            The :meth:`~pyzed.sl.Camera.get_area_export_state()` function will return :data:`~pyzed.sl.ARERA_EXPORT_STATE.AREA_EXPORT_STATE_NOT_STARTED`.

            A few meters (~3m) of translation or a full rotation should be enough to get usable spatial memory.

            However, as it should be used for relocation purposes, visiting a significant portion of the environment is recommended before exporting.

        .. code-block:: python

            # --- Main loop
            while True :
                if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
                    camera_pose = sl.Pose()
                    zed.get_position(camera_pose, sl.REFERENCE_FRAME.REFERENCE_FRAME_WORLD)

                    # Export the spatial memory for future sessions
                    zed.save_current_area("office.area") # The actual file will be created asynchronously.
                    print(repr(zed.get_area_export_state()))

            # --- Close the Camera
            zed.close() # The close method will wait for the end of the file creation using get_area_export_state().
            return 0
        """
        filename = area_file_path.encode()
        return ERROR_CODE(self.camera.saveCurrentArea(String(<char*> filename)))

    def disable_tracking(self, str area_file_path=""):
        """
        Disables the positional tracking. 

        The positional tracking is immediately stopped. If a file path is given, save_current_area(area_file_path) will be called asynchronously. See :meth:`~pyzed.sl.Camera.get_area_export_state()` to get the exportation state.

        If the tracking has been enabled, this function will automatically be called by :meth:`~pyzed.sl.Camera.close()` .

        **Parameters**
            - **area_file_path** : if set, saves the spatial memory into an '.area' file. default : (empty)
                                    area_file_path is the name and path of the database, e.g. "path/to/file/myArea1.area".

        .. note::
            The '.area' database depends on the depth map SENSING_MODE chosen during the recording. The same mode must be used to reload the database. 
        """
        filename = area_file_path.encode()
        self.camera.disableTracking(String(<char*> filename))

    def reset_tracking(self, Transform path):
        """
        Resets the tracking, and re-initializes the position with the given transformation matrix.

        **Parameters**
            - **path**  : Position of the camera in the world frame when the function is called. By default, it is set to identity.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` if the area_file_path file wasn't found, :data:`~pyzed.sl.ERROR_CODE.SUCCESS` otherwise.

        .. note::
            Please note that this function will also flush the accumulated or loaded spatial memory. 
        """
        return ERROR_CODE(self.camera.resetTracking(path.transform))

    def enable_spatial_mapping(self, SpatialMappingParameters py_spatial):
        """
        Initializes and starts the spatial mapping processes.

        The spatial mapping will create a geometric representation of the scene based on both tracking data and 3D point clouds.

        The resulting output is a :class:`~pyzed.sl.Mesh` and can be obtained by the :meth:`~pyzed.sl.Camera.extract_whole_mesh()` function or with :meth:`~pyzed.sl.Camera.retrieve_mesh_async()` after calling :meth:`~pyzed.sl.Camera.request_mesh_async()`

        **Parameters**
            - **py_spatial**  : the structure containing all the specific parameters for the spatial mapping.

        Default: a balanced parameter preset between geometric fidelity and output file size. For more information, see the :class:`~pyzed.sl.SpatialMappingParameters` documentation.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if everything went fine, :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE` otherwise.

        .. warning::
            The tracking (:meth:`~pyzed.sl.Camera.enable_tracking()` ) and the depth (:data:`~pyzed.sl.RuntimeParameters.enable_depth` ) needs to be enabled to use the spatial mapping. 
            The performance greatly depends on the spatial_mapping_parameters. \ Lower :data:`~pyzed.sl.SpatialMappingParameters.range_meter` and :data:`~pyzed.sl.SpatialMappingParameters.resolution_meter` for higher performance. If the mapping framerate is too slow in live mode, consider using an SVO file, or choose a lower mesh resolution.

        .. note::
            This features uses host memory (RAM) to store the 3D map. The maximum amount of available memory allowed can be tweaked using the :class:`~pyzed.sl.SpatialMappingParameters`.

            Exeeding the maximum memory allowed immediately stops the mapping.

        .. code-block:: python

            import pyzed.sl as sl
            def main() :
                # Create a ZED camera object
                zed = sl.Camera()

                # Set initial parameters
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720 # Use HD720 video mode (default fps: 60)
                init_params.coordinate_system = sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system (The OpenGL one)
                init_params.coordinate_units = sl.UNIT.UNIT_METER # Set units in meters

                # Open the camera
                err = zed.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS :
                    exit(-1)

                # Positional tracking needs to be enabled before using spatial mapping
                tracking_parameters sl.TrackingParameters()
                err = zed.enable_tracking(tracking_parameters)
                if err != sl.ERROR_CODE.SUCCESS :
                    exit(-1)

                # Enable spatial mapping
                mapping_parameters sl.SpatialMappingParameters()
                err = zed.enable_spatial_mapping(mapping_parameters)
                if err != sl.ERROR_CODE.SUCCESS :
                    exit(-1)

                # Grab data during 500 frames
                i = 0
                mesh = sl.Mesh() # Create a mesh object
                while i < 500 :
                    # For each new grab, mesh data is updated
                    if zed.grab() == sl.ERROR_CODE.SUCCESS :
                        # In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
                        mapping_state = zed.get_spatial_mappingState()

                        # Print spatial mapping state
                        print("Images captured: ", i << " / 500  ||  Spatial mapping state: ", repr(mapping_state))
                        i = i + 1

                # Extract, filter and save the mesh in a obj file
                print("Extracting Mesh ...")
                zed.extract_whole_mesh(mesh) # Extract the whole mesh
                print("Filtering Mesh ...")
                mesh.filter(sl.MESH_FILTER.MESH_FILTER_LOW) # Filter the mesh (remove unnecessary vertices and faces)
                print("Saving Mesh in mesh.obj ...")
                mesh.save("mesh.obj") # Save the mesh in an obj file

                # Disable tracking and mapping and close the camera
                zed.disable_spatial_mapping()
                zed.disable_tracking()
                zed.close()
                return 0

            if __name__ == "__main__" :
                main()

        """
        if py_spatial:
            return ERROR_CODE(self.camera.enableSpatialMapping(deref(py_spatial.spatial)))
        else:
            print("SpatialMappingParameters must be initialized first with SpatialMappingParameters()")

    def pause_spatial_mapping(self, status):
        """
        Pause or resume the spatial mapping processes.

        As spatial mapping runs asynchronously, using this function can pause its computation to free up processing power, and resume it again later.

         For example, it can be used to avoid mapping a specifif area or to pause the mapping when the camera is static.

        **Parameters**
            - **status** : if true, the integration is paused. If false, the spatial mapping is resumed.
        """
        if isinstance(status, bool):
            self.camera.pauseSpatialMapping(status)
        else:
            raise TypeError("Argument is not of boolean type.")

    def get_spatial_mapping_state(self):
        """
        Returns the current spatial mapping state.

        As the spatial mapping runs asynchronously, this function allows you to get reported errors or status info.

        **Returns**
            - The current state of the spatial mapping process

        **See also**
            :class:`~pyzed.sl.TRACKING_STATE`
        """
        return SPATIAL_MAPPING_STATE(self.camera.getSpatialMappingState())

    def extract_whole_mesh(self, Mesh py_mesh):
        """
        Extracts the current mesh from the spatial mapping process.

        If the mesh object to be filled already contains a previous version of the mesh, only changes will be updated, optimizing performance.

        **Parameters**
            - **mesh**  : [out] The mesh to be filled.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the mesh is filled and available, otherwise :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE`.

        .. warning::
            This is a blocking function. You should either call it in a thread or at the end of the mapping process.

            Is it can be long, calling this function in the grab loop will block the depth and tracking computation and therefore gives bad results.
        """
        return ERROR_CODE(self.camera.extractWholeMesh(deref(py_mesh.mesh)))

    def request_mesh_async(self):
        """
        Starts the mesh generation process in a non blocking thread from the spatial mapping process.

        As :class:`~pyzed.sl.Mesh` generation can be take a long time depending on the mapping resolution and covered area, this function triggers the generation of a mesh without blocking the program. You can get info about the current mesh generation using :meth:`~pyzed.sl.Camera.get_mesh_request_status_async()`, and retrieve the mesh using :meth:`~pyzed.sl.Camera.retrieve_mesh_async()` .

        .. note::
            Only one mesh generation can be done at a time, consequently while the previous launch is not done every call will be ignored.

        .. code-block:: python

            zed.request_mesh_async()
            while zed.get_mesh_request_status_async() == sl.ERROR_CODE.ERROR_CODE_FAILURE :
                # Mesh is still generating

            if zed.get_mesh_request_status_async() == sl.ERROR_CODE.SUCCESS :
                zed.retrieve_mesh_async(mesh)
                print("Number of triangles in the mesh: ", mesh.get_number_of_triangles())
        """
        self.camera.requestMeshAsync()

    def get_mesh_request_status_async(self):
        """
        Returns the mesh generation status. Useful after calling :meth:`~pyzed.sl.Camera.request_mesh_async()` to know if you can call :meth:`~pyzed.sl.Camera.request_mesh_async()`.

        **Parameters**
            - **mesh**  : [out] The mesh to be filled.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the mesh is retrieved, otherwise :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE`.
        """
        return ERROR_CODE(self.camera.getMeshRequestStatusAsync())

    def retrieve_mesh_async(self, Mesh py_mesh):
        """
        Retrieves the current generated mesh.

        After calling :meth:~pyzed.sl.Camera.request_mesh_async() , this function allows you to retrieve the generated mesh. The mesh will only be available when :meth:`~pyzed.sl.Camera.get_mesh_request_status_async()` returned :data:`~pyzed.sl.ERROR_CODE.SUCCESS`

        **Parameters**
            - **mesh**  : [out] The mesh to be filled.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the mesh is retrieved, otherwise :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE`.

        .. note::
            This function only updates chunks that need to be updated and add the new ones in order to improve update speed.

        .. warning::
            You should not modify the mesh between two calls of this function, otherwise it can lead to corrupted mesh.

        See :meth:`~pyzed.sl.Camera.request_mesh_async` for an example.
        """
        return ERROR_CODE(self.camera.retrieveMeshAsync(deref(py_mesh.mesh)))

    def request_spatial_map_async(self):
        self.camera.requestSpatialMapAsync()

    def get_spatial_map_request_status_async(self):
        return ERROR_CODE(self.camera.getSpatialMapRequestStatusAsync())

    def retrieve_spatial_map_async(self, py_mesh):
        if isinstance(py_mesh, Mesh) :
            return ERROR_CODE(self.camera.retrieveSpatialMapAsync(deref((<Mesh>py_mesh).mesh)))
        elif isinstance(py_mesh, FusedPointCloud) :
            py_mesh = <FusedPointCloud> py_mesh
            return ERROR_CODE(self.camera.retrieveSpatialMapAsync(deref((<FusedPointCloud>py_mesh).fpc)))
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    def extract_whole_spatial_map(self, py_mesh):
        if isinstance(py_mesh, Mesh) :
            return ERROR_CODE(self.camera.extractWholeSpatialMap(deref((<Mesh>py_mesh).mesh)))
        elif isinstance(py_mesh, FusedPointCloud) :
            return ERROR_CODE(self.camera.extractWholeSpatialMap(deref((<FusedPointCloud>py_mesh).fpc)))
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    def find_plane_at_hit(self, coord, Plane py_plane):
        """
        Checks the plane at the given left image coordinates. 

        This function gives the 3D plane corresponding to a given pixel in the latest left image grabbed.

         The pixel coordinates are expected to be contained between 0 and getResolution().width-1 and getResolution().height-1 .

        **Parameters**
            - **coord** : [in] The image coordinate. The coordinate must be taken from the full-size image
            - **plane** : [out] The detected plane if the function succeeded

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the mesh is retrieved, otherwise :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_FAILURE`.

        .. note::
            The reference frame is defined by the :data:`~pyzed.sl.RuntimeParameters.measure3D_reference_frame` given to the :meth:`~pyzed.sl.Camera.grab()` function. 
        """
        cdef Vector2[uint] vec = Vector2[uint](coord[0], coord[1])
        return ERROR_CODE(self.camera.findPlaneAtHit(vec, py_plane.plane))

    def find_floor_plane(self, Plane py_plane, Transform resetTrackingFloorFrame, floor_height_prior = float('nan'), Rotation world_orientation_prior = Rotation(Matrix3f().zeros()), floor_height_prior_tolerance = float('nan')) :
        """
        Detect the floor plane of the scene. 

        This function analyses the latest image and depth to estimate the floor plane of the scene. 

        It expects the floor plane to be visible and bigger than other candidate planes, like a table.

        **Parameters**
            - **floorPlane** : [out] The detected floor plane if the function succeeded 
            - **resetTrackingFloorFrame** : [out] The transform to align the tracking with the floor plane. The initial position will then be at ground height, with the axis align with the gravity. The positional tracking needs to be reset/enabled with this transform as a parameter (:data:`~pyzed.sl.TrackingParameters.initial_world_transform`)
            - **floor_height_prior** : [in] Prior set to locate the floor plane depending on the known camera distance to the ground, expressed in the same unit as the ZED. If the prior is too far from the detected floor plane, the function will return :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_PLANE_NOT_FOUND`
            - **world_orientation_prior** : [in] Prior set to locate the floor plane depending on the known camera orientation to the ground. If the prior is too far from the detected floor plane, the function will return :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_PLANE_NOT_FOUND`
            - **floor_height_prior_tolerance** : [in] Prior height tolerance, absolute value.

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if the floor plane is found and matches the priors (if defined), otherwise :data:`~pyzed.sl.ERROR_CODE.ERROR_CODE_PLANE_NOT_FOUND`

        .. note::
            The reference frame is defined by the :class:`~pyzed.sl.RuntimeParameters` (:data:`~pyzed.sl.RuntimeParameters.measure3D_reference_frame`) given to the :meth:`~pyzed.sl.Camera.grab()` function. The length unit is defined by :class:`~pyzed.sl.InitParameters` (:data:`~pyzed.sl.InitParameters.coordinate_units`). With the ZED, the assumption is made that the floor plane is the dominant plane in the scene. The ZED Mini uses the gravity as prior. 
        """
        return ERROR_CODE(self.camera.findFloorPlane(py_plane.plane, resetTrackingFloorFrame.transform, floor_height_prior, world_orientation_prior.rotation, floor_height_prior_tolerance))

    def disable_spatial_mapping(self):
        """
        Disables the spatial mapping process.

        The spatial mapping is immediately stopped.

        If the mapping has been enabled, this function will automatically be called by :meth:`~pyzed.sl.Camera.close()`.

        .. note::
            This function frees the memory allocated for th spatial mapping, consequently, mesh cannot be retrieved after this call. 
        """
        self.camera.disableSpatialMapping()


    def enable_streaming(self, StreamingParameters streaming_parameters = StreamingParameters()) :
        """
        Creates an streaming pipeline for images.

        **Parameters**
            - **streaming_parameters** : the structure containing all the specific parameters for the streaming.
        """
        return ERROR_CODE(self.camera.enableStreaming(deref(streaming_parameters.streaming)))

    def disable_streaming(self):
        """
        Disables the streaming initiated by :meth:`~pyzed.sl.Camera.enable_streaming()`

        .. note::
            This function will automatically be called by :meth:`~pyzed.sl.Camera.close()` if :meth:`~pyzed.sl.Camera.enable_streaming()` was called.
        """
        self.camera.disableStreaming()

    def is_streaming_enabled(self):
        """
        Tells if the streaming is actually sending data (true) or still in configuration (false)
        """
        return self.camera.isStreamingEnabled()


    def enable_recording(self, str video_filename,
                          compression_mode=SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS):
        """
        Creates an SVO file to be filled by :meth:`~pyzed.sl.Camera.record()`.

        This function can create AVI or SVO files based on the provided filename.

        SVO files are custom video files containing the unrectified images from the camera along with some metadata like timestamps or IMU orientation (if applicable).

        They can be used to simulate a live ZED and test a sequence with various SDK parameters.

        Depending on the application, various compression modes are available. See :data:`~pyzed.sl.SVO_COMPRESSION.SVO_COMPRESSION_MODE`.

        **Parameters**
            - **video_filename** : can be a *.svo file or a *.avi file (detected by the suffix name provided).
            - **compression_mode** : can be one of the :class:`~pyzed.sl.SVO_COMPRESSION_MODE` enum. default : :data:`~pyzed.sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS`.

        **Returns**
            - an :class:`~pyzed.sl.ERROR_CODE` that defines if file was successfully created and can be filled with images.

        .. warning::
            This function can be called multiple times during ZED lifetime, but if video_filename is already existing, the file will be erased. 

        .. note::
            :data:`~pyzed.sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_RAW` is deprecated in recording mode.

        .. code-block:: python

            import pyzed.sl as sl
            def main() :
                # Create a ZED camera object
                zed = sl.Camera()
                # Set initial parameters
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720 # Use HD720 video mode (default fps: 60)
                init_params.coordinate_units = sl.UNIT.UNIT_METER # Set units in meters
                # Open the camera
                err = zed.open(init_params)
                if (err != sl.ERROR_CODE.SUCCESS) :
                    print(repr(err))
                    exit(-1)

                # Enable video recording
                err = zed.enable_recording("myVideoFile.svo", sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS)
                if (err != sl.ERROR_CODE.SUCCESS) :
                    print(repr(err))
                    exit(-1)

                # Grab data during 500 frames
                i = 0
                while i < 500 :
                    # Grab a new frame
                    if zed.grab() == sl.ERROR_CODE.SUCCESS :
                        # Record the grabbed frame in the video file
                        zed.record()
                        i = i + 1

                zed.disable_recording()
                print("Video has been saved ...")
                zed.close()
                return 0

            if __name__ == "__main__" :
                main()
        """
        if isinstance(compression_mode, SVO_COMPRESSION_MODE):
            filename = video_filename.encode()
            return ERROR_CODE(self.camera.enableRecording(String(<char*> filename),
                                      compression_mode.value))
        else:
            raise TypeError("Argument is not of SVO_COMPRESSION_MODE type.")

    def record(self):
        """
        Records the current frame provided by :meth:`~pyzed.sl.Camera.grab()` into the file.

        Calling this function after a successful :meth:`~pyzed.sl.Camera.grab()` call saves the images into the video file opened by :meth:`~pyzed.sl.Camera.enable_recording()` .

        **Returns**
            - The recording state structure. For more details, see :class:`~pyzed.sl.RecordingState`

        .. warning::
            The latest grabbed frame will be save, so :meth:`~pyzed.sl.Camera.grab()` must be called before.

        See :meth:`~pyzed.sl.Camera.enable_recording()` for an example.
        """
        return self.camera.record()

    def disable_recording(self):
        """
        Disables the recording initiated by :meth:`~pyzed.sl.Camera.enable_recording()` and closes the generated file.

        .. note::
            This function will automatically be called by :meth:`~pyzed.sl.Camera.close()` if :meth:`~pyzed.sl.Camera.enable_recording()` was called.

        See :meth:`~pyzed.sl.Camera.enable_recording()` for an example.
        """
        self.camera.disableRecording()

    def get_sdk_version(cls):
        """
        Returns the version of the currently installed ZED SDK. 
        """
        return cls.camera.getSDKVersion().get().decode()

    def is_zed_connected(cls):
        """
        Returns the number of connected cameras.

        **Returns**
            - The number of connected cameras supported by the SDK. See (:class:`~pyzed.sl.MODEL`)

        .. warning::
            This function has been deprecated in favor of :meth:`~pyzed.sl.Camera.get_device_list()` , which returns more info about the connected devices. 
        """
        return cls.camera.isZEDconnected()

    def stickto_cpu_core(cls, int cpu_core):
        """
        **Only for Nvidia Jetson**: Sticks the calling thread to a specific CPU core.

        **Parameters**
            - **cpu_core** : int that defines the core the thread must be run on. Can be between 0 and 3. (cpu0,cpu1,cpu2,cpu3).

        **Returns**
            - :data:`~pyzed.sl.ERROR_CODE.SUCCESS` if stick is OK, otherwise returns a status error.

        .. warning::
            Function only available for Nvidia Jetson. On other platforms, the result will always be 0 and no operations will be performed. 
        """
        return ERROR_CODE(cls.camera.sticktoCPUCore(cpu_core))

    def get_device_list(cls):
        """
        List all the connected devices with their associated information.

        This function lists all the cameras available and provides their serial number, models and other information.

        **Returns**
            - The device properties for each connected camera
        """
        vect_ = cls.camera.getDeviceList()
        vect_python = []
        for i in range(vect_.size()):
            prop = DeviceProperties()
            prop.camera_state = vect_[i].camera_state
            prop.id = vect_[i].id
            prop.path = vect_[i].path.get().decode()
            prop.camera_model = vect_[i].camera_model
            prop.serial_number = vect_[i].serial_number
            vect_python.append(prop)
        return vect_python

    def get_streaming_device_list(cls):
        """
        List all the streaming devices with their associated information.

        **Returns**
            - The streaming properties for each connected camera

        .. warning::
            As this function returns an std::vector, it is only safe to use in Release mode (not Debug).

            This is due to a known compatibility issue between release (the SDK) and debug (your app) implementations of std::vector.

        .. warning::
            This function takes around 2seconds to make sure all network informations has been captured. Make sure to run this function in a thread.
        """
        vect_ = cls.camera.getStreamingDeviceList()
        vect_python = []
        for i in range(vect_.size()):
            prop = StreamingProperties()
            prop.ip = to_str(vect_[i].ip)
            prop.port = vect_[i].port
            prop.serial_number = vect_[i].serial_number
            prop.current_bitrate = vect_[i].current_bitrate
            prop.codec = vect_[i].codec
            vect_python.append(prop)
        return vect_python


def save_camera_depth_as(Camera zed, format, str name, factor=1):
    """
    Writes the current depth map into a file.

    **Parameters**
        - **zed** : the current camera object.
        - **format**  : the depth file format you desired.
        - **name**    : the name (path) in which the depth will be saved.
        - **factor**  : only for PNG and PGM, apply a gain to the depth value. default : 1. The PNG format only stores integers between 0 and 65536, if you're saving a depth map in meters, values will be rounded to the nearest integer. Set factor to 0.01 to reduce this effect by converting to millimeters. Do not forget to scale (by 1./factor) the pixel value at reading to get the real depth. The occlusions are represented by 0.

    **Returns**
        - False if something wrong happens, else return true. 

    Referenced by :meth:`~pyzed.sl.Camera.is_opened()`.
    """
    if isinstance(format, DEPTH_FORMAT) and factor <= 65536:
        name_save = name.encode()
        return saveDepthAs(zed.camera, format.value, String(<char*>name_save), factor)
    else:
        raise TypeError("Arguments must be of DEPTH_FORMAT type and factor not over 65536.")

def save_camera_point_cloud_as(Camera zed, format, str name, with_color=False):
    """
    Writes the current point cloud into a file.

    **Parameters**
        - **zed** : the current camera object.
        - **format**  : the point cloud file format you desired.
        - **name**    : the name (path) in which the point cloud will be saved.
        - **with_color** : indicates if the color must be saved. default : false.

    **Returns**
        - False if something wrong happens, else return true.

    .. note::
        The color is not saved for XYZ and VTK files. 

    Referenced by :meth:`~pyzed.sl.Camera.is_opened`

    """
    if isinstance(format, POINT_CLOUD_FORMAT):
        name_save = name.encode()
        return savePointCloudAs(zed.camera, format.value, String(<char*>name_save),
                                with_color)
    else:
        raise TypeError("Argument is not of POINT_CLOUD_FORMAT type.")

def save_mat_depth_as(Mat py_mat, format, str name, factor=1):
    if isinstance(format, DEPTH_FORMAT) and factor <= 65536:
        name_save = name.encode()
        return saveMatDepthAs(py_mat.mat, format.value, String(<char*>name_save), factor)
    else:
        raise TypeError("Arguments must be of DEPTH_FORMAT type and factor not over 65536.")


def save_mat_point_cloud_as(Mat py_mat, format, str name, with_color=False):
    if isinstance(format, POINT_CLOUD_FORMAT):
        name_save = name.encode()
        return saveMatPointCloudAs(py_mat.mat, format.value, String(<char*>name_save),
                                with_color)
    else:
        raise TypeError("Argument is not of POINT_CLOUD_FORMAT type.")
