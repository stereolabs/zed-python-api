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
from sl_c cimport to_str, ERROR_CODE as c_ERROR_CODE, toString, sleep_ms, MODEL as c_MODEL, model2str, CAMERA_STATE as c_CAMERA_STATE, String, DeviceProperties as c_DeviceProperties, Vector2, Vector3, Vector4, Matrix3f as c_Matrix3f, Matrix4f as c_Matrix4f, UNIT as c_UNIT, COORDINATE_SYSTEM as c_COORDINATE_SYSTEM, RESOLUTION as c_RESOLUTION, CAMERA_SETTINGS as c_CAMERA_SETTINGS, SELF_CALIBRATION_STATE as c_SELF_CALIBRATION_STATE, DEPTH_MODE as c_DEPTH_MODE, SENSING_MODE as c_SENSING_MODE, MEASURE as c_MEASURE, VIEW as c_VIEW, TIME_REFERENCE as c_TIME_REFERENCE, DEPTH_FORMAT as c_DEPTH_FORMAT, POINT_CLOUD_FORMAT as c_POINT_CLOUD_FORMAT, TRACKING_STATE as c_TRACKING_STATE, AREA_EXPORT_STATE as c_AREA_EXPORT_STATE, REFERENCE_FRAME as c_REFERENCE_FRAME, SPATIAL_MAPPING_STATE as c_SPATIAL_MAPPING_STATE, SVO_COMPRESSION_MODE as c_SVO_COMPRESSION_MODE, RecordingState, cameraResolution, resolution2str, statusCode2str, str2mode, depthMode2str, sensingMode2str, unit2str, str2unit, trackingState2str, spatialMappingState2str, getCurrentTimeStamp, Resolution as c_Resolution, CameraParameters as c_CameraParameters, CalibrationParameters as c_CalibrationParameters, CameraInformation as c_CameraInformation, MEM as c_MEM, COPY_TYPE as c_COPY_TYPE, MAT_TYPE as c_MAT_TYPE, Mat as c_Mat, Rotation as c_Rotation, Translation as c_Translation, Orientation as c_Orientation, Transform as c_Transform, uchar1, uchar2, uchar3, uchar4, float1, float2, float3, float4, matResolution, setToUchar1, setToUchar2, setToUchar3, setToUchar4, setToFloat1, setToFloat2, setToFloat3, setToFloat4, setValueUchar1, setValueUchar2, setValueUchar3, setValueUchar4, setValueFloat1, setValueFloat2, setValueFloat3, setValueFloat4, getValueUchar1, getValueUchar2, getValueUchar3, getValueUchar4, getValueFloat1, getValueFloat2, getValueFloat3, getValueFloat4, getPointerUchar1, getPointerUchar2, getPointerUchar3, getPointerUchar4, getPointerFloat1, getPointerFloat2, getPointerFloat3, getPointerFloat4, uint, MESH_FILE_FORMAT as c_MESH_FILE_FORMAT, MESH_TEXTURE_FORMAT as c_MESH_TEXTURE_FORMAT, MESH_FILTER as c_MESH_FILTER, PLANE_TYPE as c_PLANE_TYPE, MeshFilterParameters as c_MeshFilterParameters, Texture as c_Texture, Chunk as c_Chunk, Mesh as c_Mesh, Plane as c_Plane, CUctx_st, CUcontext, MAPPING_RESOLUTION as c_MAPPING_RESOLUTION, MAPPING_RANGE as c_MAPPING_RANGE, InputType as c_InputType, InitParameters as c_InitParameters, RuntimeParameters as c_RuntimeParameters, TrackingParameters as c_TrackingParameters, SpatialMappingParameters as c_SpatialMappingParameters, Pose as c_Pose, IMUData as c_IMUData, Camera as c_Camera, saveDepthAs, savePointCloudAs, saveMatDepthAs, saveMatPointCloudAs

from cython.operator cimport dereference as deref
from libc.string cimport memcpy
from cpython cimport bool

import enum

import numpy as np
cimport numpy as np
from math import sqrt

class ERROR_CODE(enum.Enum):
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
    sleep_ms(time)

cdef class DeviceProperties:
    cdef c_DeviceProperties c_device_properties

    def __cinit__(self):
        self.c_device_properties = c_DeviceProperties()

    @property
    def camera_state(self):
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
        return self.c_device_properties.id
    @id.setter
    def id(self, id):
        self.c_device_properties.id = id

    @property
    def path(self):
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
        return self.c_device_properties.serial_number
    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_device_properties.serial_number = serial_number

    def __str__(self):
        return to_str(toString(self.c_device_properties)).decode()

    def __repr__(self):
        return to_str(toString(self.c_device_properties)).decode()


cdef class Matrix3f:
    cdef c_Matrix3f mat
    def __cinit__(self):
        self.mat = c_Matrix3f()

    def init_matrix(self, Matrix3f matrix):
        self.mat = c_Matrix3f(matrix.mat)

    def inverse(self):
        self.mat.inverse()

    def inverse_mat(self, Matrix3f rotation):
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        self.mat.transpose()

    def transpose(self, Matrix3f rotation):
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
    cdef c_Matrix4f mat
    def __cinit__(self):
        self.mat = c_Matrix4f()

    def init_matrix(self, Matrix4f matrix):
        self.mat = c_Matrix4f(matrix.mat)

    def inverse(self):
        return ERROR_CODE(self.mat.inverse())

    def inverse_mat(self, Matrix4f rotation):
        rotation.mat.inverse(rotation.mat)
        return rotation

    def transpose(self):
        self.mat.transpose()

    def transpose(self, Matrix4f rotation):
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

    def set_sub_matrix3f(self, Matrix3f input, row=0, column=0):
        if row != 0 and row != 1 or column != 0 and column != 1:
            raise TypeError("Arguments row and column must be 0 or 1.")
        else:
            return ERROR_CODE(self.mat.setSubMatrix3f(input.mat, row, column))

    def set_sub_vector3f(self, float input0, float input1, float input2, column=3):
        return ERROR_CODE(self.mat.setSubVector3f(Vector3[float](input0, input1, input2), column))

    def set_sub_vector4f(self, float input0, float input1, float input2, float input3, column=3):
        return ERROR_CODE(self.mat.setSubVector4f(Vector4[float](input0, input1, input2, input3), column))

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
    CAMERA_SETTINGS_BRIGHTNESS = c_CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
    CAMERA_SETTINGS_CONTRAST = c_CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST
    CAMERA_SETTINGS_HUE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_HUE
    CAMERA_SETTINGS_SATURATION = c_CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION
    CAMERA_SETTINGS_GAIN = c_CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN
    CAMERA_SETTINGS_EXPOSURE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE
    CAMERA_SETTINGS_WHITEBALANCE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE
    CAMERA_SETTINGS_AUTO_WHITEBALANCE = c_CAMERA_SETTINGS.CAMERA_SETTINGS_AUTO_WHITEBALANCE
    CAMERA_SETTINGS_LAST = c_CAMERA_SETTINGS.CAMERA_SETTINGS_LAST


class SELF_CALIBRATION_STATE(enum.Enum):
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
    DEPTH_FORMAT_PNG = c_DEPTH_FORMAT.DEPTH_FORMAT_PNG
    DEPTH_FORMAT_PFM = c_DEPTH_FORMAT.DEPTH_FORMAT_PFM
    DEPTH_FORMAT_PGM = c_DEPTH_FORMAT.DEPTH_FORMAT_PGM
    DEPTH_FORMAT_LAST = c_DEPTH_FORMAT.DEPTH_FORMAT_LAST

    def __str__(self):
        return to_str(toString(<c_DEPTH_FORMAT>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_DEPTH_FORMAT>self.value)).decode()

class POINT_CLOUD_FORMAT(enum.Enum):
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
    REFERENCE_FRAME_WORLD = c_REFERENCE_FRAME.REFERENCE_FRAME_WORLD
    REFERENCE_FRAME_CAMERA = c_REFERENCE_FRAME.REFERENCE_FRAME_CAMERA
    REFERENCE_FRAME_LAST = c_REFERENCE_FRAME.REFERENCE_FRAME_LAST

    def __str__(self):
        return to_str(toString(<c_REFERENCE_FRAME>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_REFERENCE_FRAME>self.value)).decode()

class TIME_REFERENCE(enum.Enum):
    TIME_REFERENCE_IMAGE = c_TIME_REFERENCE.TIME_REFERENCE_IMAGE
    TIME_REFERENCE_CURRENT = c_TIME_REFERENCE.TIME_REFERENCE_CURRENT
    TIME_REFERENCE_LAST = c_TIME_REFERENCE.TIME_REFERENCE_LAST

    def __str__(self):
        return to_str(toString(<c_TIME_REFERENCE>self.value)).decode()

    def __repr__(self):
        return to_str(toString(<c_TIME_REFERENCE>self.value)).decode()

class SPATIAL_MAPPING_STATE(enum.Enum):
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
    MEM_CPU = c_MEM.MEM_CPU
    MEM_GPU = c_MEM.MEM_GPU


class COPY_TYPE(enum.Enum):
    COPY_TYPE_CPU_CPU = c_COPY_TYPE.COPY_TYPE_CPU_CPU
    COPY_TYPE_CPU_GPU = c_COPY_TYPE.COPY_TYPE_CPU_GPU
    COPY_TYPE_GPU_GPU = c_COPY_TYPE.COPY_TYPE_GPU_GPU
    COPY_TYPE_GPU_CPU = c_COPY_TYPE.COPY_TYPE_GPU_CPU


class MAT_TYPE(enum.Enum):
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

    def __richcmp__(Resolution left, Resolution right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height
        if op == 3:
            return left.width!=right.width or left.height!=right.height
        else:
            raise NotImplementedError()


cdef class CameraParameters:
    cdef c_CameraParameters camera_params
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


cdef class CalibrationParameters:
    cdef c_CalibrationParameters calibration
    cdef CameraParameters py_left_cam
    cdef CameraParameters py_right_cam
    cdef Vector3[float] R
    cdef Vector3[float] T

    def __cinit__(self):
        self.py_left_cam = CameraParameters()
        self.py_right_cam = CameraParameters()

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


cdef class CameraInformation:
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
        return self.camera_model

    @property
    def calibration_parameters(self):
        return self.py_calib

    @property
    def calibration_parameters_raw(self):
        return self.py_calib_raw

    @property
    def camera_imu_transform(self):
        return self.py_camera_imu_transform

    @property
    def serial_number(self):
        return self.serial_number

    @property
    def firmware_version(self):
        return self.firmware_version


cdef class Mat:
    cdef c_Mat mat
    def __cinit__(self):
        self.mat = c_Mat()

    def init_mat_type(self, width, height, mat_type, memory_type=MEM.MEM_CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat = c_Mat(width, height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_cpu(self, width, height, mat_type, ptr, step, memory_type=MEM.MEM_CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat = c_Mat(width, height, mat_type.value, ptr.encode(), step, memory_type.value)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_cpu_gpu(self, width, height, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        if isinstance(mat_type, MAT_TYPE):
             self.mat = c_Mat(width, height, mat_type.value, ptr_cpu.encode(), step_cpu, ptr_gpu.encode(), step_gpu)
        else:
            raise TypeError("Argument is not of MAT_TYPE type.")

    def init_mat_resolution(self, Resolution resolution, mat_type, memory_type):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat = c_Mat(c_Resolution(resolution.width, resolution.height), mat_type.value, memory_type.value)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_resolution_cpu(self, Resolution resolution, mat_type, ptr, step, memory_type=MEM.MEM_CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat = c_Mat(c_Resolution(resolution.width, resolution.height), mat_type.value, ptr.encode(),
                            step, memory_type.value)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    def init_mat_resolution_cpu_gpu(self, Resolution resolution, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu):
        if isinstance(mat_type, MAT_TYPE):
             self.mat = matResolution(c_Resolution(resolution.width, resolution.height), mat_type.value, ptr_cpu.encode(),
                             step_cpu, ptr_gpu.encode(), step_gpu)
        else:
            raise TypeError("Argument is not of MAT_TYPE type.")

    def init_mat(self, Mat matrix):
        self.mat = c_Mat(matrix.mat)

    def alloc_size(self, width, height, mat_type, memory_type=MEM.MEM_CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(<size_t> width, <size_t> height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    def alloc_resolution(self, Resolution resolution, mat_type, memory_type=MEM.MEM_CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(resolution.width, resolution.height, mat_type.value, memory_type.value)
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    def free(self, memory_type=None):
        if isinstance(memory_type, MEM):
            self.mat.free(memory_type.value)
        elif memory_type is None:
            self.mat.free(MEM.MEM_CPU or MEM.MEM_GPU)
        else:
            raise TypeError("Argument is not of MEM type.")

    def update_cpu_from_gpu(self):
        return ERROR_CODE(self.mat.updateCPUfromGPU())

    def update_gpu_from_cpu(self):
        return ERROR_CODE(self.mat.updateGPUfromCPU())

    def copy_to(self, cpy_type=COPY_TYPE.COPY_TYPE_CPU_CPU):
        dst = Mat()
        print(ERROR_CODE(self.mat.copyTo(dst.mat, cpy_type.value)))
        return dst

    def set_from(self, cpy_type=COPY_TYPE.COPY_TYPE_CPU_CPU):
        dst = Mat()
        print(self.mat.setFrom(dst.mat, cpy_type.value))
        return dst

    def read(self, str filepath):
        return ERROR_CODE(self.mat.read(filepath.encode()))

    def write(self, str filepath):
        return ERROR_CODE(self.mat.write(filepath.encode()))

    def set_to(self, value, memory_type=MEM.MEM_CPU):
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
        return self.mat.getWidth()

    def get_height(self):
        return self.mat.getHeight()

    def get_resolution(self):
        return Resolution(self.mat.getResolution().width, self.mat.getResolution().height)

    def get_channels(self):
        return self.mat.getChannels()

    def get_data_type(self):
        return MAT_TYPE(self.mat.getDataType())

    def get_memory_type(self):
        return MEM(self.mat.getMemoryType())

    def get_data(self, memory_type=MEM.MEM_CPU):
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
        if type(memory_type) == MEM:
            return self.mat.getStepBytes(memory_type.value)
        else:
            raise TypeError("Argument is not of MEM type.")

    def get_step(self, memory_type=MEM.MEM_CPU):
        if type(memory_type) == MEM:
            return self.mat.getStep(memory_type.value)
        else:
            raise TypeError("Argument is not of MEM type.")

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

    def clone(self, Mat py_mat):
        return ERROR_CODE(self.mat.clone(py_mat.mat))

    def move(self, Mat py_mat):
        return ERROR_CODE(self.mat.move(py_mat.mat))

    @staticmethod
    def swap(self, Mat mat1, Mat mat2):
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


cdef class Rotation(Matrix3f):
    cdef c_Rotation rotation
    def __cinit__(self):
        self.rotation = c_Rotation()

    def init_rotation(self, Rotation rot):
        self.rotation = c_Rotation(rot.rotation)
        self.mat = rot.mat

    def init_matrix(self, Matrix3f matrix):
        self.rotation = c_Rotation(matrix.mat)
        self.mat = matrix.mat

    def init_orientation(self, Orientation orient):
        self.rotation = c_Rotation(orient.orientation)
        self.mat = c_Matrix3f(self.rotation.r)

    def init_angle_translation(self, float angle, Translation axis):
        self.rotation = c_Rotation(angle, axis.translation)
        self.mat = c_Matrix3f(self.rotation.r)

    def set_orientation(self, Orientation py_orientation):
        self.rotation.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        py_orientation = Orientation()
        py_orientation.orientation = self.rotation.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.rotation.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        self.rotation.setRotationVector(Vector3[float](input0, input1, input2))

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
            self.rotation.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")

cdef class Translation:
    cdef c_Translation translation
    def __cinit__(self):
        self.translation = c_Translation()

    def init_translation(self, Translation tr):
        self.translation = c_Translation(tr.translation)

    def init_vector(self, float t1, float t2, float t3):
        self.translation = c_Translation(t1, t2, t3)

    def normalize(self):
        self.translation.normalize()

    def normalize_translation(self, Translation tr):
        py_translation = Translation()
        py_translation.translation = self.translation.normalize(tr.translation)
        return py_translation

    def size(self):
        return self.translation.size()

    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.translation(i)
        return arr

    def __mul__(Translation self, Orientation other):
        tr = Translation()
        tr.translation = self.translation * other.orientation
        return tr


cdef class Orientation:
    cdef c_Orientation orientation
    def __cinit__(self):
        self.orientation = c_Orientation()

    def init_orientation(self, Orientation orient):
        self.orientation = c_Orientation(orient.orientation)

    def init_vector(self, float v0, float v1, float v2, float v3):
        self.orientation = c_Orientation(Vector4[float](v0, v1, v2, v3))

    def init_rotation(self, Rotation rotation):
        self.orientation = c_Orientation(rotation.rotation)

    def init_translation(self, Translation tr1, Translation tr2):
        self.orientation = c_Orientation(tr1.translation, tr2.translation)

    def set_rotation_matrix(self, Rotation py_rotation):
       self.orientation.setRotationMatrix(py_rotation.rotation)

    def get_rotation_matrix(self):
        py_rotation = Rotation()
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
    def normalize_orientation(Orientation orient):
        orient.orientation.normalise()
        return orient

    def size(self):
        return self.orientation.size()

    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.orientation(i)
        return arr

    def __mul__(Orientation self, Orientation other):
        orient = Orientation()
        orient.orientation = self.orientation * other.orientation
        return orient


cdef class Transform(Matrix4f):
    cdef c_Transform transform
    def __cinit__(self):
        self.transform = c_Transform()

    def init_transform(self, Transform motion):
        self.transform = c_Transform(motion.transform)
        self.mat = motion.mat

    def init_matrix(self, Matrix4f matrix):
        self.transform = c_Transform(matrix.mat)
        self.mat = matrix.mat

    def init_rotation_translation(self, Rotation rot, Translation tr):
        self.transform = c_Transform(rot.rotation, tr.translation)
        self.mat = c_Matrix4f(self.transform.m)

    def init_orientation_translation(self, Orientation orient, Translation tr):
        self.transform = c_Transform(orient.orientation, tr.translation)
        self.mat = c_Matrix4f(self.transform.m)

    def set_rotation_matrix(self, Rotation py_rotation):
        self.transform.setRotationMatrix(<c_Rotation>py_rotation.mat)

    def get_rotation_matrix(self):
        py_rotation = Rotation()
        py_rotation.mat = self.transform.getRotationMatrix()
        return py_rotation

    def set_translation(self, Translation py_translation):
        self.transform.setTranslation(py_translation.translation)

    def get_translation(self):
        py_translation = Translation()
        py_translation.translation = self.transform.getTranslation()
        return py_translation

    def set_orientation(self, Orientation py_orientation):
        self.transform.setOrientation(py_orientation.orientation)

    def get_orientation(self):
        py_orientation = Orientation()
        py_orientation.orientation = self.transform.getOrientation()
        return py_orientation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.transform.getRotationVector()[i]
        return arr

    def set_rotation_vector(self, float input0, float input1, float input2):
        self.transform.setRotationVector(Vector3[float](input0, input1, input2))

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
            self.transform.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")



class MESH_FILE_FORMAT(enum.Enum):
    MESH_FILE_PLY = c_MESH_FILE_FORMAT.MESH_FILE_PLY
    MESH_FILE_PLY_BIN = c_MESH_FILE_FORMAT.MESH_FILE_PLY_BIN
    MESH_FILE_OBJ = c_MESH_FILE_FORMAT.MESH_FILE_OBJ
    MESH_FILE_LAST = c_MESH_FILE_FORMAT.MESH_FILE_LAST

class MESH_TEXTURE_FORMAT(enum.Enum):
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
    cdef c_MeshFilterParameters* meshFilter
    def __cinit__(self):
        self.meshFilter = new c_MeshFilterParameters(c_MESH_FILTER.MESH_FILTER_LOW)

    def __dealloc__(self):
        del self.meshFilter

    def set(self, filter=MESH_FILTER.MESH_FILTER_LOW):
        if isinstance(filter, MESH_FILTER):
            self.meshFilter.set(filter.value)
        else:
            raise TypeError("Argument is not of MESH_FILTER type.")

    def save(self, str filename):
        filename_save = filename.encode()
        return self.meshFilter.save(String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.meshFilter.load(String(<char*> filename_load))


cdef class Texture:
    cdef c_Texture texture
    def __cinit__(self):
        self.texture = c_Texture()

    @property
    def name(self):
        if not self.texture.name.empty():
            return self.texture.name.get().decode()
        else:
            return ""

    def get_data(self, Mat py_mat):
       py_mat.mat = self.texture.data
       return py_mat

    @property
    def indice_gl(self):
        return self.texture.indice_gl

    def clear(self):
        self.texture.clear()


cdef class Chunk:
    cdef c_Chunk chunk
    def __cinit__(self):
        self.chunk = c_Chunk()

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

cdef class Mesh:
    cdef c_Mesh* mesh
    def __cinit__(self):
        self.mesh = new c_Mesh()

    def __dealloc__(self):
        del self.mesh

    @property
    def chunks(self):
        list = []
        for i in range(self.mesh.chunks.size()):
            py_chunk = Chunk()
            py_chunk.chunk = self.mesh.chunks[i]
            list.append(py_chunk)
        return list

    def __getitem__(self, x):
        return self.chunks[x]

    def filter(self, MeshFilterParameters params, update_mesh=True):
        if isinstance(update_mesh, bool):
            return self.mesh.filter(deref(params.meshFilter), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    def apply_texture(self, texture_format=MESH_TEXTURE_FORMAT.MESH_TEXTURE_RGB):
        if isinstance(texture_format, MESH_TEXTURE_FORMAT):
            return self.mesh.applyTexture(texture_format.value)
        else:
            raise TypeError("Argument is not of MESH_TEXTURE_FORMAT type.")

    def save(self, str filename, typeMesh=MESH_FILE_FORMAT.MESH_FILE_OBJ, id=[]):
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.mesh.save(String(filename.encode()), typeMesh.value, id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    def load(self, str filename, update_mesh=True):
        if isinstance(update_mesh, bool):
            return self.mesh.load(String(filename.encode()), update_mesh)
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
        py_texture = Texture()
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

    def get_visible_list(self, Transform camera_pose):
        return self.mesh.getVisibleList(camera_pose.transform)

    def get_surrounding_list(self, Transform camera_pose, float radius):
        return self.mesh.getSurroundingList(camera_pose.transform, radius)

    def update_mesh_from_chunklist(self, id=[]):
        self.mesh.updateMeshFromChunkList(id)

cdef class Plane:
    cdef c_Plane plane
    def __cinit__(self):
        self.plane = c_Plane()

    @property
    def type(self):
        return self.plane.type

    def get_normal(self):
        normal = self.plane.getNormal()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = normal[i]
        return arr

    def get_center(self):
        center = self.plane.getCenter()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = center[i]
        return arr

    def get_pose(self, Transform py_pose):
        py_pose.transform = self.plane.getPose()
        return py_pose

    def get_extents(self):
        extents = self.plane.getExtents()
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = extents[i]
        return arr

    def get_plane_equation(self):
        plane_eq = self.plane.getPlaneEquation()
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = plane_eq[i]
        return arr

    def get_bounds(self):
        cdef np.ndarray arr = np.zeros((self.plane.getBounds().size(), 3))
        for i in range(self.plane.getBounds().size()):
            for j in range(3):
                arr[i,j] = self.plane.getBounds()[i].ptr()[j]
        return arr

    def extract_mesh(self):
        ext_mesh = self.plane.extractMesh()
        pymesh = Mesh()
        pymesh.mesh[0] = ext_mesh
        return pymesh

    def get_closest_distance(self, point=[0,0,0]):
        cdef Vector3[float] vec = Vector3[float](point[0], point[1], point[2])
        return self.plane.getClosestDistance(vec)




class MAPPING_RESOLUTION(enum.Enum):
    MAPPING_RESOLUTION_HIGH = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH
    MAPPING_RESOLUTION_MEDIUM  = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_MEDIUM
    MAPPING_RESOLUTION_LOW = c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_LOW


class MAPPING_RANGE(enum.Enum):
    MAPPING_RANGE_NEAR = c_MAPPING_RANGE.MAPPING_RANGE_NEAR
    MAPPING_RANGE_MEDIUM = c_MAPPING_RANGE.MAPPING_RANGE_MEDIUM
    MAPPING_RANGE_FAR = c_MAPPING_RANGE.MAPPING_RANGE_FAR

cdef class InputType:
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
        self.input.setFromCameraID(id)

    def set_from_serial_number(self, serial_number):
        self.input.setFromSerialNumber(serial_number)

    def set_from_svo_file(self, str svo_input_filename):
        filename = svo_input_filename.encode()
        self.input.setFromSVOFile(String(<char*> filename))
 

cdef class InitParameters:
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
        filename_save = filename.encode()
        return self.init.save(String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.init.load(String(<char*> filename_load))

    @property
    def camera_resolution(self):
        return RESOLUTION(self.init.camera_resolution)

    @camera_resolution.setter
    def camera_resolution(self, value):
        if isinstance(value, RESOLUTION):
            self.init.camera_resolution = value.value
        else:
            raise TypeError("Argument must be of RESOLUTION type.")

    @property
    def camera_fps(self):
        return self.init.camera_fps

    @camera_fps.setter
    def camera_fps(self, int value):
        self.init.camera_fps = value

    @property
    def camera_linux_id(self):
        return self.init.camera_linux_id

    @camera_linux_id.setter
    def camera_linux_id(self, int value):
        self.init.camera_linux_id = value

    @property
    def svo_input_filename(self):
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
        return self.init.svo_real_time_mode

    @svo_real_time_mode.setter
    def svo_real_time_mode(self, bool value):
        self.init.svo_real_time_mode = value

    @property
    def depth_mode(self):
        return DEPTH_MODE(self.init.depth_mode)

    @depth_mode.setter
    def depth_mode(self, value):
        if isinstance(value, DEPTH_MODE):
            self.init.depth_mode = value.value
        else:
            raise TypeError("Argument must be of DEPTH_MODE type.")

    @property
    def coordinate_units(self):
        return UNIT(self.init.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value):
        if isinstance(value, UNIT):
            self.init.coordinate_units = value.value
        else:
            raise TypeError("Argument must be of UNIT type.")

    @property
    def coordinate_system(self):
        return COORDINATE_SYSTEM(self.init.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, COORDINATE_SYSTEM):
            self.init.coordinate_system = value.value
        else:
            raise TypeError("Argument must be of COORDINATE_SYSTEM type.")

    @property
    def sdk_verbose(self):
        return self.init.sdk_verbose

    @sdk_verbose.setter
    def sdk_verbose(self, bool value):
        self.init.sdk_verbose = value

    @property
    def sdk_gpu_id(self):
        return self.init.sdk_gpu_id

    @sdk_gpu_id.setter
    def sdk_gpu_id(self, int value):
        self.init.sdk_gpu_id = value

    @property
    def depth_minimum_distance(self):
        return self.init.depth_minimum_distance

    @depth_minimum_distance.setter
    def depth_minimum_distance(self, float value):
        self.init.depth_minimum_distance = value

    @property
    def camera_disable_self_calib(self):
        return self.init.camera_disable_self_calib

    @camera_disable_self_calib.setter
    def camera_disable_self_calib(self, bool value):
        self.init.camera_disable_self_calib = value

    @property
    def camera_image_flip(self):
        return self.init.camera_image_flip

    @camera_image_flip.setter
    def camera_image_flip(self, bool value):
        self.init.camera_image_flip = value

    @property
    def enable_right_side_measure(self):
        return self.init.enable_right_side_measure

    @enable_right_side_measure.setter
    def enable_right_side_measure(self, bool value):
        self.init.enable_right_side_measure = value

    @property
    def camera_buffer_count_linux(self):
        return self.init.camera_buffer_count_linux

    @camera_buffer_count_linux.setter
    def camera_buffer_count_linux(self, int value):
        self.init.camera_buffer_count_linux = value

    @property
    def sdk_verbose_log_file(self):
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
        return self.init.depth_stabilization

    @depth_stabilization.setter
    def depth_stabilization(self, bool value):
        self.init.depth_stabilization = value

    @property
    def input(self):
        input_t = InputType()
        input_t.input = self.init.input
        return input_t

    @input.setter
    def input(self, InputType input_t):
        self.init.input = input_t.input

    @property
    def optional_settings_path(self):
        if not self.init.optional_settings_path.empty():
            return self.init.optional_settings_path.get().decode()
        else:
            return ""

    @optional_settings_path.setter
    def optional_settings_path(self, str value):
        value_filename = value.encode()
        self.init.optional_settings_path.set(<char*>value_filename)

    def set_from_camera_id(self, id):
        self.init.input.setFromCameraID(id)

    def set_from_serial_number(self, serial_number):
        self.init.input.setFromSerialNumber(serial_number)

    def set_from_svo_file(self, str svo_input_filename):
        filename = svo_input_filename.encode()
        self.init.input.setFromSVOFile(String(<char*> filename))


cdef class RuntimeParameters:
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
        filename_save = filename.encode()
        return self.runtime.save(String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.runtime.load(String(<char*> filename_load))

    @property
    def sensing_mode(self):
        return SENSING_MODE(self.runtime.sensing_mode)

    @sensing_mode.setter
    def sensing_mode(self, value):
        if isinstance(value, SENSING_MODE):
            self.runtime.sensing_mode = value.value
        else:
            raise TypeError("Argument must be of SENSING_MODE type.")

    @property
    def enable_depth(self):
        return self.runtime.enable_depth

    @enable_depth.setter
    def enable_depth(self, bool value):
        self.runtime.enable_depth = value

    @property
    def measure3D_reference_frame(self):
        return REFERENCE_FRAME(self.runtime.measure3D_reference_frame)

    @measure3D_reference_frame.setter
    def measure3D_reference_frame(self, value):
        if isinstance(value, REFERENCE_FRAME):
            self.runtime.measure3D_reference_frame = value.value
        else:
            raise TypeError("Argument must be of REFERENCE type.")


cdef class TrackingParameters:
    cdef c_TrackingParameters* tracking
    def __cinit__(self, Transform init_pos, _enable_memory=True, _area_path=None):
        if _area_path is None:
            self.tracking = new c_TrackingParameters(init_pos.transform, _enable_memory, String())
        else:
            raise TypeError("Argument init_pos must be initialized first with Transform().")
    
    def __dealloc__(self):
        del self.tracking

    def save(self, str filename):
        filename_save = filename.encode()
        return self.tracking.save(String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.tracking.load(String(<char*> filename_load))

    def initial_world_transform(self, Transform init_pos):
        init_pos.transform = self.tracking.initial_world_transform
        return init_pos

    def set_initial_world_transform(self, Transform value):
        self.tracking.initial_world_transform = value.transform

    @property
    def enable_spatial_memory(self):
        return self.tracking.enable_spatial_memory

    @enable_spatial_memory.setter
    def enable_spatial_memory(self, bool value):
        self.tracking.enable_spatial_memory = value

    @property
    def enable_pose_smoothing(self):
        return self.tracking.enable_pose_smoothing

    @enable_pose_smoothing.setter
    def enable_pose_smoothing(self, bool value):
        self.tracking.enable_pose_smoothing = value

    @property
    def set_floor_as_origin(self):
        return self.tracking.set_floor_as_origin

    @set_floor_as_origin.setter
    def set_floor_as_origin(self, bool value):
        self.tracking.set_floor_as_origin = value

    @property
    def enable_imu_fusion(self):
        return self.tracking.enable_imu_fusion

    @enable_imu_fusion.setter
    def enable_imu_fusion(self, bool value):
        self.tracking.enable_imu_fusion = value

    @property
    def area_file_path(self):
        if not self.tracking.area_file_path.empty():
            return self.tracking.area_file_path.get().decode()
        else:
            return ""

    @area_file_path.setter
    def area_file_path(self, str value):
        value_area = value.encode()
        self.tracking.area_file_path.set(<char*>value_area)


cdef class SpatialMappingParameters:
    cdef c_SpatialMappingParameters* spatial
    def __cinit__(self, resolution=MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH, mapping_range=MAPPING_RANGE.MAPPING_RANGE_MEDIUM,
                  max_memory_usage=2048, save_texture=True, use_chunk_only=True,
                  reverse_vertex_order=False):
        if (isinstance(resolution, MAPPING_RESOLUTION) and isinstance(mapping_range, MAPPING_RANGE) and
            isinstance(use_chunk_only, bool) and isinstance(reverse_vertex_order, bool)):
            self.spatial = new c_SpatialMappingParameters(resolution.value, mapping_range.value, max_memory_usage, save_texture,
                                                        use_chunk_only, reverse_vertex_order)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.spatial

    def set_resolution(self, resolution=MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH):
        if isinstance(resolution, MAPPING_RESOLUTION):
            self.spatial.set(<c_MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of RESOLUTION type.")

    def set_range(self, mapping_range=MAPPING_RANGE.MAPPING_RANGE_MEDIUM):
        if isinstance(mapping_range, MAPPING_RANGE):
            self.spatial.set(<c_MAPPING_RANGE> mapping_range.value)
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    def get_range_preset(self, mapping_range):
        if isinstance(mapping_range, MAPPING_RANGE):
            return self.spatial.get(<c_MAPPING_RANGE> mapping_range.value)
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    def get_resolution_preset(self, resolution):
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.get(<c_MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    def get_recommended_range(self, resolution, Camera py_cam):
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.getRecommendedRange(<c_MAPPING_RESOLUTION> resolution.value, py_cam.camera)
        elif isinstance(resolution, float):
            return self.spatial.getRecommendedRange(<float> resolution, py_cam.camera)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION or float type.")

    @property
    def max_memory_usage(self):
        return self.spatial.max_memory_usage

    @max_memory_usage.setter
    def max_memory_usage(self, int value):
        self.spatial.max_memory_usage = value

    @property
    def save_texture(self):
        return self.spatial.save_texture

    @save_texture.setter
    def save_texture(self, bool value):
        self.spatial.save_texture = value

    @property
    def use_chunk_only(self):
        return self.spatial.use_chunk_only

    @use_chunk_only.setter
    def use_chunk_only(self, bool value):
        self.spatial.use_chunk_only = value

    @property
    def reverse_vertex_order(self):
        return self.spatial.reverse_vertex_order

    @reverse_vertex_order.setter
    def reverse_vertex_order(self, bool value):
        self.spatial.reverse_vertex_order = value

    @property
    def allowed_range(self):
        return self.spatial.allowed_range

    @property
    def range_meter(self):
        return self.spatial.range_meter

    @range_meter.setter
    def range_meter(self, float value):
        self.spatial.range_meter = value

    @property
    def allowed_resolution(self):
        return self.spatial.allowed_resolution

    @property
    def resolution_meter(self):
        return self.spatial.resolution_meter

    @resolution_meter.setter
    def resolution_meter(self, float value):
        self.spatial.resolution_meter = value

cdef class Pose:
    cdef c_Pose pose
    def __cinit__(self):
        self.pose = c_Pose()

    def init_pose(self, Pose pose):
        self.pose = c_Pose(pose.pose)

    def init_transform(self, Transform pose_data, mtimestamp=0, mconfidence=0):
        self.pose = c_Pose(pose_data.transform, mtimestamp, mconfidence)

    def get_translation(self, Translation py_translation):
        py_translation.translation = self.pose.getTranslation()
        return py_translation

    def get_orientation(self, Orientation py_orientation):
        py_orientation.orientation = self.pose.getOrientation()
        return py_orientation

    def get_rotation_matrix(self, Rotation py_rotation):
        py_rotation.rotation = self.pose.getRotationMatrix()
        py_rotation.mat = self.pose.getRotationMatrix()
        return py_rotation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.pose.getRotationVector()[i]
        return arr

    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.pose.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr

    @property
    def valid(self):
        return self.pose.valid

    @valid.setter
    def valid(self, bool valid_):
        self.pose.valid = valid_

    @property
    def timestamp(self):
        return self.pose.timestamp

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.pose.timestamp = timestamp

    def pose_data(self, Transform pose_data):
        pose_data.transform = self.pose.pose_data
        pose_data.mat = self.pose.pose_data
        return pose_data

    @property
    def pose_confidence(self):
        return self.pose.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, int pose_confidence_):
        self.pose.pose_confidence = pose_confidence_

    @property
    def pose_covariance(self):
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.pose.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.pose.pose_covariance[i] = pose_covariance_[i]

cdef class IMUData:
    cdef c_IMUData imuData

    def __cinit__(self):
        self.imuData = c_IMUData()
        
    def init_imuData(self, IMUData imuData):
        self.imuData = c_IMUData(imuData.imuData)

    def init_transform(self, Transform pose_data, mtimestamp=0, mconfidence=0):
        self.imuData = c_IMUData(pose_data.transform, mtimestamp, mconfidence)

    def get_orientation_covariance(self, Matrix3f orientation_covariance):
        orientation_covariance.mat = self.imuData.orientation_covariance
        return orientation_covariance

    def get_angular_velocity(self, angular_velocity):
        for i in range(3):
            angular_velocity[i] = self.imuData.angular_velocity[i]
        return angular_velocity

    def get_linear_acceleration(self, linear_acceleration):
        for i in range(3):
            linear_acceleration[i] = self.imuData.linear_acceleration[i]
        return linear_acceleration

    def get_angular_velocity_convariance(self, Matrix3f angular_velocity_convariance):
        angular_velocity_convariance.mat = self.imuData.angular_velocity_convariance
        return angular_velocity_convariance

    def get_linear_acceleration_convariance(self, Matrix3f linear_acceleration_convariance):
        linear_acceleration_convariance.mat = self.imuData.linear_acceleration_convariance
        return linear_acceleration_convariance

    def get_translation(self, Translation py_translation):
        py_translation.translation = self.imuData.getTranslation()
        return py_translation

    def get_orientation(self, Orientation py_orientation):
        py_orientation.orientation = self.imuData.getOrientation()
        return py_orientation

    def get_rotation_matrix(self, Rotation py_rotation):
        py_rotation.rotation = self.imuData.getRotationMatrix()
        py_rotation.mat = self.imuData.getRotationMatrix()
        return py_rotation

    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.imuData.getRotationVector()[i]
        return arr

    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.imuData.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr
        
    def pose_data(self, Transform pose_data):
        pose_data.transform = self.imuData.pose_data
        pose_data.mat = self.imuData.pose_data
        return pose_data

    @property
    def valid(self):
        return self.imuData.valid

    @valid.setter
    def valid(self, bool valid_):
        self.imuData.valid = valid_
 
    @property
    def timestamp(self):
        return self.imuData.timestamp

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.imuData.timestamp = timestamp

    @property
    def pose_confidence(self):
        return self.imuData.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, int pose_confidence_):
        self.imuData.pose_confidence = pose_confidence_

    @property
    def pose_covariance(self):
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.imuData.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.imuData.pose_covariance[i] = pose_covariance_[i]


cdef class Camera:
    cdef c_Camera camera
    def __cinit__(self):
        self.camera = c_Camera()

    def close(self):
        self.camera.close()

    def open(self, InitParameters py_init):
        if py_init:
            return ERROR_CODE(self.camera.open(deref(py_init.init)))
        else:
            print("InitParameters must be initialized first with InitParameters().")

    def is_opened(self):
        return self.camera.isOpened()

    def grab(self, RuntimeParameters py_runtime):
        if py_runtime:
            return ERROR_CODE(self.camera.grab(deref(py_runtime.runtime)))
        else:
            print("RuntimeParameters must be initialized first with RuntimeParameters().")

    def retrieve_image(self, Mat py_mat, view=VIEW.VIEW_LEFT, type=MEM.MEM_CPU, width=0,
                       height=0):
        if (isinstance(view, VIEW) and isinstance(type, MEM) and isinstance(width, int) and
           isinstance(height, int)):
            return ERROR_CODE(self.camera.retrieveImage(py_mat.mat, view.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of VIEW, MEM and integer types.")

    def retrieve_measure(self, Mat py_mat, measure=MEASURE.MEASURE_DEPTH, type=MEM.MEM_CPU,
                         width=0, height=0):
        if (isinstance(measure, MEASURE) and isinstance(type, MEM) and isinstance(width, int) and
           isinstance(height, int)):
            return ERROR_CODE(self.camera.retrieveMeasure(py_mat.mat, measure.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of MEASURE, MEM and integer types.")

    def set_confidence_threshold(self, int conf_treshold_value):
        self.camera.setConfidenceThreshold(conf_treshold_value)

    def get_confidence_threshold(self):
        return self.camera.getConfidenceThreshold()

    def get_resolution(self):
        return Resolution(self.camera.getResolution().width, self.camera.getResolution().height)

    def set_depth_max_range_value(self, float depth_max_range):
        self.camera.setDepthMaxRangeValue(depth_max_range)

    def get_depth_max_range_value(self):
        return self.camera.getDepthMaxRangeValue()

    def get_depth_min_range_value(self):
        return self.camera.getDepthMinRangeValue()

    def set_svo_position(self, int frame_number):
        self.camera.setSVOPosition(frame_number)

    def get_svo_position(self):
        return self.camera.getSVOPosition()

    def get_svo_number_of_frames(self):
        return self.camera.getSVONumberOfFrames()

    def set_camera_settings(self, settings, int value, use_default=False):
        if isinstance(settings, CAMERA_SETTINGS) and isinstance(use_default, bool):
            self.camera.setCameraSettings(settings.value, value, use_default)
        else:
            raise TypeError("Arguments must be of CAMERA_SETTINGS and boolean types.")

    def get_camera_settings(self, setting):
        if isinstance(setting, CAMERA_SETTINGS):
            return self.camera.getCameraSettings(setting.value)
        else:
            raise TypeError("Argument is not of CAMERA_SETTINGS type.")

    def get_camera_fps(self):
        return self.camera.getCameraFPS()

    def set_camera_fps(self, int desired_fps):
        self.camera.setCameraFPS(desired_fps)

    def get_current_fps(self):
        return self.camera.getCurrentFPS()

    def get_camera_timestamp(self):
        return self.camera.getCameraTimestamp()

    def get_current_timestamp(self):
        return self.camera.getCurrentTimestamp()

    def get_timestamp(self, time_reference):
        if isinstance(time_reference, TIME_REFERENCE):
            return self.camera.getTimestamp(time_reference.value)
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")

    def get_frame_dropped_count(self):
        return self.camera.getFrameDroppedCount()

    def get_camera_information(self, resizer = Resolution(0, 0)):
        return CameraInformation(self, resizer)

    def get_self_calibration_state(self):
        return SELF_CALIBRATION_STATE(self.camera.getSelfCalibrationState())

    def reset_self_calibration(self):
        self.camera.resetSelfCalibration()

    def enable_tracking(self, TrackingParameters py_tracking):
        if py_tracking:
            return ERROR_CODE(self.camera.enableTracking(deref(py_tracking.tracking)))
        else:
            print("TrackingParameters must be initialized first with TrackingParameters().")
   
    def get_imu_data(self, IMUData py_imu_data, time_reference = TIME_REFERENCE.TIME_REFERENCE_CURRENT):
        if isinstance(time_reference, TIME_REFERENCE):
            return ERROR_CODE(self.camera.getIMUData(py_imu_data.imuData, time_reference.value))
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")
    
    def set_imu_prior(self, Transform transfom):
        return ERROR_CODE(self.camera.setIMUPrior(transfom.transform))

    def get_position(self, Pose py_pose, reference_frame = REFERENCE_FRAME.REFERENCE_FRAME_WORLD):
        if isinstance(reference_frame, REFERENCE_FRAME):
            return TRACKING_STATE(self.camera.getPosition(py_pose.pose, reference_frame.value))
        else:
            raise TypeError("Argument is not of REFERENCE_FRAME type.")

    def get_area_export_state(self):
        return AREA_EXPORT_STATE(self.camera.getAreaExportState())
   
    def save_current_area(self, str area_file_path):
        filename = area_file_path.encode()
        return ERROR_CODE(self.camera.saveCurrentArea(String(<char*> filename)))

    def disable_tracking(self, str area_file_path=""):
        filename = area_file_path.encode()
        self.camera.disableTracking(String(<char*> filename))

    def reset_tracking(self, Transform path):
        return ERROR_CODE(self.camera.resetTracking(path.transform))

    def enable_spatial_mapping(self, SpatialMappingParameters py_spatial):
        if py_spatial:
            return ERROR_CODE(self.camera.enableSpatialMapping(deref(py_spatial.spatial)))
        else:
            print("SpatialMappingParameters must be initialized first with SpatialMappingParameters()")

    def pause_spatial_mapping(self, status):
        if isinstance(status, bool):
            self.camera.pauseSpatialMapping(status)
        else:
            raise TypeError("Argument is not of boolean type.")

    def get_spatial_mapping_state(self):
        return SPATIAL_MAPPING_STATE(self.camera.getSpatialMappingState())

    def extract_whole_mesh(self, Mesh py_mesh):
        return ERROR_CODE(self.camera.extractWholeMesh(deref(py_mesh.mesh)))

    def request_mesh_async(self):
        self.camera.requestMeshAsync()

    def get_mesh_request_status_async(self):
        return ERROR_CODE(self.camera.getMeshRequestStatusAsync())

    def retrieve_mesh_async(self, Mesh py_mesh):
        return ERROR_CODE(self.camera.retrieveMeshAsync(deref(py_mesh.mesh)))

    def find_plane_at_hit(self, coord, Plane py_plane):
        cdef Vector2[uint] vec = Vector2[uint](coord[0], coord[1])
        return ERROR_CODE(self.camera.findPlaneAtHit(vec, py_plane.plane))

    def find_floor_plane(self, Plane py_plane, Transform resetTrackingFloorFrame, floor_height_prior = float('nan'), Rotation world_orientation_prior = Rotation(Matrix3f().zeros()), floor_height_prior_tolerance = float('nan')) :
        return ERROR_CODE(self.camera.findFloorPlane(py_plane.plane, resetTrackingFloorFrame.transform, floor_height_prior, world_orientation_prior.rotation, floor_height_prior_tolerance))

    def disable_spatial_mapping(self):
        self.camera.disableSpatialMapping()

    def enable_recording(self, str video_filename,
                          compression_mode=SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS):
        if isinstance(compression_mode, SVO_COMPRESSION_MODE):
            filename = video_filename.encode()
            return ERROR_CODE(self.camera.enableRecording(String(<char*> filename),
                                      compression_mode.value))
        else:
            raise TypeError("Argument is not of SVO_COMPRESSION_MODE type.")

    def record(self):
        return self.camera.record()

    def disable_recording(self):
        self.camera.disableRecording()

    def get_sdk_version(cls):
        return cls.camera.getSDKVersion().get().decode()

    def is_zed_connected(cls):
        return cls.camera.isZEDconnected()

    def stickto_cpu_core(cls, int cpu_core):
        return ERROR_CODE(cls.camera.sticktoCPUCore(cpu_core))

    def get_device_list(cls):
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

def save_camera_depth_as(Camera zed, format, str name, factor=1):
    if isinstance(format, DEPTH_FORMAT) and factor <= 65536:
        name_save = name.encode()
        return saveDepthAs(zed.camera, format.value, String(<char*>name_save), factor)
    else:
        raise TypeError("Arguments must be of DEPTH_FORMAT type and factor not over 65536.")

def save_camera_point_cloud_as(Camera zed, format, str name, with_color=False):
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
