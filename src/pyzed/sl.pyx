########################################################################
#
# Copyright (c) 2022, STEREOLABS.
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
# DATA, OR PROFITS;POSITIONAL_TRACKING_STATE OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
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
from libcpp.map cimport map
from sl_c cimport ( String, to_str, Camera as c_Camera, ERROR_CODE as c_ERROR_CODE, toString
                    , InitParameters as c_InitParameters, INPUT_TYPE as c_INPUT_TYPE
                    , InputType as c_InputType, RESOLUTION as c_RESOLUTION, BUS_TYPE as c_BUS_TYPE
                    , DEPTH_MODE as c_DEPTH_MODE, UNIT as c_UNIT
                    , COORDINATE_SYSTEM as c_COORDINATE_SYSTEM, CUcontext
                    , RuntimeParameters as c_RuntimeParameters
                    , REFERENCE_FRAME as c_REFERENCE_FRAME, Mat as c_Mat, Resolution as c_Resolution
                    , MAT_TYPE as c_MAT_TYPE, MEM as c_MEM, VIEW as c_VIEW, MEASURE as c_MEASURE
                    , Timestamp as c_Timestamp, TIME_REFERENCE as c_TIME_REFERENCE
                    , MODEL as c_MODEL, PositionalTrackingParameters as c_PositionalTrackingParameters
                    , Transform as c_Transform, Matrix4f as c_Matrix4f, Matrix3f as c_Matrix3f, Pose as c_Pose
                    , POSITIONAL_TRACKING_STATE as c_POSITIONAL_TRACKING_STATE
                    , POSITIONAL_TRACKING_MODE as c_POSITIONAL_TRACKING_MODE
                    , AREA_EXPORTING_STATE as c_AREA_EXPORTING_STATE, SensorsData as c_SensorsData
                    , CAMERA_MOTION_STATE as c_CAMERA_MOTION_STATE, SpatialMappingParameters as c_SpatialMappingParameters
                    , MAPPING_RESOLUTION as c_MAPPING_RESOLUTION, MAPPING_RANGE as c_MAPPING_RANGE
                    , SPATIAL_MAP_TYPE as c_SPATIAL_MAP_TYPE, SPATIAL_MAPPING_STATE as c_SPATIAL_MAPPING_STATE
                    , VIDEO_SETTINGS as c_VIDEO_SETTINGS, Rect as c_Rect, SIDE as c_SIDE
                    , RecordingParameters as c_RecordingParameters, SVO_COMPRESSION_MODE as c_SVO_COMPRESSION_MODE
                    , StreamingParameters as c_StreamingParameters, STREAMING_CODEC as c_STREAMING_CODEC
                    , RecordingStatus as c_RecordingStatus, ObjectDetectionParameters as c_ObjectDetectionParameters
                    , BodyTrackingParameters as c_BodyTrackingParameters, BodyTrackingRuntimeParameters as c_BodyTrackingRuntimeParameters
                    , BODY_TRACKING_MODEL as c_BODY_TRACKING_MODEL, OBJECT_DETECTION_MODEL as c_OBJECT_DETECTION_MODEL, Objects as c_Objects, Bodies as c_Bodies, create_object_detection_runtime_parameters
                    , ObjectDetectionRuntimeParameters as c_ObjectDetectionRuntimeParameters 
                    , DeviceProperties as c_DeviceProperties, CAMERA_STATE as c_CAMERA_STATE
                    , StreamingProperties as c_StreamingProperties, FusedPointCloud as c_FusedPointCloud
                    , Mesh as c_Mesh, Plane as c_Plane, Vector2, Vector3, Vector4, Rotation as c_Rotation
                    , CameraConfiguration as c_CameraConfiguration, SensorsConfiguration as c_SensorsConfiguration
                    , CalibrationParameters as c_CalibrationParameters, CameraParameters as c_CameraParameters
                    , SensorParameters as c_SensorParameters, SENSOR_TYPE as c_SENSOR_TYPE, SENSORS_UNIT as c_SENSORS_UNIT, HEADING_STATE as c_HEADING_STATE
                    , SENSOR_LOCATION as c_SENSOR_LOCATION, TemperatureData as c_TemperatureData, MagnetometerData as c_MagnetometerData, IMUData as c_IMUData
                    , MeshFilterParameters as c_MeshFilterParameters, MESH_FILTER as c_MESH_FILTER
                    , PointCloudChunk as c_PointCloudChunk, Chunk as c_Chunk, MESH_TEXTURE_FORMAT as c_MESH_TEXTURE_FORMAT
                    , MESH_FILE_FORMAT as c_MESH_FILE_FORMAT, PLANE_TYPE as c_PLANE_TYPE, sleep_ms as c_sleep_ms, sleep_us as c_sleep_us
                    , getCurrentTimeStamp, BarometerData as c_BarometerData, Orientation as c_Orientation
                    , Translation as c_Translation, COPY_TYPE as c_COPY_TYPE
                    , uchar1, uchar2, uchar3, uchar4, ushort1, float1, float2, float3, float4, matResolution
                    , setToUchar1, setToUchar2, setToUchar3, setToUchar4, setToUshort1, setToFloat1, setToFloat2, setToFloat3, setToFloat4
                    , setValueUchar1, setValueUchar2, setValueUchar3, setValueUchar4, setValueUshort1, setValueFloat1, setValueFloat2, setValueFloat3, setValueFloat4
                    , getValueUchar1, getValueUchar2, getValueUchar3, getValueUchar4, getValueUshort1, getValueFloat1, getValueFloat2, getValueFloat3, getValueFloat4
                    , getPointerUchar1, getPointerUchar2, getPointerUchar3, getPointerUchar4, getPointerUshort1, getPointerFloat1, getPointerFloat2, getPointerFloat3, getPointerFloat4, uint
                    , ObjectData as c_ObjectData, BodyData as c_BodyData, OBJECT_CLASS as c_OBJECT_CLASS, OBJECT_SUBCLASS as c_OBJECT_SUBCLASS
                    , OBJECT_TRACKING_STATE as c_OBJECT_TRACKING_STATE, OBJECT_ACTION_STATE as c_OBJECT_ACTION_STATE
                    , BODY_18_PARTS as c_BODY_18_PARTS, SIDE as c_SIDE, CameraInformation as c_CameraInformation, CUctx_st
                    , FLIP_MODE as c_FLIP_MODE, getResolution as c_getResolution, BatchParameters as c_BatchParameters
                    , ObjectsBatch as c_ObjectsBatch, BodiesBatch as c_BodiesBatch, getIdx as c_getIdx, BODY_FORMAT as c_BODY_FORMAT, BODY_KEYPOINTS_SELECTION as c_BODY_KEYPOINTS_SELECTION
                    , BODY_34_PARTS as c_BODY_34_PARTS, BODY_38_PARTS as c_BODY_38_PARTS
                    , generate_unique_id as c_generate_unique_id, CustomBoxObjectData as c_CustomBoxObjectData
                    , OBJECT_FILTERING_MODE as c_OBJECT_FILTERING_MODE
                    , COMM_TYPE as c_COMM_TYPE, FUSION_ERROR_CODE as c_FUSION_ERROR_CODE, SENDER_ERROR_CODE as c_SENDER_ERROR_CODE
                    , FusionConfiguration as c_FusionConfiguration, CommunicationParameters as c_CommunicationParameters
                    , InitFusionParameters as c_InitFusionParameters, CameraIdentifier as c_CameraIdentifier
                    , BodyTrackingFusionParameters as c_BodyTrackingFusionParameters, BodyTrackingFusionRuntimeParameters as c_BodyTrackingFusionRuntimeParameters
                    , PositionalTrackingFusionParameters as c_PositionalTrackingFusionParameters, POSITION_TYPE as c_POSITION_TYPE
                    , CameraMetrics as c_CameraMetrics, FusionMetrics as c_FusionMetrics, GNSSData as c_GNSSData, Fusion as c_Fusion
                    , ECEF as c_ECEF, LatLng as c_LatLng, UTM as c_UTM
                    , GeoConverter as c_GeoConverter, GeoPose as c_GeoPose
                    , readFusionConfigurationFile as c_readFusionConfigurationFile
                    , readFusionConfigurationFile2 as c_readFusionConfigurationFile2
                    , writeConfigurationFile as c_writeConfigurationFile

                    )
from cython.operator cimport (dereference as deref, postincrement)
from libc.string cimport memcpy
from cpython cimport bool
import enum

import numpy as np
cimport numpy as np
#https://github.com/cython/cython/wiki/tutorials-numpy#c-api-initialization
np.import_array()
from math import sqrt

## \defgroup Video_group Video Module
## \defgroup Depth_group Depth Sensing Module
## \defgroup Core_group Core Module
## \defgroup SpatialMapping_group Spatial Mapping Module
## \defgroup PositionalTracking_group Positional Tracking Module
## \defgroup Object_group Object Detection Module
## \defgroup Body_group Body Tracking Module
## \defgroup Sensors_group Sensors Module
## \defgroup Fusion_group Fusion Module

##
# \ref Timestamp representation and utilities.
# \ingroup Core_group
cdef class Timestamp():
    cdef c_Timestamp timestamp

    def __cinit__(self):
        self.timestamp = c_Timestamp()

    ##
    # Timestamp in nanoseconds.
    @property
    def data_ns(self):
        return self.timestamp.data_ns

    @data_ns.setter
    def data_ns(self, ns):
        self.timestamp.data_ns = ns

    ##
    # Gets the timestamp in nanoseconds.
    def get_nanoseconds(self):
        return self.timestamp.getNanoseconds()

    ##
    # Gets the timestamp in microseconds.
    def get_microseconds(self):
        return self.timestamp.getMicroseconds()

    ##
    # Gets the timestamp in milliseconds.
    def get_milliseconds(self):
        return self.timestamp.getMilliseconds()

    ##
    # Gets the timestamp in seconds.
    def get_seconds(self):
        return self.timestamp.getSeconds()

    ##
    # Sets the timestamp to a value in nanoseconds.
    def set_nanoseconds(self, t_ns: int):
        self.timestamp.setNanoseconds(t_ns)

    ##
    # Sets the timestamp to a value in microseconds.
    def set_microseconds(self, t_us: int):
        self.timestamp.setMicroseconds(t_us)

    ##
    # Sets the timestamp to a value in milliseconds.
    def set_milliseconds(self, t_ms: int):
        self.timestamp.setMilliseconds(t_ms)

    ##
    # Sets the timestamp to a value in seconds.
    def set_seconds(self, t_s: int):
        self.timestamp.setSeconds(t_s)

##
# Lists error codes in the ZED SDK.
# \ingroup Core_group
#
# | Enumerator                                         |                                                                                                                                                                 |
# |----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | CAMERA_REBOOTING                                   | The ZED camera is currently rebooting. |
# | SUCCESS                                            | Standard code for successful behavior.                                                                                                                          |
# | FAILURE                                            | Standard code for unsuccessful behavior.                                                                                                                        |
# | NO_GPU_COMPATIBLE                                  | No GPU found or CUDA capability of the device is not supported.                                                                                                 |
# | NOT_ENOUGH_GPU_MEMORY                              | Not enough GPU memory for this depth mode, try a different mode (such as PERFORMANCE), or increase the minimum depth value (see \ref InitParameters.depth_minimum_distance).                                                                          |
# | CAMERA_NOT_DETECTED                                | The ZED camera is not plugged or detected.                                                                                                                      |
# | SENSORS_NOT_INITIALIZED                            | The MCU that controls the sensors module has an invalid Serial Number. You can try to recover it launching the 'ZED Diagnostic' tool from the command line with the option '-r'. |
# | SENSORS_NOT_AVAILABLE                              | a ZED-M or ZED2/2i camera is detected but the sensors (imu,barometer...) cannot be opened. Only for ZED-M or ZED2/2i devices.                                   |
# | INVALID_RESOLUTION                                 | In case of invalid resolution parameter, such as an upsize beyond the original image size in Camera.retrieve_image                                               |
# | LOW_USB_BANDWIDTH                                  | This issue can occur when you use multiple ZED or a USB 2.0 port (bandwidth issue).                                                                            |
# | CALIBRATION_FILE_NOT_AVAILABLE                     | ZED calibration file is not found on the host machine. Use ZED Explorer or ZED Calibration to get one.                                                          |
# | INVALID_CALIBRATION_FILE                           | ZED calibration file is not valid, try to download the factory one or recalibrate your camera using 'ZED Calibration'.                                          |
# | INVALID_SVO_FILE                                   | The provided SVO file is not valid.                                                                                                                             |
# | SVO_RECORDING_ERROR                                | An recorder related error occurred (not enough free storage, invalid file).                                                                                     |
# | SVO_UNSUPPORTED_COMPRESSION                        | An SVO related error when NVIDIA based compression cannot be loaded.                                                                                            |
# | END_OF_SVOFILE_REACHED                             | SVO end of file has been reached, and no frame will be available until the SVO position is reset.                                                               |
# | INVALID_COORDINATE_SYSTEM                          | The requested coordinate system is not available.                                                                                                               |
# | INVALID_FIRMWARE                                   | The firmware of the ZED is out of date. Update to the latest version.                                                                                           |
# | INVALID_FUNCTION_PARAMETERS                        | An invalid parameter has been set for the function.                                                                                                             |
# | CUDA_ERROR                                         | In grab() only, a CUDA error has been detected in the process. Activate verbose in sl.Camera.open for more info. |
# | CAMERA_NOT_INITIALIZED                             | In grab() only, ZED SDK is not initialized. Probably a missing call to sl.Camera.open. |
# | NVIDIA_DRIVER_OUT_OF_DATE                          | Your NVIDIA driver is too old and not compatible with your current CUDA version. |
# | INVALID_FUNCTION_CALL                              | The call of the function is not valid in the current context. Could be a missing call of sl.Camera.open |
# | CORRUPTED_SDK_INSTALLATION                         | The SDK wasn't able to load its dependencies or some assets are missing, the installer should be launched. |
# | INCOMPATIBLE_SDK_VERSION                           | The installed SDK is incompatible SDK used to compile the program. |
# | INVALID_AREA_FILE                                  | The given area file does not exist, check the path. |
# | INCOMPATIBLE_AREA_FILE                             | The area file does not contain enough data to be used or the sl.DEPTH_MODE used during the creation of the area file is different from the one currently set. |
# | CAMERA_FAILED_TO_SETUP                             | Failed to open the camera at the proper resolution. Try another resolution or make sure that the UVC driver is properly installed. |
# | CAMERA_DETECTION_ISSUE                             | Your ZED can not be opened, try replugging it to another USB port or flipping the USB-C connector. |
# | CANNOT_START_CAMERA_STREAM                         | Cannot start camera stream. Make sure your camera is not already used by another process or blocked by firewall or antivirus. |
# | NO_GPU_DETECTED                                    | No GPU found, CUDA is unable to list it. Can be a driver/reboot issue. |
# | PLANE_NOT_FOUND                                    | Plane not found, either no plane is detected in the scene, at the location or corresponding to the floor, or the floor plane doesn't match the prior given |
# | MODULE_NOT_COMPATIBLE_WITH_CAMERA                  | The Object detection module is only compatible with the ZED2/ZED2i and ZED Mini |
# | MOTION_SENSORS_REQUIRED                            | The module needs the sensors to be enabled (see \ref InitParameters.sensors_required) |
# | MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION            | The module needs a newer version of CUDA |
class ERROR_CODE(enum.Enum):
    CAMERA_REBOOTING = <int>c_ERROR_CODE.CAMERA_REBOOTING
    SUCCESS = <int>c_ERROR_CODE.SUCCESS
    FAILURE = <int>c_ERROR_CODE.FAILURE
    NO_GPU_COMPATIBLE = <int>c_ERROR_CODE.NO_GPU_COMPATIBLE
    NOT_ENOUGH_GPU_MEMORY = <int>c_ERROR_CODE.NOT_ENOUGH_GPU_MEMORY
    CAMERA_NOT_DETECTED = <int>c_ERROR_CODE.CAMERA_NOT_DETECTED
    SENSORS_NOT_INITIALIZED = <int>c_ERROR_CODE.SENSORS_NOT_INITIALIZED
    SENSORS_NOT_AVAILABLE = <int>c_ERROR_CODE.SENSORS_NOT_AVAILABLE
    INVALID_RESOLUTION = <int>c_ERROR_CODE.INVALID_RESOLUTION
    LOW_USB_BANDWIDTH = <int>c_ERROR_CODE.LOW_USB_BANDWIDTH
    CALIBRATION_FILE_NOT_AVAILABLE = <int>c_ERROR_CODE.CALIBRATION_FILE_NOT_AVAILABLE
    INVALID_CALIBRATION_FILE = <int>c_ERROR_CODE.INVALID_CALIBRATION_FILE
    INVALID_SVO_FILE = <int>c_ERROR_CODE.INVALID_SVO_FILE
    SVO_RECORDING_ERROR = <int>c_ERROR_CODE.SVO_RECORDING_ERROR
    END_OF_SVOFILE_REACHED = <int>c_ERROR_CODE.END_OF_SVOFILE_REACHED
    SVO_UNSUPPORTED_COMPRESSION = <int>c_ERROR_CODE.SVO_UNSUPPORTED_COMPRESSION
    INVALID_COORDINATE_SYSTEM = <int>c_ERROR_CODE.INVALID_COORDINATE_SYSTEM
    INVALID_FIRMWARE = <int>c_ERROR_CODE.INVALID_FIRMWARE
    INVALID_FUNCTION_PARAMETERS = <int>c_ERROR_CODE.INVALID_FUNCTION_PARAMETERS
    CUDA_ERROR = <int>c_ERROR_CODE.CUDA_ERROR
    CAMERA_NOT_INITIALIZED = <int>c_ERROR_CODE.CAMERA_NOT_INITIALIZED
    NVIDIA_DRIVER_OUT_OF_DATE = <int>c_ERROR_CODE.NVIDIA_DRIVER_OUT_OF_DATE
    INVALID_FUNCTION_CALL = <int>c_ERROR_CODE.INVALID_FUNCTION_CALL
    CORRUPTED_SDK_INSTALLATION = <int>c_ERROR_CODE.CORRUPTED_SDK_INSTALLATION
    INCOMPATIBLE_SDK_VERSION = <int>c_ERROR_CODE.INCOMPATIBLE_SDK_VERSION
    INVALID_AREA_FILE = <int>c_ERROR_CODE.INVALID_AREA_FILE
    INCOMPATIBLE_AREA_FILE = <int>c_ERROR_CODE.INCOMPATIBLE_AREA_FILE
    CAMERA_FAILED_TO_SETUP = <int>c_ERROR_CODE.CAMERA_FAILED_TO_SETUP
    CAMERA_DETECTION_ISSUE = <int>c_ERROR_CODE.CAMERA_DETECTION_ISSUE
    CANNOT_START_CAMERA_STREAM = <int>c_ERROR_CODE.CANNOT_START_CAMERA_STREAM
    NO_GPU_DETECTED =<int> c_ERROR_CODE.NO_GPU_DETECTED
    PLANE_NOT_FOUND = <int>c_ERROR_CODE.PLANE_NOT_FOUND
    MODULE_NOT_COMPATIBLE_WITH_CAMERA = <int>c_ERROR_CODE.MODULE_NOT_COMPATIBLE_WITH_CAMERA
    MOTION_SENSORS_REQUIRED = <int>c_ERROR_CODE.MOTION_SENSORS_REQUIRED
    MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION = <int>c_ERROR_CODE.MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION
    LAST = <int>c_ERROR_CODE.LAST

    def __str__(self):
        return to_str(toString(<c_ERROR_CODE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_ERROR_CODE>(<unsigned int>self.value))).decode()

##
# Lists compatible ZED Camera model
#
# \ingroup Video_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | ZED        | Defines ZED Camera model |
# | ZED_M      | Defines ZED Mini (ZED-M) Camera model |
# | ZED2       | Defines ZED 2 Camera model |
# | ZED2i      | Defines ZED 2i Camera model |
# | ZED_X      | Defines ZED-X Camera model |
# | ZED_XM     | Defines ZED-X Mini Camera model |
class MODEL(enum.Enum):
    ZED = <int>c_MODEL.ZED
    ZED_M = <int>c_MODEL.ZED_M
    ZED2 = <int>c_MODEL.ZED2
    ZED2i = <int>c_MODEL.ZED2i
    ZED_X = <int>c_MODEL.ZED_X
    ZED_XM = <int>c_MODEL.ZED_XM
    LAST = <int>c_MODEL.MODEL_LAST

    def __str__(self):
        return to_str(toString(<c_MODEL>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_MODEL>(<unsigned int>self.value))).decode()

##
# Lists available input type in SDK
#
# \ingroup Video_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | USB        | USB input mode  |
# | SVO        | SVO file input mode  |
# | STREAM     | STREAM input mode (requires to use enableStreaming()/disableStreaming() on the "sender" side) |
# | GMSL       | GMSL input mode (only on NVIDIA Jetson) |

class INPUT_TYPE(enum.Enum):
    USB = <int>c_INPUT_TYPE.USB
    SVO = <int>c_INPUT_TYPE.SVO
    STREAM = <int>c_INPUT_TYPE.STREAM
    GMSL = <int>c_INPUT_TYPE.GMSL
    LAST = <int>c_INPUT_TYPE.LAST

##
# List available models for object detection module
#
# \ingroup Object_group
#
# | Enumerator               |                  |
# |--------------------------|------------------|
# | MULTI_CLASS_BOX_FAST     | Any object, bounding box based |
# | MULTI_CLASS_BOX_ACCURATE | Any object, bounding box based, more accurate but slower than the base model |
# | MULTI_CLASS_BOX_MEDIUM   | Any object, bounding box based, compromise between accuracy and speed  |
# | PERSON_HEAD_BOX_FAST     | Bounding Box detector specialized in person heads, particularly well suited for crowded environments, the person localization is also improved  |
# | PERSON_HEAD_BOX_ACCURATE | Bounding Box detector specialized in person heads, particularly well suited for crowded environments, the person localization is also improved, state of the art accuracy  |
# | CUSTOM_BOX_OBJECTS       | For external inference, using your own custom model and/or frameworks. This mode disables the internal inference engine, the 2D bounding box detection must be provided  |
class OBJECT_DETECTION_MODEL(enum.Enum):
    MULTI_CLASS_BOX_FAST = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    MULTI_CLASS_BOX_MEDIUM = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    MULTI_CLASS_BOX_ACCURATE = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    PERSON_HEAD_BOX_FAST = <int>c_OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_FAST
    PERSON_HEAD_BOX_ACCURATE = <int>c_OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_ACCURATE
    CUSTOM_BOX_OBJECTS = <int>c_OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    LAST = <int>c_OBJECT_DETECTION_MODEL.LAST

##
# List available models for body tracking module
#
# \ingroup Body_group
#
# | Enumerator               |                  |
# |--------------------------|------------------|
# | HUMAN_BODY_FAST          | Keypoints based, specific to human skeleton, real time performance even on Jetson or low end GPU cards |
# | HUMAN_BODY_ACCURATE      | Keypoints based, specific to human skeleton, state of the art accuracy, requires powerful GPU |
# | HUMAN_BODY_MEDIUM        | Keypoints based, specific to human skeleton, compromise between accuracy and speed  |
class BODY_TRACKING_MODEL(enum.Enum):
    HUMAN_BODY_FAST = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    HUMAN_BODY_ACCURATE = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    HUMAN_BODY_MEDIUM = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
    LAST = <int>c_BODY_TRACKING_MODEL.LAST

##
# Lists of supported bounding box preprocessing
#
# \ingroup Object_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | NONE             | SDK will not apply any preprocessing to the detected objects |
# | NMS3D            | SDK will remove objects that are in the same 3D position as an already tracked object (independant of class ID) |
# | NMS3D_PER_CLASS  | SDK will remove objects that are in the same 3D position as an already tracked object of the same class ID |
class OBJECT_FILTERING_MODE(enum.Enum):
    NONE = <int>c_OBJECT_FILTERING_MODE.NONE
    NMS3D = <int>c_OBJECT_FILTERING_MODE.NMS3D
    NMS3D_PER_CLASS = <int>c_OBJECT_FILTERING_MODE.NMS3D_PER_CLASS
    LAST = <int>c_OBJECT_FILTERING_MODE.LAST

##
# List of possible camera states
#
# \ingroup Video_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | AVAILABLE | Defines if the camera can be opened by the SDK |
# | NOT_AVAILABLE | Defines if the camera is already opened and unavailable |
class CAMERA_STATE(enum.Enum):
    AVAILABLE = <int>c_CAMERA_STATE.AVAILABLE
    NOT_AVAILABLE = <int>c_CAMERA_STATE.NOT_AVAILABLE
    LAST = <int>c_CAMERA_STATE.CAMERA_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_CAMERA_STATE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_CAMERA_STATE>(<unsigned int>self.value))).decode()

##
# Tells the program to wait for x ms.
# \ingroup Core_group
#
#  @param time : the number of ms to wait.
#
def sleep_ms(time: int):
    c_sleep_ms(time)

##
# Tells the program to wait for x microseconds.
# \ingroup Core_group
#
#  @param time : the number of microseconds to wait.
#
def sleep_us(time: int):
    c_sleep_us(time)


##
# Returns the actual size of the given \ref RESOLUTION as a \ref sl.Resolution object
# \ingroup Video_group
#
# @param resolution : the given \ref RESOLUTION
def get_resolution(resolution):
    if isinstance(resolution, RESOLUTION):
        out = c_getResolution(<c_RESOLUTION>(<unsigned int>resolution.value))
        res = Resolution()
        res.width = out.width
        res.height = out.height
        return res
    else:
        raise TypeError("Argument is not of RESOLUTION type.")
        
##
# Properties of a camera.
# \ingroup Video_group
#
# \note
#   A camera_model ZED_M with an id '-1' can be due to an inverted USB-C cable.
#
# \warning
#   Experimental on Windows.
#
cdef class DeviceProperties:
    cdef c_DeviceProperties c_device_properties

    def __cinit__(self):
        self.c_device_properties = c_DeviceProperties()

    ##
    # the camera state
    @property
    def camera_state(self):
        return CAMERA_STATE(<int>self.c_device_properties.camera_state)

    @camera_state.setter
    def camera_state(self, camera_state):
        if isinstance(camera_state, CAMERA_STATE):
            self.c_device_properties.camera_state = (<c_CAMERA_STATE> (<unsigned int>camera_state.value))
        else:
            raise TypeError("Argument is not of CAMERA_STATE type.")

    ##
    # the camera id (Notice that only the camera with id '0' can be used on Windows)
    @property
    def id(self):
        return self.c_device_properties.id

    @id.setter
    def id(self, id):
        self.c_device_properties.id = id

    ##
    # the camera system path
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

    ##
    # the camera model
    @property
    def camera_model(self):
        return MODEL(<int>self.c_device_properties.camera_model)

    @camera_model.setter
    def camera_model(self, camera_model):
        if isinstance(camera_model, MODEL):
            self.c_device_properties.camera_model = (<c_MODEL> (<unsigned int>camera_model.value))
        else:
            raise TypeError("Argument is not of MODEL type.")
    ##
    # the camera serial number
    @property
    def serial_number(self):
        return self.c_device_properties.serial_number

    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_device_properties.serial_number = serial_number

    ##
    # the input type
    @property
    def input_type(self):
        return INPUT_TYPE(<int>self.c_device_properties.input_type)

    @input_type.setter
    def input_type(self, value : INPUT_TYPE):
        if isinstance(value, INPUT_TYPE):
            self.c_device_properties.input_type = <c_INPUT_TYPE>(<int>value.value)
        else:
            raise TypeError("Argument is not of INPUT_TYPE type.")

    def __str__(self):
        return to_str(toString(self.c_device_properties)).decode()

    def __repr__(self):
        return to_str(toString(self.c_device_properties)).decode()


##
# Represents a generic 3*3 matrix
# \ingroup Core_group
#
# It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on.
# You can access the data with the 'r' ptr or by element attribute.
# | | | |
# |-|-|-|
# | r00 | r01 | r02 |
# | r10 | r11 | r12 |
# | r20 | r21 | r22 |
cdef class Matrix3f:
    cdef c_Matrix3f *mat
    def __cinit__(self):
        if type(self) is Matrix3f:
            self.mat = new c_Matrix3f()

    def __dealloc__(self):
        if type(self) is Matrix3f:
            del self.mat
    ##
    # Creates a \ref Matrix3f from another \ref Matrix3f
    # \param matrix : the \ref Matrix3f to copy
    def init_matrix(self, matrix: Matrix3f):
        for i in range(9):
            self.mat.r[i] = matrix.mat.r[i]

    ##
    # Inverses the matrix.
    def inverse(self):
        self.mat.inverse()

    ##
    # Inverses the \ref Matrix3f passed as a parameter.
    # \param rotation : the \ref Matrix3f to inverse
    # \return the inversed \ref Matrix3f
    def inverse_mat(self, rotation: Matrix3f):
        out = Matrix3f()
        out.mat[0] = rotation.mat.inverse(rotation.mat[0])
        return out

    ##
    # Sets the \ref Matrix3f to its transpose.
    def transpose(self):
        self.mat.transpose()

    ##
    # Returns the transpose of a \ref Matrix3f
    # \param rotation : the \ref Matrix3f to compute the transpose from.
    # \return The transpose of the given \ref Matrix3f
    def transpose_mat(self, rotation: Matrix3f):
        out = Matrix3f()
        out.mat[0] = rotation.mat.transpose(rotation.mat[0])
        return out

    ##
    # Sets the \ref Matrix3f to identity.
    # \return itself
    def set_identity(self):
        self.mat.setIdentity()
        return self

    ##
    # Creates an identity \ref Matrix3f
    # \return a \ref Matrix3f set to identity
    def identity(self):
        new_mat = Matrix3f()
        return new_mat.set_identity()

    ##
    # Sets the \ref Matrix3f to zero.
    def set_zeros(self):
        self.mat.setZeros()

    ##
    # Creates a \ref Matrix3f filled with zeros.
    # \return A \ref Matrix3f filled with zeros
    def zeros(self):
        output_mat = Matrix3f()
        output_mat.mat[0] = self.mat.zeros()
        return output_mat

    ##
    # Returns the components of the \ref Matrix3f in a string.
    # \return A string containing the components of the current of \ref Matrix3f
    def get_infos(self):
        return to_str(self.mat.getInfos()).decode()

    ##
    # Name of the matrix (optional).
    @property
    def matrix_name(self):
        if not self.mat.matrix_name.empty():
           return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, name: str):
        self.mat.matrix_name.set(name.encode()) 

    @property
    def nbElem(self):
        return 9

    ##
    # Numpy array of inner data
    @property
    def r(self):
        cdef np.ndarray arr = np.zeros(9)
        for i in range(9):
            arr[i] = self.mat.r[i]
        return arr.reshape(3, 3)

    @r.setter
    def r(self, value):
        if isinstance(value, list):
            if len(value) <= 9:
                for i in range(len(value)):
                    self.mat.r[i] = value[i]
            else:
                raise IndexError("Value list must be of length 9 max.")
        elif isinstance(value, np.ndarray):
            if value.size <= 9:
                for i in range(value.size):
                    self.mat.r[i] = value[i]
            else:
                raise IndexError("Numpy array must be of size 9.")
        else:
            raise TypeError("Argument must be numpy array or list type.")

    def __mul__(self, other):
        matrix = Matrix3f()
        if isinstance(other, Matrix3f):
            matrix.r = (self.r * other.r).reshape(9)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.r = (other * self.r).reshape(9)
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

    def __getitem__(self, key):
        return self.mat.r[<int>key[0] * 3 + <int>key[1]]

    def __setitem__(self, key, value):
        self.mat.r[<int>key[0] * 3 + <int>key[1]] = <float>value

    def __repr__(self):
        return to_str(self.mat.getInfos()).decode()
    
##
# Represents a generic fourth-dimensional matrix.
# \ingroup Core_group
# It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on.
# You can access the data by the 'm' ptr or by the element attribute.
#
# | | | | |
# |-|-|-|-|
# | r00 | r01 | r02 | tx |
# | r10 | r11 | r12 | ty |
# | r20 | r21 | r22 | tz |
# | m30 | m31 | m32 | m33 |
cdef class Matrix4f:
    cdef c_Matrix4f* mat
    def __cinit__(self):
        if type(self) is Matrix4f:
            self.mat = new c_Matrix4f()

    def __dealloc__(self):
        if type(self) is Matrix4f:
            del self.mat

    ##
    # Creates a \ref Matrix4f from another \ref Matrix4f (deep copy)
    # \param matrix : the \ref Matrix4f to copy
    def init_matrix(self, matrix: Matrix4f):
        for i in range(16):
            self.mat.m[i] = matrix.mat.m[i]

    ##
    # Inverses the matrix.
    def inverse(self):
        return ERROR_CODE(<int><int>(self.mat.inverse()))

    ##
    # Inverses the \ref Matrix4f passed as a parameter
    # \param rotation : the \ref Matrix4f to inverse
    # \return the inversed \ref Matrix4f
    def inverse_mat(self, rotation: Matrix4f):
        out = Matrix4f()
        out.mat[0] = rotation.mat.inverse(rotation.mat[0])
        return out

    ##
    # Sets the \ref Matrix4f to its transpose.
    def transpose(self):
        self.mat.transpose()

    ##
    # Returns the transpose of a \ref Matrix4f
    # \param rotation : the \ref Matrix4f to compute the transpose from.
    # \return the transposed \ref Matrix4f
    def transpose_mat(self, rotation: Matrix4f):
        out = Matrix4f()
        out.mat[0] = rotation.mat.transpose(rotation.mat[0])
        return out

    ##
    # Sets the \ref Matrix4f to identity
    # \return itself
    def set_identity(self):
        self.mat.setIdentity()
        return self

    ##
    # Creates an identity \ref Matrix4f
    # \return A \ref Matrix4f set to identity
    def identity(self):
        new_mat = Matrix4f()
        return new_mat.set_identity()

    ##
    # Sets the \ref Matrix4f to zero.
    def set_zeros(self):
        self.mat.setZeros()

    ##
    # Creates a \ref Matrix4f  filled with zeros.
    # \return A \ref Matrix4f filled with zeros.
    def zeros(self):
        output_mat = Matrix4f()
        output_mat.mat[0] = self.mat.zeros()
        return output_mat

    ##
    # Returns the components of the \ref Matrix4f in a string.
    # \return A string containing the components of the current \ref Matrix4f
    def get_infos(self):
        return to_str(self.mat.getInfos()).decode()

    ##
    # Sets a 3x3 Matrix inside the \ref Matrix4f
    # \note Can be used to set the rotation matrix when the matrix4f is a pose or an isometric matrix.
    # \param input : sub matrix to put inside the  \ref Matrix4f
    # \param row : index of the row to start the 3x3 block. Must be 0 or 1.
    # \param column : index of the column to start the 3x3 block. Must be 0 or 1.
    #
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    def set_sub_matrix3f(self, input: Matrix3f, row=0, column=0):
        if row != 0 and row != 1 or column != 0 and column != 1:
            raise TypeError("Arguments row and column must be 0 or 1.")
        else:
            return ERROR_CODE(<int>self.mat.setSubMatrix3f(input.mat[0], row, column))

    ##
    # Sets a 3x1 Vector inside the \ref Matrix4f at the specified column index.
    # \note Can be used to set the Translation/Position matrix when the matrix4f is a pose or an isometry.
    # \param input0 : first value of the 3x1 Vector to put inside the \ref Matrix4f
    # \param input1 : second value of the 3x1 Vector to put inside the \ref Matrix4f
    # \param input2 : third value of the 3x1 Vector to put inside the \ref Matrix4f
    # \param column : index of the column to start the 3x3 block. By default, it is the last column (translation for a \ref Pose ).
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    def set_sub_vector3f(self, input0: float, input1: float, input2: float, column=3):
        return ERROR_CODE(<int>self.mat.setSubVector3f(Vector3[float](input0, input1, input2), column))

    ##
    # Sets a 4x1 Vector inside the \ref Matrix4f at the specified column index.
    # \param input0 : first value of the 4x1 Vector to put inside the \ref Matrix4f
    # \param input1 : second value of the 4x1 Vector to put inside the \ref Matrix4f
    # \param input2 : third value of the 4x1 Vector to put inside the \ref Matrix4f
    # \param input3 : fourth value of the 4x1 Vector to put inside the \ref Matrix4f
    # \param column : index of the column to start the 3x3 block. By default, it is the last column (translation for a \ref Pose ).
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    def set_sub_vector4f(self, input0: float, input1: float, input2: float, input3: float, column=3):
        return ERROR_CODE(<int>self.mat.setSubVector4f(Vector4[float](input0, input1, input2, input3), column))

    ##
    # Returns the name of the matrix (optional). 
    @property
    def matrix_name(self):
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, str name):
        self.mat.matrix_name.set(name.encode())

    ##
    # Access to the content of the \ref Matrix4f as a numpy array or list.
    @property
    def m(self):
        cdef np.ndarray arr = np.zeros(16)
        for i in range(16):
            arr[i] = self.mat.m[i]
        return arr.reshape(4, 4)

    @m.setter
    def m(self, value):
        if isinstance(value, list):
            if len(value) <= 16:
                for i in range(len(value)):
                    self.mat.m[i] = value[i]
            else:
                raise IndexError("Value list must be of length 16 max.")
        elif isinstance(value, np.ndarray):
            if value.size <= 16:
                for i in range(value.size):
                    self.mat.m[i] = value[i]
            else:
                raise IndexError("Numpy array must be of size 16.")
        else:
            raise TypeError("Argument must be numpy array or list type.")

    def __mul__(self, other):
        matrix = Matrix4f()
        if isinstance(other, Matrix4f) :
            matrix.m = np.matmul(self.m, other.m).reshape(16)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, long):
            matrix.m = (other * self.m).reshape(16)
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

    def __getitem__(self, key):
        return self.mat.m[<int>key[0] * 4 + <int>key[1]]

    def __setitem__(self, key, value):
        self.mat.m[<int>key[0] * 4 + <int>key[1]] = <float>value


    def __repr__(self):
        return self.get_infos()

##
# Defines left, right, both to distinguish between left and right or both sides 
# \ingroup Video_group
#
# | Enumerator |            |
# |------------|------------|
# | LEFT | Left side only |
# | RIGHT | Right side only |
# | BOTH | Left and right side |
class SIDE(enum.Enum):
    LEFT = <int>c_SIDE.LEFT
    RIGHT = <int>c_SIDE.RIGHT
    BOTH = <int>c_SIDE.BOTH

##
# Represents the available resolution list.
# \ingroup Core_group
# \note The VGA resolution does not respect the 640*480 standard to better fit the camera sensor (672*376 is used).
#
# | Enumerator |            |
# |------------|------------|
# | HD2K    | 2208*1242 (x2), available framerates: 15 fps. |
# | HD1080  | 1920*1080 (x2), available framerates: 15, 30 fps. |
# | HD1200  | 1920*1200 (x2), available framerates: 30, 60 fps. (ZED-X(M) only) |
# | HD720   | 1280*720 (x2), available framerates: 15, 30, 60 fps |
# | SVGA    | 960*600 (x2), available framerates: 60, 120 fps. (ZED-X(M) only) |
# | VGA     | 672*376 (x2), available framerates: 15, 30, 60, 100 fps. |
class RESOLUTION(enum.Enum):
    HD2K = <int>c_RESOLUTION.HD2K
    HD1080 = <int>c_RESOLUTION.HD1080
    HD1200 = <int>c_RESOLUTION.HD1200
    HD720 = <int>c_RESOLUTION.HD720
    SVGA  = <int>c_RESOLUTION.SVGA
    VGA  = <int>c_RESOLUTION.VGA
    AUTO = <int>c_RESOLUTION.AUTO
    LAST = <int>c_RESOLUTION.LAST

##
# Lists available camera settings for the ZED camera (contrast, hue, saturation, gain...).
# \ingroup Video_group
#
# \warning GAIN and EXPOSURE are linked in auto/default mode (see \ref Camera.set_camera_settings).
# 
# Each enum defines one of those settings.
#
# | Enumerator |                         |
# |------------|-------------------------|
# | BRIGHTNESS | Defines the brightness control. Affected value should be between 0 and 8. |
# | CONTRAST | Defines the contrast control. Affected value should be between 0 and 8. |
# | HUE | Defines the hue control. Affected value should be between 0 and 11. |
# | SATURATION | Defines the saturation control. Affected value should be between 0 and 8. |
# | SHARPNESS | Defines the digital sharpening control. Affected value should be betwwen 0 and 8. |
# | GAMMA |  Defines the ISP gamma control. Affected value should be between 1 and 9. |
# | GAIN |  Defines the gain control. Affected value should be between 0 and 100 for manual control. |
# | EXPOSURE | Defines the exposure control. Affected value should be between 0 and 100 for manual control.\n The exposition is mapped linearly in a percentage of the following max values. Special case for set_exposure(0) that corresponds to 0.17072ms.\n The conversion to milliseconds depends on the framerate: <ul><li>15fps set_exposure(100) -> 19.97ms</li><li>30fps set_exposure(100) -> 19.97ms</li><li>60fps se_exposure(100) -> 10.84072ms</li><li>100fps set_exposure(100) -> 10.106624ms</li></ul> |
# | AEC_AGC | Defines if the Gain and Exposure are in automatic mode or not. Setting a Gain or Exposure through @GAIN or @EXPOSURE values will automatically set this value to 0. |
# | AEC_AGC_ROI | Defines the region of interest for automatic exposure/gain computation. To be used with the dedicated @set_camera_settings_roi/@get_camera_settings_roi functions. |
# | WHITEBALANCE_TEMPERATURE | Defines the color temperature value. Setting a value will automatically set @WHITEBALANCE_AUTO to 0. Affected value should be between 2800 and 6500 with a step of 100. |
# | WHITEBALANCE_AUTO | Defines if the White balance is in automatic mode or not |
# | LED_STATUS | Defines the status of the camera front LED. Set to 0 to disable the light, 1 to enable the light. Default value is on. Requires Camera FW 1523 at least |
# | EXPOSURE_TIME | Defines the real exposure time in microseconds. Only available for GMSL based cameras. Recommended for ZED-X/ZED-XM to control manual exposure (instead of EXPOSURE setting) |
# | ANALOG_GAIN | Defines the real analog gain (sensor) in mDB. Range is defined by Jetson DTS and by default [1000-16000].  Recommended for ZED-X/ZED-XM to control manual sensor gain (instead of GAIN setting). Only available for GMSL based cameras. |
# | DIGITAL_GAIN | Defines the real digital gain (ISP) as a factor. Range is defined by Jetson DTS and by default [1-256].  Recommended for ZED-X/ZED-XM to control manual ISP gain (instead of GAIN setting). Only available for GMSL based cameras. |
# | AUTO_EXPOSURE_TIME_RANGE | Defines the range of exposure auto control in micro seconds.Used with \ref setCameraSettings(VIDEO_SETTINGS,int,int).  Min/Max range between Max range defined in DTS. By default : [28000 - <fps_time> or 19000] us. Only available for GMSL based cameras |
# | AUTO_ANALOG_GAIN_RANGE | Defines the range of sensor gain in automatic control. Used with \ref setCameraSettings(VIDEO_SETTINGS,int,int). Min/Max range between Max range defined in DTS. By default : [1000 - 16000] mdB .  Only available for GMSL based cameras |
# | AUTO_DIGITAL_GAIN_RANGE | Defines the range of digital ISP gain in automatic control. Used with \ref setCameraSettings(VIDEO_SETTINGS,int,int). Min/Max range between Max range defined in DTS. By default : [1 - 256]. Only available for GMSL based cameras |
# | EXPOSURE_COMPENSATION | Defines the Exposure-target compensation made after auto exposure. Reduces the overall illumination target by factor of F-stops. values range is [0 - 100] (mapped between [-2.0,2.0]). Default value is 50, i.e. no compensation applied. Only available for GMSL based cameras |
# | DENOISING | Defines the level of denoising applied on both left and right images. values range is [0-100]. Default value is 50. Only available for GMSL based cameras |
class VIDEO_SETTINGS(enum.Enum):
    BRIGHTNESS = <int>c_VIDEO_SETTINGS.BRIGHTNESS
    CONTRAST = <int>c_VIDEO_SETTINGS.CONTRAST
    HUE = <int>c_VIDEO_SETTINGS.HUE
    SATURATION = <int>c_VIDEO_SETTINGS.SATURATION
    SHARPNESS = <int>c_VIDEO_SETTINGS.SHARPNESS
    GAMMA = <int>c_VIDEO_SETTINGS.GAMMA
    GAIN = <int>c_VIDEO_SETTINGS.GAIN
    EXPOSURE = <int>c_VIDEO_SETTINGS.EXPOSURE
    AEC_AGC = <int>c_VIDEO_SETTINGS.AEC_AGC
    AEC_AGC_ROI = <int>c_VIDEO_SETTINGS.AEC_AGC_ROI
    WHITEBALANCE_TEMPERATURE = <int>c_VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
    WHITEBALANCE_AUTO = <int>c_VIDEO_SETTINGS.WHITEBALANCE_AUTO
    LED_STATUS = <int>c_VIDEO_SETTINGS.LED_STATUS
    EXPOSURE_TIME = <int>c_VIDEO_SETTINGS.EXPOSURE_TIME
    ANALOG_GAIN = <int>c_VIDEO_SETTINGS.ANALOG_GAIN
    DIGITAL_GAIN = <int>c_VIDEO_SETTINGS.DIGITAL_GAIN
    AUTO_EXPOSURE_TIME_RANGE = <int>c_VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE
    AUTO_ANALOG_GAIN_RANGE = <int>c_VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE
    AUTO_DIGITAL_GAIN_RANGE = <int>c_VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE
    EXPOSURE_COMPENSATION = <int>c_VIDEO_SETTINGS.EXPOSURE_COMPENSATION
    DENOISING = <int>c_VIDEO_SETTINGS.DENOISING
    LAST = <int>c_VIDEO_SETTINGS.LAST

##
# Lists available depth computation modes.
# \ingroup Depth_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | NONE | This mode does not compute any depth map. Only rectified stereo images will be available. |
# | PERFORMANCE | Computation mode optimized for speed. |
# | QUALITY | Computation mode designed for high quality results. |
# | ULTRA | Computation mode favorising edges and sharpness. Requires more GPU memory and computation power. |
# | NEURAL | End to End Neural disparity estimation, requires AI module |
class DEPTH_MODE(enum.Enum):
    NONE = <int>c_DEPTH_MODE.NONE
    PERFORMANCE = <int>c_DEPTH_MODE.PERFORMANCE
    QUALITY = <int>c_DEPTH_MODE.QUALITY
    ULTRA = <int>c_DEPTH_MODE.ULTRA
    NEURAL = <int>c_DEPTH_MODE.NEURAL
    LAST = <int>c_DEPTH_MODE.DEPTH_MODE_LAST

##
# Lists available unit for measures.
# \ingroup Core_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | MILLIMETER | International System, 1/1000 METER. |
# | CENTIMETER | International System, 1/100 METER. |
# | METER | International System, 1 METER |
# | INCH | Imperial Unit, 1/12 FOOT |
# | FOOT | Imperial Unit, 1 FOOT |
class UNIT(enum.Enum):
    MILLIMETER = <int>c_UNIT.MILLIMETER
    CENTIMETER = <int>c_UNIT.CENTIMETER
    METER = <int>c_UNIT.METER
    INCH = <int>c_UNIT.INCH
    FOOT = <int>c_UNIT.FOOT
    LAST = <int>c_UNIT.UNIT_LAST


##
# Lists available coordinates systems for positional tracking and 3D measures.
# \image html CoordinateSystem.png
# \ingroup Core_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | IMAGE | Standard coordinates system in computer vision. Used in OpenCV : see <a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">here</a>. |
# | LEFT_HANDED_Y_UP | Left-Handed with Y up and Z forward. Used in Unity with DirectX. |
# | RIGHT_HANDED_Y_UP | Right-Handed with Y pointing up and Z backward. Used in OpenGL. |
# | RIGHT_HANDED_Z_UP | Right-Handed with Z pointing up and Y forward. Used in 3DSMax. |
# | LEFT_HANDED_Z_UP | Left-Handed with Z axis pointing up and X forward. Used in Unreal Engine. |
# | RIGHT_HANDED_Z_UP_X_FWD | Right-Handed with Z pointing up and X forward. Used in ROS (REP 103). |
class COORDINATE_SYSTEM(enum.Enum):
    IMAGE = <int>c_COORDINATE_SYSTEM.IMAGE
    LEFT_HANDED_Y_UP = <int>c_COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    RIGHT_HANDED_Y_UP = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    RIGHT_HANDED_Z_UP = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    LEFT_HANDED_Z_UP = <int>c_COORDINATE_SYSTEM.LEFT_HANDED_Z_UP
    RIGHT_HANDED_Z_UP_X_FWD = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    LAST = <int>c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_LAST

    def __str__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>(<unsigned int>self.value))).decode()

##
# Lists retrievable measures.
# \ingroup Core_group
# | Enumerator |                         |
# |------------|-------------------------|
# | DISPARITY | Disparity map. Each pixel contains 1 float. [sl.MAT_TYPE.F32_C1] (\ref sl.MAT_TYPE) |
# | DEPTH | Depth map, in \ref sl.UNIT defined in \ref sl.InitParameters. Each pixel contains 1 float. [sl.MAT_TYPE.F32_C1] (\ref sl.MAT_TYPE) |
# | CONFIDENCE | Certainty/confidence of the depth map. Each pixel contains 1 float. [sl.MAT_TYPE.F32_C1] (\ref sl.MAT_TYPE) |
# | XYZ | Point cloud. Each pixel contains 4 float (X, Y, Z, not used). [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | XYZRGBA | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the RGBA color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | XYZBGRA | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the BGRA color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | XYZARGB | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the ARGB color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | XYZABGR | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the ABGR color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | NORMALS | Normals vector. Each pixel contains 4 float (X, Y, Z, 0). [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE) |
# | DISPARITY_RIGHT | Disparity map for right sensor. Each pixel contains 1 float. [sl.MAT_TYPE.F32_C1] (\ref sl.MAT_TYPE)|
# | DEPTH_RIGHT | Depth map for right sensor. Each pixel contains 1 float. [sl.MAT_TYPE.F32_C1] (\ref sl.MAT_TYPE)|
# | XYZ_RIGHT | Point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, not used). [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | XYZRGBA_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the RGBA color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | XYZBGRA_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the BGRA color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | XYZARGB_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the ARGB color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | XYZABGR_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color). The color needs to be read as an unsigned char[4] representing the ABGR color. [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | NORMALS_RIGHT | Normals vector for right view. Each pixel contains 4 float (X, Y, Z, 0). [sl.MAT_TYPE.F32_C4] (\ref sl.MAT_TYPE)|
# | DEPTH_U16_MM | Depth map in millimeter whatever the \ref sl.UNIT defined in \ref sl.InitParameters. Invalid values are set to 0, depth values are clamped at 65000. Each pixel contains 1 unsigned short. [sl.MAT_TYPE.U16_C1] (\ref sl.MAT_TYPE)|
# | DEPTH_U16_MM_RIGHT | Depth map in millimeter for right sensor. Each pixel  contains 1 unsigned short. [sl.MAT_TYPE.U16_C1] (\ref sl.MAT_TYPE)|
class MEASURE(enum.Enum):
    DISPARITY = <int>c_MEASURE.DISPARITY
    DEPTH = <int>c_MEASURE.DEPTH
    CONFIDENCE = <int>c_MEASURE.CONFIDENCE
    XYZ = <int>c_MEASURE.XYZ
    XYZRGBA = <int>c_MEASURE.XYZRGBA
    XYZBGRA = <int>c_MEASURE.XYZBGRA
    XYZARGB = <int>c_MEASURE.XYZARGB
    XYZABGR = <int>c_MEASURE.XYZABGR
    NORMALS = <int>c_MEASURE.NORMALS
    DISPARITY_RIGHT = <int>c_MEASURE.DISPARITY_RIGHT
    DEPTH_RIGHT = <int>c_MEASURE.DEPTH_RIGHT
    XYZ_RIGHT = <int>c_MEASURE.XYZ_RIGHT
    XYZRGBA_RIGHT = <int>c_MEASURE.XYZRGBA_RIGHT
    XYZBGRA_RIGHT = <int>c_MEASURE.XYZBGRA_RIGHT
    XYZARGB_RIGHT = <int>c_MEASURE.XYZARGB_RIGHT
    XYZABGR_RIGHT = <int>c_MEASURE.XYZABGR_RIGHT
    NORMALS_RIGHT = <int>c_MEASURE.NORMALS_RIGHT
    DEPTH_U16_MM = <int>c_MEASURE.DEPTH_U16_MM
    DEPTH_U16_MM_RIGHT = <int>c_MEASURE.DEPTH_U16_MM_RIGHT
    LAST = <int>c_MEASURE.MEASURE_LAST

    def __str__(self):
        return to_str(toString(<c_MEASURE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_MEASURE>(<unsigned int>self.value))).decode()

##
# Lists available views.
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | LEFT | Left RGBA image. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE) |
# | RIGHT | Right RGBA image. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE) |
# | LEFT_GRAY | Left GRAY image. Each pixel contains 1 unsigned char. [sl.MAT_TYPE.U8_C1] (\ref sl.MAT_TYPE)|
# | RIGHT_GRAY | Right GRAY image. Each pixel contains 1 unsigned char. sl.MAT_TYPE.U8_C1 [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | LEFT_UNRECTIFIED | Left RGBA unrectified image. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | RIGHT_UNRECTIFIED | Right RGBA unrectified image. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | LEFT_UNRECTIFIED_GRAY | Left GRAY unrectified image. Each pixel contains 1 unsigned char. [sl.MAT_TYPE.U8_C1] (\ref sl.MAT_TYPE)|
# | RIGHT_UNRECTIFIED_GRAY | Right GRAY unrectified image. Each pixel contains 1 unsigned char. [sl.MAT_TYPE.U8_C1] (\ref sl.MAT_TYPE)|
# | SIDE_BY_SIDE | Left and right image (the image width is therefore doubled). Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | DEPTH | Color rendering of the depth. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE). Use [MEASURE.DEPTH](\ref MEASURE) with \ref Camera.retrieve_measure() to get depth values. |
# | CONFIDENCE | Color rendering of the depth confidence. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | NORMALS | Color rendering of the normals. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | DEPTH_RIGHT | Color rendering of the right depth mapped on right sensor, [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
# | NORMALS_RIGHT | Color rendering of the normals mapped on right sensor. Each pixel contains 4 unsigned char (B,G,R,A). [sl.MAT_TYPE.U8_C4] (\ref sl.MAT_TYPE)|
class VIEW(enum.Enum):
    LEFT = <int>c_VIEW.LEFT
    RIGHT = <int>c_VIEW.RIGHT
    LEFT_GRAY = <int>c_VIEW.LEFT_GRAY
    RIGHT_GRAY = <int>c_VIEW.RIGHT_GRAY
    LEFT_UNRECTIFIED = <int>c_VIEW.LEFT_UNRECTIFIED
    RIGHT_UNRECTIFIED = <int>c_VIEW.RIGHT_UNRECTIFIED
    LEFT_UNRECTIFIED_GRAY = <int>c_VIEW.LEFT_UNRECTIFIED_GRAY
    RIGHT_UNRECTIFIED_GRAY = <int>c_VIEW.RIGHT_UNRECTIFIED_GRAY
    SIDE_BY_SIDE = <int>c_VIEW.SIDE_BY_SIDE
    DEPTH = <int>c_VIEW.VIEW_DEPTH
    CONFIDENCE = <int>c_VIEW.VIEW_CONFIDENCE
    NORMALS = <int>c_VIEW.VIEW_NORMALS
    DEPTH_RIGHT = <int>c_VIEW.VIEW_DEPTH_RIGHT
    NORMALS_RIGHT = <int>c_VIEW.VIEW_NORMALS_RIGHT
    LAST = <int>c_VIEW.VIEW_LAST

    def __str__(self):
        return to_str(toString(<c_VIEW>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_VIEW>(<unsigned int>self.value))).decode()

##
# Lists the different states of positional tracking.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | SEARCHING | The camera is searching for a previously known position to locate itself. |
# | OK | Positional tracking is working normally. |
# | OFF | Positional tracking is not enabled. |
# | FPS_TOO_LOW | Effective FPS is too low to give proper results for motion tracking. Consider using PERFORMANCE parameters ([DEPTH_MODE.PERFORMANCE](\ref DEPTH_MODE), low camera resolution (VGA,HD720)) |
# | SEARCHING_FLOOR_PLANE | The camera is searching for the floor plane to locate itself related to it, the REFERENCE_FRAME::WORLD will be set afterward.|
class POSITIONAL_TRACKING_STATE(enum.Enum):
    SEARCHING = <int>c_POSITIONAL_TRACKING_STATE.SEARCHING
    OK = <int>c_POSITIONAL_TRACKING_STATE.OK
    OFF = <int>c_POSITIONAL_TRACKING_STATE.OFF
    FPS_TOO_LOW = <int>c_POSITIONAL_TRACKING_STATE.FPS_TOO_LOW
    SEARCHING_FLOOR_PLANE = <int>c_POSITIONAL_TRACKING_STATE.SEARCHING_FLOOR_PLANE
    LAST = <int>c_POSITIONAL_TRACKING_STATE.POSITIONAL_TRACKING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_STATE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_STATE>(<unsigned int>self.value))).decode()


##
#  Lists the mode of positional tracking that can be used.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | STANDARD | Default mode, best compromise in performance and accuracy |
# | QUALITY | Improve accuracy in more challenging scenes such as outdoor repetitive patterns like extensive fields. Currently works best with ULTRA depth mode, requires more compute power  |
class POSITIONAL_TRACKING_MODE(enum.Enum):
    STANDARD = <int>c_POSITIONAL_TRACKING_MODE.STANDARD
    QUALITY = <int>c_POSITIONAL_TRACKING_MODE.QUALITY

    def __str__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_MODE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_MODE>(<unsigned int>self.value))).decode()

##
# Lists the different states of spatial memory area export.
# \ingroup SpatialMapping_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | SUCCESS | The spatial memory file has been successfully created. |
# | RUNNING | The spatial memory is currently being written. |
# | NOT_STARTED | The spatial memory file exportation has not been called. |
# | FILE_EMPTY | The spatial memory contains no data, the file is empty. |
# | FILE_ERROR | The spatial memory file has not been written because of a wrong file name. |
# | SPATIAL_MEMORY_DISABLED | The spatial memory learning is disable, no file can be created. |
class AREA_EXPORTING_STATE(enum.Enum):
    SUCCESS = <int>c_AREA_EXPORTING_STATE.AREA_EXPORTING_STATE_SUCCESS
    RUNNING = <int>c_AREA_EXPORTING_STATE.RUNNING
    NOT_STARTED = <int>c_AREA_EXPORTING_STATE.NOT_STARTED
    FILE_EMPTY = <int>c_AREA_EXPORTING_STATE.FILE_EMPTY
    FILE_ERROR = <int>c_AREA_EXPORTING_STATE.FILE_ERROR
    SPATIAL_MEMORY_DISABLED = <int>c_AREA_EXPORTING_STATE.SPATIAL_MEMORY_DISABLED
    LAST = <int>c_AREA_EXPORTING_STATE.AREA_EXPORTING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_AREA_EXPORTING_STATE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_AREA_EXPORTING_STATE>(<unsigned int>self.value))).decode()

##
# Defines which type of position matrix is used to store camera path and pose.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | WORLD | The transform of \ref sl.Pose will contain the motion with reference to the world frame (previously called PATH). |
# | CAMERA | The transform of \ref sl.Pose will contain the motion with reference to the previous camera frame (previously called POSE). |
class REFERENCE_FRAME(enum.Enum):
    WORLD = <int>c_REFERENCE_FRAME.WORLD
    CAMERA = <int>c_REFERENCE_FRAME.CAMERA
    LAST = <int>c_REFERENCE_FRAME.REFERENCE_FRAME_LAST

    def __str__(self):
        return to_str(toString(<c_REFERENCE_FRAME>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_REFERENCE_FRAME>(<unsigned int>self.value))).decode()

##
# Lists specific and particular timestamps
#
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | IMAGE | Defines the timestamp at the time the frame has been extracted from USB stream. |
# | CURRENT | Defines the timestamp at the time of the function call. |
class TIME_REFERENCE(enum.Enum):
    IMAGE = <int>c_TIME_REFERENCE.TIME_REFERENCE_IMAGE
    CURRENT = <int>c_TIME_REFERENCE.CURRENT
    LAST = <int>c_TIME_REFERENCE.TIME_REFERENCE_LAST

    def __str__(self):
        return to_str(toString(<c_TIME_REFERENCE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_TIME_REFERENCE>(<unsigned int>self.value))).decode()

##
# Gives the spatial mapping state.
# \ingroup SpatialMapping_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | INITIALIZING | The spatial mapping is initializing. |
# | OK | The depth and tracking data were correctly integrated in the fusion algorithm. |
# | NOT_ENOUGH_MEMORY | The maximum memory dedicated to the scanning has been reached, the mesh will no longer be updated. |
# | NOT_ENABLED | Camera.enable_spatial_mapping() wasn't called (or the scanning was stopped and not relaunched). |
# | FPS_TOO_LOW | Effective FPS is too low to give proper results for spatial mapping. Consider using PERFORMANCE parameters ([DEPTH_MODE.PERFORMANCE](\ref DEPTH_MODE), low camera resolution (VGA,HD720), spatial mapping low resolution) |
class SPATIAL_MAPPING_STATE(enum.Enum):
    INITIALIZING = <int>c_SPATIAL_MAPPING_STATE.INITIALIZING
    OK = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_OK
    NOT_ENOUGH_MEMORY = <int>c_SPATIAL_MAPPING_STATE.NOT_ENOUGH_MEMORY
    NOT_ENABLED = <int>c_SPATIAL_MAPPING_STATE.NOT_ENABLED
    FPS_TOO_LOW = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_FPS_TOO_LOW
    LAST = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_LAST

##
# Lists available compression modes for SVO recording.
# \ingroup Video_group
# sl.SVO_COMPRESSION_MODE.LOSSLESS is an improvement of previous lossless compression (used in ZED Explorer), even if size may be bigger, compression time is much faster.
# | Enumerator |                         |
# |------------|-------------------------|
# | LOSSLESS | PNG/ZSTD (lossless) CPU based compression : avg size = 42% (of RAW). |
# | H264 | H264 Lossy GPU based compression : avg size ~= 1% (of RAW). Requires a NVIDIA GPU |
# | H265 | H265 Lossy GPU based compression : avg size ~= 1% (of raw). Requires a NVIDIA GPU |
# | H264_LOSSLESS | H265 Lossless GPU/Hardware based compression: avg size ~= 25% (of RAW). Provides a SSIM/PSNR result (vs RAW) >= 99.9%. Requires a NVIDIA GPU |
# | H265_LOSSLESS | H264 Lossless GPU/Hardware based compression: avg size ~= 25% (of RAW). Provides a SSIM/PSNR result (vs RAW) >= 99.9%. Requires a NVIDIA GPU |
class SVO_COMPRESSION_MODE(enum.Enum):
    LOSSLESS = <int>c_SVO_COMPRESSION_MODE.LOSSLESS
    H264 = <int>c_SVO_COMPRESSION_MODE.H264
    H265 = <int>c_SVO_COMPRESSION_MODE.H265
    H264_LOSSLESS = <int>c_SVO_COMPRESSION_MODE.H264_LOSSLESS
    H265_LOSSLESS = <int>c_SVO_COMPRESSION_MODE.H265_LOSSLESS
    LAST = <int>c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LAST

    def __str__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>(<unsigned int>self.value))).decode()

##
# Lists available memory type.
# \ingroup Core_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | CPU | CPU Memory (Processor side). |
class MEM(enum.Enum):
    CPU = <int>c_MEM.CPU

##
# Lists available copy operation on \ref Mat .
# \ingroup Core_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | CPU_CPU | copy data from CPU to CPU. |
class COPY_TYPE(enum.Enum):
    CPU_CPU = <int>c_COPY_TYPE.CPU_CPU

##
# Lists available \ref Mat formats.
# \ingroup Core_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | F32_C1 | float 1 channel. |
# | F32_C2 | float 2 channels. |
# | F32_C3 | float 3 channels. |
# | F32_C4 | float 4 channels. |
# | U8_C1 | unsigned char 1 channel. |
# | U8_C2 | unsigned char 2 channels. |
# | U8_C3 | unsigned char 3 channels. |
# | U8_C4 | unsigned char 4 channels. |
# | U16_C1 | unsigned short 1 channel. |
# | S8_C4 | signed char 4 channels. |
class MAT_TYPE(enum.Enum):
    F32_C1 = <int>c_MAT_TYPE.F32_C1
    F32_C2 = <int>c_MAT_TYPE.F32_C2
    F32_C3 = <int>c_MAT_TYPE.F32_C3
    F32_C4 = <int>c_MAT_TYPE.F32_C4
    U8_C1 = <int>c_MAT_TYPE.U8_C1
    U8_C2 = <int>c_MAT_TYPE.U8_C2
    U8_C3 = <int>c_MAT_TYPE.U8_C3
    U8_C4 = <int>c_MAT_TYPE.U8_C4
    U16_C1 = <int>c_MAT_TYPE.U16_C1
    S8_C4 = <int>c_MAT_TYPE.S8_C4

##
# Lists available sensor types
# \ingroup Sensors_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | ACCELEROMETER | Three axis Accelerometer sensor to measure the inertial accelerations |
# | GYROSCOPE | Three axis Gyroscope sensor to measure the angular velocities |
# | MAGNETOMETER | Three axis Magnetometer sensor to measure the orientation of the device respect to the earth magnetic field |
# | BAROMETER | Barometer sensor to measure the atmospheric pressure |
class SENSOR_TYPE(enum.Enum):
    ACCELEROMETER = <int>c_SENSOR_TYPE.ACCELEROMETER
    GYROSCOPE = <int>c_SENSOR_TYPE.GYROSCOPE
    MAGNETOMETER = <int>c_SENSOR_TYPE.MAGNETOMETER
    BAROMETER = <int>c_SENSOR_TYPE.BAROMETER

##
# List of the available onboard sensors measurement units. 
# \ingroup Sensors_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | M_SEC_2 | Acceleration [m/s²] |
# | DEG_SEC | Angular velocity [deg/s] |
# | U_T | Magnetic Field [uT] |
# | HPA | Atmospheric pressure [hPa] |
# | CELSIUS | Temperature [°C] |
# | HERTZ | Frequency [Hz] |
class SENSORS_UNIT(enum.Enum):
    M_SEC_2 = <int>c_SENSORS_UNIT.M_SEC_2
    DEG_SEC = <int>c_SENSORS_UNIT.DEG_SEC
    U_T = <int>c_SENSORS_UNIT.U_T
    HPA = <int>c_SENSORS_UNIT.HPA
    CELSIUS = <int>c_SENSORS_UNIT.CELSIUS
    HERTZ = <int>c_SENSORS_UNIT.HERTZ

##
# Lists available object classes
#
# \ingroup Object_group
# 
# | OBJECT_CLASS | Description |
# |-|-|
# | PERSON | For people detection |
# | VEHICLE | For vehicle detection. It can be cars, trucks, buses, motorcycles etc |
# | BAG | For bag detection (backpack, handbag, suitcase) |
# | ANIMAL | For animal detection (cow, sheep, horse, dog, cat, bird, etc) |
# | ELECTRONICS | For electronic device detection (cellphone, laptop, etc) |
# | FRUIT_VEGETABLE | For fruit and vegetable detection (banana, apple, orange, carrot, etc) |
# | SPORT | For sport-related object detection (ball) |
class OBJECT_CLASS(enum.Enum):
    PERSON = <int>c_OBJECT_CLASS.PERSON
    VEHICLE = <int>c_OBJECT_CLASS.VEHICLE
    BAG = <int>c_OBJECT_CLASS.BAG
    ANIMAL = <int>c_OBJECT_CLASS.ANIMAL
    ELECTRONICS = <int>c_OBJECT_CLASS.ELECTRONICS
    FRUIT_VEGETABLE = <int>c_OBJECT_CLASS.FRUIT_VEGETABLE
    SPORT = <int>c_OBJECT_CLASS.SPORT
    LAST = <int>c_OBJECT_CLASS.OBJECT_CLASS_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_CLASS>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_CLASS>(<unsigned int>self.value))).decode()

##
#  Available object subclass, given as hint, when using object tracking an object can change of OBJECT_SUBCLASS while keeping the same OBJECT_CLASS (i.e: frame M: MOTORBIKE, frame N:BICYCLE)
#
# \ingroup Object_group
# 
# | OBJECT_SUBCLASS | OBJECT_CLASS |
# |------------|-------------------------|
# | PERSON | PERSON |
# | PERSON_HEAD | PERSON |
# | BICYCLE | VEHICLE |
# | CAR | VEHICLE |
# | MOTORBIKE | VEHICLE |
# | BUS | VEHICLE |
# | TRUCK | VEHICLE |
# | BOAT | VEHICLES |
# | BACKPACK | BAG |
# | HANDBAG | BAG |
# | SUITCASE | BAG |
# | BIRD | ANIMAL |
# | CAT | ANIMAL |
# | DOG | ANIMAL |
# | HORSE | ANIMAL |
# | SHEEP | ANIMAL |
# | COW | ANIMAL |
# | CELLPHONE | ELECTRONICS |
# | LAPTOP | ELECTRONICS |
# | BANANA | FRUIT_VEGETABLE |
# | APPLE | FRUIT_VEGETABLE |
# | ORANGE | FRUIT_VEGETABLE |
# | CARROT | FRUIT_VEGETABLE |
# | SPORTSBALL | SPORT |
class OBJECT_SUBCLASS(enum.Enum):
    PERSON = <int>c_OBJECT_SUBCLASS.PERSON
    PERSON_HEAD = <int>c_OBJECT_SUBCLASS.PERSON_HEAD
    BICYCLE = <int>c_OBJECT_SUBCLASS.BICYCLE
    CAR = <int>c_OBJECT_SUBCLASS.CAR
    MOTORBIKE = <int>c_OBJECT_SUBCLASS.MOTORBIKE
    BUS = <int>c_OBJECT_SUBCLASS.BUS
    TRUCK = <int>c_OBJECT_SUBCLASS.TRUCK
    BOAT = <int>c_OBJECT_SUBCLASS.BOAT
    BACKPACK = <int>c_OBJECT_SUBCLASS.BACKPACK
    HANDBAG = <int>c_OBJECT_SUBCLASS.HANDBAG
    SUITCASE = <int>c_OBJECT_SUBCLASS.SUITCASE
    BIRD = <int>c_OBJECT_SUBCLASS.BIRD
    CAT = <int>c_OBJECT_SUBCLASS.CAT
    DOG = <int>c_OBJECT_SUBCLASS.DOG
    HORSE = <int>c_OBJECT_SUBCLASS.HORSE
    SHEEP = <int>c_OBJECT_SUBCLASS.SHEEP
    COW = <int>c_OBJECT_SUBCLASS.COW
    CELLPHONE = <int>c_OBJECT_SUBCLASS.CELLPHONE
    LAPTOP = <int>c_OBJECT_SUBCLASS.LAPTOP
    BANANA = <int>c_OBJECT_SUBCLASS.BANANA
    APPLE = <int>c_OBJECT_SUBCLASS.APPLE
    ORANGE = <int>c_OBJECT_SUBCLASS.ORANGE
    CARROT = <int>c_OBJECT_SUBCLASS.CARROT
    SPORTSBALL = <int>c_OBJECT_SUBCLASS.SPORTSBALL
    LAST = <int>c_OBJECT_SUBCLASS.OBJECT_SUBCLASS_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_SUBCLASS>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_SUBCLASS>(<unsigned int>self.value))).decode()

##
# Lists available object tracking states
#
# \ingroup Object_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | OFF | The tracking is not yet initialized, the object ID is not usable |
# | OK | The object is tracked |
# | SEARCHING | The object couldn't be detected in the image and is potentially occluded, the trajectory is estimated |
# | TERMINATE | This is the last searching state of the track, the track will be deleted in the next retrieve_object |
class OBJECT_TRACKING_STATE(enum.Enum):
    OFF = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_OFF
    OK = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_OK
    SEARCHING = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_SEARCHING
    TERMINATE = <int>c_OBJECT_TRACKING_STATE.TERMINATE
    LAST = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_TRACKING_STATE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_TRACKING_STATE>(<unsigned int>self.value))).decode()

##
# Gives the camera flip mode
#
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | OFF  | Default behavior |
# | ON   | Images and camera sensors data are flipped, useful when your camera is mounted upside down |
# | AUTO | In live mode: use the camera orientation (if an IMU is available) to set the flip mode, in SVO mode, read the state of this enum when recorded |
class FLIP_MODE(enum.Enum):
    OFF = <int>c_FLIP_MODE.OFF
    ON = <int>c_FLIP_MODE.ON
    AUTO = <int>c_FLIP_MODE.AUTO

    def __str__(self):
        return to_str(toString(<c_FLIP_MODE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_FLIP_MODE>(<unsigned int>self.value))).decode()

##
# Lists available object action states
#
# \ingroup Object_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | IDLE | The object is staying static. |
# | MOVING | The object is moving. |
class OBJECT_ACTION_STATE(enum.Enum):
    IDLE = <int>c_OBJECT_ACTION_STATE.IDLE
    MOVING = <int>c_OBJECT_ACTION_STATE.OBJECT_ACTION_STATE_MOVING
    LAST = <int>c_OBJECT_ACTION_STATE.OBJECT_ACTION_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_ACTION_STATE>(<unsigned int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_ACTION_STATE>(<unsigned int>self.value))).decode()


##
# Contains data of a detected object such as its bounding_box, label, id and its 3D position.
# \ingroup Object_group
cdef class ObjectData:
    cdef c_ObjectData object_data

    ##
    # Object identification number, used as a reference when tracking the object through the frames.
    # \note Is set to -1 if the object is not currently tracked. 
    @property
    def id(self):
        return self.object_data.id

    @id.setter
    def id(self, int id):
        self.object_data.id = id

    ##
    # Unique ID to help identify and track AI detections. Can be either generated externally, or using \ref generate_unique_id() or left empty
    @property
    def unique_object_id(self):
        if not self.object_data.unique_object_id.empty():
            return self.object_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.object_data.unique_object_id.set(id_.encode())


    ##
    # Object label, forwarded from \ref CustomBoxObjectData when using sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    @property
    def raw_label(self):
        return self.object_data.raw_label

    @raw_label.setter
    def raw_label(self, int raw_label):
        self.object_data.raw_label = raw_label


    ##
    # Object category. Identifies the object type. Can have the following values: \ref OBJECT_CLASS 
    @property
    def label(self):
        return OBJECT_CLASS(<int>self.object_data.label)

    @label.setter
    def label(self, label):
        if isinstance(label, OBJECT_CLASS):
            self.object_data.label = <c_OBJECT_CLASS>(<unsigned int>label.value)
        else:
            raise TypeError("Argument is not of OBJECT_CLASS type.")
   
    ##
    # Object sublabel. Identifies the object subclass. Can have the following values: \ref OBJECT_SUBCLASS
    @property
    def sublabel(self):
        return OBJECT_SUBCLASS(<int>self.object_data.sublabel)

    @sublabel.setter
    def sublabel(self, sublabel):
        if isinstance(sublabel, OBJECT_SUBCLASS):
            self.object_data.sublabel = <c_OBJECT_SUBCLASS>(<unsigned int>sublabel.value)
        else:
            raise TypeError("Argument is not of OBJECT_SUBCLASS type.")
    
    ##
    # Defines the object tracking state. Can have the following values: \ref OBJECT_TRACKING_STATE
    @property
    def tracking_state(self):
        return OBJECT_TRACKING_STATE(<int>self.object_data.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.object_data.tracking_state = <c_OBJECT_TRACKING_STATE>(<unsigned int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # Defines the object action state. Can have the following values: \ref OBJECT_ACTION_STATE
    @property
    def action_state(self):
        return OBJECT_ACTION_STATE(<int>self.object_data.action_state)

    @action_state.setter
    def action_state(self, action_state):
        if isinstance(action_state, OBJECT_ACTION_STATE):
            self.object_data.action_state = <c_OBJECT_ACTION_STATE>(<unsigned int>action_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_ACTION_STATE type.")

    ##
    # Defines the object 3D centroid. Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def position(self):
        cdef np.ndarray position = np.zeros(3)
        for i in range(3):
            position[i] = self.object_data.position[i]
        return position

    @position.setter
    def position(self, np.ndarray position):
        for i in range(3):
            self.object_data.position[i] = position[i]

    ##
    # Defines the object 3D velocity. Defined in \ref InitParameters.coordinate_units / s , expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def velocity(self):
        cdef np.ndarray velocity = np.zeros(3)
        for i in range(3):
            velocity[i] = self.object_data.velocity[i]
        return velocity

    @velocity.setter
    def velocity(self, np.ndarray velocity):
        for i in range(3):
            self.object_data.velocity[i] = velocity[i]

    ##
    # 3D bounding box of the person represented as eight 3D points. Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    # \code
    #   1 ------ 2
    #  /        /|
    # 0 ------ 3 |
    # | Object | 6
    # |        |/
    # 4 ------ 7
    # \endcode
    # \note Only available if ObjectDetectionParameters.enable_tracking is activated 
    @property
    def bounding_box(self):
        cdef np.ndarray arr = np.zeros((self.object_data.bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.object_data.bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.object_data.bounding_box[i].ptr()[j]
        return arr

    @bounding_box.setter
    def bounding_box(self, np.ndarray coordinates):
        cdef Vector3[float] vec
        self.object_data.bounding_box.clear()
        for i in range(8):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            vec[2] = coordinates[i][2]
            self.object_data.bounding_box.push_back(vec)

    ##
    # 2D bounding box of the person represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution, where [0,0] is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self):
        cdef np.ndarray arr = np.zeros((self.object_data.bounding_box_2d.size(), 2))
        for i in range(self.object_data.bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.object_data.bounding_box_2d[i].ptr()[j]
        return arr

    @bounding_box_2d.setter
    def bounding_box_2d(self, np.ndarray coordinates):
        cdef Vector2[unsigned int] vec
        self.object_data.bounding_box_2d.clear()
        for i in range(4):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            self.object_data.bounding_box_2d.push_back(vec)

    ##
    # Defines the detection confidence value of the object.
    # Values can range from 0 to 100, where lower confidence values mean that the object might not be localized perfectly or that the label (\ref OBJECT_CLASS) is uncertain.
    @property
    def confidence(self):
        return self.object_data.confidence

    @confidence.setter
    def confidence(self, float confidence):
        self.object_data.confidence = confidence

    ##
    # Defines for the bounding_box_2d the pixels which really belong to the object (set to 255) and those of the background (set to 0).
    # \warning The mask information is available only for tracked objects ([OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE)) that have a valid depth. Otherwise, it will not be initialized ([mask.is_init](\ref Mat.is_init) == False) 
    @property
    def mask(self):
        mat = Mat()
        mat.mat = self.object_data.mask
        return mat

    @mask.setter
    def mask(self, Mat mat):
        self.object_data.mask = mat.mat

    ##
    # 3D object dimensions: width, height, length 
    # \note Only available if ObjectDetectionParameters.enable_tracking is activated 
    @property
    def dimensions(self):
        cdef np.ndarray dimensions = np.zeros(3)
        for i in range(3):
            dimensions[i] = self.object_data.dimensions[i]
        return dimensions

    @dimensions.setter
    def dimensions(self, np.ndarray dimensions):
        for i in range(3):
            self.object_data.dimensions[i] = dimensions[i]
   
    ##
    # 3D bounding box of the person head, only available in [BODY_TRACKING_MODEL.HUMAN_BODY*](\ref OBJECT_DETECTION_MODEL), represented as eight 3D points. 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def head_bounding_box(self):
        cdef np.ndarray arr = np.zeros((self.object_data.head_bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.object_data.head_bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.object_data.head_bounding_box[i].ptr()[j]
        return arr

    ##
    # 2D bounding box of the person head, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL), represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution. 
    @property
    def head_bounding_box_2d(self):
        cdef np.ndarray arr = np.zeros((self.object_data.head_bounding_box_2d.size(), 2))
        for i in range(self.object_data.head_bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.object_data.head_bounding_box_2d[i].ptr()[j]
        return arr

    ##
    # 3D head centroid, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL). 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def head_position(self):
        cdef np.ndarray head_position = np.zeros(3)
        for i in range(3):
            head_position[i] = self.object_data.head_position[i]
        return head_position

    @head_position.setter
    def head_position(self, np.ndarray head_position):
        for i in range(3):
            self.object_data.head_position[i] = head_position[i]

    ##
    # Position covariance
    @property
    def position_covariance(self):
        cdef np.ndarray arr = np.zeros(6)
        for i in range(6) :
            arr[i] = self.object_data.position_covariance[i]
        return arr

    @position_covariance.setter
    def position_covariance(self, np.ndarray position_covariance_):
        for i in range(6) :
            self.object_data.position_covariance[i] = position_covariance_[i]


##
# Contains data of a detected object such as its bounding_box, label, id and its 3D position.
# \ingroup Body_group
cdef class BodyData:
    cdef c_BodyData body_data

    ##
    # Object identification number, used as a reference when tracking the object through the frames.
    # \note Only available if \ref BodyTrackingParameters.enable_tracking is activated else set to -1.
    @property
    def id(self):
        return self.body_data.id

    @id.setter
    def id(self, int id):
        self.body_data.id = id

    ##
    # Unique ID to help identify and track AI detections. Can be either generated externally, or using \ref generate_unique_id() or left empty
    @property
    def unique_object_id(self):
        if not self.body_data.unique_object_id.empty():
            return self.body_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.body_data.unique_object_id.set(id_.encode())

    ##
    # Defines the object tracking state. Can have the following values: \ref OBJECT_TRACKING_STATE
    @property
    def tracking_state(self):
        return OBJECT_TRACKING_STATE(<int>self.body_data.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.body_data.tracking_state = <c_OBJECT_TRACKING_STATE>(<unsigned int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # Defines the object action state. Can have the following values: \ref OBJECT_ACTION_STATE
    @property
    def action_state(self):
        return OBJECT_ACTION_STATE(<int>self.body_data.action_state)

    @action_state.setter
    def action_state(self, action_state):
        if isinstance(action_state, OBJECT_ACTION_STATE):
            self.body_data.action_state = <c_OBJECT_ACTION_STATE>(<unsigned int>action_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_ACTION_STATE type.")

    ##
    # Defines the object 3D centroid. Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def position(self):
        cdef np.ndarray position = np.zeros(3)
        for i in range(3):
            position[i] = self.body_data.position[i]
        return position

    @position.setter
    def position(self, np.ndarray position):
        for i in range(3):
            self.body_data.position[i] = position[i]

    ##
    # Defines the object 3D velocity. Defined in \ref InitParameters.coordinate_units / s , expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def velocity(self):
        cdef np.ndarray velocity = np.zeros(3)
        for i in range(3):
            velocity[i] = self.body_data.velocity[i]
        return velocity

    @velocity.setter
    def velocity(self, np.ndarray velocity):
        for i in range(3):
            self.body_data.velocity[i] = velocity[i]

    ##
    # 3D bounding box of the person represented as eight 3D points. Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    # \code
    #   1 ------ 2
    #  /        /|
    # 0 ------ 3 |
    # | Object | 6
    # |        |/
    # 4 ------ 7
    # \endcode
    # \note Only available if ObjectDetectionParameters.enable_tracking is activated 
    @property
    def bounding_box(self):
        cdef np.ndarray arr = np.zeros((self.body_data.bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.body_data.bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.body_data.bounding_box[i].ptr()[j]
        return arr

    @bounding_box.setter
    def bounding_box(self, np.ndarray coordinates):
        cdef Vector3[float] vec
        self.body_data.bounding_box.clear()
        for i in range(8):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            vec[2] = coordinates[i][2]
            self.body_data.bounding_box.push_back(vec)

    ##
    # 2D bounding box of the person represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution, where [0,0] is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self):
        cdef np.ndarray arr = np.zeros((self.body_data.bounding_box_2d.size(), 2))
        for i in range(self.body_data.bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.body_data.bounding_box_2d[i].ptr()[j]
        return arr

    @bounding_box_2d.setter
    def bounding_box_2d(self, np.ndarray coordinates):
        cdef Vector2[unsigned int] vec
        self.body_data.bounding_box_2d.clear()
        for i in range(4):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            self.body_data.bounding_box_2d.push_back(vec)

    ##
    # Defines the detection confidence value of the object.
    # Values can range from 0 to 100, where lower confidence values mean that the object might not be localized perfectly or that the label (\ref OBJECT_CLASS) is uncertain.
    @property
    def confidence(self):
        return self.body_data.confidence

    @confidence.setter
    def confidence(self, float confidence):
        self.body_data.confidence = confidence

    ##
    # A sample of the associated position covariance
    @property
    def keypoints_covariance(self):
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint_covariances.size(), 6), dtype=np.float32)
        for i in range(self.body_data.keypoint_covariances.size()):
            for j in range(6):
                arr[i,j] = self.body_data.keypoint_covariances[i][j]
        return arr

    ##
    # Keypoint covariance
    @property
    def keypoints_covariance(self):
        result = []
        for i in range(6):
            subresult = []
            for j in range(6):
                subresult.append(self.body_data.keypoint_covariances[i][j])
            result.append(subresult)
        return result

    @keypoints_covariance.setter
    def keypoints_covariance(self, value: list):
        if isinstance(value, list):
            if len(value) == 6:
                for i in range(6):
                    if len( value[i]) != 6:
                        raise TypeError("Argument is not of 6x6 list.")
                    for j in range(6):
                        self.body_data.keypoint_covariances[i][j] = value[i][j]
                return
        raise TypeError("Argument is not of 6x6 list.")

    ##
    # Position covariance
    @property
    def position_covariance(self):
        cdef np.ndarray arr = np.zeros(6)
        for i in range(6) :
            arr[i] = self.body_data.position_covariance[i]
        return arr

    @position_covariance.setter
    def position_covariance(self, np.ndarray position_covariance_):
        for i in range(6) :
            self.body_data.position_covariance[i] = position_covariance_[i]


    ##
    # Defines for the bounding_box_2d the pixels which really belong to the object (set to 255) and those of the background (set to 0).
    # \warning The mask information is available only for tracked objects ([OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE)) that have a valid depth. Otherwise, it will not be initialized ([mask.is_init](\ref Mat.is_init) == False) 
    @property
    def mask(self):
        mat = Mat()
        mat.mat = self.body_data.mask
        return mat

    @mask.setter
    def mask(self, Mat mat):
        self.body_data.mask = mat.mat

    ##
    # 3D object dimensions: width, height, length 
    # \note Only available if ObjectDetectionParameters.enable_tracking is activated 
    @property
    def dimensions(self):
        cdef np.ndarray dimensions = np.zeros(3)
        for i in range(3):
            dimensions[i] = self.body_data.dimensions[i]
        return dimensions

    @dimensions.setter
    def dimensions(self, np.ndarray dimensions):
        for i in range(3):
            self.body_data.dimensions[i] = dimensions[i]
   
    ##
    # A set of useful points representing the human body, expressed in 3D and only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL). 
    # We use a classic 18 points representation, the keypoint semantic and order is given by \ref BODY_18_PARTS
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    # \warning in some cases, eg. body partially out of the image, some keypoints can not be detected, they will have negative coordinates. 
    @property
    def keypoint(self):
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint.size(), 3), dtype=np.float32)
        for i in range(self.body_data.keypoint.size()):
            for j in range(3):
                arr[i,j] = self.body_data.keypoint[i].ptr()[j]
        return arr

    ##
    # 2D keypoint of the object, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL)
    # \warning in some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected, they will have non finite values. 
    @property
    def keypoint_2d(self):
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint_2d.size(), 2))
        for i in range(self.body_data.keypoint_2d.size()):
            for j in range(2):
                arr[i,j] = self.body_data.keypoint_2d[i].ptr()[j]
        return arr

    
    ##
    # 3D bounding box of the person head, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL), represented as eight 3D points. 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def head_bounding_box(self):
        cdef np.ndarray arr = np.zeros((self.body_data.head_bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.body_data.head_bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.body_data.head_bounding_box[i].ptr()[j]
        return arr

    ##
    # 2D bounding box of the person head, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL), represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution. 
    @property
    def head_bounding_box_2d(self):
        cdef np.ndarray arr = np.zeros((self.body_data.head_bounding_box_2d.size(), 2))
        for i in range(self.body_data.head_bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.body_data.head_bounding_box_2d[i].ptr()[j]
        return arr

    ##
    # 3D head centroid, only available in [DETECTION_MODEL.HUMAN_BODY*](\ref DETECTION_MODEL). 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame
    @property
    def head_position(self):
        cdef np.ndarray head_position = np.zeros(3)
        for i in range(3):
            head_position[i] = self.body_data.head_position[i]
        return head_position

    @head_position.setter
    def head_position(self, np.ndarray head_position):
        for i in range(3):
            self.body_data.head_position[i] = head_position[i]

    ##
    # Per keypoint detection confidence, can not be lower than the \ref ObjectDetectionRuntimeParameters::detection_confidence_threshold.
    # \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected, they will have non finite values.
    @property
    def keypoint_confidence(self):
        cdef np.ndarray out_arr = np.zeros(self.body_data.keypoint_confidence.size())
        for i in range(self.body_data.keypoint_confidence.size()):
            out_arr[i] = self.body_data.keypoint_confidence[i]
        return out_arr

    ##
    # Per keypoint local position (the position of the child keypoint with respect to its parent expressed in its parent coordinate frame)
    # \note it is expressed in [sl.REFERENCE_FRAME.CAMERA](\ref REFERENCE_FRAME) or [sl.REFERENCE_FRAME.WORLD](\ref REFERENCE_FRAME)
    # \warning Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL) and with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def local_position_per_joint(self):
        cdef np.ndarray arr = np.zeros((self.body_data.local_position_per_joint.size(), 3), dtype=np.float32)
        for i in range(self.body_data.local_position_per_joint.size()):
            for j in range(3):
                arr[i,j] = self.body_data.local_position_per_joint[i].ptr()[j]
        return arr

    ##
    # Per keypoint local orientation
    # \note the orientation is represented by a quaternion which is stored in a numpy array of size 4 [qx,qy,qz,qw]
    # \warning Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL) and with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def local_orientation_per_joint(self):
        cdef np.ndarray arr = np.zeros((self.body_data.local_orientation_per_joint.size(), 4), dtype=np.float32)
        for i in range(self.body_data.local_orientation_per_joint.size()):
            for j in range(4):
                arr[i,j] = self.body_data.local_orientation_per_joint[i].ptr()[j]
        return arr

    ##
    # Global root orientation of the skeleton. The orientation is also represented by a quaternion with the same format as \ref local_orientation_per_joint
    # \note the global root position is already accessible in \ref keypoint attribute by using the root index of a given \ref sl.BODY_FORMAT
    # \warning Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL) and with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def global_root_orientation(self):
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = self.body_data.global_root_orientation[i]
        return arr

##
# Generates a UUID like unique ID to help identify and track AI detections
# \ingroup Object_group
def generate_unique_id():
    return to_str(c_generate_unique_id()).decode()       

##
# Container to store the externally detected objects. The objects can be ingested using \ref sl.Camera.ingest_custom_box_objects() 
# functions to extract 3D information and tracking over time
# \ingroup Object_group
cdef class CustomBoxObjectData:
    cdef c_CustomBoxObjectData custom_box_object_data

    ##
    # Unique ID to help identify and track AI detections. Can be either generated externally, or using \ref generate_unique_id() or left empty
    @property
    def unique_object_id(self):
        if not self.custom_box_object_data.unique_object_id.empty():
            return self.custom_box_object_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.custom_box_object_data.unique_object_id.set(id_.encode())

    ##
    # 2D bounding box of the object represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution, where [0,0] is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self):
        cdef np.ndarray arr = np.zeros((self.custom_box_object_data.bounding_box_2d.size(), 2))
        for i in range(self.custom_box_object_data.bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.custom_box_object_data.bounding_box_2d[i].ptr()[j]
        return arr

    @bounding_box_2d.setter
    def bounding_box_2d(self, np.ndarray coordinates):
        cdef Vector2[unsigned int] vec
        self.custom_box_object_data.bounding_box_2d.clear()
        for i in range(4):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            self.custom_box_object_data.bounding_box_2d.push_back(vec)

    ##
    # Object label, this information is passed-through and can be used to improve object tracking 
    @property
    def label(self):
        return self.custom_box_object_data.label

    @label.setter
    def label(self, int label):
        self.custom_box_object_data.label = label

    ##
    # Detection confidence. Should be [0-1]. It can be used to improve the object tracking
    @property
    def probability(self):
        return self.custom_box_object_data.probability

    @probability.setter
    def probability(self, float probability):
        self.custom_box_object_data.probability = probability

    ##
    # Provides hypothesis about the object movements (degrees of freedom) to improve the object tracking
    # \n True: means 2 DoF projected alongside the floor plane, It is the default for objects standing on the ground such as person, vehicle, etc
    # \n False: 6 DoF full 3D movements are allowed
    @property
    def is_grounded(self):
        return self.custom_box_object_data.is_grounded

    @is_grounded.setter
    def is_grounded(self, bool is_grounded):
        self.custom_box_object_data.is_grounded = is_grounded

##
# \brief Semantic of human body parts and order of \ref sl.ObjectData.keypoint for [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT)
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | NOSE |  |
# | NECK |  |
# | RIGHT_SHOULDER |  |
# | RIGHT_ELBOW |  |
# | RIGHT_WRIST |  |
# | LEFT_SHOULDER |  |
# | LEFT_ELBOW |  |
# | LEFT_WRIST |  |
# | RIGHT_HIP |  |
# | RIGHT_KNEE |  |
# | RIGHT_ANKLE |  |
# | LEFT_HIP |  |
# | LEFT_KNEE |  |
# | LEFT_ANKLE |  |
# | RIGHT_EYE |  |
# | LEFT_EYE |  |
# | RIGHT_EAR |  |
# | LEFT_EAR |  |
class BODY_18_PARTS(enum.Enum):
    NOSE = <int>c_BODY_18_PARTS.NOSE
    NECK = <int>c_BODY_18_PARTS.NECK
    RIGHT_SHOULDER = <int>c_BODY_18_PARTS.RIGHT_SHOULDER
    RIGHT_ELBOW = <int>c_BODY_18_PARTS.RIGHT_ELBOW
    RIGHT_WRIST = <int>c_BODY_18_PARTS.RIGHT_WRIST
    LEFT_SHOULDER = <int>c_BODY_18_PARTS.LEFT_SHOULDER
    LEFT_ELBOW = <int>c_BODY_18_PARTS.LEFT_ELBOW
    LEFT_WRIST = <int>c_BODY_18_PARTS.LEFT_WRIST
    RIGHT_HIP = <int>c_BODY_18_PARTS.RIGHT_HIP
    RIGHT_KNEE = <int>c_BODY_18_PARTS.RIGHT_KNEE
    RIGHT_ANKLE = <int>c_BODY_18_PARTS.RIGHT_ANKLE
    LEFT_HIP = <int>c_BODY_18_PARTS.LEFT_HIP
    LEFT_KNEE = <int>c_BODY_18_PARTS.LEFT_KNEE
    LEFT_ANKLE = <int>c_BODY_18_PARTS.LEFT_ANKLE
    RIGHT_EYE = <int>c_BODY_18_PARTS.RIGHT_EYE
    LEFT_EYE = <int>c_BODY_18_PARTS.LEFT_EYE
    RIGHT_EAR = <int>c_BODY_18_PARTS.RIGHT_EAR
    LEFT_EAR = <int>c_BODY_18_PARTS.LEFT_EAR
    LAST = <int>c_BODY_18_PARTS.LAST

##
# \brief Semantic of human body parts and order of \ref sl.ObjectData.keypoint for [sl.BODY_FORMAT.POSE_32](\ref BODY_FORMAT)
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | PELVIS |  |
# | NAVAL_SPINE |  |
# | CHEST_SPINE |  |
# | NECK |  |
# | LEFT_CLAVICLE |  |
# | LEFT_SHOULDER |  |
# | LEFT_ELBOW |  |
# | LEFT_WRIST |  |
# | LEFT_HAND |  |
# | LEFT_HANDTIP |  |
# | LEFT_THUMB |  |
# | RIGHT_CLAVICLE |  |
# | RIGHT_SHOULDER |  |
# | RIGHT_ELBOW |  |
# | RIGHT_WRIST |  |
# | RIGHT_HAND |  |
# | RIGHT_HANDTIP |  |
# | RIGHT_THUMB |  |
# | LEFT_HIP |  |
# | LEFT_KNEE |  |
# | LEFT_ANKLE |  |
# | LEFT_FOOT |  |
# | RIGHT_HIP |  |
# | RIGHT_KNEE |  |
# | RIGHT_ANKLE |  |
# | RIGHT_FOOT |  |
# | HEAD |  |
# | NOSE |  |
# | LEFT_EYE |  |
# | LEFT_EAR |  |
# | RIGHT_EYE |  |
# | RIGHT_EAR |  |
# | LEFT_HEEL |  |
# | RIGHT_HEEL |  |
class BODY_34_PARTS(enum.Enum):
    PELVIS = <int>c_BODY_34_PARTS.PELVIS 
    NAVAL_SPINE = <int>c_BODY_34_PARTS.NAVAL_SPINE 
    CHEST_SPINE = <int>c_BODY_34_PARTS.CHEST_SPINE 
    NECK = <int>c_BODY_34_PARTS.NECK 
    LEFT_CLAVICLE = <int>c_BODY_34_PARTS.LEFT_CLAVICLE 
    LEFT_SHOULDER = <int>c_BODY_34_PARTS.LEFT_SHOULDER 
    LEFT_ELBOW = <int>c_BODY_34_PARTS.LEFT_ELBOW 
    LEFT_WRIST = <int>c_BODY_34_PARTS.LEFT_WRIST 
    LEFT_HAND = <int>c_BODY_34_PARTS.LEFT_HAND 
    LEFT_HANDTIP = <int>c_BODY_34_PARTS.LEFT_HANDTIP 
    LEFT_THUMB = <int>c_BODY_34_PARTS.LEFT_THUMB 
    RIGHT_CLAVICLE = <int>c_BODY_34_PARTS.RIGHT_CLAVICLE  
    RIGHT_SHOULDER = <int>c_BODY_34_PARTS.RIGHT_SHOULDER 
    RIGHT_ELBOW = <int>c_BODY_34_PARTS.RIGHT_ELBOW 
    RIGHT_WRIST = <int>c_BODY_34_PARTS.RIGHT_WRIST 
    RIGHT_HAND = <int>c_BODY_34_PARTS.RIGHT_HAND 
    RIGHT_HANDTIP = <int>c_BODY_34_PARTS.RIGHT_HANDTIP 
    RIGHT_THUMB = <int>c_BODY_34_PARTS.RIGHT_THUMB 
    LEFT_HIP = <int>c_BODY_34_PARTS.LEFT_HIP 
    LEFT_KNEE = <int>c_BODY_34_PARTS.LEFT_KNEE 
    LEFT_ANKLE = <int>c_BODY_34_PARTS.LEFT_ANKLE 
    LEFT_FOOT = <int>c_BODY_34_PARTS.LEFT_FOOT 
    RIGHT_HIP = <int>c_BODY_34_PARTS.RIGHT_HIP 
    RIGHT_KNEE = <int>c_BODY_34_PARTS.RIGHT_KNEE 
    RIGHT_ANKLE = <int>c_BODY_34_PARTS.RIGHT_ANKLE 
    RIGHT_FOOT = <int>c_BODY_34_PARTS.RIGHT_FOOT 
    HEAD = <int>c_BODY_34_PARTS.HEAD 
    NOSE = <int>c_BODY_34_PARTS.NOSE 
    LEFT_EYE = <int>c_BODY_34_PARTS.LEFT_EYE 
    LEFT_EAR = <int>c_BODY_34_PARTS.LEFT_EAR 
    RIGHT_EYE = <int>c_BODY_34_PARTS.RIGHT_EYE 
    RIGHT_EAR = <int>c_BODY_34_PARTS.RIGHT_EAR 
    LEFT_HEEL = <int>c_BODY_34_PARTS.LEFT_HEEL
    RIGHT_HEEL = <int>c_BODY_34_PARTS.RIGHT_HEEL
    LAST = <int>c_BODY_34_PARTS.LAST

##
# \brief Semantic of human body parts and order of \ref sl.ObjectData.keypoint for [sl.BODY_FORMAT.POSE_38](\ref BODY_FORMAT)
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | PELVIS |  |
# | SPINE_1 |  |
# | SPINE_2 |  |
# | SPINE_3 |  |
# | NECK |  |
# | NOSE |  |
# | LEFT_EYE |  |
# | RIGHT_EYE |  |
# | LEFT_EAR |  |
# | RIGHT_EAR |  |
# | LEFT_CLAVICLE |  |
# | RIGHT_CLAVICLE |  |
# | LEFT_SHOULDER |  |
# | RIGHT_SHOULDER |  |
# | LEFT_ELBOW |  |
# | RIGHT_ELBOW |  |
# | LEFT_WRIST |  |
# | RIGHT_WRIST |  |
# | LEFT_HIP |  |
# | RIGHT_HIP |  |
# | LEFT_KNEE |  |
# | RIGHT_KNEE |  |
# | LEFT_ANKLE |  |
# | RIGHT_ANKLE |  |
# | LEFT_BIG_TOE |  |
# | RIGHT_BIG_TOE |  |
# | LEFT_SMALL_TOE |  |
# | RIGHT_SMALL_TOE |  |
# | LEFT_HEEL |  |
# | RIGHT_HEEL |  |
# | LEFT_HAND_THUMB_4 |  |
# | RIGHT_HAND_THUMB_4 |  |
# | LEFT_HAND_INDEX_1 |  |
# | RIGHT_HAND_INDEX_1 |  |
# | LEFT_HAND_MIDDLE_4 |  |
# | RIGHT_HAND_MIDDLE_4 |  |
# | LEFT_HAND_PINKY_1 |  |
# | RIGHT_HAND_PINKY_1 |  |
class BODY_38_PARTS(enum.Enum):
    PELVIS = <int>c_BODY_38_PARTS.PELVIS 
    SPINE_1 = <int>c_BODY_38_PARTS.SPINE_1 
    SPINE_2 = <int>c_BODY_38_PARTS.SPINE_2 
    SPINE_3 = <int>c_BODY_38_PARTS.SPINE_3 
    NECK = <int>c_BODY_38_PARTS.NECK 
    NOSE = <int>c_BODY_38_PARTS.NOSE 
    LEFT_EYE = <int>c_BODY_38_PARTS.LEFT_EYE 
    RIGHT_EYE = <int>c_BODY_38_PARTS.RIGHT_EYE 
    LEFT_EAR = <int>c_BODY_38_PARTS.LEFT_EAR         
    RIGHT_EAR = <int>c_BODY_38_PARTS.RIGHT_EAR         
    LEFT_CLAVICLE = <int>c_BODY_38_PARTS.LEFT_CLAVICLE 
    RIGHT_CLAVICLE = <int>c_BODY_38_PARTS.RIGHT_CLAVICLE  
    LEFT_SHOULDER = <int>c_BODY_38_PARTS.LEFT_SHOULDER 
    RIGHT_SHOULDER = <int>c_BODY_38_PARTS.RIGHT_SHOULDER 
    LEFT_ELBOW = <int>c_BODY_38_PARTS.LEFT_ELBOW 
    RIGHT_ELBOW = <int>c_BODY_38_PARTS.RIGHT_ELBOW 
    LEFT_WRIST = <int>c_BODY_38_PARTS.LEFT_WRIST 
    RIGHT_WRIST = <int>c_BODY_38_PARTS.RIGHT_WRIST
    LEFT_HIP = <int>c_BODY_38_PARTS.LEFT_HIP 
    RIGHT_HIP = <int>c_BODY_38_PARTS.RIGHT_HIP 
    LEFT_KNEE = <int>c_BODY_38_PARTS.LEFT_KNEE 
    RIGHT_KNEE = <int>c_BODY_38_PARTS.RIGHT_KNEE 
    LEFT_ANKLE = <int>c_BODY_38_PARTS.LEFT_ANKLE 
    RIGHT_ANKLE = <int>c_BODY_38_PARTS.RIGHT_ANKLE 
    LEFT_BIG_TOE = <int>c_BODY_38_PARTS.LEFT_BIG_TOE 
    RIGHT_BIG_TOE = <int>c_BODY_38_PARTS.RIGHT_BIG_TOE 
    LEFT_SMALL_TOE = <int>c_BODY_38_PARTS.LEFT_SMALL_TOE 
    RIGHT_SMALL_TOE = <int>c_BODY_38_PARTS.RIGHT_SMALL_TOE 
    LEFT_HEEL = <int>c_BODY_38_PARTS.LEFT_HEEL 
    RIGHT_HEEL = <int>c_BODY_38_PARTS.RIGHT_HEEL    
    LEFT_HAND_THUMB_4 = <int>c_BODY_38_PARTS.LEFT_HAND_THUMB_4 
    RIGHT_HAND_THUMB_4 = <int>c_BODY_38_PARTS.RIGHT_HAND_THUMB_4 
    LEFT_HAND_INDEX_1 = <int>c_BODY_38_PARTS.LEFT_HAND_INDEX_1 
    RIGHT_HAND_INDEX_1 = <int>c_BODY_38_PARTS.RIGHT_HAND_INDEX_1 
    LEFT_HAND_MIDDLE_4 = <int>c_BODY_38_PARTS.LEFT_HAND_MIDDLE_4 
    RIGHT_HAND_MIDDLE_4 = <int>c_BODY_38_PARTS.RIGHT_HAND_MIDDLE_4 
    LEFT_HAND_PINKY_1 = <int>c_BODY_38_PARTS.LEFT_HAND_PINKY_1 
    RIGHT_HAND_PINKY_1 = <int>c_BODY_38_PARTS.RIGHT_HAND_PINKY_1 
    LAST = <int>c_BODY_38_PARTS.LAST


##
# \brief List of supported skeleton body model
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | BODY_18 | 18 keypoint model of COCO 18. \note local keypoint angle and position are not available with this format.  |
# | BODY_34 | 34 keypoint model. \note local keypoint angle and position are available. \warning The SDK will automatically enable fitting. |
# | BODY_38 | 38 keypoint model. \note local keypoint angle and position are available. |
class BODY_FORMAT(enum.Enum):
    BODY_18 = <int>c_BODY_FORMAT.BODY_18
    BODY_34 = <int>c_BODY_FORMAT.BODY_34
    BODY_38 = <int>c_BODY_FORMAT.BODY_38
    LAST = <int>c_BODY_FORMAT.LAST

##
# \brief Lists of supported skeleton body selection model
# \ingroup Body_group
class BODY_KEYPOINTS_SELECTION(enum.Enum):
    FULL = <int>c_BODY_KEYPOINTS_SELECTION.FULL
    UPPER_BODY = <int>c_BODY_KEYPOINTS_SELECTION.UPPER_BODY
    LAST = <int>c_BODY_KEYPOINTS_SELECTION.LAST

##
# \brief Links of human body keypoints for [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT), useful for display.
# \ingroup Body_group
BODY_18_BONES = [ (BODY_18_PARTS.NOSE, BODY_18_PARTS.NECK),
                (BODY_18_PARTS.NECK, BODY_18_PARTS.RIGHT_SHOULDER),
                (BODY_18_PARTS.RIGHT_SHOULDER, BODY_18_PARTS.RIGHT_ELBOW),
                (BODY_18_PARTS.RIGHT_ELBOW, BODY_18_PARTS.RIGHT_WRIST),
                (BODY_18_PARTS.NECK, BODY_18_PARTS.LEFT_SHOULDER),
                (BODY_18_PARTS.LEFT_SHOULDER, BODY_18_PARTS.LEFT_ELBOW),
                (BODY_18_PARTS.LEFT_ELBOW, BODY_18_PARTS.LEFT_WRIST),
                (BODY_18_PARTS.RIGHT_SHOULDER, BODY_18_PARTS.RIGHT_HIP),
                (BODY_18_PARTS.RIGHT_HIP, BODY_18_PARTS.RIGHT_KNEE),
                (BODY_18_PARTS.RIGHT_KNEE, BODY_18_PARTS.RIGHT_ANKLE),
                (BODY_18_PARTS.LEFT_SHOULDER, BODY_18_PARTS.LEFT_HIP),
                (BODY_18_PARTS.LEFT_HIP, BODY_18_PARTS.LEFT_KNEE),
                (BODY_18_PARTS.LEFT_KNEE, BODY_18_PARTS.LEFT_ANKLE),
                (BODY_18_PARTS.RIGHT_SHOULDER, BODY_18_PARTS.LEFT_SHOULDER),
                (BODY_18_PARTS.RIGHT_HIP, BODY_18_PARTS.LEFT_HIP),
                (BODY_18_PARTS.NOSE, BODY_18_PARTS.RIGHT_EYE),
                (BODY_18_PARTS.RIGHT_EYE, BODY_18_PARTS.RIGHT_EAR),
                (BODY_18_PARTS.NOSE, BODY_18_PARTS.LEFT_EYE),
                (BODY_18_PARTS.LEFT_EYE, BODY_18_PARTS.LEFT_EAR) ]

##
# \brief Links of human body keypoints for [sl.BODY_FORMAT.BODY_34](\ref BODY_FORMAT), useful for display.
# \ingroup Body_group
BODY_34_BONES = [ 
        (BODY_34_PARTS.PELVIS, BODY_34_PARTS.NAVAL_SPINE),
		(BODY_34_PARTS.NAVAL_SPINE, BODY_34_PARTS.CHEST_SPINE),
		(BODY_34_PARTS.CHEST_SPINE, BODY_34_PARTS.LEFT_CLAVICLE),
		(BODY_34_PARTS.LEFT_CLAVICLE, BODY_34_PARTS.LEFT_SHOULDER),
		(BODY_34_PARTS.LEFT_SHOULDER, BODY_34_PARTS.LEFT_ELBOW),
		(BODY_34_PARTS.LEFT_ELBOW, BODY_34_PARTS.LEFT_WRIST),
		(BODY_34_PARTS.LEFT_WRIST, BODY_34_PARTS.LEFT_HAND),
		(BODY_34_PARTS.LEFT_HAND, BODY_34_PARTS.LEFT_HANDTIP),
		(BODY_34_PARTS.LEFT_WRIST, BODY_34_PARTS.LEFT_THUMB),
		(BODY_34_PARTS.CHEST_SPINE, BODY_34_PARTS.RIGHT_CLAVICLE),
		(BODY_34_PARTS.RIGHT_CLAVICLE, BODY_34_PARTS.RIGHT_SHOULDER),
		(BODY_34_PARTS.RIGHT_SHOULDER, BODY_34_PARTS.RIGHT_ELBOW),
		(BODY_34_PARTS.RIGHT_ELBOW, BODY_34_PARTS.RIGHT_WRIST),
		(BODY_34_PARTS.RIGHT_WRIST, BODY_34_PARTS.RIGHT_HAND),
		(BODY_34_PARTS.RIGHT_HAND, BODY_34_PARTS.RIGHT_HANDTIP),
		(BODY_34_PARTS.RIGHT_WRIST, BODY_34_PARTS.RIGHT_THUMB),
		(BODY_34_PARTS.PELVIS, BODY_34_PARTS.LEFT_HIP),
		(BODY_34_PARTS.LEFT_HIP, BODY_34_PARTS.LEFT_KNEE),
		(BODY_34_PARTS.LEFT_KNEE, BODY_34_PARTS.LEFT_ANKLE),
		(BODY_34_PARTS.LEFT_ANKLE, BODY_34_PARTS.LEFT_FOOT),
		(BODY_34_PARTS.PELVIS, BODY_34_PARTS.RIGHT_HIP),
		(BODY_34_PARTS.RIGHT_HIP, BODY_34_PARTS.RIGHT_KNEE),
		(BODY_34_PARTS.RIGHT_KNEE, BODY_34_PARTS.RIGHT_ANKLE),
		(BODY_34_PARTS.RIGHT_ANKLE, BODY_34_PARTS.RIGHT_FOOT),
		(BODY_34_PARTS.CHEST_SPINE, BODY_34_PARTS.NECK),
		(BODY_34_PARTS.NECK, BODY_34_PARTS.HEAD),
		(BODY_34_PARTS.HEAD, BODY_34_PARTS.NOSE),
		(BODY_34_PARTS.NOSE, BODY_34_PARTS.LEFT_EYE),
		(BODY_34_PARTS.LEFT_EYE, BODY_34_PARTS.LEFT_EAR),
		(BODY_34_PARTS.NOSE, BODY_34_PARTS.RIGHT_EYE),
		(BODY_34_PARTS.RIGHT_EYE, BODY_34_PARTS.RIGHT_EAR),
		(BODY_34_PARTS.LEFT_ANKLE, BODY_34_PARTS.LEFT_HEEL),
		(BODY_34_PARTS.RIGHT_ANKLE, BODY_34_PARTS.RIGHT_HEEL),
		(BODY_34_PARTS.LEFT_HEEL, BODY_34_PARTS.LEFT_FOOT),
        (BODY_34_PARTS.RIGHT_HEEL, BODY_34_PARTS.RIGHT_FOOT)
 ]

##
# \brief Links of human body keypoints for [sl.BODY_FORMAT.BODY_38](\ref BODY_FORMAT), useful for display.
# \ingroup Body_group
BODY_38_BONES = [
        (BODY_38_PARTS.PELVIS, BODY_38_PARTS.SPINE_1),
        (BODY_38_PARTS.SPINE_1, BODY_38_PARTS.SPINE_2),
        (BODY_38_PARTS.SPINE_2, BODY_38_PARTS.SPINE_3),
        (BODY_38_PARTS.SPINE_3, BODY_38_PARTS.NECK),
        (BODY_38_PARTS.NECK, BODY_38_PARTS.NOSE),
        (BODY_38_PARTS.NOSE, BODY_38_PARTS.LEFT_EYE),
        (BODY_38_PARTS.LEFT_EYE, BODY_38_PARTS.LEFT_EAR),
        (BODY_38_PARTS.NOSE, BODY_38_PARTS.RIGHT_EYE),
        (BODY_38_PARTS.RIGHT_EYE, BODY_38_PARTS.RIGHT_EAR),
        (BODY_38_PARTS.SPINE_3, BODY_38_PARTS.LEFT_CLAVICLE),
        (BODY_38_PARTS.LEFT_CLAVICLE, BODY_38_PARTS.LEFT_SHOULDER),
        (BODY_38_PARTS.LEFT_SHOULDER, BODY_38_PARTS.LEFT_ELBOW),
        (BODY_38_PARTS.LEFT_ELBOW, BODY_38_PARTS.LEFT_WRIST),
        (BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_THUMB_4),
        (BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_INDEX_1),
        (BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_MIDDLE_4),
        (BODY_38_PARTS.LEFT_WRIST, BODY_38_PARTS.LEFT_HAND_PINKY_1),
        (BODY_38_PARTS.SPINE_3, BODY_38_PARTS.RIGHT_CLAVICLE),
        (BODY_38_PARTS.RIGHT_CLAVICLE, BODY_38_PARTS.RIGHT_SHOULDER),
        (BODY_38_PARTS.RIGHT_SHOULDER, BODY_38_PARTS.RIGHT_ELBOW),
        (BODY_38_PARTS.RIGHT_ELBOW, BODY_38_PARTS.RIGHT_WRIST),
        (BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_THUMB_4),
        (BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_INDEX_1),
        (BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_MIDDLE_4),
        (BODY_38_PARTS.RIGHT_WRIST, BODY_38_PARTS.RIGHT_HAND_PINKY_1),
        (BODY_38_PARTS.PELVIS, BODY_38_PARTS.LEFT_HIP),
        (BODY_38_PARTS.LEFT_HIP, BODY_38_PARTS.LEFT_KNEE),
        (BODY_38_PARTS.LEFT_KNEE, BODY_38_PARTS.LEFT_ANKLE),
        (BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_HEEL),
        (BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_BIG_TOE),
        (BODY_38_PARTS.LEFT_ANKLE, BODY_38_PARTS.LEFT_SMALL_TOE),
        (BODY_38_PARTS.PELVIS, BODY_38_PARTS.RIGHT_HIP),
        (BODY_38_PARTS.RIGHT_HIP, BODY_38_PARTS.RIGHT_KNEE),
        (BODY_38_PARTS.RIGHT_KNEE, BODY_38_PARTS.RIGHT_ANKLE),
        (BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_HEEL),
        (BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_BIG_TOE),
        (BODY_38_PARTS.RIGHT_ANKLE, BODY_38_PARTS.RIGHT_SMALL_TOE)
 ]

##
# Returns the associated index for a given \ref BODY_18_PARTS.
# \ingroup Body_group
def get_idx(part: BODY_18_PARTS):
    return c_getIdx(<c_BODY_18_PARTS>(<unsigned int>part.value))

##
# Returns the associated index for a given \ref BODY_34_PARTS.
# \ingroup Body_group
def get_idx_34(part: BODY_34_PARTS):
    return c_getIdx(<c_BODY_34_PARTS>(<unsigned int>part.value))

##
# Contains batched data of a detected object
# \ingroup Object_group
cdef class ObjectsBatch:
    cdef c_ObjectsBatch objects_batch

    ##
    # The trajectory ID
    @property
    def id(self):
        return self.objects_batch.id

    @id.setter
    def id(self, int value):
        self.objects_batch.id = value

    ##
    # Object category. Identifies the object type
    @property
    def label(self):
        return OBJECT_CLASS(<int>self.objects_batch.label)

    @label.setter
    def label(self, label):
        if isinstance(label, OBJECT_CLASS):
            self.objects_batch.label = <c_OBJECT_CLASS>(<unsigned int>label.value)
        else:
            raise TypeError("Argument is not of OBJECT_CLASS type.")

    ##
    # Object sublabel. Identifies the object subclass
    @property
    def sublabel(self):
        return OBJECT_SUBCLASS(<int>self.objects_batch.sublabel)

    @sublabel.setter
    def sublabel(self, sublabel):
        if isinstance(sublabel, OBJECT_SUBCLASS):
            self.objects_batch.sublabel = <c_OBJECT_SUBCLASS>(<unsigned int>sublabel.value)
        else:
            raise TypeError("Argument is not of c_OBJECT_SUBCLASS type.")

    ##
    # Defines the object tracking state.
    @property
    def tracking_state(self):
        return OBJECT_TRACKING_STATE(<int>self.objects_batch.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.objects_batch.tracking_state = <c_OBJECT_TRACKING_STATE>(<unsigned int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # A sample of 3d positions
    @property
    def positions(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.positions.size(), 3), dtype=np.float32)
        for i in range(self.objects_batch.positions.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.positions[i].ptr()[j]
        return arr

    ##
    # A sample of the associated position covariance
    @property
    def position_covariances(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.position_covariances.size(), 6), dtype=np.float32)
        for i in range(self.objects_batch.position_covariances.size()):
            for j in range(6):
                arr[i,j] = self.objects_batch.position_covariances[i][j]
        return arr

    ##
    # A sample of 3d velocities
    @property
    def velocities(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.velocities.size(), 3), dtype=np.float32)
        for i in range(self.objects_batch.velocities.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.velocities[i].ptr()[j]
        return arr

    ##
    # The associated position timestamp
    @property
    def timestamps(self):
        out_ts = []
        for i in range(self.objects_batch.timestamps.size()):
            ts = Timestamp()
            ts.timestamp = self.objects_batch.timestamps[i] 
            out_ts.append(ts)
        return out_ts

    ##
    # A sample of 3d bounding boxes
    @property
    def bounding_boxes(self):
        # A 3D bounding box should have 8 indices, 3 coordinates
        cdef np.ndarray arr = np.zeros((self.objects_batch.bounding_boxes.size(),8,3))
        for i in range(self.objects_batch.bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.objects_batch.bounding_boxes[i][j][k]
        return arr

    ##
    # 2D bounding box of the person represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution, [0,0] is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_boxes_2d(self):
        # A 2D bounding box should have 4 indices, 2 coordinates
        cdef np.ndarray arr = np.zeros((self.objects_batch.bounding_boxes_2d.size(),4,2))
        for i in range(self.objects_batch.bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.objects_batch.bounding_boxes_2d[i][j][k]
        return arr

    ##
    # A sample of object detection confidence
    @property
    def confidences(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.confidences.size()))
        for i in range(self.objects_batch.confidences.size()):
            arr[i] = self.objects_batch.confidences[i]
        return arr

    ##
    # A sample of the object action state
    @property
    def action_states(self):
        action_states_out = []
        for i in range(self.objects_batch.action_states.size()):
            action_states_out.append(OBJECT_ACTION_STATE(<unsigned int>self.objects_batch.action_states[i]))
        return action_states_out

    ##
	# Bounds the head with four 2D points. Expressed in pixels on the original image resolution.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL)
    @property
    def head_bounding_boxes_2d(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_bounding_boxes_2d.size(),4,2))
        for i in range(self.objects_batch.head_bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.objects_batch.head_bounding_boxes_2d[i][j][k]
        return arr

    ##
	# Bounds the head with eight 3D points. 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    @property
    def head_bounding_boxes(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_bounding_boxes.size(),8,3))
        for i in range(self.objects_batch.head_bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.objects_batch.head_bounding_boxes[i][j][k]
        return arr
		
    ##
	# 3D head centroid.
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    @property
    def head_positions(self):
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_positions.size(),3))
        for i in range(self.objects_batch.head_positions.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.head_positions[i][j]
        return arr

##
# Contains the result of the object detection module.
# \ingroup Object_group
# The detected objects are listed in \ref object_list
cdef class Objects:
    cdef c_Objects objects

    ##
    # Defines the \ref Timestamp corresponding to the frame acquisition. 
    # This value is especially useful for the async mode to synchronize the data.
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp=self.objects.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.objects.timestamp.data_ns = timestamp

    ##
    # The list of detected objects. An array of \ref ObjectData .
    @property
    def object_list(self):
        object_list_ = []
        for i in range(self.objects.object_list.size()):
            py_objectData = ObjectData()
            py_objectData.object_data = self.objects.object_list[i]
            object_list_.append(py_objectData)
        return object_list_

    @object_list.setter
    def object_list(self, objects):
        for i in range(len(objects)):
            self.objects.object_list.push_back((<ObjectData>objects[i]).object_data)

    ##
    # Defines if the object list has already been retrieved or not.
    @property
    def is_new(self):
        return self.objects.is_new

    @is_new.setter
    def is_new(self, bool is_new):
        self.objects.is_new = is_new

    ##
    # Defines if both the object tracking and the world orientation have been setup.
    @property
    def is_tracked(self):
        return self.objects.is_tracked

    @is_tracked.setter
    def is_tracked(self, bool is_tracked):
        self.objects.is_tracked = is_tracked


    ##
    # Function that looks for a given object ID in the current object list and returns the associated object if found and a status.
    # \param py_object_data [out] : the object corresponding to the given ID if found
    # \param object_data_id [in] : the input object ID
    # \return True if found False otherwise
    def get_object_data_from_id(self, py_object_data: ObjectData, object_data_id: int):
        if isinstance(py_object_data, ObjectData) :
            return self.objects.getObjectDataFromId((<ObjectData>py_object_data).object_data, object_data_id)
        else :
           raise TypeError("Argument is not of ObjectData type.") 

##
# Contains batched data of a detected object
# \ingroup Body_group
cdef class BodiesBatch:
    cdef c_BodiesBatch bodies_batch

    ##
    # The trajectory ID
    @property
    def id(self):
        return self.bodies_batch.id

    @id.setter
    def id(self, int value):
        self.bodies_batch.id = value

    ##
    # Defines the body tracking state.
    @property
    def tracking_state(self):
        return OBJECT_TRACKING_STATE(<int>self.bodies_batch.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.bodies_batch.tracking_state = <c_OBJECT_TRACKING_STATE>(<unsigned int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # A sample of 3d positions
    @property
    def positions(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.positions.size(), 3), dtype=np.float32)
        for i in range(self.bodies_batch.positions.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.positions[i].ptr()[j]
        return arr

    ##
    # A sample of the associated position covariance
    @property
    def position_covariances(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.position_covariances.size(), 6), dtype=np.float32)
        for i in range(self.bodies_batch.position_covariances.size()):
            for j in range(6):
                arr[i,j] = self.bodies_batch.position_covariances[i][j]
        return arr

    ##
    # A sample of 3d velocities
    @property
    def velocities(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.velocities.size(), 3), dtype=np.float32)
        for i in range(self.bodies_batch.velocities.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.velocities[i].ptr()[j]
        return arr

    ##
    # The associated position timestamp
    @property
    def timestamps(self):
        out_ts = []
        for i in range(self.bodies_batch.timestamps.size()):
            ts = Timestamp()
            ts.timestamp = self.bodies_batch.timestamps[i] 
            out_ts.append(ts)
        return out_ts

    ##
    # A sample of 3d bounding boxes
    @property
    def bounding_boxes(self):
        # A 3D bounding box should have 8 indices, 3 coordinates
        cdef np.ndarray arr = np.zeros((self.bodies_batch.bounding_boxes.size(),8,3))
        for i in range(self.bodies_batch.bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.bounding_boxes[i][j][k]
        return arr

    ##
    # 2D bounding box of the person represented as four 2D points starting at the top left corner and rotation clockwise.
    # Expressed in pixels on the original image resolution, [0,0] is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_boxes_2d(self):
        # A 2D bounding box should have 4 indices, 2 coordinates
        cdef np.ndarray arr = np.zeros((self.bodies_batch.bounding_boxes_2d.size(),4,2))
        for i in range(self.bodies_batch.bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.bodies_batch.bounding_boxes_2d[i][j][k]
        return arr

    ##
    # A sample of object detection confidence
    @property
    def confidences(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.confidences.size()))
        for i in range(self.bodies_batch.confidences.size()):
            arr[i] = self.bodies_batch.confidences[i]
        return arr

    ##
    # A sample of the object action state
    @property
    def action_states(self):
        action_states_out = []
        for i in range(self.bodies_batch.action_states.size()):
            action_states_out.append(OBJECT_ACTION_STATE(<unsigned int>self.bodies_batch.action_states[i]))
        return action_states_out
    
    ##
	# A sample of 2d person keypoints.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    # \warning in some cases, eg. body partially out of the image or missing depth data, some keypoints cannot be detected, they will have non finite values.
    @property
    def keypoints_2d(self):
        # 18 keypoints
        cdef np.ndarray arr = np.zeros((self.bodies_batch.keypoints_2d.size(),self.bodies_batch.keypoints_2d[0].size(),2))
        for i in range(self.bodies_batch.keypoints_2d.size()):
            for j in range(self.bodies_batch.keypoints_2d[0].size()):
                for k in range(2):
                    arr[i,j,k] = self.bodies_batch.keypoints_2d[i][j][k]
        return arr

	##
	# A sample of 3d person keypoints
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
	# \warning in some cases, eg. body partially out of the image or missing depth data, some keypoints cannot be detected, they will have non finite values.
    @property
    def keypoints(self):
        # 18 keypoints
        cdef np.ndarray arr = np.zeros((self.bodies_batch.keypoints.size(),self.bodies_batch.keypoints[0].size(),3))
        for i in range(self.bodies_batch.keypoints.size()):
            for j in range(self.bodies_batch.keypoints[0].size()):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.keypoints[i][j][k]
        return arr

    ##
	# Bounds the head with four 2D points. Expressed in pixels on the original image resolution.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL)
    @property
    def head_bounding_boxes_2d(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_bounding_boxes_2d.size(),4,2))
        for i in range(self.bodies_batch.head_bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.bodies_batch.head_bounding_boxes_2d[i][j][k]
        return arr

    ##
	# Bounds the head with eight 3D points. 
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    @property
    def head_bounding_boxes(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_bounding_boxes.size(),8,3))
        for i in range(self.bodies_batch.head_bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.head_bounding_boxes[i][j][k]
        return arr
		
    ##
	# 3D head centroid.
    # Defined in \ref InitParameters.coordinate_units, expressed in \ref RuntimeParameters.measure3D_reference_frame.
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
    @property
    def head_positions(self):
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_positions.size(),3))
        for i in range(self.bodies_batch.head_positions.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.head_positions[i][j]
        return arr

	##
	# Per keypoint detection confidence, cannot be lower than the [sl.ObjectDetectionRuntimeParameters().detection_confidence_threshold](\ref ObjectDetectionRuntimeParameters).
	# \note Not available with [DETECTION_MODEL.MULTI_CLASS_BOX*](\ref DETECTION_MODEL).
	# \warning in some cases, eg. body partially out of the image or missing depth data, some keypoints cannot be detected, they will have non finite values.
    @property
    def keypoint_confidences(self):
        cdef np.ndarray arr = np.zeros(self.bodies_batch.keypoint_confidences.size())
        for i in range(self.bodies_batch.keypoint_confidences.size()):
            arr[i] = self.bodies_batch.keypoint_confidences[i]
        return arr

##
# Contains the result of the object detection module. The detected objects are listed in \ref object_list.
# \ingroup Object_group
cdef class Bodies:
    cdef c_Bodies bodies

    ##
    # Defines the \ref Timestamp corresponding to the frame acquisition. 
    # This value is especially useful for the async mode to synchronize the data.
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp=self.bodies.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.bodies.timestamp.data_ns = timestamp

    ##
    # The list of detected bodies. An array of \ref BodiesData .
    @property
    def body_list(self):
        body_list_ = []
        for i in range(self.bodies.body_list.size()):
            py_bodyData = BodyData()
            py_bodyData.body_data = self.bodies.body_list[i]
            body_list_.append(py_bodyData)
        return body_list_

    @body_list.setter
    def body_list(self, bodies):
        for i in range(len(bodies)):
            self.bodies.body_list.push_back((<BodyData>bodies[i]).body_data)

    ##
    # Defines if the object list has already been retrieved or not.
    @property
    def is_new(self):
        return self.bodies.is_new

    @is_new.setter
    def is_new(self, bool is_new):
        self.bodies.is_new = is_new

    ##
    # Defines if both the object tracking and the world orientation have been setup.
    @property
    def is_tracked(self):
        return self.bodies.is_tracked

    @is_tracked.setter
    def is_tracked(self, bool is_tracked):
        self.bodies.is_tracked = is_tracked


    ##
    # Function that looks for a given body ID in the current body list and returns the associated body if found and a status.
    # \param py_body_data [out] : the body corresponding to the given ID if found
    # \param body_data_id [in] : the input body ID
    # \return True if found False otherwise
    def get_body_data_from_id(self, py_body_data: BodyData, body_data_id: int):
        if isinstance(py_body_data, BodyData) :
            return self.bodies.getBodyDataFromId((<BodyData>py_body_data).body_data, body_data_id)
        else :
           raise TypeError("Argument is not of ObjectData type.") 

##
# Sets batch trajectory parameters
# \ingroup Object_group
# The default constructor sets all parameters to their default settings.
# \note Parameters can be user adjusted.
cdef class BatchParameters:
    cdef c_BatchParameters* batch_params
    
    ##
    # Default constructor. Sets all parameters to their default values
    def __cinit__(self, enable=False, id_retention_time=240, batch_duration=2.0):
        self.batch_params = new c_BatchParameters(<bool>enable, <float>(id_retention_time), <float>batch_duration)

    def __dealloc__(self):
        del self.batch_params

    ##
    # Defines if the Batch option in the object detection module is enabled. Batch queueing system provides:
    # \n - Deep-Learning based re-identification
    # \n - Trajectory smoothing and filtering
    # \note To activate this option, \ref enable must be set to True.
    @property
    def enable(self):
        return self.batch_params.enable

    @enable.setter
    def enable(self, value: bool):
        self.batch_params.enable = value

    ##
    # Max retention time in seconds of a detected object. After this time, the same object will mostly have a different ID.
    @property
    def id_retention_time(self):
        return self.batch_params.id_retention_time

    @id_retention_time.setter
    def id_retention_time(self, value):
        self.batch_params.id_retention_time = value

    ##
    # Trajectories will be output in batch with the desired latency in seconds.
    # During this waiting time, re-identification of objects is done in the background.
    # Specifying a short latency will limit the search (falling in timeout) for previously seen object IDs but will be closer to real time output.
    @property
    def latency(self):
        return self.batch_params.latency

    @latency.setter
    def latency(self, value):
        self.batch_params.latency = value

##
# Sets the object detection parameters.
# \ingroup Object_group
# The default constructor sets all parameters to their default settings.
# \note Parameters can be user adjusted.
cdef class ObjectDetectionParameters:
    cdef c_ObjectDetectionParameters* object_detection

    ##
    # Constructor. Calling the constructor without any parameter will set them to their default values.
    # \param image_sync : sets \ref image_sync. Default: True
    # \param enable_tracking : sets \ref enable_tracking. Default: True
    # \param enable_segmentation : sets \ref enable_segmentation. Default: True
    # \param enable_body_fitting : sets \ref enable_body_fitting. Default: False
    # \param max_range : sets \ref max_range. Default: -1.0 (set to \ref InitParameters.depth_maximum_distance)
    # \param batch_trajectories_parameters : sets \ref batch_parameters. Default: see \ref BatchParameters default constructor  
    # \param body_format : sets \ref body_format. Default: [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT)  
    def __cinit__(self, image_sync=True, enable_tracking=True
                , enable_segmentation=False, detection_model=OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
                , max_range=-1.0 , batch_trajectories_parameters=BatchParameters()
                , filtering_mode = OBJECT_FILTERING_MODE.NMS3D
                , prediction_timeout_s = 0.2
                , allow_reduced_precision_inference = False
                , instance_module_id = 0):
        self.object_detection = new c_ObjectDetectionParameters(image_sync, enable_tracking
                                                                , enable_segmentation, <c_OBJECT_DETECTION_MODEL>(<unsigned int>detection_model.value)
                                                                , max_range, (<BatchParameters>batch_trajectories_parameters).batch_params[0]
                                                                , <c_OBJECT_FILTERING_MODE>(<unsigned int>filtering_mode.value)
                                                                , prediction_timeout_s
                                                                , allow_reduced_precision_inference
                                                                , instance_module_id)

    def __dealloc__(self):
        del self.object_detection

    ##
    # Defines if the object detection  is synchronized to the image or runs in a separate thread
    @property
    def image_sync(self):
        return self.object_detection.image_sync

    @image_sync.setter
    def image_sync(self, bool image_sync):
        self.object_detection.image_sync = image_sync

    ##
    # Defines if the object detection will track objects across images flow
    @property
    def enable_tracking(self):
        return self.object_detection.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, bool enable_tracking):
        self.object_detection.enable_tracking = enable_tracking

    ##
    # Defines if the mask object will be computed
    @property
    def enable_segmentation(self):
        return self.object_detection.enable_segmentation

    @enable_segmentation.setter
    def enable_segmentation(self, bool enable_segmentation):
        self.object_detection.enable_segmentation = enable_segmentation

    ##
    # Enable human pose estimation with skeleton keypoints output 
    @property
    def detection_model(self):
        return OBJECT_DETECTION_MODEL(<int>self.object_detection.detection_model)

    @detection_model.setter
    def detection_model(self, detection_model):
        if isinstance(detection_model, OBJECT_DETECTION_MODEL) :
            self.object_detection.detection_model = <c_OBJECT_DETECTION_MODEL>(<unsigned int>detection_model.value)
        else :
            raise TypeError()

    ##
    # Defines an upper depth range for detections
    # \n Defined in \ref InitParameters.coordinate_units
    # \n Default value is set to \ref InitParameters.depth_maximum_distance (can not be higher)
    @property
    def max_range(self):
        return self.object_detection.max_range

    @max_range.setter
    def max_range(self, float max_range):
        self.object_detection.max_range = max_range

    ##
    # Batching system (introduced in 3.5) performs short-term re-identification with deep learning and trajectories filtering.
    # \ref BatchParameters.enable needs to be set to True to use this feature (by default, it is disabled) 
    @property
    def batch_parameters(self):
        params = BatchParameters()
        params.enable = self.object_detection.batch_parameters.enable
        params.id_retention_time = self.object_detection.batch_parameters.id_retention_time
        params.latency = self.object_detection.batch_parameters.latency
        return params

    @batch_parameters.setter
    def batch_parameters(self, BatchParameters params):
        self.object_detection.batch_parameters = params.batch_params[0]


    ##
    # Filtering mode for MULTI_CLASS_BOX and Custom objects tracking
    @property
    def filtering_mode(self):
        return OBJECT_FILTERING_MODE(<int>self.object_detection.filtering_mode)

    @filtering_mode.setter
    def filtering_mode(self, filtering_mode):
        if isinstance(filtering_mode, OBJECT_FILTERING_MODE) :
            self.object_detection.filtering_mode = <c_OBJECT_FILTERING_MODE>(<unsigned int>filtering_mode.value)
        else :
            raise TypeError()

    ##
    # When an object is not detected anymore, the SDK will predict its positions during a short period of time before its state switched to SEARCHING.
    @property
    def prediction_timeout_s(self):
        return self.object_detection.prediction_timeout_s

    @prediction_timeout_s.setter
    def prediction_timeout_s(self, float prediction_timeout_s):
        self.object_detection.prediction_timeout_s = prediction_timeout_s
        
    ##
    # Allow inference to run at a lower precision to improve runtime and memory usage, 
    # it might increase the initial optimization time and could include downloading calibration data or calibration cache and slightly reduce the accuracy
    @property
    def allow_reduced_precision_inference(self):
        return self.object_detection.allow_reduced_precision_inference

    @allow_reduced_precision_inference.setter
    def allow_reduced_precision_inference(self, bool allow_reduced_precision_inference):
        self.object_detection.allow_reduced_precision_inference = allow_reduced_precision_inference

    ##
    # Defines which object detection instance to use
    @property
    def instance_module_id(self):
        return self.object_detection.instance_module_id

    @instance_module_id.setter
    def instance_module_id(self, unsigned int instance_module_id):
        self.object_detection.instance_module_id = instance_module_id



##
# Sets the object detection runtime parameters.
# \ingroup Object_group
cdef class ObjectDetectionRuntimeParameters:
    cdef c_ObjectDetectionRuntimeParameters* object_detection_rt

    ##
    # Default constructor
    # \param detection_confidence_threshold : sets \ref detection_confidence_threshold. Default: 50
    # \param object_class_filter : sets \ref object_class_filter. Default: empty list (all classes are tracked)
    # \param object_class_detection_confidence_threshold : sets \ref object_class_detection_confidence_threshold. Default: empty dict (detection_confidence_threshold value will be taken for each class)
    # \param minimum_keypoints_threshold: sets \ref minimum_keypoints_threshold. Default: 0 (all skeletons are retrieved)
    def __cinit__(self, detection_confidence_threshold=50, object_class_filter=[], object_class_detection_confidence_threshold={}):
        cdef vector[int] vec_cpy
        cdef map[int,float] map_cpy
        for object_class in object_class_filter:
            vec_cpy.push_back(<int>object_class.value)
        for k,v in object_class_detection_confidence_threshold.items():
            map_cpy[<int>k.value] = v
        self.object_detection_rt = create_object_detection_runtime_parameters(detection_confidence_threshold, vec_cpy, map_cpy)

    def __dealloc__(self):
        del self.object_detection_rt

    ##
    # Defines the confidence threshold: interval between 1 and 99. A confidence of 1 meaning a low threshold, more uncertain objects and 99 very few but very precise objects.
    # If the scene contains a lot of objects, increasing the confidence can slightly speed up the process, since every object instances are tracked.
    @property
    def detection_confidence_threshold(self):
        return self.object_detection_rt.detection_confidence_threshold

    @detection_confidence_threshold.setter
    def detection_confidence_threshold(self, float detection_confidence_threshold_):
        self.object_detection_rt.detection_confidence_threshold = detection_confidence_threshold_
 
    ##
    # Selects which object types to detect and track. By default all classes are tracked.
    # Fewer object types can slightly speed up the process, since every objects are tracked.
    #
    # \n In order to get all the available classes, the filter vector must be empty (default behaviour): 
    # \code
    # object_detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
    # object_detection_parameters_rt.object_class_filter = []
    # \endcode
    # 
    # \n To select a set of specific object classes, like vehicles, persons and animals for instance:
    # \code
    # object_detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE,sl.OBJECT_CLASS.PERSON,sl.OBJECT_CLASS.ANIMAL]
    # \endcode
    @property
    def object_class_filter(self):
        object_class_filter_out = []
        for i in range(self.object_detection_rt.object_class_filter.size()):
            object_class_filter_out.append(OBJECT_CLASS(<unsigned int>self.object_detection_rt.object_class_filter[i]))
        return object_class_filter_out

    @object_class_filter.setter
    def object_class_filter(self, object_class_filter):
        self.object_detection_rt.object_class_filter.clear()
        for i in range(len(object_class_filter)):
            self.object_detection_rt.object_class_filter.push_back(<c_OBJECT_CLASS>(<unsigned int>object_class_filter[i].value))
    
    ##
    # Defines a detection threshold for each object class. It can be empty for some classes, \ref detection_confidence_threshold will be taken as fallback/default value.
    # 
    # \n To set a specific confidence threshold per class:
    # \code
    # object_detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
    # object_detection_parameters_rt.object_class_detection_confidence_threshold = {sl.OBJECT_CLASS.VEHICLE: 30,sl.OBJECT_CLASS.PERSON: 50}
    # \endcode
    @property
    def object_class_detection_confidence_threshold(self):
        object_detection_confidence_threshold_out = {}
        cdef map[c_OBJECT_CLASS,float].iterator it = self.object_detection_rt.object_class_detection_confidence_threshold.begin()
        while(it != self.object_detection_rt.object_class_detection_confidence_threshold.end()):
            object_detection_confidence_threshold_out[OBJECT_CLASS(<unsigned int>deref(it).first)] = deref(it).second
            postincrement(it)
        return object_detection_confidence_threshold_out

    @object_class_detection_confidence_threshold.setter
    def object_class_detection_confidence_threshold(self, object_class_detection_confidence_threshold_dict):
        self.object_detection_rt.object_class_detection_confidence_threshold.clear()
        for k,v in object_class_detection_confidence_threshold_dict.items():
            self.object_detection_rt.object_class_detection_confidence_threshold[<c_OBJECT_CLASS>(<unsigned int>k.value)] = v

##
# Sets the body tracking parameters.
# \ingroup Body_group
# The default constructor sets all parameters to their default settings.
# \note Parameters can be user adjusted.
cdef class BodyTrackingParameters:
    cdef c_BodyTrackingParameters* bodyTrackingParameters

    ##
    # Constructor. Calling the constructor without any parameter will set them to their default values.
    # \param image_sync : sets \ref image_sync. Default: True
    # \param enable_tracking : sets \ref enable_tracking. Default: True
    # \param enable_segmentation : sets \ref enable_segmentation. Default: True
    # \param enable_body_fitting : sets \ref enable_body_fitting. Default: False
    # \param max_range : sets \ref max_range. Default: -1.0 (set to \ref InitParameters.depth_maximum_distance)
    # \param body_format : sets \ref body_format. Default: [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT)  
    def __cinit__(self, image_sync=True, enable_tracking=True
                , enable_segmentation=True, detection_model=BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
                , enable_body_fitting=False, max_range=-1.0
                , body_format=BODY_FORMAT.BODY_18, body_selection=BODY_KEYPOINTS_SELECTION.FULL, prediction_timeout_s = 0.2
                , allow_reduced_precision_inference = False
                , instance_module_id = 0):
        self.bodyTrackingParameters = new c_BodyTrackingParameters(image_sync, enable_tracking
                                                                , enable_segmentation
                                                                , <c_BODY_TRACKING_MODEL>(<int>detection_model.value)
                                                                , enable_body_fitting
                                                                , max_range
                                                                , <c_BODY_FORMAT>(<int>body_format.value)
                                                                , <c_BODY_KEYPOINTS_SELECTION>(<int>body_selection.value)
                                                                , prediction_timeout_s
                                                                , allow_reduced_precision_inference
                                                                , instance_module_id)

    def __dealloc__(self):
        del self.bodyTrackingParameters

    ##
    # Defines if the object detection  is synchronized to the image or runs in a separate thread
    @property
    def image_sync(self):
        return self.bodyTrackingParameters.image_sync

    @image_sync.setter
    def image_sync(self, bool image_sync):
        self.bodyTrackingParameters.image_sync = image_sync

    ##
    # Defines if the object detection will track objects across images flow
    @property
    def enable_tracking(self):
        return self.bodyTrackingParameters.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, bool enable_tracking):
        self.bodyTrackingParameters.enable_tracking = enable_tracking

    ##
    # Defines if the mask object will be computed
    @property
    def enable_segmentation(self):
        return self.bodyTrackingParameters.enable_segmentation

    @enable_segmentation.setter
    def enable_segmentation(self, bool enable_segmentation):
        self.bodyTrackingParameters.enable_segmentation = enable_segmentation

    ##
    # Enable human pose estimation with skeleton keypoints output 
    @property
    def detection_model(self):
        return BODY_TRACKING_MODEL(<int>self.bodyTrackingParameters.detection_model)

    @detection_model.setter
    def detection_model(self, detection_model):
        if isinstance(detection_model, BODY_TRACKING_MODEL) :
            self.bodyTrackingParameters.detection_model = <c_BODY_TRACKING_MODEL>(<unsigned int>detection_model.value)
        else :
            raise TypeError()

    ##
    # Defines the body format output by the SDK when \ref retrieve_objects is called.
    # \warning if set to sl.BODY_FORMAT.POSE_32, the ZED SDK will automatically enable the fitting (cf. \ref enable_body_fitting).
    @property
    def body_format(self):
        return BODY_FORMAT(<int>self.bodyTrackingParameters.body_format)

    @body_format.setter
    def body_format(self, body_format):
        if isinstance(body_format, BODY_FORMAT):
            self.bodyTrackingParameters.body_format = <c_BODY_FORMAT>(<unsigned int>body_format.value)

    ##
    # Defines if the body fitting will be applied 
    @property
    def enable_body_fitting(self):
        return self.bodyTrackingParameters.enable_body_fitting

    @enable_body_fitting.setter
    def enable_body_fitting(self, bool enable_body_fitting):
        self.bodyTrackingParameters.enable_body_fitting = enable_body_fitting

    ##
    # Defines an upper depth range for detections
    # \n Defined in \ref InitParameters.coordinate_units
    # \n Default value is set to \ref InitParameters.depth_maximum_distance (can not be higher)
    @property
    def max_range(self):
        return self.bodyTrackingParameters.max_range

    @max_range.setter
    def max_range(self, float max_range):
        self.bodyTrackingParameters.max_range = max_range

    ##
    # When an object is not detected anymore, the SDK will predict its positions during a short period of time before its state switched to SEARCHING.
    @property
    def prediction_timeout_s(self):
        return self.bodyTrackingParameters.prediction_timeout_s

    @prediction_timeout_s.setter
    def prediction_timeout_s(self, float prediction_timeout_s):
        self.bodyTrackingParameters.prediction_timeout_s = prediction_timeout_s
        
    ##
    # Allow inference to run at a lower precision to improve runtime and memory usage, 
    # it might increase the initial optimization time and could include downloading calibration data or calibration cache and slightly reduce the accuracy
    @property
    def allow_reduced_precision_inference(self):
        return self.bodyTrackingParameters.allow_reduced_precision_inference

    @allow_reduced_precision_inference.setter
    def allow_reduced_precision_inference(self, bool allow_reduced_precision_inference):
        self.bodyTrackingParameters.allow_reduced_precision_inference = allow_reduced_precision_inference

    ##
    # Defines which object detection instance to use
    @property
    def instance_module_id(self):
        return self.bodyTrackingParameters.instance_module_id

    @instance_module_id.setter
    def instance_module_id(self, unsigned int instance_module_id):
        self.bodyTrackingParameters.instance_module_id = instance_module_id



##
# Sets the object detection runtime parameters.
# \ingroup Body_group
cdef class BodyTrackingRuntimeParameters:
    cdef c_BodyTrackingRuntimeParameters* body_tracking_rt

    ##
    # Default constructor
    # \param detection_confidence_threshold : sets \ref detection_confidence_threshold. Default: 50
    # \param minimum_keypoints_threshold: sets \ref minimum_keypoints_threshold. Default: 0 (all skeletons are retrieved)
    def __cinit__(self, detection_confidence_threshold=50, minimum_keypoints_threshold=0, skeleton_smoothing=0):
        self.body_tracking_rt = new c_BodyTrackingRuntimeParameters(detection_confidence_threshold, minimum_keypoints_threshold, skeleton_smoothing)

    def __dealloc__(self):
        del self.body_tracking_rt

    ##
    # Defines the confidence threshold: interval between 1 and 99. A confidence of 1 meaning a low threshold, more uncertain objects and 99 very few but very precise objects.
    # If the scene contains a lot of bodies, increasing the confidence can slightly speed up the process, since every object instances are tracked.
    @property
    def detection_confidence_threshold(self):
        return self.body_tracking_rt.detection_confidence_threshold

    @detection_confidence_threshold.setter
    def detection_confidence_threshold(self, float detection_confidence_threshold_):
        self.body_tracking_rt.detection_confidence_threshold = detection_confidence_threshold_
 
    ##
    # Defines minimal number of keypoints per skeleton to be retrieved:
    # the SDK will outputs skeleton with more keypoints than this threshold.
    # it is useful for example to remove unstable fitting results when a skeleton is partially occluded.
    @property
    def minimum_keypoints_threshold(self):
        return self.body_tracking_rt.minimum_keypoints_threshold

    @minimum_keypoints_threshold.setter
    def minimum_keypoints_threshold(self, int minimum_keypoints_threshold_):
        self.body_tracking_rt.minimum_keypoints_threshold = minimum_keypoints_threshold_

    ##
    # this value controls the smoothing of the fitted fused skeleton.
    # it is ranged from 0 (low smoothing) and 1 (high smoothing)
    @property
    def skeleton_smoothing(self):
        return self.body_tracking_rt.skeleton_smoothing

    @skeleton_smoothing.setter
    def skeleton_smoothing(self, float skeleton_smoothing_):
        self.body_tracking_rt.skeleton_smoothing = skeleton_smoothing_


# Returns the current timestamp at the time the function is called.
# \ingroup Core_group
def get_current_timestamp():
    ts = Timestamp()
    ts.timestamp = getCurrentTimeStamp()
    return ts


##
# Width and height of an array.
# \ingroup Core_group
cdef class Resolution:
    cdef c_Resolution resolution
    def __cinit__(self, width=0, height=0):
        self.resolution.width = width
        self.resolution.height = height

    ##
    # Returns the area of the image.
    # \return The number of pixels of the array.
    def area(self):
        return self.resolution.width * self.resolution.height

    ##
    # Array width in pixels
    @property
    def width(self):
        return self.resolution.width

    @width.setter
    def width(self, value):
        self.resolution.width = value

    ##
    # Array height in pixels
    @property
    def height(self):
        return self.resolution.height

    @height.setter
    def height(self, value):
        self.resolution.height = value

    def __richcmp__(Resolution left, Resolution right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height
        if op == 3:
            return left.width!=right.width or left.height!=right.height
        else:
            raise NotImplementedError()

##
# Width and height of an array.
# \ingroup Core_group
cdef class Rect:
    cdef c_Rect rect
    def __cinit__(self, x=0, y=0, width=0, height=0):
        self.rect.x = x
        self.rect.y = y
        self.rect.width = width
        self.rect.height = height

    ##
    # Array width in pixels
    @property
    def width(self):
        return self.rect.width

    @width.setter
    def width(self, value):
        self.rect.width = value

    ##
    # Array height in pixels
    @property
    def height(self):
        return self.rect.height

    @height.setter
    def height(self, value):
        self.rect.height = value

    ##
    # x coordinate of top-left corner
    @property
    def x(self):
        return self.rect.x

    @x.setter
    def x(self, value):
        self.rect.x = value

    ##
    # y coordinate of top-left corner
    @property
    def y(self):
        return self.rect.y

    @y.setter
    def y(self, value):
        self.rect.y = value

    ##
    # Returns the area of the image.
    # \return The number of pixels of the array.
    def area(self):
        return self.rect.width * self.rect.height

    ##
    # \brief Tests if the given \ref Rect is empty (width or/and height is null) 
    # \return Returns True if rectangle is empty 
    def is_empty(self):
        return (self.rect.width * self.rect.height == 0)

    ##
    # \brief Tests if this \ref Rect contains the <target> \ref Rect.
    # \return Returns true if this rectangle contains the <target> rectangle. Otherwise returns false.
    # If proper is true, this function only returns true if the target rectangle is entirely inside this rectangle (not on the edge).
    def contains(self, target: Rect, proper = False):
        return self.rect.contains(target.rect, proper)

    ##
    # \brief Tests if this \ref Rect is contained inside the given <target> \ref Rect.
    # \return Returns true if this rectangle is inside the current target \ref Rect. Otherwise returns false.
    # If proper is true, this function only returns true if this rectangle is entirely inside the <target> rectangle (not on the edge).
    def is_contained(self, target: Rect, proper = False):
        return self.rect.isContained((<c_Rect>target.rect), proper)

    def __richcmp__(Rect left, Rect right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height and left.x==right.x and left.y==right.y
        if op == 3:
            return left.width!=right.width or left.height!=right.height or left.x!=right.x or left.y!=right.y
        else:
            raise NotImplementedError()

##
# Intrinsic parameters of a camera.
# \ingroup Depth_group
# Those information about the camera will be returned by \ref Camera.get_camera_information() .
# \note Similar to the \ref CalibrationParameters , those parameters are taken from the settings file (SNXXX.conf) and are modified during the \ref Camera.open call, represent the camera matrix corresponding to rectified or unrectified images. \nWhen filled with rectified parameters, fx,fy,cx,cy must be the same for Left and Right \ref Camera once \ref Camera.open has been called. Since distortion is corrected during rectification, distortion should not be considered on rectified images.
cdef class CameraParameters:
    cdef c_CameraParameters camera_params
    ##
    # Focal length in pixels along x axis.
    @property
    def fx(self):
        return self.camera_params.fx

    @fx.setter
    def fx(self, float fx_):
        self.camera_params.fx = fx_

    ##
    # Focal length in pixels along y axis.
    @property
    def fy(self):
        return self.camera_params.fy

    @fy.setter
    def fy(self, float fy_):
        self.camera_params.fy = fy_

    ##
    # Optical center along x axis, defined in pixels (usually close to width/2).
    @property
    def cx(self):
        return self.camera_params.cx

    @cx.setter
    def cx(self, float cx_):
        self.camera_params.cx = cx_

    ##
    # Optical center along y axis, defined in pixels (usually close to height/2).
    @property
    def cy(self):
        return self.camera_params.cy

    @cy.setter
    def cy(self, float cy_):
        self.camera_params.cy = cy_

    ##
    # A Numpy array. Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.
    @property
    def disto(self):
        cdef np.ndarray arr = np.zeros(5)
        for i in range(5):
            arr[i] = self.camera_params.disto[i]
        return arr

    ##
    # Sets the elements of the disto array.
    # \param float value1 : k1
    # \param float value2 : k2
    # \param float value3 : p1
    # \param float value4 : p2
    # \param float value5 : k3
    def set_disto(self, value1: float, value2: float, value3: float, value4: float, value5: float):
        self.camera_params.disto[0] = value1
        self.camera_params.disto[1] = value2
        self.camera_params.disto[2] = value3
        self.camera_params.disto[3] = value4
        self.camera_params.disto[4] = value5

    ##
    # Vertical field of view, in degrees.
    @property
    def v_fov(self):
        return self.camera_params.v_fov

    @v_fov.setter
    def v_fov(self, float v_fov_):
        self.camera_params.v_fov = v_fov_

    ##
    # Horizontal field of view, in degrees.
    @property
    def h_fov(self):
        return self.camera_params.h_fov

    @h_fov.setter
    def h_fov(self, float h_fov_):
        self.camera_params.h_fov = h_fov_

    ##
    # Diagonal field of view, in degrees.
    @property
    def d_fov(self):
        return self.camera_params.d_fov

    @d_fov.setter
    def d_fov(self, float d_fov_):
        self.camera_params.d_fov = d_fov_

    ##
    # Size in pixels of the images given by the camera.
    @property
    def image_size(self):
        return Resolution(self.camera_params.image_size.width, self.camera_params.image_size.height)

    @image_size.setter
    def image_size(self, Resolution size_):
        self.camera_params.image_size.width = size_.width
        self.camera_params.image_size.height = size_.height

    ##
    # Setups the parameters of a camera.
    # \param float fx_ : horizontal focal length.
    # \param float fy_ : vertical focal length.
    # \param float cx_ : horizontal optical center.
    # \param float cx_ : vertical optical center. 
    def set_up(self, fx_: float, fy_: float, cx_: float, cy_: float) :
        self.camera_params.fx = fx_
        self.camera_params.fy = fy_
        self.camera_params.cx = cx_
        self.camera_params.cy = cy_

    def scale(self, resolution: Resolution) -> CameraParameters:
        cam_params = CameraParameters()
        cam_params.camera_params = self.camera_params.scale(resolution.resolution)

##
# Intrinsic and Extrinsic parameters of the camera (translation and rotation).
# \ingroup Depth_group
# That information about the camera will be returned by \ref Camera.get_camera_information() .
# \note The calibration/rectification process, called during \ref Camera.open() , is using the raw parameters defined in the SNXXX.conf file, where XXX is the ZED Serial Number.
# \n Those values may be adjusted or not by the Self-Calibration to get a proper image alignment. After \ref Camera.open() is done (with or without Self-Calibration activated) success, most of the stereo parameters (except Baseline of course) should be 0 or very close to 0.
# \n  It means that images after rectification process (given by \ref Camera.retrieve_image() ) are aligned as if they were taken by a "perfect" stereo camera, defined by the new \ref CalibrationParameters .
# \warning \ref CalibrationParameters are returned in \ref COORDINATE_SYSTEM.IMAGE , they are not impacted by the \ref InitParameters.coordinate_system .
cdef class CalibrationParameters:
    cdef c_CalibrationParameters calibration
    cdef CameraParameters py_left_cam
    cdef CameraParameters py_right_cam
    cdef Transform py_stereo_transform

    def __cinit__(self):
        self.py_left_cam = CameraParameters()
        self.py_right_cam = CameraParameters()
        self.py_stereo_transform = Transform()
    
    def set(self):
        self.py_left_cam.camera_params = self.calibration.left_cam
        self.py_right_cam.camera_params = self.calibration.right_cam
        
        for i in range(16):
            self.py_stereo_transform.transform.m[i] = self.calibration.stereo_transform.m[i]

    ##
    # Returns the camera baseline in the \ref sl.UNIT defined in \ref sl.InitParameters
    def get_camera_baseline(self):
        return self.calibration.getCameraBaseline()

    ##
    # Intrisics \ref CameraParameters of the left camera.
    @property
    def left_cam(self):
        return self.py_left_cam

    @left_cam.setter
    def left_cam(self, CameraParameters left_cam_) :
        self.calibration.left_cam = left_cam_.camera_params
        self.set()
    
    ##
    # Intrisics \ref CameraParameters of the right camera.
    @property
    def right_cam(self):
        return self.py_right_cam

    @right_cam.setter
    def right_cam(self, CameraParameters right_cam_) :
        self.calibration.right_cam = right_cam_.camera_params
        self.set()

    ##
    # Left to Right camera transform, expressed in user coordinate system and unit (defined by \ref InitParameters).
    @property
    def stereo_transform(self):
        return self.py_stereo_transform

##
# Structure containing information about a single sensor available in the current device
# \ingroup Sensors_group
# That information about the camera sensors is available in the \ref CameraInformation struct returned by \ref Camera.get_camera_information()
# \note This object is meant to be used as a read-only container, editing any of its fields won't impact the SDK. 
cdef class SensorParameters:
    cdef c_SensorParameters c_sensor_parameters
    cdef c_SENSOR_TYPE sensor_type
    cdef float resolution
    cdef float sampling_rate
    cdef Vector2[float] sensor_range
    cdef float noise_density
    cdef float random_walk
    cdef c_SENSORS_UNIT sensor_unit
    cdef bool is_available

    def set(self):
        self.sensor_type = self.c_sensor_parameters.type
        self.resolution = self.c_sensor_parameters.resolution
        self.sampling_rate = self.c_sensor_parameters.sampling_rate
        self.sensor_range = self.c_sensor_parameters.range
        self.noise_density =  self.c_sensor_parameters.noise_density
        self.random_walk =  self.c_sensor_parameters.random_walk
        self.sensor_unit =  self.c_sensor_parameters.sensor_unit
        self.is_available =  self.c_sensor_parameters.isAvailable

    ##
    # The type of the sensor as \ref SENSOR_TYPE
    @property
    def sensor_type(self):
        return SENSOR_TYPE(<int>self.sensor_type)

    ##
    # The resolution of the sensor.
    @property
    def resolution(self):
        return self.c_sensor_parameters.resolution

    @resolution.setter
    def resolution(self, float resolution_):
        self.c_sensor_parameters.resolution = resolution_

    ##
    # The sampling rate (or ODR) of the sensor.
    @property
    def sampling_rate(self):
        return self.c_sensor_parameters.sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, float sampling_rate_):
        self.c_sensor_parameters.sampling_rate = sampling_rate_

    ##
    # The range values of the sensor. MIN: `sensor_range[0]`, MAX: `sensor_range[1]`
    @property
    def sensor_range(self):
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = self.c_sensor_parameters.range[i]
        return arr

    ##
    # Sets the minimum and the maximum values of the sensor range
    # \param float value1 : min range value of the sensor
    # \param float value2 : max range value of the sensor
    def set_sensor_range(self, value1: float, value2: float) :
        self.c_sensor_parameters.range[0] = value1
        self.c_sensor_parameters.range[1] = value2
        self.set()

    ##
    # Also known as white noise, given as continous (frequency independent). Units will be expressed in sensor_unit/√(Hz). `NAN` if the information is not available 
    @property
    def noise_density(self):
        return self.c_sensor_parameters.noise_density

    @noise_density.setter
    def noise_density(self, float noise_density_):
        self.c_sensor_parameters.noise_density = noise_density_

    ##
    # derived from the Allan Variance, given as continous (frequency independent). Units will be expressed in sensor_unit/s/√(Hz).`NAN` if the information is not available
    @property
    def random_walk(self):
        return self.c_sensor_parameters.random_walk

    @random_walk.setter
    def random_walk(self, float random_walk_):
        self.c_sensor_parameters.random_walk = random_walk_
    
    ##
    # The string relative to the measurement unit of the sensor.
    @property
    def sensor_unit(self):
        return SENSORS_UNIT(<int>self.sensor_unit)

    ##
    # Defines if the sensor is available in your camera. 
    @property
    def is_available(self):
        return self.c_sensor_parameters.isAvailable

##
# Structure containing information about all the sensors available in the current device
# \ingroup Sensors_group
# That information about the camera sensors is available in the \ref CameraInformation struct returned by \ref Camera.getCameraInformation()
# \note This object is meant to be used as a read-only container, editing any of its fields won't impact the SDK. 
cdef class SensorsConfiguration:
    cdef unsigned int firmware_version
    cdef Transform camera_imu_transform
    cdef Transform imu_magnetometer_transform
    cdef SensorParameters accelerometer_parameters
    cdef SensorParameters gyroscope_parameters
    cdef SensorParameters magnetometer_parameters
    cdef SensorParameters barometer_parameters

    def __cinit__(self, Camera py_camera, Resolution resizer=Resolution(0,0)):
        res = c_Resolution(resizer.width, resizer.height)
        caminfo = py_camera.camera.getCameraInformation(res)
        config = caminfo.sensors_configuration
        self.accelerometer_parameters = SensorParameters()
        self.accelerometer_parameters.c_sensor_parameters = config.accelerometer_parameters
        self.accelerometer_parameters.set()
        self.gyroscope_parameters = SensorParameters()
        self.gyroscope_parameters.c_sensor_parameters = config.gyroscope_parameters
        self.gyroscope_parameters.set()
        self.magnetometer_parameters = SensorParameters()
        self.magnetometer_parameters.c_sensor_parameters = config.magnetometer_parameters
        self.magnetometer_parameters.set()
        self.firmware_version = config.firmware_version
        self.barometer_parameters = SensorParameters()
        self.barometer_parameters.c_sensor_parameters = config.barometer_parameters
        self.barometer_parameters.set()
        self.camera_imu_transform = Transform()
        for i in range(16):
            self.camera_imu_transform.transform.m[i] = config.camera_imu_transform.m[i]
        self.imu_magnetometer_transform = Transform()
        for i in range(16):
            self.imu_magnetometer_transform.transform.m[i] = config.imu_magnetometer_transform.m[i]

    ##
    # Configuration of the accelerometer device
    @property
    def accelerometer_parameters(self):
        return self.accelerometer_parameters
    
    ##
    # Configuration of the gyroscope device
    @property
    def gyroscope_parameters(self):
        return self.gyroscope_parameters

    ##
    # Configuration of the magnetometer device    
    @property
    def magnetometer_parameters(self):
        return self.magnetometer_parameters

    ##
    # Configuration of the barometer device
    @property
    def barometer_parameters(self):
        return self.barometer_parameters
    
    ##
    # IMU to Left camera transform matrix, that contains rotation and translation between IMU frame and camera frame.
    @property
    def camera_imu_transform(self):
        return self.camera_imu_transform
    
    ##
    # Magnetometer to IMU transform matrix, that contains rotation and translation between IMU frame and magnetometer frame.
    @property
    def imu_magnetometer_transform(self):
        return self.imu_magnetometer_transform


    ##
    # The internal firmware version of the sensors.
    @property
    def firmware_version(self):
        return self.firmware_version

##
# Structure containing information about the camera sensor. 
# \ingroup Core_group
# That information about the camera is available in the CameraInformation struct returned by Camera::getCameraInformation()
# \note This object is meant to be used as a read-only container, editing any of its fields won't impact the SDK. 
# \note The returned py_calib and py_calib_raw values might vary between two execution due to the \ref InitParameters.camera_disable_self_calib "self-calibration" being ran in the \ref open() method.
cdef class CameraConfiguration:
    cdef CalibrationParameters py_calib
    cdef CalibrationParameters py_calib_raw
    cdef unsigned int firmware_version
    cdef c_Resolution py_res
    cdef float camera_fps

    def __cinit__(self, Camera py_camera, Resolution resizer=Resolution(0,0), int firmware_version_=0, int fps_=0, CalibrationParameters py_calib_= CalibrationParameters(), CalibrationParameters py_calib_raw_= CalibrationParameters()):
        res = c_Resolution(resizer.width, resizer.height)
        self.py_calib = CalibrationParameters()
        caminfo = py_camera.camera.getCameraInformation(res)
        camconfig = caminfo.camera_configuration
        self.py_calib.calibration = camconfig.calibration_parameters
        self.py_calib_raw = CalibrationParameters()
        self.py_calib_raw.calibration = camconfig.calibration_parameters_raw
        self.py_calib.set()
        self.py_calib_raw.set()
        self.firmware_version = camconfig.firmware_version
        self.py_res = camconfig.resolution
        self.camera_fps = camconfig.fps

    ##
    # \ref Resolution of the camera
    @property
    def resolution(self):
        return Resolution(self.py_res.width, self.py_res.height)

    ##
    # \ref FPS of the camera
    @property
    def fps(self):
        return self.camera_fps

    ##
    # Intrinsic and Extrinsic stereo \ref CalibrationParameters for rectified/undistorded images (default).
    @property
    def calibration_parameters(self):
        return self.py_calib

    ##
    # Intrinsic and Extrinsic stereo \ref CalibrationParameters for original images (unrectified/distorded).
    @property
    def calibration_parameters_raw(self):
        return self.py_calib_raw

    ##
    # The internal firmware version of the camera.
    @property
    def firmware_version(self):
        return self.firmware_version



##
# Structure containing information of a single camera (serial number, model, calibration, etc.)
# \ingroup Video_group
# That information about the camera will be returned by \ref Camera.get_camera_information()
# \note This object is meant to be used as a read-only container, editing any of its fields won't impact the SDK.
# \warning \ref CalibrationParameters are returned in \ref COORDINATE_SYSTEM.IMAGE , they are not impacted by the \ref InitParameters.coordinate_system
cdef class CameraInformation:
    cdef unsigned int serial_number
    cdef c_MODEL camera_model
    cdef c_INPUT_TYPE input_type
    cdef CameraConfiguration py_camera_configuration
    cdef SensorsConfiguration py_sensors_configuration
    
    ##
    # Constructor. Gets the \ref CameraParameters from a \ref Camera object.
    # \param py_camera : \ref Camera object.
    # \param resizer : You can specify a \ref Resolution different from default image size to get the scaled camera information. default = (0,0) meaning original image size.
    #
    # \code
    # cam = sl.Camera()
    # res = sl.Resolution(0,0)
    # cam_info = sl.CameraInformation(cam, res)
    # \endcode
    def __cinit__(self, py_camera: Camera, resizer=Resolution(0,0)):
        res = c_Resolution(resizer.width, resizer.height)
        caminfo = py_camera.camera.getCameraInformation(res)

        self.serial_number = caminfo.serial_number
        self.camera_model = caminfo.camera_model
        self.py_camera_configuration = CameraConfiguration(py_camera, resizer)
        self.py_sensors_configuration = SensorsConfiguration(py_camera, resizer)
        self.input_type = caminfo.input_type

    ##
    # Device Sensors configuration as defined in \ref SensorsConfiguration.
    @property
    def sensors_configuration(self):
        return self.py_sensors_configuration

    ##
    # Camera configuration as defined in \ref CameraConfiguration.
    @property
    def camera_configuration(self):
        return self.py_camera_configuration

    ##
    # Input type used in SDK.
    @property
    def input_type(self):
        return INPUT_TYPE(<int>self.input_type)

    ##
    # The model of the camera (ZED, ZED2 or ZED-M).
    @property
    def camera_model(self):
        return MODEL(<int>self.camera_model)

    ##
    # The serial number of the camera.
    @property
    def serial_number(self):
        return self.serial_number

##
# The \ref Mat class can handle multiple matrix formats from 1 to 4 channels, with different value types (float or uchar), and can be stored CPU and/or GPU side.
# \ingroup Core_group
#
# \ref Mat is defined in a row-major order, it means that, for an image buffer, the entire first row is stored first, followed by the entire second row, and so on.
#
# The CPU and GPU buffer aren't automatically synchronized for performance reasons, you can use \ref update_gpu_from_cpu / \ref update_gpu_from_cpu to do it. If you are using the GPU side of the \ref Mat object, you need to make sure to call \ref free before destroying the \ref Camera object. The destruction of the \ref Camera object deletes the CUDA context needed to free the \ref Mat memory.
cdef class Mat:
    cdef c_Mat mat
    ##
    # Constructor.
    # \param width : width of the matrix in pixels. Default: 0
    # \param height : height of the matrix in pixels. Default: 0
    # \param mat_type : the type of the matrix ( [MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...). Default: [MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \code
    # mat = sl.Mat(width=0, height=0, mat_type=MAT_TYPE.F32_C1, memory_type=MEM.CPU)
    # \endcode
    def __cinit__(self, width=0, height=0, mat_type=MAT_TYPE.F32_C1, memory_type=MEM.CPU):
        c_Mat(width, height, <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value)).move(self.mat)

    ##
    # Inits a new \ref Mat .
    # This function directly allocates the requested memory. It calls \ref alloc_size .
    # \param width : width of the matrix in pixels.
    # \param height : height of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...)
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_type(self, width, height, mat_type, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Inits a new \ref Mat from an existing data pointer.
    # This function doesn't allocate the memory.
    # \param width : width of the matrix in pixels.
    # \param height : height of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...)
    # \param ptr : pointer to the data array. (CPU or GPU).
    # \param step : step of the data array. (the Bytes size of one pixel row).
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_cpu(self, width: int, height: int, mat_type: MAT_TYPE, ptr, step, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(<unsigned int>mat_type.value), ptr.encode(), step, <c_MEM>(<unsigned int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Inits a new \ref Mat .
    # This function directly allocates the requested memory. It calls \ref alloc_resolution .
    # \param resolution : the size of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ... )
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_resolution(self, resolution: Resolution, mat_type: MAT_TYPE, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Inits a new \ref Mat from an existing data pointer.
    # This function doesn't allocate the memory.
    # \param resolution : the size of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...)
    # \param ptr : pointer to the data array. (CPU or GPU).
    # \param step : step of the data array. (the Bytes size of one pixel row).
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_resolution_cpu(self, resolution: Resolution, mat_type, ptr, step, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), <c_MAT_TYPE>(<unsigned int>mat_type.value), ptr.encode(), step, <c_MEM>(<unsigned int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Inits a new \ref Mat by copy (shallow copy).
    # This function doesn't allocate the memory.
    # \param mat : a \ref Mat to copy.
    def init_mat(self, matrix: Mat):
        c_Mat(matrix.mat).move(self.mat)

    ##
    # Allocates the \ref Mat memory.
    # \param width : width of the matrix in pixels.
    # \param height : height of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...)
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \warning It erases previously allocated memory.
    def alloc_size(self, width, height, mat_type, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(<size_t> width, <size_t> height, <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value))
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    ##
    # Allocates the \ref Mat memory.
    # \param resolution : the size of the matrix in pixels.
    # \param mat_type : the type of the matrix ([MAT_TYPE.F32_C1](\ref MAT_TYPE) , [MAT_TYPE.U8_C4](\ref MAT_TYPE) ...)
    # \param memory_type : defines where the buffer will be stored. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \warning It erases previously allocated memory.
    def alloc_resolution(self, resolution: Resolution, mat_type: MAT_TYPE, memory_type=MEM.CPU):
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(resolution.resolution, <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value))
            #self.mat.alloc(resolution.width, resolution.height, <c_MAT_TYPE>(<unsigned int>mat_type.value), <c_MEM>(<unsigned int>memory_type.value))
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    ##
    # Free the owned memory.
    # \param memory_type : specifies which memory you wish to free. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    def free(self, memory_type=MEM.CPU):
        if isinstance(memory_type, MEM):
            self.mat.free(<c_MEM>(<unsigned int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Copies data to another \ref Mat (deep copy).
    #
    # \param dst : the \ref Mat where the data will be copied.
    # \param cpy_type : specifies the memories that will be used for the copy. Default: [COPY_TYPE.CPU_CPU](\ref COPY_TYPE) (you cannot change the default value)
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    #
    # \note If the destination is not allocated or doesn't have a compatible \ref MAT_TYPE or \ref Resolution , current memory is freed and new memory is directly allocated.
    def copy_to(self, dst: Mat, cpy_type=COPY_TYPE.CPU_CPU):
        return ERROR_CODE(<int>self.mat.copyTo(dst.mat, <c_COPY_TYPE>(<unsigned int>cpy_type.value)))

    ##
    # Copies data from an other \ref Mat (deep copy).
    # \param src : the \ref Mat where the data will be copied from.
    # \param cpy_type : specifies the memories that will be used for the update. Default: [COPY_TYPE.CPU_CPU](\ref COPY_TYPE) (you cannot change the default value)
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    #
    # \note If the current \ref Mat doesn't have a compatible \ref MAT_TYPE or \ref Resolution with the source, current memory is freed and new memory is directly allocated.
    def set_from(self, src: Mat, cpy_type=COPY_TYPE.CPU_CPU):
        return ERROR_CODE(<int>self.mat.setFrom(<const c_Mat>src.mat, <c_COPY_TYPE>(<unsigned int>cpy_type.value)))

    ##
    # Reads an image from a file (only if [MEM.CPU](\ref MEM) is available on the current \ref Mat ).
    # Supported input files format are PNG and JPEG.
    # \param filepath : file path including the name and extension
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    #
    # \note Supported \ref MAT_TYPE are :
    # \n - [MAT_TYPE.F32_C1](\ref MAT_TYPE) for PNG/PFM/PGM
    # \n - [MAT_TYPE.F32_C3](\ref MAT_TYPE) for PCD/PLY/VTK/XYZ
    # \n - [MAT_TYPE.F32_C4](\ref MAT_TYPE) for PCD/PLY/VTK/WYZ
    # \n - [MAT_TYPE.U8_C1](\ref MAT_TYPE) for PNG/JPG
    # \n - [MAT_TYPE.U8_C3](\ref MAT_TYPE) for PNG/JPG
    # \n - [MAT_TYPE.U8_C4](\ref MAT_TYPE) for PNG/JPG
    def read(self, filepath: str):
        return ERROR_CODE(<int>self.mat.read(filepath.encode()))

    ##
    # Writes the \ref Mat (only if [MEM.CPU](\ref MEM) is available on the current \ref Mat ) into a file as an image.
    # Supported output files format are PNG and JPEG.
    # \param filepath : file path including the name and extension.
    # \param memory_type : memory type of the Mat. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \param compression_level : level of compression between 0 (lowest compression == highest size == highest quality(jpg)) and 100 (highest compression == lowest size == lowest quality(jpg)).
    # \note Specific/default value for compression_level = -1 : This will set the default quality for PNG(30) or JPEG(5).
    # \note compression_level is only supported for [U8_Cx] (\ref MAT_TYPE).
    # \return \ref ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    #
    # \note Supported \ref MAT_TYPE are :
    # \n - [MAT_TYPE.F32_C1](\ref MAT_TYPE) for PNG/PFM/PGM
    # \n - [MAT_TYPE.F32_C3](\ref MAT_TYPE) for PCD/PLY/VTK/XYZ
    # \n - [MAT_TYPE.F32_C4](\ref MAT_TYPE) for PCD/PLY/VTK/WYZ
    # \n - [MAT_TYPE.U8_C1](\ref MAT_TYPE) for PNG/JPG
    # \n - [MAT_TYPE.U8_C3](\ref MAT_TYPE) for PNG/JPG
    # \n - [MAT_TYPE.U8_C4](\ref MAT_TYPE) for PNG/JPG
    def write(self, filepath: str, memory_type=MEM.CPU, compression_level = -1):
        return ERROR_CODE(<int>self.mat.write(filepath.encode(), <c_MEM>(<unsigned int>memory_type.value), compression_level))

    ##
    # Fills the \ref Mat with the given value.
    # This function overwrites all the matrix.
    # \param value : the value to be copied all over the matrix.
    # \param memory_type : defines which buffer to fill. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    def set_to(self, value, memory_type=MEM.CPU):
        if self.get_data_type() == MAT_TYPE.U8_C1:
            return ERROR_CODE(<int>setToUchar1(self.mat, value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            return ERROR_CODE(<int>setToUchar2(self.mat, Vector2[uchar1](value[0], value[1]),
                                      <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            return ERROR_CODE(<int>setToUchar3(self.mat, Vector3[uchar1](value[0], value[1],
                                      value[2]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            return ERROR_CODE(<int>setToUchar4(self.mat, Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            return ERROR_CODE(<int>setToUshort1(self.mat, value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            return ERROR_CODE(<int>setToFloat1(self.mat, value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            return ERROR_CODE(<int>setToFloat2(self.mat, Vector2[float1](value[0], value[1]),
                                      <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            return ERROR_CODE(<int>setToFloat3(self.mat, Vector3[float1](value[0], value[1],
                                      value[2]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            return ERROR_CODE(<int>setToFloat4(self.mat, Vector4[float1](value[0], value[1], value[2],
                                      value[3]), <c_MEM>(<unsigned int>memory_type.value)))

    ##
    # Sets a value to a specific point in the matrix.
    # \param x : specifies the column.
    # \param y : specifies the row.
    # \param value : the value to be set.
    # \param memory_type : defines which memory will be updated. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    #
    # \warning not efficient for GPU, use it on sparse data.
    def set_value(self, x: int, y: int, value, memory_type=MEM.CPU):
        if self.get_data_type() == MAT_TYPE.U8_C1:
            return ERROR_CODE(<int>setValueUchar1(self.mat, x, y, value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            return ERROR_CODE(<int>setValueUchar2(self.mat, x, y, Vector2[uchar1](value[0], value[1]),
                                      <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            return ERROR_CODE(<int>setValueUchar3(self.mat, x, y, Vector3[uchar1](value[0], value[1],
                                      value[2]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            return ERROR_CODE(<int>setValueUshort1(self.mat, x, y, <ushort1>value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            return ERROR_CODE(<int>setValueUchar4(self.mat, x, y, Vector4[uchar1](value[0], value[1], value[2],
                                      value[3]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            return ERROR_CODE(<int>setValueFloat1(self.mat, x, y, value, <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            return ERROR_CODE(<int>setValueFloat2(self.mat, x, y, Vector2[float1](value[0], value[1]),
                                      <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            return ERROR_CODE(<int>setValueFloat3(self.mat, x, y, Vector3[float1](value[0], value[1],
                                      value[2]), <c_MEM>(<unsigned int>memory_type.value)))
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            return ERROR_CODE(<int>setValueFloat4(self.mat, x, y, Vector4[float1](value[0], value[1], value[2],
                                      value[3]), <c_MEM>(<unsigned int>memory_type.value)))

    ##
    # Returns the value of a specific point in the matrix.
    # \param x : specifies the column
    # \param y : specifies the row
    # \param memory_type : defines which memory should be read. Default: [MEM.CPU](\ref MEM) (you cannot change this default value)
    # \return ERROR_CODE.SUCCESS if everything went well, \ref ERROR_CODE.FAILURE otherwise.
    def get_value(self, x: int, y: int, memory_type=MEM.CPU):
        cdef uchar1 value1u
        cdef Vector2[uchar1] value2u = Vector2[uchar1](0,0)
        cdef Vector3[uchar1] value3u = Vector3[uchar1](0,0,0)
        cdef Vector4[uchar1] value4u = Vector4[uchar1](0,0,0,0)

        cdef ushort1 value1us

        cdef float1 value1f
        cdef Vector2[float1] value2f = Vector2[float1](0,0)
        cdef Vector3[float1] value3f = Vector3[float1](0,0,0)
        cdef Vector4[float1] value4f = Vector4[float1](0,0,0,0)

        if self.get_data_type() == MAT_TYPE.U8_C1:
            status = getValueUchar1(self.mat, x, y, &value1u, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), value1u
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            status = getValueUchar2(self.mat, x, y, &value2u, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value2u.ptr()[0], value2u.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            status = getValueUchar3(self.mat, x, y, &value3u, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value3u.ptr()[0], value3u.ptr()[1], value3u.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            status = getValueUchar4(self.mat, x, y, &value4u, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value4u.ptr()[0], value4u.ptr()[1], value4u.ptr()[2],
                                                         value4u.ptr()[3]])
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            status = getValueUshort1(self.mat, x, y, &value1us, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), value1us
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            status = getValueFloat1(self.mat, x, y, &value1f, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), value1f
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            status = getValueFloat2(self.mat, x, y, &value2f, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value2f.ptr()[0], value2f.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            status = getValueFloat3(self.mat, x, y, &value3f, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value3f.ptr()[0], value3f.ptr()[1], value3f.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            status = getValueFloat4(self.mat, x, y, &value4f, <c_MEM>(<unsigned int>memory_type.value))
            return ERROR_CODE(<int>status), np.array([value4f.ptr()[0], value4f.ptr()[1], value4f.ptr()[2],
                                                         value4f.ptr()[3]])

    ##
    # Returns the width of the matrix.
    # \return The width of the matrix in pixels.
    def get_width(self):
        return self.mat.getWidth()

    ##
    # Returns the height of the matrix.
    # \return The height of the matrix in pixels.
    def get_height(self):
        return self.mat.getHeight()

    ##
    # Returns the resolution of the matrix.
    # \return The resolution of the matrix in pixels.
    def get_resolution(self):
        return Resolution(self.mat.getResolution().width, self.mat.getResolution().height)

    ##
    # Returns the number of values stored in one pixel.
    # \return The number of values in a pixel. 
    def get_channels(self):
        return self.mat.getChannels()

    ##
    # Returns the format of the matrix.
    # \return The format of the current \ref Mat .
    def get_data_type(self):
        return MAT_TYPE(<int>self.mat.getDataType())

    ##
    # Returns the format of the matrix.
    # \return The format of the current \ref Mat
    def get_memory_type(self):
        return MEM(<int>self.mat.getMemoryType())

    ##
    # Returns the Mat as a Numpy Array
    # This is for convenience to mimic the PyTorch API https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html
    # This is like an alias of \ref get_data() function
    # \param force : defines if the memory of the Mat need to be duplicated or not. The fastest is deep_copy at False but the sl::Mat memory must not be released to use the numpy array.
    # \return A Numpy array containing the \ref Mat data.
    def numpy(self, force=False):
        return self.get_data(memory_type=MEM.CPU, deep_copy=force)

    ##
    # Cast the data of the \ref Mat in a Numpy array (with or without copy).
    # \param memory_type : defines which memory should be read. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \param deep_copy : defines if the memory of the Mat need to be duplicated or not. The fastest is deep_copy at False but the sl::Mat memory must not be released to use the numpy array.
    # \return A Numpy array containing the \ref Mat data.
    def get_data(self, memory_type=MEM.CPU, deep_copy=False):
        
        shape = None
        cdef np.npy_intp cython_shape[3]
        cython_shape[0] = <np.npy_intp> self.mat.getHeight()
        cython_shape[1] = <np.npy_intp> self.mat.getWidth()
        cython_shape[2] = <np.npy_intp> self.mat.getChannels()

        if self.mat.getChannels() == 1:
            shape = (self.mat.getHeight(), self.mat.getWidth())
        else:
            shape = (self.mat.getHeight(), self.mat.getWidth(), self.mat.getChannels())

        cdef size_t size = 0
        dtype = None
        nptype = None
        npdim = None
        if self.mat.getDataType() in (c_MAT_TYPE.U8_C1, c_MAT_TYPE.U8_C2, c_MAT_TYPE.U8_C3, c_MAT_TYPE.U8_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()
            dtype = np.uint8
            nptype = np.NPY_UINT8
        elif self.mat.getDataType() in (c_MAT_TYPE.F32_C1, c_MAT_TYPE.F32_C2, c_MAT_TYPE.F32_C3, c_MAT_TYPE.F32_C4):
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()*sizeof(float)
            dtype = np.float32
            nptype = np.NPY_FLOAT32
        elif self.mat.getDataType() == c_MAT_TYPE.U16_C1:
            size = self.mat.getHeight()*self.mat.getWidth()*self.mat.getChannels()*sizeof(ushort)
            dtype = np.ushort
            nptype = np.NPY_UINT16
        else:
            raise RuntimeError("Unknown Mat data_type value: {0}".format(<int>self.mat.getDataType()))

        if self.mat.getDataType() in (c_MAT_TYPE.U8_C1, c_MAT_TYPE.F32_C1, c_MAT_TYPE.U16_C1):
            npdim = 2
        else:
            npdim = 3

        cdef np.ndarray arr = np.empty(shape, dtype=dtype)

        if isinstance(memory_type, MEM):
            if deep_copy:
                if self.mat.getDataType() == c_MAT_TYPE.U8_C1:
                    memcpy(<void*>arr.data, <void*>getPointerUchar1(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.U8_C2:
                    memcpy(<void*>arr.data, <void*>getPointerUchar2(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.U8_C3:
                    memcpy(<void*>arr.data, <void*>getPointerUchar3(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.U8_C4:
                    memcpy(<void*>arr.data, <void*>getPointerUchar4(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.U16_C1:
                    memcpy(<void*>arr.data, <void*>getPointerUshort1(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.F32_C1:
                    memcpy(<void*>arr.data, <void*>getPointerFloat1(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.F32_C2:
                    memcpy(<void*>arr.data, <void*>getPointerFloat2(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.F32_C3:
                    memcpy(<void*>arr.data, <void*>getPointerFloat3(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
                elif self.mat.getDataType() == c_MAT_TYPE.F32_C4:
                    memcpy(<void*>arr.data, <void*>getPointerFloat4(self.mat, <c_MEM>(<unsigned int>memory_type.value)), size)
            else: # Thanks to BDO for the initial implementation!
                arr = np.PyArray_SimpleNewFromData(npdim, cython_shape, nptype, <void*>getPointerUchar1(self.mat, <c_MEM>(<unsigned int>memory_type.value)))
        else:
            raise TypeError("Argument is not of MEM type.")

        return arr

    ##
    # Returns the memory step in Bytes (the Bytes size of one pixel row).
    # \param memory_type : defines which memory should be read. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return The step in bytes of the specified memory.
    def get_step_bytes(self, memory_type=MEM.CPU):
        if type(memory_type) == MEM:
            return self.mat.getStepBytes(<c_MEM>(<unsigned int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Returns the memory step in number of elements (the number of values in one pixel row).
    # \param memory_type : defines which memory should be read. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return The step in number of elements.
    def get_step(self, memory_type=MEM.CPU):
        if type(memory_type) == MEM:
            return self.mat.getStep(<c_MEM>(<unsigned int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Returns the size in bytes of one pixel.
    # \return The size in bytes of a pixel.
    def get_pixel_bytes(self):
        return self.mat.getPixelBytes()

    ##
    # Returns the size in bytes of a row.
    # \return The size in bytes of a row.
    def get_width_bytes(self):
        return self.mat.getWidthBytes()

    ##
    # Returns the information about the \ref Mat into a string.
    # \return A string containing the \ref Mat information.
    def get_infos(self):
        return to_str(self.mat.getInfos()).decode()

    ##
    # Defines whether the \ref Mat is initialized or not.
    # \return True if current \ref Mat has been allocated (by the constructor or therefore).
    def is_init(self):
        return self.mat.isInit()

    ##
    # Returns whether the \ref Mat is the owner of the memory it accesses.
    #
    # If not, the memory won't be freed if the Mat is destroyed.
    # \return True if the \ref Mat is owning its memory, else false.
    def is_memory_owner(self):
        return self.mat.isMemoryOwner()

    ##
    # Duplicates \ref Mat by copy (deep copy).
    # \param py_mat : the reference to the \ref Mat to copy. This function copies the data array(s), it marks the new \ref Mat as the memory owner.
    def clone(self, py_mat: Mat):
        return ERROR_CODE(<int>self.mat.clone(py_mat.mat))

    ##
    # Moves Mat data to another \ref Mat.
    #
    # This function gives the attribute of the current \ref Mat to the specified one. (No copy).
    # \param py_mat : the \ref Mat to move.
    #
    # \note the current \ref Mat is then no more usable since it loses its attributes.
    def move(self, py_mat: Mat):
        return ERROR_CODE(<int>self.mat.move(py_mat.mat))

    ##
    # Swaps the content of the provided \ref Mat (only swaps the pointers, no data copy). Static Method.
    #
    # This function swaps the pointers of the given \ref Mat.
    # \param mat1 : the first mat.
    # \param mat2 : the second mat.
    @staticmethod
    def swap(mat1: Mat, mat2: Mat):
        cdef c_Mat tmp
        tmp = mat1.mat
        mat1.mat = mat2.mat
        mat2.mat = tmp

    ##
    # Gets the pointer of the content of the \ref Mat.
    #
    # \param memory_type : Defines which memory you want to get. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return the pointer of the content of the \ref Mat.
    def get_pointer(self, memory_type=MEM.CPU) :
        ptr = <unsigned long long>getPointerUchar1(self.mat, <c_MEM>(<unsigned int>memory_type.value))
        return ptr

    ##
    # The name of the \ref Mat (optional). In \ref verbose mode, it's used to indicate which \ref Mat is printing information. Default set to "n/a" to avoid empty string if not filled.
    @property
    def name(self):
        if not self.mat.name.empty():
            return self.mat.name.get().decode()
        else:
            return ""

    @name.setter
    def name(self, str name_):
        self.mat.name.set(name_.encode())

    ##
    # The timestamp of the \ref Mat.
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp = self.mat.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, timestamp : Timestamp):
        self.mat.timestamp.data_ns = timestamp.get_nanoseconds()

    ##
    # Whether the \ref Mat can display information or not.
    @property
    def verbose(self):
        return self.mat.verbose

    @verbose.setter
    def verbose(self, bool verbose_):
        self.mat.verbose = verbose_

    def __repr__(self):
        return self.get_infos()


##
# Designed to contain rotation data of the positional tracking. It inherits from the generic \ref Matrix3f .
# \ingroup PositionalTracking_group
cdef class Rotation(Matrix3f):
    cdef c_Rotation* rotation
    def __cinit__(self):
        if type(self) is Rotation:
            self.rotation = self.mat = new c_Rotation()
    
    def __dealloc__(self):
        if type(self) is Rotation:
            del self.rotation

    ##
    # Deep copy from another \ref Rotation .
    # \param rot : \ref Rotation to be copied.
    def init_rotation(self, rot: Rotation):
        for i in range(9):
            self.rotation.r[i] = rot.rotation.r[i]

    ##
    # Inits the \ref Rotation from a \ref Matrix3f .
    # \param matrix : \ref Matrix3f to be used.
    def init_matrix(self, matrix: Matrix3f):
        for i in range(9):
            self.rotation.r[i] = matrix.mat.r[i]

    ##
    # Inits the \ref Rotation from a \ref Orientation .
    # \param orient : \ref Orientation to be used.
    def init_orientation(self, orient: Orientation):
        self.rotation.setOrientation(orient.orientation)

    ##
    # Inits the \ref Rotation from an angle and an arbitrary 3D axis.
    # \param angle : The rotation angle in rad.
    # \param axis : the 3D axis (\ref Translation) to rotate around
    def init_angle_translation(self, angle: float, axis: Translation):
        cdef c_Rotation tmp = c_Rotation(angle, axis.translation)        
        for i in range(9):
            self.rotation.r[i] = tmp.r[i]

    ##
    # Sets the \ref Rotation from an \ref Orientation .
    # \param py_orientation : the \ref Orientation containing the rotation to set.
    def set_orientation(self, py_orientation: Orientation):
        self.rotation.setOrientation(py_orientation.orientation)

    ##
    # Returns the \ref Orientation corresponding to the current \ref Rotation .
    # \return The orientation of the current rotation.
    def get_orientation(self):
        py_orientation = Orientation()        
        py_orientation.orientation = self.rotation.getOrientation()
        return py_orientation

    ##
    # Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.
    # \return The rotation vector (numpy array)
    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.rotation.getRotationVector()[i]
        return arr

    ##
    # Sets the \ref Rotation from a rotation vector (using Rodrigues' transformation).
    # \param input0 : First float value
    # \param input1 : Second float value
    # \param input2 : Third float value
    def set_rotation_vector(self, input0: float, input1: float, input2: float):
        self.rotation.setRotationVector(Vector3[float](input0, input1, input2))

    ##
    # Converts the \ref Rotation as Euler angles.
    # \param radian : Bool to define whether the angle in is radian (True) or degree (False). Default: True
    # \return The Euler angles, as a numpy array representing the rotations arround the X, Y and Z axes.
    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.rotation.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    ##
    # Sets the \ref Rotation from the Euler angles
    # \param input0 : Roll value
    # \param input1 : Pitch value
    # \param input2 : Yaw value
    # \param radian : Bool to define whether the angle in is radian (True) or degree (False). Default: True
    def set_euler_angles(self, input0: float, input1: float, input2: float, radian=True):
        if isinstance(radian, bool):
            self.rotation.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")

##
# Designed to contain translation data of the positional tracking.
# \ingroup PositionalTracking_group
#
# \ref Translation is a vector as [tx, ty, tz]. You can access the data with the \ref get method that returns a numpy array.
cdef class Translation:
    cdef c_Translation translation
    def __cinit__(self):
        self.translation = c_Translation()

    ##
    # Deep copy from another \ref Translation
    # \param tr : \ref Translation to be copied
    def init_translation(self, tr: Translation):
        self.translation = c_Translation(tr.translation)

    ##
    # \param t1 : First float value
    # \param t2 : Second float value
    # \param t3 : Third float value
    def init_vector(self, t1: float, t2: float, t3: float):
        self.translation = c_Translation(t1, t2, t3)

    ##
    # Normalizes the current translation.
    def normalize(self):
        self.translation.normalize()

    ##
    # Gets the normalized version of a given \ref Translation .
    # \param tr : \ref Translation to be used
    # \return Another \ref Translation object, which is equal to tr.normalize.
    def normalize_translation(self, tr: Translation):
        py_translation = Translation()
        py_translation.translation = self.translation.normalize(tr.translation)
        return py_translation

    ##
    # Gets the size of the translation vector.
    # \return the vector size
    def size(self):
        return self.translation.size()

    ##
    # Computes the dot product of two \ref Translation objects
    # \param tr1 : first vector, defined ad a \ref Translation
    # \param tr2 : sencond vector, defined as a \ref Translation
    # \return dot product of tr1 and tr2
    def dot_translation(tr1: Translation, tr2: Translation):
        py_translation = Translation()
        return py_translation.translation.dot(tr1.translation,tr2.translation)

    ##
    # Gets the \ref Translation as a numpy array.
    # \return A numpy array of float with the \ref Translation values.
    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.translation(i)
        return arr

    def __mul__(Translation self, Orientation other):
        tr = Translation()
        tr.translation = self.translation * other.orientation
        return tr

##
# Designed to contain orientation (quaternion) data of the positional tracking.
# \ingroup PositionalTracking_group
#
# \ref Orientation is a vector defined as [ox, oy, oz, ow].
cdef class Orientation:
    cdef c_Orientation orientation
    def __cinit__(self):
        self.orientation = c_Orientation()

    ##
    # Deep copy from another \ref Orientation
    # \param orient : \ref Orientation to be copied.
    def init_orientation(self, orient: Orientation):
        self.orientation = c_Orientation(orient.orientation)

    ##
    # Inits \ref Orientation from float values.
    # \param v0 : ox value
    # \param v1 : oy value
    # \param v2 : oz value
    # \param v3 : ow value
    def init_vector(self, v0: float, v1: float, v2: float, v3: float):
        self.orientation = c_Orientation(Vector4[float](v0, v1, v2, v3))

    ##
    # Inits \ref Orientation from \ref Rotation .
    #
    # It converts the \ref Rotation representation to the \ref Orientation one.
    # \param rotation : \ref Rotation to be converted
    def init_rotation(self, rotation: Rotation):
        self.orientation = c_Orientation(rotation.rotation[0])

    ##
    # Inits \ref Orientation from \ref Translation
    # \param tr1 : First \ref Translation
    # \param tr2 : Second \ref Translation
    def init_translation(self, tr1: Translation, tr2: Translation):
        self.orientation = c_Orientation(tr1.translation, tr2.translation)

    ##
    # Sets the orientation from a \ref Rotation
    # \param rotation : the \ref Rotation to be used.
    def set_rotation_matrix(self, py_rotation: Rotation):
        self.orientation.setRotationMatrix(py_rotation.rotation[0])

    ##
    # Returns the current orientation as a \ref Rotation .
    # \return The rotation computed from the orientation data.
    def get_rotation_matrix(self):
        cdef c_Rotation tmp = self.orientation.getRotationMatrix()
        py_rotation = Rotation()
        for i in range(9):
            py_rotation.rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Sets the current \ref Orientation to identity.
    def set_identity(self):
        self.orientation.setIdentity()
        # return self

    ##
    # Creates an \ref Orientation initialized to identity.
    # \return An identity class \ref Orientation .
    def identity(self):
        self.orientation.identity()
        return self

    ##
    # Fills the current \ref Orientation with zeros.
    def set_zeros(self):
        self.orientation.setZeros()

    ##
    # Creates an \ref Orientation filled with zeros.
    # \return An \ref Orientation filled with zeros.
    def zeros(self, orient=Orientation()):
        (<Orientation>orient).orientation.setZeros()
        return orient

    ##
    # Normalizes the current \ref Orientation .
    def normalize(self):
        self.orientation.normalise()

    ##
    # Creates the normalized version of an existing \ref Orientation .
    # \param orient : the \ref Orientation to be used.
    # \return The normalized version of the \ref Orientation .
    @staticmethod
    def normalize_orientation(orient: Orientation):
        orient.orientation.normalise()
        return orient

    ##
    # The size of the orientation vector.
    # \return the size of the orientation vector.
    def size(self):
        return self.orientation.size()

    ##
    # Returns a numpy array of the \ref Orientation .
    # \return A numpy array of the \ref Orientation .
    def get(self):
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.orientation(i)
        return arr

    def __mul__(Orientation self, Orientation other):
        orient = Orientation()
        orient.orientation = self.orientation * other.orientation
        return orient


##
# Designed to contain translation and rotation data of the positional tracking.
# \ingroup PositionalTracking_group
# It contains the orientation as well. It can be used to create any type of Matrix4x4 or \ref Matrix4f that must be specifically used for handling a rotation and position information (OpenGL, Tracking...). It inherits from the generic \ref Matrix4f .
cdef class Transform(Matrix4f):
    cdef c_Transform *transform
    def __cinit__(self):
        if type(self) is Transform:
            self.transform = self.mat = new c_Transform()

    def __dealloc__(self):
        if type(self) is Transform:
            del self.transform

    ##
    # Deep copy from another \ref Transform
    # \param motion : \ref Transform to be copied
    def init_transform(self, motion: Transform):
        for i in range(16):
            self.transform.m[i] = motion.transform.m[i]

    ##
    # Inits \ref Transform from a \ref Matrix4f
    # \param matrix : \ref Matrix4f to be used
    def init_matrix(self, matrix: Matrix4f):
        for i in range(16):
            self.transform.m[i] = matrix.mat.m[i]

    ##
    # Inits \ref Transform from a \ref Rotation and a \ref Translation .
    # \param rot : \ref Rotation to be used.
    # \param tr : \ref Translation to be used.
    def init_rotation_translation(self, rot: Rotation, tr: Translation):
        cdef c_Transform tmp = c_Transform(rot.rotation[0], tr.translation)
        for i in range(16):
            self.transform.m[i] = tmp.m[i]

    ##
    # Inits \ref Transform from a \ref Orientation and a \ref Translation .
    # \param orient : \ref Orientation to be used
    # \param tr : \ref Translation to be used
    def init_orientation_translation(self, orient: Orientation, tr: Translation):
        cdef c_Transform tmp = c_Transform(orient.orientation, tr.translation)
        for i in range(16):
            self.transform.m[i] = tmp.m[i]

    ##
    # Sets the rotation of the current \ref Transform from a \ref Rotation .
    # \param py_rotation : the \ref Rotation to be used.
    def set_rotation_matrix(self, py_rotation: Rotation):
        self.transform.setRotationMatrix(py_rotation.rotation[0])

    ##
    # Returns the \ref Rotation of the current \ref Transform .
    # \return The \ref Rotation of the current \ref Transform .
    def get_rotation_matrix(self):
        cdef c_Rotation tmp = self.transform.getRotationMatrix()
        py_rotation = Rotation()
        for i in range(9):
            py_rotation.rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Sets the translation of the current \ref Transform from a \ref Translation .
    # \param py_translation : \ref Translation to be used.
    def set_translation(self, py_translation: Translation):
        self.transform.setTranslation(py_translation.translation)

    ##
    # Returns the \ref Translation of the current \ref Transform .
    # \return the \ref Translation created from the \ref Transform values .
    # \warning the given \ref Translation contains a copy of the \ref Transform values.
    def get_translation(self):
        py_translation = Translation()
        py_translation.translation = self.transform.getTranslation()
        return py_translation

    ##
    # Sets the orientation of the current \ref Transform from an \ref Orientation .
    # \param py_orientation : \ref Orientation to be used.
    def set_orientation(self, py_orientation: Orientation):
        self.transform.setOrientation(py_orientation.orientation)

    ##
    # Returns the \ref Orientation of the current \ref Transform .
    # \return The \ref Orientation created from the \ref Transform values.
    # \warning the given \ref Orientation contains a copy of the \ref Transform values.
    def get_orientation(self):
        py_orientation = Orientation()
        py_orientation.orientation = self.transform.getOrientation()
        return py_orientation

    ##
    # Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.
    # \return The rotation vector (numpy array)
    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.transform.getRotationVector()[i]
        return arr

    ##
    # Sets the Rotation 3x3 of the Transform with a 3x1 rotation vector (using Rodrigues' transformation).
    # \param input0 : First float value
    # \param input1 : Second float value
    # \param input2 : Third float value
    def set_rotation_vector(self, input0: float, input1: float, input2: float):
        self.transform.setRotationVector(Vector3[float](input0, input1, input2))

    ##
    # Converts the \ref Rotation of the \ref Transform as Euler angles.
    # \param radian : True if the angle is in radian, False otherwise. Default: True
    # \return The Euler angles, as 3x1 numpy array representing the rotations around the x, y and z axes.
    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.transform.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    ##
    # Sets the \ref Rotation of the \ref Transform from the Euler angles.
    # \param input0 : First float euler value.
    # \param input1 : Second float euler value.
    # \param input2 : Third float euler value.
    # \param radian : True if the angle is in radian, False otherwise. Default: True
    def set_euler_angles(self, input0: float, input1: float, input2: float, radian=True):
        if isinstance(radian, bool):
            self.transform.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")


##
# Lists available mesh file formats.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | PLY | Contains only vertices and faces. |
# | PLY_BIN | Contains only vertices and faces, encoded in binary. |
# | OBJ | Contains vertices, normals, faces and textures information if possible. |
class MESH_FILE_FORMAT(enum.Enum):
    PLY = <int>c_MESH_FILE_FORMAT.PLY
    PLY_BIN = <int>c_MESH_FILE_FORMAT.PLY_BIN
    OBJ = <int>c_MESH_FILE_FORMAT.OBJ
    LAST = <int>c_MESH_FILE_FORMAT.MESH_FILE_FORMAT_LAST

##
# Lists available mesh texture formats.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | RGB | The texture has 3 channels. |
# | RGBA | The texture has 4 channels.|
class MESH_TEXTURE_FORMAT(enum.Enum):
    RGB = <int>c_MESH_TEXTURE_FORMAT.RGB
    RGBA = <int>c_MESH_TEXTURE_FORMAT.RGBA
    LAST = <int>c_MESH_TEXTURE_FORMAT.MESH_TEXTURE_FORMAT_LAST

##
# Lists available mesh filtering intensity.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | LOW | Clean the mesh by closing small holes and removing isolated faces |
# | MEDIUM | Soft decimation and smoothing. |
# | HIGH | Decimate the number of triangles and apply a soft smooth. |
class MESH_FILTER(enum.Enum):
    LOW = <int>c_MESH_FILTER.LOW
    MEDIUM = <int>c_MESH_FILTER.MESH_FILTER_MEDIUM
    HIGH = <int>c_MESH_FILTER.HIGH

##
# Lists available plane types detected from the orientation
#
# \ingroup SpatialMapping_group
#
# | Enumerator |                  |
# |------------|------------------|
# | HORIZONTAL |                  |
# | VERTICAL | |
# | UNKNOWN | |
class PLANE_TYPE(enum.Enum):
    HORIZONTAL = <int>c_PLANE_TYPE.HORIZONTAL
    VERTICAL = <int>c_PLANE_TYPE.VERTICAL
    UNKNOWN = <int>c_PLANE_TYPE.UNKNOWN
    LAST = <int>c_PLANE_TYPE.PLANE_TYPE_LAST

##
# Defines the behavior of the \ref Mesh.filter() function.
# \ingroup SpatialMapping_group
# The constructor sets all the default parameters.
cdef class MeshFilterParameters:
    cdef c_MeshFilterParameters* meshFilter
    def __cinit__(self):
        self.meshFilter = new c_MeshFilterParameters(c_MESH_FILTER.LOW)

    def __dealloc__(self):
        del self.meshFilter

    ##
    # Set the filtering intensity.
    # \param filter : the desired \ref MESH_FILTER
    def set(self, filter=MESH_FILTER.LOW):
        if isinstance(filter, MESH_FILTER):
            self.meshFilter.set(<c_MESH_FILTER>(<unsigned int>filter.value))
        else:
            raise TypeError("Argument is not of MESH_FILTER type.")

    ##
    # Saves the current bunch of parameters into a file.
    # \param filename : the path to the file in which the parameters will be stored.
    # \return true if the file was successfully saved, otherwise false.
    def save(self, filename: str):
        filename_save = filename.encode()
        return self.meshFilter.save(String(<char*> filename_save))

    ##
    # Loads the values of the parameters contained in a file.
    # \param filename : the path to the file from which the parameters will be loaded.
    # \return true if the file was successfully loaded, otherwise false.
    def load(self, filename: str):
        filename_load = filename.encode()
        return self.meshFilter.load(String(<char*> filename_load))

##
# Represents a sub fused point cloud, it contains local vertices and colors.
# \ingroup SpatialMapping_group
# Vertices and normals have the same size.
cdef class PointCloudChunk :
    cdef c_PointCloudChunk chunk

    def __cinit__(self):
        self.chunk = c_PointCloudChunk()
    ##
    # Vertices are defined by a colored 3D point {x, y, z, rgba}. The information is stored in a numpy array.
    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 4), dtype=np.float32)
        for i in range(self.chunk.vertices.size()):
            for j in range(4):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    ##
    # Normals are defined by three components, {nx, ny, nz}. The information is stored in a numpy array.
    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3), dtype=np.float32)
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    ##
    # Timestamp of the latest update in nanoseconds.
    @property
    def timestamp(self):
        return self.chunk.timestamp

    ##
    # 3D centroid of the chunk. The information is stored in a numpy array.
    @property
    def barycenter(self):
        cdef np.ndarray arr = np.zeros(3, dtype=np.float32)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    ##
    # True if the chunk has been updated by an inner process.
    @property
    def has_been_updated(self):
        return self.chunk.has_been_updated

    ##
    # Clears all chunk data.
    def clear(self):
        self.chunk.clear()

##
# Represents a sub-mesh, it contains local vertices and triangles.
# \ingroup SpatialMapping_group
#
# Vertices and normals have the same size and are linked by id stored in triangles.
# \note uv contains data only if your mesh have textures (by loading it or after calling apply_texture)
cdef class Chunk:
    cdef c_Chunk chunk
    def __cinit__(self):
        self.chunk = c_Chunk()

    ##
    # Vertices are defined by a 3D point (numpy array).
    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 3), dtype=np.float32)
        for i in range(self.chunk.vertices.size()):
            for j in range(3):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    ##
    # List of triangles, defined as a set of three vertices. The information is stored in a numpy array
    @property
    def triangles(self):
        cdef np.ndarray arr = np.zeros((self.chunk.triangles.size(), 3), dtype = np.uint32)
        for i in range(self.chunk.triangles.size()):
            for j in range(3):
                arr[i,j] = self.chunk.triangles[i].ptr()[j]
        return arr

    ##
    # Normals are defined by three components (numpy array). Normals are defined for each vertex.
    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3), dtype=np.float32)
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    ##
    # Colors are defined by three components, {b, g, r}. Colors are defined for each vertex.
    @property
    def colors(self):
        cdef np.ndarray arr = np.zeros((self.chunk.colors.size(), 3), dtype = np.ubyte)
        for i in range(self.chunk.colors.size()):
            for j in range(3):
                arr[i,j] = self.chunk.colors[i].ptr()[j]
        return arr

    ##
    # UVs define the 2D projection of each vertex onto the texture.
    # Values are normalized [0;1], starting from the bottom left corner of the texture (as requested by opengl).
    # In order to display a textured mesh you need to bind the Texture and then draw each triangle by picking its uv values.
    # \note Contains data only if your mesh has textures (by loading it or calling \ref apply_texture).
    @property
    def uv(self):
        cdef np.ndarray arr = np.zeros((self.chunk.uv.size(), 2), dtype=np.float32)
        for i in range(self.chunk.uv.size()):
            for j in range(2):
                arr[i,j] = self.chunk.uv[i].ptr()[j]
        return arr

    ##
    # Timestamp of the latest update.
    @property
    def timestamp(self):
        return self.chunk.timestamp

    ##
    # 3D centroid of the chunk.
    @property
    def barycenter(self):
        cdef np.ndarray arr = np.zeros(3, dtype=np.float32)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    ##
    # True if the chunk has been updated by an inner process.
    @property
    def has_been_updated(self):
        return self.chunk.has_been_updated

    ##
    # Clears all chunk data.
    def clear(self):
        self.chunk.clear()

##
# A fused point cloud contains both geometric and color data of the scene captured by spatial mapping.
# \ingroup SpatialMapping_group
#
# By default the fused point cloud is defined as a set of point cloud chunks, this way we update only the required data, avoiding a time consuming remapping process every time a small part of the fused point cloud is changed.
cdef class FusedPointCloud :
    cdef c_FusedPointCloud* fpc
    def __cinit__(self):
        self.fpc = new c_FusedPointCloud()

    def __dealloc__(self):
        del self.fpc

    ##
    # contains the list of chunks
    @property
    def chunks(self):
        list = []
        for i in range(self.fpc.chunks.size()):
            py_chunk = PointCloudChunk()
            py_chunk.chunk = self.fpc.chunks[i]
            list.append(py_chunk)
        return list

    ##
    # gets a chunk from the list
    def __getitem__(self, x):
        return self.chunks[x]

    ##
    # Vertices are defined by colored 3D points {x, y, z, rgba}. The information is stored in a numpy array.
    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.fpc.vertices.size(), 4), dtype=np.float32)
        for i in range(self.fpc.vertices.size()):
            for j in range(4):
                arr[i,j] = self.fpc.vertices[i].ptr()[j]
        return arr

    ##
    # Normals are defined by three components, {nx, ny, nz}. Normals are defined for each vertices. The information is stored in a numpy array.
    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.fpc.normals.size(), 3), dtype=np.float32)
        for i in range(self.fpc.normals.size()):
            for j in range(3):
                arr[i,j] = self.fpc.normals[i].ptr()[j]
        return arr

    ##
    # Saves the current fused point cloud into a file.
    # \param filename : the path and filename of the mesh.
    # \param typeMesh : defines the file type (extension). default :  [MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT).
    # \param id : Specifies a set of chunks to be saved, if none provided all chunks are saved. default : (empty).
    # \return True if the file was successfully saved, false otherwise.
    #
    # \note Only [MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT) support textures data.
    # \note This function operates on the fused point cloud not on the chunks. This way you can save different parts of your fused point cloud (update with \ref update_from_chunklist).
    def save(self, filename: str, typeMesh=MESH_FILE_FORMAT.OBJ, id=[]):
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.fpc.save(String(filename.encode()), <c_MESH_FILE_FORMAT>(<unsigned int>typeMesh.value), id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    ##
    # Loads the fused point cloud from a file.
    # \param filename : the path and filename of the fused point cloud (do not forget the extension).
    # \param update_chunk_only : if set to false the fused point cloud data (vertices/normals) are updated otherwise only the chunk data is updated. default : true.
    # \return True if the loading was successful, false otherwise.
    # \note Updating the fused point cloud is time consuming, consider using only chunks for better performances.
    def load(self, filename: str, update_chunk_only=True):
        if isinstance(update_chunk_only, bool):
            return self.fpc.load(String(filename.encode()), update_chunk_only)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Clears all the data.
    def clear(self):
        self.fpc.clear()

    ##
    # Updates \ref vertices / \ref normals / \ref colors from chunks' data pointed by the given chunk list.
    # \param id : the indices of chunks which will be concatenated. default : (empty).
    # \note If the given chunkList is empty, all chunks will be used.
    def update_from_chunklist(self, id=[]):
        self.fpc.updateFromChunkList(id)

    ##
    # Computes the total number of triangles stored in all chunks.
    # \return The number of points stored in all chunks.
    def get_number_of_points(self):
        return self.fpc.getNumberOfPoints()


##
# A mesh contains the geometric (and optionally texture) data of the scene captured by spatial mapping.
# \ingroup SpatialMapping_group
# By default the mesh is defined as a set of chunks, this way we update only the data that has to be updated avoiding a time consuming remapping process every time a small part of the Mesh is updated.
cdef class Mesh:
    cdef c_Mesh* mesh
    def __cinit__(self):
        self.mesh = new c_Mesh()

    def __dealloc__(self):
        del self.mesh

    ##
    # contains the list of chunks
    @property
    def chunks(self):
        list_ = []
        for i in range(self.mesh.chunks.size()):        
            py_chunk = Chunk()
            py_chunk.chunk = self.mesh.chunks[i]
            list_.append(py_chunk)
        return list_

    ##
    # gets a chunk from the list
    def __getitem__(self, x):
        return self.chunks[x]

    ##
    # Filters the mesh.
    # The resulting mesh in smoothed, small holes are filled and small blobs of non connected triangles are deleted.
    # \param params : defines the filtering parameters, for more info checkout the \ref MeshFilterParameters documentation. default : preset.
    # \param update_chunk_only : if set to false the mesh data (vertices/normals/triangles) is updated otherwise only the chunk data is updated. default : true.
    # \return True if the filtering was successful, false otherwise.
    #
    # \note The filtering is a costly operation, its not recommended to call it every time you retrieve a mesh but at the end of your spatial mapping process.
    def filter(self, params=MeshFilterParameters(), update_chunk_only=True):
        if isinstance(update_chunk_only, bool):
            return self.mesh.filter(deref((<MeshFilterParameters>params).meshFilter), update_chunk_only)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Applies texture to the mesh.
    # By using this function you will get access to \ref uv, and \ref texture.
    # The number of triangles in the mesh may slightly differ before and after calling this function due to missing texture information.
    # There is only one texture for the mesh, the uv of each chunk are expressed for it in its entirety.
    # Vectors of vertices/normals and uv have now the same size.
    # \param texture_format : defines the number of channels desired for the computed texture. default : [MESH_TEXTURE_FORMAT.RGB](\ref MESH_TEXTURE_FORMAT).
    #
    # \note This function can be called as long as you do not start a new spatial mapping process, due to shared memory.
    # \note This function can require a lot of computation time depending on the number of triangles in the mesh. Its recommended to call it once a the end of your spatial mapping process.
    # 
    # \warning The save_texture parameter in \ref SpatialMappingParameters must be set as true when enabling the spatial mapping to be able to apply the textures.
    # \warning The mesh should be filtered before calling this function since \ref filter will erase the textures, the texturing is also significantly slower on non-filtered meshes.
    def apply_texture(self, texture_format=MESH_TEXTURE_FORMAT.RGB):
        if isinstance(texture_format, MESH_TEXTURE_FORMAT):
            return self.mesh.applyTexture(<c_MESH_TEXTURE_FORMAT>(<unsigned int>texture_format.value))
        else:
            raise TypeError("Argument is not of MESH_TEXTURE_FORMAT type.")

    ##
    # Saves the current Mesh into a file.
    # \param filename : the path and filename of the mesh.
    # \param typeMesh : defines the file type (extension). default : [MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT).
    # \param id : specifies a set of chunks to be saved, if none provided all chunks are saved. default : (empty)
    # \return True if the file was successfully saved, false otherwise.
    # 
    # \note Only [MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT) supports textures data.
    # \note This function operates on the Mesh not on the chunks. This way you can save different parts of your Mesh (update your Mesh with \ref update_mesh_from_chunkList).
    def save(self, filename: str, typeMesh=MESH_FILE_FORMAT.OBJ, id=[]):
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.mesh.save(String(filename.encode()), <c_MESH_FILE_FORMAT>(<unsigned int>typeMesh.value), id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    ##
    # Loads the mesh from a file.
    # \param filename : the path and filename of the mesh (do not forget the extension).
    # \param update_mesh : if set to false the mesh data (vertices/normals/triangles) are updated otherwise only the chunk's data are updated. default : false.
    # \return True if the loading was successful, false otherwise.
    #
    # \note Updating the Mesh is time consuming, consider using only Chunks for better performances.
    def load(self, filename: str, update_mesh=False):
        if isinstance(update_mesh, bool):
            return self.mesh.load(String(filename.encode()), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Clears all the data.
    def clear(self):
        self.mesh.clear()

    ##
    # Vertices are defined by a 3D point (numpy array)
    @property
    def vertices(self):
        cdef np.ndarray arr = np.zeros((self.mesh.vertices.size(), 3), dtype=np.float32)
        for i in range(self.mesh.vertices.size()):
            for j in range(3):
                arr[i,j] = self.mesh.vertices[i].ptr()[j]
        return arr

    ##
    # List of triangles, defined as a set of three vertices. The information is stored in a numpy array
    @property
    def triangles(self):
        cdef np.ndarray arr = np.zeros((self.mesh.triangles.size(), 3))
        for i in range(self.mesh.triangles.size()):
            for j in range(3):
                arr[i,j] = self.mesh.triangles[i].ptr()[j]
        return arr

    ##
    # Normals are defined by three components, {nx, ny, nz}. Normals are defined for each vertex. (numpy array)
    @property
    def normals(self):
        cdef np.ndarray arr = np.zeros((self.mesh.normals.size(), 3), dtype=np.float32)
        for i in range(self.mesh.normals.size()):
            for j in range(3):
                arr[i,j] = self.mesh.normals[i].ptr()[j]
        return arr

    ##
    # Colors are defined by three components, {b, g, r}. Colors are defined for each vertex.
    @property
    def colors(self):
        cdef np.ndarray arr = np.zeros((self.mesh.colors.size(), 3), dtype=np.ubyte)
        for i in range(self.mesh.colors.size()):
            for j in range(3):
                arr[i,j] = self.mesh.colors[i].ptr()[j]
        return arr

    ##
    # UVs define the 2D projection of each vertex onto the texture . (numpy array)
    # Values are normalized [0;1], starting from the bottom left corner of the texture (as requested by opengl).
    # In order to display a textured mesh you need to bind the Texture and then draw each triangle by picking its uv values.
    #
    # \note Contains data only if your mesh has textures (by loading it or calling \ref apply_texture
    @property
    def uv(self):
        cdef np.ndarray arr = np.zeros((self.mesh.uv.size(), 2), dtype=np.float32)
        for i in range(self.mesh.uv.size()):
            for j in range(2):
                arr[i,j] = self.mesh.uv[i].ptr()[j]
        return arr

    ##
    # Texture of the \ref Mesh
    # \return a \ref Mat containing the texture of the \ref Mesh
    # \note Contains data only if your mesh has textures (by loading it or calling \ref apply_texture).
    @property
    def texture(self):
        py_texture = Mat()
        py_texture.mat = self.mesh.texture
        return py_texture

    ##
    # Computes the total number of triangles stored in all chunks.
    # \return The number of triangles stored in all chunks.
    def get_number_of_triangles(self):
        return self.mesh.getNumberOfTriangles()

    ##
    # Computes the indices of boundary vertices. 
    # \return The indices of boundary vertices. 
    def get_boundaries(self):
        cdef np.ndarray arr = np.zeros(self.mesh.getBoundaries().size(), dtype=np.uint32)
        for i in range(self.mesh.getBoundaries().size()):
            arr[i] = self.mesh.getBoundaries()[i]
        return arr

    ##
    # Merges current chunks.
    # This can be used to merge chunks into bigger sets to improve rendering process.
    # \param faces_per_chunk : defines the new number of faces per chunk (useful for Unity that doesn't handle chunks over 65K vertices).
    # 
    # \note You should not use this function during spatial mapping process because mesh updates will revert this changes.
    def merge_chunks(self, faces_per_chunk: int):
        self.mesh.mergeChunks(faces_per_chunk)

    ##
    # Estimates the gravity vector.
    # This function looks for a dominant plane in the whole mesh considering that it is the floor (or a horizontal plane). This can be used to find the gravity and then create realistic physical interactions.
    # \return The gravity vector. (numpy array)
    def get_gravity_estimate(self):
        gravity = self.mesh.getGravityEstimate()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = gravity[i]
        return arr

    ##
    # Computes the list of visible chunks from a specific point of view.
    # \param camera_pose : the point of view, given in world reference.
    # \return The list of visible chunks.
    def get_visible_list(self, camera_pose: Transform):
        return self.mesh.getVisibleList(camera_pose.transform[0])

    ##
    # Computes the list of chunks which are close to a specific point of view.
    # \param camera_pose : the point of view, given in world reference.
    # \param radius : the radius in defined \ref UNIT
    # \return The list of chunks close to the given point.
    def get_surrounding_list(self, camera_pose: Transform, radius: float):
        return self.mesh.getSurroundingList(camera_pose.transform[0], radius)

    ##
    # Updates \ref vertices / \ref normals / \ref triangles \ref uv from chunk data pointed by given chunkList.
    # \param id : the indices of chunks which will be concatenated. Default : (empty).
    # \note If the given chunkList is empty, all chunks will be used to update the current \ref Mesh
    def update_mesh_from_chunklist(self, id=[]):
        self.mesh.updateMeshFromChunkList(id)

##
# A plane defined by a point and a normal, or a plane equation. Other elements can be extracted such as the mesh, the 3D bounds...
# \ingroup SpatialMapping_group
# \note The plane measurements are expressed in REFERENCE_FRAME defined by \ref RuntimeParameters.measure3D_reference_frame .
cdef class Plane:
    cdef c_Plane plane
    def __cinit__(self):
        self.plane = c_Plane()

    ##
    # The plane type defines the plane orientation : vertical or horizontal.
    # \warning It is deduced from the gravity vector and is therefore only available with the ZED-M. The ZED will give UNKNOWN for every plane.
    @property
    def type(self):
        return PLANE_TYPE(<int>self.plane.type)

    @type.setter
    def type(self, type_):
        if isinstance(type_, PLANE_TYPE) :
            self.plane.type = <c_PLANE_TYPE>(<unsigned int>type_.value)
        else :
            raise TypeError("Argument is not of PLANE_TYPE type")

    ##
    # Gets the plane normal vector.
    # \return \ref Plane normal vector, with normalized components (numpy array)
    def get_normal(self):
        normal = self.plane.getNormal()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = normal[i]
        return arr

    ##
    # Gets the plane center point.
    # \return \ref Plane center point (numpy array)
    def get_center(self):
        center = self.plane.getCenter()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = center[i]
        return arr

    ##
    # Gets the plane pose relative to the global reference frame.
    # \param py_pose : a \ref Transform or it creates one by default.
    # \return A transformation matrix (rotation and translation) which gives the plane pose. Can be used to transform the global reference frame center (0,0,0) to the plane center.
    def get_pose(self, py_pose = Transform()):
        tmp =  self.plane.getPose()
        for i in range(16):
            (<Transform>py_pose).transform.m[i] = tmp.m[i]
        return py_pose

    ##
    # Gets the width and height of the bounding rectangle around the plane contours.
    # \return Width and height of the bounding plane contours (numpy array)
    # \warning This value is expressed in the plane reference frame.
    def get_extents(self):
        extents = self.plane.getExtents()
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = extents[i]
        return arr

    ##
    # Gets the plane equation.
    # \return \ref Plane equation, in the form : ax+by+cz=d, the returned values are (a,b,c,d) (numpy array)
    def get_plane_equation(self):
        plane_eq = self.plane.getPlaneEquation()
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = plane_eq[i]
        return arr

    ##
    # Gets the polygon bounds of the plane.
    # \return Vector of 3D points forming a polygon corresponding to the current visible limits of the plane (numpy array)
    def get_bounds(self):
        cdef np.ndarray arr = np.zeros((self.plane.getBounds().size(), 3))
        for i in range(self.plane.getBounds().size()):
            for j in range(3):
                arr[i,j] = self.plane.getBounds()[i].ptr()[j]
        return arr

    ##
    # Computes and returns the mesh of the bounds polygon.
    # \return A mesh representing the plane delimited by the visible bounds
    def extract_mesh(self):
        ext_mesh = self.plane.extractMesh()
        pymesh = Mesh()
        pymesh.mesh[0] = ext_mesh
        return pymesh

    ##
    # Gets the distance between the input point and the projected point alongside the normal vector onto the plane. This corresponds to the closest point on the plane.
    # \param point : The 3D point to project into the plane. Default: [0,0,0]
    # \return The Euclidian distance between the input point and the projected point
    def get_closest_distance(self, point=[0,0,0]):
        cdef Vector3[float] vec = Vector3[float](point[0], point[1], point[2])
        return self.plane.getClosestDistance(vec)

    ##
    # Clears all the data.
    def clear(self):
        self.plane.clear()

##
# Lists the spatial mapping resolution presets.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | HIGH | Creates a detailed geometry, requires lots of memory. |
# | MEDIUM | Small variations in the geometry will disappear, useful for big objects |
# | LOW | Keeps only huge variations of the geometry, useful for outdoor purposes. |
class MAPPING_RESOLUTION(enum.Enum):
    HIGH = <int>c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_HIGH
    MEDIUM  = <int>c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_MEDIUM
    LOW = <int>c_MAPPING_RESOLUTION.MAPPING_RESOLUTION_LOW

##
# Lists the spatial mapping depth range presets.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | SHORT | Only depth close to the camera will be used during spatial mapping. |
# | MEDIUM | Medium depth range.  |
# | LONG | Takes into account objects that are far, useful for outdoor purposes. |
# | AUTO | Depth range will be computed based on current \ref Camera states and parameters. |
class MAPPING_RANGE(enum.Enum):
    SHORT = <int>c_MAPPING_RANGE.SHORT
    MEDIUM = <int>c_MAPPING_RANGE.MAPPING_RANGE_MEDIUM
    LONG = <int>c_MAPPING_RANGE.LONG
    AUTO = <int>c_MAPPING_RANGE.AUTO

##
# Lists the types of spatial maps that can be created.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | MESH | Represents a surface with faces, 3D points are linked by edges, no color information. |
# | FUSED_POINT_CLOUD | Geometry is represented by a set of 3D colored points. |
class SPATIAL_MAP_TYPE(enum.Enum):
    MESH = <int>c_SPATIAL_MAP_TYPE.MESH
    FUSED_POINT_CLOUD = <int>c_SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

class BUS_TYPE(enum.Enum):
    USB = <int>c_BUS_TYPE.USB
    GMSL = <int>c_BUS_TYPE.GMSL
    AUTO = <int>c_BUS_TYPE.AUTO
    LAST = <int>c_BUS_TYPE.LAST

##
# Defines the input type used in the ZED SDK. Can be used to select a specific camera with ID or serial number, or a svo file.
# \ingroup Video_group
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

    ##
    # Set the input as the camera with specified id
    # \param id : The desired camera ID
    def set_from_camera_id(self, id: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO):
        self.input.setFromCameraID(id, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Set the input as the camera with specified serial number
    # \param serial_number : The desired camera serial_number
    def set_from_serial_number(self, serial_number: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO):
        self.input.setFromSerialNumber(serial_number, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Set the input as the svo specified with the filename
    # \param svo_input_filename : The path to the desired SVO file
    def set_from_svo_file(self, svo_input_filename: str):
        filename = svo_input_filename.encode()
        self.input.setFromSVOFile(String(<char*> filename))

    ##
    # Set the input to stream with the specified ip and port
    # \param sender_ip : The IP address of the streaming sender
    # \param port : The port on which to listen. Default: 30000
    def set_from_stream(self, sender_ip: str, port=30000):
        sender_ip_ = sender_ip.encode()
        self.input.setFromStream(String(<char*>sender_ip_), port)
 
    def get_type(self) -> INPUT_TYPE:
        return INPUT_TYPE(<int>self.input.getType())

    def get_configuration(self) -> str:
        return to_str(self.input.getConfiguration()).decode()

    def is_init(self) -> bool:
        return self.input.isInit()

##
# Holds the options used to initialize the \ref Camera object.
# \ingroup Video_group
# Once passed to the \ref Camera.open() function, these settings will be set for the entire execution life time of the \ref Camera.
# You can get further information in the detailed description bellow.
#
# This structure allows you to select multiple parameters for the \ref Camera such as the selected camera, its resolution, depth mode, coordinate system, and unit, of measurement.
# Once filled with the desired options, it should be passed to the \ref Camera.open() function.
# \code
#
#        import pyzed.sl as sl
#
#        def main() :
#            zed = sl.Camera() # Create a ZED camera object
#            init_params = sl.InitParameters() # Set initial parameters
#            init_params.sdk_verbose = 0  # Disable verbose mode
#            init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD1080 video mode
#            init_params.camera_fps = 30 # Set fps at 30
#            # Other parameters are left to their default values
#
#            # Open the camera
#            err = zed.open(init_params)
#            if err != sl.ERROR_CODE.SUCCESS :
#                exit(-1)
#
#            # Close the camera
#            zed.close()
#            return 0
#
#        if __name__ == "__main__" :
#            main()
#
# \endcode
#
# With its default values, it opens the ZED camera in live mode at \ref RESOLUTION.HD720 and sets the depth mode to \ref DEPTH_MODE.PERFORMANCE
# You can customize it to fit your application.
# The parameters can also be saved and reloaded using its \ref save() and \ref load() functions.
cdef class InitParameters:
    cdef c_InitParameters* init
    ##
    # Constructor.
    # \param camera_resolution : the chosen \ref camera_resolution
    # \param camera_fps : the chosen \ref camera_fps
    # \param svo_real_time_mode : activates \ref svo_real_time_mode
    # \param depth_mode : the chosen \ref depth_mode
    # \param coordinate_units : the chosen \ref coordinate_units
    # \param coordinate_system : the chosen \ref coordinate_system
    # \param sdk_verbose : sets \ref sdk_verbose
    # \param sdk_gpu_id : the chosen \ref sdk_gpu_id
    # \param depth_minimum_distance : the chosen \ref depth_minimum_distance
    # \param depth_maximum_distance : the chosen \ref depth_maximum_distance
    # \param camera_disable_self_calib : activates \ref camera_disable_self_calib
    # \param camera_image_flip : sets \ref camera_image_flip
    # \param enable_right_side_measure : activates \ref enable_right_side_measure
    # \param sdk_verbose_log_file : the chosen \ref sdk_verbose_log_file
    # \param depth_stabilization : activates \ref depth_stabilization
    # \param input_t : the chosen input_t (\ref InputType )
    # \param optional_settings_path : the chosen \ref optional_settings_path
    # \param sensors_required : activates \ref sensors_required
    # \param enable_image_enhancement : activates \ref enable_image_enhancement
    # \param optional_opencv_calibration_file : sets \ref optional_opencv_calibration_file
    # \param open_timeout_sec : sets \ref open_timeout_sec
    # \param async_grab_camera_recovery : sets \ref async_grab_camera_recovery
    # \param grab_compute_capping_fps : sets \ref grab_compute_capping_fps
    #
    # \code
    # params = sl.InitParameters(camera_resolution=RESOLUTION.HD720, camera_fps=30, depth_mode=DEPTH_MODE.PERFORMANCE)
    # \endcode
    def __cinit__(self, camera_resolution=RESOLUTION.HD720, camera_fps=0,
                  svo_real_time_mode=False,
                  depth_mode=DEPTH_MODE.PERFORMANCE,
                  coordinate_units=UNIT.MILLIMETER,
                  coordinate_system=COORDINATE_SYSTEM.IMAGE,
                  sdk_verbose=0, sdk_gpu_id=-1, depth_minimum_distance=-1.0, depth_maximum_distance=-1.0, camera_disable_self_calib=False,
                  camera_image_flip=FLIP_MODE.AUTO, enable_right_side_measure=False,
                  sdk_verbose_log_file="", depth_stabilization=1, input_t=InputType(),
                  optional_settings_path="",sensors_required=False,
                  enable_image_enhancement=True, optional_opencv_calibration_file="", 
                  open_timeout_sec=5.0, async_grab_camera_recovery=False, grab_compute_capping_fps=0):
        if (isinstance(camera_resolution, RESOLUTION) and isinstance(camera_fps, int) and
            isinstance(svo_real_time_mode, bool) and isinstance(depth_mode, DEPTH_MODE) and
            isinstance(coordinate_units, UNIT) and
            isinstance(coordinate_system, COORDINATE_SYSTEM) and isinstance(sdk_verbose, int) and
            isinstance(sdk_gpu_id, int) and isinstance(depth_minimum_distance, float) and
            isinstance(depth_maximum_distance, float) and
            isinstance(camera_disable_self_calib, bool) and isinstance(camera_image_flip, FLIP_MODE) and
            isinstance(enable_right_side_measure, bool) and
            isinstance(sdk_verbose_log_file, str) and isinstance(depth_stabilization, int) and
            isinstance(input_t, InputType) and isinstance(optional_settings_path, str) and
            isinstance(optional_opencv_calibration_file, str) and
            isinstance(open_timeout_sec, float) and
            isinstance(async_grab_camera_recovery, bool) and
            isinstance(grab_compute_capping_fps, float) or isinstance(grab_compute_capping_fps, int)) :

            filelog = sdk_verbose_log_file.encode()
            fileoption = optional_settings_path.encode()
            filecalibration = optional_opencv_calibration_file.encode()
            self.init = new c_InitParameters(<c_RESOLUTION>(<unsigned int>camera_resolution.value), camera_fps,
                                            svo_real_time_mode, <c_DEPTH_MODE>(<unsigned int>depth_mode.value),
                                            <c_UNIT>(<unsigned int>coordinate_units.value), <c_COORDINATE_SYSTEM>(<unsigned int>coordinate_system.value), sdk_verbose, sdk_gpu_id,
                                            <float>(depth_minimum_distance), <float>(depth_maximum_distance), camera_disable_self_calib
                                            , <c_FLIP_MODE>(<unsigned int>camera_image_flip.value),
                                            enable_right_side_measure,
                                            String(<char*> filelog), depth_stabilization,
                                            <CUcontext> 0, (<InputType>input_t).input, String(<char*> fileoption), sensors_required, enable_image_enhancement,
                                            String(<char*> filecalibration), <float>(open_timeout_sec), 
                                            async_grab_camera_recovery, <float>(grab_compute_capping_fps))
        else:
            raise TypeError("Argument is not of right type.")

    def __dealloc__(self):
        del self.init

    ##
    # This function saves the current set of parameters into a file to be reloaded with the \ref load() function.
    # \param filename : the path to the file in which the parameters will be stored
    # \return True if file was successfully saved, otherwise false.
    # 
    # \code
    #
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # init_params.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read
    # init_params.save("initParameters.conf") # Export the parameters into a file
    #
    # \endcode
    def save(self, filename: str):
        filename_save = filename.encode()
        return self.init.save(String(<char*> filename_save))

    ##
    # This function sets the other parameters from the values contained in a previously saved file.
    # \param filename : the path to the file from which the parameters will be loaded.
    # \return True if the file was successfully loaded, otherwise false.
    #
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.load("initParameters.conf") # Load the init_params from a previously exported file
    # \endcode
    def load(self, filename: str):
        filename_load = filename.encode()
        return self.init.load(String(<char*> filename_load))

    ##
    # Defines the chosen camera resolution. Small resolutions offer higher framerate and lower computation time.
    # In most situations, the \ref RESOLUTION.HD720 at 60 fps is the best balance between image quality and framerate.
    # Available resolutions are listed here: \ref RESOLUTION
    @property
    def camera_resolution(self):
        return RESOLUTION(<int>self.init.camera_resolution)

    @camera_resolution.setter
    def camera_resolution(self, value):
        if isinstance(value, RESOLUTION):
            self.init.camera_resolution = <c_RESOLUTION>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of RESOLUTION type.")

    ##
    # Requested camera frame rate. If set to 0, the highest FPS of the specified \ref InitParameters.camera_resolution will be used.
    # See \ref RESOLUTION for a list of supported framerates.
    # default 0
    # \note If the requested camera_fps is unsupported, the closest available FPS will be used.
    @property
    def camera_fps(self):
        return self.init.camera_fps

    @camera_fps.setter
    def camera_fps(self, int value):
        self.init.camera_fps = value

    ##
    # Force the motion sensors opening of the ZED 2 / ZED-M to open the camera.
    # default : false
    # If set to false, the SDK will try to <b>open and use</b> the IMU (second USB device on USB2.0) and will open the camera successfully even if the sensors failed to open.
    # This can be used for example when using a USB3.0 only extension cable (some fiber extension for example).
    # This parameter only impacts the LIVE mode.
    # If set to true, the camera will fail to open if the sensors cannot be opened. This parameter should be used when the IMU data must be available, such as Object Detection module or when the gravity is needed.
    @property
    def sensors_required(self):
        return self.init.sensors_required

    @sensors_required.setter
    def sensors_required(self, value: bool):
        self.init.sensors_required = value

    # Enable or Disable the Enhanced Contrast Technology, to improve image quality.
    # default : true.
    # If set to true, image enhancement will be activated in camera ISP. Otherwise, the image will not be enhanced by the IPS.
    # This only works for firmware version starting from 1523 and up.
    @property
    def enable_image_enhancement(self):
        return self.init.enable_image_enhancement

    @enable_image_enhancement.setter
    def enable_image_enhancement(self, value: bool):
        self.init.enable_image_enhancement = value

    ##
    # When playing back an SVO file, each call to \ref Camera.grab() will extract a new frame and use it.
    # However, this ignores the real capture rate of the images saved in the SVO file.
    # Enabling this parameter will bring the SDK closer to a real simulation when playing back a file by using the images' timestamps. However, calls to \ref Camera.grab() will return an error when trying to play too fast, and frames will be dropped when playing too slowly.
    @property
    def svo_real_time_mode(self):
        return self.init.svo_real_time_mode

    @svo_real_time_mode.setter
    def svo_real_time_mode(self, value: bool):
        self.init.svo_real_time_mode = value

    ##
    # The SDK offers several \ref DEPTH_MODE options offering various levels of performance and accuracy.
    # This parameter allows you to set the \ref DEPTH_MODE that best matches your needs.
    # default \ref DEPTH_MODE.PERFORMANCE
    @property
    def depth_mode(self):
        return DEPTH_MODE(<int>self.init.depth_mode)

    @depth_mode.setter
    def depth_mode(self, value):
        if isinstance(value, DEPTH_MODE):
            self.init.depth_mode = <c_DEPTH_MODE>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of DEPTH_MODE type.")

    ##
    # This parameter allows you to select the unit to be used for all metric values of the SDK. (depth, point cloud, tracking, mesh, and others).
    # default : \ref UNIT.MILLIMETER
    @property
    def coordinate_units(self):
        return UNIT(<int>self.init.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value):
        if isinstance(value, UNIT):
            self.init.coordinate_units = <c_UNIT>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of UNIT type.")

    ##
    # Positional tracking, point clouds and many other features require a given \ref COORDINATE_SYSTEM to be used as reference. This parameter allows you to select the \ref COORDINATE_SYSTEM use by the \ref Camera to return its measures.
    # default : \ref COORDINATE_SYSTEM.IMAGE
    @property
    def coordinate_system(self):
        return COORDINATE_SYSTEM(<int>self.init.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, COORDINATE_SYSTEM):
            self.init.coordinate_system = <c_COORDINATE_SYSTEM>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of COORDINATE_SYSTEM type.")

    ##
    # This parameter allows you to enable the verbosity of the SDK to get a variety of runtime information in the console. When developing an application, enabling verbose mode (sdk_verbose >= 1) can help you understand the current SDK behavior.
    # However, this might not be desirable in a shipped version.
    # default : 0 = no verbose message
    # \note The verbose messages can also be exported into a log file. See \ref sdk_verbose_log_file for more
    @property
    def sdk_verbose(self):
        return self.init.sdk_verbose

    @sdk_verbose.setter
    def sdk_verbose(self, value: int):
        self.init.sdk_verbose = value

    ##
    # By default the SDK will use the most powerful NVIDIA graphics card found. However, when running several applications, or using several cameras at the same time, splitting the load over available GPUs can be useful. This parameter allows you to select the GPU used by the \ref Camera using an ID from 0 to n-1 GPUs in your PC.
    # default : -1
    #
    # \note A non-positive value will search for all CUDA capable devices and select the most powerful.
    @property
    def sdk_gpu_id(self):
        return self.init.sdk_gpu_id

    @sdk_gpu_id.setter
    def sdk_gpu_id(self, value: int):
        self.init.sdk_gpu_id = value

    ##
    # This parameter allows you to specify the minimum depth value (from the camera) that will be computed, measured in the \ref UNIT you define.
    # In stereovision (the depth technology used by the camera), looking for closer depth values can have a slight impact on performance. However, this difference is almost invisible on modern GPUs.
    # In cases of limited computational power, increasing this value can provide better performance.
    # default : (-1) corresponding to 700 mm for a ZED/ZED2 and 200 mm for ZED Mini.
    # \note With a ZED camera you can decrease this value to 300 mm whereas you can set it to 100 mm using a ZED Mini and 200 mm for a ZED2. In any case this value cannot be greater than 3 meters.
    # Specific value (0): This will set the depth minimum distance to the minimum authorized value :
    #     - 300mm for ZED
    #     - 100mm for ZED-M
    #     - 200mm for ZED2
    @property
    def depth_minimum_distance(self):
        return  self.init.depth_minimum_distance

    @depth_minimum_distance.setter
    def depth_minimum_distance(self, value: float):
        self.init.depth_minimum_distance = value

    ##
    # Defines the current maximum distance that can be computed in the defined \ref UNIT.
    # When estimating the depth, the SDK uses this upper limit to turn higher values into TOO_FAR ones (unavailable depth values).
    # \note Changing this value has no impact on performance and doesn't affect the positional tracking nor the spatial mapping. (Only the depth, point cloud, normals)
    @property
    def depth_maximum_distance(self):
        return self.init.depth_maximum_distance

    @depth_maximum_distance.setter
    def depth_maximum_distance(self, value: float):
        self.init.depth_maximum_distance = value

    ##
    # At initialization, the \ref Camera runs a self-calibration process that corrects small offsets from the device's factory calibration.
    # A drawback is that calibration parameters will sligtly change from one (live) run to another, which can be an issue for repeatability.
    # If set to true, self-calibration will be disabled and calibration parameters won't be optimized (using the parameters of the conf file).
    # default : false
    # \note In most situations, self calibration should remain enabled.
    @property
    def camera_disable_self_calib(self):
        return self.init.camera_disable_self_calib

    @camera_disable_self_calib.setter
    def camera_disable_self_calib(self, value: bool):
        self.init.camera_disable_self_calib = value

    ##
    # If you are using the camera upside down, setting this parameter to FLIP_MODE.ON will cancel its rotation. The images will be horizontally flipped.
    # default : FLIP_MODE.AUTO
    @property
    def camera_image_flip(self):
        return FLIP_MODE(self.init.camera_image_flip)

    @camera_image_flip.setter
    def camera_image_flip(self, value):
        if isinstance(value, FLIP_MODE):
            self.init.camera_image_flip = <c_FLIP_MODE>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of FLIP_MODE type.")

    ##
    # By default, the SDK only computes a single depth map, aligned with the left camera image.
    # This parameter allows you to enable the \ref DEPTH.DEPTH_RIGHT and other <XXX>.RIGHT at the cost of additional computational time.
    # For example, mixed reality passthrough applications require one depth map per eye, so this parameter can be activated.
    # default : false
    @property
    def enable_right_side_measure(self):
        return self.init.enable_right_side_measure

    @enable_right_side_measure.setter
    def enable_right_side_measure(self, value: bool):
        self.init.enable_right_side_measure = value

    ##
    # When \ref sdk_verbose is enabled, this parameter allows you to redirect both the SDK verbose messages and your own application messages to a file.
    # default : (empty) Should contain the path to the file to be written. A file will be created if missing.
    # \note Setting this parameter to any value will redirect all standard output print calls of the entire program. This means that your own standard output print calls will be redirected to the log file.
    # \warning The log file won't be cleared after successive executions of the application. This means that it can grow indefinitely if not cleared. 
    @property
    def sdk_verbose_log_file(self):
        if not self.init.sdk_verbose_log_file.empty():
            return self.init.sdk_verbose_log_file.get().decode()
        else:
            return ""

    @sdk_verbose_log_file.setter
    def sdk_verbose_log_file(self, value: str):
        value_filename = value.encode()
        self.init.sdk_verbose_log_file.set(<char*>value_filename)

    ##
	# Regions of the generated depth map can oscillate from one frame to another. These oscillations result from a lack of texture (too homogeneous) on an object and by image noise.
	# This parameter control a stabilization filter that reduces these oscillations. In the range [0-100], 0 is disable (raw depth), smoothness is linear from 1 to 100.
    # \note The stabilization uses the positional tracking to increase its accuracy, so the Positional Tracking module will be enabled automatically when set to a value different from 0
    @property
    def depth_stabilization(self):
        return self.init.depth_stabilization

    @depth_stabilization.setter
    def depth_stabilization(self, value: int):
        self.init.depth_stabilization = value

    ##
    # The SDK can handle different input types:
    #   - Select a camera by its ID (/dev/video<i>X</i> on Linux, and 0 to N cameras connected on Windows)
    #   - Select a camera by its serial number
    #   - Open a recorded sequence in the SVO file format
    #   - Open a streaming camera from its IP address and port
    #
    # This parameter allows you to select to desired input. It should be used like this:
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # input_t = sl.InputType()
    # input_t.set_from_camera_id(0) # Selects the camera with ID = 0
    # init_params.input = input_t
    # init_params.set_from_camera_id(0) # You can also use this
    # \endcode
    #
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # input_t = sl.InputType()
    # input_t.set_from_serial_number(1010) # Selects the camera with serial number = 101
    # init_params.input = input_t
    # init_params.set_from_serial_number(1010) # You can also use this
    # \endcode
    #
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # input_t = sl.InputType()
    # input_t.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read
    # init_params.input = input_t
    # init_params.set_from_svo_file("/path/to/file.svo")  # You can also use this
    # \endcode
    # 
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # input_t = sl.InputType()
    # input_t.set_from_stream("192.168.1.42")
    # init_params.input = input_t
    # init_params.set_from_stream("192.168.1.42") # You can also use this
    # \endcode
    #
    # Available cameras and their ID/serial can be listed using \ref get_device_list() and \ref get_streaming_device_list()
    # Each \ref Camera will create its own memory (CPU and GPU), therefore the number of ZED used at the same time can be limited by the configuration of your computer. (GPU/CPU memory and capabilities)
    #
    # default : empty
    # See \ref InputType for complementary information.
    # 
    # \warning Using the ZED SDK Python API, using init_params.input.set_from_XXX won't work, use init_params.set_from_XXX instead
    # @property
    # def input(self):
    #    input_t = InputType()
    #    input_t.input = self.init.input
    #    return input_t

    # @input.setter
    def input(self, input_t: InputType):
        self.init.input = input_t.input

    input = property(None, input)

    ##
    # Set the optional path where the SDK has to search for the settings file (SN<XXXX>.conf file). This file contains the calibration information of the camera.
    #
    # default : (empty). The SNXXX.conf file will be searched in the default directory (/usr/local/zed/settings/ for Linux or C:/ProgramData/stereolabs/settings for Windows)
    #
    # \note if a path is specified and no file has been found, the SDK will search on the default path (see default) for the *.conf file.
    #
    # Automatic download of conf file (through ZED Explorer or the installer) will still download the file in the default path. If you want to use another path by using this entry, make sure to copy the file in the proper location.
    #
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # home = "/path/to/home"
    # path= home+"/Documents/settings/" # assuming /path/to/home/Documents/settings/SNXXXX.conf exists. Otherwise, it will be searched in /usr/local/zed/settings/
    # init_params.optional_settings_path = path
    # \endcode
    @property
    def optional_settings_path(self):
        if not self.init.optional_settings_path.empty():
            return self.init.optional_settings_path.get().decode()
        else:
            return ""

    @optional_settings_path.setter
    def optional_settings_path(self, value: str):
        value_filename = value.encode()
        self.init.optional_settings_path.set(<char*>value_filename)

    ##
    # Set an optional file path where the SDK can find a file containing the calibration information of the camera computed by OpenCV.
    # \note Using this will disable the factory calibration of the camera.
    # \warning Erroneous calibration values can lead to poor SDK modules accuracy.
    @property
    def optional_opencv_calibration_file(self):
        if not self.init.optional_opencv_calibration_file.empty():
            return self.init.optional_opencv_calibration_file.get().decode()
        else:
            return ""

    @optional_opencv_calibration_file.setter
    def optional_opencv_calibration_file(self, value: str):
        value_filename = value.encode()
        self.init.optional_opencv_calibration_file.set(<char*>value_filename)

    ##
    # Defines a timeout in seconds after which an error is reported if the \ref sl.Camera.open() command fails.
    # Set to '-1' to try to open the camera endlessly without returning error in case of failure.
    # Set to '0' to return error in case of failure at the first attempt.
    # Default : 5.0f
    # \note This parameter only impacts the LIVE mode.
    @property
    def open_timeout_sec(self):
        return self.init.open_timeout_sec

    @open_timeout_sec.setter
    def open_timeout_sec(self, value: float):
        self.init.open_timeout_sec = value

    ##
    # Define the behavior of the automatic camera recovery during grab() function call. When async is enabled and there's an issue with the communication with the camera
    #  the grab() will exit after a short period and return the ERROR_CODE::CAMERA_REBOOTING warning. The recovery will run in the background until the correct communication is restored.
    #  When async_grab_camera_recovery is false, the grab() function is blocking and will return only once the camera communication is restored or the timeout is reached. 
    # The default behavior is synchronous, like previous ZED SDK versions
    @property
    def async_grab_camera_recovery(self):
        return self.init.async_grab_camera_recovery

    @async_grab_camera_recovery.setter
    def async_grab_camera_recovery(self, value: bool):
        self.init.async_grab_camera_recovery = value

    ##
    # Define a computation upper limit to the grab frequency. 
    # This can be useful to get a known constant fixed rate or limit the computation load while keeping a short exposure time by setting a high camera capture framerate.
    # \n The value should be inferior to the InitParameters::camera_fps and strictly positive. It has no effect when reading an SVO file.
    # \n This is an upper limit and won't make a difference if the computation is slower than the desired compute capping fps.
    # \note Internally the grab function always tries to get the latest available image while respecting the desired fps as much as possible.
    @property
    def grab_compute_capping_fps(self):
        return self.init.grab_compute_capping_fps

    @grab_compute_capping_fps.setter
    def grab_compute_capping_fps(self, value: float):
        self.init.grab_compute_capping_fps = value

    ##
    # Call of \ref InputType.set_from_camera_id function of \ref input
    # \param id : The desired camera ID
    def set_from_camera_id(self, id: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO):
        self.init.input.setFromCameraID(id, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Call of \ref InputType.set_from_serial_number function of \ref input
    # \param serial_number : The desired camera serial_number
    def set_from_serial_number(self, serial_number: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO):
        self.init.input.setFromSerialNumber(serial_number, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Call of \ref InputType.set_from_svo_file function of \ref input
    # \param svo_input_filename : The path to the desired SVO file
    def set_from_svo_file(self, svo_input_filename: str):
        filename = svo_input_filename.encode()
        self.init.input.setFromSVOFile(String(<char*> filename))

    ##
    # Call of \ref InputType.set_from_stream function of \ref input
    # \param sender_ip : The IP address of the streaming sender
    # \param port : The port on which to listen. Default: 30000
    def set_from_stream(self, sender_ip: str, port=30000):
        sender_ip_ = sender_ip.encode()
        self.init.input.setFromStream(String(<char*>sender_ip_), port)

##
# Parameters that define the behavior of the \ref Camera.grab.
# \ingroup Depth_group
# Default values are enabled.
# You can customize it to fit your application and then save it to create a preset that can be loaded for further executions.
cdef class RuntimeParameters:
    cdef c_RuntimeParameters* runtime
    ##
    # Constructor.
    # \param enable_depth : activates \ref enable_depth
    # \param confidence_threshold : chosen \ref confidence_threshold
    # \param texture_confidence_threshold : chosen \ref texture_confidence_threshold
    # \param measure3D_reference_frame : chosen \ref measure3D_reference_frame
    #
    # \code
    # params = sl.RuntimeParameters(enable_depth=True)
    # \endcode
    def __cinit__(self, enable_depth=True, enable_fill_mode=False,
                  confidence_threshold = 100, texture_confidence_threshold = 100,
                  measure3D_reference_frame=REFERENCE_FRAME.CAMERA, remove_saturated_areas = True):
        if (isinstance(enable_depth, bool)
            and isinstance(enable_fill_mode, bool)
            and isinstance(confidence_threshold, int) and
            isinstance(measure3D_reference_frame, REFERENCE_FRAME)
            and isinstance(remove_saturated_areas, bool)):

            self.runtime = new c_RuntimeParameters(enable_depth, enable_fill_mode, confidence_threshold, texture_confidence_threshold,
                                                 <c_REFERENCE_FRAME>(<unsigned int>measure3D_reference_frame.value),remove_saturated_areas)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.runtime

    ##
    # Saves the current set of parameters into a file.
    # \param filename : the path to the file in which the parameters will be stored.
    # \return true if the file was successfully saved, otherwise false.
    def save(self, filename: str):
        filename_save = filename.encode()
        return self.runtime.save(String(<char*> filename_save))

    ##
    # Loads the values of the parameters contained in a file.
    # \param filename : the path to the file from which the parameters will be loaded.
    # \return true if the file was successfully loaded, otherwise false.
    def load(self, filename: str):
        filename_load = filename.encode()
        return self.runtime.load(String(<char*> filename_load))

    ##
    # Defines if the depth map should be computed.
    # If false, only the images are available.
    # default : True
    @property
    def enable_depth(self):
        return self.runtime.enable_depth

    @enable_depth.setter
    def enable_depth(self, value: bool):
        self.runtime.enable_depth = value

    ##
    # Defines if the depth map should be completed or not, similar to the removed SENSING_MODE::FILL
    # Enabling this will override the confidence values confidence_threshold and texture_confidence_threshold as well as remove_saturated_areas
    @property
    def enable_fill_mode(self):
        return self.runtime.enable_fill_mode

    @enable_fill_mode.setter
    def enable_fill_mode(self, value: bool):
        self.runtime.enable_fill_mode = value

    ##
    # Provides 3D measures (point cloud and normals) in the desired reference frame.
    # default : [REFERENCE_FRAME.CAMERA](\ref REFERENCE_FRAME)
    @property
    def measure3D_reference_frame(self):
        return REFERENCE_FRAME(<int>self.runtime.measure3D_reference_frame)

    @measure3D_reference_frame.setter
    def measure3D_reference_frame(self, value):
        if isinstance(value, REFERENCE_FRAME):
            self.runtime.measure3D_reference_frame = <c_REFERENCE_FRAME>(<unsigned int>value.value)
        else:
            raise TypeError("Argument must be of REFERENCE type.")

    ##
    # Threshold to reject depth values based on their confidence.
    # 
    # Each depth pixel has a corresponding confidence. (\ref MEASURE.CONFIDENCE)
    # A lower value means more confidence and precision (but less density). An upper value reduces filtering (more density, less certainty).
    # \n - \b setConfidenceThreshold(100) will allow values from \b 0 to \b 100. (no filtering)
    # \n - \b setConfidenceThreshold(90) will allow values from \b 10 to \b 100. (filtering lowest confidence values)
    # \n - \b setConfidenceThreshold(30) will allow values from \b 70 to \b 100. (keeping highest confidence values and lowering the density of the depth map)
    # The value should be in [1,100].
    # \n By default, the confidence threshold is set at 100, meaning that no depth pixel will be rejected.
    @property
    def confidence_threshold(self):
        return self.runtime.confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value):
        self.runtime.confidence_threshold = value

    ##
    # Threshold to reject depth values based on their texture confidence.
    # A lower value means more confidence and precision (but less density). An upper value reduces filtering (more density, less certainty). 
    # The value should be in [1,100]. By default, the confidence threshold is set at 100, meaning that no depth pixel will be rejected.
    @property
    def texture_confidence_threshold(self):
        return self.runtime.texture_confidence_threshold

    @texture_confidence_threshold.setter
    def texture_confidence_threshold(self, value):
        self.runtime.texture_confidence_threshold = value

    ##
    # Defines if the saturated area (Luminance>=255) must be removed from depth map estimationd.
    # default : True
    @property
    def remove_saturated_areas(self):
        return self.runtime.remove_saturated_areas

    @remove_saturated_areas.setter
    def remove_saturated_areas(self, value: bool):
        self.runtime.remove_saturated_areas = value
##
# Parameters for positional tracking initialization.
# \ingroup PositionalTracking_group
# A default constructor is enabled and set to its default parameters.
# You can customize it to fit your application and then save it to create a preset that can be loaded for further executions.
# \note Parameters can be user adjusted.
cdef class PositionalTrackingParameters:
    cdef c_PositionalTrackingParameters* tracking
    ##
    # Constructor.
    # \param _init_pos : chosen initial camera position in the world frame (\ref Transform)
    # \param _enable_memory : activates \ref enable_memory
    # \param _enable_pose_smoothing : activates \ref enable_pose_smoothing
    # \param _area_path : chosen \ref area_path
    # \param _set_floor_as_origin : activates \ref set_floor_as_origin
    # \param _enable_imu_fusion : activates \ref enable_imu_fusion
    # \param _set_as_static : activates \ref set_as_static
    # \param _depth_min_range : activates \ref depth_min_range
    # \param _set_gravity_as_origin : This setting allows you to set the odometry world using sensors data.
    # \param _mode : Positional tracking mode used. Can be used to improve accuracy in some type of scene at the cost of longer runtime
    # \code
    # params = sl.PositionalTrackingParameters(init_pos=Transform(), _enable_pose_smoothing=True)
    # \endcode
    def __cinit__(self, _init_pos=Transform(), _enable_memory=True, _enable_pose_smoothing=False, _area_path=None,
                  _set_floor_as_origin=False, _enable_imu_fusion=True, _set_as_static=False, _depth_min_range=-1, _set_gravity_as_origin=True, _mode=POSITIONAL_TRACKING_MODE.STANDARD):
        if _area_path is None:
            self.tracking = new c_PositionalTrackingParameters((<Transform>_init_pos).transform[0], _enable_memory, _enable_pose_smoothing, String(), _set_floor_as_origin, _enable_imu_fusion, _set_as_static, _depth_min_range, _set_gravity_as_origin, <c_POSITIONAL_TRACKING_MODE>(<unsigned int>_mode.value))
        else :
            area_path = _area_path.encode()
            self.tracking = new c_PositionalTrackingParameters((<Transform>_init_pos).transform[0], _enable_memory, _enable_pose_smoothing, String(<char*> area_path), _set_floor_as_origin, _enable_imu_fusion, _set_as_static, _depth_min_range, _set_gravity_as_origin, <c_POSITIONAL_TRACKING_MODE>(<unsigned int>_mode.value))
    
    def __dealloc__(self):
        del self.tracking

    ##
    # Saves the current set of parameters into a file.
    # \param filename : the path to the file in which the parameters will be stored.
    # \return true if the file was successfully saved, otherwise false.
    def save(self, filename: str):
        filename_save = filename.encode()
        return self.tracking.save(String(<char*> filename_save))

    ##
    # Loads the values of the parameters contained in a file.
    # \param filename : the path to the file from which the parameters will be loaded.
    # \return true if the file was successfully loaded, otherwise false.
    def load(self, filename: str):
        filename_load = filename.encode()
        return self.tracking.load(String(<char*> filename_load))

    ##
    # Gets the position of the camera in the world frame when camera is started. By default it should be identity.
    # \param init_pos : \ref Transform to be returned, by default it creates one
    # \return Position of the camera in the world frame when camera is started.
    # \note The camera frame (defines the reference frame for the camera) is by default positioned at the world frame when tracking is started.
    def initial_world_transform(self, init_pos = Transform()):
        for i in range(16):
            (<Transform>init_pos).transform.m[i] = self.tracking.initial_world_transform.m[i]
        return init_pos

    ##
    # Set the position of the camera in the world frame when camera is started.
    # \param value : \ref Transform input
    def set_initial_world_transform(self, value: Transform):
        for i in range(16):
            self.tracking.initial_world_transform.m[i] = value.transform.m[i]

    ##
    # This mode enables the camera to learn and remember its surroundings. This helps correct positional tracking drift and position different cameras relative to each other in space.
    # default : true
    #
    # \warning This mode requires few resources to run and greatly improves tracking accuracy. We recommend to leave it on by default.
    @property
    def enable_area_memory(self):
        return self.tracking.enable_area_memory

    @enable_area_memory.setter
    def enable_area_memory(self, value: bool):
        self.tracking.enable_area_memory = value

    ##
    # This mode enables smooth pose correction for small drift correction.
    # default : false
    @property
    def enable_pose_smoothing(self):
        return self.tracking.enable_pose_smoothing

    @enable_pose_smoothing.setter
    def enable_pose_smoothing(self, value: bool):
        self.tracking.enable_pose_smoothing = value

    ##
    # This mode initializes the tracking aligned with the floor plane to better position the camera in space
    # \note The floor plane detection is launched in the background until it is found. The tracking is in SEARCHING state.
    # \warning This feature works best with the ZED-M since it needs an IMU to classify the floor. The ZED needs to look at the floor during the initialization for optimum results.
    @property
    def set_floor_as_origin(self):
        return self.tracking.set_floor_as_origin

    @set_floor_as_origin.setter
    def set_floor_as_origin(self, value: bool):
        self.tracking.set_floor_as_origin = value

    ##
    # This setting allows you to enable or disable the IMU fusion. When set to false, only the optical odometry will be used.
    # default : true
    # \note This setting has no impact on the tracking of a ZED camera, only the ZED Mini uses a built-in IMU.
    @property
    def enable_imu_fusion(self):
        return self.tracking.enable_imu_fusion

    @enable_imu_fusion.setter
    def enable_imu_fusion(self, value: bool):
        self.tracking.enable_imu_fusion = value

    ##
    # Area localization file that describes the surroundings (previously saved).
    # default : (empty)
    # \note Loading an area file will start a searching phase during which the camera will try to position itself in the previously learned area
    # \warning The area file describes a specific location. If you are using an area file describing a different location, the tracking function will continuously search for a position and may not find a correct one.
    # \warning The '.area' file can only be used with the same depth mode (\ref MODE) as the one used during area recording.
    @property
    def area_file_path(self):
        if not self.tracking.area_file_path.empty():
            return self.tracking.area_file_path.get().decode()
        else:
            return ""

    @area_file_path.setter
    def area_file_path(self, value: str):
        value_area = value.encode()
        self.tracking.area_file_path.set(<char*>value_area)

    ##
    # This setting allows you define the camera as static. If true, it will not move in the environment. This allows you to set its position using the initial world transform.
    # All SDK functionalities requiring positional tracking will be enabled.
    # \ref Camera.get_position() will return the value set as initial world transform for the PATH, and identify as the POSE.
    @property
    def set_as_static(self):
        return self.tracking.set_as_static

    @set_as_static.setter
    def set_as_static(self, value: bool):
        self.tracking.set_as_static = value
    
    ##
    # This setting allows you to change the minimum depth used by the SDK for Positional Tracking.
    # It may be useful for example if any steady objects are in front of the camera and may perturbate the positional tracking algorithm.
    # default : -1, no minimum depth
    @property
    def depth_min_range(self):
        return self.tracking.depth_min_range

    @depth_min_range.setter
    def depth_min_range(self, value):
        self.tracking.depth_min_range = value

    ##
    # This setting allows you to override 2 of the 3 rotations from initial_world_transform using the IMU gravity
    @property
    def set_gravity_as_origin(self):
        return self.tracking.set_gravity_as_origin

    @set_gravity_as_origin.setter
    def set_gravity_as_origin(self, value: bool):
        self.tracking.set_gravity_as_origin = value

##
# List of possible camera states.
# \ingroup Video_group
#
# | Enumerator |                 |
# |------------|-----------------|
# | H264 | AVCHD/H264 encoding used in image streaming. |
# | H265 | HEVC/H265 encoding used in image streaming. |
class STREAMING_CODEC(enum.Enum):
    H264 = <int>c_STREAMING_CODEC.STREAMING_CODEC_H264
    H265 = <int>c_STREAMING_CODEC.STREAMING_CODEC_H265
    LAST = <int>c_STREAMING_CODEC.STREAMING_CODEC_LAST

##
# Properties of all streaming devices
# \ingroup Video_group
cdef class StreamingProperties:
    cdef c_StreamingProperties c_streaming_properties

    ##
    # the streaming IP of the device
    @property
    def ip(self):
        return to_str(self.c_streaming_properties.ip).decode()

    @ip.setter
    def ip(self, str ip_):
        self.c_streaming_properties.ip = String(ip_.encode())

    ##
    # the streaming port
    @property
    def port(self):
        return self.c_streaming_properties.port

    @port.setter
    def port(self, port_):
         self.c_streaming_properties.port = port_

    ##
    # the serial number of the streaming device
    @property
    def serial_number(self):
        return self.c_streaming_properties.serial_number

    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_streaming_properties.serial_number=serial_number

    ##
    # the current bitrate of encoding of the streaming device
    @property
    def current_bitrate(self):
        return self.c_streaming_properties.current_bitrate

    @current_bitrate.setter
    def current_bitrate(self, current_bitrate):
        self.c_streaming_properties.current_bitrate=current_bitrate

    ##
    # the current codec used for compression in streaming device
    @property
    def codec(self):
        return STREAMING_CODEC(<int>self.c_streaming_properties.codec)

    @codec.setter
    def codec(self, codec):
        self.c_streaming_properties.codec = <c_STREAMING_CODEC>(<unsigned int>codec.value)


##
# Sets the streaming parameters.
# \ingroup Video_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be user adjusted.
cdef class StreamingParameters:
    cdef c_StreamingParameters* streaming
    ##
    # Constructor.
    # \param codec : the chosen \ref codec
    # \param port : the chosen \ref port
    # \param bitrate : the chosen \ref bitrate
    # \param gop_size : the chosen \ref gop_size
    # \param adaptative_bitrate : activtates \ref adaptative_bitrate
    # \param chunk_size : the chosen \ref chunk_size
    # \param target_framerate : the chosen \ref target_framerate
    #
    # \code
    # params = sl.StreamingParameters(port=30000)
    # \endcode
    def __cinit__(self, codec=STREAMING_CODEC.H265, port=30000, bitrate=8000, gop_size=-1, adaptative_bitrate=False, chunk_size=32768,target_framerate=0):
            self.streaming = new c_StreamingParameters(<c_STREAMING_CODEC>(<unsigned int>codec.value), port, bitrate, gop_size, adaptative_bitrate, chunk_size,target_framerate)

    def __dealloc__(self):
        del self.streaming

    ##
    # Defines a single chunk size
    # \note Stream buffers are divided in X number of chunks where each chunk is "chunk_size" bits long.
    # \note Default value is 32768. You can lower this value if network generates a lot of packet lost : this will generate more chunks for a single image, but each chunk sent will be lighter to avoid inside-chunk corruption.
    # \note Available range : [8192 - 65000]
    @property
    def chunk_size(self):
        return self.streaming.chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self.streaming.chunk_size = value

    ##
    # Defines the codec used for streaming.
    # \warning If HEVC is used, make sure the receiving host is compatible with H265 decoding (Pascal NVIDIA card or newer). If not, prefer to use H264 since every compatible NVIDIA card supports H264 decoding
    @property
    def codec(self):
        return STREAMING_CODEC(<int>self.streaming.codec)

    @codec.setter
    def codec(self, codec):
        self.streaming.codec = <c_STREAMING_CODEC>(<unsigned int>codec.value)

    ##
    # Defines the port the data will be streamed on.
    # \warning port must be an even number. Any odd number will be rejected.
    @property
    def port(self):
        return self.streaming.port

    @port.setter
    def port(self, value: ushort1):
        self.streaming.port = value

    ##
    # Defines the streaming bitrate in Kbits/s
    @property
    def bitrate(self):
        return self.streaming.bitrate

    @bitrate.setter
    def bitrate(self, value: uint):
        self.streaming.bitrate = value

    ##
    # Enable/Disable adaptive bitrate
    # \note Bitrate will be adjusted regarding the number of packet loss during streaming.
    # \note if activated, bitrate can vary between [bitrate/4, bitrate]
    # \warning Bitrate will be adjusted regarding the number of packet loss during streaming.
    @property
    def adaptative_bitrate(self):
        return self.streaming.adaptative_bitrate

    @adaptative_bitrate.setter
    def adaptative_bitrate(self, value: bool):
        self.streaming.adaptative_bitrate = value

    ##
    # Defines the gop size in frame unit.
    # \note if value is set to -1, the gop size will match 2 seconds, depending on camera fps.
    # \note The gop size determines the maximum distance between IDR/I-frames. Very high GOP sizes will result in slightly more efficient compression, especially on static scenes. But it can result in more latency if IDR/I-frame packet are lost during streaming.
    # \note Default value is -1. Maximum allowed value is 256 (frames).
    @property
    def gop_size(self):
        return self.streaming.gop_size

    @gop_size.setter
    def gop_size(self, value: int):
        self.streaming.gop_size = value
    
    ##
    # \brief defines the target framerate for the streaming output.
    # This framerate must be below or equal to the camera framerate. Allowed framerates are 15,30, 60 or 100 if possible.
    # Any other values will be discarded and camera FPS will be taken.
    # \note By default 0 means that the camera framerate will be taken
    @property
    def target_framerate(self):
        return self.streaming.target_framerate

    @target_framerate.setter
    def target_framerate(self, value: int):
        self.streaming.target_framerate = value


##
# Sets the recording parameters.
# \ingroup Video_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be user adjusted.
cdef class RecordingParameters:

    cdef c_RecordingParameters *record
    ##
    # Constructor.
    # \param video_filename : the chosen \ref video_filename
    # \param compression_mode : the chosen \ref compression_mode
    # \param target_framerate : the chosen \ref target_framerate
    # \param bitrate : the chosen \ref bitrate
    # \param transcode_streaming_input : enables \ref transcode_streaming_input
    #
    # \code
    # params = sl.RecordingParameters(video_filename="record.svo",compression_mode=SVO_COMPRESSION_MODE.H264)
    # \endcode
    def __cinit__(self, video_filename="myRecording.svo", compression_mode=SVO_COMPRESSION_MODE.H264, target_framerate=0,
                    bitrate=0, transcode_streaming_input=False):
        if (isinstance(compression_mode, SVO_COMPRESSION_MODE)) :
            video_filename_c = video_filename.encode()
            self.record = new c_RecordingParameters(String(<char*> video_filename_c), 
                                                    <c_SVO_COMPRESSION_MODE>(<unsigned int>compression_mode.value),
                                                    target_framerate, bitrate, transcode_streaming_input)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.record

    ##
    # filename of the SVO file.
    @property
    def video_filename(self):
        return to_str(self.record.video_filename).decode()

    @video_filename.setter
    def video_filename(self, video_filename):
        video_filename_c = video_filename.encode()
        self.record.video_filename = String(<char*> video_filename_c)

    ##
    # compression_mode : can be one of the \ref SVO_COMPRESSION_MODE enum
    @property
    def compression_mode(self):
        return SVO_COMPRESSION_MODE(<int>self.record.compression_mode)

    @compression_mode.setter
    def compression_mode(self, compression_mode):
        if isinstance(compression_mode, SVO_COMPRESSION_MODE) :
            self.record.compression_mode = <c_SVO_COMPRESSION_MODE>(<unsigned int>compression_mode.value)
        else :
            raise TypeError()

    ##
    # \brief defines the target framerate for the streaming output.
    # This framerate must be below or equal to the camera framerate. Allowed framerates are 15,30, 60 or 100 if possible.
    # Any other values will be discarded and camera FPS will be taken.
    # \note By default 0 means that the camera framerate will be taken
    @property
    def target_framerate(self):
        return self.record.target_framerate

    @target_framerate.setter
    def target_framerate(self, value: int):
        self.record.target_framerate = value

    ##
    # \brief overrides default bitrate of the SVO file, in KBits/s. Only works if \ref SVO_COMPRESSION_MODE is H264 or H265.
    # 0 means default values (depends on the resolution)
    # \note Available range : 0 or [1000 - 60000]
    @property
    def bitrate(self):
        return self.record.bitrate

    @bitrate.setter
    def bitrate(self, value: int):
        self.record.bitrate = value

    ##
    # \brief In case of streaming input, if set to false, it will avoid decoding/re-encoding and convert directly streaming input into a SVO file.
    # This saves an encoding session and can be especially useful on NVIDIA Geforce cards where the number of encoding session is limited.
    # \note compression_mode, target_framerate and bitrate will be ignored in this mode.
    @property
    def transcode_streaming_input(self):
        return self.record.transcode_streaming_input

    @transcode_streaming_input.setter
    def transcode_streaming_input(self, value):
        self.record.transcode_streaming_input = value

##
# Sets the spatial mapping parameters.
# \ingroup SpatialMapping_group
#
# Instantiating with the default constructor will set all parameters to their default values.
# You can customize these values to fit your application, and then save them to a preset to be loaded in future executions.
# \note Users can adjust these parameters as they see fit.
cdef class SpatialMappingParameters:
    cdef c_SpatialMappingParameters* spatial
    ##
    # Constructor.
    # \param resolution : the chosen \ref MAPPING_RESOLUTION
    # \param mapping_range : the chosen \ref MAPPING_RANGE
    # \param max_memory_usage : the chosen \ref max_memory_usage
    # \param save_texture : activates \ref save_texture
    # \param use_chunk_only : activates \ref use_chunk_only
    # \param reverse_vertex_order : activates \ref reverse_vertex_order
    # \param map_type : the chosen \ref map_type
    #
    # \code
    # params = sl.SpatialMappingParameters(resolution=MAPPING_RESOLUTION.HIGH)
    # \endcode
    def __cinit__(self, resolution=MAPPING_RESOLUTION.MEDIUM, mapping_range=MAPPING_RANGE.AUTO,
                  max_memory_usage=2048, save_texture=False, use_chunk_only=False,
                  reverse_vertex_order=False, map_type=SPATIAL_MAP_TYPE.MESH):
        if (isinstance(resolution, MAPPING_RESOLUTION) and isinstance(mapping_range, MAPPING_RANGE) and
            isinstance(use_chunk_only, bool) and isinstance(reverse_vertex_order, bool) and isinstance(map_type, SPATIAL_MAP_TYPE)):
            self.spatial = new c_SpatialMappingParameters(<c_MAPPING_RESOLUTION>(<unsigned int>resolution.value),
                                                          <c_MAPPING_RANGE>(<unsigned int>mapping_range.value),
                                                          max_memory_usage, save_texture,
                                                          use_chunk_only, reverse_vertex_order,
                                                          <c_SPATIAL_MAP_TYPE>(<unsigned int>map_type.value))
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.spatial

    ##
    # Sets the resolution corresponding to the given \ref MAPPING_RESOLUTION preset.
    # \param resolution : the desired \ref MAPPING_RESOLUTION. default : [MAPPING_RESOLUTION.HIGH](\ref MAPPING_RESOLUTION).
    def set_resolution(self, resolution=MAPPING_RESOLUTION.HIGH):
        if isinstance(resolution, MAPPING_RESOLUTION):
            self.spatial.set(<c_MAPPING_RESOLUTION> (<unsigned int>resolution.value))
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    ##
    # Sets the range corresponding to the given \ref MAPPING_RANGE preset.
    # \param mapping_range : the desired \ref MAPPING_RANGE . default : [MAPPING_RANGE.AUTO](\ref MAPPING_RANGE)
    def set_range(self, mapping_range=MAPPING_RANGE.AUTO):
        if isinstance(mapping_range, MAPPING_RANGE):
            self.spatial.set(<c_MAPPING_RANGE> (<unsigned int>mapping_range.value))
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    ##
    # Returns the maximum value of depth corresponding to the given \ref MAPPING_RANGE presets.
    # \param range : the desired \ref MAPPING_RANGE . default : [MAPPING_RANGE.AUTO](\ref MAPPING_RANGE)
    # \return The maximum value of depth.
    def get_range_preset(self, mapping_range=MAPPING_RANGE.AUTO):
        if isinstance(mapping_range, MAPPING_RANGE):
            return self.spatial.get(<c_MAPPING_RANGE> (<unsigned int>mapping_range.value))
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    ##
    # Returns the resolution corresponding to the given \ref MAPPING_RESOLUTION preset.
    # \param resolution : the desired \ref MAPPING_RESOLUTION . default : [MAPPING_RESOLUTION.HIGH](\ref MAPPING_RESOLUTION)
    # \return The resolution in meter
    def get_resolution_preset(self, resolution=MAPPING_RESOLUTION.HIGH):
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.get(<c_MAPPING_RESOLUTION> (<unsigned int>resolution.value))
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    ##
    # Returns the recommended maximum depth value for the given resolution
    # \param resolution : the desired resolution, either defined by a \ref MAPPING_RESOLUTION preset or a resolution value in meters
    # \param py_cam : the \ref Camera object which will run the spatial mapping.
    # \return The maximum value of depth in meters.
    def get_recommended_range(self, resolution, py_cam: Camera):
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.getRecommendedRange(<c_MAPPING_RESOLUTION> (<unsigned int>resolution.value), py_cam.camera)
        elif isinstance(resolution, float):
            return self.spatial.getRecommendedRange(<float> resolution, py_cam.camera)
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION or float type.")

    ##
    # The type of spatial map to be created. This dictates the format that will be used for the mapping(e.g. mesh, point cloud). See \ref SPATIAL_MAP_TYPE
    @property
    def map_type(self):
        return SPATIAL_MAP_TYPE(<int>self.spatial.map_type)

    @map_type.setter
    def map_type(self, value):
        self.spatial.map_type = <c_SPATIAL_MAP_TYPE>(<unsigned int>value.value)

    ## 
    # The maximum CPU memory (in mega bytes) allocated for the meshing process.
    @property
    def max_memory_usage(self):
        return self.spatial.max_memory_usage

    @max_memory_usage.setter
    def max_memory_usage(self, value: int):
        self.spatial.max_memory_usage = value

    ##
    # Set to true if you want to be able to apply the texture to your mesh after its creation.
    # \note This option will consume more memory.
    # \note This option is only available for \ref SPATIAL_MAP_TYPE.MESH
    @property
    def save_texture(self):
        return self.spatial.save_texture

    @save_texture.setter
    def save_texture(self, value: bool):
        self.spatial.save_texture = value

    ##
    # Set to false if you want to ensure consistency between the mesh and its inner chunk data.
    # \note Updating the mesh is time-consuming. Setting this to true results in better performance.
    @property
    def use_chunk_only(self):
        return self.spatial.use_chunk_only

    @use_chunk_only.setter
    def use_chunk_only(self, value: bool):
        self.spatial.use_chunk_only = value

    ##
    # Specify if the order of the vertices of the triangles needs to be inverted. If your display process does not handle front and back face culling you can use this to set it right.
    # \note This option is only available for \ref SPATIAL_MAP_TYPE.MESH
    @property
    def reverse_vertex_order(self):
        return self.spatial.reverse_vertex_order

    @reverse_vertex_order.setter
    def reverse_vertex_order(self, value: bool):
        self.spatial.reverse_vertex_order = value

    ##
    # Gets the range of the minimal/maximal depth value allowed by the spatial mapping in a numpy array.
    # The first value of the array is the minimum value allowed.
    # The second value of the array is the maximum value allowed.
    @property
    def allowed_range(self):
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_range.first
        arr[1] = self.spatial.allowed_range.second
        return arr

    ##
    # Depth range in meters. Can be different from the value set by \ref Camera.set_depth_max_range_value()
    # Set to 0 by default. In this case, the range is computed from resolution_meter and from the current internal parameters to fit your application.
    # Deprecated : Since SDK 2.6, we recommend leaving this to 0.
    @property
    def range_meter(self):
        return self.spatial.range_meter

    @range_meter.setter
    def range_meter(self, value: float):
        self.spatial.range_meter = value

    ##
    # Gets the range of the maximal depth value allowed by the spatial mapping in a numpy array.
    # The first value of the array is the minimum value allowed.
    # The second value of the array is the maximum value allowed.
    @property
    def allowed_resolution(self):
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_resolution.first
        arr[1] = self.spatial.allowed_resolution.second
        return arr

    ##
    # Spatial mapping resolution in meters, should fit \ref allowed_resolution
    @property
    def resolution_meter(self):
        return self.spatial.resolution_meter

    @resolution_meter.setter
    def resolution_meter(self, value: float):
        self.spatial.resolution_meter = value
    ##
    # Saves the current set of parameters into a file
    # \param filename : the path to the file in which the parameters will be stored.
    # \return true if the file was successfully saved, otherwise false.
    def save(self, filename: str):
        filename_save = filename.encode()
        return self.spatial.save(String(<char*> filename_save))

    ##
    # Loads the values of the parameters contained in a file.
    # \param filename : the path to the file from which the parameters will be loaded.
    # \return true if the file was successfully loaded, otherwise false.
    def load(self, filename: str):
        filename_load = filename.encode()
        return self.spatial.load(String(<char*> filename_load))

##
# Contains positional tracking data which gives the position and orientation of the ZED in 3D space.
# \ingroup PositionalTracking_group
# Different representations of position and orientation can be retrieved, along with timestamp and pose confidence.
cdef class Pose:
    cdef c_Pose pose
    def __cinit__(self):
        self.pose = c_Pose()

    ##
    # Deep copy from another \ref Pose
    # \param pose : the \ref Pose to copy
    def init_pose(self, pose: Pose):
        self.pose = c_Pose(pose.pose)

    ##
    # Inits \ref Pose from pose data
    # 
    # \param pose_data : \ref Transform containing pose data to copy
    # \param timestamp : pose timestamp
    # \param confidence : pose confidence
    def init_transform(self, pose_data: Transform, timestamp=0, confidence=0):
        self.pose = c_Pose(pose_data.transform[0], timestamp, confidence)

    ##
    # Returns the translation from the pose.
    # \param py_translation : \ref Translation to be returned. It creates one by default.
    # \return The (3x1) translation vector
    def get_translation(self, py_translation = Translation()):
        (<Translation>py_translation).translation = self.pose.getTranslation()
        return py_translation

    ##
    # Returns the orientation from the pose.
    # \param py_orientation : \ref Orientation to be returned. It creates one by default.
    # \return The (3x1) orientation vector
    def get_orientation(self, py_orientation = Orientation()):
        (<Orientation>py_orientation).orientation = self.pose.getOrientation()
        return py_orientation

    ##
    # Returns the rotation (3x3) from the pose.
    # \param py_rotation : \ref Rotation to be returned. It creates one by default.
    # \return The (3x3) rotation matrix
    # \warning The given \ref Rotation contains a copy of the \ref Transform values. Not references.
    def get_rotation_matrix(self, py_rotation = Rotation()):    
        cdef c_Rotation tmp = self.pose.getRotationMatrix()
        for i in range(9):
            (<Rotation>py_rotation).rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Returns the rotation (3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula) from the pose.
    # \return The (3x1) rotation vector (numpy array)
    def get_rotation_vector(self):
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.pose.getRotationVector()[i]
        return arr

    ##
    # Converts the \ref Rotation of the \ref Transform as Euler angles.
    # \param radian : True if the angle in is radian, False if it is in degree. Default : True.
    # \return The Euler angles, as a float3 representing the rotations arround the X, Y and Z axes. (numpy array)
    def get_euler_angles(self, radian=True):
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.pose.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr

    ##
    # boolean that indicates if tracking is activated or not. You should check that first if something went wrong.
    @property
    def valid(self):
        return self.pose.valid

    @valid.setter
    def valid(self, valid_: bool):
        self.pose.valid = valid_

    ##
    # \ref Timestamp of the pose. This timestamp should be compared with the camera timestamp for synchronization.
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp = self.pose.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.pose.timestamp.data_ns = timestamp

    ##
    # Gets the 4x4 Matrix which contains the rotation (3x3) and the translation. \ref Orientation is extracted from this transform as well.
    # \param pose_data : \ref Transform to be returned. It creates one by default.
    # \return the pose data \ref Transform
    def pose_data(self, pose_data = Transform()):
        for i in range(16):
            (<Transform>pose_data).transform.m[i] = self.pose.pose_data.m[i]
        return pose_data

    ##
    # Confidence/Quality of the pose estimation for the target frame.
    # A confidence metric of the tracking [0-100], 0 means that the tracking is lost, 100 means that the tracking can be fully trusted.
    @property
    def pose_confidence(self):
        return self.pose.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, pose_confidence_: int):
        self.pose.pose_confidence = pose_confidence_

    ##
    # 6x6 Pose covariance of translation (the first 3 values) and rotation in so3 (the last 3 values) (numpy array)
    # \note Computed only if \ref PositionalTrackingParameters.enable_spatial_memory is disabled.
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


##
# Lists different states of the camera motion
#
# \ingroup Sensors_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | STATIC | The camera is static. |
# | MOVING | The camera is moving. |
# | FALLING | The camera is falling. |
class CAMERA_MOTION_STATE(enum.Enum):
    STATIC = <int>c_CAMERA_MOTION_STATE.STATIC
    MOVING = <int>c_CAMERA_MOTION_STATE.MOVING
    FALLING = <int>c_CAMERA_MOTION_STATE.FALLING
    LAST = <int>c_CAMERA_MOTION_STATE.CAMERA_MOTION_STATE_LAST

##
# Defines the location of each sensor for \ref TemperatureData .
# \ingroup Sensors_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | IMU | The IMU sensor location |
# | BAROMETER | The Barometer sensor location |
# | ONBOARD_LEFT | The Temperature sensor left location |
# | ONBOARD_RIGHT | The Temperature sensor right location |
class SENSOR_LOCATION(enum.Enum):
    IMU = <int>c_SENSOR_LOCATION.IMU
    BAROMETER = <int>c_SENSOR_LOCATION.BAROMETER
    ONBOARD_LEFT = <int>c_SENSOR_LOCATION.ONBOARD_LEFT
    ONBOARD_RIGHT = <int>c_SENSOR_LOCATION.ONBOARD_RIGHT
    LAST = <int>c_SENSOR_LOCATION.SENSOR_LOCATION_LAST

##
# Contains Barometer sensor data.
# \ingroup Sensors_group
cdef class BarometerData:
    cdef c_BarometerData barometerData

    def __cinit__(self):
        self.barometerData = c_BarometerData()

    ##
    # Defines if the sensor is available
    @property
    def is_available(self):
        return self.barometerData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.barometerData.is_available = is_available

    ##
    # Barometer ambient air pressure in hPa
    @property
    def pressure(self):
        return self.barometerData.pressure

    @pressure.setter
    def pressure(self, pressure: float):
        self.barometerData.pressure=pressure

    ##
    # Relative altitude from first camera position (at open() time)
    @property
    def relative_altitude(self):
        return self.barometerData.relative_altitude

    @relative_altitude.setter
    def relative_altitude(self, alt: float):
        self.barometerData.relative_altitude = alt

    ##
    # Defines the sensors data timestamp
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp = self.barometerData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.barometerData.timestamp.data_ns = timestamp

    ##
    # Realtime data acquisition rate [Hz]
    @property
    def effective_rate(self):
        return self.barometerData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.barometerData.effective_rate = rate

##
# Contains sensors temperature data.
# \ingroup Sensors_group
cdef class TemperatureData:
    cdef c_TemperatureData temperatureData

    def __cinit__(self):
        self.temperatureData = c_TemperatureData()

    ##
    # Gets temperature of sensor location
    # \param location : the sensor location ( \ref SENSOR_LOCATION )
    # \return temperature of sensor location
    def get(self, location):
        cdef float value
        value = 0
        if isinstance(location,SENSOR_LOCATION):
            err = ERROR_CODE(<int>self.temperatureData.get(<c_SENSOR_LOCATION>(<unsigned int>(location.value)), value))
            if err == ERROR_CODE.SUCCESS :
                return value
            else :
                return -1
        else:
            raise TypeError("Argument not of type SENSOR_LOCATION")


##
# Defines the magnetic heading state for \ref MagnetometerData
# \ingroup Sensors_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | GOOD | The heading is reliable and not affected by iron interferences |
# | OK | The heading is reliable, but affected by slight iron interferences |
# | NOT_GOOD | The heading is not reliable because affected by strong iron interferences |
# | NOT_CALIBRATED | The magnetometer has not been calibrated |
# | MAG_NOT_AVAILABLE | The magnetomer sensor is not available |
class HEADING_STATE(enum.Enum):
    GOOD = <int>c_HEADING_STATE.GOOD
    OK = <int>c_HEADING_STATE.OK
    NOT_GOOD = <int>c_HEADING_STATE.NOT_GOOD
    NOT_CALIBRATED = <int>c_HEADING_STATE.NOT_CALIBRATED
    MAG_NOT_AVAILABLE = <int>c_HEADING_STATE.MAG_NOT_AVAILABLE
    HEADING_STATE_LAST = <int>c_HEADING_STATE.HEADING_STATE_LAST

##
# Contains magnetometer sensor data.
# \ingroup Sensors_group
cdef class MagnetometerData:
    cdef c_MagnetometerData magnetometerData

    def __cinit__(self):
        self.magnetometerData

    ##
    # Defines if the sensor is available
    @property
    def is_available(self):
        return self.magnetometerData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.magnetometerData.is_available = is_available

    ##
    # Realtime data acquisition rate [Hz]
    @property
    def effective_rate(self):
        return self.magnetometerData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.magnetometerData.effective_rate = rate

    ##
    # The camera heading in degrees relative to the magnetic North Pole.
    # \note The magnetic North Pole has an offset with respect to the geographic North Pole, depending on the geographic position of the camera.
    # \note To get a correct magnetic heading the magnetometer sensor must be calibrated using the ZED Sensor Viewer tool
    @property
    def magnetic_heading(self):
        return self.magnetometerData.magnetic_heading

    @magnetic_heading.setter
    def magnetic_heading(self, heading: float):
        self.magnetometerData.magnetic_heading = heading

    ##
    # The accuracy of the magnetic heading measure in the range [0.0,1.0].
    # \note A negative value means that the magnetometer must be calibrated using the ZED Sensor Viewer tool
    @property
    def magnetic_heading_accuracy(self):
        return self.magnetometerData.magnetic_heading_accuracy

    @magnetic_heading_accuracy.setter
    def magnetic_heading_accuracy(self, accuracy: float):
        self.magnetometerData.magnetic_heading_accuracy = accuracy

    ##
    # The state of the \ref magnetic_heading value
    @property
    def magnetic_heading_state(self):
        return HEADING_STATE(<int>self.magnetometerData.magnetic_heading_state)

    @magnetic_heading_state.setter
    def magnetic_heading_state(self, state):
        if isinstance(state, HEADING_STATE):
            self.magnetometerData.magnetic_heading_state = <c_HEADING_STATE>(<unsigned int>state.value)
        else:
            raise TypeError("Argument is not of HEADING_STATE type.")

    ##
    # (3x1) Vector for magnetometer raw values (uncalibrated). In other words, the current magnetic field (uT), along with the x, y, and z axes.
    # \return the magnetic field array
    # \note The magnetometer raw values are affected by soft and hard iron interferences. 
    # The sensor must be calibrated, placing the camera in the working environment, using the ZED Sensor Viewer tool.
    # \note Not available in SVO or Stream mode.
    def get_magnetic_field_uncalibrated(self):
        cdef np.ndarray magnetic_field = np.zeros(3)
        for i in range(3):
            magnetic_field[i] = self.magnetometerData.magnetic_field_uncalibrated[i]
        return magnetic_field

    ##
    # (3x1) Vector for magnetometer values (after user calibration). In other words, the current calibrated and normalized magnetic field (uT), along with the x, y, and z axes.
    # \return the magnetic field array
    # \note To calibrate the magnetometer sensor please use the ZED Sensor Viewer tool after placing the camera in the final operating environment
    def get_magnetic_field_calibrated(self):
        cdef np.ndarray magnetic_field = np.zeros(3)
        for i in range(3):
            magnetic_field[i] = self.magnetometerData.magnetic_field_calibrated[i]
        return magnetic_field

    ##
    # Defines the sensors data timestamp
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp = self.magnetometerData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.magnetometerData.timestamp.data_ns = timestamp


##
# Contains all sensors data (except image sensors) to be used for positional tracking or environment study.
# \ingroup Sensors_group
cdef class SensorsData:
    cdef c_SensorsData sensorsData

    def __cinit__(self):
        self.sensorsData = c_SensorsData()

    ##
    # Copy constructor.
    # \param sensorsData : \ref SensorsData object to be copied from
    def init_sensorsData(self, sensorsData: SensorsData):
        self.sensorsData = sensorsData.sensorsData

    ##
    # Indicates if the camera is static, moving or falling
    @property
    def camera_moving_state(self):
        return CAMERA_MOTION_STATE(<int>self.sensorsData.camera_moving_state)

    @camera_moving_state.setter
    def camera_moving_state(self, state):
        if isinstance(state, CAMERA_MOTION_STATE):
            self.sensorsData.camera_moving_state = <c_CAMERA_MOTION_STATE>(<unsigned int>(state.value))
        else:
            raise TypeError("Argument not of type CAMERA_MOTION_STATE")

    ##
    # Indicates if the Sensors data has been taken during a frame capture on sensor.
    # If value is 1, SensorsData has been retrieved during a left sensor frame acquisition (the time precision is linked to the IMU rate, therefore 800Hz == 1.3ms)
    # If value is 0, the data has not been taken during a frame acquisition.
    @property
    def image_sync_trigger(self):
        return self.sensorsData.image_sync_trigger

    @image_sync_trigger.setter
    def image_sync_trigger(self, image_sync_trigger: int):
        self.sensorsData.image_sync_trigger = image_sync_trigger


    ##
    # Gets the \ref IMUData
    # \return the \ref IMUData
    def get_imu_data(self):
        imu_data = IMUData()
        imu_data.imuData = self.sensorsData.imu
        return imu_data

    ##
    # Gets the \ref BarometerData
    # \return the \ref BarometerData
    def get_barometer_data(self):
        barometer_data = BarometerData()
        barometer_data.barometerData = self.sensorsData.barometer
        return barometer_data

    ##
    # Gets the \ref MagnetometerData
    # \return the \ref MagnetometerData
    def get_magnetometer_data(self):
        magnetometer_data = MagnetometerData()
        magnetometer_data.magnetometerData = self.sensorsData.magnetometer
        return magnetometer_data

    ##
    # Gets the \ref TemperatureData
    # \return the \ref TemperatureData
    def get_temperature_data(self):
        temperature_data = TemperatureData()
        temperature_data.temperatureData = self.sensorsData.temperature
        return temperature_data


##
# Contains the IMU sensor data.
# \ingroup Sensors_group
cdef class IMUData:
    cdef c_IMUData imuData

    def __cinit__(self):
        self.imuData = c_IMUData()
    
    ##
    # Gets the (3x1) Vector for raw angular velocity of the gyroscope, given in deg/s.
    # Values are uncorrected from IMU calibration.
    # In other words, the current velocity at which the sensor is rotating around the x, y, and z axes.
    # \param angular_velocity_uncalibrated : An array to be returned. It creates one by default.
    # \return The uncalibrated angular velocity (3x1) vector in an array
    # \note Those values are the exact raw values from the IMU. 
    # \note Not available in SVO or Stream mode
    def get_angular_velocity_uncalibrated(self, angular_velocity_uncalibrated = [0, 0, 0]):
        for i in range(3):
            angular_velocity_uncalibrated[i] = self.imuData.angular_velocity_uncalibrated[i]
        return angular_velocity_uncalibrated    
        
    ##
    # Gets the (3x1) Vector for uncalibrated angular velocity of the gyroscope, given in deg/s.
    # Values are corrected from bias, scale and misalignment.
    # In other words, the current velocity at which the sensor is rotating around the x, y, and z axes.
    # \param angular_velocity : An array to be returned. It creates one by default.
    # \return The angular velocity (3x1) vector in an array
    # \note Those values can be directly ingested in a IMU fusion algorithm to extract quaternion
    # \note Not available in SVO or Stream mode
    def get_angular_velocity(self, angular_velocity = [0, 0, 0]):
        for i in range(3):
            angular_velocity[i] = self.imuData.angular_velocity[i]
        return angular_velocity

    ##
    # Gets the (3x1) Vector for linear acceleration of the gyroscope, given in m/s^2.
    # In other words, the current acceleration of the sensor, along with the x, y, and z axes.
    # \param linear_acceleration : An array to be returned. It creates one by default.
    # \return The linear acceleration (3x1) vector in an array
    # \note Those values can be directly ingested in a IMU fusion algorithm to extract quaternion
    def get_linear_acceleration(self, linear_acceleration = [0, 0, 0]):
        for i in range(3):
            linear_acceleration[i] = self.imuData.linear_acceleration[i]
        return linear_acceleration

    ##
    # Gets the (3x1) Vector for uncalibrated linear acceleration of the gyroscope, given in m/s^2.
    # Values are uncorrected from IMU calibration.
    # In other words, the current acceleration of the sensor, along with the x, y, and z axes.
    # \param linear_acceleration_uncalibrated : An array to be returned. It creates one by default.
    # \return The uncalibrated linear acceleration (3x1) vector in an array
    # \note Those values are the exact raw values from the IMU. 
    # \note Those values can be directly ingested in a IMU fusion algorithm to extract quaternion.
    # \note Not available in SVO or Stream mode
    def get_linear_acceleration_uncalibrated(self, linear_acceleration_uncalibrated = [0, 0, 0]):
        for i in range(3):
            linear_acceleration_uncalibrated[i] = self.imuData.linear_acceleration_uncalibrated[i]
        return linear_acceleration_uncalibrated

    ##
    # Gets the (3x3) Covariance matrix for angular velocity (x,y,z axes)
    # \param angular_velocity_covariance : \ref Matrix3f to be returned. It creates one by default.
    # \return The (3x3) Covariance matrix for angular velocity
    # \note Not available in SVO or Stream mode
    def get_angular_velocity_covariance(self, angular_velocity_covariance = Matrix3f()):        
        for i in range(9):
            (<Matrix3f>angular_velocity_covariance).mat.r[i] = self.imuData.angular_velocity_covariance.r[i]
        return angular_velocity_covariance
        
    ##
    # Gets the (3x3) Covariance matrix for linear acceleration (x,y,z axes)
    # \param linear_acceleration_covariance : \ref Matrix3f to be returned. It creates one by default.
    # \return The (3x3) Covariance matrix for linear acceleration
    # \note Not available in SVO or Stream mode
    def get_linear_acceleration_covariance(self, linear_acceleration_covariance = Matrix3f()):
        for i in range(9):
            (<Matrix3f>linear_acceleration_covariance).mat.r[i] = self.imuData.linear_acceleration_covariance.r[i]
        return linear_acceleration_covariance

    ##
    # Defines if the sensor is available in your camera.
    @property
    def is_available(self):
        return self.imuData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.imuData.is_available = is_available

    ##
    # Defines the sensors data timestamp
    @property
    def timestamp(self):
        ts = Timestamp()
        ts.timestamp = self.imuData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.imuData.timestamp.data_ns = timestamp

    ##
    # Realtime data acquisition rate [Hz]
    @property
    def effective_rate(self):
        return self.imuData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.imuData.effective_rate = rate

    ##
    # (3x3) 3x3 Covariance matrix for pose orientation (x,y,z axes)
    # \param pose_covariance : \ref Matrix3f  to be returned. It creates one by default.
    # \return the \ref Matrix3f to be returned
    def get_pose_covariance(self, pose_covariance = Matrix3f()):
        for i in range(9):
            (<Matrix3f>pose_covariance).mat.r[i] = self.imuData.pose_covariance.r[i]
        return pose_covariance


    ##
    # IMU pose (IMU 6-dof fusion)
    # \param pose : \ref Transform() to be returned. It creates one by default.
    # \return the \ref Transform to be returned
    def get_pose(self, pose = Transform()):
        for i in range(16):
            (<Transform>pose).transform.m[i] = self.imuData.pose.m[i]
        return pose

##
# Recording structure that contains information about SVO.
# \ingroup Video_group
cdef class RecordingStatus:
    cdef c_RecordingStatus recordingState

    ##
    # Recorder status, true if enabled
    @property
    def is_recording(self):
        return self.recordingState.is_recording

    @is_recording.setter
    def is_recording(self, value: bool):
        self.recordingState.is_recording = value

    ##
    # Recorder status, true if the pause is enabled
    @property
    def is_paused(self):
        return self.recordingState.is_recording

    @is_paused.setter
    def is_paused(self, value: bool):
        self.recordingState.is_paused = value

    ##
    # Status of current frame. True for success or false if the frame couldn't be written in the SVO file.
    @property
    def status(self):
        return self.recordingState.status

    @status.setter
    def status(self, value: bool):
        self.recordingState.status = value

    ##
    # Compression time for the current frame in ms.
    @property
    def current_compression_time(self):
        return self.recordingState.current_compression_time

    @current_compression_time.setter
    def current_compression_time(self, value: double):
        self.recordingState.current_compression_time = value

    ##
    # Compression ratio (% of raw size) for the current frame.
    @property
    def current_compression_ratio(self):
        return self.recordingState.current_compression_ratio

    @current_compression_ratio.setter
    def current_compression_ratio(self, value: double):
        self.recordingState.current_compression_ratio = value

    ##
    # Average compression time in ms since beginning of recording.
    @property
    def average_compression_time(self):
        return self.recordingState.average_compression_time

    @average_compression_time.setter
    def average_compression_time(self, value: double):
        self.recordingState.average_compression_time = value

    ##
    # Average compression ratio (% of raw size) since beginning of recording.
    @property
    def average_compression_ratio(self):
        return self.recordingState.average_compression_ratio

    @average_compression_ratio.setter
    def average_compression_ratio(self, value: double):
        self.recordingState.average_compression_ratio = value


##
# This class is the main interface with the camera and the SDK features, such as: video, depth, tracking, mapping, and more. Find more information in the detailed description below.
# \ingroup Video_group
# 
# A standard program will use the \ref Camera class like this:
# \code
#
#        import pyzed.sl as sl
#
#        def main():
#            # --- Initialize a Camera object and open the ZED
#            # Create a ZED camera object
#            zed = sl.Camera()
#
#            # Set configuration parameters
#            init_params = sl.InitParameters()
#            init_params.camera_resolution = sl.RESOLUTION.HD720 #Use HD720 video mode
#            init_params.camera_fps = 60 # Set fps at 60
#
#            # Open the camera
#            err = zed.open(init_params)
#            if err != sl.ERROR_CODE.SUCCESS :
#                print(repr(err))
#                exit(-1)
#
#            runtime_param = sl.RuntimeParameters()
#
#            # --- Main loop grabing images and depth values
#            # Capture 50 frames and stop
#            i = 0
#            image = sl.Mat()
#            depth = sl.Mat()
#            while i < 50 :
#                # Grab an image
#                if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
#                    # Display a pixel color
#                    zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
#                    center_rgb = image.get_value(image.get_width() / 2, image.get_height() / 2)
#                    print("Image ", i, " center pixel R:", int(center_rgb[0]), " G:", int(center_rgb[1]), " B:", int(center_rgb[2]))
#
#                    # Display a pixel depth
#                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Get the depth map
#                    center_depth = depth.get_value(depth.get_width() / 2, depth.get_height() /2)
#                    print("Image ", i," center depth:", center_depth)
#
#                    i = i+1
#
#            # --- Close the Camera
#            zed.close()
#            return 0
#
#        if __name__ == "__main__":
#            main()
#
# \endcode
cdef class Camera:
    cdef c_Camera camera
    def __cinit__(self):
        self.camera = c_Camera()

    ##
    # If \ref open() has been called, this function will close the connection to the camera (or the SVO file) and free the corresponding memory.
    #
    # If \ref open() wasn't called or failed, this function won't have any effect.
    # \note If an asynchronous task is running within the \ref Camera object, like \ref save_area_map(), this function will wait for its completion.
    # The \ref open() function can then be called if needed.
    # \warning If the CUDA context was created by \ref open(), this function will destroy it. Please make sure to delete your GPU \ref sl.Mat objects before the context is destroyed.
    def close(self):
        self.camera.close()

    ##
    # Opens the ZED camera from the provided \ref InitParameters.
    # This function will also check the hardware requirements and run a self-calibration.
    # \param py_init : a structure containing all the initial parameters. default : a preset of \ref InitParameters.
    # \return An error code giving information about the internal process. If \ref ERROR_CODE "SUCCESS" is returned, the camera is ready to use. Every other code indicates an error and the program should be stopped.
    #
    # Here is the proper way to call this function:
    #
    # \code
    # zed = sl.Camera() # Create a ZED camera object
    #
    # init_params = sl.InitParameters() # Set configuration parameters
    # init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode
    # init_params.camera_fps = 60 # Set fps at 60
    #
    # # Open the camera
    # err = zed.open(init_params)
    # if (err != sl.ERROR_CODE.SUCCESS) :
    #   print(repr(err)) # Display the error
    #   exit(-1)
    # \endcode
    #
    # \note
    # If you are having issues opening a camera, the diagnostic tool provided in the SDK can help you identify to problems.
    # If this function is called on an already opened camera, \ref Camera.close() will be called.
    def open(self, py_init=InitParameters()):
        if py_init:
            return ERROR_CODE(<int>self.camera.open(deref((<InitParameters>py_init).init)))
        else:
            print("InitParameters must be initialized first with InitParameters().")


    ##
    # Reports if the camera has been successfully opened. It has the same behavior as checking if \ref open() returns \ref ERROR_CODE "SUCCESS".
    # \return true if the ZED is already setup, otherwise false.
    def is_opened(self):
        return self.camera.isOpened()

    ##
    # This function will grab the latest images from the camera, rectify them, and compute the measurements based on the \ref RuntimeParameters provided (depth, point cloud, tracking, etc.)
    # As measures are created in this function, its execution can last a few milliseconds, depending on your parameters and your hardware.
    # The exact duration will mostly depend on the following parameters:
    # 
    #   - \ref InitParameters.enable_right_side_measure : Activating this parameter increases computation time
    #   - \ref InitParameters.depth_mode : \ref DEPTH_MODE "PERFORMANCE" will run faster than \ref DEPTH_MODE "ULTRA"
    #   - \ref enable_positional_tracking() : Activating the tracking is an additional load
    #   - \ref RuntimeParameters.enable_depth : Avoiding the depth computation must be faster. However, it is required by most SDK features (tracking, spatial mapping, plane estimation, etc.)
    #   - \ref RuntimeParameters.remove_saturated_areas : Remove saturated areas from depth estimation . Recommended to True.
    #
    # This function is meant to be called frequently in the main loop of your application.
    # \note Since ZED SDK 3.0, this function is blocking. It means that grab() will wait until a new frame is detected and available. If no new frames is available until timeout is reached, grab() will return \ref ERROR_CODE.CAMERA_NOT_DETECTED since the camera has probably been disconnected.
    # 
    # \param py_runtime : a structure containing all the runtime parameters. default : a preset of \ref RuntimeParameters.
    # \param Returning \ref ERROR_CODE "SUCCESS" means that no problem was encountered. Returned errors can be displayed using \ref toString(error)
    #
    # \code
    # # Set runtime parameters after opening the camera
    # runtime_param = sl.RuntimeParameters()
    #
    # image = sl.Mat()
    # while True :
    # # Grab an image
    # if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
    #   zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image     
    #   # Use the image for your application
    # \endcode
    def grab(self, py_runtime=RuntimeParameters()):
        if py_runtime:
            return ERROR_CODE(<int>self.camera.grab(deref((<RuntimeParameters>py_runtime).runtime)))
        else:
            print("RuntimeParameters must be initialized first with RuntimeParameters().")

    ##
    # Retrieves images from the camera (or SVO file).
    #
    # Multiple images are available along with a view of various measures for display purposes.
    # Available images and views are listed \ref VIEW "here".
    # As an example, \ref VIEW "VIEW.DEPTH" can be used to get a gray-scale version of the depth map, but the actual depth values can be retrieved using \ref retrieve_measure() .
    #
    # <b>Memory</b>
    # \n By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
    # If your application can use GPU images, using the <b>type</b> parameter can increase performance by avoiding this copy.
    # If the provided \ref Mat object is already allocated  and matches the requested image format, memory won't be re-allocated.
    #
    # <b>Image size</b>
    # \n By default, images are returned in the resolution provided by \ref get_camera_information() in \ref CameraInformation.camera_resolution
    # However, you can request custom resolutions. For example, requesting a smaller image can help you speed up your application.
    # 
    # \param py_mat : \b [out] the \ref Mat to store the image.
    # \param view  : defines the image you want (see \ref VIEW). default : [VIEW.LEFT](\ref VIEW).
    # \param type : defines on which memory the image should be allocated. default :  [MEM.CPU](\ref MEM) (you cannot change this default value)
    # \param resolution : if specified, defines the \ref Resolution of the output mat. If set to (0,0) , the ZED resolution will be taken. default : (0,0).
    # \return An \ref ERROR_CODE :
    # \n - [ERROR_CODE.SUCCESS](\ref ERROR_CODE) if the method succeeded,
    # \n - [ERROR_CODE.INVALID_FUNCTION_PARAMETERS](\ref ERROR_CODE) if the view mode requires a module not enabled ([VIEW.DEPTH](\ref DEPTH) with [DEPTH_MODE.NONE](\ref DEPTH_MODE) for example),
    # \n - [ERROR_CODE.INVALID_RESOLUTION](\ref ERROR_CODE) if the resolution is higher than \ref CameraInformation.camera_resolution provided by \ref get_camera_information() 
    #
    # \note As this function retrieves the images grabbed by the \ref grab() function, it should be called afterwards.
    #
    # \code
    # # create sl.Mat objects to store the images
    # left_image = sl.Mat()
    # depth_view = sl.Mat()
    # while True :
    # # Grab an image
    # if zed.grab() == sl.ERROR_CODE.SUCCESS : # A new image is available if grab() returns SUCCESS
    #     zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
    #     zed.retrieve_image(depth_view, sl.VIEW.DEPTH) # Get a grayscale preview of the depth map
    #
    #     # Display the center pixel colors
    #     left_center = left_image.get_value(left_image.get_width() / 2, left_image.get_height() / 2)
    #     print("left_image center pixel R:", int(left_center[0]), " G:", int(left_center[1]), " B:", int(left_center[2]))
    #
    #     depth_center = depth_view.get_value(depth_view.get_width() / 2, depth_view.get_height() / 2)
    #     print("depth_view center pixel R:", int(depth_venter[1]), " G:", int(depth_center[1]), " B:", int(depth_center[2]))
    # \endcode
    def retrieve_image(self, py_mat: Mat, view=VIEW.LEFT, type=MEM.CPU, resolution=Resolution(0,0)):
        if (isinstance(view, VIEW) and isinstance(type, MEM)):
            return ERROR_CODE(<int>self.camera.retrieveImage(py_mat.mat, <c_VIEW>(<unsigned int>view.value), <c_MEM>(<unsigned int>type.value), (<Resolution>resolution).resolution))
        else:
            raise TypeError("Arguments must be of VIEW, MEM and integer types.")

    ##
    # Computed measures, like depth, point cloud, or normals, can be retrieved using this method.
    # 
    # Multiple measures are available after a \ref Camera.grab() call. A full list is available here.
    #
    # <b>Memory</b>
    # By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
    # If your application can use GPU images, using the \b type parameter can increase performance by avoiding this copy.
    # If the provided \ref Mat object is already allocated and matches the requested image format, memory won't be re-allocated.
    # 
    # <b>Measure size</b>
    # By default, measures are returned in the resolution provided by \ref get_camera_information() in \ref CameraInformations.camera_resolution .
    # However, custom resolutions can be requested. For example, requesting a smaller measure can help you speed up your application.
    #
    # \param py_mat : \b [out] the \ref Mat to store the measures
    # \param measure : defines the measure you want. (see \ref MEASURE), default : [MEASURE.DEPTH](\ref MEASURE)
    # \param type : defines on which memory the mat should be allocated. default : [MEM.CPU](\ref MEM) (you cannot change this default value)
    # \param resolution : if specified, defines the resolution of the output mat. If set to \ref Resolution (0,0) , the ZED resolution will be taken. default : (0,0).
    # \return An \ref ERROR_CODE
    # \n - [ERROR_CODE.SUCCESS](\ref ERROR_CODE) if the method succeeded,
    # \n - [ERROR_CODE.INVALID_FUNCTION_PARAMETERS](\ref ERROR_CODE) if the view mode requires a module not enabled ([VIEW.DEPTH](\ref DEPTH) with [DEPTH_MODE.NONE](\ref DEPTH_MODE for example),
    # \n - [ERROR_CODE.INVALID_RESOLUTION](\ref ERROR_CODE) if the resolution is higher than \ref CameraInformation.camera_resolution provided by \ref get_camera_information() 
    # \n - [ERROR_CODE.FAILURE](\ref ERROR_CODE) if another error occured.
    #
    # \note As this function retrieves the measures computed by the \ref grab() function, it should be called after.
    #
    # \code
    # depth_map = sl.Mat()
    # point_cloud = sl.Mat()
    # resolution = zed.get_camera_information().camera_resolution
    # x = int(resolution.width / 2) # Center coordinates
    # y = int(resolution.height / 2)
    #
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image
    #
    #         zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU) # Get the depth map
    #
    #         # Read a depth value
    #         center_depth = depth_map.get_value(x, y sl.MEM.CPU) # each depth map pixel is a float value
    #         if isnormal(center_depth) : # + Inf is "too far", -Inf is "too close", Nan is "unknown/occlusion"
    #             print("Depth value at center: ", center_depth, " ", init_params.coordinate_units)
    #         zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU) # Get the point cloud
    #
    #         # Read a point cloud value
    #         err, pc_value = point_cloud.get_value(x, y) # each point cloud pixel contains 4 floats, so we are using a numpy array
    #         
    #         # Get 3D coordinates
    #         if (isnormal(pc_value[2])) :
    #             print("Point cloud coordinates at center: X=", pc_value[0], ", Y=", pc_value[1], ", Z=", pc_value[2])
    #         
    #        # Get color information using Python struct package to unpack the unsigned char array containing RGBA values
    #        import struct
    #        packed = struct.pack('f', pc_value[3])
    #        char_array = struct.unpack('BBBB', packed)
    #        print("Color values at center: R=", char_array[0], ", G=", char_array[1], ", B=", char_array[2], ", A=", char_array[3])
    #     
    # \endcode
    def retrieve_measure(self, py_mat: Mat, measure=MEASURE.DEPTH, type=MEM.CPU, resolution=Resolution(0,0)):
        if (isinstance(measure, MEASURE) and isinstance(type, MEM)):
            return ERROR_CODE(<int>self.camera.retrieveMeasure(py_mat.mat, <c_MEASURE>(<unsigned int>measure.value), <c_MEM>(<unsigned int>type.value), (<Resolution>resolution).resolution))
        else:
            raise TypeError("Arguments must be of MEASURE, MEM and integer types.")

    ##
    # Defines a region of interest to focus on for all the SDK, discarding other parts.
    # \param roi_mask: the \ref Mat defining the requested region of interest, all pixel set to 0 will be discard. If empty, set all pixels as valid, 
    # otherwise should fit the resolution of the current instance and its type should be U8_C1.
    # \return An ERROR_CODE if something went wrong.
    def set_region_of_interest(self, py_mat: Mat):
        return ERROR_CODE(<int>self.camera.setRegionOfInterest(py_mat.mat))

    def start_publishing(self, communication_parameters : CommunicationParameters):
        return ERROR_CODE(<int>self.camera.startPublishing(communication_parameters.communicationParameters))

    ##
    # Sets the playback cursor to the desired frame number in the SVO file.
    #
    # This function allows you to move around within a played-back SVO file. After calling, the next call to \ref grab() will read the provided frame number.
    #
    # \param frame_number : the number of the desired frame to be decoded.
    # 
    # \note Works only if the camera is open in SVO playback mode.
    #
    # \code
    #
    # import pyzed.sl as sl
    #
    # def main() :
    #
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #
    #     # Set configuration parameters
    #     init_params = sl.InitParameters()
    #     init_params.set_from_svo_file("path/to/my/file.svo")
    #
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Loop between frame 0 and 50
    #     i = 0
    #     left_image = sl.Mat()
    #     while zed.get_svo_position() < zed.get_svo_number_of_frames()-1 :
    #
    #         print("Current frame: ", zed.get_svo_position())
    #
    #         # Loop if we reached frame 50
    #         if zed.get_svo_position() == 50 :
    #             zed.set_svo_position(0)
    #
    #         # Grab an image
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #             zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
    #
    #         # Use the image in your application
    #
    #         # Close the Camera
    #         zed.close()
    #         return 0
    #
    # if __name__ == "__main__" :
    #     main()
    #
    # \endcode
    def set_svo_position(self, frame_number: int):
        self.camera.setSVOPosition(frame_number)

    ##
    # Returns the current playback position in the SVO file.
    #
    # The position corresponds to the number of frames already read from the SVO file, starting from 0 to n.
    # 
    # Each \ref grab() call increases this value by one (except when using \ref InitParameters.svo_real_time_mode).
    # \return The current frame position in the SVO file. Returns -1 if the SDK is not reading an SVO.
    # 
    # \note Works only if the camera is open in SVO playback mode.
    #
    # See \ref set_svo_position() for an example.
    def get_svo_position(self):
        return self.camera.getSVOPosition()

    ##
    # Returns the number of frames in the SVO file.
    #
    # \return The total number of frames in the SVO file (-1 if the SDK is not reading a SVO).
    #
    # \note Works only if the camera is open in SVO reading mode.
    def get_svo_number_of_frames(self):
        return self.camera.getSVONumberOfFrames()

    ##
    # Sets the value of the requested \ref VIDEO_SETTINGS "camera setting" (gain, brightness, hue, exposure, etc.)
    #
    # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
    #
    # \param settings : the setting to be set
    # \param value : the value to set, default : auto mode
    #
    # \code
    # # Set the gain to 50
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
    # \endcode
    #
    # \warning Setting [VIDEO_SETTINGS.EXPOSURE](\ref VIDEO_SETTINGS) or [VIDEO_SETTINGS.GAIN](\ref VIDEO_SETTINGS) to default will automatically sets the other to default.
    #
    # \note Works only if the camera is opened in live mode.
    def set_camera_settings(self, settings: VIDEO_SETTINGS, value=-1):
        if isinstance(settings, VIDEO_SETTINGS) :
            return ERROR_CODE(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), value))
        else:
            raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")

    def set_camera_settings_range(self, settings: VIDEO_SETTINGS, min=-1, max=-1):
        if isinstance(settings, VIDEO_SETTINGS) :
            return ERROR_CODE(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), min, max))
        else:
            raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")

    ##
    # Sets the ROI of the requested \ref VIDEO_SETTINGS "camera setting" (AEC_AGC_ROI)
    #
    # \param settings : the setting to be set
    # \param roi : the requested ROI
    # \param eye : the requested side. Default: \ref SIDE "SIDE.BOTH"
    # \param reset : cancel the manual ROI and reset it to the full image. Default: False
    #
    # \code
    #   roi = sl.Rect(42, 56, 120, 15)
    #   zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
    # \endcode
    #
    # \note Works only if the camera is opened in live mode.
    def set_camera_settings_roi(self, settings: VIDEO_SETTINGS, roi: Rect, eye = SIDE.BOTH, reset = False):
        if isinstance(settings, VIDEO_SETTINGS) :
            return ERROR_CODE(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<unsigned int>settings.value), roi.rect, <c_SIDE>(<unsigned int>eye.value), reset))
        else:
            raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")
    
    ##
    # Returns the current value of the requested \ref VIDEO_SETTINGS "camera setting". (gain, brightness, hue, exposure, etc.)
    # 
    # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
    # 
    # \param setting : the requested setting.
    # \return The current value for the corresponding setting. Returns -1 if encounters an error.
    #
    # \code
    # gain = zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
    # print("Current gain value: ", gain)
    # \endcode
    #
    # \note Works only if the camera is open in live mode. (Settings aren't exported in the SVO file format)            
    def get_camera_settings(self, setting: VIDEO_SETTINGS) -> (ERROR_CODE, int):
        cdef int value
        if isinstance(setting, VIDEO_SETTINGS):
            error_code = ERROR_CODE(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<unsigned int&>setting.value), value))
            return error_code, value
        else:
            raise TypeError("Argument is not of VIDEO_SETTINGS type.")

    def get_camera_settings_range(self, setting: VIDEO_SETTINGS) -> (ERROR_CODE, int, int):
        cdef int min
        cdef int max
        if isinstance(setting, VIDEO_SETTINGS):
            error_code = ERROR_CODE(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<unsigned int>setting.value), <int&>min, <int&>max))
            return error_code, min, max
        else:
            raise TypeError("Argument is not of VIDEO_SETTINGS type.")

    ##
    # Returns the current value of the currently used ROI for the \ref VIDEO_SETTINGS "camera setting" (AEC_AGC_ROI)
    # 
    # \param setting : the requested setting.
    # \param roi : the current ROI used
    # \param eye : the requested side. Default: \ref SIDE "SIDE.BOTH"
    # \return An \ref ERROR_CODE
    #
    # \code
    # roi = sl.Rect()
    # err = zed.get_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
    # print("Current ROI for AEC_AGC: " + str(roi_.x) + " " + str(roi_.y)+ " " + str(roi_.width) + " " + str(roi_.height))
    # \endcode
    #
    # \note Works only if the camera is open in live mode. (Settings aren't exported in the SVO file format)       
    def get_camera_settings_roi(self, setting: VIDEO_SETTINGS, roi: Rect, eye = SIDE.BOTH):
        if isinstance(setting, VIDEO_SETTINGS) and isinstance(eye, SIDE):
            return ERROR_CODE(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<unsigned int>setting.value), roi.rect, <c_SIDE>(<unsigned int>eye.value)))
        else:
            raise TypeError("Argument is not of SIDE type.")

    ##
    # Returns the current framerate at which the \ref grab() method is successfully called.
    #
    # The returned value is based on the difference of camera \ref get_timestamp() "timestamps" between two successful grab() calls.
    #
    # \return The current SDK framerate
    #
    # \warning The returned framerate (number of images grabbed per second) can be lower than \ref InitParameters.camera_fps if the \ref grab() function runs slower than the image stream or is called too often.
    #
    # \code
    # current_fps = zed.get_current_fps()
    # print("Current framerate: ", current_fps)
    # \endcode
    def get_current_fps(self):
        return self.camera.getCurrentFPS()

    ##
    # Returns the timestamp in the requested \ref TIME_REFERENCE
    #
    # - When requesting the [TIME_REFERENCE.IMAGE](\ref TIME_REFERENCE) timestamp, the UNIX nanosecond timestamp of the latest \ref grab() "grabbed" image will be returned.
    # This value corresponds to the time at which the entire image was available in the PC memory. As such, it ignores the communication time that corresponds to 2 or 3 frame-time based on the fps (ex: 33.3ms to 50ms at 60fps).
    #
    # - When requesting the [TIME_REFERENCE.CURRENT](\ref TIME_REFERENCE) timestamp, the current UNIX nanosecond timestamp is returned.
    #
    # This function can also be used when playing back an SVO file.
    #
    # \param time_reference : The selected \ref TIME_REFERENCE.
    # \return The \ref Timestamp in nanosecond. 0 if not available (SVO file without compression).
    #
    # \note As this function returns UNIX timestamps, the reference it uses is common across several \ref Camera instances.
    #
    # \code
    # last_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
    # current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
    # print("Latest image timestamp: ", last_image_timestamp.get_nanoseconds(), "ns from Epoch.")
    # print("Current timestamp: ", current_timestamp.get_nanoseconds(), "ns from Epoch.")
    # \endcode 
    def get_timestamp(self, time_reference: TIME_REFERENCE):
        if isinstance(time_reference, TIME_REFERENCE):
            ts = Timestamp()
            ts.timestamp = self.camera.getTimestamp(<c_TIME_REFERENCE>(<unsigned int>time_reference.value))
            return ts
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")

    ##
    # Returns the number of frames dropped since \ref grab() was called for the first time.
    #
    # A dropped frame corresponds to a frame that never made it to the grab function.
    # This can happen if two frames were extracted from the camera when grab() is called. The older frame will be dropped so as to always use the latest (which minimizes latency)
    # \return The number of frames dropped since the first \ref grab() call.
    def get_frame_dropped_count(self):
        return self.camera.getFrameDroppedCount()


    ##
    # Gets the current range of perceived depth.
    def get_current_min_max_depth(self,min: float,max: float):
        return ERROR_CODE(<int>self.camera.getCurrentMinMaxDepth(min.float,max.float))

    ##
    # Returns the calibration parameters, serial number and other information about the camera being used.
    # As calibration parameters depend on the image resolution, you can provide a custom resolution as a parameter to get scaled information.
    # When reading an SVO file, the parameters will correspond to the camera used for recording.
    # 
    # \param resizer : You can specify a size different from default image size to get the scaled camera information. default = (0,0) meaning original image size.
    # \return \ref CameraInformation containing the calibration parameters of the ZED, as well as serial number and firmware version.
    #
    # \note The returned parameters might vary between two execution due to the \ref InitParameters.camera_disable_self_calib "self-calibration" being ran in the \ref open() method.
    def get_camera_information(self, resizer = Resolution(0, 0)):
        return CameraInformation(self, resizer)

    ##
    # Returns the runtime parameters used. Corresponds to the structure sent when the \ref grab() function was called
    # \return \ref RuntimeParameters containing the parameters that defines the behavior of the \ref grab()
    def get_runtime_parameters(self) :
        runtime = RuntimeParameters()
        runtime.runtime.measure3D_reference_frame = self.camera.getRuntimeParameters().measure3D_reference_frame
        runtime.runtime.enable_depth = self.camera.getRuntimeParameters().enable_depth
        runtime.runtime.confidence_threshold = self.camera.getRuntimeParameters().confidence_threshold
        runtime.runtime.texture_confidence_threshold = self.camera.getRuntimeParameters().texture_confidence_threshold
        runtime.runtime.remove_saturated_areas = self.camera.getRuntimeParameters().remove_saturated_areas
        return runtime

    ##
    # Returns the init parameters used. Corresponds to the structure sent when the \ref open() function was called
    #
    # \return \ref InitParameters containing the parameters used to initialize the \ref Camera object.
    def get_init_parameters(self) :
        init = InitParameters()
        init.init.camera_resolution = self.camera.getInitParameters().camera_resolution
        init.init.camera_fps = self.camera.getInitParameters().camera_fps
        init.init.camera_image_flip = self.camera.getInitParameters().camera_image_flip
        init.init.camera_disable_self_calib = self.camera.getInitParameters().camera_disable_self_calib
        init.init.enable_right_side_measure = self.camera.getInitParameters().enable_right_side_measure
        init.init.svo_real_time_mode = self.camera.getInitParameters().svo_real_time_mode
        init.init.depth_mode = self.camera.getInitParameters().depth_mode
        init.init.depth_stabilization = self.camera.getInitParameters().depth_stabilization
        init.init.depth_minimum_distance = self.camera.getInitParameters().depth_minimum_distance
        init.init.depth_maximum_distance = self.camera.getInitParameters().depth_maximum_distance
        init.init.coordinate_units = self.camera.getInitParameters().coordinate_units
        init.init.coordinate_system = self.camera.getInitParameters().coordinate_system
        init.init.sdk_gpu_id = self.camera.getInitParameters().sdk_gpu_id
        init.init.sdk_verbose = self.camera.getInitParameters().sdk_verbose
        init.init.sdk_verbose_log_file = self.camera.getInitParameters().sdk_verbose_log_file
        init.init.input = self.camera.getInitParameters().input
        init.init.optional_settings_path = self.camera.getInitParameters().optional_settings_path
        init.init.async_grab_camera_recovery = self.camera.getInitParameters().async_grab_camera_recovery
        init.init.grab_compute_capping_fps = self.camera.getInitParameters().grab_compute_capping_fps
        return init

    ##
    # Returns the positional tracking parameters used. Corresponds to the structure sent when the \ref Camera.enable_positional_tracking() function was called.
    #
    # \return \ref PositionalTrackingParameters containing the parameters used for positional tracking initialization.
    def get_positional_tracking_parameters(self) :
        tracking = PositionalTrackingParameters()
        tracking.tracking.initial_world_transform = self.camera.getPositionalTrackingParameters().initial_world_transform
        tracking.tracking.enable_area_memory = self.camera.getPositionalTrackingParameters().enable_area_memory
        tracking.tracking.enable_pose_smoothing = self.camera.getPositionalTrackingParameters().enable_pose_smoothing
        tracking.tracking.set_floor_as_origin = self.camera.getPositionalTrackingParameters().set_floor_as_origin
        tracking.tracking.area_file_path = self.camera.getPositionalTrackingParameters().area_file_path
        tracking.tracking.enable_imu_fusion  = self.camera.getPositionalTrackingParameters().enable_imu_fusion
        tracking.tracking.set_as_static = self.camera.getPositionalTrackingParameters().set_as_static
        tracking.tracking.depth_min_range = self.camera.getPositionalTrackingParameters().depth_min_range
        tracking.tracking.set_gravity_as_origin = self.camera.getPositionalTrackingParameters().set_gravity_as_origin
        return tracking

    ## 
    # Returns the spatial mapping parameters used. Corresponds to the structure sent when the \ref Camera.enable_spatial_mapping() function was called.
    #
    # \return \ref SpatialMappingParameters containing the parameters used for spatial mapping initialization.
    def get_spatial_mapping_parameters(self) :
        spatial = SpatialMappingParameters()
        spatial.spatial.resolution_meter =  self.camera.getSpatialMappingParameters().resolution_meter
        spatial.spatial.range_meter =  self.camera.getSpatialMappingParameters().range_meter
        spatial.spatial.save_texture =  self.camera.getSpatialMappingParameters().save_texture
        spatial.spatial.use_chunk_only =  self.camera.getSpatialMappingParameters().use_chunk_only
        spatial.spatial.max_memory_usage =  self.camera.getSpatialMappingParameters().max_memory_usage
        spatial.spatial.reverse_vertex_order =  self.camera.getSpatialMappingParameters().reverse_vertex_order
        spatial.spatial.map_type =  self.camera.getSpatialMappingParameters().map_type
        return spatial

    ##
    # Returns the object detection parameters used. Corresponds to the structure sent when the \ref Camera.enable_object_detection() function was called
    #
    # \return \ref ObjectDetectionParameters containing the parameters used for object detection initialization.
    def get_object_detection_parameters(self, instance_module_id=0) :
        object_detection = ObjectDetectionParameters()
        object_detection.object_detection.image_sync = self.camera.getObjectDetectionParameters(instance_module_id).image_sync
        object_detection.object_detection.enable_tracking = self.camera.getObjectDetectionParameters(instance_module_id).enable_tracking
        object_detection.object_detection.max_range = self.camera.getObjectDetectionParameters(instance_module_id).max_range
        object_detection.object_detection.prediction_timeout_s = self.camera.getObjectDetectionParameters(instance_module_id).prediction_timeout_s
        object_detection.object_detection.instance_module_id = instance_module_id
        object_detection.object_detection.enable_segmentation = self.camera.getObjectDetectionParameters(instance_module_id).enable_segmentation
        return object_detection

    ##
    # Returns the object detection parameters used. Correspond to the structure send when the \ref enable_body_tracking() function was called.
    #
    # \return \ref BodyTrackingParameters containing the parameters used for object detection initialization.
    def get_body_tracking_parameters(self, instance_id = 0):
        body_params = BodyTrackingParameters()
        body_params.bodyTrackingParameters.image_sync = self.camera.getBodyTrackingParameters(instance_id).image_sync
        body_params.bodyTrackingParameters.enable_tracking = self.camera.getBodyTrackingParameters(instance_id).enable_tracking
        body_params.bodyTrackingParameters.enable_segmentation = self.camera.getBodyTrackingParameters(instance_id).enable_segmentation
        body_params.bodyTrackingParameters.detection_model = self.camera.getBodyTrackingParameters(instance_id).detection_model
        body_params.bodyTrackingParameters.enable_body_fitting = self.camera.getBodyTrackingParameters(instance_id).enable_body_fitting
        body_params.bodyTrackingParameters.body_format = self.camera.getBodyTrackingParameters(instance_id).body_format
        body_params.bodyTrackingParameters.body_selection = self.camera.getBodyTrackingParameters(instance_id).body_selection
        body_params.bodyTrackingParameters.max_range = self.camera.getBodyTrackingParameters(instance_id).max_range
        body_params.bodyTrackingParameters.prediction_timeout_s = self.camera.getBodyTrackingParameters(instance_id).prediction_timeout_s
        body_params.bodyTrackingParameters.allow_reduced_precision_inference = self.camera.getBodyTrackingParameters(instance_id).allow_reduced_precision_inference
        body_params.bodyTrackingParameters.instance_module_id = self.camera.getBodyTrackingParameters(instance_id).instance_module_id
        return body_params
  
    ##
    # Returns the streaming parameters used. Corresponds to the structure sent when the \ref Camera.enable_streaming() function was called.
    # 
    # \return \ref StreamingParameters containing the parameters used for streaming initialization.
    def get_streaming_parameters(self):
        stream = StreamingParameters()
        stream.streaming.codec = self.camera.getStreamingParameters().codec
        stream.streaming.port = self.camera.getStreamingParameters().port
        stream.streaming.bitrate = self.camera.getStreamingParameters().bitrate
        stream.streaming.gop_size = self.camera.getStreamingParameters().gop_size
        stream.streaming.adaptative_bitrate = self.camera.getStreamingParameters().adaptative_bitrate
        stream.streaming.chunk_size = self.camera.getStreamingParameters().chunk_size
        stream.streaming.target_framerate = self.camera.getStreamingParameters().target_framerate
        return stream

    ##
    # Initializes and starts the positional tracking processes.
    #
    # This function allows you to enable the position estimation of the SDK. It only has to be called once in the camera's lifetime.
    #
    # When enabled, the position will be updated at each grab call.
    # Tracking-specific parameters can be set by providing \ref PositionalTrackingParameters to this function.
    #
    # \param py_tracking : structure containing all the \ref PositionalTrackingParameters . default : a preset of \ref PositionalTrackingParameters.
    # \return \ref ERROR_CODE.FAILURE if the \ref area_file_path file wasn't found, \ref ERROR_CODE.SUCCESS otherwise.
    #
    # \warning The positional tracking feature benefits from a high framerate. We found HD720@60fps to be the best compromise between image quality and framerate.
    #
    # \code
    #
    # import pyzed.sl as sl
    # def main() :
    #     # --- Initialize a Camera object and open the ZED
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #
    #     # Set configuration parameters
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode
    #     init_params.camera_fps = 60 # Set fps at 60
    # 
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Set tracking parameters
    #     track_params = sl.PositionalTrackingParameters()
    #     track_params.enable_spatial_memory = True
    #
    #     # Enable positional tracking
    #     err = zed.enable_tracking(track_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print("Tracking error: ", repr(err))
    #         exit(-1)
    #
    #     # --- Main loop
    #     while True :
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
    #             camera_pose = sl.Pose()
    #             zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
    #             print("Camera position: X=", camera_pose.get_translation()[0], " Y=", camera_pose.get_translation()[1], " Z=", camera_pose.get_translation()[2])
    #
    #     # --- Close the Camera
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    #
    # \endcode
    def enable_positional_tracking(self, py_tracking=PositionalTrackingParameters()):
        if isinstance(py_tracking, PositionalTrackingParameters):
            return ERROR_CODE(<int>self.camera.enablePositionalTracking(deref((<PositionalTrackingParameters>py_tracking).tracking)))
        else:
            raise TypeError("Argument is not of PositionalTrackingParameters type.")

    ##
    # Performs a new self calibration process.
    # In some cases, due to temperature changes or strong vibrations, the stereo calibration becomes less accurate.
    # Use this function to update the self-calibration data and get more reliable depth values.
    # \note The self calibration will occur at the next \ref grab() call.
    # \note This function is similar to the previous reset_self_calibration() used in 2.X SDK versions.
    # \warning New values will then be available in \ref get_camera_information(), be sure to get them to still have consistent 2D <-> 3D conversion.
    def update_self_calibration(self):
        self.camera.updateSelfCalibration()

    ##
    # Initializes and starts the Deep Learning detection module.
    #
    # - Human skeleton detection with the \ref DETECTION_MODEL::HUMAN_BODY_FAST or \ref DETECTION_MODEL::HUMAN_BODY_ACCURATE.
    # This model only detects humans but also provides a full skeleton map for each person.
    #
    # Detected objects can be retrieved using the \ref retrieve_bodies() function.
    
    # As detecting and tracking the objects is CPU and GPU-intensive, the module can be used synchronously or asynchronously using \ref BodyTrackingParameters::image_sync.
    # - <b>Synchronous:</b> the \ref retrieve_bodies() function will be blocking during the detection.
    # - <b>Asynchronous:</b> the detection is running in the background, and \ref retrieve_bodies() will immediately return the last objects detected.
    #
    # \note - Only one detection model can be used at the time.
    # \note - <b>This Depth Learning detection module is only available for ZED2 cameras</b>
    # \note - This feature uses AI to locate objects and requires a powerful GPU. A GPU with at least 3GB of memory is recommended.
    #
    # \param object_detection_parameters : Structure containing all specific parameters for object detection.
    # For more information, see the \ref BodyTrackingParameters documentation.
    # \return
    #     - \ref ERROR_CODE::SUCCESS : if everything went fine.
    #     - \ref ERROR_CODE::CORRUPTED_SDK_INSTALLATION : if the AI model is missing or corrupted. In this case, the SDK needs to be reinstalled.
    #     - \ref ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_CAMERA : if the camera used does not have a IMU (ZED Camera). the IMU gives the gravity vector that helps in the 3D box localization. Therefore the Body detection module is available only for ZED-M and ZED2 camera model.
    #     - \ref ERROR_CODE::MOTION_SENSORS_REQUIRED : if the camera model is correct (ZED2) but the IMU is missing. It probably happens because InitParameters::sensors_required was set to false and that IMU has not been found.
    #     - \ref ERROR_CODE::INVALID_FUNCTION_CALL : if one of the BodyTracking parameter is not compatible with other modules parameters (For example, depth mode has been set to NONE).
    #     - \ref ERROR_CODE::FAILURE : otherwise.
    #
    # \code
    #
    # import pyzed.sl as sl
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    # 
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Set tracking parameters
    #     track_params = sl.PositionalTrackingParameters()
    #     track_params.enable_spatial_memory = True
    #
    #     # Set the object detection parameters
    #     object_detection_params = sl.BodyTrackingParameters()
    #     object_detection_params.image_sync = True
    #
    #     # Enable the object detection
    #     err = zed.enable_body_tracking(object_detection_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Grab an image and detect objects on it
    #     objects = sl.Bodies()
    #     while True :
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #             zed.retrieve_bodies(objects)
    #             print(len(objects.object_list), " objects detected\n")
    #             # Use the objects in your application
    # 
    #     # Close the camera
    #     zed.disable_body_tracking()
    #     zed.close()
    #
    # if __name__ == "__main__":
    #     main()
    # \endcode
    def enable_body_tracking(self, body_tracking_parameters : BodyTrackingParameters = BodyTrackingParameters()) -> ERROR_CODE:
        if isinstance(body_tracking_parameters, BodyTrackingParameters):
            return ERROR_CODE(<int>self.camera.enableBodyTracking(deref(body_tracking_parameters.bodyTrackingParameters)))
        else:
            raise TypeError("Argument is not of BodyTrackingParameters type.")

    ##
    # Pauses or resumes the object detection processes.
    #
    # If the object detection has been enabled with  \ref BodyTrackingParameters::image_sync set to false (running asynchronously), this function will pause processing.
    #
    # While in pause, calling this function with <i>status = false</i> will resume the object detection.
    # The \ref retrieveBodies function will keep on returning the last objects detected while in pause.
    #
    #\param status : If true, object detection is paused. If false, object detection is resumed.
    def pause_body_tracking(self, status : bool, instance_id : int = 0):
        return self.camera.pauseBodyTracking(status, instance_id)

    ##
    # Disables the Body Detection process.
    #
    # The object detection module immediately stops and frees its memory allocations.
    # If the object detection has been enabled, this function will automatically be called by \ref close().
    def disable_body_tracking(self, instance_id : int = 0, force_disable_all_instances : bool = False):
        return self.camera.disableBodyTracking(instance_id, force_disable_all_instances)

    ##
    # Retrieve objects detected by the object detection module
    #
    # This function returns the result of the object detection, whether the module is running synchronously or asynchronously.
    #
    # - <b>Asynchronous:</b> this function immediately returns the last objects detected. If the current detection isn't done, the objects from the last detection will be returned, and \ref Bodies::is_new will be set to false.
    # - <b>Synchronous:</b> this function executes detection and waits for it to finish before returning the detected objects.
    #
    # It is recommended to keep the same \ref Bodies object as the input of all calls to this function. This will enable the identification and the tracking of every objects detected.
    #
    # \param objects : The detected objects will be saved into this object. If the object already contains data from a previous detection, it will be updated, keeping a unique ID for the same person.
    # \param parameters : Body detection runtime settings, can be changed at each detection. In async mode, the parameters update is applied on the next iteration.
    #
    # \return \ref SUCCESS if everything went fine, \ref ERROR_CODE::FAILURE otherwise
    #
    # \code
    # objects = sl.Bodies() # Unique Bodies to be updated after each grab
    # --- Main loop
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image from the camera
    #         zed.retrieve_bodies(objects)
    #         for object in objects.object_list:
    #             print(object.label)
    # \endcode
    def retrieve_bodies(self, bodies : Bodies, body_tracking_runtime_parameters : BodyTrackingRuntimeParameters = BodyTrackingRuntimeParameters(), instance_id : int = 0) -> ERROR_CODE:
        return ERROR_CODE(<int>self.camera.retrieveBodies(bodies.bodies, deref(body_tracking_runtime_parameters.body_tracking_rt), instance_id))

    ##
    # Tells if the object detection module is enabled
    def is_body_tracking_enabled(self, instance_id : int = 0):
        return self.camera.isBodyTrackingEnabled(instance_id)

    ##
    # Retrieves the Sensors (IMU,magnetometer,barometer) Data at a specific time reference
    # 
    # Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" gives you the latest sensors data received. Getting all the data requires to call this function at 800Hz in a thread.
    # Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" gives you the sensors data at the time of the latest image \ref grab() "grabbed".
    #
    # \ref SensorsData object contains the previous \ref IMUData structure that was used in ZED SDK v2.X:
    # For IMU data, the values are provided in 2 ways :
    # <b>Time-fused</b> pose estimation that can be accessed using:
    #   <ul><li>\ref data.imu.pose</li>
    # <b>Raw values</b> from the IMU sensor:
    #   <ul><li>\ref data.imu.angular_velocity, corresponding to the gyroscope</li>
    #   <li>\ref data.imu.linear_acceleration, corresponding to the accelerometer</li></ul>
    # both gyroscope and accelerometer are synchronized. The delta time between previous and current values can be calculated using <li>\ref data.imu.timestamp</li>
    #
    # \note The IMU quaternion (fused data) is given in the specified \ref COORDINATE_SYSTEM of \ref InitParameters.
    #
    # \warning In SVO reading mode, the \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" is currently not available (yielding \ref ERROR_CODE.INVALID_FUNCTION_PARAMETERS .
    # * Only the quaternion data and barometer data (if available) at \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" are available. Other values will be set to 0.
    #
    def get_sensors_data(self, py_sensors_data: SensorsData, time_reference = TIME_REFERENCE.CURRENT):
        if isinstance(time_reference, TIME_REFERENCE):
            return ERROR_CODE(<int>self.camera.getSensorsData(py_sensors_data.sensorsData, <c_TIME_REFERENCE>(<unsigned int>time_reference.value)))
        else:
            raise TypeError("Argument is not of TIME_REFERENCE type.")

    ##
    # Set an optional IMU orientation hint that will be used to assist the tracking during the next \ref grab().
    # 
    # This function can be used to assist the positional tracking rotation while using a ZED Mini.
    # 
    # \note This function is only effective if a ZED Mini (ZED-M) is used.
    # 
    # It needs to be called before the \ref grab() function.
    # \param transform : \ref Transform to be ingested into IMU fusion. Note that only the rotation is used.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS"  if the transform has been passed, \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" otherwise (e.g. when used with a ZED camera which doesn't have IMU data).
    def set_imu_prior(self, transfom: Transform):
        return ERROR_CODE(<int>self.camera.setIMUPrior(transfom.transform[0]))

    ##
    # Retrieves the estimated position and orientation of the camera in the specified \ref REFERENCE_FRAME "reference frame".
    #
    # Using \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD", the returned pose relates to the initial position of the camera.  (\ref PositionalTrackingParameters.initial_world_transform )
    # Using \ref REFERENCE_FRAME "REFERENCE_FRAME.CAMERA", the returned pose relates to the previous position of the camera.
    #
    # If the tracking has been initialized with \ref PositionalTrackingParameters.enable_area_memory to true (default), this function can return \ref POSITIONAL_TRACKING_STATE "POSITIONAL_TRACKING_STATE::SEARCHING".
    # This means that the tracking lost its link to the initial referential and is currently trying to relocate the camera. However, it will keep on providing position estimations.
    # 
    # \param camera_pose \b [out]: the pose containing the position of the camera and other information (timestamp, confidence)
    # \param reference_frame : defines the reference from which you want the pose to be expressed. Default : \ref REFERENCE_FRAME "REFERENCE_FRAME::WORLD".
    # \return The current \ref POSITIONAL_TRACKING_STATE "state" of the tracking process.
    #
    # \n Extract Rotation Matrix : camera_pose.get_rotation()
    # \n Extract Translation Vector : camera_pose.get_translation()
    # \n Extract Orientation / quaternion : camera_pose.get_orientation()
    #
    # \code
    # while True :
    # if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
    #     camera_pose = sl.Pose()
    #     zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
    #
    #     print("Camera position: X=", camera_pose.get_translation().[0], " Y=", camera_pose.get_translation()[1], " Z=", camera_pose.get_translation()[2])
    #     print("Camera Euler rotation: X=", camera_pose.get_euler_angles()[0], " Y=", camera_pose.get_euler_angles()[1], " Z=", camera_pose.get_euler_angles()[2])
    #     print("Camera Rodrigues rotation: X=", camera_pose.get_rotation_vector()[0], " Y=", camera_pose.get_rotation_vector()[1], " Z=", camera_pose.get_rotation_vector()[2])
    #     print("Camera quaternion orientation: X=", camera_pose.get_orientation()[0], " Y=", camera_pose.get_orientation()[1], " Z=", camera_pose.get_orientation()[2], " W=", camera_pose.get_orientation()[3])
    # \endcode
    def get_position(self, py_pose: Pose, reference_frame = REFERENCE_FRAME.WORLD):
        if isinstance(reference_frame, REFERENCE_FRAME):
            return POSITIONAL_TRACKING_STATE(<int>self.camera.getPosition(py_pose.pose, <c_REFERENCE_FRAME>(<unsigned int>reference_frame.value)))
        else:
            raise TypeError("Argument is not of REFERENCE_FRAME type.")

    ##
    # Returns the state of the spatial memory export process.
    #
    # As \ref Camera.save_area_map() only starts the exportation, this function allows you to know when the exportation finished or if it failed.
    # \return The current \ref AREA_EXPORTING_STATE "state" of the spatial memory export process.
    def get_area_export_state(self):
        return AREA_EXPORTING_STATE(<int>self.camera.getAreaExportState())

    ##
    # Saves the current area learning file. The file will contain spatial memory data generated by the tracking.
    #
    # If the tracking has been initialized with \ref PositionalTrackingParameters.enable_area_memory to true (default), the function allows you to export the spatial memory.
    # Reloading the exported file in a future session with \ref PositionalTrackingParameters.area_file_path initializes the tracking within the same referential.
    # This function is asynchronous, and only triggers the file generation. You can use \ref get_area_export_state() to get the export state.
    # The positional tracking keeps running while exporting.
    #
    # \param area_file_path : saves the spatial memory database in an '.area' file.
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if the \ref area_file_path file wasn't found, \ref ERROR_CODE.SUCCESS otherwise.
    # 
    # See \ref get_area_export_state()
    #
    # \note Please note that this function will also flush the area database that was built / loaded.
    # 
    # \warning If the camera wasn't moved during the tracking session, or not enough, the spatial memory won't be usable and the file won't be exported.
    # The \ref get_area_export_state() will return \ref AREA_EXPORTING_STATE "AREA_EXPORTING_STATE.NOT_STARTED"
    # A few meters (~3m) of translation or a full rotation should be enough to get usable spatial memory.
    # However, as it should be used for relocation purposes, visiting a significant portion of the environment is recommended before exporting.
    #
    # \code
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS : # Grab an image and computes the tracking
    #         camera_pose = Pose()
    #         zed.get_position(camera_pose, REFERENCE_FRAME.WORLD)
    #
    # # Export the spatial memory for future sessions
    # zed.save_area_map("office.area") # The actual file will be created asynchronously.
    # print(repr(zed.get_area_export_state()))
    #
    # # Close the camera
    # zed.close()
    # \endcode
    def save_area_map(self, area_file_path=""):
        filename = (<str>area_file_path).encode()
        return ERROR_CODE(<int>self.camera.saveAreaMap(String(<char*>filename)))

    ##
    # Disables the positional tracking.
    #
    # The positional tracking is immediately stopped. If a file path is given, \ref save_area_map() will be called asynchronously. See \ref get_area_export_state() to get the exportation state.
    # If the tracking has been enabled, this function will automatically be called by \ref close() .
    # 
    # \param area_file_path : if set, saves the spatial memory into an '.area' file. default : (empty)
    # \n area_file_path is the name and path of the database, e.g. "path/to/file/myArea1.area".
    #
    def disable_positional_tracking(self, area_file_path=""):
        filename = (<str>area_file_path).encode()
        self.camera.disablePositionalTracking(String(<char*> filename))
    
    ##
    # Tells if the tracking module is enabled.
    def is_positional_tracking_enabled(self):
        return self.camera.isPositionalTrackingEnabled()

    ##
    # Resets the tracking, and re-initializes the position with the given transformation matrix.
    # \param path : Position of the camera in the world frame when the function is called. By default, it is set to identity.
    # \return \ref ERROR_CODE.SUCCESS if the tracking has been reset, ERROR_CODE.FAILURE otherwise.
    #
    # \note Please note that this function will also flush the accumulated or loaded spatial memory.
    def reset_positional_tracking(self, path: Transform):
        return ERROR_CODE(<int>self.camera.resetPositionalTracking(path.transform[0]))

    ##
    # Initializes and starts the spatial mapping processes.
    #
    # The spatial mapping will create a geometric representation of the scene based on both tracking data and 3D point clouds.
    # The resulting output can be a \ref Mesh or a \ref FusedPointCloud. It can be be obtained by calling \ref extract_whole_spatial_map() or \ref retrieve_spatial_map_async().
    # Note that \ref retrieve_spatial_map_async should be called after \ref request_spatial_map_async().
    # 
    # \param py_spatial : the structure containing all the specific parameters for the spatial mapping.
    # Default: a balanced parameter preset between geometric fidelity and output file size. For more information, see the \ref SpatialMappingParameters documentation.
    # \return \ref ERROR_CODE.SUCCESS if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \warning The tracking (\ref enable_positional_tracking() ) and the depth (\ref RuntimeParameters.enable_depth ) needs to be enabled to use the spatial mapping.
    # \warning The performance greatly depends on the spatial_mapping_parameters.
    # \warning Lower SpatialMappingParameters.range_meter and SpatialMappingParameters.resolution_meter for higher performance.
    # If the mapping framerate is too slow in live mode, consider using an SVO file, or choose a lower mesh resolution.
    #
    # \note This features uses host memory (RAM) to store the 3D map. The maximum amount of available memory allowed can be tweaked using the SpatialMappingParameters.
    # Exceeding the maximum memory allowed immediately stops the mapping.
    #
    # \code
    # import pyzed.sl as sl
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #
    #     # Set initial parameters
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
    #     init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system (The OpenGL one)
    #     init_params.coordinate_units = sl.UNIT.METER # Set units in meters
    #
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         exit(-1)
    #
    #     # Positional tracking needs to be enabled before using spatial mapping
    #     tracking_parameters sl.PositionalTrackingParameters()
    #     err = zed.enable_tracking(tracking_parameters)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         exit(-1)
    #
    #     # Enable spatial mapping
    #     mapping_parameters sl.SpatialMappingParameters()
    #     err = zed.enable_spatial_mapping(mapping_parameters)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         exit(-1)
    #
    #     # Grab data during 500 frames
    #     i = 0
    #     mesh = sl.Mesh() # Create a mesh object
    #     while i < 500 :
    #     # For each new grab, mesh data is updated
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #         # In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
    #         mapping_state = zed.get_spatial_mapping_state()
    #
    #         # Print spatial mapping state
    #         print("Images captured: ", i << " / 500  ||  Spatial mapping state: ", repr(mapping_state))
    #         i = i + 1
    #
    #     # Extract, filter and save the mesh in a obj file
    #     print("Extracting Mesh ...")
    #     zed.extract_whole_spatial_map(mesh) # Extract the whole mesh
    #     print("Filtering Mesh ...")
    #     mesh.filter(sl.MESH_FILTER.LOW) # Filter the mesh (remove unnecessary vertices and faces)
    #     print("Saving Mesh in mesh.obj ...")
    #     mesh.save("mesh.obj") # Save the mesh in an obj file
    #
    #     # Disable tracking and mapping and close the camera
    #     zed.disable_spatial_mapping()
    #     zed.disable_tracking()
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    # \endcode
    def enable_spatial_mapping(self, py_spatial=SpatialMappingParameters()):
        if isinstance(py_spatial, SpatialMappingParameters):
            return ERROR_CODE(<int>self.camera.enableSpatialMapping(deref((<SpatialMappingParameters>py_spatial).spatial)))
        else:
            raise TypeError("SpatialMappingParameters must be initialized first with SpatialMappingParameters()")

    ##
    # Pauses or resumes the spatial mapping processes.
    # 
    # As spatial mapping runs asynchronously, using this function can pause its computation to free some processing power, and resume it again later.
    # For example, it can be used to avoid mapping a specific area or to pause the mapping when the camera is static.
    # \param status : if true, the integration is paused. If false, the spatial mapping is resumed.
    def pause_spatial_mapping(self, status: bool):
        if isinstance(status, bool):
            self.camera.pauseSpatialMapping(status)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Returns the current spatial mapping state.
    #
    # As the spatial mapping runs asynchronously, this function allows you to get reported errors or status info.
    # \return The current state of the spatial mapping process
    # 
    # See also \ref SPATIAL_MAPPING_STATE
    def get_spatial_mapping_state(self):
        return SPATIAL_MAPPING_STATE(<int>self.camera.getSpatialMappingState())

    ##
    # Starts the spatial map generation process in a non blocking thread from the spatial mapping process.
    # 
    # The spatial map generation can take a long time depending on the mapping resolution and covered area. This function will trigger the generation of a mesh without blocking the program.
    # You can get info about the current generation using \ref get_spatial_map_request_status_async(), and retrieve the mesh using \ref retrieve_spatial_map_async().
    #
    # \note Only one mesh can be generated at a time. If the previous mesh generation is not over, new calls of the function will be ignored.
    # 
    #
    # \code
    # cam.request_spatial_map_async()
    # while cam.get_spatial_map_request_status_async() == sl.ERROR_CODE.FAILURE :
    #     # Mesh is generating
    #
    # mesh = sl.Mesh()
    # if cam.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS :
    #     cam.retrieve_spatial_map_async(mesh)
    #     nb_triangles = mesh.get_number_of_triangles()
    #     print("Number of triangles in the mesh: ", mesh)
    # \endcode
    def request_spatial_map_async(self):
        self.camera.requestSpatialMapAsync()

    ##
    # Returns the spatial map generation status. This status allows to know if the mesh can be retrieved by calling \ref retrieve_spatial_map_async()
    # \return \ref ERROR_CODE.SUCCESS if the mesh is ready and not yet retrieved, otherwise \ref ERROR_CODE.FAILURE
    def get_spatial_map_request_status_async(self):
        return ERROR_CODE(<int>self.camera.getSpatialMapRequestStatusAsync())

    ##
    # Retrieves the current generated spatial map.
    # 
    # After calling \ref retrieve_spatial_map_async() , this function allows you to retrieve the generated mesh or fused point cloud. The \ref Mesh or \ref FusedPointCloud will only be available when \ref get_spatial_map_request_status_async() returns \ref ERROR_CODE.SUCCESS
    #
    # \param py_mesh : \b [out] The \ref Mesh or \ref FusedPointCloud to be filled with the generated spatial map.
    # \return \ref ERROR_CODE.SUCCESS if the mesh is retrieved, otherwise \ref ERROR_CODE.FAILURE
    # 
    # \note This function only updates the necessary chunks and adds the new ones in order to improve update speed.
    # \warning You should not modify the mesh / fused point cloud between two calls of this function, otherwise it can lead to corrupted mesh / fused point cloud .
    #
    # See \ref request_spatial_map_async() for an example.
    def retrieve_spatial_map_async(self, py_mesh):
        if isinstance(py_mesh, Mesh) :
            return ERROR_CODE(<int>self.camera.retrieveSpatialMapAsync(deref((<Mesh>py_mesh).mesh)))
        elif isinstance(py_mesh, FusedPointCloud) :
            py_mesh = <FusedPointCloud> py_mesh
            return ERROR_CODE(<int>self.camera.retrieveSpatialMapAsync(deref((<FusedPointCloud>py_mesh).fpc)))
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    ##
    # Extracts the current spatial map from the spatial mapping process.
    #
    # If the object to be filled already contains a previous version of the mesh / fused point cloud, only changes will be updated, optimizing performance.
    #
    # \param py_mesh : \b [out] The \ref Mesh or \ref FuesedPointCloud to be filled with the generated spatial map.
    #
    # \return \ref ERROR_CODE.SUCCESS if the mesh is filled and available, otherwise \ref ERROR_CODE.FAILURE
    #
    # \warning This is a blocking function. You should either call it in a thread or at the end of the mapping process.
    # The extraction can be long, calling this function in the grab loop will block the depth and tracking computation giving bad results.
    def extract_whole_spatial_map(self, py_mesh):
        if isinstance(py_mesh, Mesh) :
            return ERROR_CODE(<int>self.camera.extractWholeSpatialMap(deref((<Mesh>py_mesh).mesh)))
        elif isinstance(py_mesh, FusedPointCloud) :
            return ERROR_CODE(<int>self.camera.extractWholeSpatialMap(deref((<FusedPointCloud>py_mesh).fpc)))
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    ##
    # Checks the plane at the given left image coordinates.
    # 
    # This function gives the 3D plane corresponding to a given pixel in the latest left image \ref grab() "grabbed".
    # The pixel coordinates are expected to be contained between 0 and \ref CameraInformations.camera_resolution "get_camera_information().camera_resolution.width-1" and  \ref CameraInformations.camera_resolution "get_camera_information().camera_resolution.height-1"
    # 
    # \param coord :  \b [in] The image coordinate. The coordinate must be taken from the full-size image
    # \param plane : \b [out] The detected plane if the function succeeded
    # \return \ref ERROR_CODE.SUCCESS if a plane is found otherwise \ref ERROR_CODE.PLANE_NOT_FOUND 
    #
    # \note The reference frame is defined by the \ref RuntimeParameters.measure3D_reference_frame given to the \ref grab() function.
    def find_plane_at_hit(self, coord, py_plane: Plane):
        cdef Vector2[uint] vec = Vector2[uint](coord[0], coord[1])
        return ERROR_CODE(<int>self.camera.findPlaneAtHit(vec, py_plane.plane))

    ##
    # Detect the floor plane of the scene.
    # 
    # This function analyses the latest image and depth to estimate the floor plane of the scene.
    # 
    # It expects the floor plane to be visible and bigger than other candidate planes, like a table.
    # 
    # \param py_plane : \b [out] The detected floor plane if the function succeeded
    # \param resetTrackingFloorFrame : \b [out] The transform to align the tracking with the floor plane. The initial position will then be at ground height, with the axis align with the gravity. The positional tracking needs to be reset/enabled
    # \param floor_height_prior : \b [in] Prior set to locate the floor plane depending on the known camera distance to the ground, expressed in the same unit as the ZED. If the prior is too far from the detected floor plane, the function will return \ref ERROR_CODE.PLANE_NOT_FOUND
    # \param world_orientation_prior : \b [in] Prior set to locate the floor plane depending on the known camera  orientation to the ground. If the prior is too far from the detected floor plane, the function will return \ref ERROR_CODE.PLANE_NOT_FOUND
    # \param floor_height_prior_tolerance : \b [in] Prior height tolerance, absolute value.
    # \return \ref ERROR_CODE.SUCCESS if the floor plane is found and matches the priors (if defined), otherwise \ref ERROR_CODE.PLANE_NOT_FOUND
    #
    # \note The reference frame is defined by the \ref RuntimeParameters (measure3D_reference_frame) given to the \ref grab() function. The length unit is defined by \ref InitParameters (coordinate_units). With the ZED, the assumption is made that the floor plane is the dominant plane in the scene. The ZED Mini uses the gravity as prior.
    #
    def find_floor_plane(self, py_plane: Plane, resetTrackingFloorFrame: Transform, floor_height_prior = float('nan'), world_orientation_prior = Rotation(Matrix3f().zeros()), floor_height_prior_tolerance = float('nan')) :
        return ERROR_CODE(<int>self.camera.findFloorPlane(py_plane.plane, resetTrackingFloorFrame.transform[0], floor_height_prior, (<Rotation>world_orientation_prior).rotation[0], floor_height_prior_tolerance))

    ##
    # Disables the spatial mapping process.
    # The spatial mapping is immediately stopped.
    # If the mapping has been enabled, this function will automatically be called by \ref close() .
    # \note This function frees the memory allocated for th spatial mapping, consequently, mesh cannot be retrieved after this call.
    def disable_spatial_mapping(self):
        self.camera.disableSpatialMapping()


    ##
    # Creates a streaming pipeline for images.
    # \param streaming_parameters : the structure containing all the specific parameters for the streaming.
    #
    # \code
    # import pyzed.sl as sl
    #
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #
    #     # Set initial parameters
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
    #
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #        print(repr(err))
    #        exit(-1)
    #
    #     # Enable streaming
    #     stream_params = sl.StreamingParameters()
    #     stream_params.port = 30000
    #     stream_params.bitrate = 8000
    #     err = zed.enable_streaming(stream_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Grab data during 500 frames
    #     i = 0
    #     while i < 500 :
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #             i = i+1
    #
    #     zed.disable_streaming()
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()  
    # \endcode
    #
    # \return an \ref ERROR_CODE that defines if the stream was started.
    # \n Possible Error Code :
    # \n - \ref ERROR_CODE.SUCCESS if the streaming was successfully started
    # \n - \ref ERROR_CODE.INVALID_FUNCTION_CALL if open() was not successfully called before.
    # \n - \ref ERROR_CODE.FAILURE if streaming RTSP protocol was not able to start.
    # \n - \ref ERROR_CODE.NO_GPU_COMPATIBLE if streaming codec is not supported (in this case, use H264 codec).
    def enable_streaming(self, streaming_parameters = StreamingParameters()) :
        return ERROR_CODE(<int>self.camera.enableStreaming(deref((<StreamingParameters>streaming_parameters).streaming)))

    ##
    # Disables the streaming initiated by \ref enable_straming()
    # \note This function will automatically be called by \ref close() if enable_streaming() was called.
    #
    # See \ref enable_streaming() for an example.
    def disable_streaming(self):
        self.camera.disableStreaming()

    ##
    # Tells if the streaming is actually sending data (true) or still in configuration (false)
    def is_streaming_enabled(self):
        return self.camera.isStreamingEnabled()


    ##
    # Creates an SVO file to be filled by \ref record().
    # 
    # SVO files are custom video files containing the un-rectified images from the camera along with some meta-data like timestamps or IMU orientation (if applicable).
    # They can be used to simulate a live ZED and test a sequence with various SDK parameters.
    # Depending on the application, various compression modes are available. See \ref SVO_COMPRESSION_MODE.
    # 
    # \param record : \ref RecordingParameters such as filename and compression mode
    # 
    # \return an \ref ERROR_CODE that defines if SVO file was successfully created and can be filled with images.
    # 
    # \warning This function can be called multiple times during ZED lifetime, but if video_filename is already existing, the file will be erased.
    #
    # 
    # \code
    # import pyzed.sl as sl
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #     # Set initial parameters
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
    #     init_params.coordinate_units = sl.UNIT.METER # Set units in meters
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if (err != sl.ERROR_CODE.SUCCESS) :
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Enable video recording
    #     record_params = RecordingParameters("myVideoFile.svo, sl.SVO_COMPRESSION_MODE.HD264)
    #     err = zed.enable_recording(record_params)
    #     if (err != sl.ERROR_CODE.SUCCESS) :
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Grab data during 500 frames
    #     i = 0
    #     while i < 500 :
    #         # Grab a new frame
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #             # Record the grabbed frame in the video file
    #             i = i + 1
    # 
    #     zed.disable_recording()
    #     print("Video has been saved ...")
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    # \endcode
    def enable_recording(self, record: RecordingParameters):
        if isinstance(record, RecordingParameters):
            return ERROR_CODE(<int>self.camera.enableRecording(deref(record.record)))
        else:
            raise TypeError("Argument is not of RecordingParameters type.")

    ##
    # Disables the recording initiated by \ref enable_recording() and closes the generated file.
    #
    # \note This function will automatically be called by \ref close() if \ref enable_recording() was called.
    # 
    # See \ref enable_recording() for an example.
    def disable_recording(self):
        self.camera.disableRecording()

    ##
    # Get the recording information
    # \return The recording state structure. For more details, see \ref RecordingStatus.
    def get_recording_status(self):
        state = RecordingStatus()
        state.is_recording = self.camera.getRecordingStatus().is_recording
        state.is_paused = self.camera.getRecordingStatus().is_paused
        state.status = self.camera.getRecordingStatus().status
        state.current_compression_time = self.camera.getRecordingStatus().current_compression_time
        state.current_compression_ratio = self.camera.getRecordingStatus().current_compression_ratio
        state.average_compression_time = self.camera.getRecordingStatus().average_compression_time
        state.average_compression_ratio = self.camera.getRecordingStatus().average_compression_ratio
        return state

    ##
    # Pauses or resumes the recording.
    # \param status : if true, the recording is paused. If false, the recording is resumed.
    def pause_recording(self, value=True):
        self.camera.pauseRecording(value)

    ##
    # Returns the recording parameters used. Corresponds to the structure sent when the \ref enable_recording() function was called
    #
    # \return \ref RecordingParameters containing the parameters used for streaming initialization.
    def get_recording_parameters(self):
        param = RecordingParameters()
        param.record.video_filename = self.camera.getRecordingParameters().video_filename
        param.record.compression_mode = self.camera.getRecordingParameters().compression_mode
        param.record.target_framerate = self.camera.getRecordingParameters().target_framerate
        param.record.bitrate = self.camera.getRecordingParameters().bitrate
        param.record.transcode_streaming_input = self.camera.getRecordingParameters().transcode_streaming_input
        return param

    ##
    # Initializes and starts the object detection module.
    # 
    # The object detection module will detect and track objects, people or animals in range of the camera, the full list of detectable objects is available in \ref OBJECT_CLASS.
    # 
    # Detected objects can be retrieved using the \ref retrieve_objects() function.
    #
    # As detecting and tracking the objects is CPU and GPU-intensive, the module can be used synchronously or asynchronously using \ref ObjectDetectionParameters.image_sync .
    # - <b>Synchronous:</b> the \ref retrieve_objects() function will be blocking during the detection.
    # - <b>Asynchronous:</b> the detection is running in the background, and \ref retrieve_objects() will immediately return the last objects detected.
    # 
    # \param object_detection_parameters : Structure containing all specific parameters for object detection.
    # 
    # For more information, see the \ref ObjectDetectionParameters documentation
    # \return
    # \ref ERROR_CODE.SUCCESS if everything went fine
    # \ref ERROR_CODE.OBJECT_DETECTION_NOT_AVAILABLE if the AI model is missing or corrupted. In this case, the SDK needs to be reinstalled
    # \ref ERROR_CODE.OBJECT_DETECTION_MODULE_NOT_COMPATIBLE_WITH_CAMERA if the camera used does not have a IMU (ZED Camera). the IMU gives the gravity vector that helps in the 3D box localization. Therefore the Object detection module is available only for ZED-M and ZED2 camera models.
    # \ref ERROR_CODE.SENSORS_NOT_DETECTED if the camera model is correct (ZED-M or ZED2) but the IMU is missing. It probably happens because \ref InitParameters.sensors_required was set to true
    # \ref ERROR_CODE.INVALID_FUNCTION_CALL if one of the  \ref ObjectDetection parameter is not compatible with other modules parameters (For example, depth mode has been set to NONE).
    # \ref ERROR_CODE.FAILURE otherwise.
    #
    # \note This feature uses AI to locate objects and requires a powerful GPU. A GPU with at least 3GB of memory is recommended.
    #
    # \code
    # import pyzed.sl as sl
    #
    # def main():
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    # 
    #     # Open the camera
    #     err = zed.open()
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Set the object detection parameters
    #     object_detection_params = sl.ObjectDetectionParameters()
    #     object_detection_params.image_sync = True
    #
    #     # Enable the object detection
    #     err = zed.enable_object_detection(object_detection_params)
    #     if err != sl.ERROR_CODE.SUCCESS :
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Grab an image and detect objects on it
    #     objects = sl.Objects()
    #     while True :
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #             zed.retrieve_objects(objects)
    #             print(len(objects.object_list), " objects detected\n")
    #             # Use the objects in your application
    # 
    #     # Close the camera
    #     zed.disable_object_detection()
    #     zed.close()
    #
    # if __name__ == "__main__":
    #     main()
    # \endcode
    def enable_object_detection(self, object_detection_parameters = ObjectDetectionParameters()) :
        if isinstance(object_detection_parameters, ObjectDetectionParameters):
            return ERROR_CODE(<int>self.camera.enableObjectDetection(deref((<ObjectDetectionParameters>object_detection_parameters).object_detection)))
        else:
            raise TypeError("Argument is not of ObjectDetectionParameters type.")

    ##
    # Disables the Object Detection process.
    #
    # The object detection module immediately stops and frees its memory allocations.
    # If the object detection has been enabled, this function will automatically be called by \ref close().
    def disable_object_detection(self, instance_module_id=0):
        self.camera.disableObjectDetection(instance_module_id)

    ##
    # Pauses or resumes the object detection processes.
    #
    # If the object detection has been enabled with  \ref ObjectDetectionParameters.image_sync set to false (running asynchronously), this function will pause processing.
    # While in pause, calling this function with <i>status = false</i> will resume the object detection.
    # The \ref retrieve_objects function will keep on returning the last objects detected while in pause.
    #
    # \param status : If true, object detection is paused. If false, object detection is resumed.
    def pause_object_detection(self, status: bool, instance_module_id=0):
        if isinstance(status, bool):
            self.camera.pauseObjectDetection(status, instance_module_id)
        else:
            raise TypeError("Argument is not of boolean type.")


    ##
    # Retrieve objects detected by the object detection module.
    #
    # This function returns the result of the object detection, whether the module is running synchronously or asynchronously.
    #
    # - <b>Asynchronous:</b> this function immediately returns the last objects detected. If the current detection isn't done, the objects from the last detection will be returned, and \ref Objects::is_new will be set to false.
    # - <b>Synchronous:</b> this function executes detection and waits for it to finish before returning the detected objects.
    #
    # It is recommended to keep the same \ref Objects object as the input of all calls to this function. This will enable the identification and the tracking of every objects detected.
    # 
    # \param py_objects : [in,out] The detected objects will be saved into this object. If the object already contains data from a previous detection, it will be updated, keeping a unique ID for the same person.
    # \param object_detection_parameters : [in] Object detection runtime settings, can be changed at each detection. In async mode, the parameters update is applied on the next iteration.
    # \return \ref ERROR_CODE.SUCCESS if everything went fine, \ref ERROR_CODE.FAILURE otherwise
    #
    # \code
    # objects = sl.Objects()
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #         zed.retrieve_objects(objects)
    #         object_list = objects.object_list
    #         for i in range(len(object_list)) :
    #             print(repr(object_list[i].label))
    # \endcode
    def retrieve_objects(self, py_objects: Objects, object_detection_parameters=ObjectDetectionRuntimeParameters(), instance_module_id=0):
        if isinstance(py_objects, Objects) :
            return ERROR_CODE(<int>self.camera.retrieveObjects((<Objects>py_objects).objects, deref((<ObjectDetectionRuntimeParameters>object_detection_parameters).object_detection_rt), instance_module_id))
        else :
           raise TypeError("Argument is not of Objects type.") 

    ##
    # Get a batch of detected objects.
    # \warning This function needs to be called after \ref retrieve_objects, otherwise trajectories will be empty.
    # It is the \ref retrieve_objects function that ingests the current/live objects into the batching queue.
    # \param trajectories : list of \ref ObjectsBatch that will be filled by the batching queue process. An empty list should be passed to the function
    # \return [ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went fine, [ERROR_CODE.INVALID_FUNCTION_CALL](\ref ERROR_CODE) if batching module is not available (TensorRT!=7.1) or if object tracking was not enabled.
    # 
    # \code
    # objects = sl.Objects()                                        # Unique Objects to be updated after each grab
    # while True:                                                   # Main loop
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:                   # Grab an image from the camera
    #         zed.retrieve_objects(objects)                         # Call retrieve_objects so that objects are ingested in the batching system
    #         trajectories = []                                     # Create an empty list of trajectories 
    #         zed.get_objects_batch(trajectories)                   # Get batch of objects
    #         print("Size of batch : {}".format(len(trajectories)))
    # \endcode
    def get_objects_batch(self, trajectories: list[ObjectsBatch], instance_module_id=0):
        cdef vector[c_ObjectsBatch] output_trajectories
        if trajectories is not None:
            status = self.camera.getObjectsBatch(output_trajectories, instance_module_id)
            for trajectory in output_trajectories:
                curr = ObjectsBatch()
                curr.objects_batch = trajectory
                trajectories.append(curr)
            return ERROR_CODE(<int>status)
        else:
            raise TypeError("Argument is not of the right type")

    ##
    # Feed the 3D Object tracking function with your own 2D bounding boxes from your own detection algorithm.
    # \param objects_in : list of \ref CustomBoxObjectData.
    # \return [ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went fine
    # \note The detection should be done on the current grabbed left image as the internal process will use all current available data to extract 3D information and perform object tracking.
    def ingest_custom_box_objects(self, objects_in: list[CustomBoxObjectData], instance_module_id=0):
        cdef vector[c_CustomBoxObjectData] custom_obj
        if objects_in is not None:
            # Convert input list into C vector
            for i in range(len(objects_in)):
                custom_obj.push_back((<CustomBoxObjectData>objects_in[i]).custom_box_object_data) 
            status = self.camera.ingestCustomBoxObjects(custom_obj, instance_module_id)
            return ERROR_CODE(<int>status)
        else:
            raise TypeError("Argument is not of the right type")

    ##
    # Returns the version of the currently installed ZED SDK.
    @staticmethod
    def get_sdk_version():
        cls = Camera()
        return to_str(cls.camera.getSDKVersion()).decode()

    ##
    # Lists all the connected devices with their associated information.
    # This function lists all the cameras available and provides their serial number, models and other information.
    # \return The device properties for each connected camera
    @staticmethod
    def get_device_list():
        cls = Camera()
        vect_ = cls.camera.getDeviceList()
        vect_python = []
        for i in range(vect_.size()):
            prop = DeviceProperties()
            prop.camera_state = CAMERA_STATE(<int> vect_[i].camera_state)
            prop.id = vect_[i].id
            prop.path = vect_[i].path.get().decode()
            prop.camera_model = MODEL(<int>vect_[i].camera_model)
            prop.serial_number = vect_[i].serial_number
            vect_python.append(prop)
        return vect_python

    ##
    # Lists all the streaming devices with their associated information.
    # 
    # \return The streaming properties for each connected camera
    #
    # \warning As this function returns an std::vector, it is only safe to use in Release mode (not Debug).
    # This is due to a known compatibility issue between release (the SDK) and debug (your app) implementations of std::vector.
    @staticmethod
    def get_streaming_device_list():
        cls = Camera()
        vect_ = cls.camera.getStreamingDeviceList()
        vect_python = []
        for i in range(vect_.size()):
            prop = StreamingProperties()
            prop.ip = vect_[i].ip.get().decode()
            prop.port = vect_[i].port
            prop.serial_number = vect_[i].serial_number
            prop.current_bitrate = vect_[i].current_bitrate
            prop.codec = STREAMING_CODEC(<int>vect_[i].codec)
            vect_python.append(prop)
        return vect_python

    ##
    # Performs an hardware reset of the ZED 2.
    # 
    # \param sn : Serial number of the camera to reset, or 0 to reset the first camera detected.
    # \param fullReboot : If set to True, performs a full reboot (Sensors and Video modules). Default: True
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.CAMERA_NOT_DETECTED" if no camera was detected, \ref ERROR_CODE "ERROR_CODE.FAILURE"  otherwise.
    #
    # \note This function only works for ZED2 and ZED2i cameras.
    # 
    # \warning This function will invalidate any sl.Camera object, since the device is rebooting.
    @staticmethod
    def reboot(sn : int, fullReboot: bool =True):
        cls = Camera()
        return ERROR_CODE(<int>cls.camera.reboot(sn, fullReboot))

##
# Lists the types of communications available for Fusion app.
# \ingroup Fusion_group
class COMM_TYPE(enum.Enum):
    LOCAL_NETWORK = <int>c_COMM_TYPE.LOCAL_NETWORK
    INTRA_PROCESS = <int>c_COMM_TYPE.INTRA_PROCESS
    LAST = <int>c_COMM_TYPE.LAST

##
# Lists the types of error that can be raised by the Fusion
#
# \ingroup Fusion_group
# 
# | Enumerator     |                  |
# |----------------|------------------|
# | WRONG_BODY_FORMAT | Senders use different body format, consider to change them. |
# | NOT_ENABLE | The following module was not enabled |
# | INPUT_FEED_MISMATCH | Some source are provided by SVO and some sources are provided by LIVE stream |
# | CONNECTION_TIMED_OUT | Connection timed out ... impossible to reach the sender... this may be due to ZED Hub absence |
# | SHARED_MEMORY_LEAK | Detect multiple instance of SHARED_MEMORY communicator ... only one is authorized |
# | BAD_IP_ADDRESS | The IP format provided is wrong, please provide IP in this format a.b.c.d where (a, b, c, d) are numbers between 0 and 255. |
# | CONNECTION_ERROR | Something goes bad in the connection between sender and receiver. |
# | FAILURE | Standard code for unsuccessful behavior. |
# | SUCCESS |  |
# | FUSION_ERRATIC_FPS | Some big differences has been observed between senders FPS |
# | FUSION_FPS_TOO_LOW | At least one sender has fps lower than 10 FPS |
class FUSION_ERROR_CODE(enum.Enum):
    WRONG_BODY_FORMAT = <int>c_FUSION_ERROR_CODE.WRONG_BODY_FORMAT
    NOT_ENABLE = <int>c_FUSION_ERROR_CODE.NOT_ENABLE
    INPUT_FEED_MISMATCH = <int>c_FUSION_ERROR_CODE.INPUT_FEED_MISMATCH
    CONNECTION_TIMED_OUT = <int>c_FUSION_ERROR_CODE.CONNECTION_TIMED_OUT
    MEMORY_ALREADY_USED = <int>c_FUSION_ERROR_CODE.MEMORY_ALREADY_USED
    BAD_IP_ADDRESS = <int>c_FUSION_ERROR_CODE.BAD_IP_ADDRESS
    FAILURE = <int>c_FUSION_ERROR_CODE.FAILURE
    SUCCESS = <int>c_FUSION_ERROR_CODE.SUCCESS
    FUSION_ERRATIC_FPS = <int>c_FUSION_ERROR_CODE.FUSION_ERRATIC_FPS
    FUSION_FPS_TOO_LOW = <int>c_FUSION_ERROR_CODE.FUSION_FPS_TOO_LOW
    NO_NEW_DATA_AVAILABLE = <int>c_FUSION_ERROR_CODE.NO_NEW_DATA_AVAILABLE
    INVALID_TIMESTAMP = <int>c_FUSION_ERROR_CODE.INVALID_TIMESTAMP
    INVALID_COVARIANCE = <int>c_FUSION_ERROR_CODE.INVALID_COVARIANCE
    
    def __str__(self):
        return to_str(toString(<c_FUSION_ERROR_CODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_FUSION_ERROR_CODE>(<int>self.value))).decode()

##
# Lists the types of error that can be raised during the Fusion by senders
#
# \ingroup Fusion_group
# 
# | Enumerator     |                  |
# |----------------|------------------|
# | DISCONNECTED | the sender has been disconnected |
# | SUCCESS |  |
# | GRAB_ERROR | the sender has encountered an grab error |
# | ERRATIC_FPS | the sender does not run with a constant frame rate |
# | FPS_TOO_LOW | fps lower than 10 FPS |
class SENDER_ERROR_CODE(enum.Enum):
    DISCONNECTED = <int>c_SENDER_ERROR_CODE.DISCONNECTED
    SUCCESS = <int>c_SENDER_ERROR_CODE.SUCCESS
    GRAB_ERROR = <int>c_SENDER_ERROR_CODE.GRAB_ERROR
    ERRATIC_FPS = <int>c_SENDER_ERROR_CODE.ERRATIC_FPS
    FPS_TOO_LOW = <int>c_SENDER_ERROR_CODE.FPS_TOO_LOW

    def __str__(self):
        return to_str(toString(<c_SENDER_ERROR_CODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_SENDER_ERROR_CODE>(<int>self.value))).decode()

##
# Change the type of outputed position (raw data or fusion data projected into zed camera)
#
# \ingroup Fusion_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | RAW | The output position will be the raw position data |
# | FUSION | The output position will be the fused position projected into the requested camera repository |
class POSITION_TYPE(enum.Enum):
        RAW  = <int>c_POSITION_TYPE.RAW
        FUSION  = <int>c_POSITION_TYPE.FUSION
        LAST  = <int>c_POSITION_TYPE.LAST

##
# Holds the communication parameter to configure the connection between senders and receiver
# \ingroup Fusion_group
cdef class CommunicationParameters:
    cdef c_CommunicationParameters communicationParameters

    ##
    # Default constructor. All the parameters are set to their default and optimized values.
    def __cinit__(self):
        self.communicationParameters = c_CommunicationParameters()

    ##
    # Setup the communication to used shared memory for intra process workflow, senders and receiver in different threads.
    def set_for_shared_memory(self):
        return self.communicationParameters.setForSharedMemory()

    ##
    # Setup local Network connection information
    def set_for_local_network(self, port : int, ip : str = ""):
        if ip == "":
            return self.communicationParameters.setForLocalNetwork(<int>port)
        return self.communicationParameters.setForLocalNetwork(ip.encode('utf-8'), <int>port)

    ##
    # The comm port used for streaming the data
    @property
    def port(self):
        return self.communicationParameters.getPort()

    ##
    # The IP address of the sender
    @property
    def ip_address(self):
        return self.communicationParameters.getIpAddress().decode()

    ##
    # The type of the used communication
    @property
    def comm_type(self):
        return COMM_TYPE(<int>self.communicationParameters.getType())

##
# useful struct to store the Fusion configuration, can be read from /write to a Json file.
# \ingroup Fusion_group
cdef class FusionConfiguration:
    cdef c_FusionConfiguration fusionConfiguration
    cdef Transform pose

    def __cinit__(self):
        self.pose = Transform()

    ##
    # The serial number of the used ZED camera.
    @property
    def serial_number(self):
        return self.fusionConfiguration.serial_number

    @serial_number.setter
    def serial_number(self, value: int):
        self.fusionConfiguration.serial_number = value

    ##
    # The communication parameters to connect this camera to the Fusion
    @property
    def communication_parameters(self):
        cp = CommunicationParameters()
        cp.communicationParameters = self.fusionConfiguration.communication_parameters
        return cp

    @communication_parameters.setter
    def communication_parameters(self, communication_parameters : CommunicationParameters):
        self.fusionConfiguration.communication_parameters = communication_parameters.communicationParameters

    ##
    # The WORLD Pose of the camera for Fusion
    @property
    def pose(self):
        for i in range(16):
            self.pose.transform.m[i] = self.fusionConfiguration.pose.m[i]
        return self.pose

    @pose.setter
    def pose(self, transform : Transform):
        self.fusionConfiguration.pose = deref(transform.transform)

    ##
    # The input type for the current camera.
    @property
    def input_type(self):
        inp = InputType()
        inp.input = self.fusionConfiguration.input_type
        return inp

    @input_type.setter
    def input_type(self, input_type : InputType):
        self.fusionConfiguration.input_type = input_type.input

##
# Read a Configuration JSON file to configure a fusion process 
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON file containing the configuration
# \param serial_number : the serial number of the ZED Camera you want to retrieve
# \param coord_system : the COORDINATE_SYSTEM in which you want the World Pose to be in
# \param unit : the UNIT in which you want the World Pose to be in
#
# \return a \ref FusionConfiguration for the requested camera
# \note empty if no data were found for the requested camera
def read_fusion_configuration_file_from_serial(self, json_config_filename : str, serial_number : int, coord_system : COORDINATE_SYSTEM, unit: UNIT) -> FusionConfiguration:
    fusion_configuration = FusionConfiguration()
    fusion_configuration.fusionConfiguration = c_readFusionConfigurationFile(json_config_filename.encode('utf-8'), serial_number, <c_COORDINATE_SYSTEM>(<int>coord_system.value), <c_UNIT>(<int>unit.value))
    return fusion_configuration

##
# Read a Configuration JSON file to configure a fusion process
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON file containing the configuration
# \param coord_system : the COORDINATE_SYSTEM in which you want the World Pose to be in
# \param unit : the UNIT in which you want the World Pose to be in
#
# \return a vector of \ref FusionConfiguration for all the camera present in the file
# \note empty if no data were found for the requested camera
def read_fusion_configuration_file(json_config_filename : str, coord_system : COORDINATE_SYSTEM, unit: UNIT) -> list[FusionConfiguration]:
    cdef vector[c_FusionConfiguration] fusion_configurations = c_readFusionConfigurationFile2(json_config_filename.encode('utf-8'), <c_COORDINATE_SYSTEM>(<int>coord_system.value), <c_UNIT>(<int>unit.value))
    return_list = []
    for item in fusion_configurations:
        fc = FusionConfiguration()
        fc.fusionConfiguration = item
        return_list.append(fc)
    return return_list

##
# Write a Configuration JSON file to configure a fusion process
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON that will contain the information
# \param fusion_configurations: a vector of \ref FusionConfiguration listing all the camera configurations
# \param coord_sys : the COORDINATE_SYSTEM in which the World Pose is
# \param unit : the UNIT in which the World Pose is
def write_configuration_file(json_config_filename : str, fusion_configurations : list, coord_sys : COORDINATE_SYSTEM, unit: UNIT):
    cdef vector[c_FusionConfiguration] confs
    for fusion_configuration in fusion_configurations:
        cast_conf = <FusionConfiguration>fusion_configuration
        confs.push_back(cast_conf.fusionConfiguration)

    c_writeConfigurationFile(json_config_filename.encode('utf-8'), confs, <c_COORDINATE_SYSTEM>(<int>coord_sys.value), <c_UNIT>(<int>unit.value))


cdef class PositionalTrackingFusionParameters:
    cdef c_PositionalTrackingFusionParameters positionalTrackingFusionParameters

    ##
    # Is the GNSS fusion enabled
    @property
    def enable_GNSS_fusion(self):
        return self.positionalTrackingFusionParameters.enable_GNSS_fusion

    @enable_GNSS_fusion.setter
    def enable_GNSS_fusion(self, value: bool):
        self.positionalTrackingFusionParameters.enable_GNSS_fusion = value

    ##
    # Is the gnss fusion enabled
    @property
    def gnss_ignore_threshold(self):
        return self.positionalTrackingFusionParameters.gnss_ignore_threshold

    @gnss_ignore_threshold.setter
    def gnss_ignore_threshold(self, value: float):
        self.positionalTrackingFusionParameters.gnss_ignore_threshold = value

##
# Holds the options used to initialize the body tracking module of the \ref Fusion.
# \ingroup Fusion_group
cdef class BodyTrackingFusionParameters:
    cdef c_BodyTrackingFusionParameters bodyTrackingFusionParameters

    ##
    # Defines if the object detection will track objects across images flow
    @property
    def enable_tracking(self):
        return self.bodyTrackingFusionParameters.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, value: bool):
        self.bodyTrackingFusionParameters.enable_tracking = value

    ##
    # Defines if the body fitting will be applied
    @property
    def enable_body_fitting(self):
        return self.bodyTrackingFusionParameters.enable_body_fitting

    @enable_body_fitting.setter
    def enable_body_fitting(self, value: bool):
        self.bodyTrackingFusionParameters.enable_body_fitting = value

##
# Holds the options used to change the behavior of the body tracking module at runtime.
# \ingroup Fusion_group
cdef class BodyTrackingFusionRuntimeParameters:
    cdef c_BodyTrackingFusionRuntimeParameters bodyTrackingFusionRuntimeParameters

    ##
    # if the fused skeleton has less than skeleton_minimum_allowed_keypoints keypoints, it will be discarded
    @property
    def skeleton_minimum_allowed_keypoints(self):
        return self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_keypoints

    @skeleton_minimum_allowed_keypoints.setter
    def skeleton_minimum_allowed_keypoints(self, value: int):
        self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_keypoints = value

    ##
    # if a skeleton was detected in less than skeleton_minimum_allowed_camera cameras, it will be discarded
    @property
    def skeleton_minimum_allowed_camera(self):
        return self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_camera

    @skeleton_minimum_allowed_camera.setter
    def skeleton_minimum_allowed_camera(self, value: int):
        self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_camera = value

    ##
    # this value controls the smoothing of the tracked or fitted fused skeleton. it is ranged from 0 (low smoothing) and 1 (high smoothing)
    @property
    def skeleton_smoothing(self):
        return self.bodyTrackingFusionRuntimeParameters.skeleton_smoothing

    @skeleton_smoothing.setter
    def skeleton_smoothing(self, value: float):
        self.bodyTrackingFusionRuntimeParameters.skeleton_smoothing = value

##
# Holds the metrics of a sender in the fusion process.
# \ingroup Fusion_group
cdef class CameraMetrics :
    cdef c_CameraMetrics cameraMetrics

    ##
    # gives the fps of the received data
    @property
    def received_fps(self):
        return self.cameraMetrics.received_fps

    @received_fps.setter
    def received_fps(self, value: float):
        self.cameraMetrics.received_fps = value

    ##
    # gives the latency (in second) of the received data
    @property
    def received_latency(self):
        return self.cameraMetrics.received_latency

    @received_latency.setter
    def received_latency(self, value: float):
        self.cameraMetrics.received_latency = value

    ##
    # gives the latency (in second) after Fusion synchronization
    @property
    def synced_latency(self):
        return self.cameraMetrics.synced_latency

    @synced_latency.setter
    def synced_latency(self, value: float):
        self.cameraMetrics.synced_latency = value

    ##
    # if no data present is set to false
    @property
    def is_present(self):
        return self.cameraMetrics.is_present

    @is_present.setter
    def is_present(self, value: bool):
        self.cameraMetrics.is_present = value

    ##
    # percent of detection par image during the last second in %, a low values means few detections occurs lately
    @property
    def ratio_detection(self):
        return self.cameraMetrics.ratio_detection

    @ratio_detection.setter
    def ratio_detection(self, value: float):
        self.cameraMetrics.ratio_detection = value

    ##
    # percent of detection par image during the last second in %, a low values means few detections occurs lately
    @property
    def delta_ts(self):
        return self.cameraMetrics.delta_ts

    @delta_ts.setter
    def delta_ts(self, value: float):
        self.cameraMetrics.delta_ts = value

##
# Holds the metrics of the fusion process.
# \ingroup Fusion_group
cdef class FusionMetrics:
    cdef c_FusionMetrics fusionMetrics

    ##
    # reset the current metrics
    def reset(self):
        return self.fusionMetrics.reset()
    
    ##
    # mean number of camera that provides data during the past second
    @property
    def mean_camera_fused(self):
        return self.fusionMetrics.mean_camera_fused

    @mean_camera_fused.setter
    def mean_camera_fused(self, value: float):
        self.fusionMetrics.mean_camera_fused = value

    ##
    # the standard deviation of the data timestamp fused, the lower the better
    @property
    def mean_stdev_between_camera(self):
        return self.fusionMetrics.mean_stdev_between_camera

    @mean_stdev_between_camera.setter
    def mean_stdev_between_camera(self, value: float):
        self.fusionMetrics.mean_stdev_between_camera = value

    ##
    # the sender metrics
    @property
    def camera_individual_stats(self):
        cdef map[c_CameraIdentifier, c_CameraMetrics] temp_map = self.fusionMetrics.camera_individual_stats
        cdef map[c_CameraIdentifier, c_CameraMetrics].iterator it = temp_map.begin()
        returned_value = {}

        while(it != temp_map.end()):
            cam_id = CameraIdentifier()
            cam_id.cameraIdentifier = <c_CameraIdentifier>(deref(it).first)
            cam_metrics = CameraMetrics()
            cam_metrics.cameraMetrics = <c_CameraMetrics>(deref(it).second)
            returned_value[cam_id] = cam_metrics
            postincrement(it) # Increment the iterator to the net element

        return returned_value

    @camera_individual_stats.setter
    def camera_individual_stats(self, value: dict):
        cdef map[c_CameraIdentifier, c_CameraMetrics] temp_map
        for key in value:
            if isinstance(key, CameraIdentifier) and isinstance(value[key], CameraMetrics):
                cam_id = <CameraIdentifier>key
                cam_metrics = <CameraMetrics>CameraMetrics()
                temp_map[cam_id.cameraIdentifier] = cam_metrics.cameraMetrics

        self.fusionMetrics.camera_individual_stats = temp_map

##
# Used to identify a specific camera in the Fusion API
# \ingroup Fusion_group
cdef class CameraIdentifier:
    cdef c_CameraIdentifier cameraIdentifier

    def __cinit__(self, serial_number : int = 0):
        if serial_number == 0:
            self.cameraIdentifier = c_CameraIdentifier()
        self.cameraIdentifier = c_CameraIdentifier(serial_number)

    @property
    def serial_number(self):
        return self.cameraIdentifier.sn

    @serial_number.setter
    def serial_number(self, value: int):
        self.cameraIdentifier.sn = value

##
# Coordinates in ECEF format
cdef class ECEF:
    cdef c_ECEF ecef

    ##
    # x coordinate of ECEF
    @property
    def x(self):
        return self.ecef.x

    @x.setter
    def x(self, value: double):
        self.ecef.x = value

    ##
    # y coordinate of ECEF
    @property
    def y(self):
        return self.ecef.y

    @y.setter
    def y(self, value: double):
        self.ecef.y = value

    ##
    # z coordinate of ECEF
    @property
    def z(self):
        return self.ecef.z

    @z.setter
    def z(self, value: double):
        self.ecef.z = value

##
# Coordinates in LatLng format
cdef class LatLng:
    cdef c_LatLng latLng

    ##
    # Get the latitude coordinate
    #
    # \param in_radian: is the output should be in radian or degree
    # \return float
    def get_latitude(self, in_radian : bool = True):
        return self.latLng.getLatitude(in_radian)

    ##
    # Get the longitude coordinate
    #
    # \param in_radian: is the output should be in radian or degree
    # \return float
    def get_longitude(self, in_radian=True):
        return self.latLng.getLongitude(in_radian)

    ##
    # Get the altitude coordinate
    #
    # \return float
    def get_altitude(self):
        return self.latLng.getAltitude()
    
    ##
    # Get the coordinates in radians (default) or in degrees
    #
    # \param latitude: latitude coordinate
    # \param longitude: longitude coordinate
    # \param altitude:  altitude coordinate
    # \@param in_radian: should we expresse output in radians or in degrees
    def get_coordinates(self, in_radian=True):
        cdef double lat, lng, alt
        self.latLng.getCoordinates(lat, lng, alt, in_radian)
        return lat, lng , alt
    
    ##
    # Set the coordinates in radians (default) or in degrees
    #
    # \param latitude: latitude coordinate
    # \param longitude: longitude coordinate
    # \param altitude:  altitude coordinate
    # \@param in_radian: is input are in radians or in degrees
    def set_coordinates(self, latitude: double, longitude: double, altitude: double, in_radian=True):
        self.latLng.setCoordinates(latitude, longitude, altitude, in_radian)

##
# Coordinate in UTM format
cdef class UTM:
    cdef c_UTM utm

    ##
    # Northing coordinate
    @property
    def northing(self):
        return self.utm.northing

    @northing.setter
    def northing(self, value: double):
        self.utm.northing = value

    ##
    # Easting coordinate
    @property
    def easting(self):
        return self.utm.easting

    @easting.setter
    def easting(self, value: double):
        self.utm.easting = value

    ##
    # Gamma coordinate
    @property
    def gamma(self):
        return self.utm.gamma

    @gamma.setter
    def gamma(self, value: double):
        self.utm.gamma = value

    ##
    # UTMZone if the coordinate
    @property
    def UTM_zone(self):
        return self.utm.UTMZone.decode()

    @UTM_zone.setter
    def UTM_zone(self, value: str):
        self.utm.UTMZone = value.encode('utf-8')

##
# Purely static class for Geo functions
# \ingroup Fusion_group
cdef class GeoConverter:
    ##
    # Convert ECEF coordinates to Lat/Long coordinates
    @staticmethod
    def ecef2latlng(input: ECEF) -> LatLng:
        cdef c_LatLng temp
        c_GeoConverter.ecef2latlng(input.ecef, temp)
        result = LatLng()
        result.latLng = temp
        return result

    ##
    # Convert ECEF coordinates to UTM coordinates
    @staticmethod
    def ecef2utm(input: ECEF) -> UTM:
        cdef c_UTM temp
        c_GeoConverter.ecef2utm(input.ecef, temp)
        result = UTM()
        result.utm.easting = temp.easting
        result.utm.northing = temp.northing
        result.utm.gamma = temp.gamma
        result.utm.UTMZone = temp.UTMZone
        return result

    ##
    # Convert Lat/Long coordinates to ECEF coordinates
    @staticmethod
    def latlng2ecef(input: LatLng) -> ECEF:
        cdef c_ECEF temp
        c_GeoConverter.latlng2ecef(input.latLng, temp)
        result = ECEF()
        result.ecef.x = temp.x
        result.ecef.y = temp.y
        result.ecef.z = temp.z
        return result 

    ##
    # Convert Lat/Long coordinates to UTM coordinates
    @staticmethod
    def latlng2utm(input: LatLng) -> UTM:
        cdef c_UTM temp
        c_GeoConverter.latlng2utm(input.latLng, temp)
        result = UTM()
        result.utm.easting = temp.easting
        result.utm.northing = temp.northing
        result.utm.gamma = temp.gamma
        result.utm.UTMZone = temp.UTMZone
        return result

    ##
    # Convert UTM coordinates to ECEF coordinates
    @staticmethod
    def utm2ecef(input: UTM) -> ECEF:
        cdef c_ECEF temp
        c_GeoConverter.utm2ecef(input.utm, temp)
        result = ECEF()
        result.ecef.x = temp.x
        result.ecef.y = temp.y
        result.ecef.z = temp.z
        return result 

    ##
    # Convert UTM coordinates to Lat/Long coordinates
    @staticmethod
    def utm2latlng(input: UTM) -> LatLng:
        cdef c_LatLng temp
        c_GeoConverter.utm2latlng(input.utm, temp)
        result = LatLng()
        result.latLng = temp
        return result

##
# Holds Geo data
# \ingroup Fusion_group
cdef class GeoPose:
    cdef c_GeoPose geopose
    cdef Transform pose_data 

    ##
    # Default constructor
    def __cinit__(self):
        self.geopose = c_GeoPose()
        self.pose_data = Transform()

    ##
    # the 4x4 Matrix defining the pose
    @property
    def pose_data(self):
        for i in range(16):
            self.pose_data.transform.m[i] = self.geopose.pose_data.m[i]

        return self.pose_data

    @pose_data.setter
    def pose_data(self, transform : Transform):
        self.geopose.pose_data = deref(transform.transform)

    ##
    # the pose covariance
    @property
    def pose_covariance(self):
        arr = []
        for i in range(39):
            arr[i] = self.geopose.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, value):
        if isinstance(value, list):
            if len(value) == 36:
                for i in range(len(value)):
                    self.geopose.pose_covariance[i] = value[i]
            else:
                raise IndexError("Value list must be of length 36.")
        else:
            raise TypeError("Argument must be list type.")

    ##
    # the horizontal accuracy
    @property
    def horizontal_accuracy(self):
        return self.geopose.horizontal_accuracy

    @horizontal_accuracy.setter
    def horizontal_accuracy(self, value: double):
        self.geopose.horizontal_accuracy = value

    ##
    # the vertical accuracy
    @property
    def vertical_accuracy(self):
        return self.geopose.vertical_accuracy

    @vertical_accuracy.setter
    def vertical_accuracy(self, value: double):
        self.geopose.vertical_accuracy = value

    ##
    # the latitude
    @property
    def latlng_coordinates(self):
        result = LatLng()
        result.latLng = self.geopose.latlng_coordinates
        return result

    @latlng_coordinates.setter
    def latlng_coordinates(self, value: LatLng):
        self.geopose.latlng_coordinates = value.latLng

##
# Contains all gnss data to be used for positional tracking as prior.
# \ingroup Sensors_group
cdef class GNSSData:

    cdef c_GNSSData gnss_data

    ##
    # Get the coordinates of GNSSData. The LatLng coordinates could be expressed in degrees or radians.
    #
    # \param latitude: latitude coordinate
    # \param longitude: longitude coordinate
    # \param altitude: altitude coordinate
    # \param is_radian: is the inputs are exppressed in radians or in degrees
    def get_coordinates(self, in_radian=True):
        cdef double lat, lng , alt
        self.gnss_data.getCoordinates(lat, lng, alt, in_radian)
        return lat, lng , alt
    
    ##
    # Set the LatLng coordinates of GNSSData. The LatLng coordinates could be expressed in degrees or radians.
    #
    # \param latitude: latitude coordinate
    # \param longitude: longitude coordinate
    # \param altitude: altitude coordinate
    # \param is_radian: should we express outpu in radians or in degrees
    def set_coordinates(self, latitude: double, longitude: double, altitude: double, in_radian=True):
        self.gnss_data.setCoordinates(latitude, longitude, altitude, in_radian)

    ##
    # latitude standard deviation
    @property
    def latitude_std(self):
        return self.gnss_data.latitude_std

    @latitude_std.setter
    def latitude_std(self, value: double):
        self.gnss_data.latitude_std = value

    ##
    # longitude standard deviation
    @property
    def longitude_std(self):
        return self.gnss_data.longitude_std

    @longitude_std.setter
    def longitude_std(self, value: double):
        self.gnss_data.longitude_std = value

    ##
    # altitude standard deviation
    @property
    def altitude_std(self):
        return self.gnss_data.altitude_std

    @altitude_std.setter
    def altitude_std(self, value: double):
        self.gnss_data.altitude_std = value

    ##
    # \ref Timestamp in the PC clock
    @property
    def ts(self):
        ts = Timestamp()
        ts.timestamp = self.gnss_data.ts
        return  ts

    @ts.setter
    def ts(self, value: Timestamp):
        self.gnss_data.ts = value.timestamp
    
    ##
    # Position covariance in meter
    @property
    def position_covariances(self):
        result = []
        for i in range(9):
                result.append(self.gnss_data.position_covariance[i])
        return result

    @position_covariances.setter
    def position_covariances(self, value:list[float]):
        if isinstance(value, list):
            if len(value) == 9:
                for i in range(9):
                    self.gnss_data.position_covariance[i] = value[i]
                return
        raise TypeError("Argument is not of 9-sized list.")

##
# Holds the options used to initialize the \ref Fusion object.
# \ingroup Fusion_group
cdef class InitFusionParameters:
    cdef c_InitFusionParameters* initFusionParameters

    def __cinit__(self, coordinate_unit : UNIT = UNIT.MILLIMETER, coordinate_system : COORDINATE_SYSTEM = COORDINATE_SYSTEM.IMAGE, output_performance_metrics : bool = False, verbose_ : bool = False, timeout_period_number : int = 20):
        self.initFusionParameters = new c_InitFusionParameters(
            <c_UNIT>(<int>coordinate_unit.value), 
            <c_COORDINATE_SYSTEM>(<int>coordinate_system.value),  
            output_performance_metrics, verbose_, 
            timeout_period_number
        )

    def __dealloc__(self):
        del self.initFusionParameters

    ##
    # This parameter allows you to select the unit to be used for all metric values of the SDK. (depth, point cloud, tracking, mesh, and others).
    # default : \ref UNIT.MILLIMETER
    @property
    def coordinate_units(self):
        return UNIT(<int>self.initFusionParameters.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value: UNIT):
        self.initFusionParameters.coordinate_units = <c_UNIT>(<int>value.value)

    ##
    # Positional tracking, point clouds and many other features require a given \ref COORDINATE_SYSTEM to be used as reference.
    # This parameter allows you to select the \ref COORDINATE_SYSTEM used by the \ref Camera to return its measures.
    # This defines the order and the direction of the axis of the coordinate system.
    # default : \ref COORDINATE_SYSTEM "COORDINATE_SYSTEM::IMAGE"
    @property
    def coordinate_system(self):
        return UNIT(<int>self.initFusionParameters.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value: COORDINATE_SYSTEM):
        self.initFusionParameters.coordinate_system = <c_COORDINATE_SYSTEM>(<int>value.value)

    ##
    # It allows users to extract some stats of the Fusion API like drop frame of each camera, latency, etc
    @property
    def output_performance_metrics(self):
        return self.initFusionParameters.output_performance_metrics

    @output_performance_metrics.setter
    def output_performance_metrics(self, value: bool):
        self.initFusionParameters.output_performance_metrics = value
        
    ##
    # Enable the verbosity mode of the SDK
    @property
    def verbose(self):
        return self.initFusionParameters.verbose

    @verbose.setter
    def verbose(self, value: bool):
        self.initFusionParameters.verbose = value

    ##
    # If specified change the number of period necessary for a source to go in timeout without data. For example, if you set this to 5 then, if any source do not receive data during 5 period, these sources will go to timeout and will be ignored.
    @property
    def timeout_period_number(self):
        return self.initFusionParameters.timeout_period_number

    @timeout_period_number.setter
    def timeout_period_number(self, value: int):
        self.initFusionParameters.timeout_period_number = value


##
# Holds Fusion process data and functions
# \ingroup Fusion_group
cdef class Fusion:
    cdef c_Fusion fusion

    # def __cinit__(self):
    #     self.fusion = c_Fusion()
    
    # def __dealloc__(self):
    #     del self.fusion

    ##
    # FusionHandler initialisation
    #
    # \note Initializes memory/generic data
    def init(self, init_fusion_parameters : InitFusionParameters):
        return FUSION_ERROR_CODE(<int>self.fusion.init(deref(init_fusion_parameters.initFusionParameters)))

    ##
    # FusionHandler close.
    #
    # \note Free memory/generic data
    def close(self):
        return self.fusion.close()

    ##
    # adds a camera to the multi camera handler
    # \param uuid : unique ID that is associated with the camera for easy access.
    # \param json_config_filename : a json configuration file. it should contains the extrinsic calibration of each camera as well as the communication type and configuration of each camera in the system. The same file should be passed to sl::Camera::startPublishing(std::string json_config_filename) of each sender
    def subscribe(self, uuid : CameraIdentifier, communication_parameters: CommunicationParameters, pose: Transform) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.subscribe(uuid.cameraIdentifier, communication_parameters.communicationParameters, deref(pose.transform)))

    def update_pose(self, uuid : CameraIdentifier, pose: Transform) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.updatePose(uuid.cameraIdentifier, deref(pose.transform)))

    ##
    # get the metrics of the Fusion process, for the fused data as well as individual camera provider data
    # \param metrics
    # \return FUSION_STATUS
    def get_process_metrics(self) -> (FUSION_ERROR_CODE, FusionMetrics):
        cdef c_FusionMetrics temp_fusion_metrics
        err = FUSION_ERROR_CODE(<int>self.fusion.getProcessMetrics(temp_fusion_metrics))
        metrics = FusionMetrics()
        metrics.fusionMetrics = temp_fusion_metrics
        return err, metrics

    ##
    # returns the state of each connected data senders.
    # \return the individual state of each connected senders
    def get_sender_state(self) -> dict:
        cdef map[c_CameraIdentifier, c_SENDER_ERROR_CODE] tmp
        tmp = self.fusion.getSenderState()
        cdef map[c_CameraIdentifier, c_SENDER_ERROR_CODE].iterator it = tmp.begin()
        result = {}

        while(it != tmp.end()):
            cam = CameraIdentifier()
            cam.cameraIdentifier = deref(it).first
            err = SENDER_ERROR_CODE(<int>(<c_SENDER_ERROR_CODE>deref(it).second))
            result[cam] = err
            postincrement(it)
        return result

    ##
    # Runs the main function of the Fusion, this trigger the retrieve and sync of all connected senders and updates the enables modules
    # \return SUCCESS if it goes as it should, otherwise it returns an error code. 
    def process(self) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.process())
    
    ##
    # enables Object detection fusion module
    # \param parameters defined by \ref sl::ObjectDetectionFusionParameters
    def enable_body_tracking(self, params : BodyTrackingFusionParameters) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.enableBodyTracking(params.bodyTrackingFusionParameters))
    
    ##
    # retrieves a list of objects (in sl::Objects class type) seen by all cameras and merged as if it was seen by a single super-camera.
    # \note Internal calls retrieveObjects() for all listed cameras, then merged into a single sl::Objects
    # \param objs: list of objects seen by all available cameras
    # \note Only the 3d informations is available in the returned object.
    # For this version, a person is detected if at least it is seen by 2 cameras.
    def retrieve_bodies(self, bodies : Bodies, parameters : BodyTrackingFusionRuntimeParameters, uuid : CameraIdentifier = CameraIdentifier(0)) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.retrieveBodies(bodies.bodies, parameters.bodyTrackingFusionRuntimeParameters, uuid.cameraIdentifier))
    
    ##
    # disables object detection fusion module
    def disable_body_tracking(self):
        return self.fusion.disableBodyTracking()

    ##
    # enable positional tracking fusion.
    # \note note that for the alpha version of the API, the positional tracking fusion doesn't support the area memory feature
    # \param params positional tracking fusion parameters
    # \return FUSION_STATUS
    def enable_positionnal_tracking(self) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.enablePositionalTracking())
    
    ##
    # Add GNSS that will be used by fusion for computing fused pose.
    # \param _gnss_data GPS data put in sl::GNSSData format
    def ingest_gnss_data(self, gnss_data : GNSSData):
        return FUSION_ERROR_CODE(<int>self.fusion.ingestGNSSData(gnss_data.gnss_data))

    ##
    # Get the Fused Position of the camera system
    # \param camera_pose will contain the camera pose in world position (world position is given by the calibration of the cameras system)
    # \param reference_frame defines the reference from which you want the pose to be expressed. Default : \ref REFERENCE_FRAME "REFERENCE_FRAME::WORLD".
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process
    def get_position(self, camera_pose : Pose, reference_frame : REFERENCE_FRAME = REFERENCE_FRAME.WORLD, uuid: CameraIdentifier = CameraIdentifier(), position_type : POSITION_TYPE = POSITION_TYPE.FUSION):
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.getPosition(camera_pose.pose, <c_REFERENCE_FRAME>(<int>reference_frame.value), uuid.cameraIdentifier, <c_POSITION_TYPE>(<int>position_type.value)))

    ##
    # returns the current GNSS data
    # \param out [out]: the current GNSS data
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process
    def get_current_gnss_data(self, gnss_data : GNSSData):
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.getCurrentGNSSData(gnss_data.gnss_data))

    ##
    # returns the current GeoPose
    # \param pose [out]: the current GeoPose
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process
    def get_geo_pose(self, pose : GeoPose) -> POSITIONAL_TRACKING_STATE:
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.getGeoPose(pose.geopose))

    ##
    # returns the current GeoPose
    # \param in: the current GeoPose
    # \param out [out]: the current GeoPose
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process
    def geo_to_camera(self, input : LatLng, output : Pose) -> POSITIONAL_TRACKING_STATE:
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.Geo2Camera(input.latLng, output.pose))

    ##
    # returns the current GeoPose
    # \param pose [out]: the current GeoPose
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process
    def camera_to_geo(self, input : Pose, output : GeoPose) -> POSITIONAL_TRACKING_STATE:
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.Camera2Geo(input.pose, output.geopose))

    ##
    # disable the positional tracking 
    def disable_positionnal_tracking(self):
        return self.fusion.disablePositionalTracking()

