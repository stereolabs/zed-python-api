########################################################################
#
# Copyright (c) 2024, STEREOLABS.
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
from libcpp.unordered_set cimport unordered_set
from libc.math cimport NAN
from libc.stdint cimport uint64_t
from libc.string cimport const_char, memcpy
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from .sl_c cimport ( String, to_str, Camera as c_Camera, ERROR_CODE as c_ERROR_CODE, toString
                    , InitParameters as c_InitParameters, INPUT_TYPE as c_INPUT_TYPE
                    , InputType as c_InputType, RESOLUTION as c_RESOLUTION, BUS_TYPE as c_BUS_TYPE
                    , DEPTH_MODE as c_DEPTH_MODE, UNIT as c_UNIT
                    , COORDINATE_SYSTEM as c_COORDINATE_SYSTEM, CUcontext
                    , RuntimeParameters as c_RuntimeParameters
                    , REFERENCE_FRAME as c_REFERENCE_FRAME, Mat as c_Mat, Resolution as c_Resolution
                    , blobFromImage as c_blobFromImage, blobFromImages as c_blobFromImages
                    , isCameraOne as c_isCameraOne, isResolutionAvailable as c_isResolutionAvailable, isFPSAvailable as c_isFPSAvailable, supportHDR as c_supportHDR
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
                    , REGION_OF_INTEREST_AUTO_DETECTION_STATE as c_REGION_OF_INTEREST_AUTO_DETECTION_STATE
                    , VIDEO_SETTINGS as c_VIDEO_SETTINGS, Rect as c_Rect, SIDE as c_SIDE
                    , RecordingParameters as c_RecordingParameters, SVO_COMPRESSION_MODE as c_SVO_COMPRESSION_MODE
                    , StreamingParameters as c_StreamingParameters, STREAMING_CODEC as c_STREAMING_CODEC
                    , RecordingStatus as c_RecordingStatus, ObjectDetectionParameters as c_ObjectDetectionParameters
                    , BodyTrackingParameters as c_BodyTrackingParameters, BodyTrackingRuntimeParameters as c_BodyTrackingRuntimeParameters
                    , HealthStatus as c_HealthStatus
                    , AI_MODELS as c_AI_MODELS, BODY_TRACKING_MODEL as c_BODY_TRACKING_MODEL, OBJECT_DETECTION_MODEL as c_OBJECT_DETECTION_MODEL
                    , Objects as c_Objects, Bodies as c_Bodies, create_object_detection_runtime_parameters
                    , ObjectDetectionRuntimeParameters as c_ObjectDetectionRuntimeParameters, PlaneDetectionParameters as c_PlaneDetectionParameters
                    , CustomObjectDetectionProperties as c_CustomObjectDetectionProperties, CustomObjectDetectionRuntimeParameters as c_CustomObjectDetectionRuntimeParameters
                    , RegionOfInterestParameters as c_RegionOfInterestParameters
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
                    , ObjectData as c_ObjectData, BodyData as c_BodyData, OBJECT_CLASS as c_OBJECT_CLASS, MODULE as c_MODULE, OBJECT_SUBCLASS as c_OBJECT_SUBCLASS
                    , OBJECT_TRACKING_STATE as c_OBJECT_TRACKING_STATE, OBJECT_ACTION_STATE as c_OBJECT_ACTION_STATE
                    , BODY_18_PARTS as c_BODY_18_PARTS, SIDE as c_SIDE, CameraInformation as c_CameraInformation, CUctx_st
                    , CameraOneInformation as c_CameraOneInformation, CameraOneConfiguration as c_CameraOneConfiguration
                    , FLIP_MODE as c_FLIP_MODE, getResolution as c_getResolution, BatchParameters as c_BatchParameters
                    , ObjectsBatch as c_ObjectsBatch, BodiesBatch as c_BodiesBatch, getIdx as c_getIdx
                    , INFERENCE_PRECISION as c_INFERENCE_PRECISION, BODY_FORMAT as c_BODY_FORMAT, BODY_KEYPOINTS_SELECTION as c_BODY_KEYPOINTS_SELECTION
                    , BODY_34_PARTS as c_BODY_34_PARTS, BODY_38_PARTS as c_BODY_38_PARTS
                    , generate_unique_id as c_generate_unique_id, CustomBoxObjectData as c_CustomBoxObjectData, CustomMaskObjectData as c_CustomMaskObjectData
                    , OBJECT_FILTERING_MODE as c_OBJECT_FILTERING_MODE, OBJECT_ACCELERATION_PRESET as c_OBJECT_ACCELERATION_PRESET
                    , COMM_TYPE as c_COMM_TYPE, FUSION_ERROR_CODE as c_FUSION_ERROR_CODE, SENDER_ERROR_CODE as c_SENDER_ERROR_CODE
                    , FusionConfiguration as c_FusionConfiguration, CommunicationParameters as c_CommunicationParameters
                    , InitFusionParameters as c_InitFusionParameters, CameraIdentifier as c_CameraIdentifier
                    , SynchronizationParameter as c_SynchronizationParameter
                    , BodyTrackingFusionParameters as c_BodyTrackingFusionParameters, BodyTrackingFusionRuntimeParameters as c_BodyTrackingFusionRuntimeParameters
                    , ObjectDetectionFusionParameters as c_ObjectDetectionFusionParameters
                    , PositionalTrackingFusionParameters as c_PositionalTrackingFusionParameters, GNSSCalibrationParameters as c_GNSSCalibrationParameters, POSITION_TYPE as c_POSITION_TYPE
                    , FUSION_REFERENCE_FRAME as c_FUSION_REFERENCE_FRAME, GNSS_FUSION_STATUS as c_GNSS_FUSION_STATUS
                    , CameraMetrics as c_CameraMetrics, FusionMetrics as c_FusionMetrics, GNSSData as c_GNSSData, Fusion as c_Fusion
                    , ECEF as c_ECEF, LatLng as c_LatLng, UTM as c_UTM 
                    , GeoConverter as c_GeoConverter, GeoPose as c_GeoPose
                    , readFusionConfiguration as c_readFusionConfiguration
                    , readFusionConfigurationFile as c_readFusionConfigurationFile
                    , readFusionConfigurationFile2 as c_readFusionConfigurationFile2
                    , writeConfigurationFile as c_writeConfigurationFile
                    , SVOData as c_SVOData
                    , PositionalTrackingStatus as c_PositionalTrackingStatus
                    , ODOMETRY_STATUS as c_ODOMETRY_STATUS
                    , SPATIAL_MEMORY_STATUS as c_SPATIAL_MEMORY_STATUS
                    , POSITIONAL_TRACKING_FUSION_STATUS as c_POSITIONAL_TRACKING_FUSION_STATUS
                    , FusedPositionalTrackingStatus as c_FusedPositionalTrackingStatus
                    , GNSS_STATUS as c_GNSS_STATUS
                    , GNSS_MODE as c_GNSS_MODE
                    , GNSS_FUSION_STATUS as c_GNSS_FUSION_STATUS
                    , CameraOne as c_CameraOne
                    , InitParametersOne as c_InitParametersOne
                    , Landmark as c_Landmark
                    , Landmark2D as c_Landmark2D
                    , ENU as c_ENU
                    )
from cython.operator cimport (dereference as deref, postincrement)
from cpython cimport bool
import enum
import json
from cython.operator import dereference, postincrement


cdef bint CUPY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

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
# Structure representing timestamps with  utilities.
# \ingroup Core_group
cdef class Timestamp():
    cdef c_Timestamp timestamp

    def __cinit__(self):
        self.timestamp = c_Timestamp()

    ##
    # Timestamp in nanoseconds.
    @property
    def data_ns(self) -> int:
        return self.timestamp.data_ns

    @data_ns.setter
    def data_ns(self, ns):
        self.timestamp.data_ns = ns

    ##
    # Returns the timestamp in nanoseconds.
    def get_nanoseconds(self) -> int:
        return self.timestamp.getNanoseconds()

    ##
    # Returns the timestamp in microseconds.
    def get_microseconds(self) -> int:
        return self.timestamp.getMicroseconds()

    ##
    # Returns the timestamp in milliseconds.
    def get_milliseconds(self) -> int:
        return self.timestamp.getMilliseconds()

    ##
    # Returns the timestamp in seconds.
    def get_seconds(self) -> int:
        return self.timestamp.getSeconds()

    ##
    # Sets the timestamp to a value in nanoseconds.
    def set_nanoseconds(self, t_ns: int) -> None:
        self.timestamp.setNanoseconds(t_ns)

    ##
    # Sets the timestamp to a value in microseconds.
    def set_microseconds(self, t_us: int) -> None:
        self.timestamp.setMicroseconds(t_us)

    ##
    # Sets the timestamp to a value in milliseconds.
    def set_milliseconds(self, t_ms: int) -> None:
        self.timestamp.setMilliseconds(t_ms)

    ##
    # Sets the timestamp to a value in seconds.
    def set_seconds(self, t_s: int) -> None:
        self.timestamp.setSeconds(t_s)

##
# Lists error codes in the ZED SDK.
# \ingroup Core_group
#
# | Enumerator                                         |                                                                                                                                                                 |
# |----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | CONFIGURATION_FALLBACK                             | The operation could not proceed with the target configuration but did success with a fallback.                                                         |
# | SENSORS_DATA_REQUIRED                              | The input data does not contains the high frequency sensors data, this is usually because it requires newer SVO/Streaming. In order to work this modules needs inertial data present in it input.                                                |
# | CORRUPTED_FRAME                                    | The image could be corrupted, Enabled with the parameter InitParameters.enable_image_validity_check.      
# | CAMERA_REBOOTING                                   | The camera is currently rebooting.                                                                                                                              |
# | SUCCESS                                            | Standard code for successful behavior.                                                                                                                          |
# | FAILURE                                            | Standard code for unsuccessful behavior.                                                                                                                        |
# | NO_GPU_COMPATIBLE                                  | No GPU found or CUDA capability of the device is not supported.                                                                                                 |
# | NOT_ENOUGH_GPU_MEMORY                              | Not enough GPU memory for this depth mode. Try a different mode (such as \ref DEPTH_MODE "PERFORMANCE"), or increase the minimum depth value (see \ref InitParameters.depth_minimum_distance). |
# | CAMERA_NOT_DETECTED                                | No camera was detected.                                                                                                                       |
# | SENSORS_NOT_INITIALIZED                            | The MCU that controls the sensors module has an invalid serial number. You can try to recover it by launching the <b>ZED Diagnostic</b> tool from the command line with the option <code>-r</code>. |
# | SENSORS_NOT_AVAILABLE                              | A camera with sensor is detected but the sensors (IMU, barometer, ...) cannot be opened. Only the \ref MODEL "MODEL.ZED" does not has sensors. Unplug/replug is required. |
# | INVALID_RESOLUTION                                 | In case of invalid resolution parameter, such as an upsize beyond the original image size in Camera.retrieve_image.                                             |
# | LOW_USB_BANDWIDTH                                  | Insufficient bandwidth for the correct use of the camera. This issue can occur when you use multiple cameras or a USB 2.0 port.                                |
# | CALIBRATION_FILE_NOT_AVAILABLE                     | The calibration file of the camera is not found on the host machine. Use <b>ZED Explorer</b> or <b>ZED Calibration</b> to download the factory calibration file. |
# | INVALID_CALIBRATION_FILE                           | The calibration file is not valid. Try to download the factory calibration file or recalibrate your camera using <b>ZED Calibration</b>.                                     |
# | INVALID_SVO_FILE                                   | The provided SVO file is not valid.                                                                                                                             |
# | SVO_RECORDING_ERROR                                | An error occurred while trying to record an SVO (not enough free storage, invalid file, ...).                                                                   |
# | SVO_UNSUPPORTED_COMPRESSION                        | An SVO related error, occurs when NVIDIA based compression cannot be loaded.                                                                                            |
# | END_OF_SVOFILE_REACHED                             | SVO end of file has been reached.\n No frame will be available until the SVO position is reset.                                                               |
# | INVALID_COORDINATE_SYSTEM                          | The requested coordinate system is not available.                                                                                                               |
# | INVALID_FIRMWARE                                   | The firmware of the camera is out of date. Update to the latest version.                                                                                        |
# | INVALID_FUNCTION_PARAMETERS                        | Invalid parameters have been given for the function.                                                                                                            |
# | CUDA_ERROR                                         | A CUDA error has been detected in the process, in sl.Camera.grab() or sl.Camera.retrieve_xxx() only. Activate verbose in sl.Camera.open() for more info.        |
# | CAMERA_NOT_INITIALIZED                             | The ZED SDK is not initialized. Probably a missing call to sl.Camera.open().                                                                                    |
# | NVIDIA_DRIVER_OUT_OF_DATE                          | Your NVIDIA driver is too old and not compatible with your current CUDA version.                                                                                |
# | INVALID_FUNCTION_CALL                              | The call of the function is not valid in the current context. Could be a missing call of sl.Camera.open().                                                      |
# | CORRUPTED_SDK_INSTALLATION                         | The ZED SDK was not able to load its dependencies or some assets are missing. Reinstall the ZED SDK or check for missing dependencies (cuDNN, TensorRT).        |
# | INCOMPATIBLE_SDK_VERSION                           | The installed ZED SDK is incompatible with the one used to compile the program.                                                                                 |
# | INVALID_AREA_FILE                                  | The given area file does not exist. Check the path.                                                                                                             |
# | INCOMPATIBLE_AREA_FILE                             | The area file does not contain enough data to be used or the sl.DEPTH_MODE used during the creation of the area file is different from the one currently set.   |
# | CAMERA_FAILED_TO_SETUP                             | Failed to open the camera at the proper resolution. Try another resolution or make sure that the UVC driver is properly installed.                              |
# | CAMERA_DETECTION_ISSUE                             | Your camera can not be opened. Try replugging it to another port or flipping the USB-C connector (if there is one).                                             |
# | CANNOT_START_CAMERA_STREAM                         | Cannot start the camera stream. Make sure your camera is not already used by another process or blocked by firewall or antivirus.                               |
# | NO_GPU_DETECTED                                    | No GPU found. CUDA is unable to list it. Can be a driver/reboot issue.                                                                                          |
# | PLANE_NOT_FOUND                                    | Plane not found. Either no plane is detected in the scene, at the location or corresponding to the floor, or the floor plane doesn't match the prior given.     |
# | MODULE_NOT_COMPATIBLE_WITH_CAMERA                  | The module you try to use is not compatible with your camera sl.MODEL. \note \ref MODEL "sl.MODEL.ZED" does not has an IMU and does not support the AI modules. |
# | MOTION_SENSORS_REQUIRED                            | The module needs the sensors to be enabled (see \ref InitParameters.sensors_required). |
# | MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION            | The module needs a newer version of CUDA. |
class ERROR_CODE(enum.Enum):
    CONFIGURATION_FALLBACK = <int>c_ERROR_CODE.CONFIGURATION_FALLBACK
    SENSORS_DATA_REQUIRED = <int>c_ERROR_CODE.SENSORS_DATA_REQUIRED
    CORRUPTED_FRAME = <int>c_ERROR_CODE.CORRUPTED_FRAME
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
        return to_str(toString(<c_ERROR_CODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_ERROR_CODE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, ERROR_CODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, ERROR_CODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, ERROR_CODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, ERROR_CODE):
            return self.value >= other.value
        return NotImplemented


# The C++ enum being wrapped through Cython causes runtime overhead
# When converting between the C++ enum and Python, there's overhead from:
#  - Creating a new Python object for the enum value
#  - Table lookups for the enum names
#  - Type checking and validation
#  - Possible memory allocation
# To compensate for that, we cache the conversion in a dict instead
# We could optimize further with a PyArray instead but profiling on a Nano 8GB already shows quite good runtimes
cdef dict _error_code_cache = {}
def _initialize_error_codes():
    global _error_code_cache
    if not _error_code_cache:  # Only initialize if not already done
        for error_code in ERROR_CODE:  # Iterate through the existing Python enum
            _error_code_cache[error_code.value] = error_code
_initialize_error_codes()


##
# Lists ZED camera model.
#
# \ingroup Video_group
#
# | Enumerator |                  |
# |------------|------------------|
# | ZED        | ZED camera model |
# | ZED_M      | ZED Mini (ZED M) camera model |
# | ZED2       | ZED 2 camera model |
# | ZED2i      | ZED 2i camera model |
# | ZED_X      | ZED X camera model |
# | ZED_XM     | ZED X Mini (ZED XM) camera model |
# | ZED_X_HDR  | ZED X HDR camera model |
# | ZED_X_HDR_MINI | ZED X HDR Mini camera model |
# | ZED_X_HDR_MAX | ZED X HDR Wide camera model |
# | VIRTUAL_ZED_X | Virtual ZED X generated from 2 ZED X One |
# | ZED_XONE_GS   | ZED X One with global shutter AR0234 sensor |
# | ZED_XONE_UHD  | ZED X One with 4K rolling shutter IMX678 sensor |
# | ZED_XONE_HDR  | ZED X One HDR |
class MODEL(enum.Enum):
    ZED = <int>c_MODEL.ZED
    ZED_M = <int>c_MODEL.ZED_M
    ZED2 = <int>c_MODEL.ZED2
    ZED2i = <int>c_MODEL.ZED2i
    ZED_X = <int>c_MODEL.ZED_X
    ZED_XM = <int>c_MODEL.ZED_XM
    ZED_X_HDR = <int>c_MODEL.ZED_X_HDR
    ZED_X_HDR_MINI = <int>c_MODEL.ZED_X_HDR_MINI
    ZED_X_HDR_MAX = <int>c_MODEL.ZED_X_HDR_MAX
    VIRTUAL_ZED_X = <int>c_MODEL.VIRTUAL_ZED_X
    ZED_XONE_GS = <int>c_MODEL.ZED_XONE_GS
    ZED_XONE_UHD = <int>c_MODEL.ZED_XONE_UHD
    ZED_XONE_HDR = <int>c_MODEL.ZED_XONE_HDR
    LAST = <int>c_MODEL.MODEL_LAST

    def __str__(self):
        return to_str(toString(<c_MODEL>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_MODEL>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, MODEL):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MODEL):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MODEL):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MODEL):
            return self.value >= other.value
        return NotImplemented

##
# Lists available input types in the ZED SDK.
#
# \ingroup Video_group
#
# | Enumerator     |                  |
# |------------|------------------|
# | USB        | USB input mode  |
# | SVO        | SVO file input mode  |
# | STREAM     | STREAM input mode (requires to use \ref Camera.enable_streaming "enable_streaming()" / \ref Camera.disable_streaming "disable_streaming()" on the "sender" side) |
# | GMSL       | GMSL input mode (only on NVIDIA Jetson) |

class INPUT_TYPE(enum.Enum):
    USB = <int>c_INPUT_TYPE.USB
    SVO = <int>c_INPUT_TYPE.SVO
    STREAM = <int>c_INPUT_TYPE.STREAM
    GMSL = <int>c_INPUT_TYPE.GMSL
    LAST = <int>c_INPUT_TYPE.LAST

    def __lt__(self, other):
        if isinstance(other, INPUT_TYPE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, INPUT_TYPE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, INPUT_TYPE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, INPUT_TYPE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available AI models.
# \ingroup Object_group
#
# | Enumerator               |                  |
# |--------------------------|------------------|
# | MULTI_CLASS_DETECTION | Related to [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST](\ref OBJECT_DETECTION_MODEL) |
# | MULTI_CLASS_MEDIUM_DETECTION | Related to [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM](\ref OBJECT_DETECTION_MODEL) |
# | MULTI_CLASS_ACCURATE_DETECTION | Related to [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE](\ref OBJECT_DETECTION_MODEL) |
# | HUMAN_BODY_FAST_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST](\ref BODY_TRACKING_MODEL) |
# | HUMAN_BODY_MEDIUM_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM](\ref BODY_TRACKING_MODEL) |
# | HUMAN_BODY_ACCURATE_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE](\ref BODY_TRACKING_MODEL) |
# | HUMAN_BODY_38_FAST_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST](\ref BODY_TRACKING_MODEL) |
# | HUMAN_BODY_38_MEDIUM_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST](\ref BODY_TRACKING_MODEL) |
# | HUMAN_BODY_38_ACCURATE_DETECTION | Related to [sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST](\ref BODY_TRACKING_MODEL) |
# | PERSON_HEAD_DETECTION | Related to [sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_FAST](\ref OBJECT_DETECTION_MODEL) |
# | PERSON_HEAD_ACCURATE_DETECTION | Related to [sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_ACCURATE](\ref OBJECT_DETECTION_MODEL) |
# | REID_ASSOCIATION | Related to sl.BatchParameters.enable |
# | NEURAL_LIGHT_DEPTH | Related to [sl.DEPTH_MODE.NEURAL_LIGHT_DEPTH](\ref DEPTH_MODE) |
# | NEURAL_DEPTH | Related to [sl.DEPTH_MODE.NEURAL](\ref DEPTH_MODE) |
# | NEURAL_PLUS_DEPTH | Related to [sl.DEPTH_MODE.NEURAL_PLUS_DEPTH](\ref DEPTH_MODE) |
class AI_MODELS(enum.Enum):
    MULTI_CLASS_DETECTION = <int>c_AI_MODELS.MULTI_CLASS_DETECTION
    MULTI_CLASS_MEDIUM_DETECTION = <int>c_AI_MODELS.MULTI_CLASS_MEDIUM_DETECTION
    MULTI_CLASS_ACCURATE_DETECTION = <int>c_AI_MODELS.MULTI_CLASS_ACCURATE_DETECTION
    HUMAN_BODY_FAST_DETECTION = <int>c_AI_MODELS.HUMAN_BODY_FAST_DETECTION
    HUMAN_BODY_MEDIUM_DETECTION = <int>c_AI_MODELS.HUMAN_BODY_MEDIUM_DETECTION
    HUMAN_BODY_ACCURATE_DETECTION = <int>c_AI_MODELS.HUMAN_BODY_ACCURATE_DETECTION
    HUMAN_BODY_38_FAST_DETECTION = <int>c_AI_MODELS.HUMAN_BODY_38_FAST_DETECTION
    HUMAN_BODY_38_MEDIUM_DETECTION = <int>c_AI_MODELS.HUMAN_BODY_38_MEDIUM_DETECTION
    HUMAN_BODY_38_ACCURATE_DETECTION = <int>c_AI_MODELS. HUMAN_BODY_38_ACCURATE_DETECTION
    PERSON_HEAD_DETECTION = <int>c_AI_MODELS.PERSON_HEAD_DETECTION
    PERSON_HEAD_ACCURATE_DETECTION = <int>c_AI_MODELS.PERSON_HEAD_ACCURATE_DETECTION
    REID_ASSOCIATION = <int>c_AI_MODELS.REID_ASSOCIATION
    NEURAL_LIGHT_DEPTH = <int>c_AI_MODELS.NEURAL_LIGHT_DEPTH
    NEURAL_DEPTH = <int>c_AI_MODELS.NEURAL_DEPTH
    NEURAL_PLUS_DEPTH = <int>c_AI_MODELS.NEURAL_PLUS_DEPTH
    LAST = <int>c_OBJECT_DETECTION_MODEL.LAST

    def __lt__(self, other):
        if isinstance(other, AI_MODELS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, AI_MODELS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, AI_MODELS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, AI_MODELS):
            return self.value >= other.value
        return NotImplemented

##
# Lists available models for the object detection module.
#
# \ingroup Object_group
#
# | Enumerator               |                  |
# |--------------------------|------------------|
# | MULTI_CLASS_BOX_FAST     | Any objects, bounding box based. |
# | MULTI_CLASS_BOX_ACCURATE | Any objects, bounding box based, more accurate but slower than the base model. |
# | MULTI_CLASS_BOX_MEDIUM   | Any objects, bounding box based, compromise between accuracy and speed. |
# | PERSON_HEAD_BOX_FAST     | Bounding box detector specialized in person heads particularly well suited for crowded environments. The person localization is also improved. |
# | PERSON_HEAD_BOX_ACCURATE | Bounding box detector specialized in person heads, particularly well suited for crowded environments. The person localization is also improved, more accurate but slower than the base model. |
# | CUSTOM_BOX_OBJECTS       | For external inference, using your own custom model and/or frameworks. This mode disables the internal inference engine, the 2D bounding box detection must be provided. |
class OBJECT_DETECTION_MODEL(enum.Enum):
    MULTI_CLASS_BOX_FAST = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    MULTI_CLASS_BOX_MEDIUM = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    MULTI_CLASS_BOX_ACCURATE = <int>c_OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    PERSON_HEAD_BOX_FAST = <int>c_OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_FAST
    PERSON_HEAD_BOX_ACCURATE = <int>c_OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_ACCURATE
    CUSTOM_BOX_OBJECTS = <int>c_OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    CUSTOM_YOLOLIKE_BOX_OBJECTS = <int>c_OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    LAST = <int>c_OBJECT_DETECTION_MODEL.LAST

    def __lt__(self, other):
        if isinstance(other, OBJECT_DETECTION_MODEL):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_DETECTION_MODEL):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_DETECTION_MODEL):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_DETECTION_MODEL):
            return self.value >= other.value
        return NotImplemented

##
# Lists available models for the body tracking module.
#
# \ingroup Body_group
#
# | Enumerator               |                  |
# |--------------------------|------------------|
# | HUMAN_BODY_FAST          | Keypoints based, specific to human skeleton, real time performance even on Jetson or low end GPU cards. |
# | HUMAN_BODY_ACCURATE      | Keypoints based, specific to human skeleton, state of the art accuracy, requires powerful GPU. |
# | HUMAN_BODY_MEDIUM        | Keypoints based, specific to human skeleton, compromise between accuracy and speed.  |
class BODY_TRACKING_MODEL(enum.Enum):
    HUMAN_BODY_FAST = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    HUMAN_BODY_ACCURATE = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    HUMAN_BODY_MEDIUM = <int>c_BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
    LAST = <int>c_BODY_TRACKING_MODEL.LAST

    def __lt__(self, other):
        if isinstance(other, BODY_TRACKING_MODEL):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, BODY_TRACKING_MODEL):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, BODY_TRACKING_MODEL):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, BODY_TRACKING_MODEL):
            return self.value >= other.value
        return NotImplemented

##
# Lists supported bounding box preprocessing.
#
# \ingroup Object_group
#
# | Enumerator       |                  |
# |------------------|------------------|
# | NONE             | The ZED SDK will not apply any preprocessing to the detected objects. |
# | NMS3D            | The ZED SDK will remove objects that are in the same 3D position as an already tracked object (independent of class id). |
# | NMS3D_PER_CLASS  | The ZED SDK will remove objects that are in the same 3D position as an already tracked object of the same class id. |
class OBJECT_FILTERING_MODE(enum.Enum):
    NONE = <int>c_OBJECT_FILTERING_MODE.NONE
    NMS3D = <int>c_OBJECT_FILTERING_MODE.NMS3D
    NMS3D_PER_CLASS = <int>c_OBJECT_FILTERING_MODE.NMS3D_PER_CLASS
    LAST = <int>c_OBJECT_FILTERING_MODE.LAST

    def __lt__(self, other):
        if isinstance(other, OBJECT_FILTERING_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_FILTERING_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_FILTERING_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_FILTERING_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists supported presets for maximum acceleration allowed for a given tracked object.
#
# \ingroup Object_group
#
# | Enumerator |                  |
# |------------|------------------|
# | DEFAULT    | The ZED SDK will automatically determine the appropriate maximum acceleration. |
# | LOW        | Suitable for objects with relatively low maximum acceleration (e.g., a person walking). |
# | MEDIUM     | Suitable for objects with moderate maximum acceleration (e.g., a person running). |
# | HIGH       | Suitable for objects with high maximum acceleration (e.g., a car accelerating, a kicked sports ball). |
class OBJECT_ACCELERATION_PRESET(enum.Enum):
    DEFAULT = <int>c_OBJECT_ACCELERATION_PRESET.ACC_PRESET_DEFAULT
    LOW = <int>c_OBJECT_ACCELERATION_PRESET.ACC_PRESET_LOW
    MEDIUM = <int>c_OBJECT_ACCELERATION_PRESET.ACC_PRESET_MEDIUM
    HIGH = <int>c_OBJECT_ACCELERATION_PRESET.ACC_PRESET_HIGH
    LAST = <int>c_OBJECT_ACCELERATION_PRESET.ACC_PRESET_LAST

    def __lt__(self, other):
        if isinstance(other, OBJECT_ACCELERATION_PRESET):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_ACCELERATION_PRESET):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_ACCELERATION_PRESET):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_ACCELERATION_PRESET):
            return self.value >= other.value
        return NotImplemented

##
# Lists possible camera states.
#
# \ingroup Video_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | AVAILABLE      | The camera can be opened by the ZED SDK. |
# | NOT_AVAILABLE  | The camera is already opened and unavailable. |
class CAMERA_STATE(enum.Enum):
    AVAILABLE = <int>c_CAMERA_STATE.AVAILABLE
    NOT_AVAILABLE = <int>c_CAMERA_STATE.NOT_AVAILABLE
    LAST = <int>c_CAMERA_STATE.CAMERA_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_CAMERA_STATE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_CAMERA_STATE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, CAMERA_STATE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, CAMERA_STATE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, CAMERA_STATE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, CAMERA_STATE):
            return self.value >= other.value
        return NotImplemented

##
# Lists possible sides on which to get data from.
# \ingroup Video_group
#
# | Enumerator |            |
# |------------|------------|
# | LEFT       | Left side only. |
# | RIGHT      | Right side only. |
# | BOTH       | Left and right side. |
class SIDE(enum.Enum):
    LEFT = <int>c_SIDE.LEFT
    RIGHT = <int>c_SIDE.RIGHT
    BOTH = <int>c_SIDE.BOTH

    def __lt__(self, other):
        if isinstance(other, SIDE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SIDE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SIDE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SIDE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available resolutions.
# \ingroup Core_group
# \note The VGA resolution does not respect the 640*480 standard to better fit the camera sensor (672*376 is used).
# \warning All resolutions are not available for every camera.
# \warning You can find the available resolutions for each camera in <a href="https://www.stereolabs.com/docs/video/camera-controls#selecting-a-resolution">our documentation</a>.
#
# | Enumerator |            |
# |------------|------------|
# | HD4K    | 3856x2180 for imx678 mono |
# | QHDPLUS | 3800x1800 |
# | HD2K    | 2208*1242 (x2) \n Available FPS: 15 |
# | HD1080  | 1920*1080 (x2) \n Available FPS: 15, 30 |
# | HD1200  | 1920*1200 (x2) \n Available FPS: 15, 30, 60 |
# | HD1536  | 1920*1536 (x2) \n Available FPS: 30 |
# | HD720   | 1280*720 (x2) \n Available FPS: 15, 30, 60 |
# | SVGA    | 960*600 (x2) \n Available FPS: 15, 30, 60, 120 |
# | VGA     | 672*376 (x2) \n Available FPS: 15, 30, 60, 100 |
# | AUTO    | Select the resolution compatible with the camera: <ul><li>ZED X/X Mini: HD1200</li><li>other cameras: HD720</li></ul> |
class RESOLUTION(enum.Enum):
    HD4K = <int>c_RESOLUTION.HD4K
    QHDPLUS = <int>c_RESOLUTION.QHDPLUS
    HD2K = <int>c_RESOLUTION.HD2K
    HD1080 = <int>c_RESOLUTION.HD1080
    HD1200 = <int>c_RESOLUTION.HD1200
    HD1536 = <int>c_RESOLUTION.HD1536
    HD720 = <int>c_RESOLUTION.HD720
    SVGA  = <int>c_RESOLUTION.SVGA
    VGA  = <int>c_RESOLUTION.VGA
    AUTO = <int>c_RESOLUTION.AUTO
    LAST = <int>c_RESOLUTION.LAST

    def __lt__(self, other):
        if isinstance(other, RESOLUTION):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, RESOLUTION):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, RESOLUTION):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, RESOLUTION):
            return self.value >= other.value
        return NotImplemented

##
# Blocks the execution of the current thread for \b time milliseconds.
# \ingroup Core_group
# \param time : Number of milliseconds to wait.
def sleep_ms(time: int) -> None:
    c_sleep_ms(time)

##
# Blocks the execution of the current thread for \b time microseconds.
# \ingroup Core_group
# \param time : Number of microseconds to wait.
def sleep_us(time: int) -> None:
    c_sleep_us(time)


##
# Gets the corresponding sl.Resolution from an sl.RESOLUTION.
# \ingroup Video_group
#
# \param resolution : The wanted sl.RESOLUTION.
# \return The sl.Resolution corresponding to sl.RESOLUTION given as argument.
def get_resolution(resolution: RESOLUTION) -> Resolution:
    if isinstance(resolution, RESOLUTION):
        out = c_getResolution(<c_RESOLUTION>(<int>resolution.value))
        res = Resolution()
        res.width = out.width
        res.height = out.height
        return res
    else:
        raise TypeError("Argument is not of RESOLUTION type.")
        
##
# Class containing information about the properties of a camera.
# \ingroup Video_group
#
# \note A \ref camera_model \ref MODEL "sl.MODEL.ZED_M" with an id '-1' can be due to an inverted USB-C cable.
cdef class DeviceProperties:
    cdef c_DeviceProperties c_device_properties

    def __cinit__(self):
        self.c_device_properties = c_DeviceProperties()

    ##
    # State of the camera.
    #
    # Default: \ref CAMERA_STATE "sl.CAMERA_STATE.NOT_AVAILABLE"
    @property
    def camera_state(self) -> CAMERA_STATE:
        return CAMERA_STATE(<int>self.c_device_properties.camera_state)

    @camera_state.setter
    def camera_state(self, camera_state):
        if isinstance(camera_state, CAMERA_STATE):
            self.c_device_properties.camera_state = (<c_CAMERA_STATE> (<int>camera_state.value))
        else:
            raise TypeError("Argument is not of CAMERA_STATE type.")

    ##
    # Id of the camera.
    #
    # Default: -1
    @property
    def id(self) -> int:
        return self.c_device_properties.id

    @id.setter
    def id(self, int id):
        self.c_device_properties.id = id

    ##
    # System path of the camera.
    @property
    def path(self) -> str:
        if not self.c_device_properties.path.empty():
            return self.c_device_properties.path.get().decode()
        else:
            return ""

    @path.setter
    def path(self, str path):
        path_ = path.encode()
        self.c_device_properties.path = (String(<char*> path_))

    ##
    # i2c port of the camera.
    @property
    def i2c_port(self) -> int:
        return self.c_device_properties.i2c_port

    @i2c_port.setter
    def i2c_port(self, int i2c_port):
        self.c_device_properties.i2c_port = i2c_port

    ##
    # Model of the camera.
    @property
    def camera_model(self) -> MODEL:
        return MODEL(<int>self.c_device_properties.camera_model)

    @camera_model.setter
    def camera_model(self, camera_model: MODEL):
        if isinstance(camera_model, MODEL):
            self.c_device_properties.camera_model = (<c_MODEL> (<int>camera_model.value))
        else:
            raise TypeError("Argument is not of MODEL type.")

    ##
    # Serial number of the camera.
    #
    # Default: 0
    # \warning Not provided for Windows.
    @property
    def serial_number(self) -> int:
        return self.c_device_properties.serial_number

    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_device_properties.serial_number = serial_number

    ##
    # sensor_address when available (ZED-X HDR/XOne HDR only)
    @property
    def identifier(self) -> np.numpy[np.uint8]:
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.c_device_properties.identifier[i]
        return arr

    @identifier.setter
    def identifier(self, object identifier):
        if isinstance(identifier, list):
            if len(identifier) != 3:
                raise IndexError("identifier List must be of length 3.")
        elif isinstance(identifier, np.ndarray):
            if identifier.size != 3:
                raise IndexError("identifier Numpy array must be of size 3.")
        else:
            raise TypeError("Argument must be numpy array or list type.")
        for i in range(3):
            self.c_device_properties.identifier[i] = identifier[i]

    ##
    # Badge name (zedx_ar0234)
    @property
    def camera_badge(self) -> str:
        if not self.c_device_properties.camera_badge.empty():
            return self.c_device_properties.camera_badge.get().decode()
        else:
            return ""

    @camera_badge.setter
    def camera_badge(self, str camera_badge):
        camera_badge_ = camera_badge.encode()
        self.c_device_properties.camera_badge = (String(<char*> camera_badge_))

    ##
    # Name of sensor (zedx)
    @property
    def camera_sensor_model(self) -> str:
        if not self.c_device_properties.camera_sensor_model.empty():
            return self.c_device_properties.camera_sensor_model.get().decode()
        else:
            return ""

    @camera_sensor_model.setter
    def camera_sensor_model(self, str camera_sensor_model):
        camera_sensor_model_ = camera_sensor_model.encode()
        self.c_device_properties.camera_sensor_model = (String(<char*> camera_sensor_model_))

    ##
    # Name of Camera in DT (ZED_CAM1)
    @property
    def camera_name(self) -> str:
        if not self.c_device_properties.camera_name.empty():
            return self.c_device_properties.camera_name.get().decode()
        else:
            return ""

    @camera_name.setter
    def camera_name(self, str camera_name):
        camera_name_ = camera_name.encode()
        self.c_device_properties.camera_name = (String(<char*> camera_name_))

    ##
    # Input type of the camera.
    @property
    def input_type(self) -> INPUT_TYPE:
        return INPUT_TYPE(<int>self.c_device_properties.input_type)

    @input_type.setter
    def input_type(self, value : INPUT_TYPE):
        if isinstance(value, INPUT_TYPE):
            self.c_device_properties.input_type = <c_INPUT_TYPE>(<int>value.value)
        else:
            raise TypeError("Argument is not of INPUT_TYPE type.")

    ##
    # sensor_address when available (ZED-X HDR/XOne HDR only)
    @property
    def sensor_address_left(self) -> int:
        return self.c_device_properties.sensor_address_left

    @sensor_address_left.setter
    def sensor_address_left(self, int sensor_address_left):
        self.c_device_properties.sensor_address_left = sensor_address_left

    ##
    # sensor_address when available (ZED-X HDR/XOne HDR only)
    @property
    def sensor_address_right(self) -> int:
        return self.c_device_properties.sensor_address_right

    @sensor_address_right.setter
    def sensor_address_right(self, int sensor_address_right):
        self.c_device_properties.sensor_address_right = sensor_address_right

    def __str__(self):
        return to_str(toString(self.c_device_properties)).decode()

    def __repr__(self):
        return to_str(toString(self.c_device_properties)).decode()


##
# Class representing a generic 3*3 matrix.
# \ingroup Core_group
#
# It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on.
# \n The data value of the matrix can be accessed with the \ref r() method.
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
    # Copy the values from another sl.Matrix3f.
    # \param matrix : sl.Matrix3f to copy.
    def init_matrix(self, matrix: Matrix3f) -> None:
        for i in range(9):
            self.mat.r[i] = matrix.mat.r[i]

    ##
    # Sets the sl.Matrix3f to its inverse.
    def inverse(self) -> None:
        self.mat.inverse()

    ##
    # Returns the inverse of a sl.Matrix3f.
    # \param rotation : sl.Matrix3f to compute the inverse from.
    # \return The inverse of the sl.Matrix3f given as input.
    def inverse_mat(self, rotation: Matrix3f) -> Matrix3f:
        out = Matrix3f()
        out.mat[0] = rotation.mat.inverse(rotation.mat[0])
        return out

    ##
    # Sets the sl.Matrix3f to its transpose.
    def transpose(self) -> None:
        self.mat.transpose()

    ##
    # Returns the transpose of a sl.Matrix3f.
    # \param rotation : sl.Matrix3f to compute the transpose from.
    # \return The transpose of the sl.Matrix3f given as input.
    def transpose_mat(self, rotation: Matrix3f) -> Matrix3f:
        out = Matrix3f()
        out.mat[0] = rotation.mat.transpose(rotation.mat[0])
        return out

    ##
    # Sets the sl.Matrix3f to identity.
    # \return itself
    def set_identity(self) -> Matrix3f:
        self.mat.setIdentity()
        return self

    ##
    # Creates an identity sl.Matrix3f.
    # \return A sl.Matrix3f set to identity.
    def identity(self) -> Matrix3f:
        new_mat = Matrix3f()
        return new_mat.set_identity()

    ##
    # Sets the sl.Matrix3f to zero.
    def set_zeros(self) -> None:
        self.mat.setZeros()

    ##
    # Creates a sl.Matrix3f filled with zeros.
    # \return A sl.Matrix3f filled with zeros.
    def zeros(self) -> Matrix3f:
        output_mat = Matrix3f()
        output_mat.mat[0] = self.mat.zeros()
        return output_mat

    ##
    # Returns the components of the sl.Matrix3f in a string.
    # \return A string containing the components of the current sl.Matrix3f.
    def get_infos(self) -> str:
        return to_str(self.mat.getInfos()).decode()

    ##
    # Name of the matrix (optional).
    @property
    def matrix_name(self) -> str:
        if not self.mat.matrix_name.empty():
           return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, name: str):
        self.mat.matrix_name.set(name.encode()) 

    @property
    def nbElem(self) -> int:
        return 9

    ##
    # 3*3 numpy array of inner data.
    @property
    def r(self) -> np.numpy[float][float]:
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
        elif isinstance(other, float) or isinstance(other, int):
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
# Class representing a generic 4*4 matrix.
# \ingroup Core_group
#
# It is defined in a row-major order, it means that, in the value buffer, the entire first row is stored first, followed by the entire second row, and so on.
# \n The data value of the matrix can be accessed with the \ref r() method.
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
    # Copy the values from another sl.Matrix4f.
    # \param matrix : sl.Matrix4f to copy.
    def init_matrix(self, matrix: Matrix4f) -> None:
        for i in range(16):
            self.mat.m[i] = matrix.mat.m[i]

    ##
    # Sets the sl.Matrix4f to its inverse.
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if the inverse has been computed, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) is not (det = 0).
    def inverse(self) -> ERROR_CODE:
        return _error_code_cache.get(<int>(self.mat.inverse()), ERROR_CODE.FAILURE)

    ##
    # Returns the inverse of a sl.Matrix4f.
    # \param rotation : sl.Matrix4f  to compute the inverse from.
    # \return The inverse of the sl.Matrix4f given as input.
    def inverse_mat(self, rotation: Matrix4f) -> Matrix4f:
        out = Matrix4f()
        out.mat[0] = rotation.mat.inverse(rotation.mat[0])
        return out

    ##
    # 	Sets the sl.Matrix4f to its transpose.
    def transpose(self) -> None:
        self.mat.transpose()

    ##
    # Returns the transpose of a sl.Matrix4f.
    # \param rotation : sl.Matrix4f to compute the transpose from.
    # \return The transpose of the sl.Matrix4f given as input.
    def transpose_mat(self, rotation: Matrix4f) -> Matrix4f:
        out = Matrix4f()
        out.mat[0] = rotation.mat.transpose(rotation.mat[0])
        return out

    ##
    # Sets the sl.Matrix4f to identity.
    # \return itself
    def set_identity(self) -> Matrix4f:
        self.mat.setIdentity()
        return self

    ##
    # Creates an identity sl.Matrix4f.
    # \return A sl.Matrix3f set to identity.
    def identity(self) -> Matrix4f:
        new_mat = Matrix4f()
        return new_mat.set_identity()

    ##
    # Sets the sl.Matrix4f to zero.
    def set_zeros(self) -> None:
        self.mat.setZeros()

    ##
    # Creates a sl.Matrix4f filled with zeros.
    # \return A sl.Matrix4f filled with zeros.
    def zeros(self) -> Matrix4f:
        output_mat = Matrix4f()
        output_mat.mat[0] = self.mat.zeros()
        return output_mat

    ##
    # Returns the components of the sl.Matrix4f in a string.
    # \return A string containing the components of the current sl.Matrix4f.
    def get_infos(self) -> str:
        return to_str(self.mat.getInfos()).decode()

    ##
    # Sets a sl.Matrix3f inside the sl.Matrix4f.
    # \note Can be used to set the rotation matrix when the sl.Matrix4f is a pose or an isometric matrix.
    # \param input : Sub-matrix to put inside the sl.Matrix4f.
    # \param row : Index of the row to start the 3x3 block. Must be 0 or 1.
    # \param column : Index of the column to start the 3x3 block. Must be 0 or 1.
    #
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    def set_sub_matrix3f(self, input: Matrix3f, row=0, column=0) -> ERROR_CODE:
        if row != 0 and row != 1 or column != 0 and column != 1:
            raise TypeError("Arguments row and column must be 0 or 1.")
        else:
            return _error_code_cache.get(<int>self.mat.setSubMatrix3f(input.mat[0], row, column), ERROR_CODE.FAILURE)

    ##
    # Sets a 3x1 Vector inside the sl.Matrix4f at the specified column index.
    # \note Can be used to set the translation/position matrix when the sl.Matrix4f is a pose or an isometry.
    # \param input0 : First value of the 3x1 Vector to put inside the sl.Matrix4f.
    # \param input1 : Second value of the 3x1 Vector to put inside the sl.Matrix4f.
    # \param input2 : Third value of the 3x1 Vector to put inside the sl.Matrix4f.
    # \param column : Index of the column to start the 3x3 block. By default, it is the last column (translation for a sl.Pose).
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    def set_sub_vector3f(self, input0: float, input1: float, input2: float, column=3) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.setSubVector3f(Vector3[float](input0, input1, input2), column), ERROR_CODE.FAILURE)

    ##
    # Sets a 4x1 Vector inside the sl.Matrix4f at the specified column index.
    # \param input0 : First value of the 4x1 Vector to put inside the sl.Matrix4f.
    # \param input1 : Second value of the 4x1 Vector to put inside the sl.Matrix4f.
    # \param input2 : Third value of the 4x1 Vector to put inside the sl.Matrix4f.
    # \param input3 : Fourth value of the 4x1 Vector to put inside the sl.Matrix4f.
    # \param column : Index of the column to start the 3x3 block. By default, it is the last column (translation for a sl.Pose).
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    def set_sub_vector4f(self, input0: float, input1: float, input2: float, input3: float, column=3) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.setSubVector4f(Vector4[float](input0, input1, input2, input3), column), ERROR_CODE.FAILURE)

    ##
    # Returns the name of the matrix (optional). 
    @property
    def matrix_name(self) -> str:
        if not self.mat.matrix_name.empty():
            return self.mat.matrix_name.get().decode()
        else:
            return ""

    @matrix_name.setter
    def matrix_name(self, str name):
        self.mat.matrix_name.set(name.encode())

    ##
    # 4*4 numpy array of inner data.
    @property
    def m(self) -> np.numpy[float][float]:
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
        elif isinstance(other, float) or isinstance(other, int):
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
# Lists available camera settings for the camera (contrast, hue, saturation, gain, ...).
# \ingroup Video_group
#
# \warning All \ref VIDEO_SETTINGS are not supported for all camera models. You can find the supported \ref VIDEO_SETTINGS for each ZED camera in our <a href="https://www.stereolabs.com/docs/video/camera-controls#adjusting-camera-settings">documentation</a>.\n\n
# GAIN and EXPOSURE are linked in auto/default mode (see \ref sl.Camera.set_camera_settings()).
#
# | Enumerator |                         |
# |------------|-------------------------|
# | BRIGHTNESS | Brightness control \n Affected value should be between 0 and 8. \note Not available for ZED X/X Mini cameras. |
# | CONTRAST   | Contrast control \n Affected value should be between 0 and 8. \note Not available for ZED X/X Mini cameras. |
# | HUE        | Hue control \n Affected value should be between 0 and 11. \note Not available for ZED X/X Mini cameras. |
# | SATURATION | Saturation control \n Affected value should be between 0 and 8. |
# | SHARPNESS  | Digital sharpening control \n Affected value should be between 0 and 8. |
# | GAMMA      | ISP gamma control \n Affected value should be between 1 and 9. |
# | GAIN       | Gain control \n Affected value should be between 0 and 100 for manual control. \note If EXPOSURE is set to -1 (automatic mode), then GAIN will be automatic as well. |
# | EXPOSURE   | Exposure control \n Affected value should be between 0 and 100 for manual control.\n The exposition is mapped linearly in a percentage of the following max values.\n Special case for <code>EXPOSURE = 0</code> that corresponds to 0.17072ms.\n The conversion to milliseconds depends on the framerate: <ul><li>15fps & <code>EXPOSURE = 100</code> -> 19.97ms</li><li>30fps & <code>EXPOSURE = 100</code> -> 19.97ms</li><li>60fps & <code>EXPOSURE = 100</code> -> 10.84072ms</li><li>100fps & <code>EXPOSURE = 100</code> -> 10.106624ms</li></ul> |
# | AEC_AGC    | Defines if the GAIN and EXPOSURE are in automatic mode or not.\n Setting GAIN or EXPOSURE values will automatically set this value to 0. |
# | AEC_AGC_ROI | Defines the region of interest for automatic exposure/gain computation.\n To be used with the dedicated \ref Camera.set_camera_settings_roi "set_camera_settings_roi()" / \ref Camera.get_camera_settings_roi "get_camera_settings_roi()" methods. |
# | WHITEBALANCE_TEMPERATURE | Color temperature control \n Affected value should be between 2800 and 6500 with a step of 100.\note Setting a value will automatically set WHITEBALANCE_AUTO to 0. |
# | WHITEBALANCE_AUTO | Defines if the white balance is in automatic mode or not. |
# | LED_STATUS | Status of the front LED of the camera.\n Set to 0 to disable the light, 1 to enable the light.\n Default value is on. \note Requires camera firmware 1523 at least. |
# | EXPOSURE_TIME | Real exposure time control in microseconds. \note Only available for ZED X/X Mini cameras.\note Replace EXPOSURE setting. |
# | ANALOG_GAIN | Real analog gain (sensor) control in mDB.\n The range is defined by Jetson DTS and by default [1000-16000]. \note Only available for ZED X/X Mini cameras.\note Replace GAIN settings. |
# | DIGITAL_GAIN | Real digital gain (ISP) as a factor.\n The range is defined by Jetson DTS and by default [1-256]. \note Only available for ZED X/X Mini cameras.\note Replace GAIN settings. |
# | AUTO_EXPOSURE_TIME_RANGE | Range of exposure auto control in micro seconds.\n Used with \ref Camera.set_camera_settings_range "set_camera_settings_range()".\n Min/max range between max range defined in DTS.\n By default: [28000 - <fps_time> or 19000] us. \note Only available for ZED X/X Mini cameras. |
# | AUTO_ANALOG_GAIN_RANGE | Range of sensor gain in automatic control.\n Used with \ref Camera.set_camera_settings_range "set_camera_settings_range()".\n Min/max range between max range defined in DTS.\n By default: [1000 - 16000] mdB. \note Only available for ZED X/X Mini cameras. |
# | AUTO_DIGITAL_GAIN_RANGE | Range of digital ISP gain in automatic control.\n Used with \ref Camera.set_camera_settings_range "set_camera_settings_range()".\n Min/max range between max range defined in DTS.\n By default: [1 - 256]. \note Only available for ZED X/X Mini cameras. |
# | EXPOSURE_COMPENSATION | Exposure-target compensation made after auto exposure.\n Reduces the overall illumination target by factor of F-stops.\n Affected value should be between 0 and 100 (mapped between [-2.0,2.0]).\n Default value is 50, i.e. no compensation applied. \note Only available for ZED X/X Mini cameras. |
# | DENOISING  | Level of denoising applied on both left and right images.\n Affected value should be between 0 and 100.\n Default value is 50. \note Only available for ZED X/X Mini cameras. |
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

    def __lt__(self, other):
        if isinstance(other, VIDEO_SETTINGS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, VIDEO_SETTINGS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, VIDEO_SETTINGS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, VIDEO_SETTINGS):
            return self.value >= other.value
        return NotImplemented

##
# Lists available depth computation modes.
# \ingroup Depth_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | NONE       | No depth map computation.\n Only rectified stereo images will be available. |
# | PERFORMANCE | Computation mode optimized for speed. |
# | QUALITY    | Computation mode designed for challenging areas with untextured surfaces. |
# | ULTRA      | Computation mode that favors edges and sharpness.\n Requires more GPU memory and computation power. |
# | NEURAL_LIGHT     | End to End Neural disparity estimation. \n Requires AI module. |
# | NEURAL     | End to End Neural disparity estimation.\n Requires AI module. |
# | NEURAL_PLUS     | End to End Neural disparity estimation. More precise but requires more GPU memory and computation power. \n Requires AI module. |
class DEPTH_MODE(enum.Enum):
    NONE = <int>c_DEPTH_MODE.NONE
    PERFORMANCE = <int>c_DEPTH_MODE.PERFORMANCE
    QUALITY = <int>c_DEPTH_MODE.QUALITY
    ULTRA = <int>c_DEPTH_MODE.ULTRA
    NEURAL_LIGHT = <int>c_DEPTH_MODE.NEURAL_LIGHT
    NEURAL = <int>c_DEPTH_MODE.NEURAL
    NEURAL_PLUS = <int>c_DEPTH_MODE.NEURAL_PLUS
    LAST = <int>c_DEPTH_MODE.DEPTH_MODE_LAST

    def __lt__(self, other):
        if isinstance(other, DEPTH_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, DEPTH_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, DEPTH_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, DEPTH_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available units for measures.
# \ingroup Core_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | MILLIMETER | International System (1/1000 meters) |
# | CENTIMETER | International System (1/100 meters) |
# | METER      | International System (1 meter) |
# | INCH       | Imperial Unit (1/12 feet) |
# | FOOT       | Imperial Unit (1 foot)  |
class UNIT(enum.Enum):
    MILLIMETER = <int>c_UNIT.MILLIMETER
    CENTIMETER = <int>c_UNIT.CENTIMETER
    METER = <int>c_UNIT.METER
    INCH = <int>c_UNIT.INCH
    FOOT = <int>c_UNIT.FOOT
    LAST = <int>c_UNIT.UNIT_LAST

    def __lt__(self, other):
        if isinstance(other, UNIT):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, UNIT):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, UNIT):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, UNIT):
            return self.value >= other.value
        return NotImplemented


##
# Lists available coordinates systems for positional tracking and 3D measures.
# \image html CoordinateSystem.webp
# \ingroup Core_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | IMAGE | Standard coordinates system in computer vision.\n Used in OpenCV: see <a href="http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">here</a>. |
# | LEFT_HANDED_Y_UP | Left-handed with Y up and Z forward.\n Used in Unity with DirectX. |
# | RIGHT_HANDED_Y_UP | Right-handed with Y pointing up and Z backward.\n Used in OpenGL. |
# | RIGHT_HANDED_Z_UP | Right-handed with Z pointing up and Y forward.\n Used in 3DSMax. |
# | LEFT_HANDED_Z_UP | Left-handed with Z axis pointing up and X forward.\n Used in Unreal Engine. |
# | RIGHT_HANDED_Z_UP_X_FWD | Right-handed with Z pointing up and X forward.\n Used in ROS (REP 103). |
class COORDINATE_SYSTEM(enum.Enum):
    IMAGE = <int>c_COORDINATE_SYSTEM.IMAGE
    LEFT_HANDED_Y_UP = <int>c_COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    RIGHT_HANDED_Y_UP = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    RIGHT_HANDED_Z_UP = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    LEFT_HANDED_Z_UP = <int>c_COORDINATE_SYSTEM.LEFT_HANDED_Z_UP
    RIGHT_HANDED_Z_UP_X_FWD = <int>c_COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    LAST = <int>c_COORDINATE_SYSTEM.COORDINATE_SYSTEM_LAST

    def __str__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_COORDINATE_SYSTEM>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, COORDINATE_SYSTEM):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, COORDINATE_SYSTEM):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, COORDINATE_SYSTEM):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, COORDINATE_SYSTEM):
            return self.value >= other.value
        return NotImplemented

##
# Lists retrievable measures.
# \ingroup Core_group
# | Enumerator |                         |
# |------------|-------------------------|
# | DISPARITY  | Disparity map. Each pixel contains 1 float.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C1" |
# | DEPTH      | Depth map in sl.UNIT defined in sl.InitParameters.coordinate_units. Each pixel contains 1 float.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C1" |
# | CONFIDENCE | Certainty/confidence of the depth map. Each pixel contains 1 float.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C1" |
# | XYZ        | Point cloud. Each pixel contains 4 float (X, Y, Z, not used).\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZRGBA    | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color).\n The color should to be read as an unsigned char[4] representing the RGBA color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZBGRA    | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color).\n The color should to be read as an unsigned char[4] representing the BGRA color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZARGB    | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color).\n The color should to be read as an unsigned char[4] representing the ARGB color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZABGR    | Colored point cloud. Each pixel contains 4 float (X, Y, Z, color).\n The color should to be read as an unsigned char[4] representing the ABGR color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | NORMALS    | Normal vectors map. Each pixel contains 4 float (X, Y, Z, 0).\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | DISPARITY_RIGHT |  Disparity map for right sensor. Each pixel contains 1 float.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C1" |
# | DEPTH_RIGHT | Depth map for right sensor. Each pixel contains 1 float.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C1" |
# | XYZ_RIGHT  | Point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, not used).\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZRGBA_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color).\n The color needs to be read as an unsigned char[4] representing the RGBA color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZBGRA_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color).\n The color needs to be read as an unsigned char[4] representing the BGRA color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZARGB_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color).\n The color needs to be read as an unsigned char[4] representing the ARGB color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | XYZABGR_RIGHT | Colored point cloud for right sensor. Each pixel contains 4 float (X, Y, Z, color).\n The color needs to be read as an unsigned char[4] representing the ABGR color.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | NORMALS_RIGHT | Normal vectors map for right view. Each pixel contains 4 float (X, Y, Z, 0).\n Type: \ref MAT_TYPE "sl.MAT_TYPE.F32_C4" |
# | DEPTH_U16_MM | Depth map in millimeter whatever the sl.UNIT defined in sl.InitParameters.coordinate_units.\n Invalid values are set to 0 and depth values are clamped at 65000.\n Each pixel contains 1 unsigned short.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.U16_C1" |
# | DEPTH_U16_MM_RIGHT | Depth map in millimeter for right sensor. Each pixel contains 1 unsigned short.\n Type: \ref MAT_TYPE "sl.MAT_TYPE.U16_C1" |
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
        return to_str(toString(<c_MEASURE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_MEASURE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, MEASURE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MEASURE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MEASURE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MEASURE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available views.
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | LEFT       | Left BGRA image. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" |
# | RIGHT      | Right BGRA image. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" |
# | LEFT_GRAY  | Left gray image. Each pixel contains 1 unsigned char.\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C1"|
# | RIGHT_GRAY | Right gray image. Each pixel contains 1 unsigned char.\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C1"|
# | LEFT_UNRECTIFIED | Left BGRA unrectified image. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" |
# | RIGHT_UNRECTIFIED | Right BGRA unrectified image. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" |
# | LEFT_UNRECTIFIED_GRAY | Left gray unrectified image. Each pixel contains 1 unsigned char.\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C1" |
# | RIGHT_UNRECTIFIED_GRAY | Right gray unrectified image. Each pixel contains 1 unsigned char.\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C1" |
# | SIDE_BY_SIDE | Left and right image (the image width is therefore doubled). Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" |
# | DEPTH      | Color rendering of the depth. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" \note Use \ref MEASURE "sl.MEASURE.DEPTH" with sl.Camera.retrieve_measure() to get depth values. |
# | CONFIDENCE | Color rendering of the depth confidence. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" \note Use \ref MEASURE "sl.MEASURE.CONFIDENCE" with sl.Camera.retrieve_measure() to get confidence values. |
# | NORMALS    | Color rendering of the normals. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" \note Use \ref MEASURE "sl.MEASURE.NORMALS" with sl.Camera.retrieve_measure() to get normal values. |
# | DEPTH_RIGHT | Color rendering of the right depth mapped on right sensor. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" \note Use \ref MEASURE "sl.MEASURE.DEPTH_RIGHT" with sl.Camera.retrieve_measure() to get depth right values. |
# | NORMALS_RIGHT | Color rendering of the normals mapped on right sensor. Each pixel contains 4 unsigned char (B, G, R, A).\n Type: \ref sl.MAT_TYPE "sl.MAT_TYPE.U8_C4" \note Use \ref MEASURE "sl.MEASURE.NORMALS_RIGHT" with sl.Camera.retrieve_measure() to get normal right values. |
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
        return to_str(toString(<c_VIEW>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_VIEW>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, VIEW):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, VIEW):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, VIEW):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, VIEW):
            return self.value >= other.value
        return NotImplemented

##
# Lists the different states of positional tracking.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | SEARCHING  | \warn DEPRECATED: This state is no longer in use. |
# | OK         | The positional tracking is functioning normally. |
# | OFF        | The positional tracking is currently disabled. |
# | FPS_TOO_LOW | The effective FPS is too low to provide accurate motion tracking results. Consider adjusting performance parameters (e.g., depth mode, camera resolution) to improve tracking quality.|
# | SEARCHING_FLOOR_PLANE | The camera is currently searching for the floor plane to establish its position relative to it. The world reference frame will be set afterward. |
# | UNAVAILABLE | The tracking module was unable to perform tracking from the previous frame to the current frame. |
class POSITIONAL_TRACKING_STATE(enum.Enum):
    SEARCHING = <int>c_POSITIONAL_TRACKING_STATE.SEARCHING
    OK = <int>c_POSITIONAL_TRACKING_STATE.OK
    OFF = <int>c_POSITIONAL_TRACKING_STATE.OFF
    FPS_TOO_LOW = <int>c_POSITIONAL_TRACKING_STATE.FPS_TOO_LOW
    SEARCHING_FLOOR_PLANE = <int>c_POSITIONAL_TRACKING_STATE.SEARCHING_FLOOR_PLANE
    UNAVAILABLE = <int>c_POSITIONAL_TRACKING_STATE.UNAVAILABLE
    LAST = <int>c_POSITIONAL_TRACKING_STATE.POSITIONAL_TRACKING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_STATE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_STATE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_STATE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_STATE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_STATE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_STATE):
            return self.value >= other.value
        return NotImplemented

##
# Report the status of current odom tracking.
# \ingroup PositionalTracking_group
#
# | Enumerator |                            |
# |:----------:|:---------------------------|
# | OK         | The positional tracking module successfully tracked from the previous frame to the current frame. |
# | UNAVAILABLE | The positional tracking module failed to track from the previous frame to the current frame. |
class ODOMETRY_STATUS(enum.Enum):
    OK = <int>c_ODOMETRY_STATUS.OK
    UNAVAILABLE = <int>c_ODOMETRY_STATUS.UNAVAILABLE
    LAST = <int>c_ODOMETRY_STATUS.LAST

    def __str__(self):
        return to_str(toString(<c_ODOMETRY_STATUS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_ODOMETRY_STATUS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, ODOMETRY_STATUS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, ODOMETRY_STATUS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, ODOMETRY_STATUS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, ODOMETRY_STATUS):
            return self.value >= other.value
        return NotImplemented

##
# Report the status of current map tracking.
# \ingroup PositionalTracking_group
#
# | Enumerator  |                            |
# |:-----------:|:---------------------------|
# | OK          | The positional tracking module is operating normally. |
# | LOOP_CLOSED | The positional tracking module detected a loop and corrected its position. |
# | SEARCHING   | The positional tracking module is searching for recognizable areas in the global map to relocate.  |
class SPATIAL_MEMORY_STATUS(enum.Enum):
    OK = <int>c_SPATIAL_MEMORY_STATUS.OK
    LOOP_CLOSED = <int>c_SPATIAL_MEMORY_STATUS.LOOP_CLOSED
    SEARCHING = <int>c_SPATIAL_MEMORY_STATUS.SEARCHING
    LAST = <int>c_SPATIAL_MEMORY_STATUS.LAST

    def __str__(self):
        return to_str(toString(<c_SPATIAL_MEMORY_STATUS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_SPATIAL_MEMORY_STATUS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, SPATIAL_MEMORY_STATUS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SPATIAL_MEMORY_STATUS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SPATIAL_MEMORY_STATUS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SPATIAL_MEMORY_STATUS):
            return self.value >= other.value
        return NotImplemented

##
# Report the status of the positional tracking fusion.
# \ingroup PositionalTracking_group
#
# | Enumerator |                            |
# |:----------:|:---------------------------|
# | VISUAL_INERTIAL | The positional tracking module is fusing visual and inertial data. |
# | VISUAL | The positional tracking module is fusing visual data only. |
# | INERTIAL | The positional tracking module is fusing inertial data only. |
# | GNSS | The positional tracking module is fusing GNSS data only. |
# | VISUAL_INERTIAL_GNSS | The positional tracking module is fusing visual, inertial, and GNSS data. |
# | VISUAL_GNSS | The positional tracking module is fusing visual and GNSS data. |
# | INERTIAL_GNSS | The positional tracking module is fusing inertial and GNSS data. |
# | UNAVAILABLE | The positional tracking module is unavailable. |
class POSITIONAL_TRACKING_FUSION_STATUS(enum.Enum):
    VISUAL_INERTIAL = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.VISUAL_INERTIAL
    VISUAL = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.VISUAL
    INERTIAL = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.INERTIAL
    GNSS = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.GNSS
    VISUAL_INERTIAL_GNSS = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.VISUAL_INERTIAL_GNSS
    VISUAL_GNSS = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.VISUAL_GNSS
    INERTIAL_GNSS = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.INERTIAL_GNSS
    UNAVAILABLE = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.UNAVAILABLE
    LAST = <int>c_POSITIONAL_TRACKING_FUSION_STATUS.LAST

    def __str__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_FUSION_STATUS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_FUSION_STATUS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_FUSION_STATUS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_FUSION_STATUS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_FUSION_STATUS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_FUSION_STATUS):
            return self.value >= other.value
        return NotImplemented

##
# Lists that represents the status of the of GNSS signal.
# \ingroup Sensors_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | UNKNOWN    | No GNSS fix data is available. |
# | SINGLE    | Single Point Positioning. |
# | DGNSS | Differential GNSS. |
# | PPS | Precise Positioning Service. |
# | RTK_FLOAT | Real Time Kinematic Float. |
# | RTK_FIX |  Real Time Kinematic Fixed. |
class GNSS_STATUS(enum.Enum):
    UNKNOWN = <int>c_GNSS_STATUS.UNKNOWN
    SINGLE = <int>c_GNSS_STATUS.SINGLE
    DGNSS = <int>c_GNSS_STATUS.DGNSS
    PPS = <int>c_GNSS_STATUS.PPS
    RTK_FLOAT = <int>c_GNSS_STATUS.RTK_FLOAT
    RTK_FIX = <int>c_GNSS_STATUS.RTK_FIX
    LAST = <int>c_GNSS_STATUS.LAST

    def __str__(self):
        return to_str(toString(<c_GNSS_STATUS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_GNSS_STATUS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, GNSS_STATUS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, GNSS_STATUS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, GNSS_STATUS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, GNSS_STATUS):
            return self.value >= other.value
        return NotImplemented

##
# Lists that represents the mode of GNSS signal.
# \ingroup Sensors_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | UNKNOWN    | No GNSS fix data is available. |
# | NO_FIX    | No GNSS fix is available. |
# | FIX_2D | 2D GNSS fix, providing latitude and longitude coordinates but without altitude information. |
# | FIX_3D | 3D GNSS fix, providing latitude, longitude, and altitude coordinates. |
class GNSS_MODE(enum.Enum):
    UNKNOWN = <int>c_GNSS_MODE.UNKNOWN
    NO_FIX = <int>c_GNSS_MODE.NO_FIX
    FIX_2D = <int>c_GNSS_MODE.FIX_2D
    FIX_3D = <int>c_GNSS_MODE.FIX_3D
    LAST = <int>c_GNSS_MODE.LAST

    def __str__(self):
        return to_str(toString(<c_GNSS_MODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_GNSS_MODE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, GNSS_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, GNSS_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, GNSS_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, GNSS_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists that represents the current GNSS fusion status
# \ingroup Sensors_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | OK    | The GNSS fusion module is calibrated and working successfully. |
# | OFF    | The GNSS fusion module is not enabled. |
# | CALIBRATION_IN_PROGRESS | Calibration of the GNSS/VIO fusion module is in progress. |
# | RECALIBRATION_IN_PROGRESS | Re-alignment of GNSS/VIO data is in progress, leading to potentially inaccurate global position. |
class GNSS_FUSION_STATUS(enum.Enum):
    OK = <int>c_GNSS_FUSION_STATUS.OK
    OFF = <int>c_GNSS_FUSION_STATUS.OFF
    CALIBRATION_IN_PROGRESS = <int>c_GNSS_FUSION_STATUS.CALIBRATION_IN_PROGRESS
    RECALIBRATION_IN_PROGRESS = <int>c_GNSS_FUSION_STATUS.RECALIBRATION_IN_PROGRESS
    LAST = <int>c_GNSS_FUSION_STATUS.LAST
    def __str__(self):
        return to_str(toString(<c_GNSS_FUSION_STATUS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_GNSS_FUSION_STATUS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, GNSS_FUSION_STATUS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, GNSS_FUSION_STATUS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, GNSS_FUSION_STATUS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, GNSS_FUSION_STATUS):
            return self.value >= other.value
        return NotImplemented

##
# Represents a 3d landmark.
cdef class Landmark:
    cdef c_Landmark landmark

    ##
    # The ID of the landmark.
    @property
    def id(self) -> int:
        return self.landmark.id

    @id.setter
    def id(self, int id):
        self.landmark.id = id

    ##
    # The position of the landmark.
    @property
    def position(self) -> list[float]:
        return [self.landmark.position[0], self.landmark.position[1], self.landmark.position[2]]

    @position.setter
    def position(self, position: list[float]):
        if(len(position) == 3):
            self.landmark.position[0] = position[0]
            self.landmark.position[1] = position[1]
            self.landmark.position[2] = position[2]
        else:
            raise ValueError("position must be a list of 3 floats")

##
# Represents the projection of a 3d landmark in the image.
cdef class Landmark2D:
    cdef c_Landmark2D landmark2d

    ##
    # Unique identifier of the corresponding landmark.
    @property
    def id(self) -> int:
        return int(self.landmark2d.id)
    
    ##
    # The position of the landmark in the image.
    @property
    def position(self) -> np.array:
        cdef Vector2[unsigned int] vec
        vec = self.landmark2d.image_position
        return np.array(vec[0], vec[1])

    @position.setter
    def position(self, list position):
        cdef Vector2[unsigned int] vec
        if(len(position) == 2):
            vec[0] = position[0]
            vec[1] = position[1]
            self.landmark2d.image_position = vec
        else:
            raise ValueError("position must be a list of 2 ints")

##
# Lists the different status of the positional tracking
# \ingroup Positional_tracking_group
cdef class PositionalTrackingStatus:
    cdef c_PositionalTrackingStatus positional_status

    ##
    # Represents the current state of Visual-Inertial Odometry (VIO) tracking between the previous frame and the current frame.
    @property
    def odometry_status(self) -> ODOMETRY_STATUS:
        return ODOMETRY_STATUS(<int>self.positional_status.odometry_status)

    @odometry_status.setter
    def odometry_status(self, odometry_status):
        self.positional_status.odometry_status = (<c_ODOMETRY_STATUS> (<int>odometry_status))

    ##
    # Represents the current state of camera tracking in the global map.
    @property
    def spatial_memory_status(self) -> SPATIAL_MEMORY_STATUS:
        return SPATIAL_MEMORY_STATUS(<int>self.positional_status.spatial_memory_status)

    @spatial_memory_status.setter
    def spatial_memory_status(self, spatial_memory_status):
        self.positional_status.spatial_memory_status = (<c_SPATIAL_MEMORY_STATUS> (<int>spatial_memory_status))

    ##
    # Represents the current state of the positional tracking fusion.
    @property
    def tracking_fusion_status(self) -> POSITIONAL_TRACKING_FUSION_STATUS:
        return POSITIONAL_TRACKING_FUSION_STATUS(<int>self.positional_status.tracking_fusion_status)

    @tracking_fusion_status.setter
    def tracking_fusion_status(self, tracking_fusion_status):
        self.positional_status.tracking_fusion_status = (<c_POSITIONAL_TRACKING_FUSION_STATUS> (<int>tracking_fusion_status))


cdef class FusedPositionalTrackingStatus:
    cdef c_FusedPositionalTrackingStatus positional_status

    @property
    def odometry_status(self) -> ODOMETRY_STATUS:
        return ODOMETRY_STATUS(<int>self.positional_status.odometry_status)

    @odometry_status.setter
    def odometry_status(self, odometry_status):
        self.positional_status.odometry_status = (<c_ODOMETRY_STATUS> (<int>odometry_status))

    @property
    def spatial_memory_status(self) -> SPATIAL_MEMORY_STATUS:
        return SPATIAL_MEMORY_STATUS(<int>self.positional_status.spatial_memory_status)

    @spatial_memory_status.setter
    def spatial_memory_status(self, spatial_memory_status):
        self.positional_status.spatial_memory_status = (<c_SPATIAL_MEMORY_STATUS> (<int>spatial_memory_status))

    @property
    def tracking_fusion_status(self) -> POSITIONAL_TRACKING_FUSION_STATUS:
        return POSITIONAL_TRACKING_FUSION_STATUS(<int>self.positional_status.tracking_fusion_status)

    @tracking_fusion_status.setter
    def tracking_fusion_status(self, tracking_fusion_status):
        self.positional_status.tracking_fusion_status = (<c_POSITIONAL_TRACKING_FUSION_STATUS> (<int>tracking_fusion_status))

    @property
    def gnss_status(self) -> GNSS_STATUS:
        return GNSS_STATUS(<int>self.positional_status.gnss_status)

    @gnss_status.setter
    def gnss_status(self, gnss_status):
        self.positional_status.gnss_status = (<c_GNSS_STATUS> (<int>gnss_status))

    @property
    def gnss_mode(self) -> GNSS_MODE:
        return GNSS_STATUS(<int>self.positional_status.gnss_mode)

    @gnss_mode.setter
    def gnss_mode(self, gnss_mode):
        self.positional_status.gnss_mode = (<c_GNSS_MODE> (<int>gnss_mode))

    @property
    def gnss_fusion_status(self) -> GNSS_FUSION_STATUS:
        return GNSS_FUSION_STATUS(<int>self.positional_status.gnss_fusion_status)

    @gnss_fusion_status.setter
    def gnss_fusion_status(self, gnss_fusion_status):
        self.positional_status.gnss_fusion_status = (<c_GNSS_FUSION_STATUS> (<int>gnss_fusion_status))

##
# Lists the mode of positional tracking that can be used.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | GEN_1   | Default mode. Best compromise in performance. |
# | GEN_2   | Second generation of positional tracking, better accuracy but slower performance. |
# | GEN_3   | Next generation of positional tracking, allow better compromise between performance and accuracy. |
class POSITIONAL_TRACKING_MODE(enum.Enum):
    GEN_1 = <int>c_POSITIONAL_TRACKING_MODE.GEN_1
    GEN_2 = <int>c_POSITIONAL_TRACKING_MODE.GEN_2
    GEN_3 = <int>c_POSITIONAL_TRACKING_MODE.GEN_3

    def __str__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_MODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_POSITIONAL_TRACKING_MODE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, POSITIONAL_TRACKING_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists the different states of spatial memory area export.
# \ingroup SpatialMapping_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | SUCCESS    | The spatial memory file has been successfully created. |
# | RUNNING    | The spatial memory is currently being written. |
# | NOT_STARTED | The spatial memory file exportation has not been called. |
# | FILE_EMPTY | The spatial memory contains no data, the file is empty. |
# | FILE_ERROR | The spatial memory file has not been written because of a wrong file name. |
# | SPATIAL_MEMORY_DISABLED | The spatial memory learning is disabled. No file can be created. |
class AREA_EXPORTING_STATE(enum.Enum):
    SUCCESS = <int>c_AREA_EXPORTING_STATE.AREA_EXPORTING_STATE_SUCCESS
    RUNNING = <int>c_AREA_EXPORTING_STATE.RUNNING
    NOT_STARTED = <int>c_AREA_EXPORTING_STATE.NOT_STARTED
    FILE_EMPTY = <int>c_AREA_EXPORTING_STATE.FILE_EMPTY
    FILE_ERROR = <int>c_AREA_EXPORTING_STATE.FILE_ERROR
    SPATIAL_MEMORY_DISABLED = <int>c_AREA_EXPORTING_STATE.SPATIAL_MEMORY_DISABLED
    LAST = <int>c_AREA_EXPORTING_STATE.AREA_EXPORTING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_AREA_EXPORTING_STATE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_AREA_EXPORTING_STATE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, AREA_EXPORTING_STATE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, AREA_EXPORTING_STATE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, AREA_EXPORTING_STATE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, AREA_EXPORTING_STATE):
            return self.value >= other.value
        return NotImplemented

##
# Lists possible types of position matrix used to store camera path and pose.
# \ingroup PositionalTracking_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | WORLD      | The transform of sl.Pose will contain the motion with reference to the world frame (previously called sl.PATH). |
# | CAMERA     | The transform of sl.Pose will contain the motion with reference to the previous camera frame (previously called sl.POSE). |
class REFERENCE_FRAME(enum.Enum):
    WORLD = <int>c_REFERENCE_FRAME.WORLD
    CAMERA = <int>c_REFERENCE_FRAME.CAMERA
    LAST = <int>c_REFERENCE_FRAME.REFERENCE_FRAME_LAST

    def __str__(self):
        return to_str(toString(<c_REFERENCE_FRAME>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_REFERENCE_FRAME>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, REFERENCE_FRAME):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, REFERENCE_FRAME):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, REFERENCE_FRAME):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, REFERENCE_FRAME):
            return self.value >= other.value
        return NotImplemented

##
# Lists possible time references for timestamps or data.
#
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | IMAGE      | The requested timestamp or data will be at the time of the frame extraction. |
# | CURRENT    | The requested timestamp or data will be at the time of the function call. |
class TIME_REFERENCE(enum.Enum):
    IMAGE = <int>c_TIME_REFERENCE.TIME_REFERENCE_IMAGE
    CURRENT = <int>c_TIME_REFERENCE.CURRENT
    LAST = <int>c_TIME_REFERENCE.TIME_REFERENCE_LAST

    def __str__(self):
        return to_str(toString(<c_TIME_REFERENCE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_TIME_REFERENCE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, TIME_REFERENCE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, TIME_REFERENCE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, TIME_REFERENCE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, TIME_REFERENCE):
            return self.value >= other.value
        return NotImplemented

##
# Lists the different states of spatial mapping.
# \ingroup SpatialMapping_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | INITIALIZING | The spatial mapping is initializing. |
# | OK         | The depth and tracking data were correctly integrated in the mapping algorithm. |
# | NOT_ENOUGH_MEMORY | The maximum memory dedicated to the scanning has been reached.\n The mesh will no longer be updated. |
# | NOT_ENABLED | sl.Camera.enable_spatial_mapping() wasn't called or the scanning was stopped and not relaunched. |
# | FPS_TOO_LOW | The effective FPS is too low to give proper results for spatial mapping.\n Consider using performance parameters (\ref DEPTH_MODE "sl.DEPTH_MODE.PERFORMANCE", \ref MAPPING_RESOLUTION "sl.MAPPING_RESOLUTION.LOW", low camera resolution (\ref RESOLUTION "sl.RESOLUTION.VGA/SVGA" or \ref RESOLUTION "sl.RESOLUTION.HD720"). |
class SPATIAL_MAPPING_STATE(enum.Enum):
    INITIALIZING = <int>c_SPATIAL_MAPPING_STATE.INITIALIZING
    OK = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_OK
    NOT_ENOUGH_MEMORY = <int>c_SPATIAL_MAPPING_STATE.NOT_ENOUGH_MEMORY
    NOT_ENABLED = <int>c_SPATIAL_MAPPING_STATE.NOT_ENABLED
    FPS_TOO_LOW = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_FPS_TOO_LOW
    LAST = <int>c_SPATIAL_MAPPING_STATE.SPATIAL_MAPPING_STATE_LAST

##
# Lists the different states of region of interest auto detection.
# \ingroup Depth_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | RUNNING    | The region of interest auto detection is initializing. |
# | READY      | The region of interest mask is ready, if auto_apply was enabled, the region of interest mask is being used |
# | NOT_ENABLED | The region of interest auto detection is not enabled |
class REGION_OF_INTEREST_AUTO_DETECTION_STATE(enum.Enum):
    RUNNING = <int>c_REGION_OF_INTEREST_AUTO_DETECTION_STATE.RUNNING
    READY = <int>c_REGION_OF_INTEREST_AUTO_DETECTION_STATE.READY
    NOT_ENABLED = <int>c_REGION_OF_INTEREST_AUTO_DETECTION_STATE.NOT_ENABLED
    LAST = <int>c_REGION_OF_INTEREST_AUTO_DETECTION_STATE.REGION_OF_INTEREST_AUTO_DETECTION_STATE_LAST

##
# Lists available compression modes for SVO recording.
# \ingroup Video_group
# \note LOSSLESS is an improvement of previous lossless compression (used in ZED Explorer), even if size may be bigger, compression time is much faster.
#
# | Enumerator |                         |
# |------------|-------------------------|
# | LOSSLESS   | PNG/ZSTD (lossless) CPU based compression.\n Average size: 42% of RAW |
# | H264       | H264 (AVCHD) GPU based compression.\n Average size: 1% of RAW \note Requires a NVIDIA GPU. |
# | H265       | H265 (HEVC) GPU based compression.\n Average size: 1% of RAW \note Requires a NVIDIA GPU. |
# | H264_LOSSLESS | H264 Lossless GPU/Hardware based compression.\n Average size: 25% of RAW \n Provides a SSIM/PSNR result (vs RAW) >= 99.9%. \note Requires a NVIDIA GPU. |
# | H265_LOSSLESS | H265 Lossless GPU/Hardware based compression.\n Average size: 25% of RAW \n Provides a SSIM/PSNR result (vs RAW) >= 99.9%. \note Requires a NVIDIA GPU. |
class SVO_COMPRESSION_MODE(enum.Enum):
    LOSSLESS = <int>c_SVO_COMPRESSION_MODE.LOSSLESS
    H264 = <int>c_SVO_COMPRESSION_MODE.H264
    H265 = <int>c_SVO_COMPRESSION_MODE.H265
    H264_LOSSLESS = <int>c_SVO_COMPRESSION_MODE.H264_LOSSLESS
    H265_LOSSLESS = <int>c_SVO_COMPRESSION_MODE.H265_LOSSLESS
    LAST = <int>c_SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LAST

    def __str__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_SVO_COMPRESSION_MODE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, SVO_COMPRESSION_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SVO_COMPRESSION_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SVO_COMPRESSION_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SVO_COMPRESSION_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available memory type.
# \ingroup Core_group
# \note The ZED SDK Python wrapper does not support GPU data storage/access.
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | CPU        | Data will be stored on the CPU (processor side). |
# | GPU        | Data will be stored on the GPU |
# | BOTH        | Data will be stored on both the CPU and GPU memory |
class MEM(enum.Enum):
    CPU = <int>c_MEM.CPU
    GPU = <int>c_MEM.GPU
    BOTH = <int>c_MEM.BOTH

##
# Lists available copy operation on sl.Mat.
# \ingroup Core_group
# \note The ZED SDK Python wrapper does not support GPU data storage/access.
#
# | Enumerator |                         |
# |------------|-------------------------|
# | CPU_CPU    | Copy data from CPU to CPU. |
# | GPU_CPU    | Copy data from GPU to CPU. |
# | CPU_GPU    | Copy data from CPU to GPU. |
# | GPU_GPU    | Copy data from GPU to GPU. |
class COPY_TYPE(enum.Enum):
    CPU_CPU = <int>c_COPY_TYPE.CPU_CPU
    GPU_CPU = <int>c_COPY_TYPE.GPU_CPU
    CPU_GPU = <int>c_COPY_TYPE.CPU_GPU
    GPU_GPU = <int>c_COPY_TYPE.GPU_GPU

##
# Lists available sl.Mat formats.
# \ingroup Core_group
# \note sl.Mat type depends on image or measure type.
# \note For the dependencies, see sl.VIEW and sl.MEASURE.
#
# | Enumerator |                         |
# |------------|-------------------------|
# | F32_C1     | 1-channel matrix of float |
# | F32_C2     | 2-channel matrix of float |
# | F32_C3     | 3-channel matrix of float |
# | F32_C4     | 4-channel matrix of float |
# | U8_C1      | 1-channel matrix of unsigned char |
# | U8_C2      | 2-channel matrix of unsigned char |
# | U8_C3      | 3-channel matrix of unsigned char |
# | U8_C4      | 4-channel matrix of unsigned char |
# | U16_C1     | 1-channel matrix of unsigned short |
# | S8_C4      | 4-channel matrix of signed char |
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
# Lists available sensor types.
# \ingroup Sensors_group
# \note Sensors are not available on \ref MODEL "sl.MODEL.ZED".
#
# | Enumerator |                         |
# |------------|-------------------------|
# | ACCELEROMETER | Three-axis accelerometer sensor to measure the inertial accelerations. |
# | GYROSCOPE | Three-axis gyroscope sensor to measure the angular velocities. |
# | MAGNETOMETER | Three-axis magnetometer sensor to measure the orientation of the device with respect to the Earth's magnetic field. |
# | BAROMETER | Barometer sensor to measure the atmospheric pressure. |
class SENSOR_TYPE(enum.Enum):
    ACCELEROMETER = <int>c_SENSOR_TYPE.ACCELEROMETER
    GYROSCOPE = <int>c_SENSOR_TYPE.GYROSCOPE
    MAGNETOMETER = <int>c_SENSOR_TYPE.MAGNETOMETER
    BAROMETER = <int>c_SENSOR_TYPE.BAROMETER

##
# Lists available measurement units of onboard sensors.
# \ingroup Sensors_group
# \note Sensors are not available on \ref MODEL "sl.MODEL.ZED".
#
# | Enumerator |                         |
# |------------|-------------------------|
# | M_SEC_2    | m/s² (acceleration)     |
# | DEG_SEC    | deg/s (angular velocity) |
# | U_T        | μT (magnetic field)     |
# | HPA        | hPa (atmospheric pressure) |
# | CELSIUS    | °C (temperature)        |
# | HERTZ      | Hz (frequency)          |
class SENSORS_UNIT(enum.Enum):
    M_SEC_2 = <int>c_SENSORS_UNIT.M_SEC_2
    DEG_SEC = <int>c_SENSORS_UNIT.DEG_SEC
    U_T = <int>c_SENSORS_UNIT.U_T
    HPA = <int>c_SENSORS_UNIT.HPA
    CELSIUS = <int>c_SENSORS_UNIT.CELSIUS
    HERTZ = <int>c_SENSORS_UNIT.HERTZ

##
# Lists available module
#
# \ingroup Video_group
# 
# | MODULE | Description |
# |--------------|-------------|
# | ALL       | All modules |
# | DEPTH      | For the depth module (includes all 'measures' in retrieveMeasure) |
# | POSITIONAL_TRACKING          | For the positional tracking module |
# | OBJECT_DETECTION       | For the object detection module |
# | BODY_TRACKING  | For the body tracking module |
# | SPATIAL_MAPPING  | For the spatial mapping module |
class MODULE(enum.Enum):
    ALL = <int>c_MODULE.ALL
    DEPTH = <int>c_MODULE.DEPTH
    POSITIONAL_TRACKING = <int>c_MODULE.POSITIONAL_TRACKING
    OBJECT_DETECTION = <int>c_MODULE.OBJECT_DETECTION
    BODY_TRACKING = <int>c_MODULE.BODY_TRACKING
    SPATIAL_MAPPING = <int>c_MODULE.SPATIAL_MAPPING
    LAST = <int>c_MODULE.MODULE_LAST

    def __str__(self):
        return to_str(toString(<c_MODULE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_MODULE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, MODULE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MODULE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MODULE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MODULE):
            return self.value >= other.value
        return NotImplemented

##
# Lists available object classes.
#
# \ingroup Object_group
# 
# | OBJECT_CLASS | Description |
# |--------------|-------------|
# | PERSON       | For people detection |
# | VEHICLE      | For vehicle detection (cars, trucks, buses, motorcycles, etc.) |
# | BAG          | For bag detection (backpack, handbag, suitcase, etc.) |
# | ANIMAL       | For animal detection (cow, sheep, horse, dog, cat, bird, etc.) |
# | ELECTRONICS  | For electronic device detection (cellphone, laptop, etc.) |
# | FRUIT_VEGETABLE | For fruit and vegetable detection (banana, apple, orange, carrot, etc.) |
# | SPORT        | For sport-related object detection (sport ball, etc.) |
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
        return to_str(toString(<c_OBJECT_CLASS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_CLASS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, OBJECT_CLASS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_CLASS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_CLASS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_CLASS):
            return self.value >= other.value
        return NotImplemented

##
#  List available object subclasses.
#
# Given as hint, when using object tracking an object can change of sl.OBJECT_SUBCLASS while keeping the same sl.OBJECT_CLASS
# (i.e.: frame n: MOTORBIKE, frame n+1: BICYCLE).
# \ingroup Object_group
# 
# | OBJECT_SUBCLASS | OBJECT_CLASS |
# |-----------------|--------------|
# | PERSON          | PERSON       |
# | PERSON_HEAD     | PERSON       |
# | BICYCLE         | VEHICLE      |
# | CAR             | VEHICLE      |
# | MOTORBIKE       | VEHICLE      |
# | BUS             | VEHICLE      |
# | TRUCK           | VEHICLE      |
# | BOAT            | VEHICLE      |
# | BACKPACK        | BAG          |
# | HANDBAG         | BAG          |
# | SUITCASE        | BAG          |
# | BIRD            | ANIMAL       |
# | CAT             | ANIMAL       |
# | DOG             | ANIMAL       |
# | HORSE           | ANIMAL       |
# | SHEEP           | ANIMAL       |
# | COW             | ANIMAL       |
# | CELLPHONE       | ELECTRONICS  |
# | LAPTOP          | ELECTRONICS  |
# | BANANA          | FRUIT_VEGETABLE |
# | APPLE           | FRUIT_VEGETABLE |
# | ORANGE          | FRUIT_VEGETABLE |
# | CARROT          | FRUIT_VEGETABLE |
# | SPORTSBALL      | SPORT        |
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
        return to_str(toString(<c_OBJECT_SUBCLASS>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_SUBCLASS>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, OBJECT_SUBCLASS):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_SUBCLASS):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_SUBCLASS):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_SUBCLASS):
            return self.value >= other.value
        return NotImplemented

##
# Lists the different states of object tracking.
#
# \ingroup Object_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | OFF        | The tracking is not yet initialized.\n The object id is not usable. |
# | OK         | The object is tracked.  |
# | SEARCHING  | The object could not be detected in the image and is potentially occluded.\n The trajectory is estimated. |
# | TERMINATE  | This is the last searching state of the track.\n The track will be deleted in the next sl.Camera.retrieve_objects(). |
class OBJECT_TRACKING_STATE(enum.Enum):
    OFF = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_OFF
    OK = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_OK
    SEARCHING = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_SEARCHING
    TERMINATE = <int>c_OBJECT_TRACKING_STATE.TERMINATE
    LAST = <int>c_OBJECT_TRACKING_STATE.OBJECT_TRACKING_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_TRACKING_STATE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_TRACKING_STATE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, OBJECT_TRACKING_STATE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_TRACKING_STATE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_TRACKING_STATE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_TRACKING_STATE):
            return self.value >= other.value
        return NotImplemented

##
# Lists possible flip modes of the camera.
#
# \ingroup Video_group
#
# | Enumerator |                         |
# |------------|-------------------------|
# | OFF        | No flip applied. Default behavior. |
# | ON         | Images and camera sensors' data are flipped useful when your camera is mounted upside down. |
# | AUTO       | In LIVE mode, use the camera orientation (if an IMU is available) to set the flip mode.\n In SVO mode, read the state of this enum when recorded. |
class FLIP_MODE(enum.Enum):
    OFF = <int>c_FLIP_MODE.OFF
    ON = <int>c_FLIP_MODE.ON
    AUTO = <int>c_FLIP_MODE.AUTO

    def __str__(self):
        return to_str(toString(<c_FLIP_MODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_FLIP_MODE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, FLIP_MODE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, FLIP_MODE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, FLIP_MODE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, FLIP_MODE):
            return self.value >= other.value
        return NotImplemented

##
# Lists the different states of an object's actions.
#
# \ingroup Object_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | IDLE       | The object is staying static. |
# | MOVING     | The object is moving.   |
class OBJECT_ACTION_STATE(enum.Enum):
    IDLE = <int>c_OBJECT_ACTION_STATE.IDLE
    MOVING = <int>c_OBJECT_ACTION_STATE.OBJECT_ACTION_STATE_MOVING
    LAST = <int>c_OBJECT_ACTION_STATE.OBJECT_ACTION_STATE_LAST

    def __str__(self):
        return to_str(toString(<c_OBJECT_ACTION_STATE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_OBJECT_ACTION_STATE>(<int>self.value))).decode()

    def __lt__(self, other):
        if isinstance(other, OBJECT_ACTION_STATE):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OBJECT_ACTION_STATE):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OBJECT_ACTION_STATE):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OBJECT_ACTION_STATE):
            return self.value >= other.value
        return NotImplemented


##
# Class containing data of a detected object such as its \ref bounding_box, \ref label, \ref id and its 3D \ref position.
# \ingroup Object_group
cdef class ObjectData:
    cdef c_ObjectData object_data

    ##
    # Object identification number.
    # It is used as a reference when tracking the object through the frames.
    # \note Only available if sl.ObjectDetectionParameters.enable_tracking is activated.
    # \note Otherwise, it will be set to -1.
    @property
    def id(self) -> int:
        return self.object_data.id

    @id.setter
    def id(self, int id):
        self.object_data.id = id

    ##
    # Unique id to help identify and track AI detections.
    # It can be either generated externally, or by using \ref generate_unique_id() or left empty.
    @property
    def unique_object_id(self) -> str:
        if not self.object_data.unique_object_id.empty():
            return self.object_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.object_data.unique_object_id.set(id_.encode())

    ##
    # Object raw label.
    # It is forwarded from sl.CustomBoxObjectData when using [sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS](\ref OBJECT_DETECTION_MODEL).
    @property
    def raw_label(self) -> int:
        return self.object_data.raw_label

    @raw_label.setter
    def raw_label(self, int raw_label):
        self.object_data.raw_label = raw_label

    ##
    # Object class/category to identify the object type.
    @property
    def label(self) -> OBJECT_CLASS:
        return OBJECT_CLASS(<int>self.object_data.label)

    @label.setter
    def label(self, label):
        if isinstance(label, OBJECT_CLASS):
            self.object_data.label = <c_OBJECT_CLASS>(<int>label.value)
        else:
            raise TypeError("Argument is not of OBJECT_CLASS type.")

    ##
    # Object sub-class/sub-category to identify the object type.
    @property
    def sublabel(self) -> OBJECT_SUBCLASS:
        return OBJECT_SUBCLASS(<int>self.object_data.sublabel)

    @sublabel.setter
    def sublabel(self, sublabel):
        if isinstance(sublabel, OBJECT_SUBCLASS):
            self.object_data.sublabel = <c_OBJECT_SUBCLASS>(<int>sublabel.value)
        else:
            raise TypeError("Argument is not of OBJECT_SUBCLASS type.")

    ##
    # Object tracking state.
    @property
    def tracking_state(self) -> OBJECT_TRACKING_STATE:
        return OBJECT_TRACKING_STATE(<int>self.object_data.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.object_data.tracking_state = <c_OBJECT_TRACKING_STATE>(<int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # Object action state.
    @property
    def action_state(self) -> OBJECT_ACTION_STATE:
        return OBJECT_ACTION_STATE(<int>self.object_data.action_state)

    @action_state.setter
    def action_state(self, action_state):
        if isinstance(action_state, OBJECT_ACTION_STATE):
            self.object_data.action_state = <c_OBJECT_ACTION_STATE>(<int>action_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_ACTION_STATE type.")

    ##
    # Object 3D centroid.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def position(self) -> np.array[float]:
        cdef np.ndarray position = np.zeros(3)
        for i in range(3):
            position[i] = self.object_data.position[i]
        return position

    @position.setter
    def position(self, np.ndarray position):
        for i in range(3):
            self.object_data.position[i] = position[i]

    ##
    # Object 3D velocity.
    # \note It is defined in ```sl.InitParameters.coordinate_units / s``` and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def velocity(self) -> np.array[float]:
        cdef np.ndarray velocity = np.zeros(3)
        for i in range(3):
            velocity[i] = self.object_data.velocity[i]
        return velocity

    @velocity.setter
    def velocity(self, np.ndarray velocity):
        for i in range(3):
            self.object_data.velocity[i] = velocity[i]

    ##
    # 3D bounding box of the object represented as eight 3D points.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \code
    #    1 ------ 2
    #   /        /|
    #  0 ------ 3 |
    #  | Object | 6
    #  |        |/
    #  4 ------ 7
    # \endcode
    @property
    def bounding_box(self) -> np.array[float][float]:
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
    # 2D bounding box of the object represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self) -> np.array[int][int]:
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
    # Detection confidence value of the object.
    # From 0 to 100, a low value means the object might not be localized perfectly or the label (sl.OBJECT_CLASS) is uncertain.
    @property
    def confidence(self) -> float:
        return self.object_data.confidence

    @confidence.setter
    def confidence(self, float confidence):
        self.object_data.confidence = confidence

    ##
    # Mask defining which pixels which belong to the object (in \ref bounding_box_2d and set to 255) and those of the background (set to 0).
    # \warning The mask information is only available for tracked objects ([sl.OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE)) that have a valid depth.
    # \warning Otherwise, the mask will not be initialized (```mask.is_init() == False```).
    @property
    def mask(self) -> Mat:
        mat = Mat()
        mat.mat = self.object_data.mask
        return mat

    @mask.setter
    def mask(self, Mat mat):
        self.object_data.mask = mat.mat

    ##
    # 3D object dimensions: width, height, length.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def dimensions(self) -> np.array[float]:
        cdef np.ndarray dimensions = np.zeros(3)
        for i in range(3):
            dimensions[i] = self.object_data.dimensions[i]
        return dimensions

    @dimensions.setter
    def dimensions(self, np.ndarray dimensions):
        for i in range(3):
            self.object_data.dimensions[i] = dimensions[i]
   
    ##
    # 3D bounding box of the head of the object (a person) represented as eight 3D points.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_bounding_box(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.object_data.head_bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.object_data.head_bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.object_data.head_bounding_box[i].ptr()[j]
        return arr

    ##
    # 2D bounding box of the head of the object (a person) represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_bounding_box_2d(self) -> np.array[int][int]:
        cdef np.ndarray arr = np.zeros((self.object_data.head_bounding_box_2d.size(), 2))
        for i in range(self.object_data.head_bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.object_data.head_bounding_box_2d[i].ptr()[j]
        return arr

    ##
    # 3D centroid of the head of the object (a person).
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_position(self) -> np.array[float]:
        cdef np.ndarray head_position = np.zeros(3)
        for i in range(3):
            head_position[i] = self.object_data.head_position[i]
        return head_position

    @head_position.setter
    def head_position(self, np.ndarray head_position):
        for i in range(3):
            self.object_data.head_position[i] = head_position[i]

    ##
    # Covariance matrix of the 3D position.
    # \note It is represented by its upper triangular matrix value
    # \code
    #      = [p0, p1, p2]
    #        [p1, p3, p4]
    #        [p2, p4, p5]
    # \endcode
    # where pi is ```position_covariance[i]```
    @property
    def position_covariance(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(6)
        for i in range(6) :
            arr[i] = self.object_data.position_covariance[i]
        return arr

    @position_covariance.setter
    def position_covariance(self, np.ndarray position_covariance_):
        for i in range(6) :
            self.object_data.position_covariance[i] = position_covariance_[i]


##
# Class containing data of a detected body/person such as its \ref bounding_box, \ref id and its 3D \ref position.
# \ingroup Body_group
cdef class BodyData:
    cdef c_BodyData body_data

    ##
    # Body/person identification number.
    # It is used as a reference when tracking the body through the frames.
    # \note Only available if sl.BodyTrackingParameters.enable_tracking is activated.
    # \note Otherwise, it will be set to -1.
    @property
    def id(self) -> int:
        return self.body_data.id

    @id.setter
    def id(self, int id):
        self.body_data.id = id

    ##
    # Unique id to help identify and track AI detections.
    # It can be either generated externally, or by using \ref generate_unique_id() or left empty.
    @property
    def unique_object_id(self) -> str:
        if not self.body_data.unique_object_id.empty():
            return self.body_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.body_data.unique_object_id.set(id_.encode())

    ##
    # Body/person tracking state.
    @property
    def tracking_state(self) -> OBJECT_TRACKING_STATE:
        return OBJECT_TRACKING_STATE(<int>self.body_data.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.body_data.tracking_state = <c_OBJECT_TRACKING_STATE>(<int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # Body/person action state.
    @property
    def action_state(self) -> OBJECT_ACTION_STATE:
        return OBJECT_ACTION_STATE(<int>self.body_data.action_state)

    @action_state.setter
    def action_state(self, action_state):
        if isinstance(action_state, OBJECT_ACTION_STATE):
            self.body_data.action_state = <c_OBJECT_ACTION_STATE>(<int>action_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_ACTION_STATE type.")

    ##
    # Body/person 3D centroid.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def position(self) -> np.array[float]:
        cdef np.ndarray position = np.zeros(3)
        for i in range(3):
            position[i] = self.body_data.position[i]
        return position

    @position.setter
    def position(self, np.ndarray position):
        for i in range(3):
            self.body_data.position[i] = position[i]

    ##
    # Body/person 3D velocity.
    # \note It is defined in ```sl.InitParameters.coordinate_units / s``` and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def velocity(self) -> np.array[float]:
        cdef np.ndarray velocity = np.zeros(3)
        for i in range(3):
            velocity[i] = self.body_data.velocity[i]
        return velocity

    @velocity.setter
    def velocity(self, np.ndarray velocity):
        for i in range(3):
            self.body_data.velocity[i] = velocity[i]

    ##
    # 3D bounding box of the body/person represented as eight 3D points.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \code
    #    1 ------ 2
    #   /        /|
    #  0 ------ 3 |
    #  | Object | 6
    #  |        |/
    #  4 ------ 7
    # \endcode
    @property
    def bounding_box(self) -> np.array[float][float]:
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
    # 2D bounding box of the body/person represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self) -> np.array[int][int]:
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
    # Detection confidence value of the body/person.
    # From 0 to 100, a low value means the body might not be localized perfectly.
    @property
    def confidence(self) -> float:
        return self.body_data.confidence

    @confidence.setter
    def confidence(self, float confidence):
        self.body_data.confidence = confidence

    ##
    # NumPy array of detection covariance for each keypoint.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. Their covariances will be 0.
    @property
    def keypoints_covariance(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint_covariances.size(), 6), dtype=np.float32)
        for i in range(self.body_data.keypoint_covariances.size()):
            for j in range(6):
                arr[i,j] = self.body_data.keypoint_covariances[i][j]
        return arr

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
    # Covariance matrix of the 3D position.
    # \note It is represented by its upper triangular matrix value
    # \code
    #      = [p0, p1, p2]
    #        [p1, p3, p4]
    #        [p2, p4, p5]
    # \endcode
    # where pi is ```position_covariance[i]```
    @property
    def position_covariance(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(6)
        for i in range(6) :
            arr[i] = self.body_data.position_covariance[i]
        return arr

    @position_covariance.setter
    def position_covariance(self, np.ndarray position_covariance_):
        for i in range(6) :
            self.body_data.position_covariance[i] = position_covariance_[i]


    ##
    # Mask defining which pixels which belong to the body/person (in \ref bounding_box_2d and set to 255) and those of the background (set to 0).
    # \warning The mask information is only available for tracked bodies ([sl.OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE)) that have a valid depth.
    # \warning Otherwise, the mask will not be initialized (```mask.is_init() == False```).
    @property
    def mask(self) -> Mat:
        mat = Mat()
        mat.mat = self.body_data.mask
        return mat

    @mask.setter
    def mask(self, Mat mat):
        self.body_data.mask = mat.mat

    ##
    # 3D body/person dimensions: width, height, length.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def dimensions(self) -> np.array[float]:
        cdef np.ndarray dimensions = np.zeros(3)
        for i in range(3):
            dimensions[i] = self.body_data.dimensions[i]
        return dimensions

    @dimensions.setter
    def dimensions(self, np.ndarray dimensions):
        for i in range(3):
            self.body_data.dimensions[i] = dimensions[i]
   
    ##
    # Set of useful points representing the human body in 3D.
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. They will have non finite values.
    @property
    def keypoint(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint.size(), 3), dtype=np.float32)
        for i in range(self.body_data.keypoint.size()):
            for j in range(3):
                arr[i,j] = self.body_data.keypoint[i].ptr()[j]
        return arr

    ##
    # Set of useful points representing the human body in 2D.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \warning In some cases, eg. body partially out of the image, some keypoints can not be detected. They will have negatives coordinates.
    @property
    def keypoint_2d(self) -> np.array[int][int]:
        cdef np.ndarray arr = np.zeros((self.body_data.keypoint_2d.size(), 2))
        for i in range(self.body_data.keypoint_2d.size()):
            for j in range(2):
                arr[i,j] = self.body_data.keypoint_2d[i].ptr()[j]
        return arr

    
    ##
    # 3D bounding box of the head of the body/person represented as eight 3D points.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def head_bounding_box(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.body_data.head_bounding_box.size(), 3), dtype=np.float32)
        for i in range(self.body_data.head_bounding_box.size()):
            for j in range(3):
                arr[i,j] = self.body_data.head_bounding_box[i].ptr()[j]
        return arr

    ##
    # 2D bounding box of the head of the body/person represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    @property
    def head_bounding_box_2d(self) -> np.array[int][int]:
        cdef np.ndarray arr = np.zeros((self.body_data.head_bounding_box_2d.size(), 2))
        for i in range(self.body_data.head_bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.body_data.head_bounding_box_2d[i].ptr()[j]
        return arr

    ##
    # 3D centroid of the head of the body/person.
    # \note It is defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def head_position(self) -> np.array[float]:
        cdef np.ndarray head_position = np.zeros(3)
        for i in range(3):
            head_position[i] = self.body_data.head_position[i]
        return head_position

    @head_position.setter
    def head_position(self, np.ndarray head_position):
        for i in range(3):
            self.body_data.head_position[i] = head_position[i]

    ##
    # NumPy array of detection confidences for each keypoint.
    # \note They can not be lower than the sl.BodyTrackingRuntimeParameters.detection_confidence_threshold.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. They will have non finite values.
    @property
    def keypoint_confidence(self) -> np.array[float]:
        cdef np.ndarray out_arr = np.zeros(self.body_data.keypoint_confidence.size())
        for i in range(self.body_data.keypoint_confidence.size()):
            out_arr[i] = self.body_data.keypoint_confidence[i]
        return out_arr

    ##
    # NumPy array of local position (position of the child keypoint with respect to its parent expressed in its parent coordinate frame) for each keypoint.
    # \note They are expressed in [sl.REFERENCE_FRAME.CAMERA](\ref REFERENCE_FRAME) or [sl.REFERENCE_FRAME.WORLD](\ref REFERENCE_FRAME).
    # \warning Not available with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def local_position_per_joint(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.body_data.local_position_per_joint.size(), 3), dtype=np.float32)
        for i in range(self.body_data.local_position_per_joint.size()):
            for j in range(3):
                arr[i,j] = self.body_data.local_position_per_joint[i].ptr()[j]
        return arr

    ##
    # NumPy array of local orientation for each keypoint.
    # \note The orientation is represented by a quaternion.
    # \warning Not available with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def local_orientation_per_joint(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.body_data.local_orientation_per_joint.size(), 4), dtype=np.float32)
        for i in range(self.body_data.local_orientation_per_joint.size()):
            for j in range(4):
                arr[i,j] = self.body_data.local_orientation_per_joint[i].ptr()[j]
        return arr

    ##
    # Global root orientation of the skeleton (NumPy array).
    # The orientation is also represented by a quaternion.
    # \note The global root position is already accessible in \ref keypoint attribute by using the root index of a given sl.BODY_FORMAT.
    # \warning Not available with [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT).
    @property
    def global_root_orientation(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = self.body_data.global_root_orientation[i]
        return arr

##
# Generate a UUID like unique id to help identify and track AI detections.
# \ingroup Object_group
def generate_unique_id():
    return to_str(c_generate_unique_id()).decode()       

##
# Class that store externally detected objects.
# \ingroup Object_group
#
# The objects can be ingested with sl.Camera.ingest_custom_box_objects() to extract 3D and tracking information over time.
cdef class CustomBoxObjectData:
    cdef c_CustomBoxObjectData custom_box_object_data

    ##
    # Unique id to help identify and track AI detections.
    # It can be either generated externally, or by using \ref generate_unique_id() or left empty.
    @property
    def unique_object_id(self) -> str:
        if not self.custom_box_object_data.unique_object_id.empty():
            return self.custom_box_object_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.custom_box_object_data.unique_object_id.set(id_.encode())

    ##
    # 2D bounding box of the object represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self) -> np.array[int][int]:
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
    # Object label.
    # This information is passed-through and can be used to improve object tracking.
    # \note It should define an object class. This means that any similar object (in classification) should share the same label number.
    @property
    def label(self) -> int:
        return self.custom_box_object_data.label

    @label.setter
    def label(self, int label):
        self.custom_box_object_data.label = label

    ##
    # Detection confidence value of the object.
    # \note The value should be in ```[0-1]```.
    # \note It can be used to improve the object tracking.
    @property
    def probability(self) -> float:
        return self.custom_box_object_data.probability

    @probability.setter
    def probability(self, float probability):
        self.custom_box_object_data.probability = probability

    ##
    # Provide hypothesis about the object movements (degrees of freedom or DoF) to improve the object tracking.
    # - true: 2 DoF projected alongside the floor plane. Case for object standing on the ground such as person, vehicle, etc. 
    # \n The projection implies that the objects cannot be superposed on multiple horizontal levels. 
    # - false: 6 DoF (full 3D movements are allowed).
    #
    # \note This parameter cannot be changed for a given object tracking id.
    # \note It is advised to set it by labels to avoid issues.
    @property
    def is_grounded(self) -> bool:
        return self.custom_box_object_data.is_grounded

    @is_grounded.setter
    def is_grounded(self, bool is_grounded):
        self.custom_box_object_data.is_grounded = is_grounded

    ##
    # Provide hypothesis about the object staticity to improve the object tracking.
    #  - true: the object will be assumed to never move nor being moved.
    #  - false: the object will be assumed to be able to move or being moved.
    @property
    def is_static(self) -> bool:
        return self.custom_box_object_data.is_static

    @is_static.setter
    def is_static(self, bool is_static):
        self.custom_box_object_data.is_static = is_static

    ##
    # Maximum tracking time threshold (in seconds) before dropping the tracked object when unseen for this amount of time.
    #  By default, let the tracker decide internally based on the internal sub class of the tracked object.
    @property
    def tracking_timeout(self) -> float:
        return self.custom_box_object_data.tracking_timeout

    @tracking_timeout.setter
    def tracking_timeout(self, float tracking_timeout):
        self.custom_box_object_data.tracking_timeout = tracking_timeout

    ##
    # Maximum tracking distance threshold (in meters) before dropping the tracked object when unseen for this amount of meters.
    #  By default, do not discard tracked object based on distance.
    #  Only valid for static object.
    @property
    def tracking_max_dist(self) -> float:
        return self.custom_box_object_data.tracking_max_dist

    @tracking_max_dist.setter
    def tracking_max_dist(self, float tracking_max_dist):
        self.custom_box_object_data.tracking_max_dist = tracking_max_dist

##
# Class storing externally detected objects.
# \ingroup Object_group
#
# The objects can be ingested with sl.Camera.ingest_custom_mask_objects() to extract 3D and tracking information over time.
cdef class CustomMaskObjectData:
    cdef c_CustomMaskObjectData custom_mask_object_data

    ##
    # Unique id to help identify and track AI detections.
    # It can be either generated externally, or by using \ref generate_unique_id() or left empty.
    @property
    def unique_object_id(self) -> str:
        if not self.custom_mask_object_data.unique_object_id.empty():
            return self.custom_mask_object_data.unique_object_id.get().decode()
        else:
            return ""

    @unique_object_id.setter
    def unique_object_id(self, str id_):
        self.custom_mask_object_data.unique_object_id.set(id_.encode())

    ##
    # 2D bounding box of the object represented as four 2D points starting at the top left corner and rotation clockwise.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_box_2d(self) -> np.array[int][int]:
        cdef np.ndarray arr = np.zeros((self.custom_mask_object_data.bounding_box_2d.size(), 2))
        for i in range(self.custom_mask_object_data.bounding_box_2d.size()):
            for j in range(2):
                arr[i,j] = self.custom_mask_object_data.bounding_box_2d[i].ptr()[j]
        return arr

    @bounding_box_2d.setter
    def bounding_box_2d(self, np.ndarray coordinates):
        cdef Vector2[unsigned int] vec
        self.custom_mask_object_data.bounding_box_2d.clear()
        for i in range(4):
            vec[0] = coordinates[i][0]
            vec[1] = coordinates[i][1]
            self.custom_mask_object_data.bounding_box_2d.push_back(vec)

    ##
    # Object label.
    # This information is passed-through and can be used to improve object tracking.
    # \note It should define an object class. This means that any similar object (in classification) should share the same label number.
    @property
    def label(self) -> int:
        return self.custom_mask_object_data.label

    @label.setter
    def label(self, int label):
        self.custom_mask_object_data.label = label

    ##
    # Detection confidence value of the object.
    # \note The value should be in ```[0-1]```.
    # \note It can be used to improve the object tracking.
    @property
    def probability(self) -> float:
        return self.custom_mask_object_data.probability

    @probability.setter
    def probability(self, float probability):
        self.custom_mask_object_data.probability = probability

    ##
    # Provide hypothesis about the object movements (degrees of freedom or DoF) to improve the object tracking.
    # - true: 2 DoF projected alongside the floor plane. Case for object standing on the ground such as person, vehicle, etc. 
    # \n The projection implies that the objects cannot be superposed on multiple horizontal levels. 
    # - false: 6 DoF (full 3D movements are allowed).
    #
    # \note This parameter cannot be changed for a given object tracking id.
    # \note It is advised to set it by labels to avoid issues.
    @property
    def is_grounded(self) -> bool:
        return self.custom_mask_object_data.is_grounded

    @is_grounded.setter
    def is_grounded(self, bool is_grounded):
        self.custom_mask_object_data.is_grounded = is_grounded

    ##
    # Provide hypothesis about the object staticity to improve the object tracking.
    #  - true: the object will be assumed to never move nor being moved.
    #  - false: the object will be assumed to be able to move or being moved.
    @property
    def is_static(self) -> bool:
        return self.custom_box_object_data.is_static

    @is_static.setter
    def is_static(self, bool is_static):
        self.custom_box_object_data.is_static = is_static

    ##
    # Maximum tracking time threshold (in seconds) before dropping the tracked object when unseen for this amount of time.
    #  By default, let the tracker decide internally based on the internal sub class of the tracked object.
    @property
    def tracking_timeout(self) -> float:
        return self.custom_box_object_data.tracking_timeout

    @tracking_timeout.setter
    def tracking_timeout(self, float tracking_timeout):
        self.custom_box_object_data.tracking_timeout = tracking_timeout

    ##
    # Maximum tracking distance threshold (in meters) before dropping the tracked object when unseen for this amount of meters.
    #  By default, do not discard tracked object based on distance.
    #  Only valid for static object.
    @property
    def tracking_max_dist(self) -> float:
        return self.custom_box_object_data.tracking_max_dist

    @tracking_max_dist.setter
    def tracking_max_dist(self, float tracking_max_dist):
        self.custom_box_object_data.tracking_max_dist = tracking_max_dist

    ##
    # Mask defining which pixels which belong to the object (in \ref bounding_box_2d and set to 255) and those of the background (set to 0).
    @property
    def box_mask(self) -> Mat:
        mat = Mat()
        mat.mat = self.custom_mask_object_data.box_mask
        return mat

    @box_mask.setter
    def box_mask(self, Mat mat):
        self.custom_mask_object_data.box_mask = mat.mat

##
# \brief Semantic of human body parts and order of \ref sl.BodyData.keypoint for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_18".
# \ingroup Body_group
# 
# | BODY_18_PARTS | Keypoint number         |
# |---------------|-------------------------|
# | NOSE          | 0                       |
# | NECK          | 1                       |
# | RIGHT_SHOULDER | 2                      |
# | RIGHT_ELBOW   | 3                       |
# | RIGHT_WRIST   | 4                       |
# | LEFT_SHOULDER | 5                       |
# | LEFT_ELBOW    | 6                       |
# | LEFT_WRIST    | 7                       |
# | RIGHT_HIP     | 8                       |
# | RIGHT_KNEE    | 9                       |
# | RIGHT_ANKLE   | 10                      |
# | LEFT_HIP      | 11                      |
# | LEFT_KNEE     | 12                      |
# | LEFT_ANKLE    | 13                      |
# | RIGHT_EYE     | 14                      |
# | LEFT_EYE      | 15                      |
# | RIGHT_EAR     | 16                      |
# | LEFT_EAR      | 17                      |
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
# \brief Semantic of human body parts and order of \ref sl.BodyData.keypoint for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_34".
# \ingroup Body_group
# 
# | BODY_34_PARTS | Keypoint number         |
# |---------------|-------------------------|
# | PELVIS        | 0                       |
# | NAVAL_SPINE   | 1                       |
# | CHEST_SPINE   | 2                       |
# | NECK          | 3                       |
# | LEFT_CLAVICLE | 4                       |
# | LEFT_SHOULDER | 5                       |
# | LEFT_ELBOW    | 6                       |
# | LEFT_WRIST    | 7                       |
# | LEFT_HAND     | 8                       |
# | LEFT_HANDTIP  | 9                       |
# | LEFT_THUMB    | 10                      |
# | RIGHT_CLAVICLE | 11                     |
# | RIGHT_SHOULDER | 12                     |
# | RIGHT_ELBOW   | 13                      |
# | RIGHT_WRIST   | 14                      |
# | RIGHT_HAND    | 15                      |
# | RIGHT_HANDTIP | 16                      |
# | RIGHT_THUMB   | 17                      |
# | LEFT_HIP      | 18                      |
# | LEFT_KNEE     | 19                      |
# | LEFT_ANKLE    | 20                      |
# | LEFT_FOOT     | 21                      |
# | RIGHT_HIP     | 22                      |
# | RIGHT_KNEE    | 23                      |
# | RIGHT_ANKLE   | 24                      |
# | RIGHT_FOOT    | 25                      |
# | HEAD          | 26                      |
# | NOSE          | 27                      |
# | LEFT_EYE      | 28                      |
# | LEFT_EAR      | 29                      |
# | RIGHT_EYE     | 30                      |
# | RIGHT_EAR     | 31                      |
# | LEFT_HEEL     | 32                      |
# | RIGHT_HEEL    | 33                      |
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
# \brief Semantic of human body parts and order of \ref sl.BodyData.keypoint for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_38".
# \ingroup Body_group
# 
# | BODY_38_PARTS | Keypoint number         |
# |---------------|-------------------------|
# | PELVIS        | 0                       |
# | SPINE_1       | 1                       |
# | SPINE_2       | 2                       |
# | SPINE_3       | 3                       |
# | NECK          | 4                       |
# | NOSE          | 5                       |
# | LEFT_EYE      | 6                       |
# | RIGHT_EYE     | 7                       |
# | LEFT_EAR      | 8                       |
# | RIGHT_EAR     | 9                       |
# | LEFT_CLAVICLE | 10                      |
# | RIGHT_CLAVICLE | 11                     |
# | LEFT_SHOULDER | 12                      |
# | RIGHT_SHOULDER | 13                     |
# | LEFT_ELBOW    | 14                      |
# | RIGHT_ELBOW   | 15                      |
# | LEFT_WRIST    | 16                      |
# | RIGHT_WRIST   | 17                      |
# | LEFT_HIP      | 18                      |
# | RIGHT_HIP     | 19                      |
# | LEFT_KNEE     | 20                      |
# | RIGHT_KNEE    | 21                      |
# | LEFT_ANKLE    | 22                      |
# | RIGHT_ANKLE   | 23                      |
# | LEFT_BIG_TOE  | 24                      |
# | RIGHT_BIG_TOE | 25                      |
# | LEFT_SMALL_TOE | 26                     |
# | RIGHT_SMALL_TOE | 27                    |
# | LEFT_HEEL     | 28                      |
# | RIGHT_HEEL    | 29                      |
# | LEFT_HAND_THUMB_4 | 30                  |
# | RIGHT_HAND_THUMB_4 | 31                 |
# | LEFT_HAND_INDEX_1 | 32                  |
# | RIGHT_HAND_INDEX_1 | 33                 |
# | LEFT_HAND_MIDDLE_4 | 34                 |
# | RIGHT_HAND_MIDDLE_4 | 35                |
# | LEFT_HAND_PINKY_1 | 36                  |
# | RIGHT_HAND_PINKY_1 | 37 |
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
# \brief Report the actual inference precision used
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | FP32       |                         |
# | FP16       |                         |
# | INT8       |                         |
class INFERENCE_PRECISION(enum.Enum):
    FP32 = <int>c_INFERENCE_PRECISION.FP32
    FP16 = <int>c_INFERENCE_PRECISION.FP16
    INT8 = <int>c_INFERENCE_PRECISION.INT8
    LAST = <int>c_INFERENCE_PRECISION.LAST

##
# \brief Lists supported skeleton body models.
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | BODY_18 | 18-keypoint model \n Basic body model |
# | BODY_34 | 34-keypoint model \note Requires body fitting enabled. |
# | BODY_38 | 38-keypoint model \n Including simplified face, hands and feet.\note Early Access |
class BODY_FORMAT(enum.Enum):
    BODY_18 = <int>c_BODY_FORMAT.BODY_18
    BODY_34 = <int>c_BODY_FORMAT.BODY_34
    BODY_38 = <int>c_BODY_FORMAT.BODY_38
    LAST = <int>c_BODY_FORMAT.LAST

##
# \brief Lists supported models for skeleton keypoints selection.
# \ingroup Body_group
# 
# | Enumerator |                         |
# |------------|-------------------------|
# | FULL       | Full keypoint model     |
# | UPPER_BODY | Upper body keypoint model \n Will output only upper body (from hip). |
class BODY_KEYPOINTS_SELECTION(enum.Enum):
    FULL = <int>c_BODY_KEYPOINTS_SELECTION.FULL
    UPPER_BODY = <int>c_BODY_KEYPOINTS_SELECTION.UPPER_BODY
    LAST = <int>c_BODY_KEYPOINTS_SELECTION.LAST

##
# \brief Lists links of human body keypoints for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_18".
# \ingroup Body_group
# Useful for display.
BODY_18_BONES = [
        (BODY_18_PARTS.NOSE, BODY_18_PARTS.NECK),
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
        (BODY_18_PARTS.LEFT_EYE, BODY_18_PARTS.LEFT_EAR)
]

##
# \brief Lists links of human body keypoints for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_34".
# \ingroup Body_group
# Useful for display.
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
# \brief Lists links of human body keypoints for \ref BODY_FORMAT "sl.BODY_FORMAT.BODY_38".
# \ingroup Body_group
# Useful for display.
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
# Return associated index of each sl.BODY_18_PARTS.
# \ingroup Body_group
def get_idx(part: BODY_18_PARTS) -> int:
    return c_getIdx(<c_BODY_18_PARTS>(<int>part.value))

##
# Return associated index of each sl.BODY_34_PARTS.
# \ingroup Body_group
def get_idx_34(part: BODY_34_PARTS) -> int:
    return c_getIdx(<c_BODY_34_PARTS>(<int>part.value))

##
# Return associated index of each sl.BODY_38_PARTS.
# \ingroup Body_group
def get_idx_38(part: BODY_38_PARTS) -> int:
    return c_getIdx(<c_BODY_38_PARTS>(<int>part.value))

##
# Class containing batched data of a detected objects from the object detection module.
# \ingroup Object_group
#
# This class can be used to store trajectories.
cdef class ObjectsBatch:
    cdef c_ObjectsBatch objects_batch

    ##
    # Id of the batch.
    @property
    def id(self) -> int:
        return self.objects_batch.id

    @id.setter
    def id(self, int value):
        self.objects_batch.id = value

    ##
    # Objects class/category to identify the object type.
    @property
    def label(self) -> OBJECT_CLASS:
        return OBJECT_CLASS(<int>self.objects_batch.label)

    @label.setter
    def label(self, label):
        if isinstance(label, OBJECT_CLASS):
            self.objects_batch.label = <c_OBJECT_CLASS>(<int>label.value)
        else:
            raise TypeError("Argument is not of OBJECT_CLASS type.")

    ##
    # Objects sub-class/sub-category to identify the object type.
    @property
    def sublabel(self) -> OBJECT_SUBCLASS:
        return OBJECT_SUBCLASS(<int>self.objects_batch.sublabel)

    @sublabel.setter
    def sublabel(self, sublabel):
        if isinstance(sublabel, OBJECT_SUBCLASS):
            self.objects_batch.sublabel = <c_OBJECT_SUBCLASS>(<int>sublabel.value)
        else:
            raise TypeError("Argument is not of c_OBJECT_SUBCLASS type.")

    ##
    # Objects tracking state.
    @property
    def tracking_state(self) -> OBJECT_TRACKING_STATE:
        return OBJECT_TRACKING_STATE(<int>self.objects_batch.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.objects_batch.tracking_state = <c_OBJECT_TRACKING_STATE>(<int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # NumPy array of positions for each object.
    @property
    def positions(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.positions.size(), 3), dtype=np.float32)
        for i in range(self.objects_batch.positions.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.positions[i].ptr()[j]
        return arr

    ##
    # NumPy array of positions' covariances for each object.
    @property
    def position_covariances(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.position_covariances.size(), 6), dtype=np.float32)
        for i in range(self.objects_batch.position_covariances.size()):
            for j in range(6):
                arr[i,j] = self.objects_batch.position_covariances[i][j]
        return arr

    ##
    # NumPy array of 3D velocities for each object.
    @property
    def velocities(self)-> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.velocities.size(), 3), dtype=np.float32)
        for i in range(self.objects_batch.velocities.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.velocities[i].ptr()[j]
        return arr

    ##
    # List of timestamps for each object.
    @property
    def timestamps(self) -> list[Timestamp]:
        out_ts = []
        for i in range(self.objects_batch.timestamps.size()):
            ts = Timestamp()
            ts.timestamp = self.objects_batch.timestamps[i] 
            out_ts.append(ts)
        return out_ts

    ##
    # NumPy array of 3D bounding boxes for each object.
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \code
    #    1 ------ 2
    #   /        /|
    #  0 ------ 3 |
    #  | Object | 6
    #  |        |/
    #  4 ------ 7
    # \endcode
    @property
    def bounding_boxes(self) -> np.array[float][float][float]:
        # A 3D bounding box should have 8 indices, 3 coordinates
        cdef np.ndarray arr = np.zeros((self.objects_batch.bounding_boxes.size(),8,3))
        for i in range(self.objects_batch.bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.objects_batch.bounding_boxes[i][j][k]
        return arr

    ##
    # NumPy array of 2D bounding boxes for each object.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_boxes_2d(self) -> np.array[int][int][int]:
        # A 2D bounding box should have 4 indices, 2 coordinates
        cdef np.ndarray arr = np.zeros((self.objects_batch.bounding_boxes_2d.size(),4,2))
        for i in range(self.objects_batch.bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.objects_batch.bounding_boxes_2d[i][j][k]
        return arr

    ##
    # NumPy array of confidences for each object.
    @property
    def confidences(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.confidences.size()))
        for i in range(self.objects_batch.confidences.size()):
            arr[i] = self.objects_batch.confidences[i]
        return arr

    ##
    # List of action states for each object.
    @property
    def action_states(self) -> list[OBJECT_ACTION_STATE]:
        action_states_out = []
        for i in range(self.objects_batch.action_states.size()):
            action_states_out.append(OBJECT_ACTION_STATE(<int>self.objects_batch.action_states[i]))
        return action_states_out

    ##
	# NumPy array of 2D bounding box of the head for each object (person).
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_bounding_boxes_2d(self) -> np.array[int][int][int]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_bounding_boxes_2d.size(),4,2))
        for i in range(self.objects_batch.head_bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.objects_batch.head_bounding_boxes_2d[i][j][k]
        return arr

    ##
	# NumPy array of 3D bounding box of the head for each object (person).
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_bounding_boxes(self) -> np.array[float][float][float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_bounding_boxes.size(),8,3))
        for i in range(self.objects_batch.head_bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.objects_batch.head_bounding_boxes[i][j][k]
        return arr
		
    ##
	# NumPy array of 3D centroid of the head for each object (person).
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \warning Not available with [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL).
    @property
    def head_positions(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.objects_batch.head_positions.size(),3))
        for i in range(self.objects_batch.head_positions.size()):
            for j in range(3):
                arr[i,j] = self.objects_batch.head_positions[i][j]
        return arr

##
# Class containing the results of the object detection module.
# \ingroup Object_group
#
# The detected objects are listed in \ref object_list.
cdef class Objects:
    cdef c_Objects objects

    ##
    # Timestamp corresponding to the frame acquisition.
    # This value is especially useful for the async mode to synchronize the data.
    @property
    def timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp=self.objects.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.objects.timestamp.data_ns = timestamp

    ##
    # List of detected objects.
    @property
    def object_list(self) -> list[ObjectData]:
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
    # Whether \ref object_list has already been retrieved or not.
    # Default: False
    @property
    def is_new(self) -> bool:
        return self.objects.is_new

    @is_new.setter
    def is_new(self, bool is_new):
        self.objects.is_new = is_new

    ##
    # Whether both the object tracking and the world orientation has been setup.
    # Default: False
    @property
    def is_tracked(self) -> bool:
        return self.objects.is_tracked

    @is_tracked.setter
    def is_tracked(self, bool is_tracked):
        self.objects.is_tracked = is_tracked


    ##
    # Method that looks for a given object id in the current objects list.
    # \param py_object_data[out] : sl.ObjectData to fill if the search succeeded.
    # \param object_data_id[in] : Id of the sl.ObjectData to search.
    # \return True if found, otherwise False.
    def get_object_data_from_id(self, py_object_data: ObjectData, object_data_id: int) -> bool:
        if isinstance(py_object_data, ObjectData) :
            return self.objects.getObjectDataFromId((<ObjectData>py_object_data).object_data, object_data_id)
        else :
           raise TypeError("Argument is not of ObjectData type.") 

##
# Class containing batched data of a detected bodies/persons from the body tracking module.
# \ingroup Body_group
cdef class BodiesBatch:
    cdef c_BodiesBatch bodies_batch

    ##
    # Id of the batch.
    @property
    def id(self) -> int:
        return self.bodies_batch.id

    @id.setter
    def id(self, int value):
        self.bodies_batch.id = value

    ##
    # Bodies/persons tracking state.
    @property
    def tracking_state(self) -> OBJECT_TRACKING_STATE:
        return OBJECT_TRACKING_STATE(<int>self.bodies_batch.tracking_state)

    @tracking_state.setter
    def tracking_state(self, tracking_state):
        if isinstance(tracking_state, OBJECT_TRACKING_STATE):
            self.bodies_batch.tracking_state = <c_OBJECT_TRACKING_STATE>(<int>tracking_state.value)
        else:
            raise TypeError("Argument is not of OBJECT_TRACKING_STATE type.")

    ##
    # NumPy array of positions for each body/person.
    @property
    def positions(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.positions.size(), 3), dtype=np.float32)
        for i in range(self.bodies_batch.positions.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.positions[i].ptr()[j]
        return arr

    ##
    # NumPy array of positions' covariances for each body/person.
    @property
    def position_covariances(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.position_covariances.size(), 6), dtype=np.float32)
        for i in range(self.bodies_batch.position_covariances.size()):
            for j in range(6):
                arr[i,j] = self.bodies_batch.position_covariances[i][j]
        return arr

    ##
    # NumPy array of 3D velocities for each body/person.
    @property
    def velocities(self)-> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.velocities.size(), 3), dtype=np.float32)
        for i in range(self.bodies_batch.velocities.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.velocities[i].ptr()[j]
        return arr

    ##
    # List of timestamps for each body/person.
    @property
    def timestamps(self) -> list[Timestamp]:
        out_ts = []
        for i in range(self.bodies_batch.timestamps.size()):
            ts = Timestamp()
            ts.timestamp = self.bodies_batch.timestamps[i] 
            out_ts.append(ts)
        return out_ts

    ##
    # NumPy array of 3D bounding boxes for each body/person.
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    # \code
    #    1 ------ 2
    #   /        /|
    #  0 ------ 3 |
    #  | Object | 6
    #  |        |/
    #  4 ------ 7
    # \endcode
    @property
    def bounding_boxes(self) -> np.array[float][float][float]:
        # A 3D bounding box should have 8 indices, 3 coordinates
        cdef np.ndarray arr = np.zeros((self.bodies_batch.bounding_boxes.size(),8,3))
        for i in range(self.bodies_batch.bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.bounding_boxes[i][j][k]
        return arr

    ##
    # NumPy array of 2D bounding boxes for each body/person.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    # \code
    # A ------ B
    # | Object |
    # D ------ C
    # \endcode
    @property
    def bounding_boxes_2d(self) -> np.array[int][int][int]:
        # A 2D bounding box should have 4 indices, 2 coordinates
        cdef np.ndarray arr = np.zeros((self.bodies_batch.bounding_boxes_2d.size(),4,2))
        for i in range(self.bodies_batch.bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.bodies_batch.bounding_boxes_2d[i][j][k]
        return arr

    ##
    # NumPy array of confidences for each body/person.
    @property
    def confidences(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.confidences.size()))
        for i in range(self.bodies_batch.confidences.size()):
            arr[i] = self.bodies_batch.confidences[i]
        return arr

    ##
    # List of action states for each body/person.
    @property
    def action_states(self) -> list[OBJECT_ACTION_STATE]:
        action_states_out = []
        for i in range(self.bodies_batch.action_states.size()):
            action_states_out.append(OBJECT_ACTION_STATE(<int>self.bodies_batch.action_states[i]))
        return action_states_out
    
    ##
	# NumPy array of 2D keypoints for each body/person.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. They will have non finite values.
    @property
    def keypoints_2d(self) -> np.array[int][int][int]:
        # 18 keypoints
        cdef np.ndarray arr = np.zeros((self.bodies_batch.keypoints_2d.size(),self.bodies_batch.keypoints_2d[0].size(),2))
        for i in range(self.bodies_batch.keypoints_2d.size()):
            for j in range(self.bodies_batch.keypoints_2d[0].size()):
                for k in range(2):
                    arr[i,j,k] = self.bodies_batch.keypoints_2d[i][j][k]
        return arr

	##
	# NumPy array of 3D keypoints for each body/person.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. They will have non finite values.
    @property
    def keypoints(self) -> np.array[float][float][float]:
        # 18 keypoints
        cdef np.ndarray arr = np.zeros((self.bodies_batch.keypoints.size(),self.bodies_batch.keypoints[0].size(),3))
        for i in range(self.bodies_batch.keypoints.size()):
            for j in range(self.bodies_batch.keypoints[0].size()):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.keypoints[i][j][k]
        return arr

    ##
	# NumPy array of 2D bounding box of the head for each body/person.
    # \note Expressed in pixels on the original image resolution, ```[0, 0]``` is the top left corner.
    @property
    def head_bounding_boxes_2d(self) -> np.array[int][int][int]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_bounding_boxes_2d.size(),4,2))
        for i in range(self.bodies_batch.head_bounding_boxes_2d.size()):
            for j in range(4):
                for k in range(2):
                    arr[i,j,k] = self.objects_batch.head_bounding_boxes_2d[i][j][k]
        return arr

    ##
	# NumPy array of 3D bounding box of the head for each body/person.
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def head_bounding_boxes(self) -> np.array[float][float][float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_bounding_boxes.size(),8,3))
        for i in range(self.bodies_batch.head_bounding_boxes.size()):
            for j in range(8):
                for k in range(3):
                    arr[i,j,k] = self.bodies_batch.head_bounding_boxes[i][j][k]
        return arr
		
    ##
	# NumPy array of 3D centroid of the head for each body/person.
    # \note They are defined in sl.InitParameters.coordinate_units and expressed in sl.RuntimeParameters.measure3D_reference_frame.
    @property
    def head_positions(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.bodies_batch.head_positions.size(),3))
        for i in range(self.bodies_batch.head_positions.size()):
            for j in range(3):
                arr[i,j] = self.bodies_batch.head_positions[i][j]
        return arr

	##
	# NumPy array of detection confidences NumPy array for each keypoint for each body/person.
    # \note They can not be lower than the sl.BodyTrackingRuntimeParameters.detection_confidence_threshold.
    # \warning In some cases, eg. body partially out of the image or missing depth data, some keypoints can not be detected. They will have non finite values.
    @property
    def keypoint_confidences(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros(self.bodies_batch.keypoint_confidences.size())
        for i in range(self.bodies_batch.keypoint_confidences.size()):
            arr[i] = self.bodies_batch.keypoint_confidences[i]
        return arr

##
#  Class containing the results of the body tracking module.
# \ingroup Body_group
#
# The detected bodies/persons are listed in \ref body_list.
cdef class Bodies:
    cdef c_Bodies bodies

    ##
    # Timestamp corresponding to the frame acquisition.
    # This value is especially useful for the async mode to synchronize the data.
    @property
    def timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp=self.bodies.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.bodies.timestamp.data_ns = timestamp

    ##
    # List of detected bodies/persons.
    @property
    def body_list(self) -> list[BodyData]:
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
    # Whether \ref object_list has already been retrieved or not.
    # Default: False
    @property
    def is_new(self) -> bool:
        return self.bodies.is_new

    @is_new.setter
    def is_new(self, bool is_new):
        self.bodies.is_new = is_new

    ##
    # Whether both the body tracking and the world orientation has been setup.
    # Default: False
    @property
    def is_tracked(self) -> bool:
        return self.bodies.is_tracked

    @is_tracked.setter
    def is_tracked(self, bool is_tracked):
        self.bodies.is_tracked = is_tracked
        
    ##
    # Body format used in sl.BodyTrackingParameters.body_format parameter.
    @property
    def body_format(self) -> BODY_FORMAT:
        return self.bodies.body_format

    @body_format.setter
    def body_format(self, body_format):
        if isinstance(body_format, BODY_FORMAT) :
            self.bodies.body_format = <c_BODY_FORMAT>(<int>body_format.value)
        else :
            raise TypeError()

    ##
    # Status of the actual inference precision mode used to detect the bodies/persons.
    # \note It depends on the GPU hardware support, the sl.BodyTrackingParameters.allow_reduced_precision_inference input parameter and the model support.
    @property
    def inference_precision_mode(self) -> INFERENCE_PRECISION:
        return self.bodies.inference_precision_mode

    @inference_precision_mode.setter
    def inference_precision_mode(self, inference_precision_mode):
        if isinstance(inference_precision_mode, INFERENCE_PRECISION) :
            self.bodies.inference_precision_mode = <c_INFERENCE_PRECISION>(<int>inference_precision_mode.value)
        else :
            raise TypeError()

    ##
    # Method that looks for a given body id in the current bodies list.
    # \param py_body_data[out] : sl.BodyData to fill if the search succeeded.
    # \param body_data_id[in] : Id of the sl.BodyData to search.
    # \return True if found, otherwise False.
    def get_body_data_from_id(self, py_body_data: BodyData, body_data_id: int) -> bool:
        if isinstance(py_body_data, BodyData) :
            return self.bodies.getBodyDataFromId((<BodyData>py_body_data).body_data, body_data_id)
        else :
           raise TypeError("Argument is not of ObjectData type.") 

##
# Class containing a set of parameters for batch object detection.
# \ingroup Object_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class BatchParameters:
    cdef c_BatchParameters* batch_params

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # param enable : Activates \ref enable
    # param id_retention_time : Chosen \ref id_retention_time
    # param batch_duration : Chosen \ref latency
    def __cinit__(self, enable=False, id_retention_time=240, batch_duration=2.0) -> BatchParameters:
        self.batch_params = new c_BatchParameters(<bool>enable, <float>(id_retention_time), <float>batch_duration)

    def __dealloc__(self):
        del self.batch_params

    ##
    # Whether to enable the batch option in the object detection module.
    # Batch queueing system provides:
    # - deep-learning based re-identification
    # - trajectory smoothing and filtering
    #
    # Default: False
    # \note To activate this option, \ref enable must be set to True.
    @property
    def enable(self) -> bool:
        return self.batch_params.enable

    @enable.setter
    def enable(self, value: bool):
        self.batch_params.enable = value

    ##
    # Max retention time in seconds of a detected object.
    # After this time, the same object will mostly have a different id.
    @property
    def id_retention_time(self) -> float:
        return self.batch_params.id_retention_time

    @id_retention_time.setter
    def id_retention_time(self, value):
        self.batch_params.id_retention_time = value

    ##
    # Trajectories will be output in batch with the desired latency in seconds.
    # During this waiting time, re-identification of objects is done in the background.
    # \note Specifying a short latency will limit the search (falling in timeout) for previously seen object ids but will be closer to real time output.
    # \note Specifying a long latency will reduce the change of timeout in re-identification but increase difference with live output.
    @property
    def latency(self) -> float:
        return self.batch_params.latency

    @latency.setter
    def latency(self, value):
        self.batch_params.latency = value

##
# Class containing a set of parameters for the object detection module.
# \ingroup Object_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class ObjectDetectionParameters:
    cdef c_ObjectDetectionParameters* object_detection

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # \param enable_tracking : Activates \ref enable_tracking
    # \param enable_segmentation : Activates \ref enable_segmentation
    # \param detection_model : Chosen \ref detection_model
    # \param max_range : Chosen \ref max_range
    # \param batch_trajectories_parameters : Chosen \ref batch_parameters
    # \param filtering_mode : Chosen \ref filtering_mode
    # \param prediction_timeout_s : Chosen \ref prediction_timeout_s
    # \param allow_reduced_precision_inference : Activates \ref allow_reduced_precision_inference
    # \param instance_module_id : Chosen \ref instance_module_id
    def __cinit__(self, enable_tracking=True
                , enable_segmentation=False, detection_model=OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
                , max_range=-1.0 , batch_trajectories_parameters=BatchParameters()
                , filtering_mode = OBJECT_FILTERING_MODE.NMS3D
                , prediction_timeout_s = 0.2
                , allow_reduced_precision_inference = False
                , instance_module_id = 0
                , fused_objects_group_name = ""
                , custom_onnx_file = ""
                , custom_onnx_dynamic_input_shape = Resolution(512, 512)
    ) -> ObjectDetectionParameters:
        res = c_Resolution(custom_onnx_dynamic_input_shape.width, custom_onnx_dynamic_input_shape.height)
        fused_objects_group_name = (<str>fused_objects_group_name).encode()
        custom_onnx_filename = (<str>custom_onnx_file).encode()
        self.object_detection = new c_ObjectDetectionParameters(enable_tracking
                                                                , enable_segmentation, <c_OBJECT_DETECTION_MODEL>(<int>detection_model.value)
                                                                , max_range, (<BatchParameters>batch_trajectories_parameters).batch_params[0]
                                                                , <c_OBJECT_FILTERING_MODE>(<int>filtering_mode.value)
                                                                , prediction_timeout_s
                                                                , allow_reduced_precision_inference
                                                                , instance_module_id
                                                                , String(<char*>fused_objects_group_name)
                                                                , String(<char*>custom_onnx_filename)
                                                                , res
                                                                )

    def __dealloc__(self):
        del self.object_detection

    ##
    # Whether the object detection system includes object tracking capabilities across a sequence of images.
    # Default: True
    @property
    def enable_tracking(self) -> bool:
        return self.object_detection.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, bool enable_tracking):
        self.object_detection.enable_tracking = enable_tracking

    ##
    # Whether the object masks will be computed.
    # Default: False
    @property
    def enable_segmentation(self) -> bool:
        return self.object_detection.enable_segmentation

    @enable_segmentation.setter
    def enable_segmentation(self, bool enable_segmentation):
        self.object_detection.enable_segmentation = enable_segmentation

    ##
    # sl.OBJECT_DETECTION_MODEL to use.
    # Default: [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST](\ref OBJECT_DETECTION_MODEL)
    @property
    def detection_model(self) -> OBJECT_DETECTION_MODEL:
        return OBJECT_DETECTION_MODEL(<int>self.object_detection.detection_model)

    @detection_model.setter
    def detection_model(self, detection_model):
        if isinstance(detection_model, OBJECT_DETECTION_MODEL) :
            self.object_detection.detection_model = <c_OBJECT_DETECTION_MODEL>(<int>detection_model.value)
        else :
            raise TypeError()

    ##
    # In a multi camera setup, specify which group this model belongs to.
    # 
    #   In a multi camera setup, multiple cameras can be used to detect objects and multiple detector having similar output layout can see the same object.
    #   Therefore, Fusion will fuse together the outputs received by multiple detectors only if they are part of the same \ref fused_objects_group_name.
    # 
    #   \note This parameter is not used when not using a multi-camera setup and must be set in a multi camera setup.
    @property
    def fused_objects_group_name(self) -> str:
        if not self.object_detection.fused_objects_group_name.empty():
            return self.object_detection.fused_objects_group_name.get().decode()
        else:
            return ""

    @fused_objects_group_name.setter
    def fused_objects_group_name(self, fused_objects_group_name: str):
        self.object_detection.fused_objects_group_name.set(fused_objects_group_name.encode())

    ##
    # Path to the YOLO-like onnx file for custom object detection ran in the ZED SDK.
    # 
    # When `detection_model` is \ref OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS, a onnx model must be passed so that the ZED SDK can optimize it for your GPU and run inference on it.
    # 
    # The resulting optimized model will be saved for re-use in the future.
    # 
    #   \attention - The model must be a YOLO-like model.
    #   \attention - The caching uses the `custom_onnx_file` string along with your GPU specs to decide whether to use the cached optmized model or to optimize the passed onnx model.
    #                If you want to use a different model (i.e. an onnx with different weights), you must use a different `custom_onnx_file` string or delete the cached optimized model in
    #                <ZED Installation path>/resources.
    # 
    #   \note This parameter is useless when detection_model is not \ref OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS.
    @property
    def custom_onnx_file(self) -> str:
        return to_str(self.object_detection.custom_onnx_file).decode()
    
    @custom_onnx_file.setter
    def custom_onnx_file(self, custom_onnx_file: str):
        custom_onnx_filename = (<str>custom_onnx_file).encode()
        self.object_detection.custom_onnx_file = String(<char*>custom_onnx_filename)

    ##
    # \brief Resolution to the YOLO-like onnx file for custom object detection ran in the ZED SDK. This resolution defines the input tensor size for dynamic shape ONNX model only. The batch and channel dimensions are automatically handled, it assumes it's color images like default YOLO models.
    # 
    #   \note This parameter is only used when detection_model is \ref OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS and the provided ONNX file is using dynamic shapes.
    #   \attention - Multiple model only support squared images
    # 
    #   \default Squared images 512x512 (input tensor will be 1x3x512x512)
    @property
    def custom_onnx_dynamic_input_shape(self) -> Resolution:
        res = Resolution()
        res.width = self.object_detection.custom_onnx_dynamic_input_shape.width
        res.height = self.object_detection.custom_onnx_dynamic_input_shape.height
        return res

    @custom_onnx_dynamic_input_shape.setter
    def custom_onnx_dynamic_input_shape(self, custom_onnx_dynamic_input_shape: Resolution):
        self.object_detection.custom_onnx_dynamic_input_shape.width = custom_onnx_dynamic_input_shape.width
        self.object_detection.custom_onnx_dynamic_input_shape.height = custom_onnx_dynamic_input_shape.height

    ##
    # Upper depth range for detections.
    # Default: -1 (value set in sl.InitParameters.depth_maximum_distance)
    # \note The value cannot be greater than sl.InitParameters.depth_maximum_distance and its unit is defined in sl.InitParameters.coordinate_units.
    @property
    def max_range(self) -> float:
        return self.object_detection.max_range

    @max_range.setter
    def max_range(self, float max_range):
        self.object_detection.max_range = max_range

    ##
    # Batching system parameters.
    # Batching system (introduced in 3.5) performs short-term re-identification with deep-learning and trajectories filtering.
    # \n sl.BatchParameters.enable must to be true to use this feature (by default disabled).
    @property
    def batch_parameters(self) -> BatchParameters:
        params = BatchParameters()
        params.enable = self.object_detection.batch_parameters.enable
        params.id_retention_time = self.object_detection.batch_parameters.id_retention_time
        params.latency = self.object_detection.batch_parameters.latency
        return params

    @batch_parameters.setter
    def batch_parameters(self, BatchParameters params):
        self.object_detection.batch_parameters = params.batch_params[0]

    ##
    # Filtering mode that should be applied to raw detections.
    # Default: [sl.OBJECT_FILTERING_MODE.NMS_3D](\ref OBJECT_FILTERING_MODE) (same behavior as previous ZED SDK version)
    # \note This parameter is only used in detection model [sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_XXX](\ref OBJECT_DETECTION_MODEL)
    # and [sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS](\ref OBJECT_DETECTION_MODEL).
    # \note For custom object, it is recommended to use [sl.OBJECT_FILTERING_MODE.NMS_3D_PER_CLASS](\ref OBJECT_FILTERING_MODE)
    # or [sl.OBJECT_FILTERING_MODE.NONE](\ref OBJECT_FILTERING_MODE).
    # \note In this case, you might need to add your own NMS filter before ingesting the boxes into the object detection module.
    @property
    def filtering_mode(self) -> OBJECT_FILTERING_MODE:
        return OBJECT_FILTERING_MODE(<int>self.object_detection.filtering_mode)

    @filtering_mode.setter
    def filtering_mode(self, filtering_mode):
        if isinstance(filtering_mode, OBJECT_FILTERING_MODE) :
            self.object_detection.filtering_mode = <c_OBJECT_FILTERING_MODE>(<int>filtering_mode.value)
        else :
            raise TypeError()

    ##
    # Prediction duration of the ZED SDK when an object is not detected anymore before switching its state to [sl.OBJECT_TRACKING_STATE.SEARCHING](\ref OBJECT_TRACKING_STATE).
    # It prevents the jittering of the object state when there is a short misdetection.
    # \n The user can define their own prediction time duration.
    # \n Default: 0.2
    # \note During this time, the object will have [sl.OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE) state even if it is not detected.
    # \note The duration is expressed in seconds.
    # \warning \ref prediction_timeout_s will be clamped to 1 second as the prediction is getting worse with time.
    # \warning Setting this parameter to 0 disables the ZED SDK predictions.
    @property
    def prediction_timeout_s(self) -> float:
        return self.object_detection.prediction_timeout_s

    @prediction_timeout_s.setter
    def prediction_timeout_s(self, float prediction_timeout_s):
        self.object_detection.prediction_timeout_s = prediction_timeout_s
        
    ##
    # Whether to allow inference to run at a lower precision to improve runtime and memory usage.
    # It might increase the initial optimization time and could include downloading calibration data or calibration cache and slightly reduce the accuracy.
    # \note The fp16 is automatically enabled if the GPU is compatible and provides a speed up of almost x2 and reduce memory usage by almost half, no precision loss.
    # \note This setting allow int8 precision which can speed up by another x2 factor (compared to fp16, or x4 compared to fp32) and half the fp16 memory usage, however some accuracy could be lost.
    # \note The accuracy loss should not exceed 1-2% on the compatible models.
    # \note The current compatible models are all [sl.AI_MODELS.HUMAN_BODY_XXXX](\ref AI_MODELS).
    @property
    def allow_reduced_precision_inference(self) -> bool:
        return self.object_detection.allow_reduced_precision_inference

    @allow_reduced_precision_inference.setter
    def allow_reduced_precision_inference(self, bool allow_reduced_precision_inference):
        self.object_detection.allow_reduced_precision_inference = allow_reduced_precision_inference

    ##
    # Id of the module instance.
    # This is used to identify which object detection module instance is used.
    @property
    def instance_module_id(self) -> int:
        return self.object_detection.instance_module_id

    @instance_module_id.setter
    def instance_module_id(self, unsigned int instance_module_id):
        self.object_detection.instance_module_id = instance_module_id

##
# Class containing a set of runtime parameters for the object detection module.
# \ingroup Object_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class ObjectDetectionRuntimeParameters:
    cdef c_ObjectDetectionRuntimeParameters* object_detection_rt

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # \param detection_confidence_threshold : Chosen \ref detection_confidence_threshold
    # \param object_class_filter : Chosen \ref object_class_filter
    # \param object_class_detection_confidence_threshold : Chosen \ref object_class_detection_confidence_threshold
    def __cinit__(self, detection_confidence_threshold=50, object_class_filter=[], object_class_detection_confidence_threshold={}) -> ObjectDetectionRuntimeParameters:
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
    # Confidence threshold.
    # From 1 to 100, with 1 meaning a low threshold, more uncertain objects and 99 very few but very precise objects.
    # \n Default: 20
    # \note If the scene contains a lot of objects, increasing the confidence can slightly speed up the process, since every object instance is tracked.
    # \note \ref detection_confidence_threshold is used as a fallback when sl::ObjectDetectionRuntimeParameters.object_class_detection_confidence_threshold is partially set.
    @property
    def detection_confidence_threshold(self) -> float:
        return self.object_detection_rt.detection_confidence_threshold

    @detection_confidence_threshold.setter
    def detection_confidence_threshold(self, float detection_confidence_threshold_):
        self.object_detection_rt.detection_confidence_threshold = detection_confidence_threshold_
 
    ##
    # Defines which object types to detect and track.
    # Default: [] (all classes are tracked)
    # \note Fewer object types can slightly speed up the process since every object is tracked.
    # \note Will output only the selected classes.
    #
    # In order to get all the available classes, the filter list must be empty :
    # \code
    # object_class_filter = {};
    # \endcode
    #
    # To select a set of specific object classes, like vehicles, persons and animals for instance:
    # \code
    # object_class_filter = {sl.OBJECT_CLASS.VEHICLE, sl.OBJECT_CLASS.PERSON, sl.OBJECT_CLASS.ANIMAL};
    # \endcode
    @property
    def object_class_filter(self) -> list[OBJECT_CLASS]:
        object_class_filter_out = []
        for i in range(self.object_detection_rt.object_class_filter.size()):
            object_class_filter_out.append(OBJECT_CLASS(<int>self.object_detection_rt.object_class_filter[i]))
        return object_class_filter_out

    @object_class_filter.setter
    def object_class_filter(self, object_class_filter):
        self.object_detection_rt.object_class_filter.clear()
        for i in range(len(object_class_filter)):
            self.object_detection_rt.object_class_filter.push_back(<c_OBJECT_CLASS>(<int>object_class_filter[i].value))
    
    ##
    # Dictonary of confidence thresholds for each class (can be empty for some classes).
    # \note sl.ObjectDetectionRuntimeParameters.detection_confidence_threshold will be taken as fallback/default value.
    @property
    def object_class_detection_confidence_threshold(self) -> dict:
        object_detection_confidence_threshold_out = {}
        cdef map[c_OBJECT_CLASS,float].iterator it = self.object_detection_rt.object_class_detection_confidence_threshold.begin()
        while(it != self.object_detection_rt.object_class_detection_confidence_threshold.end()):
            object_detection_confidence_threshold_out[OBJECT_CLASS(<int>deref(it).first)] = deref(it).second
            postincrement(it)
        return object_detection_confidence_threshold_out

    @object_class_detection_confidence_threshold.setter
    def object_class_detection_confidence_threshold(self, object_class_detection_confidence_threshold_dict):
        self.object_detection_rt.object_class_detection_confidence_threshold.clear()
        for k,v in object_class_detection_confidence_threshold_dict.items():
            self.object_detection_rt.object_class_detection_confidence_threshold[<c_OBJECT_CLASS>(<int>k.value)] = v


##
# Class containing a set of runtime properties of a certain class ID for the object detection module using a custom model.
# \ingroup Object_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class CustomObjectDetectionProperties:
    cdef c_CustomObjectDetectionProperties* custom_object_detection_props

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # \param detection_confidence_threshold : Chosen \ref detection_confidence_threshold
    # \param object_class_filter : Chosen \ref object_class_filter
    # \param object_class_detection_confidence_threshold : Chosen \ref object_class_detection_confidence_threshold
    def __cinit__(self,
                  bool enabled = True,
                  float detection_confidence_threshold = 20.,
                  bool is_grounded = True,
                  bool is_static = False,
                  float tracking_timeout = -1.,
                  float tracking_max_dist = -1.,
                  float max_box_width_normalized = -1.,
                  float min_box_width_normalized = -1.,
                  float max_box_height_normalized = -1.,
                  float min_box_height_normalized = -1.,
                  float max_box_width_meters = -1.,
                  float min_box_width_meters = -1.,
                  float max_box_height_meters = -1.,
                  float min_box_height_meters = -1.,
                  native_mapped_class: OBJECT_SUBCLASS = OBJECT_SUBCLASS.LAST,
                  object_acceleration_preset: OBJECT_ACCELERATION_PRESET = OBJECT_ACCELERATION_PRESET.DEFAULT,
                  float max_allowed_acceleration = NAN
    ) -> CustomObjectDetectionProperties:
        if not isinstance(native_mapped_class, OBJECT_SUBCLASS):
            raise ValueError("native_mapped_class must be an OBJECT_SUBCLASS")
        if not isinstance(object_acceleration_preset, OBJECT_ACCELERATION_PRESET):
            raise ValueError("object_acceleration_preset must be an OBJECT_ACCELERATION_PRESET")
        self.custom_object_detection_props = new c_CustomObjectDetectionProperties(enabled,
                                                                                   detection_confidence_threshold,
                                                                                   is_grounded,
                                                                                   is_static,
                                                                                   tracking_timeout,
                                                                                   tracking_max_dist,
                                                                                   max_box_width_normalized,
                                                                                   min_box_width_normalized,
                                                                                   max_box_height_normalized,
                                                                                   min_box_height_normalized,
                                                                                   max_box_width_meters,
                                                                                   min_box_width_meters,
                                                                                   max_box_height_meters,
                                                                                   min_box_height_meters,
                                                                                   <c_OBJECT_SUBCLASS>(<int>native_mapped_class.value),
                                                                                   <c_OBJECT_ACCELERATION_PRESET>(<int>object_acceleration_preset.value),
                                                                                   max_allowed_acceleration)

    def __dealloc__(self):
        if self.custom_object_detection_props is not NULL:
            del self.custom_object_detection_props

    ##
    # Whether the object object is kept or not
    @property
    def enabled(self) -> bool:
        return self.custom_object_detection_props.enabled

    @enabled.setter
    def enabled(self, bool enabled):
        self.custom_object_detection_props.enabled = enabled

    ##
    # Confidence threshold.
    #
    # From 1 to 100, with 1 meaning a low threshold, more uncertain objects and 99 very few but very precise objects.
    # Default: 20.f
    #
    # \note If the scene contains a lot of objects, increasing the confidence can slightly speed up the process, since every object instance is tracked.
    @property
    def detection_confidence_threshold(self) -> float:
        return self.custom_object_detection_props.detection_confidence_threshold

    @detection_confidence_threshold.setter
    def detection_confidence_threshold(self, float detection_confidence_threshold):
        self.custom_object_detection_props.detection_confidence_threshold = detection_confidence_threshold

    ##
    # Provide hypothesis about the object movements (degrees of freedom or DoF) to improve the object tracking.
    # - true: 2 DoF projected alongside the floor plane. Case for object standing on the ground such as person, vehicle, etc.
    #         The projection implies that the objects cannot be superposed on multiple horizontal levels.
    # - false: 6 DoF (full 3D movements are allowed).
    #
    # \note This parameter cannot be changed for a given object tracking id.
    # \note It is advised to set it by labels to avoid issues.
    @property
    def is_grounded(self) -> bool:
        return self.custom_object_detection_props.is_grounded

    @is_grounded.setter
    def is_grounded(self, bool is_grounded):
        self.custom_object_detection_props.is_grounded = is_grounded

    ##
    # Provide hypothesis about the object staticity to improve the object tracking.
    #   - true: the object will be assumed to never move nor being moved.
    #   - false: the object will be assumed to be able to move or being moved.
    @property
    def is_static(self) -> bool:
        return self.custom_object_detection_props.is_static

    @is_static.setter
    def is_static(self, bool is_static):
        self.custom_object_detection_props.is_static = is_static

    ##
    # Maximum tracking time threshold (in seconds) before dropping the tracked object when unseen for this amount of time.
    #
    # By default, let the tracker decide internally based on the internal sub class of the tracked object.
    @property
    def tracking_timeout(self) -> float:
        return self.custom_object_detection_props.tracking_timeout

    @tracking_timeout.setter
    def tracking_timeout(self, float tracking_timeout):
        self.custom_object_detection_props.tracking_timeout = tracking_timeout

    ##
    # Maximum tracking distance threshold (in meters) before dropping the tracked object when unseen for this amount of meters.
    #
    # By default, do not discard tracked object based on distance.
    # Only valid for static object.
    @property
    def tracking_max_dist(self) -> float:
        return self.custom_object_detection_props.tracking_max_dist

    @tracking_max_dist.setter
    def tracking_max_dist(self, float tracking_max_dist):
        self.custom_object_detection_props.tracking_max_dist = tracking_max_dist

    ##
    # Maximum allowed width normalized to the image size.
    #
    # Any prediction bigger than that will be filtered out.
    # Default: -1 (no filtering)
    @property
    def max_box_width_normalized(self) -> float:
        return self.custom_object_detection_props.max_box_width_normalized

    @max_box_width_normalized.setter
    def max_box_width_normalized(self, float max_box_width_normalized):
        self.custom_object_detection_props.max_box_width_normalized = max_box_width_normalized

    ##
    # Minimum allowed width normalized to the image size.
    #
    # Any prediction smaller than that will be filtered out.
    # Default: -1 (no filtering)
    @property
    def min_box_width_normalized(self) -> float:
        return self.custom_object_detection_props.min_box_width_normalized

    @min_box_width_normalized.setter
    def min_box_width_normalized(self, float min_box_width_normalized):
        self.custom_object_detection_props.min_box_width_normalized = min_box_width_normalized

    ##
    # Maximum allowed height normalized to the image size.
    #
    # Any prediction bigger than that will be filtered out.
    # Default: -1 (no filtering)
    @property
    def max_box_height_normalized(self) -> float:
        return self.custom_object_detection_props.max_box_height_normalized

    @max_box_height_normalized.setter
    def max_box_height_normalized(self, float max_box_height_normalized):
        self.custom_object_detection_props.max_box_height_normalized = max_box_height_normalized

    ##
    # Minimum allowed height normalized to the image size.
    #
    # Any prediction smaller than that will be filtered out.
    # Default: -1 (no filtering)
    @property
    def min_box_height_normalized(self) -> float:
        return self.custom_object_detection_props.min_box_height_normalized

    @min_box_height_normalized.setter
    def min_box_height_normalized(self, float min_box_height_normalized):
        self.custom_object_detection_props.min_box_height_normalized = min_box_height_normalized

    ##
    # Maximum allowed 3D width.
    #
    # Any prediction bigger than that will be either discarded (if object is tracked and in SEARCHING state) or clamped.
    # Default: -1 (no filtering)
    @property
    def max_box_width_meters(self) -> float:
        return self.custom_object_detection_props.max_box_width_meters

    @max_box_width_meters.setter
    def max_box_width_meters(self, float max_box_width_meters):
        self.custom_object_detection_props.max_box_width_meters = max_box_width_meters

    ##
    # Minimum allowed 3D width.
    #
    # Any prediction smaller than that will be either discarded (if object is tracked and in SEARCHING state) or clamped.
    # Default: -1 (no filtering)
    @property
    def min_box_width_meters(self) -> float:
        return self.custom_object_detection_props.min_box_width_meters

    @min_box_width_meters.setter
    def min_box_width_meters(self, float min_box_width_meters):
        self.custom_object_detection_props.min_box_width_meters = min_box_width_meters

    ##
    # Maximum allowed 3D height.
    #
    # Any prediction bigger than that will be either discarded (if object is tracked and in SEARCHING state) or clamped.
    # Default: -1 (no filtering)
    @property
    def max_box_height_meters(self) -> float:
        return self.custom_object_detection_props.max_box_height_meters

    @max_box_height_meters.setter
    def max_box_height_meters(self, float max_box_height_meters):
        self.custom_object_detection_props.max_box_height_meters = max_box_height_meters

    ##
    # Minimum allowed 3D height.
    #
    # Any prediction smaller than that will be either discarded (if object is tracked and in SEARCHING state) or clamped.
    # Default: -1 (no filtering)
    @property
    def min_box_height_meters(self) -> float:
        return self.custom_object_detection_props.min_box_height_meters

    @min_box_height_meters.setter
    def min_box_height_meters(self, float min_box_height_meters):
        self.custom_object_detection_props.min_box_height_meters = min_box_height_meters

    ##
    # For increased accuracy, the native \ref sl::OBJECT_SUBCLASS mapping, if any.
    #
    # Native objects have refined internal parameters for better 3D projection and tracking accuracy.
    # If one of the custom objects can be mapped to one the native \ref sl::OBJECT_SUBCLASS, this can help to boost the tracking accuracy.
    # Default: no mapping
    @property
    def native_mapped_class(self) -> OBJECT_SUBCLASS:
        return OBJECT_SUBCLASS(<int>self.custom_object_detection_props.native_mapped_class)

    @native_mapped_class.setter
    def native_mapped_class(self, native_mapped_class: OBJECT_SUBCLASS):
        if not isinstance(native_mapped_class, OBJECT_SUBCLASS):
            raise ValueError("native_mapped_class must be an OBJECT_SUBCLASS value")
        self.custom_object_detection_props.native_mapped_class =  <c_OBJECT_SUBCLASS>(<int>native_mapped_class.value)

    ##
    # Preset defining the expected maximum acceleration of the tracked object.
    #
    # Determines how the ZED SDK interprets object acceleration, affecting tracking behavior and predictions.
    # Default: Default
    @property
    def object_acceleration_preset(self) -> OBJECT_ACCELERATION_PRESET:
        return OBJECT_ACCELERATION_PRESET(<int>self.custom_object_detection_props.object_acceleration_preset)

    @object_acceleration_preset.setter
    def object_acceleration_preset(self, object_acceleration_preset: OBJECT_ACCELERATION_PRESET):
        if not isinstance(object_acceleration_preset, OBJECT_ACCELERATION_PRESET):
            raise ValueError("object_acceleration_preset must be an OBJECT_ACCELERATION_PRESET value")
        self.custom_object_detection_props.object_acceleration_preset =  <c_OBJECT_ACCELERATION_PRESET>(<int>object_acceleration_preset.value)

    ##
    # Manually override the acceleration preset.
    #
    # If set, this value takes precedence over the selected preset, allowing for a custom maximum acceleration.
    # Unit is m/s^2.
    @property
    def max_allowed_acceleration(self) -> float:
        return self.custom_object_detection_props.max_allowed_acceleration

    @max_allowed_acceleration.setter
    def max_allowed_acceleration(self, float max_allowed_acceleration):
        self.custom_object_detection_props.max_allowed_acceleration = max_allowed_acceleration


##
# Class containing a set of runtime parameters for the object detection module using your own model ran by the SDK.
# \ingroup Object_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class CustomObjectDetectionRuntimeParameters:
    cdef c_CustomObjectDetectionRuntimeParameters* custom_object_detection_rt
    cdef public CustomObjectDetectionProperties _object_detection_properties

    ##
    # Default constructor.
    def __cinit__(self,
                  CustomObjectDetectionProperties object_detection_properties = None,
                  dict object_class_detection_properties = None):
        # Create default properties if none provided
        if object_detection_properties is None:
            object_detection_properties = CustomObjectDetectionProperties()

        self._object_detection_properties = object_detection_properties

        # Initialize the C++ unordered_map
        cdef unordered_map[int, c_CustomObjectDetectionProperties] cpp_map

        # Fill the map if dictionary provided
        if object_class_detection_properties is not None:
            for key, value in object_class_detection_properties.items():
                if not isinstance(value, CustomObjectDetectionProperties):
                    raise TypeError(f"Value for key {key} must be CustomObjectDetectionProperties")
                cpp_map[int(key)] = deref((<CustomObjectDetectionProperties>value).custom_object_detection_props)

        # Create the C++ object
        self.custom_object_detection_rt = new c_CustomObjectDetectionRuntimeParameters(
            deref(self._object_detection_properties.custom_object_detection_props),
            cpp_map
        )

    def __dealloc__(self):
        if self.custom_object_detection_rt is not NULL:
            del self.custom_object_detection_rt

    ##
    # Global object detection properties.
    #
    # \note \ref object_detection_properties is used as a fallback when sl::CustomObjectDetectionRuntimeParameters.object_class_detection_properties is partially set.
    @property
    def object_detection_properties(self) -> CustomObjectDetectionProperties:
        return self._object_detection_properties

    @object_detection_properties.setter
    def object_detection_properties(self, CustomObjectDetectionProperties props):
        if props is None:
            raise ValueError("object_detection_properties cannot be None")
        self._object_detection_properties = props
        self.custom_object_detection_rt.object_detection_properties = deref(props.custom_object_detection_props)

    ##
    # Per class object detection properties.
    @property
    def object_class_detection_properties(self) -> dict:
        """Get the dictionary of class-specific detection properties"""
        result = {}
        cdef pair[int, c_CustomObjectDetectionProperties] item

        for item in self.custom_object_detection_rt.object_class_detection_properties:
            # Create a new CustomObjectDetectionProperties instance for each item
            props = CustomObjectDetectionProperties(
                item.second.enabled,
                item.second.detection_confidence_threshold,
                item.second.is_grounded,
                item.second.is_static,
                item.second.tracking_timeout,
                item.second.tracking_max_dist,
                item.second.max_box_width_normalized,
                item.second.min_box_width_normalized,
                item.second.max_box_height_normalized,
                item.second.min_box_height_normalized,
                item.second.max_box_width_meters,
                item.second.min_box_width_meters,
                item.second.max_box_height_meters,
                item.second.min_box_height_meters,
                OBJECT_SUBCLASS(<int>item.second.native_mapped_class),
                OBJECT_ACCELERATION_PRESET(<int>item.second.object_acceleration_preset),
                item.second.max_allowed_acceleration
            )
            result[item.first] = props

        return result

    @object_class_detection_properties.setter
    def object_class_detection_properties(self, dict props_dict):
        """Set the dictionary of class-specific detection properties"""
        if props_dict is None:
            props_dict = {}

        # Clear existing map
        self.custom_object_detection_rt.object_class_detection_properties.clear()

        # Fill with new values
        for key, value in props_dict.items():
            if not isinstance(value, CustomObjectDetectionProperties):
                raise TypeError(f"Value for key {key} must be CustomObjectDetectionProperties")
            self.custom_object_detection_rt.object_class_detection_properties[int(key)] = \
                deref((<CustomObjectDetectionProperties>value).custom_object_detection_props)

##
# Class containing a set of parameters for the body tracking module.
# \ingroup Body_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class BodyTrackingParameters:
    cdef c_BodyTrackingParameters* bodyTrackingParameters

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # \param enable_tracking : Activates \ref enable_tracking
    # \param enable_segmentation : Activates \ref enable_segmentation
    # \param detection_model : Chosen \ref detection_model
    # \param enable_body_fitting : Activates \ref enable_body_fitting
    # \param max_range : Chosen \ref max_range
    # \param body_format : Chosen \ref body_format
    # \param body_selection : Chosen \ref body_selection
    # \param prediction_timeout_s : Chosen \ref prediction_timeout_s
    # \param allow_reduced_precision_inference : Activates \ref allow_reduced_precision_inference
    # \param instance_module_id : Chosen \ref instance_module_id
    def __cinit__(self, enable_tracking=True
                , enable_segmentation=True, detection_model=BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
                , enable_body_fitting=False, max_range=-1.0
                , body_format=BODY_FORMAT.BODY_18, body_selection=BODY_KEYPOINTS_SELECTION.FULL, prediction_timeout_s = 0.2
                , allow_reduced_precision_inference = False
                , instance_module_id = 0) -> BodyTrackingParameters:
        self.bodyTrackingParameters = new c_BodyTrackingParameters(enable_tracking
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
        if self.bodyTrackingParameters is not NULL:
            del self.bodyTrackingParameters

    ##
    # Whether the body tracking system includes body/person tracking capabilities across a sequence of images.
    # Default: True
    @property
    def enable_tracking(self) -> bool:
        return self.bodyTrackingParameters.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, bool enable_tracking):
        self.bodyTrackingParameters.enable_tracking = enable_tracking

    ##
    # Whether the body/person masks will be computed.
    # Default: False
    @property
    def enable_segmentation(self) -> bool:
        return self.bodyTrackingParameters.enable_segmentation

    @enable_segmentation.setter
    def enable_segmentation(self, bool enable_segmentation):
        self.bodyTrackingParameters.enable_segmentation = enable_segmentation

    ##
    # sl.BODY_TRACKING_MODEL to use.
    # Default: [sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE](\ref BODY_TRACKING_MODEL)
    @property
    def detection_model(self) -> BODY_TRACKING_MODEL:
        return BODY_TRACKING_MODEL(<int>self.bodyTrackingParameters.detection_model)

    @detection_model.setter
    def detection_model(self, detection_model):
        if isinstance(detection_model, BODY_TRACKING_MODEL) :
            self.bodyTrackingParameters.detection_model = <c_BODY_TRACKING_MODEL>(<int>detection_model.value)
        else :
            raise TypeError()

    ##
    # Body format to be outputted by the ZED SDK with sl.Camera.retrieve_bodies().
    # Default: [sl.BODY_FORMAT.BODY_18](\ref BODY_FORMAT)
    @property
    def body_format(self) -> BODY_FORMAT:
        return BODY_FORMAT(<int>self.bodyTrackingParameters.body_format)

    @body_format.setter
    def body_format(self, body_format):
        if isinstance(body_format, BODY_FORMAT):
            self.bodyTrackingParameters.body_format = <c_BODY_FORMAT>(<int>body_format.value)

    ##
    # Selection of keypoints to be outputted by the ZED SDK with sl.Camera.retrieve_bodies().
    # Default: [sl.BODY_KEYPOINTS_SELECTION.FULL](\ref BODY_KEYPOINTS_SELECTION)
    @property
    def body_selection(self) -> BODY_KEYPOINTS_SELECTION:
        return BODY_KEYPOINTS_SELECTION(<int>self.bodyTrackingParameters.body_selection)

    @body_selection.setter
    def body_selection(self, body_selection):
        if isinstance(body_selection, BODY_KEYPOINTS_SELECTION):
            self.bodyTrackingParameters.body_selection = <c_BODY_KEYPOINTS_SELECTION>(<int>body_selection.value)

    ##
    # Whether to apply the body fitting.
    # Default: False
    @property
    def enable_body_fitting(self) -> bool:
        return self.bodyTrackingParameters.enable_body_fitting

    @enable_body_fitting.setter
    def enable_body_fitting(self, bool enable_body_fitting):
        self.bodyTrackingParameters.enable_body_fitting = enable_body_fitting

    ##
    # Upper depth range for detections.
    # Default: -1 (value set in sl.InitParameters.depth_maximum_distance)
    # \note The value cannot be greater than sl.InitParameters.depth_maximum_distance and its unit is defined in sl.InitParameters.coordinate_units.
    @property
    def max_range(self) -> float:
        return self.bodyTrackingParameters.max_range

    @max_range.setter
    def max_range(self, float max_range):
        self.bodyTrackingParameters.max_range = max_range

    ##
    # Prediction duration of the ZED SDK when an object is not detected anymore before switching its state to [sl.OBJECT_TRACKING_STATE.SEARCHING](\ref OBJECT_TRACKING_STATE).
    # It prevents the jittering of the object state when there is a short misdetection.
    # \n The user can define their own prediction time duration.
    # \n Default: 0.2
    # \note During this time, the object will have [sl.OBJECT_TRACKING_STATE.OK](\ref OBJECT_TRACKING_STATE) state even if it is not detected.
    # \note The duration is expressed in seconds.
    # \warning \ref prediction_timeout_s will be clamped to 1 second as the prediction is getting worse with time.
    # \warning Setting this parameter to 0 disables the ZED SDK predictions.
    @property
    def prediction_timeout_s(self) -> float:
        return self.bodyTrackingParameters.prediction_timeout_s

    @prediction_timeout_s.setter
    def prediction_timeout_s(self, float prediction_timeout_s):
        self.bodyTrackingParameters.prediction_timeout_s = prediction_timeout_s
        
    ##
    # Whether to allow inference to run at a lower precision to improve runtime and memory usage.
    # It might increase the initial optimization time and could include downloading calibration data or calibration cache and slightly reduce the accuracy.
    # \note The fp16 is automatically enabled if the GPU is compatible and provides a speed up of almost x2 and reduce memory usage by almost half, no precision loss.
    # \note This setting allow int8 precision which can speed up by another x2 factor (compared to fp16, or x4 compared to fp32) and half the fp16 memory usage, however some accuracy could be lost.
    # \note The accuracy loss should not exceed 1-2% on the compatible models.
    # \note The current compatible models are all [sl.AI_MODELS.HUMAN_BODY_XXXX](\ref AI_MODELS).
    @property
    def allow_reduced_precision_inference(self) -> bool:
        return self.bodyTrackingParameters.allow_reduced_precision_inference

    @allow_reduced_precision_inference.setter
    def allow_reduced_precision_inference(self, bool allow_reduced_precision_inference):
        self.bodyTrackingParameters.allow_reduced_precision_inference = allow_reduced_precision_inference

    ##
    # Id of the module instance.
    # This is used to identify which body tracking module instance is used.
    @property
    def instance_module_id(self) -> int:
        return self.bodyTrackingParameters.instance_module_id

    @instance_module_id.setter
    def instance_module_id(self, unsigned int instance_module_id):
        self.bodyTrackingParameters.instance_module_id = instance_module_id



##
# Class containing a set of runtime parameters for the body tracking module.
# \ingroup Body_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class BodyTrackingRuntimeParameters:
    cdef c_BodyTrackingRuntimeParameters* body_tracking_rt

    ##
    # Default constructor.
    # All the parameters are set to their default values.
    # \param detection_confidence_threshold : Chosen \ref detection_confidence_threshold
    # \param minimum_keypoints_threshold : Chosen \ref minimum_keypoints_threshold
    # \param skeleton_smoothing : Chosen \ref skeleton_smoothing
    def __cinit__(self, detection_confidence_threshold=50, minimum_keypoints_threshold=0, skeleton_smoothing=0) -> BodyTrackingRuntimeParameters:
        self.body_tracking_rt = new c_BodyTrackingRuntimeParameters(detection_confidence_threshold, minimum_keypoints_threshold, skeleton_smoothing)

    def __dealloc__(self):
        del self.body_tracking_rt

    ##
    # Confidence threshold.
    # From 1 to 100, with 1 meaning a low threshold, more uncertain objects and 99 very few but very precise objects.
    # \n Default: 20
    # \note If the scene contains a lot of objects, increasing the confidence can slightly speed up the process, since every object instance is tracked.
    @property
    def detection_confidence_threshold(self) -> float:
        return self.body_tracking_rt.detection_confidence_threshold

    @detection_confidence_threshold.setter
    def detection_confidence_threshold(self, float detection_confidence_threshold_):
        self.body_tracking_rt.detection_confidence_threshold = detection_confidence_threshold_
 
    ##
    # Minimum threshold for the keypoints.
    # The ZED SDK will only output the keypoints of the skeletons with threshold greater than this value.
    # \n Default: 0
    # \note It is useful, for example, to remove unstable fitting results when a skeleton is partially occluded.
    @property
    def minimum_keypoints_threshold(self) -> int:
        return self.body_tracking_rt.minimum_keypoints_threshold

    @minimum_keypoints_threshold.setter
    def minimum_keypoints_threshold(self, int minimum_keypoints_threshold_):
        self.body_tracking_rt.minimum_keypoints_threshold = minimum_keypoints_threshold_

    ##
    # Control of the smoothing of the fitted fused skeleton.
    # It is ranged from 0 (low smoothing) and 1 (high smoothing).
    # \n Default: 0
    @property
    def skeleton_smoothing(self) -> float:
        return self.body_tracking_rt.skeleton_smoothing

    @skeleton_smoothing.setter
    def skeleton_smoothing(self, float skeleton_smoothing_):
        self.body_tracking_rt.skeleton_smoothing = skeleton_smoothing_

##
# Class containing a set of parameters for the plane detection functionality.
# \ingroup SpatialMapping_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class PlaneDetectionParameters:
    cdef c_PlaneDetectionParameters* plane_detection_params

    ##
    # Default constructor.
    # Values:
    # - \ref max_distance_threshold : 0.15 meters
    # - \ref normal_similarity_threshold : 15.0 degrees
    def __cinit__(self) -> PlaneDetectionParameters:
        self.plane_detection_params = new c_PlaneDetectionParameters()

    def __dealloc__(self):
        del self.plane_detection_params

    ##
    # Controls the spread of plane by checking the position difference.
    # Default: 0.15 meters
    @property
    def max_distance_threshold(self) -> float:
        return self.plane_detection_params.max_distance_threshold

    @max_distance_threshold.setter
    def max_distance_threshold(self, float max_distance_threshold_):
        self.plane_detection_params.max_distance_threshold = max_distance_threshold_

    ##
    # Controls the spread of plane by checking the angle difference.
    # Default: 15 degrees
    @property
    def normal_similarity_threshold(self) -> float:
        return self.plane_detection_params.normal_similarity_threshold

    @normal_similarity_threshold.setter
    def normal_similarity_threshold(self, float normal_similarity_threshold_):
        self.plane_detection_params.normal_similarity_threshold = normal_similarity_threshold_


cdef class RegionOfInterestParameters:
    cdef c_RegionOfInterestParameters* roi_params

    def __cinit__(self) -> RegionOfInterestParameters:
        self.roi_params = new c_RegionOfInterestParameters()

    def __dealloc__(self):
        del self.roi_params

    ##
    # Filtering how far object in the ROI should be considered, this is useful for a vehicle for instance
    # Default: 2.5 meters
    @property
    def depth_far_threshold_meters(self) -> float:
        return self.roi_params.depth_far_threshold_meters
    
    @depth_far_threshold_meters.setter
    def depth_far_threshold_meters(self, float depth_far_threshold_meters_):
        self.roi_params.depth_far_threshold_meters = depth_far_threshold_meters_
    
    ##
    # By default consider only the lower half of the image, can be useful to filter out the sky
    # Default: 0.5, correspond to the lower half of the image
    @property
    def image_height_ratio_cutoff(self) -> float:
        return self.roi_params.image_height_ratio_cutoff
    
    @image_height_ratio_cutoff.setter
    def image_height_ratio_cutoff(self, float image_height_ratio_cutoff_):
        self.roi_params.image_height_ratio_cutoff = image_height_ratio_cutoff_
    
    ##
    # Once computed the ROI computed will be automatically applied
    # Default: Enabled
    @property
    def auto_apply_module(self) -> set[MODULE]:
        auto_apply_module_out = set()
        cdef unordered_set[c_MODULE].iterator it = self.roi_params.auto_apply_module.begin()
        while(it != self.roi_params.auto_apply_module.end()):
            auto_apply_module_out.add(MODULE(deref(it)))
            postincrement(it)
        return auto_apply_module_out
    
    @auto_apply_module.setter
    def auto_apply_module(self, auto_apply_module):
        self.roi_params.auto_apply_module.clear()
        for v in auto_apply_module:
            self.roi_params.auto_apply_module.insert(<c_MODULE> (<int>v.value))

# Returns the current timestamp at the time the function is called.
# \ingroup Core_group
def get_current_timestamp() -> Timestamp:
    ts = Timestamp()
    ts.timestamp = getCurrentTimeStamp()
    return ts


##
# Structure containing the width and height of an image.
# \ingroup Core_group
cdef class Resolution:
    cdef c_Resolution resolution
    def __cinit__(self, width=0, height=0):
        self.resolution.width = width
        self.resolution.height = height

    ##
    # Area (width * height) of the image.
    def area(self) -> int:
        return self.resolution.width * self.resolution.height

    ##
    # Width of the image in pixels.
    @property
    def width(self) -> int:
        return self.resolution.width

    @width.setter
    def width(self, value):
        self.resolution.width = value

    ##
    # Height of the image in pixels.
    @property
    def height(self) -> int:
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
# Class defining a 2D rectangle with top-left corner coordinates and width/height in pixels.
# \ingroup Core_group
cdef class Rect:
    cdef c_Rect rect
    def __cinit__(self, x=0, y=0, width=0, height=0):
        self.rect.x = x
        self.rect.y = y
        self.rect.width = width
        self.rect.height = height

    ##
    # Width of the rectangle in pixels.
    @property
    def width(self) -> int:
        return self.rect.width

    @width.setter
    def width(self, value):
        self.rect.width = value

    ##
    # Height of the rectangle in pixels.
    @property
    def height(self) -> int:
        return self.rect.height

    @height.setter
    def height(self, value):
        self.rect.height = value

    ##
    # x coordinates of top-left corner.
    @property
    def x(self) -> int:
        return self.rect.x

    @x.setter
    def x(self, value):
        self.rect.x = value

    ##
    # y coordinates of top-left corner.
    @property
    def y(self) -> int:
        return self.rect.y

    @y.setter
    def y(self, value):
        self.rect.y = value

    ##
    # Returns the area of the rectangle.
    def area(self) -> int:
        return self.rect.width * self.rect.height

    ##
    # Tests if the given sl.Rect is empty (width or/and height is null).
    def is_empty(self) -> bool:
        return (self.rect.width * self.rect.height == 0)

    ##
    # Tests if this sl.Rect contains the <b>target</b> sl.Rect.
    # \return True if this rectangle contains the <target> rectangle, otherwise False.
    # \note This method only returns true if the target rectangle is entirely inside this rectangle (not on the edge).
    def contains(self, target: Rect, proper = False) -> bool:
        return self.rect.contains(target.rect, proper)

    ##
    # \brief Tests if this sl.Rect is contained inside the given <b>target</b> sl.Rect.
    # \return True if this rectangle is inside the current <b>target</b> sl.Rect, otherwise False.
    # \note This method only returns True if this rectangle is entirely inside the <target> rectangle (not on the edge).
    def is_contained(self, target: Rect, proper = False) -> bool:
        return self.rect.isContained((<c_Rect>target.rect), proper)

    def __richcmp__(Rect left, Rect right, int op):
        if op == 2:
            return left.width==right.width and left.height==right.height and left.x==right.x and left.y==right.y
        if op == 3:
            return left.width!=right.width or left.height!=right.height or left.x!=right.x or left.y!=right.y
        else:
            raise NotImplementedError()

##
# Class containing the intrinsic parameters of a camera.
# \ingroup Depth_group
# That information about the camera will be returned by sl.Camera.get_camera_information().
# \note Similar to the sl.CalibrationParameters, those parameters are taken from the settings file (SNXXX.conf) and are modified during the sl.Camera.open() call when running a self-calibration).
# \note Those parameters given after sl.Camera.open() call, represent the camera matrix corresponding to rectified or unrectified images.
# \note When filled with rectified parameters, fx, fy, cx, cy must be the same for left and right camera once sl.Camera.open() has been called.
# \note Since distortion is corrected during rectification, distortion should not be considered on rectified images.
cdef class CameraParameters:
    cdef c_CameraParameters camera_params
    ##
    # Focal length in pixels along x axis.
    @property
    def fx(self) -> float:
        return self.camera_params.fx

    @fx.setter
    def fx(self, float fx_):
        self.camera_params.fx = fx_

    ##
    # Focal length in pixels along y axis.
    @property
    def fy(self) -> float:
        return self.camera_params.fy

    @fy.setter
    def fy(self, float fy_):
        self.camera_params.fy = fy_

    ##
    # Optical center along x axis, defined in pixels (usually close to width / 2).
    @property
    def cx(self) -> float:
        return self.camera_params.cx

    @cx.setter
    def cx(self, float cx_):
        self.camera_params.cx = cx_

    ##
    # Optical center along y axis, defined in pixels (usually close to height / 2).
    @property
    def cy(self) -> float:
        return self.camera_params.cy

    @cy.setter
    def cy(self, float cy_):
        self.camera_params.cy = cy_

    ##
    # Distortion factor : [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4].
    #
    # Radial (k1, k2, k3, k4, k5, k6), Tangential (p1,p2) and Prism (s1, s2, s3, s4) distortion.
    @property
    def disto(self) -> list[float]:
        cdef np.ndarray arr = np.zeros(12)
        for i in range(12):
            arr[i] = self.camera_params.disto[i]
        return arr

    ##
    # Sets the elements of the disto array.
    # \param value1 : k1
    # \param value2 : k2
    # \param value3 : p1
    # \param value4 : p2
    # \param value5 : k3
    def set_disto(self, value1: float, value2: float, value3: float, value4: float, value5: float) -> None:
        self.camera_params.disto[0] = value1
        self.camera_params.disto[1] = value2
        self.camera_params.disto[2] = value3
        self.camera_params.disto[3] = value4
        self.camera_params.disto[4] = value5

    ##
    # Vertical field of view, in degrees.
    @property
    def v_fov(self) -> float:
        return self.camera_params.v_fov

    @v_fov.setter
    def v_fov(self, float v_fov_):
        self.camera_params.v_fov = v_fov_

    ##
    # Horizontal field of view, in degrees.
    @property
    def h_fov(self) -> float:
        return self.camera_params.h_fov

    @h_fov.setter
    def h_fov(self, float h_fov_):
        self.camera_params.h_fov = h_fov_

    ##
    # Diagonal field of view, in degrees.
    @property
    def d_fov(self) -> float:
        return self.camera_params.d_fov

    @d_fov.setter
    def d_fov(self, float d_fov_):
        self.camera_params.d_fov = d_fov_

    ##
    # Size in pixels of the images given by the camera.
    @property
    def image_size(self) -> Resolution:
        return Resolution(self.camera_params.image_size.width, self.camera_params.image_size.height)

    @image_size.setter
    def image_size(self, Resolution size_):
        self.camera_params.image_size.width = size_.width
        self.camera_params.image_size.height = size_.height

    ##
    # Real focal length in millimeters.
    @property
    def focal_length_metric(self) -> float:
        return self.camera_params.focal_length_metric

    @focal_length_metric.setter
    def focal_length_metric(self, float focal_length_metric_):
        self.camera_params.focal_length_metric = focal_length_metric_

    ##
    # Setups the parameters of a camera.
    # \param fx_ : Horizontal focal length
    # \param fy_ : Vertical focal length
    # \param cx_ : Horizontal optical center
    # \param cx_ : Vertical optical center.
    def set_up(self, fx_: float, fy_: float, cx_: float, cy_: float) -> None:
        self.camera_params.fx = fx_
        self.camera_params.fy = fy_
        self.camera_params.cx = cx_
        self.camera_params.cy = cy_
    
    ##
    # Return the sl.CameraParameters for another resolution.
    # \param resolution : Resolution in which to get the new sl.CameraParameters.
    # \return The sl.CameraParameters for the resolution given as input.
    def scale(self, resolution: Resolution) -> CameraParameters:
        cam_params = CameraParameters()
        cam_params.camera_params = self.camera_params.scale(resolution.resolution)

##
# Class containing intrinsic and extrinsic parameters of the camera (translation and rotation).
# \ingroup Depth_group
# 
# That information about the camera will be returned by sl.Camera.get_camera_information().
# \note The calibration/rectification process, called during sl.Camera.open(), is using the raw parameters defined in the SNXXX.conf file, where XXX is the serial number of the camera.
# \note Those values may be adjusted or not by the self-calibration to get a proper image alignment.
# \note After sl.Camera.open() is done (with or without self-calibration activated), most of the stereo parameters (except baseline of course) should be 0 or very close to 0.
# \note It means that images after rectification process (given by sl.Camera.retrieve_image()) are aligned as if they were taken by a "perfect" stereo camera, defined by the new sl.CalibrationParameters.
# \warning CalibrationParameters are returned in \ref COORDINATE_SYSTEM "sl.COORDINATE_SYSTEM.IMAGE", they are not impacted by the \ref InitParameters "sl.InitParameters.coordinate_system".
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
    # Returns the baseline of the camera in the sl.UNIT defined in sl.InitParameters.coordinate_units.
    def get_camera_baseline(self) -> float:
        return self.calibration.getCameraBaseline()

    ##
    # Intrinsic sl.CameraParameters of the left camera.
    @property
    def left_cam(self) -> CameraParameters:
        return self.py_left_cam

    @left_cam.setter
    def left_cam(self, CameraParameters left_cam_) :
        self.calibration.left_cam = left_cam_.camera_params
        self.set()
    
    ##
    # Intrinsic sl.CameraParameters of the right camera.
    @property
    def right_cam(self) -> CameraParameters:
        return self.py_right_cam

    @right_cam.setter
    def right_cam(self, CameraParameters right_cam_) :
        self.calibration.right_cam = right_cam_.camera_params
        self.set()

    ##
    # Left to right camera transform, expressed in user coordinate system and unit (defined by \ref InitParameters "sl.InitParameters.coordinate_system").
    @property
    def stereo_transform(self) -> Transform:
        return self.py_stereo_transform

##
# Class containing information about a single sensor available in the current device.
# \ingroup Sensors_group
# 
# Information about the camera sensors is available in the sl.CameraInformation struct returned by sl.Camera.get_camera_information().
# \note This class is meant to be used as a read-only container.
# \note Editing any of its fields will not impact the ZED SDK.
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
    # Type of the sensor.
    @property
    def sensor_type(self) -> SENSOR_TYPE:
        return SENSOR_TYPE(<int>self.sensor_type)

    ##
    # Resolution of the sensor.
    @property
    def resolution(self) -> float:
        return self.c_sensor_parameters.resolution

    @resolution.setter
    def resolution(self, float resolution_):
        self.c_sensor_parameters.resolution = resolution_

    ##
    # Sampling rate (or ODR) of the sensor.
    @property
    def sampling_rate(self) -> float:
        return self.c_sensor_parameters.sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, float sampling_rate_):
        self.c_sensor_parameters.sampling_rate = sampling_rate_

    ##
    # Range (NumPy array) of the sensor (minimum: `sensor_range[0]`, maximum: `sensor_range[1]`). 
    @property
    def sensor_range(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = self.c_sensor_parameters.range[i]
        return arr

    ##
    # Sets the minimum and the maximum values of the sensor range.
    # \param float value1 : Minimum of the range to set.
    # \param float value2 : Maximum of the range to set.
    def set_sensor_range(self, value1: float, value2: float) -> None:
        self.c_sensor_parameters.range[0] = value1
        self.c_sensor_parameters.range[1] = value2
        self.set()

    ##
    # White noise density given as continuous (frequency-independent).
    # \note The units will be expressed in ```sensor_unit / √(Hz)```.
    # \note `NAN` if the information is not available.
    @property
    def noise_density(self) -> float:
        return self.c_sensor_parameters.noise_density

    @noise_density.setter
    def noise_density(self, float noise_density_):
        self.c_sensor_parameters.noise_density = noise_density_

    ##
    # Random walk derived from the Allan Variance given as continuous (frequency-independent).
    # \note The units will be expressed in ```sensor_unit / √(Hz)```.
    # \note `NAN` if the information is not available.
    @property
    def random_walk(self) -> float:
        return self.c_sensor_parameters.random_walk

    @random_walk.setter
    def random_walk(self, float random_walk_):
        self.c_sensor_parameters.random_walk = random_walk_
    
    ##
    # Unit of the sensor.
    @property
    def sensor_unit(self) -> SENSORS_UNIT:
        return SENSORS_UNIT(<int>self.sensor_unit)

    ##
    # Whether the sensor is available in your camera.
    @property
    def is_available(self) -> bool:
        return self.c_sensor_parameters.isAvailable

##
# Class containing information about all the sensors available in the current device.
# \ingroup Sensors_group
# 
# Information about the camera sensors is available in the sl.CameraInformation struct returned by sl.Camera.get_camera_information().
# \note This class is meant to be used as a read-only container.
# \note Editing any of its fields will not impact the ZED SDK.
cdef class SensorsConfiguration:
    cdef unsigned int firmware_version
    cdef Transform camera_imu_transform
    cdef Transform imu_magnetometer_transform
    cdef SensorParameters accelerometer_parameters
    cdef SensorParameters gyroscope_parameters
    cdef SensorParameters magnetometer_parameters
    cdef SensorParameters barometer_parameters

    def __cinit__(self, py_camera, Resolution resizer=Resolution(0,0)):
        if isinstance(py_camera, Camera):
            self.__set_from_camera(py_camera, resizer)
        else:
            IF UNAME_SYSNAME == u"Linux":
                if isinstance(py_camera, CameraOne):
                    self.__set_from_cameraone(py_camera, resizer)
                raise TypeError("Argument is not of Camera or CameraOne type.")
            ELSE:
                raise TypeError("Argument is not of Camera type.")

    def __set_from_camera(self, Camera py_camera, Resolution resizer=Resolution(0,0)):
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

    IF UNAME_SYSNAME == u"Linux":
        def __set_from_cameraone(self, CameraOne py_camera, Resolution resizer=Resolution(0,0)):
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
    # Configuration of the accelerometer.
    @property
    def accelerometer_parameters(self) -> SensorParameters:
        return self.accelerometer_parameters
    
    ##
    # Configuration of the gyroscope.
    @property
    def gyroscope_parameters(self) -> SensorParameters:
        return self.gyroscope_parameters

    ##
    # Configuration of the magnetometer.    
    @property
    def magnetometer_parameters(self) -> SensorParameters:
        return self.magnetometer_parameters

    ##
    # Configuration of the barometer.
    @property
    def barometer_parameters(self) -> SensorParameters:
        return self.barometer_parameters
    
    ##
    # IMU to left camera transform matrix.
    # \note It contains the rotation and translation between the IMU frame and camera frame.
    @property
    def camera_imu_transform(self) -> Transform:
        return self.camera_imu_transform
    
    ##
    # Magnetometer to IMU transform matrix.
    # \note It contains rotation and translation between IMU frame and magnetometer frame.
    @property
    def imu_magnetometer_transform(self) -> Transform:
        return self.imu_magnetometer_transform

    ##
    # Firmware version of the sensor module.
    # \note 0 if no sensors are available ([sl.MODEL.ZED](\ref MODEL)).
    @property
    def firmware_version(self) -> int:
        return self.firmware_version

    ##
    # Checks if a sensor is available on the device.
    # \param sensor_type : Sensor type to check.
    # \return True if the sensor is available on the device, otherwise False.
    def is_sensor_available(self, sensor_type) -> bool:
        if isinstance(sensor_type, SENSOR_TYPE):
            if sensor_type == SENSOR_TYPE.ACCELEROMETER:
                return self.accelerometer_parameters.is_available
            elif sensor_type == SENSOR_TYPE.GYROSCOPE:
                return self.gyroscope_parameters.is_available
            elif sensor_type == SENSOR_TYPE.MAGNETOMETER:
                return self.magnetometer_parameters.is_available
            elif sensor_type == SENSOR_TYPE.BAROMETER:
                return self.barometer_parameters.is_available
            else:
                return False
        else:
           raise TypeError("Argument is not of SENSOR_TYPE type.") 

##
# Structure containing information about the camera sensor. 
# \ingroup Core_group
# 
# Information about the camera is available in the sl.CameraInformation struct returned by sl.Camera.get_camera_information().
# \note This object is meant to be used as a read-only container, editing any of its field won't impact the SDK.
# \warning sl.CalibrationParameters are returned in sl.COORDINATE_SYSTEM.IMAGE, they are not impacted by the sl.InitParameters.coordinate_system.
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
    # Resolution of the camera.
    @property
    def resolution(self) -> Resolution:
        return Resolution(self.py_res.width, self.py_res.height)

    ##
    # FPS of the camera.
    @property
    def fps(self) -> float:
        return self.camera_fps

    ##
    # Intrinsics and extrinsic stereo parameters for rectified/undistorted images.
    @property
    def calibration_parameters(self) -> CalibrationParameters:
        return self.py_calib

    ##
    # Intrinsics and extrinsic stereo parameters for unrectified/distorted images.
    @property
    def calibration_parameters_raw(self) -> CalibrationParameters:
        return self.py_calib_raw

    ##
    # Internal firmware version of the camera.
    @property
    def firmware_version(self) -> int:
        return self.firmware_version



##
# Structure containing information of a single camera (serial number, model, calibration, etc.)
# \ingroup Core_group
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
    # Default constructor.
    # Gets the sl.CameraParameters from a sl.Camera object.
    # \param py_camera : sl.Camera object.
    # \param resizer : You can specify a sl.Resolution different from default image size to get the scaled camera information. Default: (0, 0) (original image size)
    #
    # \code
    # cam = sl.Camera()
    # res = sl.Resolution(0,0)
    # cam_info = sl.CameraInformation(cam, res)
    # \endcode
    def __cinit__(self, py_camera: Camera, resizer=Resolution(0,0)) -> CameraInformation:
        res = c_Resolution(resizer.width, resizer.height)
        caminfo = py_camera.camera.getCameraInformation(res)

        self.serial_number = caminfo.serial_number
        self.camera_model = caminfo.camera_model
        self.py_camera_configuration = CameraConfiguration(py_camera, resizer)
        self.py_sensors_configuration = SensorsConfiguration(py_camera, resizer)
        self.input_type = caminfo.input_type

    ##
    # Sensors configuration parameters stored in a sl.SensorsConfiguration.
    @property
    def sensors_configuration(self) -> SensorsConfiguration:
        return self.py_sensors_configuration

    ##
    # Camera configuration parameters stored in a sl.CameraConfiguration.
    @property
    def camera_configuration(self) -> CameraConfiguration:
        return self.py_camera_configuration

    ##
    # Input type used in the ZED SDK.
    @property
    def input_type(self) -> INPUT_TYPE:
        return INPUT_TYPE(<int>self.input_type)

    ##
    # Model of the camera (see sl.MODEL).
    @property
    def camera_model(self) -> MODEL:
        return MODEL(<int>self.camera_model)

    ##
    # Serial number of the camera.
    @property
    def serial_number(self) -> int:
        return self.serial_number


##
# Class representing 1 to 4-channel matrix of float or uchar, stored on CPU and/or GPU side.
# \ingroup Core_group
#
# This class is defined in a row-major order, meaning that for an image buffer, the rows are stored consecutively from top to bottom.
# \note The ZED SDK Python wrapper does not support GPU data storage/access.
cdef class Mat:
    cdef c_Mat mat
    ##
    # Default constructor.
    # \param width : Width of the matrix in pixels. Default: 0
    # \param height : Height of the matrix in pixels. Default: 0
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \code
    # mat = sl.Mat(width=0, height=0, mat_type=sl.MAT_TYPE.F32_C1, memory_type=sl.MEM.CPU)
    # \endcode
    def __cinit__(self, width=0, height=0, mat_type=MAT_TYPE.F32_C1, memory_type=MEM.CPU) -> Mat:
        c_Mat(width, height, <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value)).move(self.mat)

    ##
    # Initilizes a new sl.Mat and allocates the requested memory by calling \ref alloc_size().
    # \param width : Width of the matrix in pixels. Default: 0
    # \param height : Height of the matrix in pixels. Default: 0
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_type(self, width, height, mat_type, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Initilizes a new sl.Mat from an existing data pointer.
    # This method does not allocate the memory.
    # \param width : Width of the matrix in pixels.
    # \param height : Height of the matrix in pixels.
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param ptr : Pointer to the data array.
    # \param step : Step of the data array (bytes size of one pixel row).
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_cpu(self, width: int, height: int, mat_type: MAT_TYPE, ptr, step, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(width, height, <c_MAT_TYPE>(<int>mat_type.value), ptr.encode(), step, <c_MEM>(<int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Initilizes a new sl.Mat and allocates the requested memory by calling \ref alloc_size().
    # \param resolution : Size of the matrix in pixels.
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_resolution(self, resolution: Resolution, mat_type: MAT_TYPE, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Initilizes a new sl.Mat from an existing data pointer.
    # This method does not allocate the memory.
    # \param resolution : the size of the matrix in pixels.
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param ptr : Pointer to the data array (CPU or GPU).
    # \param step : Step of the data array (bytes size of one pixel row).
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    def init_mat_resolution_cpu(self, resolution: Resolution, mat_type, ptr, step, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            c_Mat(c_Resolution(resolution.width, resolution.height), <c_MAT_TYPE>(<int>mat_type.value), ptr.encode(), step, <c_MEM>(<int>memory_type.value)).move(self.mat)
        else:
            raise TypeError("Argument are not of MAT_TYPE or MEM type.")

    ##
    # Initilizes a new sl.Mat by copy (shallow copy).
    # This method does not allocate the memory.
    # \param mat : sl.Mat to copy.
    def init_mat(self, matrix: Mat) -> None:
        c_Mat(matrix.mat).move(self.mat)

    ##
    # Allocates the sl.Mat memory.
    # \param width : Width of the matrix in pixels.
    # \param height : Height of the matrix in pixels.
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \warning It erases previously allocated memory.
    def alloc_size(self, width, height, mat_type, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(<size_t> width, <size_t> height, <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value))
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    ##
    # Allocates the sl.Mat memory.
    # \param resolution : Size of the matrix in pixels.
    # \param mat_type : Type of the matrix ([sl.MAT_TYPE.F32_C1](\ref MAT_TYPE), [sl.MAT_TYPE.U8_C4](\ref MAT_TYPE), etc.).\n Default: [sl.MAT_TYPE.F32_C1](\ref MAT_TYPE)
    # \param memory_type : Where the buffer will be stored. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    #
    # \warning It erases previously allocated memory.
    def alloc_resolution(self, resolution: Resolution, mat_type: MAT_TYPE, memory_type=MEM.CPU) -> None:
        if isinstance(mat_type, MAT_TYPE) and isinstance(memory_type, MEM):
            self.mat.alloc(resolution.resolution, <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value))
            #self.mat.alloc(resolution.width, resolution.height, <c_MAT_TYPE>(<int>mat_type.value), <c_MEM>(<int>memory_type.value))
        else:
            raise TypeError("Arguments must be of Mat and MEM types.")

    ##
    # Free the owned memory.
    # \param memory_type : Specifies which memory you wish to free. Default: [sl.MEM.CPU](\ref MEM) (you cannot change this default value)
    def free(self, memory_type=MEM.CPU) -> None:
        if isinstance(memory_type, MEM):
            self.mat.free(<c_MEM>(<int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Copies data to another sl.Mat (deep copy).
    #
    # \param dst : sl.Mat where the data will be copied to.
    # \param cpy_type : Specifies the memory that will be used for the copy. Default: [sl.COPY_TYPE.CPU_CPU](\ref COPY_TYPE) (you cannot change this default value)
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \note If the destination is not allocated or does not have a compatible sl.MAT_TYPE or sl.Resolution,
    # current memory is freed and new memory is directly allocated.
    def copy_to(self, dst: Mat, cpy_type=COPY_TYPE.CPU_CPU) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.copyTo(dst.mat, <c_COPY_TYPE>(<int>cpy_type.value)), ERROR_CODE.FAILURE)

    ##
    # Downloads data from DEVICE (GPU) to HOST (CPU), if possible.
    #  \note If no CPU or GPU memory are available for this sl::Mat, some are directly allocated.
    #  \note If verbose is set to true, you have information in case of failure.
    def update_cpu_from_gpu(self) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.updateCPUfromGPU(), ERROR_CODE.FAILURE)

    ##
    # Uploads data from HOST (CPU) to DEVICE (GPU), if possible.
    # \note If no CPU or GPU memory are available for this sl::Mat, some are directly allocated.
    # \note If verbose is set to true, you have information in case of failure.
    def update_gpu_from_cpu(self) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.updateGPUfromCPU(), ERROR_CODE.FAILURE)

    ##
    # Copies data from an other sl.Mat (deep copy).
    # \param src : sl.Mat where the data will be copied from.
    # \param cpy_type : Specifies the memory that will be used for the copy. Default: [sl.COPY_TYPE.CPU_CPU](\ref COPY_TYPE) (you cannot change this default value)
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \note If the destination is not allocated or does not have a compatible sl.MAT_TYPE or sl.Resolution,
    # current memory is freed and new memory is directly allocated.
    def set_from(self, src: Mat, cpy_type=COPY_TYPE.CPU_CPU) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.setFrom(<const c_Mat>src.mat, <c_COPY_TYPE>(<int>cpy_type.value)), ERROR_CODE.FAILURE)

    ##
    # Reads an image from a file (only if [sl.MEM.CPU](\ref MEM) is available on the current sl.Mat).
    # Supported input files format are PNG and JPEG.
    # \param filepath : Path of the file to read from (including the name and extension).
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \note Supported sl.MAT_TYPE are :
    # - [MAT_TYPE.F32_C1](\ref MAT_TYPE) for PNG/PFM/PGM
    # - [MAT_TYPE.F32_C3](\ref MAT_TYPE) for PCD/PLY/VTK/XYZ
    # - [MAT_TYPE.F32_C4](\ref MAT_TYPE) for PCD/PLY/VTK/WYZ
    # - [MAT_TYPE.U8_C1](\ref MAT_TYPE) for PNG/JPG
    # - [MAT_TYPE.U8_C3](\ref MAT_TYPE) for PNG/JPG
    # - [MAT_TYPE.U8_C4](\ref MAT_TYPE) for PNG/JPG
    def read(self, filepath: str) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.read(filepath.encode()), ERROR_CODE.FAILURE)

    ##
    # Writes the sl.Mat (only if [sl.MEM.CPU](\ref MEM) is available on the current sl.Mat) into a file as an image.
    # Supported output files format are PNG and JPEG.
    # \param filepath : Path of the file to write (including the name and extension).
    # \param memory_type : Memory type of the sl.Mat. Default: [sl.MEM.CPU](\ref MEM) (you cannot change the default value)
    # \param compression_level : Level of compression between 0 (lowest compression == highest size == highest quality(jpg)) and 100 (highest compression == lowest size == lowest quality(jpg)).
    # \note Specific/default value for compression_level = -1 : This will set the default quality for PNG(30) or JPEG(5).
    # \note compression_level is only supported for [U8_Cx] (\ref MAT_TYPE).
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \note Supported sl.MAT_TYPE are :
    # - [MAT_TYPE.F32_C1](\ref MAT_TYPE) for PNG/PFM/PGM
    # - [MAT_TYPE.F32_C3](\ref MAT_TYPE) for PCD/PLY/VTK/XYZ
    # - [MAT_TYPE.F32_C4](\ref MAT_TYPE) for PCD/PLY/VTK/WYZ
    # - [MAT_TYPE.U8_C1](\ref MAT_TYPE) for PNG/JPG
    # - [MAT_TYPE.U8_C3](\ref MAT_TYPE) for PNG/JPG
    # - [MAT_TYPE.U8_C4](\ref MAT_TYPE) for PNG/JPG
    def write(self, filepath: str, memory_type=MEM.CPU, compression_level = -1) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.write(filepath.encode(), <c_MEM>(<int>memory_type.value), compression_level), ERROR_CODE.FAILURE)

    ##
    # Fills the sl.Mat with the given value.
    # This method overwrites all the matrix.
    # \param value : Value to be copied all over the matrix.
    # \param memory_type : Which buffer to fill. Default: [sl.MEM.CPU](\ref MEM) (you cannot change the default value)
    def set_to(self, value, memory_type=MEM.CPU) -> ERROR_CODE:
        if self.get_data_type() == MAT_TYPE.U8_C1:
            return _error_code_cache.get(<int>setToUchar1(self.mat, value, <c_MEM>(<int>memory_type.value)), ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            return _error_code_cache.get(<int>setToUchar2(self.mat, Vector2[uchar1](value[0], value[1]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            return _error_code_cache.get(<int>setToUchar3(self.mat, Vector3[uchar1](value[0], value[1], value[2]), <c_MEM>(<int>memory_type.value)),
                                        ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            return _error_code_cache.get(<int>setToUchar4(self.mat, Vector4[uchar1](value[0], value[1], value[2], value[3]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            return _error_code_cache.get(<int>setToUshort1(self.mat, value, <c_MEM>(<int>memory_type.value)), ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            return _error_code_cache.get(<int>setToFloat1(self.mat, value, <c_MEM>(<int>memory_type.value)), ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            return _error_code_cache.get(<int>setToFloat2(self.mat, Vector2[float1](value[0], value[1]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            return _error_code_cache.get(<int>setToFloat3(self.mat, Vector3[float1](value[0], value[1], value[2]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            return _error_code_cache.get(<int>setToFloat4(self.mat, Vector4[float1](value[0], value[1], value[2], value[3]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)

    ##
    # Sets a value to a specific point in the matrix.
    # \param x : Column of the point to change.
    # \param y : Row of the point to change.
    # \param value : Value to be set.
    # \param memory_type : Which memory will be updated.
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \warning Not efficient for [sl.MEM.GPU](\ref MEM), use it on sparse data.
    def set_value(self, x: int, y: int, value, memory_type=MEM.CPU) -> ERROR_CODE:
        if self.get_data_type() == MAT_TYPE.U8_C1:
            return _error_code_cache.get(<int>setValueUchar1(self.mat, x, y, value, <c_MEM>(<int>memory_type.value)), ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            return _error_code_cache.get(<int>setValueUchar2(self.mat, x, y, Vector2[uchar1](value[0], value[1]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            return _error_code_cache.get(<int>setValueUchar3(self.mat, x, y, Vector3[uchar1](value[0], value[1], value[2]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            return _error_code_cache.get(<int>setValueUshort1(self.mat, x, y, <ushort1>value, <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            return _error_code_cache.get(<int>setValueUchar4(self.mat, x, y, Vector4[uchar1](value[0], value[1], value[2], value[3]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            return _error_code_cache.get(<int>setValueFloat1(self.mat, x, y, value, <c_MEM>(<int>memory_type.value)), ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            return _error_code_cache.get(<int>setValueFloat2(self.mat, x, y, Vector2[float1](value[0], value[1]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            return _error_code_cache.get(<int>setValueFloat3(self.mat, x, y, Vector3[float1](value[0], value[1], value[2]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            return _error_code_cache.get(<int>setValueFloat4(self.mat, x, y, Vector4[float1](value[0], value[1], value[2], value[3]), <c_MEM>(<int>memory_type.value)),
                                         ERROR_CODE.FAILURE)

    ##
    # Returns the value of a specific point in the matrix.
    # \param x : Column of the point to get the value from.
    # \param y : Row of the point to get the value from.
    # \param memory_type : Which memory should be read.
    # \return [sl.ERROR_CODE.SUCCESS](\ref ERROR_CODE) if everything went well, [sl.ERROR_CODE.FAILURE](\ref ERROR_CODE) otherwise.
    #
    # \warning Not efficient for [sl.MEM.GPU](\ref MEM), use it on sparse data.
    def get_value(self, x: int, y: int, memory_type=MEM.CPU) -> ERROR_CODE:
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
            status = getValueUchar1(self.mat, x, y, &value1u, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), value1u
        elif self.get_data_type() == MAT_TYPE.U8_C2:
            status = getValueUchar2(self.mat, x, y, &value2u, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value2u.ptr()[0], value2u.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.U8_C3:
            status = getValueUchar3(self.mat, x, y, &value3u, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value3u.ptr()[0], value3u.ptr()[1], value3u.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.U8_C4:
            status = getValueUchar4(self.mat, x, y, &value4u, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value4u.ptr()[0], value4u.ptr()[1], value4u.ptr()[2],
                                                         value4u.ptr()[3]])
        elif self.get_data_type() == MAT_TYPE.U16_C1:
            status = getValueUshort1(self.mat, x, y, &value1us, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), value1us
        elif self.get_data_type() == MAT_TYPE.F32_C1:
            status = getValueFloat1(self.mat, x, y, &value1f, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), value1f
        elif self.get_data_type() == MAT_TYPE.F32_C2:
            status = getValueFloat2(self.mat, x, y, &value2f, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value2f.ptr()[0], value2f.ptr()[1]])
        elif self.get_data_type() == MAT_TYPE.F32_C3:
            status = getValueFloat3(self.mat, x, y, &value3f, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value3f.ptr()[0], value3f.ptr()[1], value3f.ptr()[2]])
        elif self.get_data_type() == MAT_TYPE.F32_C4:
            status = getValueFloat4(self.mat, x, y, &value4f, <c_MEM>(<int>memory_type.value))
            return _error_code_cache.get(<int>status, ERROR_CODE.FAILURE), np.array([value4f.ptr()[0], value4f.ptr()[1], value4f.ptr()[2],
                                                         value4f.ptr()[3]])

    ##
    # Returns the width of the matrix.
    # \return Width of the matrix in pixels.
    def get_width(self) -> int:
        return self.mat.getWidth()

    ##
    # Returns the height of the matrix.
    # \return Height of the matrix in pixels.
    def get_height(self) -> int:
        return self.mat.getHeight()

    ##
    # Returns the resolution (width and height) of the matrix.
    # \return Resolution of the matrix in pixels.
    def get_resolution(self) -> Resolution:
        return Resolution(self.mat.getResolution().width, self.mat.getResolution().height)

    ##
    # Returns the number of values stored in one pixel.
    # \return Number of values in a pixel.
    def get_channels(self) -> int:
        return self.mat.getChannels()

    ##
    # Returns the format of the matrix.
    # \return Format of the current sl.Mat.
    def get_data_type(self) -> MAT_TYPE:
        return MAT_TYPE(<int>self.mat.getDataType())

    ##
    # Returns the type of memory (CPU and/or GPU).
    # \return Type of allocated memory.
    def get_memory_type(self) -> MEM:
        return MEM(<int>self.mat.getMemoryType())

    ##
    # Returns the sl.Mat as a NumPy array.
    # This is for convenience to mimic the [PyTorch API](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
    # \n This is like an alias of \ref get_data() method.
    # \param force : Whether the memory of the sl.Mat need to be duplicated.
    # \return NumPy array containing the sl.Mat data.
    # \note The fastest is \b force at False but the sl.Mat memory must not be released to use the NumPy array.
    def numpy(self, force=False) -> np.array:
        return self.get_data(memory_type=MEM.CPU, deep_copy=force)

    ##
    # Cast the data of the sl.Mat in a NumPy array (with or without copy).
    # \param memory_type : Which memory should be read. Default: [MEM.CPU](\ref MEM) (you cannot change the default value)
    # \param deep_copy : Whether the memory of the sl.Mat need to be duplicated.
    # \return NumPy array containing the sl.Mat data.
    # \note The fastest is \b deep_copy at False but the sl.Mat memory must not be released to use the NumPy array.
    def get_data(self, memory_type=MEM.CPU, deep_copy=False) -> np.array:

        if not isinstance(memory_type, MEM):
            raise TypeError("Argument is not of MEM type.")

        if memory_type.value == MEM.BOTH.value:
            raise ValueError("MEM.BOTH is not supported for get_data() method.")

        if self.get_memory_type().value != memory_type.value and self.get_memory_type().value != MEM.BOTH.value:
            raise ValueError("Provided MEM type doesn't match Mat's memory_type.")

        cdef np.npy_intp cython_shape[3]
        cython_shape[0] = <np.npy_intp> self.mat.getHeight()
        cython_shape[1] = <np.npy_intp> self.mat.getWidth()
        cython_shape[2] = <np.npy_intp> self.mat.getChannels()

        cdef size_t size = 0
        dtype = None
        nptype = None
        npdim = None
        itemsize = None
        if self.mat.getDataType() in (c_MAT_TYPE.U8_C1, c_MAT_TYPE.U8_C2, c_MAT_TYPE.U8_C3, c_MAT_TYPE.U8_C4):
            itemsize = 1
            dtype = np.uint8
            nptype = np.NPY_UINT8
        elif self.mat.getDataType() in (c_MAT_TYPE.F32_C1, c_MAT_TYPE.F32_C2, c_MAT_TYPE.F32_C3, c_MAT_TYPE.F32_C4):
            itemsize = sizeof(float)
            dtype = np.float32
            nptype = np.NPY_FLOAT32
        elif self.mat.getDataType() == c_MAT_TYPE.U16_C1:
            itemsize = sizeof(ushort)
            dtype = np.ushort
            nptype = np.NPY_UINT16
        else:
            raise RuntimeError("Unknown Mat data_type value: {0}".format(<int>self.mat.getDataType()))

        shape = None
        if self.mat.getChannels() == 1:
            shape = (self.mat.getHeight(), self.mat.getWidth())
            strides = (self.get_step_bytes(memory_type), self.get_pixel_bytes())
        else:
            shape = (self.mat.getHeight(), self.mat.getWidth(), self.mat.getChannels())
            strides = (self.get_step_bytes(memory_type), self.get_pixel_bytes(), itemsize)

        size = self.mat.getHeight()*self.get_step(memory_type)*self.mat.getChannels()*itemsize

        if self.mat.getDataType() in (c_MAT_TYPE.U8_C1, c_MAT_TYPE.F32_C1, c_MAT_TYPE.U16_C1):
            npdim = 2
        else:
            npdim = 3

        cdef np.ndarray nparr  # Placeholder for the np.ndarray since memcpy on CPU only works on cdef types
        arr = None  # Could be either an `np.ndarray` or a `cp.ndarray` 

        if memory_type.value == MEM.CPU.value and deep_copy:
            nparr = np.empty(shape, dtype=dtype)
            if self.mat.getDataType() == c_MAT_TYPE.U8_C1:
                memcpy(<void*>nparr.data, <void*>getPointerUchar1(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.U8_C2:
                memcpy(<void*>nparr.data, <void*>getPointerUchar2(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.U8_C3:
                memcpy(<void*>nparr.data, <void*>getPointerUchar3(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.U8_C4:
                memcpy(<void*>nparr.data, <void*>getPointerUchar4(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.U16_C1:
                memcpy(<void*>nparr.data, <void*>getPointerUshort1(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.F32_C1:
                memcpy(<void*>nparr.data, <void*>getPointerFloat1(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.F32_C2:
                memcpy(<void*>nparr.data, <void*>getPointerFloat2(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.F32_C3:
                memcpy(<void*>nparr.data, <void*>getPointerFloat3(self.mat, <c_MEM>(<int>memory_type.value)), size)
            elif self.mat.getDataType() == c_MAT_TYPE.F32_C4:
                memcpy(<void*>nparr.data, <void*>getPointerFloat4(self.mat, <c_MEM>(<int>memory_type.value)), size)

            # This is the workaround since cdef statements couldn't be performed in sub-scopes
            arr = nparr

        elif memory_type.value == MEM.CPU.value and not deep_copy:
            # Thanks to BDO for the initial implementation!
            arr = np.PyArray_SimpleNewFromData(npdim, cython_shape, nptype, <void*>getPointerUchar1(self.mat, <c_MEM>(<int>memory_type.value)))

        ##
        # Thanks to @Rad-hi for the implementation! https://github.com/stereolabs/zed-python-api/pull/241
        # Ref: https://stackoverflow.com/questions/71344734/cupy-array-construction-from-existing-gpu-pointer
        # Ref: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.runtime.memcpy.html
        # Ref: https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_Toolkit_Reference_Manual.pdf
        elif CUPY_AVAILABLE and memory_type.value == MEM.GPU.value and deep_copy:
            in_mem_shape = (self.mat.getHeight(), self.get_step(memory_type), self.mat.getChannels())
            dst_arr = cp.empty(in_mem_shape, dtype=dtype)
            dst_ptr = dst_arr.data.ptr
            src_ptr = self.get_pointer(memory_type=memory_type)
            TRANSFER_KIND_GPU_GPU = 3
            cp.cuda.runtime.memcpy(dst_ptr, src_ptr, size, TRANSFER_KIND_GPU_GPU)
            arr = cp.ndarray(shape, dtype=dtype, memptr=dst_arr.data, strides=strides)

        ##
        # Ref: https://github.com/cupy/cupy/issues/4644
        # Ref: https://docs.cupy.dev/en/stable/user_guide/interoperability.html#device-memory-pointers
        elif CUPY_AVAILABLE and memory_type.value == MEM.GPU.value and not deep_copy:
            src_ptr = self.get_pointer(memory_type=memory_type)
            mem = cp.cuda.UnownedMemory(src_ptr, size, self)
            memptr = cp.cuda.MemoryPointer(mem, offset=0)
            arr = cp.ndarray(shape, dtype=dtype, memptr=memptr, strides=strides)

        return arr

    ##
    # Returns the memory step in bytes (size of one pixel row).
    # \param memory_type : Specifies whether you want [sl.MEM.CPU](\ref MEM) or [sl.MEM.GPU](\ref MEM) step.\n Default: [sl.MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return The step in bytes of the specified memory.
    def get_step_bytes(self, memory_type=MEM.CPU) -> int:
        if type(memory_type) == MEM:
            return self.mat.getStepBytes(<c_MEM>(<int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Returns the memory step in number of elements (size in one pixel row).
    # \param memory_type : Specifies whether you want [sl.MEM.CPU](\ref MEM) or [sl.MEM.GPU](\ref MEM) step.\n Default: [sl.MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return The step in number of elements.
    def get_step(self, memory_type=MEM.CPU) -> int:
        if type(memory_type) == MEM:
            return self.mat.getStep(<c_MEM>(<int>memory_type.value))
        else:
            raise TypeError("Argument is not of MEM type.")

    ##
    # Returns the size of one pixel in bytes.
    # \return Size of a pixel in bytes.
    def get_pixel_bytes(self) -> int:
        return self.mat.getPixelBytes()

    ##
    # Returns the size of a row in bytes.
    # \return Size of a row in bytes.
    def get_width_bytes(self) -> int:
        return self.mat.getWidthBytes()

    ##
    # Returns the information about the sl.Mat into a string.
    # \return String containing the sl.Mat information.
    def get_infos(self) -> str:
        return to_str(self.mat.getInfos()).decode()

    ##
    # Returns whether the sl.Mat is initialized or not.
    # \return True if current sl.Mat has been allocated (by the constructor or therefore).
    def is_init(self) -> bool:
        return self.mat.isInit()

    ##
    # Returns whether the sl.Mat is the owner of the memory it accesses.
    #
    # If not, the memory won't be freed if the sl.Mat is destroyed.
    # \return True if the sl.Mat is owning its memory, else False.
    def is_memory_owner(self) -> bool:
        return self.mat.isMemoryOwner()

    ##
    # Duplicates a sl.Mat by copy (deep copy).
    # \param py_mat : sl.Mat to copy.
    #
    # This method copies the data array(s) and it marks the new sl.Mat as the memory owner.
    def clone(self, py_mat: Mat) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.clone(py_mat.mat), ERROR_CODE.FAILURE)

    ##
    # Moves the data of the sl.Mat to another sl.Mat.
    #
    # This method gives the attribute of the current s.Mat to the specified one. (No copy.)
    # \param py_mat : sl.Mat to move to.
    # \note : The current sl.Mat is then no more usable since its loose its attributes.
    def move(self, py_mat: Mat) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.move(py_mat.mat), ERROR_CODE.FAILURE)

    ##
    #  Convert the color channels of the Mat (RGB<->BGR or RGBA<->BGRA)
    # This methods works only on 8U_C4 or 8U_C3
    def convert_color_inplace(self, memory_type=MEM.CPU) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.mat.convertColor(<c_MEM>(<unsigned int>memory_type.value)), ERROR_CODE.FAILURE)

    ##
    # Convert the color channels of the Mat into another Mat
    # This methods works only on 8U_C4 if remove_alpha_channels is enabled, or 8U_C4 and 8U_C3 if swap_RB_channels is enabled
    # The inplace method sl::Mat::convertColor can be used for only swapping the Red and Blue channel efficiently
    @staticmethod
    def convert_color(mat1: Mat, mat2: Mat, swap_RB_channels: bool, remove_alpha_channels: bool, memory_type=MEM.CPU) -> ERROR_CODE:
        cls = Mat()
        return _error_code_cache.get(<int>cls.mat.convertColor(mat1.mat, mat2.mat, swap_RB_channels, remove_alpha_channels, <c_MEM>(<unsigned int>memory_type.value)), ERROR_CODE.FAILURE)
        
    ##
    # Swaps the content of the provided sl::Mat (only swaps the pointers, no data copy).
    # \param mat1 : First matrix to swap.
    # \param mat2 : Second matrix to swap.
    @staticmethod
    def swap(mat1: Mat, mat2: Mat) -> None:
        cdef c_Mat tmp
        tmp = mat1.mat
        mat1.mat = mat2.mat
        mat2.mat = tmp

    ##
    # Gets the pointer of the content of the sl.Mat.
    # \param memory_type : Which memory you want to get. Default: [sl.MEM.CPU](\ref MEM) (you cannot change the default value)
    # \return Pointer of the content of the sl.Mat.
    def get_pointer(self, memory_type=MEM.CPU) -> int:
        ptr = <unsigned long long>getPointerUchar1(self.mat, <c_MEM>(<int>memory_type.value))
        return ptr

    ##
    # The name of the sl.Mat (optional).
    # In \ref verbose mode, it iss used to indicate which sl.Mat is printing information.
    # \n Default set to "n/a" to avoid empty string if not filled.
    @property
    def name(self) -> str:
        if not self.mat.name.empty():
            return self.mat.name.get().decode()
        else:
            return ""

    @name.setter
    def name(self, str name_):
        self.mat.name.set(name_.encode())

    ##
    # Timestamp of the last manipulation of the data of the matrix by a method/function.
    @property
    def timestamp(self) -> int:
        ts = Timestamp()
        ts.timestamp = self.mat.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, timestamp : Timestamp):
        self.mat.timestamp.data_ns = timestamp.get_nanoseconds()

    ##
    # Whether the sl.Mat can display information.
    @property
    def verbose(self) -> bool:
        return self.mat.verbose

    @verbose.setter
    def verbose(self, bool verbose_):
        self.mat.verbose = verbose_

    def __repr__(self):
        return self.get_infos()

##
# \brief Convert an image into a GPU Tensor in planar channel configuration (NCHW), ready to use for deep learning model
# \param image_in : input image to convert
# \param tensor_out : output GPU tensor
# \param resolution_out : resolution of the output image, generally square, although not mandatory
# \param scalefactor : Scale factor applied to each pixel value, typically to convert the char value into [0-1] float
# \param mean : mean, statistic to normalized the pixel values, applied AFTER the scale. For instance for imagenet statistics the mean would be sl::float3(0.485, 0.456, 0.406)
# \param stddev : standard deviation, statistic to normalized the pixel values, applied AFTER the scale. For instance for imagenet statistics the standard deviation would be sl::float3(0.229, 0.224, 0.225)
# \param keep_aspect_ratio : indicates if the original width and height ratio should be kept using padding (sometimes refer to as letterboxing) or if the image should be stretched
# \param swap_RB_channels : indicates if the Red and Blue channels should be swapped (RGB<->BGR or RGBA<->BGRA)
# \return ERROR_CODE : The error code gives information about the success of the function
#
# Example usage, for a 416x416 squared RGB image (letterboxed), with a scale factor of 1/255, and using the imagenet statistics for normalization:
# \code
#
#    image = sl.Mat()
#    blob = sl.Mat()
#    resolution = sl.Resolution(416,416)
#    scale = 1.0/255.0 # Scale factor to apply to each pixel value
#    keep_aspect_ratio = True # Add padding to keep the aspect ratio
#    swap_RB_channels = True # ZED SDK outputs BGR images, so we need to swap the R and B channels
#    zed.retrieve_image(image, sl.VIEW.LEFT, type=sl.MEM.GPU) # Get the ZED image (GPU only is more efficient in that case)
#    err = sl.blob_from_image(image, blob, resolution, scale, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), keep_aspect_ratio, swap_RB_channels)
#    # By default the blob is in GPU memory, you can move it to CPU memory if needed
#    blob.update_cpu_from_gpu()
#
# \endcode

def blob_from_image(mat1: Mat, mat2: Mat, resolution: Resolution, scale: float, mean: tuple, stdev: tuple, keep_aspect_ratio: bool, swap_RB_channels: bool) -> ERROR_CODE:
    return _error_code_cache.get(<int>c_blobFromImage(mat1.mat, mat2.mat, resolution.resolution, scale, Vector3[float](mean[0], mean[1], mean[2]), Vector3[float](stdev[0], stdev[1], stdev[2]), keep_aspect_ratio, swap_RB_channels), ERROR_CODE.FAILURE)

##
# \brief Check if the camera is a ZED One (Monocular) or ZED (Stereo)
# \param camera_model : The camera model to check
#
def is_camera_one(camera_model: MODEL) -> bool:
    return c_isCameraOne(<c_MODEL>(<int>camera_model.value))

##
# \brief Check if a resolution is available for a given camera model
# \param resolution : Resolution to check
# \param camera_model : The camera model to check
#
def is_resolution_available(resolution: RESOLUTION, camera_model: MODEL) -> bool:
    return c_isResolutionAvailable(<c_RESOLUTION>(<int>resolution.value), <c_MODEL>(<int>camera_model.value))

##
# \brief Check if a frame rate is available for a given resolution and camera model
# \param fps : Frame rate to check
# \param resolution : Resolution to check
# \param camera_model : The camera model to check
#
def is_FPS_available(int fps, resolution: RESOLUTION, camera_model: MODEL) -> bool:
    return c_isFPSAvailable(fps, <c_RESOLUTION>(<int>resolution.value), <c_MODEL>(<int>camera_model.value))

##
# \brief Check if a resolution for a given camera model is available for HDR
# \param resolution : Resolution to check
# \param camera_model : The camera model to check
#
def is_HDR_available(resolution: RESOLUTION, camera_model: MODEL) -> bool:
    return c_supportHDR(<c_RESOLUTION>(<int>resolution.value), <c_MODEL>(<int>camera_model.value))

##
# Class representing a rotation for the positional tracking module.
# \ingroup PositionalTracking_group
#
# It inherits from the generic sl.Matrix3f class.
cdef class Rotation(Matrix3f):
    cdef c_Rotation* rotation
    def __cinit__(self):
        if type(self) is Rotation:
            self.rotation = self.mat = new c_Rotation()
    
    def __dealloc__(self):
        if type(self) is Rotation:
            del self.rotation

    ##
    # Deep copy from another sl.Rotation.
    # \param rot : sl.Rotation to copy.
    def init_rotation(self, rot: Rotation) -> None:
        for i in range(9):
            self.rotation.r[i] = rot.rotation.r[i]

    ##
    # Initializes the sl.Rotation from a sl.Matrix3f.
    # \param matrix : sl.Matrix3f to be used.
    def init_matrix(self, matrix: Matrix3f) -> None:
        for i in range(9):
            self.rotation.r[i] = matrix.mat.r[i]

    ##
    # Initializes the sl.Rotation from an sl.Orientation.
    # \param orient : sl.Orientation to be used.
    def init_orientation(self, orient: Orientation) -> None:
        self.rotation.setOrientation(orient.orientation)

    ##
    # Initializes the sl.Rotation from an angle and an axis.
    # \param angle : Rotation angle in radian.
    # \param axis : 3D axis to rotate around.
    def init_angle_translation(self, angle: float, axis: Translation) -> None:
        cdef c_Rotation tmp = c_Rotation(angle, axis.translation)        
        for i in range(9):
            self.rotation.r[i] = tmp.r[i]

    ##
    # Sets the sl.Rotation from an sl.Orientation.
    # \param py_orientation : sl.Orientation containing the rotation to set.
    def set_orientation(self, py_orientation: Orientation) -> None:
        self.rotation.setOrientation(py_orientation.orientation)

    ##
    # Returns the sl.Orientation corresponding to the current sl.Rotation.
    # \return Rotation of the current orientation.
    def get_orientation(self) -> Orientation:
        py_orientation = Orientation()        
        py_orientation.orientation = self.rotation.getOrientation()
        return py_orientation

    ##
    # Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.
    # \return Rotation vector (NumPy array) created from the sl.Orientation values.
    def get_rotation_vector(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.rotation.getRotationVector()[i]
        return arr

    ##
    # Sets the sl.Rotation from a rotation vector (using Rodrigues' transformation).
    # \param input0 : ```rx``` component of the rotation vector.
    # \param input1 : ```ry``` component of the rotation vector.
    # \param input2 : ```rz``` component of the rotation vector.
    def set_rotation_vector(self, input0: float, input1: float, input2: float) -> None:
        self.rotation.setRotationVector(Vector3[float](input0, input1, input2))

    ##
    # Converts the sl.Rotation into Euler angles.
    # \param radian : Whether the angle will be returned in radian or degree. Default: True
    # \return Euler angles (NumPy array) created from the sl.Rotation values representing the rotations around the X, Y and Z axes using YZX convention.
    def get_euler_angles(self, radian=True) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.rotation.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    ##
    # Sets the sl.Rotation from Euler angles.
    # \param input0 : Roll value.
    # \param input1 : Pitch value.
    # \param input2 : Yaw value.
    # \param radian : Whether the angle is in radian or degree. Default: True
    def set_euler_angles(self, input0: float, input1: float, input2: float, radian=True) -> None:
        if isinstance(radian, bool):
            self.rotation.setEulerAngles(Vector3[float](input0, input1, input2), radian)
        else:
            raise TypeError("Argument is not of boolean type.")

##
# Class representing a translation for the positional tracking module.
# \ingroup PositionalTracking_group
# 
# sl.Translation is a vector as ```[tx, ty, tz]```.
# \n You can access the data with the \ref get() method that returns a NumPy array.
cdef class Translation:
    cdef c_Translation translation
    def __cinit__(self):
        self.translation = c_Translation()

    ##
    # Deep copy from another sl.Translation.
    # \param tr : sl.Translation to copy.
    def init_translation(self, tr: Translation) -> None:
        self.translation = c_Translation(tr.translation)

    ##
    # Initializes the sl.Translation with its components.
    # \param t1 : First component.
    # \param t2 : Second component.
    # \param t3 : Third component.
    def init_vector(self, t1: float, t2: float, t3: float) -> None:
        self.translation = c_Translation(t1, t2, t3)

    ##
    # Normalizes the current sl.Translation.
    def normalize(self) -> None:
        self.translation.normalize()

    ##
    # Gets the normalized sl.Translation of a given sl.Translation.
    # \param tr : sl.Translation to be get the normalized translation from.
    # \return Another sl.Translation object equal to [\b tr.normalize()](\ref normalize).
    def normalize_translation(self, tr: Translation) -> Translation:
        py_translation = Translation()
        py_translation.translation = self.translation.normalize(tr.translation)
        return py_translation

    ##
    # Gets the size of the sl.Translation.
    # \return Size of the sl.Translation.
    def size(self) -> int:
        return self.translation.size()

    ##
    # Computes the dot product of two sl.Translation objects.
    # \param tr1 : First sl.Translation to get the dot product from.
    # \param tr2 : Sencond sl.Translation to get the dot product from.
    # \return Dot product of \b tr1 and \b tr2.
    def dot_translation(tr1: Translation, tr2: Translation) -> float:
        py_translation = Translation()
        return py_translation.translation.dot(tr1.translation,tr2.translation)

    ##
    # Gets the sl.Translation as an NumPy array.
    # \return NumPy array containing the components of the sl.Translation.
    def get(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.translation(i)
        return arr

    def __mul__(Translation self, Orientation other):
        tr = Translation()
        tr.translation = self.translation * other.orientation
        return tr

##
# Class representing an orientation/quaternion for the positional tracking module.
# \ingroup PositionalTracking_group
#
# sl.Orientation is a vector defined as ```[ox, oy, oz, ow]```.
cdef class Orientation:
    cdef c_Orientation orientation
    def __cinit__(self):
        self.orientation = c_Orientation()

    ##
    # Deep copy from another sl.Orientation.
    # \param orient : sl.Orientation to copy.
    def init_orientation(self, orient: Orientation) -> None:
        self.orientation = c_Orientation(orient.orientation)

    ##
    # Initializes the sl.Orientation with its components.
    # \param v0 : ox component.
    # \param v1 : oy component.
    # \param v2 : oz component.
    # \param v3 : ow component.
    def init_vector(self, v0: float, v1: float, v2: float, v3: float) -> None:
        self.orientation = c_Orientation(Vector4[float](v0, v1, v2, v3))

    ##
    # Initializes the sl.Orientation from an sl.Rotation.
    #
    # It converts the sl.Rotation representation to the sl.Orientation one.
    # \param rotation : sl.Rotation to be used.
    def init_rotation(self, rotation: Rotation) -> None:
        self.orientation = c_Orientation(rotation.rotation[0])

    ##
    # Initializes the sl.Orientation from a vector represented by two sl.Translation.
    # \param tr1 : First point of the vector.
    # \param tr2 : Second point of the vector.
    def init_translation(self, tr1: Translation, tr2: Translation) -> None:
        self.orientation = c_Orientation(tr1.translation, tr2.translation)

    ##
    # Sets the rotation component of the current sl.Transform from an sl.Rotation.
    # \param py_rotation : sl.Rotation to be used.
    def set_rotation_matrix(self, py_rotation: Rotation) -> None:
        self.orientation.setRotationMatrix(py_rotation.rotation[0])

    ##
    # Returns the current sl.Orientation as an sl.Rotation.
    # \return The rotation computed from the orientation data.
    def get_rotation_matrix(self) -> Rotation:
        cdef c_Rotation tmp = self.orientation.getRotationMatrix()
        py_rotation = Rotation()
        for i in range(9):
            py_rotation.rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Sets the current sl.Orientation to identity.
    def set_identity(self) -> None:
        self.orientation.setIdentity()
        # return self

    ##
    # Creates an sl.Orientation initialized to identity.
    # \return Identity sl.Orientation.
    def identity(self, orient=Orientation()) -> Orientation:
        (<Orientation>orient).orientation.setIdentity()
        return orient

    ##
    # Fills the current sl.Orientation with zeros.
    def set_zeros(self) -> None:
        self.orientation.setZeros()

    ##
    # Creates an sl.Orientation filled with zeros.
    # \return sl.Orientation filled with zeros.
    def zeros(self, orient=Orientation()) -> Orientation:
        (<Orientation>orient).orientation.setZeros()
        return orient

    ##
    # Normalizes the current sl.Orientation.
    def normalize(self) -> None:
        self.orientation.normalise()

    ##
    # Gets the normalized sl.Orientation of a given sl.Orientation.
    # \param orient : sl.Orientation to be get the normalized orientation from.
    # \return Another sl.Orientation object equal to [\b orient.normalize()](\ref normalize).
    @staticmethod
    def normalize_orientation(orient: Orientation) -> Orientation:
        orient.orientation.normalise()
        return orient

    ##
    # Gets the size of the sl.Orientation.
    # \return Size of the sl.Orientation.
    def size(self) -> int:
        return self.orientation.size()

    ##
    # Returns a numpy array of the \ref Orientation .
    # \return A numpy array of the \ref Orientation .
    def get(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(self.size())
        for i in range(self.size()):
            arr[i] = <float>self.orientation(i)
        return arr

    def __mul__(Orientation self, Orientation other):
        orient = Orientation()
        orient.orientation = self.orientation * other.orientation
        return orient


##
# Class representing a transformation (translation and rotation) for the positional tracking module.
# \ingroup PositionalTracking_group
# 
# It can be used to create any type of Matrix4x4 or sl::Matrix4f that must be specifically used for handling a rotation and position information (OpenGL, Tracking, etc.).
# \n It inherits from the generic sl::Matrix4f class.
cdef class Transform(Matrix4f):
    cdef c_Transform *transform
    def __cinit__(self):
        if type(self) is Transform:
            self.transform = self.mat = new c_Transform()

    def __dealloc__(self):
        if type(self) is Transform:
            del self.transform

    ##
    # Deep copy from another sl.Transform.
    # \param motion : sl.Transform to copy.
    def init_transform(self, motion: Transform) -> None:
        for i in range(16):
            self.transform.m[i] = motion.transform.m[i]

    ##
    # Initializes the sl.Transform from a sl.Matrix4f.
    # \param matrix : sl.Matrix4f to be used.
    def init_matrix(self, matrix: Matrix4f) -> None:
        for i in range(16):
            self.transform.m[i] = matrix.mat.m[i]

    ##
    # Initializes the sl.Transform from an sl.Rotation and a sl.Translation.
    # \param rot : sl.Rotation to be used.
    # \param tr : sl.Translation to be used.
    def init_rotation_translation(self, rot: Rotation, tr: Translation) -> None:
        cdef c_Transform tmp = c_Transform(rot.rotation[0], tr.translation)
        for i in range(16):
            self.transform.m[i] = tmp.m[i]

    ##
    # Initializes the sl.Transform from an sl.Orientation and a sl.Translation.
    # \param orient : \ref Orientation to be used
    # \param tr : \ref Translation to be used
    def init_orientation_translation(self, orient: Orientation, tr: Translation) -> None:
        cdef c_Transform tmp = c_Transform(orient.orientation, tr.translation)
        for i in range(16):
            self.transform.m[i] = tmp.m[i]

    ##
    # Sets the rotation component of the current sl.Transform from an sl.Rotation.
    # \param py_rotation : sl.Rotation to be used.
    def set_rotation_matrix(self, py_rotation: Rotation) -> None:
        self.transform.setRotationMatrix(py_rotation.rotation[0])

    ##
    # Returns the sl.Rotation corresponding to the current sl.Transform.
    # \return sl.Rotation created from the sl.Transform values.
    # \warning The given sl.Rotation contains a copy of the sl.Transform values.
    def get_rotation_matrix(self) -> Rotation:
        cdef c_Rotation tmp = self.transform.getRotationMatrix()
        py_rotation = Rotation()
        for i in range(9):
            py_rotation.rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Sets the translation component of the current sl.Transform from an sl.Translation.
    # \param py_translation : sl.Translation to be used.
    def set_translation(self, py_translation: Translation) -> None:
        self.transform.setTranslation(py_translation.translation)

    ##
    # Returns the sl.Translation corresponding to the current sl.Transform.
    # \return sl.Translation created from the sl.Transform values.
    # \warning The given sl.Translation contains a copy of the sl.Transform values.
    def get_translation(self) -> Translation:
        py_translation = Translation()
        py_translation.translation = self.transform.getTranslation()
        return py_translation

    ##
    # Sets the orientation component of the current sl.Transform from an sl.Orientation.
    # \param py_orientation : sl.Orientation to be used.
    def set_orientation(self, py_orientation: Orientation) -> None:
        self.transform.setOrientation(py_orientation.orientation)

    ##
    # Returns the sl.Orientation corresponding to the current sl.Transform.
    # \return sl.Orientation created from the sl.Transform values.
    # \warning The given sl.Orientation contains a copy of the sl.Transform values.
    def get_orientation(self) -> Orientation:
        py_orientation = Orientation()
        py_orientation.orientation = self.transform.getOrientation()
        return py_orientation

    ##
    # Returns the 3x1 rotation vector obtained from 3x3 rotation matrix using Rodrigues formula.
    # \return Rotation vector (NumPy array) created from the sl.Transform values.
    def get_rotation_vector(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.transform.getRotationVector()[i]
        return arr

    ##
    # Sets the rotation component of the sl.Transform with a 3x1 rotation vector (using Rodrigues' transformation).
    # \param input0 : ```rx``` component of the rotation vector.
    # \param input1 : ```ry``` component of the rotation vector.
    # \param input2 : ```rz``` component of the rotation vector.
    def set_rotation_vector(self, input0: float, input1: float, input2: float) -> None:
        self.transform.setRotationVector(Vector3[float](input0, input1, input2))

    ##
    # Converts the rotation component of the sl.Transform into Euler angles.
    # \param radian : Whether the angle will be returned in radian or degree. Default: True
    # \return Euler angles (Numpy array) created from the sl.Transform values representing the rotations around the X, Y and Z axes using YZX convention.
    def get_euler_angles(self, radian=True) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.transform.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of boolean type.")
        return arr

    ##
    # Sets the rotation component of the sl.Transform from Euler angles.
    # \param input0 : Roll value.
    # \param input1 : Pitch value.
    # \param input2 : Yaw value.
    # \param radian : Whether the angle is in radian or degree. Default: True
    def set_euler_angles(self, input0: float, input1: float, input2: float, radian=True) -> None:
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
# | PLY            | Contains only vertices and faces. |
# | PLY_BIN        | Contains only vertices and faces encoded in binary. |
# | OBJ            | Contains vertices, normals, faces, and texture information (if possible). |
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
# | RGB            | The texture will be on 3 channels. |
# | RGBA           | The texture will be on 4 channels. |
class MESH_TEXTURE_FORMAT(enum.Enum):
    RGB = <int>c_MESH_TEXTURE_FORMAT.RGB
    RGBA = <int>c_MESH_TEXTURE_FORMAT.RGBA
    LAST = <int>c_MESH_TEXTURE_FORMAT.MESH_TEXTURE_FORMAT_LAST

##
# Lists available mesh filtering intensities.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | LOW            | Clean the mesh by closing small holes and removing isolated faces. |
# | MEDIUM         | Soft faces decimation and smoothing. |
# | HIGH           | Drastically reduce the number of faces and apply a soft smooth. |
class MESH_FILTER(enum.Enum):
    LOW = <int>c_MESH_FILTER.LOW
    MEDIUM = <int>c_MESH_FILTER.MESH_FILTER_MEDIUM
    HIGH = <int>c_MESH_FILTER.HIGH

##
# Lists the available plane types detected based on the orientation.
#
# \ingroup SpatialMapping_group
#
# | Enumerator |                  |
# |------------|------------------|
# | HORIZONTAL | Horizontal plane, such as a tabletop, floor, etc. |
# | VERTICAL   | Vertical plane, such as a wall. |
# | UNKNOWN    | Unknown plane orientation. |
class PLANE_TYPE(enum.Enum):
    HORIZONTAL = <int>c_PLANE_TYPE.HORIZONTAL
    VERTICAL = <int>c_PLANE_TYPE.VERTICAL
    UNKNOWN = <int>c_PLANE_TYPE.UNKNOWN
    LAST = <int>c_PLANE_TYPE.PLANE_TYPE_LAST

##
# Class containing a set of parameters for the [mesh filtration](\ref Mesh.filter) functionality.
# \ingroup SpatialMapping_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class MeshFilterParameters:
    cdef c_MeshFilterParameters* meshFilter
    def __cinit__(self):
        self.meshFilter = new c_MeshFilterParameters(c_MESH_FILTER.LOW)

    def __dealloc__(self):
        del self.meshFilter

    ##
    # Set the filtering intensity.
    # \param filter : Desired sl.MESH_FILTER.
    def set(self, filter=MESH_FILTER.LOW) -> None:
        if isinstance(filter, MESH_FILTER):
            self.meshFilter.set(<c_MESH_FILTER>(<int>filter.value))
        else:
            raise TypeError("Argument is not of MESH_FILTER type.")

    ##
    # Saves the current set of parameters into a file to be reloaded with the \ref load() method.
    # \param filename : Name of the file which will be created to store the parameters.
    # \return True if the file was successfully saved, otherwise False.
    # \warning For security reasons, the file must not already exist.
    # \warning In case a file already exists, the method will return False and existing file will not be updated.
    def save(self, filename: str) -> bool:
        filename_save = filename.encode()
        return self.meshFilter.save(String(<char*> filename_save))

    ##
    # Loads a set of parameters from the values contained in a previously \ref save() "saved" file.
    # \param filename : Path to the file from which the parameters will be loaded.
    # \return True if the file was successfully loaded, otherwise False.
    def load(self, filename: str) -> bool:
        filename_load = filename.encode()
        return self.meshFilter.load(String(<char*> filename_load))

##
# Class representing a sub-point cloud containing local vertices and colors.
# \ingroup SpatialMapping_group
#
# \note \ref vertices and \ref normals have the same size.
cdef class PointCloudChunk :
    cdef c_PointCloudChunk chunk

    def __cinit__(self):
        self.chunk = c_PointCloudChunk()
    ##
    # NumPy array of vertices.
    # Vertices are defined by a colored 3D point ```[x, y, z, rgba]```.
    @property
    def vertices(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 4), dtype=np.float32)
        for i in range(self.chunk.vertices.size()):
            for j in range(4):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    ##
    # NumPy array of normals.
    # Normals are defined by three components ```[nx, ny, nz]```.
    # \note A normal is defined for each vertex.
    @property
    def normals(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3), dtype=np.float32)
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    ##
    # Timestamp of the latest update.
    @property
    def timestamp(self) -> int:
        return self.chunk.timestamp

    ##
    # 3D centroid of the chunk.
    @property
    def barycenter(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3, dtype=np.float32)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    ##
    # Whether the point cloud chunk has been updated by an inner process.
    @property
    def has_been_updated(self) -> bool:
        return self.chunk.has_been_updated

    ##
    # Clears all data.
    def clear(self) -> None:
        self.chunk.clear()

##
# Class representing a sub-mesh containing local vertices and triangles.
# \ingroup SpatialMapping_group
#
# Vertices and normals have the same size and are linked by id stored in triangles.
# \note \ref uv contains data only if your mesh have textures (by loading it or after calling sl.Mesh.apply_texture()).
cdef class Chunk:
    cdef c_Chunk chunk
    def __cinit__(self):
        self.chunk = c_Chunk()

    ##
    # NumPy array of vertices.
    # Vertices are defined by a 3D point ```[x, y, z]```.
    @property
    def vertices(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.chunk.vertices.size(), 3), dtype=np.float32)
        for i in range(self.chunk.vertices.size()):
            for j in range(3):
                arr[i,j] = self.chunk.vertices[i].ptr()[j]
        return arr

    ##
    # NumPy array of triangles/faces.
    # Triangle defined as a set of three vertices indexes ```[v1, v2, v3]```.
    @property
    def triangles(self) -> np.array[int]:
        cdef np.ndarray arr = np.zeros((self.chunk.triangles.size(), 3), dtype = np.uint32)
        for i in range(self.chunk.triangles.size()):
            for j in range(3):
                arr[i,j] = self.chunk.triangles[i].ptr()[j]
        return arr

    ##
    # NumPy array of normals.
    # Normals are defined by three components ```[nx, ny, nz]```.
    # \note A normal is defined for each vertex.
    @property
    def normals(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.chunk.normals.size(), 3), dtype=np.float32)
        for i in range(self.chunk.normals.size()):
            for j in range(3):
                arr[i,j] = self.chunk.normals[i].ptr()[j]
        return arr

    ##
    # NumPy array of colors.
    # Colors are defined by three components ```[r, g, b]```.
    # \note A color is defined for each vertex.
    @property
    def colors(self) -> np.array[int]:
        cdef np.ndarray arr = np.zeros((self.chunk.colors.size(), 3), dtype = np.ubyte)
        for i in range(self.chunk.colors.size()):
            for j in range(3):
                arr[i,j] = self.chunk.colors[i].ptr()[j]
        return arr

    ##
    # UVs defines the 2D projection of each vertices onto the texture.
    # Values are normalized [0, 1] and start from the bottom left corner of the texture (as requested by OpenGL).
    # \n In order to display a textured mesh you need to bind the texture and then draw each triangle by picking its uv values.
    # \note Contains data only if your mesh has textures (by loading it or calling sl.Mesh.apply_texture()).
    @property
    def uv(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.chunk.uv.size(), 2), dtype=np.float32)
        for i in range(self.chunk.uv.size()):
            for j in range(2):
                arr[i,j] = self.chunk.uv[i].ptr()[j]
        return arr

    ##
    # Timestamp of the latest update.
    @property
    def timestamp(self) -> int:
        return self.chunk.timestamp

    ##
    # 3D centroid of the chunk.
    @property
    def barycenter(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3, dtype=np.float32)
        for i in range(3):
            arr[i] = self.chunk.barycenter[i]
        return arr

    ##
    # Whether the chunk has been updated by an inner process.
    @property
    def has_been_updated(self) -> bool:
        return self.chunk.has_been_updated

    ##
    # Clears all data.
    def clear(self) -> None:
        self.chunk.clear()

##
# Class representing a fused point cloud and containing the geometric and color data of the scene captured by the spatial mapping module.
# \ingroup SpatialMapping_group
#
# By default the fused point cloud is defined as a set of point cloud chunks.
# \n This way we update only the required data, avoiding a time consuming remapping process every time a small part of the sl.FusedPointCloud cloud is changed.
cdef class FusedPointCloud :
    cdef c_FusedPointCloud* fpc
    def __cinit__(self):
        self.fpc = new c_FusedPointCloud()

    def __dealloc__(self):
        del self.fpc

    ##
    # List of chunks constituting the sl.FusedPointCloud.
    @property
    def chunks(self) -> list[PointCloudChunk]:
        list = []
        for i in range(self.fpc.chunks.size()):
            py_chunk = PointCloudChunk()
            py_chunk.chunk = self.fpc.chunks[i]
            list.append(py_chunk)
        return list

    ##
    # Gets a chunk from \ref chunks.
    def __getitem__(self, x) -> PointCloudChunk:
        return self.chunks[x]

    ##
    # NumPy array of vertices.
    # Vertices are defined by a colored 3D point ```[x, y, z, rgba]```.
    @property
    def vertices(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.fpc.vertices.size(), 4), dtype=np.float32)
        for i in range(self.fpc.vertices.size()):
            for j in range(4):
                arr[i,j] = self.fpc.vertices[i].ptr()[j]
        return arr

    ##
    # NumPy array of normals.
    # Normals are defined by three components ```[nx, ny, nz]```.
    # \note A normal is defined for each vertex.
    @property
    def normals(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.fpc.normals.size(), 3), dtype=np.float32)
        for i in range(self.fpc.normals.size()):
            for j in range(3):
                arr[i,j] = self.fpc.normals[i].ptr()[j]
        return arr

    ##
    # Saves the current sl.FusedPointCloud into a file.
    # \param filename : Path of the file to store the fused point cloud in.
    # \param typeMesh : File extension type. Default: [sl.MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT).
    # \param id : Set of chunks to be saved. Default: (empty) (all chunks are saved)
    # \return True if the file was successfully saved, otherwise False.
    #
    # \note This method operates on the sl.FusedPointCloud not on \ref chunks.
    # \note This way you can save different parts of your sl.FusedPointCloud by updating it with \ref update_from_chunklist().
    def save(self, filename: str, typeMesh=MESH_FILE_FORMAT.OBJ, id=[]) -> bool:
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.fpc.save(String(filename.encode()), <c_MESH_FILE_FORMAT>(<int>typeMesh.value), id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    ##
    # Loads the fused point cloud from a file.
    # \param filename : Path of the file to load the fused point cloud from.
    # \param update_chunk_only : Whether to only load data in \ref chunks (and not \ref vertices / \ref normals).\n Default: False.
    # \return True if the mesh was successfully loaded, otherwise False.
    #
    # \note Updating a sl.FusedPointCloud is time consuming. Consider using only \ref chunks for better performances.
    def load(self, filename: str, update_chunk_only=False) -> bool:
        if isinstance(update_chunk_only, bool):
            return self.fpc.load(String(filename.encode()), update_chunk_only)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Clears all the data.
    def clear(self) -> None:
        self.fpc.clear()

    ##
    # Updates \ref vertices and \ref normals from chunk data pointed by the given list of id.
    # \param id : Indices of chunks which will be concatenated. Default: (empty).
    # \note If the given list of id is empty, all chunks will be used to update the current sl.FusedPointCloud.
    def update_from_chunklist(self, id=[]) -> None:
        self.fpc.updateFromChunkList(id)

    ##
    # Computes the total number of points stored in all chunks.
    # \return The number of points stored in all chunks.
    def get_number_of_points(self) -> int:
        return self.fpc.getNumberOfPoints()


##
# Class representing a mesh and containing the geometric (and optionally texture) data of the scene captured by the spatial mapping module.
# \ingroup SpatialMapping_group
# 
# By default the mesh is defined as a set of chunks.
# \n This way we update only the data that has to be updated avoiding a time consuming remapping process every time a small part of the sl.Mesh is updated.
cdef class Mesh:
    cdef c_Mesh* mesh
    def __cinit__(self):
        self.mesh = new c_Mesh()

    def __dealloc__(self):
        del self.mesh

    ##
    # List of chunks constituting the sl.Mesh.
    @property
    def chunks(self) -> list[Chunk]:
        list_ = []
        for i in range(self.mesh.chunks.size()):        
            py_chunk = Chunk()
            py_chunk.chunk = self.mesh.chunks[i]
            list_.append(py_chunk)
        return list_

    ##
    # Gets a chunk from \ref chunks.
    def __getitem__(self, x) -> Chunk:
        return self.chunks[x]

    ##
    # Filters the mesh.
    # The resulting mesh is smoothed, small holes are filled, and small blobs of non-connected triangles are deleted.
    # \param params : Filtering parameters. Default: a preset of sl.MeshFilterParameters.
    # \param update_chunk_only : Whether to only update \ref chunks (and not \ref vertices / \ref normals / \ref triangles).\n Default: False.
    # \return True if the mesh was successfully filtered, otherwise False.
    #
    # \note The filtering is a costly operation.
    # \note It is not recommended to call it every time you retrieve a mesh but only at the end of your spatial mapping process.
    def filter(self, params=MeshFilterParameters(), update_chunk_only=False) -> bool:
        if isinstance(update_chunk_only, bool):
            return self.mesh.filter(deref((<MeshFilterParameters>params).meshFilter), update_chunk_only)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Applies a texture to the mesh.
    # By using this method you will get access to \ref uv, and \ref texture.
    # \n The number of triangles in the mesh may slightly differ before and after calling this method due to missing texture information.
    # \n There is only one texture for the mesh, the uv of each chunk are expressed for it in its entirety.
    # \n NumPy arrays of \ref vertices / \ref normals and \ref uv have now the same size.
    # \param texture_format : Number of channels desired for the computed texture.\n Default: [sl.MESH_TEXTURE_FORMAT.RGB](\ref MESH_TEXTURE_FORMAT).
    # \return True if the mesh was successfully textured, otherwise False.
    #
    # \note This method can be called as long as you do not start a new spatial mapping process (due to shared memory).
    # \note This method can require a lot of computation time depending on the number of triangles in the mesh.
    # \note It is recommended to call it once at the end of your spatial mapping process.
    # 
    # \warning The sl.SpatialMappingParameters.save_texture parameter must be set to True when enabling the spatial mapping to be able to apply the textures.
    # \warning The mesh should be filtered before calling this method since \ref filter() will erase the textures.
    # \warning The texturing is also significantly slower on non-filtered meshes.
    def apply_texture(self, texture_format=MESH_TEXTURE_FORMAT.RGB) -> bool:
        if isinstance(texture_format, MESH_TEXTURE_FORMAT):
            return self.mesh.applyTexture(<c_MESH_TEXTURE_FORMAT>(<int>texture_format.value))
        else:
            raise TypeError("Argument is not of MESH_TEXTURE_FORMAT type.")

    ##
    # Saves the current sl.Mesh into a file.
    # \param filename : Path of the file to store the mesh in.
    # \param typeMesh : File extension type. Default: [sl.MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT).
    # \param id : Set of chunks to be saved. Default: (empty) (all chunks are saved)
    # \return True if the file was successfully saved, otherwise False.
    # 
    # \note Only [sl.MESH_FILE_FORMAT.OBJ](\ref MESH_FILE_FORMAT) supports textures data.
    # \note This method operates on the sl.Mesh not on \ref chunks.
    # \note This way you can save different parts of your sl.Mesh by updating it with \ref update_mesh_from_chunklist().
    def save(self, filename: str, typeMesh=MESH_FILE_FORMAT.OBJ, id=[]) -> bool:
        if isinstance(typeMesh, MESH_FILE_FORMAT):
            return self.mesh.save(String(filename.encode()), <c_MESH_FILE_FORMAT>(<int>typeMesh.value), id)
        else:
            raise TypeError("Argument is not of MESH_FILE_FORMAT type.")

    ##
    # Loads the mesh from a file.
    # \param filename : Path of the file to load the mesh from.
    # \param update_mesh : Whether to only load data in \ref chunks (and not \ref vertices / \ref normals / \ref triangles).\n Default: False.
    # \return True if the mesh was successfully loaded, otherwise False.
    #
    # \note Updating a sl::Mesh is time consuming. Consider using only \ref chunks for better performances.
    def load(self, filename: str, update_mesh=False) -> bool:
        if isinstance(update_mesh, bool):
            return self.mesh.load(String(filename.encode()), update_mesh)
        else:
            raise TypeError("Argument is not of boolean type.")

    ##
    # Clears all the data.
    def clear(self) -> None:
        self.mesh.clear()

    ##
    # NumPy array of vertices.
    # Vertices are defined by a 3D point ```[x, y, z]```.
    @property
    def vertices(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.mesh.vertices.size(), 3), dtype=np.float32)
        for i in range(self.mesh.vertices.size()):
            for j in range(3):
                arr[i,j] = self.mesh.vertices[i].ptr()[j]
        return arr

    ##
    # NumPy array of triangles/faces.
    # Triangle defined as a set of three vertices indexes ```[v1, v2, v3]```.
    @property
    def triangles(self) -> np.array[int]:
        cdef np.ndarray arr = np.zeros((self.mesh.triangles.size(), 3))
        for i in range(self.mesh.triangles.size()):
            for j in range(3):
                arr[i,j] = self.mesh.triangles[i].ptr()[j]
        return arr

    ##
    # NumPy array of normals.
    # Normals are defined by three components ```[nx, ny, nz]```.
    # \note A normal is defined for each vertex.
    @property
    def normals(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.mesh.normals.size(), 3), dtype=np.float32)
        for i in range(self.mesh.normals.size()):
            for j in range(3):
                arr[i,j] = self.mesh.normals[i].ptr()[j]
        return arr

    ##
    # NumPy array of colors.
    # Colors are defined by three components ```[r, g, b]```.
    # \note A color is defined for each vertex.
    @property
    def colors(self) -> np.array[int]:
        cdef np.ndarray arr = np.zeros((self.mesh.colors.size(), 3), dtype=np.ubyte)
        for i in range(self.mesh.colors.size()):
            for j in range(3):
                arr[i,j] = self.mesh.colors[i].ptr()[j]
        return arr

    ##
    # UVs defines the 2D projection of each vertices onto the texture.
    # Values are normalized [0, 1] and start from the bottom left corner of the texture (as requested by OpenGL).
    # In order to display a textured mesh you need to bind the texture and then draw each triangle by picking its uv values.
    # \note Contains data only if your mesh has textures (by loading it or calling sl.Mesh.apply_texture()).
    @property
    def uv(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros((self.mesh.uv.size(), 2), dtype=np.float32)
        for i in range(self.mesh.uv.size()):
            for j in range(2):
                arr[i,j] = self.mesh.uv[i].ptr()[j]
        return arr

    ##
    # Texture of the sl.Mesh.
    # \note Contains data only if your mesh has textures (by loading it or calling sl.Mesh.apply_texture()).
    @property
    def texture(self) -> Mat:
        py_texture = Mat()
        py_texture.mat = self.mesh.texture
        return py_texture

    ##
    # Computes the total number of triangles stored in all chunks.
    # \return The number of triangles stored in all chunks.
    def get_number_of_triangles(self) -> int:
        return self.mesh.getNumberOfTriangles()

    ##
    # Compute the indices of boundary vertices.
    # \return The indices of boundary vertices.
    def get_boundaries(self) -> np.array[int]:
        cdef np.ndarray arr = np.zeros(self.mesh.getBoundaries().size(), dtype=np.uint32)
        for i in range(self.mesh.getBoundaries().size()):
            arr[i] = self.mesh.getBoundaries()[i]
        return arr

    ##
    # Merges current chunks.
    # This method can be used to merge chunks into bigger sets to improve rendering process.
    # \param faces_per_chunk : Number of faces per chunk.
    #
    # \note This method is useful for Unity, which does not handle chunks with more than 65K vertices.
    # \warning This method should not be called during spatial mapping process since mesh updates will revert this changes.
    def merge_chunks(self, faces_per_chunk: int) -> None:
        self.mesh.mergeChunks(faces_per_chunk)

    ##
    # Estimates the gravity vector.
    # This method looks for a dominant plane in the whole mesh considering that it is the floor (or a horizontal plane).
    # \return The estimated gravity vector (NumPy array).
    #
    # \note This can be used to find the gravity to create realistic physical interactions.
    def get_gravity_estimate(self) -> np.array[float]:
        gravity = self.mesh.getGravityEstimate()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = gravity[i]
        return arr

    ##
    # Computes the id list of visible chunks from a specific point of view.
    # \param camera_pose : Point of view (given in the same reference as the vertices).
    # \return The list of id of visible chunks.
    def get_visible_list(self, camera_pose: Transform) -> list[int]:
        return self.mesh.getVisibleList(camera_pose.transform[0])

    ##
    # Computes the id list of chunks close to a specific point of view.
    # \param camera_pose : Point of view (given in the same reference as the vertices).
    # \param radius : Radius determining closeness (given in the same unit as the mesh).
    # \return The list of id of chunks close to the given point.
    def get_surrounding_list(self, camera_pose: Transform, radius: float) -> list[int]:
        return self.mesh.getSurroundingList(camera_pose.transform[0], radius)

    ##
    # Updates \ref vertices / \ref normals / \ref triangles / \ref uv from chunk data pointed by the given list of id.
    # \param id : Indices of chunks which will be concatenated. Default: (empty).
    # \note If the given list of id is empty, all chunks will be used to update the current sl.Mesh.
    def update_mesh_from_chunklist(self, id=[]) -> None:
        self.mesh.updateMeshFromChunkList(id)

##
# Class representing a plane defined by a point and a normal, or a plane equation.
# \ingroup SpatialMapping_group
#
# Other elements can be extracted such as the mesh, the 3D bounds, etc.
# \note The plane measurements are expressed in reference defined by sl.RuntimeParameters.measure3D_reference_frame.
cdef class Plane:
    cdef c_Plane plane
    def __cinit__(self):
        self.plane = c_Plane()

    ##
    # Type of the plane defined by its orientation.
    # \note It is deduced from the gravity vector and is therefore not available with on [sl.MODEL.ZED](\ref MODEL).
    # \note [sl.MODEL.ZED](\ref MODEL) will give [sl.PLANE_TYPE.UNKNOWN](\ref PLANE_TYPE) for every planes.
    @property
    def type(self) -> PLANE_TYPE:
        return PLANE_TYPE(<int>self.plane.type)

    @type.setter
    def type(self, type_):
        if isinstance(type_, PLANE_TYPE) :
            self.plane.type = <c_PLANE_TYPE>(<int>type_.value)
        else :
            raise TypeError("Argument is not of PLANE_TYPE type")

    ##
    # Gets the plane normal vector.
    # \return sl.Plane normalized normal vector (NumPy array).
    def get_normal(self) -> np.array[float]:
        normal = self.plane.getNormal()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = normal[i]
        return arr

    ##
    # Gets the plane center point
    # \return sl.Plane center point 
    def get_center(self) -> np.array[float]:
        center = self.plane.getCenter()
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = center[i]
        return arr

    ##
    # Gets the plane pose relative to the global reference frame.
    # \param py_pose : sl.Transform to fill (or it creates one by default).
    # \return Transformation matrix (rotation and translation) of the plane pose.
    # \note Can be used to transform the global reference frame center ```(0, 0, 0)``` to the plane center.
    def get_pose(self, py_pose = Transform()) -> Transform:
        tmp =  self.plane.getPose()
        for i in range(16):
            (<Transform>py_pose).transform.m[i] = tmp.m[i]
        return py_pose

    ##
    # Gets the width and height of the bounding rectangle around the plane contours.
    # \return Width and height of the bounding plane contours (NumPy array).
    # \warning This value is expressed in the plane reference frame.
    def get_extents(self) -> np.array[float]:
        extents = self.plane.getExtents()
        cdef np.ndarray arr = np.zeros(2)
        for i in range(2):
            arr[i] = extents[i]
        return arr

    ##
    # Gets the plane equation.
    # \return Plane equation coefficients ```[a, b, c, d]``` (NumPy array).
    # \note The plane equation has the following form: ```ax + by + cz = d```.
    def get_plane_equation(self) -> np.array[float]:
        plane_eq = self.plane.getPlaneEquation()
        cdef np.ndarray arr = np.zeros(4)
        for i in range(4):
            arr[i] = plane_eq[i]
        return arr

    ##
    # Gets the polygon bounds of the plane.
    # \return Vector of 3D points forming a polygon bounds corresponding to the current visible limits of the plane (NumPy array).
    def get_bounds(self) -> np.array[float][float]:
        cdef np.ndarray arr = np.zeros((self.plane.getBounds().size(), 3))
        for i in range(self.plane.getBounds().size()):
            for j in range(3):
                arr[i,j] = self.plane.getBounds()[i].ptr()[j]
        return arr

    ##
    # Compute and return the mesh of the bounds polygon.
    # \return sl::Mesh representing the plane delimited by the visible bounds.
    def extract_mesh(self) -> Mesh:
        ext_mesh = self.plane.extractMesh()
        pymesh = Mesh()
        pymesh.mesh[0] = ext_mesh
        return pymesh

    ##
    # Gets the distance between the input point and the projected point alongside the normal vector onto the plane (the closest point on the plane).
    # \param point : Point to project into the plane.
    # \return The Euclidean distance between the input point and the projected point.
    def get_closest_distance(self, point=[0,0,0]) -> float:
        cdef Vector3[float] vec = Vector3[float](point[0], point[1], point[2])
        return self.plane.getClosestDistance(vec)

    ##
    # Clears all the data.
    def clear(self) -> None:
        self.plane.clear()

##
# Lists the spatial mapping resolution presets.
# \ingroup SpatialMapping_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | HIGH           | Creates a detailed geometry.\n Requires lots of memory. |
# | MEDIUM         | Small variations in the geometry will disappear.\n Useful for big objects. |
# | LOW            | Keeps only huge variations of the geometry.\n Useful for outdoor purposes. |
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
# | SHORT          | Only depth close to the camera will be used during spatial mapping. |
# | MEDIUM         | Medium depth range.  |
# | LONG           | Takes into account objects that are far.\n Useful for outdoor purposes. |
# | AUTO           | Depth range will be computed based on current sl.Camera state and parameters. |
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
# | MESH           | The geometry is represented by a set of vertices connected by edges and forming faces.\n No color information is available. |
# | FUSED_POINT_CLOUD | The geometry is represented by a set of 3D colored points. |
class SPATIAL_MAP_TYPE(enum.Enum):
    MESH = <int>c_SPATIAL_MAP_TYPE.MESH
    FUSED_POINT_CLOUD = <int>c_SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

##
# Lists available LIVE input type in the ZED SDK.
# \ingroup Video_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | USB            | USB input mode   |
# | GMSL           | GMSL input mode \note Only on NVIDIA Jetson. |
# | AUTO           | Automatically select the input type.\n Trying first for available USB cameras, then GMSL. |
class BUS_TYPE(enum.Enum):
    USB = <int>c_BUS_TYPE.USB
    GMSL = <int>c_BUS_TYPE.GMSL
    AUTO = <int>c_BUS_TYPE.AUTO
    LAST = <int>c_BUS_TYPE.LAST

##
# Class defining the input type used in the ZED SDK.
# \ingroup Video_group
# It can be used to select a specific camera with an id or serial number, or from a SVO file.
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
    # Set the input as the camera with specified id (for USB or GMSL cameras only).
    # \param id : Id of the camera to open.
    # \param bus_type : Whether the camera is a USB or a GMSL camera.
    def set_from_camera_id(self, id: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
        self.input.setFromCameraID(id, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Set the input as the camera with specified serial number (for USB or GMSL cameras).
    # \param camera_serial_number : Serial number of the camera to open.
    # \param bus_type : Whether the camera is a USB or a GMSL camera.
    def set_from_serial_number(self, serial_number: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
        self.input.setFromSerialNumber(serial_number, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Set the input as the svo specified with the filename
    # \param svo_input_filename : The path to the desired SVO file
    def set_from_svo_file(self, svo_input_filename: str) -> None:
        filename = svo_input_filename.encode()
        self.input.setFromSVOFile(String(<char*> filename))

    ##
    # Set the input to stream with the specified ip and port
    # \param sender_ip : The IP address of the streaming sender
    # \param port : The port on which to listen. Default: 30000
    # \note The protocol used for the streaming module is based on RTP/RTCP.
    # \warning Port must be even number, since the port+1 is used for control data.
    def set_from_stream(self, sender_ip: str, port=30000) -> None:
        sender_ip_ = sender_ip.encode()
        self.input.setFromStream(String(<char*>sender_ip_), port)

    ##
    # Returns the current input type.
    def get_type(self) -> INPUT_TYPE:
        return INPUT_TYPE(<int>self.input.getType())
    
    ##
    # Returns the current input configuration as a string e.g: SVO name, serial number, streaming ip, etc.
    def get_configuration(self) -> str:
        return to_str(self.input.getConfiguration()).decode()
    
    ##
    # Check whether the input is set.
    def is_init(self) -> bool:
        return self.input.isInit()

##
# Class containing the options used to initialize the sl.Camera object.
# \ingroup Video_group
# 
# This class allows you to select multiple parameters for the sl.Camera such as the selected camera, resolution, depth mode, coordinate system, and units of measurement.
# \n Once filled with the desired options, it should be passed to the sl.Camera.open() method.
#
# \code
#
#        import pyzed.sl as sl
#
#        def main() :
#            zed = sl.Camera() # Create a ZED camera object
#
#            init_params = sl.InitParameters()   # Set initial parameters
#            init_params.sdk_verbose = 0  # Disable verbose mode
#
#            # Use the camera in LIVE mode
#            init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD1080 video mode
#            init_params.camera_fps = 30 # Set fps at 30
#
#            # Or use the camera in SVO (offline) mode
#            #init_params.set_from_svo_file("xxxx.svo")
#
#            # Or use the camera in STREAM mode
#            #init_params.set_from_stream("192.168.1.12", 30000)
#
#            # Other parameters are left to their default values
#
#            # Open the camera
#            err = zed.open(init_params)
#            if err != sl.ERROR_CODE.SUCCESS:
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
# With its default values, it opens the camera in live mode at \ref RESOLUTION "sl.RESOLUTION.HD720"
# (or \ref RESOLUTION "sl.RESOLUTION.HD1200" for the ZED X/X Mini) and sets the depth mode to \ref DEPTH_MODE "sl.DEPTH_MODE.NEURAL"
# \n You can customize it to fit your application.
# \note The parameters can also be saved and reloaded using its \ref save() and \ref load() methods.
cdef class InitParameters:
    cdef c_InitParameters* init
    ##
    # Default constructor.
    #
    # All the parameters are set to their default and optimized values.
    # \param camera_resolution : Chosen \ref camera_resolution
    # \param camera_fps : Chosen \ref camera_fps
    # \param svo_real_time_mode : Activates \ref svo_real_time_mode
    # \param depth_mode : Chosen \ref depth_mode
    # \param coordinate_units : Chosen \ref coordinate_units
    # \param coordinate_system : Chosen \ref coordinate_system
    # \param sdk_verbose : Sets \ref sdk_verbose
    # \param sdk_gpu_id : Chosen \ref sdk_gpu_id
    # \param depth_minimum_distance : Chosen \ref depth_minimum_distance
    # \param depth_maximum_distance : Chosen \ref depth_maximum_distance
    # \param camera_disable_self_calib : Activates \ref camera_disable_self_calib
    # \param camera_image_flip : Sets \ref camera_image_flip
    # \param enable_right_side_measure : Activates \ref enable_right_side_measure
    # \param sdk_verbose_log_file : Chosen \ref sdk_verbose_log_file
    # \param depth_stabilization : Activates \ref depth_stabilization
    # \param input_t : Chosen input_t (\ref InputType )
    # \param optional_settings_path : Chosen \ref optional_settings_path
    # \param sensors_required : Activates \ref sensors_required
    # \param enable_image_enhancement : Activates \ref enable_image_enhancement
    # \param optional_opencv_calibration_file : Sets \ref optional_opencv_calibration_file
    # \param open_timeout_sec : Sets \ref open_timeout_sec
    # \param async_grab_camera_recovery : Sets \ref async_grab_camera_recovery
    # \param grab_compute_capping_fps : Sets \ref grab_compute_capping_fps
    # \param enable_image_validity_check : Sets \ref enable_image_validity_check
    # \param maximum_working_resolution : Sets \ref maximum_working_resolution
    #
    # \code
    # params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30, depth_mode=sl.DEPTH_MODE.NEURAL)
    # \endcode
    def __cinit__(self, camera_resolution=RESOLUTION.AUTO, camera_fps=0,
                  svo_real_time_mode=False,
                  depth_mode=DEPTH_MODE.NEURAL,
                  coordinate_units=UNIT.MILLIMETER,
                  coordinate_system=COORDINATE_SYSTEM.IMAGE,
                  sdk_verbose=1, sdk_gpu_id=-1, depth_minimum_distance=-1.0, depth_maximum_distance=-1.0, camera_disable_self_calib=False,
                  camera_image_flip=FLIP_MODE.OFF, enable_right_side_measure=False,
                  sdk_verbose_log_file="", depth_stabilization=30, input_t=InputType(),
                  optional_settings_path="",sensors_required=False,
                  enable_image_enhancement=True, optional_opencv_calibration_file="", 
                  open_timeout_sec=5.0, async_grab_camera_recovery=False, grab_compute_capping_fps=0,
                  enable_image_validity_check=False, async_image_retrieval=False, maximum_working_resolution=Resolution(0,0)) -> InitParameters:
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
            isinstance(grab_compute_capping_fps, float) or isinstance(grab_compute_capping_fps, int) and
            isinstance(async_image_retrieval, bool) and
            isinstance(enable_image_validity_check, int) or isinstance(enable_image_validity_check, bool)) :

            filelog = sdk_verbose_log_file.encode()
            fileoption = optional_settings_path.encode()
            filecalibration = optional_opencv_calibration_file.encode()
            self.init = new c_InitParameters(<c_RESOLUTION>(<int>camera_resolution.value), camera_fps,
                                            svo_real_time_mode, <c_DEPTH_MODE>(<int>depth_mode.value),
                                            <c_UNIT>(<int>coordinate_units.value), <c_COORDINATE_SYSTEM>(<int>coordinate_system.value), sdk_verbose, sdk_gpu_id,
                                            <float>(depth_minimum_distance), <float>(depth_maximum_distance), camera_disable_self_calib
                                            , <c_FLIP_MODE>(<int>camera_image_flip.value),
                                            enable_right_side_measure,
                                            String(<char*> filelog), depth_stabilization,
                                            <CUcontext> 0, (<InputType>input_t).input, String(<char*> fileoption), sensors_required, enable_image_enhancement,
                                            String(<char*> filecalibration), <float>(open_timeout_sec), 
                                            async_grab_camera_recovery, <float>(grab_compute_capping_fps), <bool>(async_image_retrieval),
                                            <int>(enable_image_validity_check),  (<Resolution>maximum_working_resolution).resolution)
        else:
            raise TypeError("Argument is not of right type.")

    def __dealloc__(self):
        del self.init

    ##
    # Saves the current set of parameters into a file to be reloaded with the \ref load() method.
    # \param filename : Name of the file which will be created to store the parameters (extension '.yml' will be added if not set).
    # \return True if file was successfully saved, otherwise False.
    # \warning For security reason, the file must not exist.
    # \warning In case a file already exists, the method will return False and existing file will not be updated
    #
    # \code
    # init_params = sl.InitParameters()  # Set initial parameters
    # init_params.sdk_verbose = 1 # Enable verbose mode
    # init_params.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read
    # init_params.save("initParameters.conf") # Export the parameters into a file
    # \endcode
    def save(self, filename: str) -> bool:
        filename_save = filename.encode()
        return self.init.save(String(<char*> filename_save))

    ##
    # Loads a set of parameters from the values contained in a previously \ref save() "saved" file.
    # \param filename : Path to the file from which the parameters will be loaded  (extension '.yml' will be added at the end of the filename if not set).
    # \return True if the file was successfully loaded, otherwise false.
    #
    # \code
    # init_params = sl.InitParameters()  # Set initial parameters
    # init_params.load("initParameters.conf") # Load the init_params from a previously exported file
    # \endcode
    def load(self, filename: str) -> bool:
        filename_load = filename.encode()
        return self.init.load(String(<char*> filename_load))

    ##
    # Desired camera resolution.
    # \note Small resolutions offer higher framerate and lower computation time.
    # \note In most situations, \ref RESOLUTION "sl.RESOLUTION.HD720" at 60 FPS is the best balance between image quality and framerate.
    #
    # Default: <ul>
    # <li>ZED X/X Mini: \ref RESOLUTION "sl.RESOLUTION.HD1200"</li>
    # <li>other cameras: \ref RESOLUTION "sl.RESOLUTION.HD720"</li></ul>
    # \note Available resolutions are listed here: sl.RESOLUTION.
    @property
    def camera_resolution(self) -> RESOLUTION:
        return RESOLUTION(<int>self.init.camera_resolution)

    @camera_resolution.setter
    def camera_resolution(self, value):
        if isinstance(value, RESOLUTION):
            self.init.camera_resolution = <c_RESOLUTION>(<int>value.value)
        else:
            raise TypeError("Argument must be of RESOLUTION type.")

    ##
    # Requested camera frame rate.
    #
    # If set to 0, the highest FPS of the specified \ref camera_resolution will be used.
    # \n Default: 0
    # \n\n See sl.RESOLUTION for a list of supported frame rates.
    # \note If the requested \ref camera_fps is unsupported, the closest available FPS will be used.
    @property
    def camera_fps(self) -> int:
        return self.init.camera_fps

    @camera_fps.setter
    def camera_fps(self, int value):
        self.init.camera_fps = value

    ##
    # Enable async image retrieval.
    #
    # If set to true will camera image retrieve at a framerate different from \ref grab() application framerate. This is useful for recording SVO or sending camera stream at different rate than application.
    # \n Default: false
    @property
    def async_image_retrieval(self) -> bool:
        return self.init.async_image_retrieval

    @async_image_retrieval.setter
    def async_image_retrieval(self, bool value):
        self.init.async_image_retrieval = value

    ##
    # Requires the successful opening of the motion sensors before opening the camera.
    #
    # Default: False.
    #
    # \note If set to false, the ZED SDK will try to <b>open and use</b> the IMU (second USB device on USB2.0) and will open the camera successfully even if the sensors failed to open.
    #
    # This can be used for example when using a USB3.0 only extension cable (some fiber extension for example).
    # \note This parameter only impacts the LIVE mode.
    # \note If set to true, sl.Camera.open() will fail if the sensors cannot be opened.
    # \note This parameter should be used when the IMU data must be available, such as object detection module or when the gravity is needed.
    # 
    # \n\note This setting is not taken into account for \ref MODEL "sl.MODEL.ZED" camera since it does not include sensors.
    @property
    def sensors_required(self) -> bool:
        return self.init.sensors_required

    @sensors_required.setter
    def sensors_required(self, value: bool):
        self.init.sensors_required = value

    ##
    # Enable the Enhanced Contrast Technology, to improve image quality.
    #
    # Default: True.
    # 
    # \n If set to true, image enhancement will be activated in camera ISP. Otherwise, the image will not be enhanced by the IPS.
    # \note This only works for firmware version starting from 1523 and up.
    @property
    def enable_image_enhancement(self) -> bool:
        return self.init.enable_image_enhancement

    @enable_image_enhancement.setter
    def enable_image_enhancement(self, value: bool):
        self.init.enable_image_enhancement = value

    ##
    # Defines if sl.Camera object return the frame in real time mode.
    #
    # When playing back an SVO file, each call to sl.Camera.grab() will extract a new frame and use it.
    # \n However, it ignores the real capture rate of the images saved in the SVO file.
    # \n Enabling this parameter will bring the SDK closer to a real simulation when playing back a file by using the images' timestamps.
    # \n Default: False
    # \note sl.Camera.grab() will return an error when trying to play too fast, and frames will be dropped when playing too slowly.
    @property
    def svo_real_time_mode(self) -> bool:
        return self.init.svo_real_time_mode

    @svo_real_time_mode.setter
    def svo_real_time_mode(self, value: bool):
        self.init.svo_real_time_mode = value

    ##
    # sl.DEPTH_MODE to be used.
    #
    # The ZED SDK offers several sl.DEPTH_MODE, offering various levels of performance and accuracy.
    # \n This parameter allows you to set the sl.DEPTH_MODE that best matches your needs.
    # \n Default: \ref DEPTH_MODE "sl.DEPTH_MODE.NEURAL"
    # \note Available depth mode are listed here: sl.DEPTH_MODE.
    @property
    def depth_mode(self) -> DEPTH_MODE:
        return DEPTH_MODE(<int>self.init.depth_mode)

    @depth_mode.setter
    def depth_mode(self, value):
        if isinstance(value, DEPTH_MODE):
            self.init.depth_mode = <c_DEPTH_MODE>(<int>value.value)
        else:
            raise TypeError("Argument must be of DEPTH_MODE type.")

    ##
    # Unit of spatial data (depth, point cloud, tracking, mesh, etc.) for retrieval.
    #
    # Default: \ref UNIT "sl.UNIT.MILLIMETER"
    @property
    def coordinate_units(self) -> UNIT:
        return UNIT(<int>self.init.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value):
        if isinstance(value, UNIT):
            self.init.coordinate_units = <c_UNIT>(<int>value.value)
        else:
            raise TypeError("Argument must be of UNIT type.")

    ##
    # sl.COORDINATE_SYSTEM to be used as reference for positional tracking, mesh, point clouds, etc.
    #
    # This parameter allows you to select the sl.COORDINATE_SYSTEM used by the sl.Camera object to return its measures.
    # \n This defines the order and the direction of the axis of the coordinate system.
    # \n Default: \ref COORDINATE_SYSTEM "sl.COORDINATE_SYSTEM.IMAGE"
    @property
    def coordinate_system(self) -> COORDINATE_SYSTEM:
        return COORDINATE_SYSTEM(<int>self.init.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, COORDINATE_SYSTEM):
            self.init.coordinate_system = <c_COORDINATE_SYSTEM>(<int>value.value)
        else:
            raise TypeError("Argument must be of COORDINATE_SYSTEM type.")

    ##
    # Enable the ZED SDK verbose mode.
    #
    # This parameter allows you to enable the verbosity of the ZED SDK to get a variety of runtime information in the console.
    # \n When developing an application, enabling verbose (<code>\ref sdk_verbose >= 1</code>) mode can help you understand the current ZED SDK behavior.
    # \n However, this might not be desirable in a shipped version.
    # \n Default: 0 (no verbose message)
    # \note The verbose messages can also be exported into a log file.
    # \note See \ref sdk_verbose_log_file for more.
    @property
    def sdk_verbose(self) -> int:
        return self.init.sdk_verbose

    @sdk_verbose.setter
    def sdk_verbose(self, value: int):
        self.init.sdk_verbose = value

    ##
    # NVIDIA graphics card id to use.
    #
    # By default the SDK will use the most powerful NVIDIA graphics card found.
    # \n However, when running several applications, or using several cameras at the same time, splitting the load over available GPUs can be useful.
    # \n This parameter allows you to select the GPU used by the sl.Camera using an ID from 0 to n-1 GPUs in your PC.
    # \n Default: -1
    # \note A non-positive value will search for all CUDA capable devices and select the most powerful.
    @property
    def sdk_gpu_id(self) -> int:
        return self.init.sdk_gpu_id

    @sdk_gpu_id.setter
    def sdk_gpu_id(self, value: int):
        self.init.sdk_gpu_id = value

    ##
    # Minimum depth distance to be returned, measured in the sl.UNIT defined in \ref coordinate_units.
    #
    # This parameter allows you to specify the minimum depth value (from the camera) that will be computed.
    #
    # \n In stereovision (the depth technology used by the camera), looking for closer depth values can have a slight impact on performance and memory consumption.
    # \n On most of modern GPUs, performance impact will be low. However, the impact of memory footprint will be visible.
    # \n In cases of limited computation power, increasing this value can provide better performance.
    # \n Default: -1 (corresponding values are available <a href="https://www.stereolabs.com/docs/depth-sensing/depth-settings#depth-range">here</a>)
    #
    # \note \ref depth_minimum_distance value cannot be greater than 3 meters.
    # \note 0 will imply that \ref depth_minimum_distance is set to the minimum depth possible for each camera
    # (those values are available <a href="https://www.stereolabs.com/docs/depth-sensing/depth-settings#depth-range">here</a>).
    @property
    def depth_minimum_distance(self) -> float:
        return  self.init.depth_minimum_distance

    @depth_minimum_distance.setter
    def depth_minimum_distance(self, value: float):
        self.init.depth_minimum_distance = value

    ##
    # Maximum depth distance to be returned, measured in the sl.UNIT defined in \ref coordinate_units.
    #
    # When estimating the depth, the ZED SDK uses this upper limit to turn higher values into <b>inf</b> ones.
    # \note Changing this value has no impact on performance and doesn't affect the positional tracking nor the spatial mapping.
    # \note It only change values the depth, point cloud and normals.
    @property
    def depth_maximum_distance(self) -> float:
        return self.init.depth_maximum_distance

    @depth_maximum_distance.setter
    def depth_maximum_distance(self, value: float):
        self.init.depth_maximum_distance = value

    ##
    # Disables the self-calibration process at camera opening.
    #
    # At initialization, sl.Camera runs a self-calibration process that corrects small offsets from the device's factory calibration.
    # \n A drawback is that calibration parameters will slightly change from one (live) run to another, which can be an issue for repeatability.
    # \n If set to true, self-calibration will be disabled and calibration parameters won't be optimized, raw calibration parameters from the configuration file will be used.
    # \n Default: false
    # \note In most situations, self calibration should remain enabled.
    # \note You can also trigger the self-calibration at anytime after sl.Camera.open() by calling sl.Camera.update_self_calibration(), even if this parameter is set to true.
    @property
    def camera_disable_self_calib(self) -> bool:
        return self.init.camera_disable_self_calib

    @camera_disable_self_calib.setter
    def camera_disable_self_calib(self, value: bool):
        self.init.camera_disable_self_calib = value

    ##
    # Defines if a flip of the images is needed.
    #
    # If you are using the camera upside down, setting this parameter to \ref FLIP_MODE "sl.FLIP_MODE.ON" will cancel its rotation.
    # \n The images will be horizontally flipped.
    # \n Default: \ref FLIP_MODE "sl.FLIP_MODE.AUTO"
    # \note From ZED SDK 3.2 a new sl.FLIP_MODE enum was introduced to add the automatic flip mode detection based on the IMU gravity detection.
    # \note This does not work on \ref MODEL "sl.MODEL.ZED" cameras since they do not have the necessary sensors.
    @property
    def camera_image_flip(self) -> FLIP_MODE:
        return FLIP_MODE(self.init.camera_image_flip)

    @camera_image_flip.setter
    def camera_image_flip(self, value):
        if isinstance(value, FLIP_MODE):
            self.init.camera_image_flip = <c_FLIP_MODE>(<int>value.value)
        else:
            raise TypeError("Argument must be of FLIP_MODE type.")

    ##
    # Enable the measurement computation on the right images.
    #
    # By default, the ZED SDK only computes a single depth map, aligned with the left camera image.
    # \n This parameter allows you to enable \ref MEASURE "sl.MEASURE.DEPTH_RIGHT" and other \ref MEASURE "sl.MEASURE.XXX_RIGHT" at the cost of additional computation time.
    # \n For example, mixed reality pass-through applications require one depth map per eye, so this parameter can be activated.
    # \n Default: False
    @property
    def enable_right_side_measure(self) -> bool:
        return self.init.enable_right_side_measure

    @enable_right_side_measure.setter
    def enable_right_side_measure(self, value: bool):
        self.init.enable_right_side_measure = value

    ##
    # File path to store the ZED SDK logs (if \ref sdk_verbose is enabled).
    #
    # The file will be created if it does not exist.
    # \n Default: ""
    #
    # \note Setting this parameter to any value will redirect all standard output print calls of the entire program.
    # \note This means that your own standard output print calls will be redirected to the log file.
    # \warning The log file won't be cleared after successive executions of the application.
    # \warning This means that it can grow indefinitely if not cleared. 
    @property
    def sdk_verbose_log_file(self) -> str:
        if not self.init.sdk_verbose_log_file.empty():
            return self.init.sdk_verbose_log_file.get().decode()
        else:
            return ""

    @sdk_verbose_log_file.setter
    def sdk_verbose_log_file(self, value: str):
        value_filename = value.encode()
        self.init.sdk_verbose_log_file.set(<char*>value_filename)

    ##
	# Defines whether the depth needs to be stabilized and to what extent.
    #
    # Regions of generated depth map can oscillate from one frame to another.
    # \n These oscillations result from a lack of texture (too homogeneous) on an object and by image noise.
    # \n This parameter controls a stabilization filter that reduces these oscillations.
    # \n In the range [0-100]: <ul>
    # <li>0 disable the depth stabilization (raw depth will be return)</li>
    # <li>stabilization smoothness is linear from 1 to 100</li></ul>
    # Default: 30
    #
    # \note The stabilization uses the positional tracking to increase its accuracy, 
    # so the positional tracking module will be enabled automatically when set to a value different from 0.
    # \note Note that calling sl.Camera.enable_positional_tracking() with your own parameters afterwards is still possible.
    @property
    def depth_stabilization(self) -> int:
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
    # def input(self) -> InputType:
    #    input_t = InputType()
    #    input_t.input = self.init.input
    #    return input_t

    # @input.setter
    def input(self, input_t: InputType):
        self.init.input = input_t.input

    input = property(None, input)

    ##
    # Optional path where the ZED SDK has to search for the settings file (<i>SN<XXXX>.conf</i> file).
    #
    # This file contains the calibration information of the camera.
    # \n Default: ""
    #
    # \note The settings file will be searched in the default directory: <ul>
    # <li><b>Linux</b>: <i>/usr/local/zed/settings/</i></li> 
    # <li><b>Windows</b>: <i>C:/ProgramData/stereolabs/settings</i></li></ul>
    # 
    # \note If a path is specified and no file has been found, the ZED SDK will search the settings file in the default directory.
    # \note An automatic download of the settings file (through <b>ZED Explorer</b> or the installer) will still download the files on the default path.
    #
    # \code
    # init_params = sl.InitParameters() # Set initial parameters
    # home = "/path/to/home"
    # path = home + "/Documents/settings/" # assuming /path/to/home/Documents/settings/SNXXXX.conf exists. Otherwise, it will be searched in /usr/local/zed/settings/
    # init_params.optional_settings_path = path
    # \endcode
    @property
    def optional_settings_path(self) -> str:
        if not self.init.optional_settings_path.empty():
            return self.init.optional_settings_path.get().decode()
        else:
            return ""

    @optional_settings_path.setter
    def optional_settings_path(self, value: str):
        value_filename = value.encode()
        self.init.optional_settings_path.set(<char*>value_filename)

    ##
    # Optional path where the ZED SDK can find a file containing the calibration information of the camera computed by OpenCV.
    #
    # \note Using this will disable the factory calibration of the camera.
    # \note The file must be in a XML/YAML/JSON formatting provided by OpenCV.
    # \note It also must contain the following keys: Size, K_LEFT (intrinsic left), K_RIGHT (intrinsic right),
    # D_LEFT (distortion left), D_RIGHT (distortion right), R (extrinsic rotation), T (extrinsic translation).
    # \warning Erroneous calibration values can lead to poor accuracy in all ZED SDK modules.
    @property
    def optional_opencv_calibration_file(self) -> str:
        if not self.init.optional_opencv_calibration_file.empty():
            return self.init.optional_opencv_calibration_file.get().decode()
        else:
            return ""

    @optional_opencv_calibration_file.setter
    def optional_opencv_calibration_file(self, value: str):
        value_filename = value.encode()
        self.init.optional_opencv_calibration_file.set(<char*>value_filename)

    ##
    # Define a timeout in seconds after which an error is reported if the sl.Camera.open() method fails.
    #
    # Set to '-1' to try to open the camera endlessly without returning error in case of failure.
    # \n Set to '0' to return error in case of failure at the first attempt.
    # \n Default: 5.0
    # \note This parameter only impacts the LIVE mode.
    @property
    def open_timeout_sec(self) -> float:
        return self.init.open_timeout_sec

    @open_timeout_sec.setter
    def open_timeout_sec(self, value: float):
        self.init.open_timeout_sec = value

    ##
    # Define the behavior of the automatic camera recovery during sl.Camera.grab() method call.
    #
    # When async is enabled and there's an issue with the communication with the sl.Camera object,
    # sl.Camera.grab() will exit after a short period and return the \ref ERROR_CODE "sl.ERROR_CODE.CAMERA_REBOOTING" warning.
    # \n The recovery will run in the background until the correct communication is restored.
    # \n When \ref async_grab_camera_recovery is false, the sl.Camera.grab() method is blocking and will return
    # only once the camera communication is restored or the timeout is reached. 
    # \n Default: False
    @property
    def async_grab_camera_recovery(self) -> bool:
        return self.init.async_grab_camera_recovery

    @async_grab_camera_recovery.setter
    def async_grab_camera_recovery(self, value: bool):
        self.init.async_grab_camera_recovery = value

    ##
    # Define a computation upper limit to the grab frequency.
    #
    # This can be useful to get a known constant fixed rate or limit the computation load while keeping a short exposure time by setting a high camera capture framerate.
    # \n The value should be inferior to the sl.InitParameters.camera_fps and strictly positive.
    # \note  It has no effect when reading an SVO file.
    #
    # This is an upper limit and won't make a difference if the computation is slower than the desired compute capping FPS.
    # \note Internally the sl.Camera.grab() method always tries to get the latest available image while respecting the desired FPS as much as possible.
    @property
    def grab_compute_capping_fps(self) -> float:
        return self.init.grab_compute_capping_fps

    @grab_compute_capping_fps.setter
    def grab_compute_capping_fps(self, value: float):
        self.init.grab_compute_capping_fps = value
    
    ##
    # Enable or disable the image validity verification.
    # This will perform additional verification on the image to identify corrupted data. This verification is done in the sl.Camera.grab() method and requires some computations.
    # \n If an issue is found, the sl.Camera.grab() method will output a warning as [sl.ERROR_CODE.CORRUPTED_FRAME](\ref ERROR_CODE).
    # \n This version doesn't detect frame tearing currently.
    # \n Default: False (disabled)
    @property
    def enable_image_validity_check(self) -> int:
        return self.init.enable_image_validity_check

    @enable_image_validity_check.setter
    def enable_image_validity_check(self, value: int):
        self.init.enable_image_validity_check = value

    ##
    #  Set a maximum size for all SDK output, like retrieveImage and retrieveMeasure functions.
    # 
    # This will override the default (0,0) and instead of outputting native image size sl::Mat, the ZED SDK will take this size as default.
    # A custom lower size can also be used at runtime, but not bigger. This is used for internal optimization of compute and memory allocations
    # 
    # The default is similar to previous version with (0,0), meaning native image size
    # 
    # \note: if maximum_working_resolution field are lower than 64, it will be interpreted as dividing scale factor;
    # - maximum_working_resolution = sl::Resolution(1280, 16) -> 1280 x (image_height/2) = 1280 x half height
    # - maximum_working_resolution = sl::Resolution(4, 4) -> (image_width/4) x (image_height/4) = quarter size   
    # 
    @property
    def maximum_working_resolution(self) -> Resolution:
        return Resolution(self.init.maximum_working_resolution.width, self.init.maximum_working_resolution.height)

    @maximum_working_resolution.setter
    def maximum_working_resolution(self, Resolution value):
        self.init.maximum_working_resolution = c_Resolution(value.width, value.height)

    ##
    # Defines the input source with a camera id to initialize and open an sl.Camera object from.
    # \param id : Id of the desired camera to open.
    # \param bus_type : sl.BUS_TYPE of the desired camera to open.
    def set_from_camera_id(self, id: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
        self.init.input.setFromCameraID(id, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Defines the input source with a serial number to initialize and open an sl.Camera object from.
    # \param serial_number : Serial number of the desired camera to open.
    # \param bus_type : sl.BUS_TYPE of the desired camera to open.
    def set_from_serial_number(self, serial_number: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
        self.init.input.setFromSerialNumber(serial_number, <c_BUS_TYPE>(<int>(bus_type.value)))

    ##
    # Defines the input source with an SVO file to initialize and open an sl.Camera object from.
    # \param svo_input_filename : Path to the desired SVO file to open.
    def set_from_svo_file(self, svo_input_filename: str) -> None:
        filename = svo_input_filename.encode()
        self.init.input.setFromSVOFile(String(<char*> filename))

    ##
    # Defines the input source from a stream to initialize and open an sl.Camera object from.
    # \param sender_ip : IP address of the streaming sender.
    # \param port : Port on which to listen. Default: 30000
    def set_from_stream(self, sender_ip: str, port=30000) -> None:
        sender_ip_ = sender_ip.encode()
        self.init.input.setFromStream(String(<char*>sender_ip_), port)

##
# Class containing parameters that defines the behavior of sl.Camera.grab().
# \ingroup Depth_group
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class RuntimeParameters:
    cdef c_RuntimeParameters* runtime
    ##
    # Default constructor.
    #
    # All the parameters are set to their default values.
    # \param enable_depth : Activates \ref enable_depth
    # \param enable_fill_mode : Activates \ref enable_fill_mode
    # \param confidence_threshold : Chosen \ref confidence_threshold
    # \param texture_confidence_threshold : Chosen \ref texture_confidence_threshold
    # \param measure3D_reference_frame : Chosen \ref measure3D_reference_frame
    # \param remove_saturated_areas : Activates \ref remove_saturated_areas
    def __cinit__(self, enable_depth=True, enable_fill_mode=False,
                  confidence_threshold = 95, texture_confidence_threshold = 100,
                  measure3D_reference_frame=REFERENCE_FRAME.CAMERA, remove_saturated_areas = True) -> RuntimeParameters:
        if (isinstance(enable_depth, bool)
            and isinstance(enable_fill_mode, bool)
            and isinstance(confidence_threshold, int) and
            isinstance(measure3D_reference_frame, REFERENCE_FRAME)
            and isinstance(remove_saturated_areas, bool)):

            self.runtime = new c_RuntimeParameters(enable_depth, enable_fill_mode, confidence_threshold, texture_confidence_threshold,
                                                 <c_REFERENCE_FRAME>(<int>measure3D_reference_frame.value),remove_saturated_areas)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.runtime

    ##
    # Saves the current set of parameters into a file to be reloaded with the \ref load() method.
    # \param filename : Name of the file which will be created to store the parameters (extension '.yml' will be added if not set).
    # \return True if the file was successfully saved, otherwise False.
    # \warning For security reasons, the file must not already exist.
    # \warning In case a file already exists, the method will return False and existing file will not be updated.
    def save(self, filename: str) -> bool:
        filename_save = filename.encode()
        return self.runtime.save(String(<char*> filename_save))

    ##
    # Loads a set of parameters from the values contained in a previously \ref save() "saved" file.
    # \param filename : Path to the file from which the parameters will be loaded (extension '.yml' will be added at the end of the filename if not detected).
    # \return True if the file was successfully loaded, otherwise False.
    def load(self, filename: str) -> bool:
        filename_load = filename.encode()
        return self.runtime.load(String(<char*> filename_load))

    ##
    # Defines if the depth map should be computed.
    #
    # Default: True
    # \note If set to False, only the images are available.
    @property
    def enable_depth(self) -> bool:
        return self.runtime.enable_depth

    @enable_depth.setter
    def enable_depth(self, value: bool):
        self.runtime.enable_depth = value

    ##
    # Defines if the depth map should be completed or not.
    #
    # Default: False
    # \note It is similar to the removed sl.SENSING_MODE.FILL.
    # \warning Enabling this will override the confidence values \ref confidence_threshold and \ref texture_confidence_threshold as well as \ref remove_saturated_areas.
    @property
    def enable_fill_mode(self) -> bool:
        return self.runtime.enable_fill_mode

    @enable_fill_mode.setter
    def enable_fill_mode(self, value: bool):
        self.runtime.enable_fill_mode = value

    ##
    # Reference frame in which to provides the 3D measures (point cloud, normals, etc.).
    #
    # Default: \ref REFERENCE_FRAME "sl.REFERENCE_FRAME.CAMERA"
    @property
    def measure3D_reference_frame(self) -> REFERENCE_FRAME:
        return REFERENCE_FRAME(<int>self.runtime.measure3D_reference_frame)

    @measure3D_reference_frame.setter
    def measure3D_reference_frame(self, value):
        if isinstance(value, REFERENCE_FRAME):
            self.runtime.measure3D_reference_frame = <c_REFERENCE_FRAME>(<int>value.value)
        else:
            raise TypeError("Argument must be of REFERENCE type.")

    ##
    # Threshold to reject depth values based on their confidence.
    #
    # Each depth pixel has a corresponding confidence (\ref MEASURE "sl.MEASURE.CONFIDENCE") in the range [1, 100].
    # \n Decreasing this value will remove depth data from both objects edges and low textured areas, to keep only confident depth estimation data.
    # \n Default: 95 (no depth pixel will be rejected)
    # \note Pixels with a value close to 100 are not to be trusted. Accurate depth pixels tends to be closer to lower values.
    # \note It can be seen as a probability of error, scaled to 100.
    @property
    def confidence_threshold(self) -> int:
        return self.runtime.confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value):
        self.runtime.confidence_threshold = value

    ##
    # Threshold to reject depth values based on their texture confidence.
    #
    # The texture confidence range is [1, 100].
    # \n Decreasing this value will remove depth data from image areas which are uniform.
    # \n Default: 100 (no depth pixel will be rejected)
    # \note Pixels with a value close to 100 are not to be trusted. Accurate depth pixels tends to be closer to lower values.
    @property
    def texture_confidence_threshold(self) -> int:
        return self.runtime.texture_confidence_threshold

    @texture_confidence_threshold.setter
    def texture_confidence_threshold(self, value):
        self.runtime.texture_confidence_threshold = value

    ##
    # Defines if the saturated area (luminance>=255) must be removed from depth map estimation.
    #
    # Default: True
    # \note It is recommended to keep this parameter at True because saturated area can create false detection.
    @property
    def remove_saturated_areas(self) -> bool:
        return self.runtime.remove_saturated_areas

    @remove_saturated_areas.setter
    def remove_saturated_areas(self, value: bool):
        self.runtime.remove_saturated_areas = value
##
# Class containing a set of parameters for the positional tracking module initialization.
# \ingroup PositionalTracking_group
# 
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class PositionalTrackingParameters:
    cdef c_PositionalTrackingParameters* tracking
    ##
    # Default constructor.
    # \param _init_pos : Chosen initial camera position in the world frame (\ref Transform)
    # \param _enable_memory : Activates \ref enable_memory
    # \param _enable_pose_smoothing : Activates \ref enable_pose_smoothing
    # \param _area_path : Chosen \ref area_path
    # \param _set_floor_as_origin : Activates \ref set_floor_as_origin
    # \param _enable_imu_fusion : Activates \ref enable_imu_fusion
    # \param _set_as_static : Activates \ref set_as_static
    # \param _depth_min_range : Activates \ref depth_min_range
    # \param _set_gravity_as_origin : Activates \ref set_gravity_as_origin
    # \param _mode : Chosen \ref mode
    # 
    # \code
    # params = sl.PositionalTrackingParameters(init_pos=sl.Transform(), _enable_pose_smoothing=True)
    # \endcode
    def __cinit__(self, _init_pos=Transform(), _enable_memory=True, _enable_pose_smoothing=False, _area_path=None,
                  _set_floor_as_origin=False, _enable_imu_fusion=True, _set_as_static=False, _depth_min_range=-1,
                  _set_gravity_as_origin=True, _mode=POSITIONAL_TRACKING_MODE.GEN_1) -> PositionalTrackingParameters:
        if _area_path is None:
            self.tracking = new c_PositionalTrackingParameters((<Transform>_init_pos).transform[0], _enable_memory, _enable_pose_smoothing, String(), _set_floor_as_origin, _enable_imu_fusion, _set_as_static, _depth_min_range, _set_gravity_as_origin, <c_POSITIONAL_TRACKING_MODE>(<int>_mode.value))
        else :
            area_path = _area_path.encode()
            self.tracking = new c_PositionalTrackingParameters((<Transform>_init_pos).transform[0], _enable_memory, _enable_pose_smoothing, String(<char*> area_path), _set_floor_as_origin, _enable_imu_fusion, _set_as_static, _depth_min_range, _set_gravity_as_origin, <c_POSITIONAL_TRACKING_MODE>(<int>_mode.value))
    
    def __dealloc__(self):
        del self.tracking

    ##
    # Saves the current set of parameters into a file to be reloaded with the \ref load() method.
    # \param filename : Name of the file which will be created to store the parameters.
    # \return True if the file was successfully saved, otherwise False.
    # \warning For security reasons, the file must not already exist.
    # \warning In case a file already exists, the method will return False and existing file will not be updated.
    def save(self, filename: str) -> bool:
        filename_save = filename.encode()
        return self.tracking.save(String(<char*> filename_save))

    ##
    # Loads a set of parameters from the values contained in a previously \ref save() "saved" file.
    # \param filename : Path to the file from which the parameters will be loaded.
    # \return True if the file was successfully loaded, otherwise False.
    def load(self, filename: str) -> bool:
        filename_load = filename.encode()
        return self.tracking.load(String(<char*> filename_load))

    ##
    # Position of the camera in the world frame when the camera is started.
    # Use this sl.Transform to place the camera frame in the world frame.
    # \n Default: Identity matrix.
    # 
    # \note The camera frame (which defines the reference frame for the camera) is by default positioned at the world frame when tracking is started.
    def initial_world_transform(self, init_pos = Transform()) -> Transform:
        for i in range(16):
            (<Transform>init_pos).transform.m[i] = self.tracking.initial_world_transform.m[i]
        return init_pos

    ##
    # Set the position of the camera in the world frame when the camera is started.
    # \param value : Position of the camera in the world frame when the camera will start.
    def set_initial_world_transform(self, value: Transform) -> None:
        for i in range(16):
            self.tracking.initial_world_transform.m[i] = value.transform.m[i]

    ##
    # Whether the camera can remember its surroundings.
    # This helps correct positional tracking drift and can be helpful for positioning different cameras relative to one other in space.
    # \n Default: true
    #
    # \warning This mode requires more resources to run, but greatly improves tracking accuracy.
    # \warning We recommend leaving it on by default.
    @property
    def enable_area_memory(self) -> bool:
        return self.tracking.enable_area_memory

    @enable_area_memory.setter
    def enable_area_memory(self, value: bool):
        self.tracking.enable_area_memory = value

    ##
    # Whether to enable smooth pose correction for small drift correction.
    # Default: False
    @property
    def enable_pose_smoothing(self) -> bool:
        return self.tracking.enable_pose_smoothing

    @enable_pose_smoothing.setter
    def enable_pose_smoothing(self, value: bool):
        self.tracking.enable_pose_smoothing = value

    ##
    # Initializes the tracking to be aligned with the floor plane to better position the camera in space.
    # Default: False
    # \note This launches floor plane detection in the background until a suitable floor plane is found.
    # \note The tracking will start in [sl.POSITIONAL_TRACKING_STATE.SEARCHING](\ref POSITIONAL_TRACKING_STATE) state.
    # \warning This features does not work with [sl.MODEL.ZED](\ref MODEL) since it needs an IMU to classify the floor.
    # \warning The camera needs to look at the floor during initialization for optimum results.
    @property
    def set_floor_as_origin(self) -> bool:
        return self.tracking.set_floor_as_origin

    @set_floor_as_origin.setter
    def set_floor_as_origin(self, value: bool):
        self.tracking.set_floor_as_origin = value

    ##
    # Whether to enable the IMU fusion.
    # When set to False, only the optical odometry will be used.
    # \n Default: True
    # \note This setting has no impact on the tracking of a camera.
    # \note [sl.MODEL.ZED](\ref MODEL) does not have an IMU.
    @property
    def enable_imu_fusion(self) -> bool:
        return self.tracking.enable_imu_fusion

    @enable_imu_fusion.setter
    def enable_imu_fusion(self, value: bool):
        self.tracking.enable_imu_fusion = value

    ##
    # Path of an area localization file that describes the surroundings (saved from a previous tracking session).
    # Default: (empty)
    # \note Loading an area file will start a search phase, during which the camera will try to position itself in the previously learned area.
    # \warning The area file describes a specific location. If you are using an area file describing a different location, the tracking function will continuously search for a position and may not find a correct one.
    # \warning The '.area' file can only be used with the same depth mode (sl.DEPTH_MODE) as the one used during area recording.
    @property
    def area_file_path(self) -> str:
        if not self.tracking.area_file_path.empty():
            return self.tracking.area_file_path.get().decode()
        else:
            return ""

    @area_file_path.setter
    def area_file_path(self, value: str):
        value_area = value.encode()
        self.tracking.area_file_path.set(<char*>value_area)

    ##
    # Whether to define the camera as static.
    # If true, it will not move in the environment. This allows you to set its position using \ref initial_world_transform.
    # \n All ZED SDK functionalities requiring positional tracking will be enabled without additional computation.
    # \n sl.Camera.get_position() will return the value set as \ref initial_world_transform.
    # Default: False
    @property
    def set_as_static(self) -> bool:
        return self.tracking.set_as_static

    @set_as_static.setter
    def set_as_static(self, value: bool):
        self.tracking.set_as_static = value
    
    ##
    # Minimum depth used by the ZED SDK for positional tracking.
    # It may be useful for example if any steady objects are in front of the camera and may perturb the positional tracking algorithm.
    # \n Default: -1 (no minimum depth)
    @property
    def depth_min_range(self) -> float:
        return self.tracking.depth_min_range

    @depth_min_range.setter
    def depth_min_range(self, value):
        self.tracking.depth_min_range = value

    ##
    # Whether to override 2 of the 3 rotations from \ref initial_world_transform using the IMU gravity.
    # Default: True
    # \note This parameter does nothing on [sl.ZED.MODEL](\ref MODEL) since it does not have an IMU.
    @property
    def set_gravity_as_origin(self) -> bool:
        return self.tracking.set_gravity_as_origin

    @set_gravity_as_origin.setter
    def set_gravity_as_origin(self, value: bool):
        self.tracking.set_gravity_as_origin = value

    ##
    # Positional tracking mode used.
    # Can be used to improve accuracy in some types of scene at the cost of longer runtime.
    # \n Default: [sl.POSITIONAL_TRACKING_MODE.GEN_1](\ref POSITIONAL_TRACKING_MODE)
    @property
    def mode(self) -> POSITIONAL_TRACKING_MODE:
        return POSITIONAL_TRACKING_MODE(<int>self.tracking.mode)

    @mode.setter
    def mode(self, value: POSITIONAL_TRACKING_MODE):
        self.tracking.mode = <c_POSITIONAL_TRACKING_MODE>(<int>value.value)

##
# Lists the different encoding types for image streaming.
# \ingroup Video_group
#
# | Enumerator |                 |
# |------------|-----------------|
# | H264       | AVCHD/H264 encoding |
# | H265       | HEVC/H265 encoding |
class STREAMING_CODEC(enum.Enum):
    H264 = <int>c_STREAMING_CODEC.STREAMING_CODEC_H264
    H265 = <int>c_STREAMING_CODEC.STREAMING_CODEC_H265
    LAST = <int>c_STREAMING_CODEC.STREAMING_CODEC_LAST

##
# Class containing information about the properties of a streaming device. 
# \ingroup Video_group
cdef class StreamingProperties:
    cdef c_StreamingProperties c_streaming_properties

    ##
    # IP address of the streaming device.
    #
    # Default: ""
    @property
    def ip(self) -> str:
        return to_str(self.c_streaming_properties.ip).decode()

    @ip.setter
    def ip(self, str ip_):
        self.c_streaming_properties.ip = String(ip_.encode())

    ##
    # Streaming port of the streaming device.
    #
    # Default: 0
    @property
    def port(self) -> int:
        return self.c_streaming_properties.port

    @port.setter
    def port(self, port_):
         self.c_streaming_properties.port = port_

    ##
    # Serial number of the streaming camera.
    #
    # Default: 0
    @property
    def serial_number(self) -> int:
        return self.c_streaming_properties.serial_number

    @serial_number.setter
    def serial_number(self, serial_number):
        self.c_streaming_properties.serial_number=serial_number

    ##
    # Current bitrate of encoding of the streaming device.
    #
    # Default: 0
    @property
    def current_bitrate(self) -> int:
        return self.c_streaming_properties.current_bitrate

    @current_bitrate.setter
    def current_bitrate(self, current_bitrate):
        self.c_streaming_properties.current_bitrate=current_bitrate

    ##
    # Current codec used for compression in streaming device.
    #
    # Default: \ref STREAMING_CODEC "sl.STREAMING_CODEC.H265"
    @property
    def codec(self) -> STREAMING_CODEC:
        return STREAMING_CODEC(<int>self.c_streaming_properties.codec)

    @codec.setter
    def codec(self, codec):
        self.c_streaming_properties.codec = <c_STREAMING_CODEC>(<int>codec.value)


##
# Class containing the options used to stream with the ZED SDK.
# \ingroup Video_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class StreamingParameters:
    cdef c_StreamingParameters* streaming
    ##
    # Default constructor.
    #
    # All the parameters are set to their default values.
    # \param codec : Chosen \ref codec
    # \param port : Chosen \ref port
    # \param bitrate : Chosen \ref bitrate
    # \param gop_size : Chosen \ref gop_size
    # \param adaptative_bitrate : Activtates \ref adaptative_bitrate
    # \param chunk_size : Chosen \ref chunk_size
    # \param target_framerate : Chosen \ref target_framerate
    #
    # \code
    # params = sl.StreamingParameters(port=30000)
    # \endcode
    def __cinit__(self, codec=STREAMING_CODEC.H265, port=30000, bitrate=0, gop_size=-1, adaptative_bitrate=False, chunk_size=16084,target_framerate=0) -> StreamingParameters:
            self.streaming = new c_StreamingParameters(<c_STREAMING_CODEC>(<int>codec.value), port, bitrate, gop_size, adaptative_bitrate, chunk_size,target_framerate)

    def __dealloc__(self):
        del self.streaming

    ##
    # Size of a single chunk.
    #
    # Default: 16084
    # \note Stream buffers are divided into X number of chunks where each chunk is  \ref chunk_size bytes long.
    # \note You can lower \ref chunk_size value if network generates a lot of packet lost: this will
    # generates more chunk for a single image, but each chunk sent will be lighter to avoid inside-chunk corruption.
    # \note Increasing this value can decrease latency.
    #
    # \n \note Available range: [1024 - 65000]
    @property
    def chunk_size(self) -> int:
        return self.streaming.chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self.streaming.chunk_size = value

    ##
    # Encoding used for streaming.
    @property
    def codec(self) -> STREAMING_CODEC:
        return STREAMING_CODEC(<int>self.streaming.codec)

    @codec.setter
    def codec(self, codec):
        self.streaming.codec = <c_STREAMING_CODEC>(<int>codec.value)

    ##
    # Port used for streaming.
    # \warning Port must be an even number. Any odd number will be rejected.
    # \warning Port must be opened.
    @property
    def port(self) -> int:
        return self.streaming.port

    @port.setter
    def port(self, value: ushort1):
        self.streaming.port = value

    ##
    # Defines the streaming bitrate in Kbits/s
    # | STREAMING_CODEC  | RESOLUTION   | FPS   | Bitrate (kbps) |
    # |------------------|--------------|-------|----------------|
    # | H264             |  HD2K        |   15  |     8500       |
    # | H264             |  HD1080      |   30  |    12500       |
    # | H264             |  HD720       |   60  |     7000       |
    # | H265             |  HD2K        |   15  |     7000       |
    # | H265             |  HD1080      |   30  |    11000       |
    # | H265             |  HD720       |   60  |     6000       |
    #  
    # Default: 0 (it will be set to the best value depending on your resolution/FPS)
    # \note Available range: [1000 - 60000]
    @property
    def bitrate(self) -> int:
        return self.streaming.bitrate

    @bitrate.setter
    def bitrate(self, value: uint):
        self.streaming.bitrate = value

    ##
    # Defines whether the adaptive bitrate is enable.
    #
    # Default: False
    # \note Bitrate will be adjusted depending the number of packet dropped during streaming.
    # \note If activated, the bitrate can vary between [bitrate/4, bitrate].
    # \warning Currently, the adaptive bitrate only works when "sending" device is a NVIDIA Jetson (X1, X2, Xavier, Nano).
    @property
    def adaptative_bitrate(self) -> bool:
        return self.streaming.adaptative_bitrate

    @adaptative_bitrate.setter
    def adaptative_bitrate(self, value: bool):
        self.streaming.adaptative_bitrate = value

    ##
    # GOP size in number of frames.
    # 
    # Default: -1 (the GOP size will last at maximum 2 seconds, depending on camera FPS)
    # \note The GOP size determines the maximum distance between IDR/I-frames. Very high GOP size will result in slightly more efficient compression, especially on static scenes. But latency will increase.
    # \note Maximum value: 256
    @property
    def gop_size(self) -> int:
        return self.streaming.gop_size

    @gop_size.setter
    def gop_size(self, value: int):
        self.streaming.gop_size = value
    
    ##
    # Framerate for the streaming output.
    #
    # Default: 0 (camera framerate will be taken)
    # \warning This framerate must be below or equal to the camera framerate.
    # \warning Allowed framerates are 15, 30, 60 or 100 if possible.
    # \warning Any other values will be discarded and camera FPS will be taken.
    @property
    def target_framerate(self) -> int:
        return self.streaming.target_framerate

    @target_framerate.setter
    def target_framerate(self, value: int):
        self.streaming.target_framerate = value


##
# Class containing the options used to record.
# \ingroup Video_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class RecordingParameters:

    cdef c_RecordingParameters *record
    ##
    # Default constructor.
    #
    # All the parameters are set to their default values.
    # \param video_filename : Chosen \ref video_filename
    # \param compression_mode : Chosen \ref compression_mode
    # \param target_framerate : Chosen \ref target_framerate
    # \param bitrate : Chosen \ref bitrate
    # \param transcode_streaming_input : Enables \ref transcode_streaming_input
    #
    # \code
    # params = sl.RecordingParameters(video_filename="record.svo",compression_mode=SVO_COMPRESSION_MODE.H264)
    # \endcode
    def __cinit__(self, video_filename="myRecording.svo2", compression_mode=SVO_COMPRESSION_MODE.H264, target_framerate=0,
                    bitrate=0, transcode_streaming_input=False) -> RecordingParameters:
        if (isinstance(compression_mode, SVO_COMPRESSION_MODE)) :
            video_filename_c = video_filename.encode()
            self.record = new c_RecordingParameters(String(<char*> video_filename_c), 
                                                    <c_SVO_COMPRESSION_MODE>(<int>compression_mode.value),
                                                    target_framerate, bitrate, transcode_streaming_input)
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.record

    ##
    # Filename of the file to save the recording into.
    @property
    def video_filename(self) -> str:
        return to_str(self.record.video_filename).decode()

    @video_filename.setter
    def video_filename(self, video_filename):
        video_filename_c = video_filename.encode()
        self.record.video_filename = String(<char*> video_filename_c)

    ##
    # Compression mode the recording.
    #
    # Default: \ref SVO_COMPRESSION_MODE "sl.SVO_COMPRESSION_MODE.H264"
    @property
    def compression_mode(self) -> SVO_COMPRESSION_MODE:
        return SVO_COMPRESSION_MODE(<int>self.record.compression_mode)

    @compression_mode.setter
    def compression_mode(self, compression_mode):
        if isinstance(compression_mode, SVO_COMPRESSION_MODE) :
            self.record.compression_mode = <c_SVO_COMPRESSION_MODE>(<int>compression_mode.value)
        else :
            raise TypeError()

    ##
    # Framerate for the recording file.
    #
    # Default: 0 (camera framerate will be taken)
    # \warning This framerate must be below or equal to the camera framerate and camera framerate must be a multiple of the target framerate.
    # \warning It means that it must respect <code> camera_framerate%target_framerate == 0</code>.
    # \warning Allowed framerates are 15,30, 60 or 100 if possible.
    # \warning Any other values will be discarded and camera FPS will be taken.
    @property
    def target_framerate(self) -> int:
        return self.record.target_framerate

    @target_framerate.setter
    def target_framerate(self, value: int):
        self.record.target_framerate = value

    ##
    # Overrides the default bitrate of the SVO file, in kbits/s.
    #
    # Default: 0 (the default values associated with the resolution)
    # \note Only works if \ref compression_mode is H264 or H265.
    # \note Available range: 0 or [1000 - 60000]
    @property
    def bitrate(self) -> int:
        return self.record.bitrate

    @bitrate.setter
    def bitrate(self, value: int):
        self.record.bitrate = value

    ##
    # Defines whether to decode and re-encode a streaming source.
    #
    # Default: False
    # \note If set to False, it will avoid decoding/re-encoding and convert directly streaming input into a SVO file.
    # \note This saves a encoding session and can be especially useful on NVIDIA Geforce cards where the number of encoding session is limited.
    # \note \ref compression_mode, \ref target_framerate and \ref bitrate will be ignored in this mode.
    @property
    def transcode_streaming_input(self) -> bool:
        return self.record.transcode_streaming_input

    @transcode_streaming_input.setter
    def transcode_streaming_input(self, value):
        self.record.transcode_streaming_input = value

##
# Class containing a set of parameters for the spatial mapping module.
# \ingroup SpatialMapping_group
#
# The default constructor sets all parameters to their default settings.
# \note Parameters can be adjusted by the user.
cdef class SpatialMappingParameters:
    cdef c_SpatialMappingParameters* spatial
    ##
    # Default constructor.
    # Sets all parameters to their default and optimized values.
    # \param resolution : Chosen \ref MAPPING_RESOLUTION
    # \param mapping_range : Chosen \ref MAPPING_RANGE
    # \param max_memory_usage : Chosen \ref max_memory_usage
    # \param save_texture : Activates \ref save_texture
    # \param use_chunk_only : Activates \ref use_chunk_only
    # \param reverse_vertex_order : Activates \ref reverse_vertex_order
    # \param map_type : Chosen \ref map_type
    #
    # \code
    # params = sl.SpatialMappingParameters(resolution=sl.MAPPING_RESOLUTION.HIGH)
    # \endcode
    def __cinit__(self, resolution=MAPPING_RESOLUTION.MEDIUM, mapping_range=MAPPING_RANGE.AUTO,
                  max_memory_usage=2048, save_texture=False, use_chunk_only=False,
                  reverse_vertex_order=False, map_type=SPATIAL_MAP_TYPE.MESH) -> SpatialMappingParameters:
        if (isinstance(resolution, MAPPING_RESOLUTION) and isinstance(mapping_range, MAPPING_RANGE) and
            isinstance(use_chunk_only, bool) and isinstance(reverse_vertex_order, bool) and isinstance(map_type, SPATIAL_MAP_TYPE)):
            self.spatial = new c_SpatialMappingParameters(<c_MAPPING_RESOLUTION>(<int>resolution.value),
                                                          <c_MAPPING_RANGE>(<int>mapping_range.value),
                                                          max_memory_usage, save_texture,
                                                          use_chunk_only, reverse_vertex_order,
                                                          <c_SPATIAL_MAP_TYPE>(<int>map_type.value))
        else:
            raise TypeError()

    def __dealloc__(self):
        del self.spatial

    ##
    # Sets the resolution to a sl.MAPPING_RESOLUTION preset.
    # \param resolution: The desired sl.MAPPING_RESOLUTION. Default: [sl.MAPPING_RESOLUTION.HIGH](\ref MAPPING_RESOLUTION)
    def set_resolution(self, resolution=MAPPING_RESOLUTION.HIGH) -> None:
        if isinstance(resolution, MAPPING_RESOLUTION):
            self.spatial.set(<c_MAPPING_RESOLUTION> (<int>resolution.value))
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    ##
    # Sets the range to a sl.MAPPING_RANGE preset.
    # \param mapping_range: The desired [sl.MAPPING_RANGE](\ref MAPPING_RANGE). Default: [sl.MAPPING_RANGE::AUTO](\ref MAPPING_RANGE)
    def set_range(self, mapping_range=MAPPING_RANGE.AUTO) -> None:
        if isinstance(mapping_range, MAPPING_RANGE):
            self.spatial.set(<c_MAPPING_RANGE> (<int>mapping_range.value))
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    ##
    # Returns the value corresponding to a sl.MAPPING_RANGE preset in meters.
    # \param mapping_range: The desired [sl.MAPPING_RANGE](\ref MAPPING_RANGE). Default: [sl.MAPPING_RANGE::AUTO](\ref MAPPING_RANGE)
    # \return The value of \b mapping_range in meters.
    def get_range_preset(self, mapping_range=MAPPING_RANGE.AUTO) -> float:
        if isinstance(mapping_range, MAPPING_RANGE):
            return self.spatial.get(<c_MAPPING_RANGE> (<int>mapping_range.value))
        else:
            raise TypeError("Argument is not of MAPPING_RANGE type.")

    ##
    # Returns the value corresponding to a sl.MAPPING_RESOLUTION preset in meters.
    # \param resolution: The desired sl.MAPPING_RESOLUTION. Default: [sl.MAPPING_RESOLUTION.HIGH](\ref MAPPING_RESOLUTION)
    # \return The value of \b resolution in meters.
    def get_resolution_preset(self, resolution=MAPPING_RESOLUTION.HIGH) -> float:
        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.get(<c_MAPPING_RESOLUTION> (<int>resolution.value))
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION type.")

    ##
    # Returns the recommended maximum depth value corresponding to a resolution.
    # \param resolution : The desired resolution, either defined by a sl.MAPPING_RESOLUTION preset or a resolution value in meters.
    # \param py_cam : The sl.Camera object which will run the spatial mapping.
    # \return The maximum value of depth in meters.
    def get_recommended_range(self, resolution, py_cam: Camera) -> float:
        if not isinstance(py_cam, Camera):
            raise TypeError("Argument is not of Camera type.")
        if py_cam.camera == NULL:
            raise RuntimeError("Camera is not opened.")

        if isinstance(resolution, MAPPING_RESOLUTION):
            return self.spatial.getRecommendedRange(<c_MAPPING_RESOLUTION> (<int>resolution.value), deref(py_cam.camera))
        elif isinstance(resolution, float):
            return self.spatial.getRecommendedRange(<float> resolution, deref(py_cam.camera))
        else:
            raise TypeError("Argument is not of MAPPING_RESOLUTION or float type.")

    ##
    # The type of spatial map to be created.
    # This dictates the format that will be used for the mapping (e.g. mesh, point cloud).
    # \n See [sl.SPATIAL_MAP_TYPE](\ref SPATIAL_MAP_TYPE).
    @property
    def map_type(self) -> SPATIAL_MAP_TYPE:
        return SPATIAL_MAP_TYPE(<int>self.spatial.map_type)

    @map_type.setter
    def map_type(self, value):
        self.spatial.map_type = <c_SPATIAL_MAP_TYPE>(<int>value.value)

    ## 
    # The maximum CPU memory (in MB) allocated for the meshing process.
    # Default: 2048
    @property
    def max_memory_usage(self) -> int:
        return self.spatial.max_memory_usage

    @max_memory_usage.setter
    def max_memory_usage(self, value: int):
        self.spatial.max_memory_usage = value

    ##
    # Whether to save the texture.
    # If set to true, you will be able to apply the texture to your mesh after it is created.
    # \n Default: False
    # \note This option will consume more memory.
    # \note This option is only available for [sl.SPATIAL_MAP_TYPE.MESH](\ref SPATIAL_MAP_TYPE).
    @property
    def save_texture(self) -> bool:
        return self.spatial.save_texture

    @save_texture.setter
    def save_texture(self, value: bool):
        self.spatial.save_texture = value

    ##
    # Whether to only use chunks.
    # If set to False, you will ensure consistency between the mesh and its inner chunk data.
    # \n Default: False
    # \note Updating the mesh is time-consuming.
    # \note Setting this to True results in better performance.
    @property
    def use_chunk_only(self) -> bool:
        return self.spatial.use_chunk_only

    @use_chunk_only.setter
    def use_chunk_only(self, value: bool):
        self.spatial.use_chunk_only = value

    ##
    # Whether to inverse the order of the vertices of the triangles.
    # If your display process does not handle front and back face culling, you can use this to correct it.
    # \n Default: False
    # \note This option is only available for [sl.SPATIAL_MAP_TYPE.MESH](\ref SPATIAL_MAP_TYPE).
    @property
    def reverse_vertex_order(self) -> bool:
        return self.spatial.reverse_vertex_order

    @reverse_vertex_order.setter
    def reverse_vertex_order(self, value: bool):
        self.spatial.reverse_vertex_order = value

    ##
    # The maximum depth allowed by spatial mapping:
    # - \b allowed_range.first is the minimum value allowed
    # - \b allowed_range.second is the maximum value allowed
    @property
    def allowed_range(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_range.first
        arr[1] = self.spatial.allowed_range.second
        return arr

    ##
    # Depth range in meters.
    # Can be different from the value set by sl.InitParameters.depth_maximum_distance.
    # \note Set to 0 by default. In this case, the range is computed from \ref resolution_meter
    # and from the current internal parameters to fit your application.
    @property
    def range_meter(self) -> float:
        return self.spatial.range_meter

    @range_meter.setter
    def range_meter(self, value: float):
        self.spatial.range_meter = value

    ##
    # The resolution allowed by the spatial mapping:
    # - \b allowed_resolution.first is the minimum value allowed
    # - \b allowed_resolution.second is the maximum value allowed
    @property
    def allowed_resolution(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(2)
        arr[0] = self.spatial.allowed_resolution.first
        arr[1] = self.spatial.allowed_resolution.second
        return arr

    ##
    # Spatial mapping resolution in meters.
    # Default: 0.05
    # \note It should fit \ref allowed_resolution.
    @property
    def resolution_meter(self) -> float:
        return self.spatial.resolution_meter

    @resolution_meter.setter
    def resolution_meter(self, value: float):
        self.spatial.resolution_meter = value

    ##
    # Control the integration rate of the current depth into the mapping process.
    # This parameter controls how many times a stable 3D points should be seen before it is integrated into the spatial mapping.
    # \n Default: 0 (this will define the stability counter based on the mesh resolution, the higher the resolution, the higher the stability counter)
    @property
    def stability_counter(self) -> int:
        return self.spatial.stability_counter

    @stability_counter.setter
    def stability_counter(self, value: int):
        self.spatial.stability_counter = value

    ##
    # Saves the current set of parameters into a file to be reloaded with the \ref load() method.
    # \param filename : Name of the file which will be created to store the parameters (extension '.yml' will be added if not set).
    # \return True if the file was successfully saved, otherwise False.
    # \warning For security reasons, the file must not already exist.
    # \warning In case a file already exists, the method will return False and existing file will not be updated.
    def save(self, filename: str) -> bool:
        filename_save = filename.encode()
        return self.spatial.save(String(<char*> filename_save))

    ##
    # Loads a set of parameters from the values contained in a previously \ref save() "saved" file.
    # \param filename : Path to the file from which the parameters will be loaded (extension '.yml' will be added at the end of the filename if not detected).
    # \return True if the file was successfully loaded, otherwise False.
    def load(self, filename: str) -> bool:
        filename_load = filename.encode()
        return self.spatial.load(String(<char*> filename_load))

##
# Class containing positional tracking data giving the position and orientation of the camera in 3D space.
# \ingroup PositionalTracking_group
#
# Different representations of position and orientation can be retrieved, along with timestamp and pose confidence.
cdef class Pose:
    cdef c_Pose pose
    def __cinit__(self):
        self.pose = c_Pose()

    ##
    # Deep copy from another sl.Pose.
    # \param pose : sl.Pose to copy.
    def init_pose(self, pose: Pose) -> None:
        self.pose = c_Pose(pose.pose)

    ##
    # Initializes the sl.Pose from a sl.Transform.
    # \param pose_data : sl.Transform containing pose data to copy.
    # \param timestamp : Timestamp of the pose data.
    # \param confidence : Confidence of the pose data.
    def init_transform(self, pose_data: Transform, timestamp=0, confidence=0) -> None:
        self.pose = c_Pose(pose_data.transform[0], timestamp, confidence)

    ##
    # Returns the sl.Translation corresponding to the current sl.Pose.
    # \param py_translation : sl.Translation to be returned. It creates one by default.
    # \return sl.Translation filled with values from the sl.Pose.
    def get_translation(self, py_translation = Translation()) -> Translation:
        (<Translation>py_translation).translation = self.pose.getTranslation()
        return py_translation

    ##
    # Returns the sl.Orientation corresponding to the current sl.Pose.
    # \param py_orientation : sl.Orientation to be returned. It creates one by default.
    # \return sl.Orientation filled with values from the sl.Pose.
    def get_orientation(self, py_orientation = Orientation()) -> Orientation:
        (<Orientation>py_orientation).orientation = self.pose.getOrientation()
        return py_orientation

    ##
    # Returns the sl.Rotation corresponding to the current sl.Pose.
    # \param py_rotation : sl.Rotation to be returned. It creates one by default.
    # \return sl.Rotation filled with values from the sl.Pose.
    def get_rotation_matrix(self, py_rotation = Rotation()) -> Rotation:
        cdef c_Rotation tmp = self.pose.getRotationMatrix()
        for i in range(9):
            (<Rotation>py_rotation).rotation.r[i] = tmp.r[i]
        return py_rotation

    ##
    # Returns the the 3x1 rotation vector (obtained from 3x3 rotation matrix using Rodrigues formula) corresponding to the current sl.Pose.
    # \param py_rotation : sl.Rotation to be returned. It creates one by default.
    # \return Rotation vector (NumPy array) created from the sl.Pose values.
    def get_rotation_vector(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        for i in range(3):
            arr[i] = self.pose.getRotationVector()[i]
        return arr

    ##
    # Converts the rotation component of the sl.Pose into Euler angles.
    # \param radian : Whether the angle will be returned in radian or degree. Default: True
    # \return Euler angles (Numpy array) created from the sl.Pose values representing the rotations around the X, Y and Z axes using YZX convention.
    def get_euler_angles(self, radian=True) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(3)
        if isinstance(radian, bool):
            for i in range(3):
                arr[i] = self.pose.getEulerAngles(radian)[i]
        else:
            raise TypeError("Argument is not of bool type.")
        return arr

    ##
    # Whether the tracking is activated or not.
    # \note You should check that first if something is wrong.
    @property
    def valid(self) -> bool:
        return self.pose.valid

    @valid.setter
    def valid(self, valid_: bool):
        self.pose.valid = valid_

    ##
    # sl.Timestamp of the sl.Pose.
    # This timestamp should be compared with the camera timestamp for synchronization.
    @property
    def timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp = self.pose.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.pose.timestamp.data_ns = timestamp

    ##
    # sl.Transform containing the rotation and translation data of the sl.Pose.
    # \param pose_data : sl.Transform to be returned. It creates one by default.
    # \return sl.Transform containing the rotation and translation data of the sl.Pose.
    def pose_data(self, pose_data = Transform()) -> Transform:
        for i in range(16):
            (<Transform>pose_data).transform.m[i] = self.pose.pose_data.m[i]
        return pose_data

    ##
    # Confidence/quality of the pose estimation for the target frame.
    # A confidence metric of the tracking [0-100] with:
    # - 0: tracking is lost
    # - 100: tracking can be fully trusted
    @property
    def pose_confidence(self) -> int:
        return self.pose.pose_confidence

    @pose_confidence.setter
    def pose_confidence(self, pose_confidence_: int):
        self.pose.pose_confidence = pose_confidence_

    ##
    # 6x6 pose covariance matrix (NumPy array) of translation (the first 3 values) and rotation in so3 (the last 3 values).
    # \note Computed only if \ref PositionalTrackingParameters.enable_spatial_memory is disabled.
    @property
    def pose_covariance(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.pose.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.pose.pose_covariance[i] = pose_covariance_[i]
    
    ##
    # Twist of the camera available in reference camera.
    # This expresses velocity in free space, broken into its linear and angular parts.
    @property
    def twist(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(6)
        for i in range(6):
            arr[i] = self.pose.twist[i]
        return arr

    @twist.setter
    def twist(self, np.ndarray twist_):
        for i in range(6):
            self.pose.twist[i] = twist_[i]

    ##
    # Row-major representation of the 6x6 twist covariance matrix of the camera.
    # This expresses the uncertainty of the twist.
    @property
    def twist_covariance(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36):
            arr[i] = self.pose.twist_covariance[i]
        return arr

    @twist_covariance.setter
    def twist_covariance(self, np.ndarray twist_covariance_):
        for i in range(36):
            self.pose.twist_covariance[i] = twist_covariance_[i]

##
# Lists different states of the camera motion.
# \ingroup Sensors_group
#
# | Enumerator |                  |
# |------------|------------------|
# | STATIC     | The camera is static. |
# | MOVING     | The camera is moving. |
# | FALLING    | The camera is falling. |
class CAMERA_MOTION_STATE(enum.Enum):
    STATIC = <int>c_CAMERA_MOTION_STATE.STATIC
    MOVING = <int>c_CAMERA_MOTION_STATE.MOVING
    FALLING = <int>c_CAMERA_MOTION_STATE.FALLING
    LAST = <int>c_CAMERA_MOTION_STATE.CAMERA_MOTION_STATE_LAST

##
# Lists possible locations of temperature sensors.
# \ingroup Sensors_group
#
# | Enumerator |                  |
# |------------|------------------|
# | IMU        | The temperature sensor is in the IMU. |
# | BAROMETER  | The temperature sensor is in the barometer. |
# | ONBOARD_LEFT | The temperature sensor is next to the left image sensor. |
# | ONBOARD_RIGHT | The temperature sensor is next to the right image sensor. |
class SENSOR_LOCATION(enum.Enum):
    IMU = <int>c_SENSOR_LOCATION.IMU
    BAROMETER = <int>c_SENSOR_LOCATION.BAROMETER
    ONBOARD_LEFT = <int>c_SENSOR_LOCATION.ONBOARD_LEFT
    ONBOARD_RIGHT = <int>c_SENSOR_LOCATION.ONBOARD_RIGHT
    LAST = <int>c_SENSOR_LOCATION.SENSOR_LOCATION_LAST

##
# Class containing data from the barometer sensor.
# \ingroup Sensors_group
cdef class BarometerData:
    cdef c_BarometerData barometerData

    def __cinit__(self):
        self.barometerData = c_BarometerData()

    ##
    # Whether the barometer sensor is available in your camera.
    @property
    def is_available(self) -> bool:
        return self.barometerData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.barometerData.is_available = is_available

    ##
    # Ambient air pressure in hectopascal (hPa).
    @property
    def pressure(self) -> float:
        return self.barometerData.pressure

    @pressure.setter
    def pressure(self, pressure: float):
        self.barometerData.pressure=pressure

    ##
    # Relative altitude from first camera position (at sl.Camera.open() time).
    @property
    def relative_altitude(self) -> float:
        return self.barometerData.relative_altitude

    @relative_altitude.setter
    def relative_altitude(self, alt: float):
        self.barometerData.relative_altitude = alt

    ##
    # Data acquisition timestamp.
    @property
    def timestamp(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp = self.barometerData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.barometerData.timestamp.data_ns = timestamp

    ##
    # Realtime data acquisition rate in hertz (Hz).
    @property
    def effective_rate(self) -> float:
        return self.barometerData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.barometerData.effective_rate = rate

##
# Class containing data from the temperature sensors.
# \ingroup Sensors_group
cdef class TemperatureData:
    cdef c_TemperatureData temperatureData

    def __cinit__(self):
        self.temperatureData = c_TemperatureData()

    ##
    # Gets the temperature value at a temperature sensor location.
    # \param location : Location of the temperature sensor to request.
    # \return Temperature at the requested location.
    def get(self, location) -> float:
        cdef float value
        value = 0
        if isinstance(location,SENSOR_LOCATION):
            err = _error_code_cache.get(<int>self.temperatureData.get(<c_SENSOR_LOCATION>(<int>(location.value)), value), ERROR_CODE.FAILURE)
            if err == ERROR_CODE.SUCCESS :
                return value
            else :
                return -1
        else:
            raise TypeError("Argument not of type SENSOR_LOCATION")


##
# Lists the different states of the magnetic heading.
# \ingroup Sensors_group
#
# | Enumerator |                  |
# |------------|------------------|
# | GOOD       | The heading is reliable and not affected by iron interferences. |
# | OK         | The heading is reliable, but affected by slight iron interferences. |
# | NOT_GOOD   | The heading is not reliable because affected by strong iron interferences. |
# | NOT_CALIBRATED | The magnetometer has not been calibrated. |
# | MAG_NOT_AVAILABLE | The magnetometer sensor is not available. |
class HEADING_STATE(enum.Enum):
    GOOD = <int>c_HEADING_STATE.GOOD
    OK = <int>c_HEADING_STATE.OK
    NOT_GOOD = <int>c_HEADING_STATE.NOT_GOOD
    NOT_CALIBRATED = <int>c_HEADING_STATE.NOT_CALIBRATED
    MAG_NOT_AVAILABLE = <int>c_HEADING_STATE.MAG_NOT_AVAILABLE
    HEADING_STATE_LAST = <int>c_HEADING_STATE.HEADING_STATE_LAST

##
# Class containing data from the magnetometer sensor.
# \ingroup Sensors_group
cdef class MagnetometerData:
    cdef c_MagnetometerData magnetometerData

    def __cinit__(self):
        self.magnetometerData

    ##
    # Whether the magnetometer sensor is available in your camera.
    @property
    def is_available(self) -> bool:
        return self.magnetometerData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.magnetometerData.is_available = is_available

    ##
    # Realtime data acquisition rate in hertz (Hz).
    @property
    def effective_rate(self) -> float:
        return self.magnetometerData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.magnetometerData.effective_rate = rate

    ##
    # Camera heading in degrees relative to the magnetic North Pole.
    # \note The magnetic North Pole has an offset with respect to the geographic North Pole, depending on the geographic position of the camera.
    # \note To get a correct magnetic heading, the magnetometer sensor must be calibrated using \b ZED \b Sensor \b Viewer tool.
    @property
    def magnetic_heading(self) -> float:
        return self.magnetometerData.magnetic_heading

    @magnetic_heading.setter
    def magnetic_heading(self, heading: float):
        self.magnetometerData.magnetic_heading = heading

    ##
    # Accuracy of \ref magnetic_heading measure in the range [0.0, 1.0].
    # \note A negative value means that the magnetometer must be calibrated using \b ZED \b Sensor \b Viewer tool.
    @property
    def magnetic_heading_accuracy(self) -> float:
        return self.magnetometerData.magnetic_heading_accuracy

    @magnetic_heading_accuracy.setter
    def magnetic_heading_accuracy(self, accuracy: float):
        self.magnetometerData.magnetic_heading_accuracy = accuracy

    ##
    # State of \ref magnetic_heading.
    @property
    def magnetic_heading_state(self) -> HEADING_STATE:
        return HEADING_STATE(<int>self.magnetometerData.magnetic_heading_state)

    @magnetic_heading_state.setter
    def magnetic_heading_state(self, state):
        if isinstance(state, HEADING_STATE):
            self.magnetometerData.magnetic_heading_state = <c_HEADING_STATE>(<int>state.value)
        else:
            raise TypeError("Argument is not of HEADING_STATE type.")

    ##
    # Gets the uncalibrated magnetic field local vector in microtesla (μT).
    # \note The magnetometer raw values are affected by soft and hard iron interferences.
    # \note The sensor must be calibrated by placing the camera in the working environment and using \b ZED \b Sensor \b Viewer tool.
    # \note Not available in SVO or STREAM mode.
    def get_magnetic_field_uncalibrated(self) -> np.array[float]:
        cdef np.ndarray magnetic_field = np.zeros(3)
        for i in range(3):
            magnetic_field[i] = self.magnetometerData.magnetic_field_uncalibrated[i]
        return magnetic_field

    ##
    # Gets the magnetic field local vector in microtesla (μT).
    # \note To calibrate the magnetometer sensor, please use \b ZED \b Sensor \b Viewer tool after placing the camera in the final operating environment.
    def get_magnetic_field_calibrated(self) -> np.array[float]:
        cdef np.ndarray magnetic_field = np.zeros(3)
        for i in range(3):
            magnetic_field[i] = self.magnetometerData.magnetic_field_calibrated[i]
        return magnetic_field

    ##
    # Data acquisition timestamp.
    @property
    def timestamp(self) -> int:
        ts = Timestamp()
        ts.timestamp = self.magnetometerData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.magnetometerData.timestamp.data_ns = timestamp


##
# Class containing all sensors data (except image sensors) to be used for positional tracking or environment study.
# \ingroup Sensors_group
#
# \note Some data are not available in SVO and streaming input mode.
# \note They are specified by a note "Not available in SVO or STREAM mode." in the documentation of a specific data.
# \note If nothing is mentioned in the documentation, they are available in all input modes.
cdef class SensorsData:
    cdef c_SensorsData sensorsData

    def __cinit__(self):
        self.sensorsData = c_SensorsData()

    ##
    # Copy constructor.
    # \param sensorsData : sl.SensorsData object to copy.
    def init_sensorsData(self, sensorsData: SensorsData) -> None:
        self.sensorsData = sensorsData.sensorsData

    ##
    # Motion state of the camera.
    @property
    def camera_moving_state(self) -> CAMERA_MOTION_STATE:
        return CAMERA_MOTION_STATE(<int>self.sensorsData.camera_moving_state)

    @camera_moving_state.setter
    def camera_moving_state(self, state):
        if isinstance(state, CAMERA_MOTION_STATE):
            self.sensorsData.camera_moving_state = <c_CAMERA_MOTION_STATE>(<int>(state.value))
        else:
            raise TypeError("Argument not of type CAMERA_MOTION_STATE")

    ##
    # Indicates if the sensors data has been taken during a frame capture on sensor.
    # If the value is 1, the data has been retrieved during a left sensor frame acquisition (the time precision is linked to the IMU rate, therefore 800Hz == 1.3ms).
    # \n If the value is 0, the data has not been taken during a frame acquisition.
    @property
    def image_sync_trigger(self) -> int:
        return self.sensorsData.image_sync_trigger

    @image_sync_trigger.setter
    def image_sync_trigger(self, image_sync_trigger: int):
        self.sensorsData.image_sync_trigger = image_sync_trigger


    ##
    # Gets the IMU data.
    # \return sl.IMUData containing the IMU data.
    def get_imu_data(self) -> IMUData:
        imu_data = IMUData()
        imu_data.imuData = self.sensorsData.imu
        return imu_data

    ##
    # Gets the barometer data.
    # \return sl.BarometerData containing the barometer data.
    def get_barometer_data(self) -> BarometerData:
        barometer_data = BarometerData()
        barometer_data.barometerData = self.sensorsData.barometer
        return barometer_data

    ##
    # Gets the magnetometer data.
    # \return sl.MagnetometerData containing the magnetometer data.
    def get_magnetometer_data(self) -> MagnetometerData:
        magnetometer_data = MagnetometerData()
        magnetometer_data.magnetometerData = self.sensorsData.magnetometer
        return magnetometer_data

    ##
    # Gets the temperature data.
    # \return sl.TemperatureData containing the temperature data.
    def get_temperature_data(self) -> TemperatureData:
        temperature_data = TemperatureData()
        temperature_data.temperatureData = self.sensorsData.temperature
        return temperature_data


##
# Class containing data from the IMU sensor.
# \ingroup Sensors_group
cdef class IMUData:
    cdef c_IMUData imuData

    def __cinit__(self):
        self.imuData = c_IMUData()
    
    ##
    # Gets the angular velocity vector (3x1) of the gyroscope in deg/s (uncorrected from the IMU calibration).
    # \param angular_velocity_uncalibrated : List to be returned. It creates one by default.
    # \return List fill with the raw angular velocity vector.
    # \note The value is the exact raw values from the IMU.
    # \note Not available in SVO or STREAM mode.
    def get_angular_velocity_uncalibrated(self, angular_velocity_uncalibrated = [0, 0, 0]) -> list[float]:
        for i in range(3):
            angular_velocity_uncalibrated[i] = self.imuData.angular_velocity_uncalibrated[i]
        return angular_velocity_uncalibrated    
        
    ##
    # Gets the angular velocity vector (3x1) of the gyroscope in deg/s.
    # The value is corrected from bias, scale and misalignment.
    # \param angular_velocity : List to be returned. It creates one by default.
    # \return List fill with the angular velocity vector.
    # \note The value can be directly ingested in an IMU fusion algorithm to extract a quaternion.
    # \note Not available in SVO or STREAM mode.
    def get_angular_velocity(self, angular_velocity = [0, 0, 0]) -> list[float]:
        for i in range(3):
            angular_velocity[i] = self.imuData.angular_velocity[i]
        return angular_velocity

    ##
    # Gets the linear acceleration vector (3x1) of the gyroscope in m/s².
    # The value is corrected from bias, scale and misalignment.
    # \param linear_acceleration : List to be returned. It creates one by default.
    # \return List fill with the linear acceleration vector.
    # \note The value can be directly ingested in an IMU fusion algorithm to extract a quaternion.
    # \note Not available in SVO or STREAM mode.
    def get_linear_acceleration(self, linear_acceleration = [0, 0, 0]) -> list[float]:
        for i in range(3):
            linear_acceleration[i] = self.imuData.linear_acceleration[i]
        return linear_acceleration

    ##
    # Gets the linear acceleration vector (3x1) of the gyroscope in m/s² (uncorrected from the IMU calibration).
    # The value is corrected from bias, scale and misalignment.
    # \param linear_acceleration_uncalibrated : List to be returned. It creates one by default.
    # \return List fill with the raw linear acceleration vector.
    # \note The value is the exact raw values from the IMU.
    # \note Not available in SVO or STREAM mode.
    def get_linear_acceleration_uncalibrated(self, linear_acceleration_uncalibrated = [0, 0, 0]) -> list[float]:
        for i in range(3):
            linear_acceleration_uncalibrated[i] = self.imuData.linear_acceleration_uncalibrated[i]
        return linear_acceleration_uncalibrated

    ##
    # Gets the covariance matrix of the angular velocity of the gyroscope in deg/s (\ref get_angular_velocity()).
    # \param angular_velocity_covariance : sl.Matrix3f to be returned. It creates one by default.
    # \return sl.Matrix3f filled with the covariance matrix of the angular velocity.
    # \note Not available in SVO or STREAM mode.
    def get_angular_velocity_covariance(self, angular_velocity_covariance = Matrix3f()) -> Matrix3f:
        for i in range(9):
            (<Matrix3f>angular_velocity_covariance).mat.r[i] = self.imuData.angular_velocity_covariance.r[i]
        return angular_velocity_covariance
        
    ##
    # Gets the covariance matrix of the linear acceleration of the gyroscope in deg/s (\ref get_angular_velocity()).
    # \param linear_acceleration_covariance : sl.Matrix3f to be returned. It creates one by default.
    # \return sl.Matrix3f filled with the covariance matrix of the linear acceleration.
    # \note Not available in SVO or STREAM mode.
    def get_linear_acceleration_covariance(self, linear_acceleration_covariance = Matrix3f()) -> Matrix3f:
        for i in range(9):
            (<Matrix3f>linear_acceleration_covariance).mat.r[i] = self.imuData.linear_acceleration_covariance.r[i]
        return linear_acceleration_covariance

    ##
    # Whether the IMU sensor is available in your camera.
    @property
    def is_available(self) -> bool:
        return self.imuData.is_available

    @is_available.setter
    def is_available(self, is_available: bool):
        self.imuData.is_available = is_available

    ##
    # Data acquisition timestamp.
    @property
    def timestamp(self) -> int:
        ts = Timestamp()
        ts.timestamp = self.imuData.timestamp
        return ts

    @timestamp.setter
    def timestamp(self, unsigned long long timestamp):
        self.imuData.timestamp.data_ns = timestamp

    ##
    # Realtime data acquisition rate in hertz (Hz).
    @property
    def effective_rate(self) -> float:
        return self.imuData.effective_rate

    @effective_rate.setter
    def effective_rate(self, rate: float):
        self.imuData.effective_rate = rate

    ##
    # Covariance matrix of the IMU pose (\ref get_pose()).
    # \param pose_covariance : sl.Matrix3f to be returned. It creates one by default.
    # \return sl.Matrix3f filled with the covariance matrix.
    def get_pose_covariance(self, pose_covariance = Matrix3f()) -> Matrix3f:
        for i in range(9):
            (<Matrix3f>pose_covariance).mat.r[i] = self.imuData.pose_covariance.r[i]
        return pose_covariance


    ##
    # IMU pose (IMU 6-DoF fusion).
    # \param pose : sl.Transform() to be returned. It creates one by default.
    # \return sl.Transform filled with the IMU pose.
    def get_pose(self, pose = Transform()) -> Transform:
        for i in range(16):
            (<Transform>pose).transform.m[i] = self.imuData.pose.m[i]
        return pose

##
# Structure containing the self diagnostic results of the image/depth
# That information can be retrieved by sl::Camera::get_health_status(), and enabled by sl::InitParameters::enable_image_validity_check 
# \n
# The default value of sl::InitParameters::enable_image_validity_check is enabled using the fastest setting, 
# the integer given can be increased to include more advanced and heavier processing to detect issues (up to 3).
# \ingroup Video_group
cdef class HealthStatus:
    cdef c_HealthStatus healthStatus

    ##
    # \brief Indicates if the Health check is enabled
    @property
    def enabled(self) -> bool:
        return self.healthStatus.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.healthStatus.enabled = value
    
    ##
    # \brief This status indicates poor image quality
    # It can indicates camera issue, like incorrect manual video settings, damaged hardware, corrupted video stream from the camera, 
    # dirt or other partial or total occlusion, stuck ISP (black/white/green/purple images, incorrect exposure, etc), blurry images
    # It also includes widely different left and right images which leads to unavailable depth information
    # In case of very low light this will be reported by this status and the dedicated \ref HealthStatus::low_lighting
    # 
    # \note: Frame tearing is currently not detected. Advanced blur detection requires heavier processing and is enabled only when setting \ref Initparameters::enable_image_validity_check to 3 and above
    @property
    def low_image_quality(self) -> bool:
        return self.healthStatus.low_image_quality

    @low_image_quality.setter
    def low_image_quality(self, value: bool):
        self.healthStatus.low_image_quality = value

    ##
    # \brief This status indicates low light scene.
    # As the camera are passive sensors working in the visible range, they requires some external light to operate.
    # This status warns if the lighting condition become suboptimal and worst.
    # This is based on the scene illuminance in LUX for the ZED X cameras series (available with \ref VIDEO_SETTINGS::SCENE_ILLUMINANCE)
    # For other camera models or when using SVO files, this is based on computer vision processing from the image characteristics.
    @property
    def low_lighting(self) -> bool:
        return self.healthStatus.low_lighting

    @low_lighting.setter
    def low_lighting(self, value: bool):
        self.healthStatus.low_lighting = value

    ##
    # \brief This status indicates low depth map reliability
    # If the image are unreliable or if the scene condition are very challenging this status report a warning.
    # This is using the depth confidence and general depth distribution. Typically due to obstructed eye (included very close object, 
    # strong occlusions) or degraded condition like heavy fog/water on the optics
    @property
    def low_depth_reliability(self) -> bool:
        return self.healthStatus.low_depth_reliability

    @low_depth_reliability.setter
    def low_depth_reliability(self, value: bool):
        self.healthStatus.low_depth_reliability = value

    ##
    # \brief This status indicates motion sensors data reliability issue.
    # This indicates the IMU is providing low quality data. Possible underlying can be regarding the data stream like corrupted data, 
    # timestamp inconsistency, resonance frequencies, saturated sensors / very high acceleration or rotation, shocks
    @property
    def low_motion_sensors_reliability(self) -> bool:
        return self.healthStatus.low_motion_sensors_reliability

    @low_motion_sensors_reliability.setter
    def low_motion_sensors_reliability(self, value: bool):
        self.healthStatus.low_motion_sensors_reliability = value


##
# Class containing information about the status of the recording.
# \ingroup Video_group
cdef class RecordingStatus:
    cdef c_RecordingStatus recordingState

    ##
    # Report if the recording has been enabled.
    @property
    def is_recording(self) -> bool:
        return self.recordingState.is_recording

    @is_recording.setter
    def is_recording(self, value: bool):
        self.recordingState.is_recording = value

    ##
    # Report if the recording has been paused.
    @property
    def is_paused(self) -> bool:
        return self.recordingState.is_recording

    @is_paused.setter
    def is_paused(self, value: bool):
        self.recordingState.is_paused = value

    ##
    # Status of current frame.
    #
    # True for success or False if the frame could not be written in the SVO file.
    @property
    def status(self) -> bool:
        return self.recordingState.status

    @status.setter
    def status(self, value: bool):
        self.recordingState.status = value

    ##
    # Compression time for the current frame in milliseconds.
    @property
    def current_compression_time(self) -> float:
        return self.recordingState.current_compression_time

    @current_compression_time.setter
    def current_compression_time(self, value: double):
        self.recordingState.current_compression_time = value

    ##
    # Compression ratio (% of raw size) for the current frame.
    @property
    def current_compression_ratio(self) -> float:
        return self.recordingState.current_compression_ratio

    @current_compression_ratio.setter
    def current_compression_ratio(self, value: double):
        self.recordingState.current_compression_ratio = value

    ##
    # Average compression time in milliseconds since beginning of recording.
    @property
    def average_compression_time(self) -> float:
        return self.recordingState.average_compression_time

    @average_compression_time.setter
    def average_compression_time(self, value: double):
        self.recordingState.average_compression_time = value

    ##
    # Average compression ratio (% of raw size) since beginning of recording.
    @property
    def average_compression_ratio(self) -> float:
        return self.recordingState.average_compression_ratio

    @average_compression_ratio.setter
    def average_compression_ratio(self, value: double):
        self.recordingState.average_compression_ratio = value

    ##
    # Number of frames ingested in SVO encoding/writing.
    @property
    def number_frames_ingested(self) -> int:
        return self.recordingState.number_frames_ingested

    @number_frames_ingested.setter
    def number_frames_ingested(self, value: int):
        self.recordingState.number_frames_ingested = value

    ##
    # Number of frames effectively encoded and written. Might be different from the number of frames ingested. The difference will show the encoder latency
    @property
    def number_frames_encoded(self) -> int:
        return self.recordingState.number_frames_encoded

    @number_frames_encoded.setter
    def number_frames_encoded(self, value: int):
        self.recordingState.number_frames_encoded = value


##
# This class serves as the primary interface between the camera and the various features provided by the SDK.
# It enables seamless integration and access to a wide array of capabilities, including video streaming, depth sensing, object tracking, mapping, and much more.  
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
#            init_params.camera_resolution = sl.RESOLUTION.HD720    # Use HD720 video mode for USB cameras
#            # init_params.camera_resolution = sl.RESOLUTION.HD1200 # Use HD1200 video mode for GMSL cameras
#            init_params.camera_fps = 60                            # Set fps at 60
#
#            # Open the camera
#            err = zed.open(init_params)
#            if err != sl.ERROR_CODE.SUCCESS:
#                print(repr(err))
#                exit(-1)
#
#            runtime_param = sl.RuntimeParameters()
#
#            # --- Main loop grabbing images and depth values
#            # Capture 50 frames and stop
#            i = 0
#            image = sl.Mat()
#            depth = sl.Mat()
#            while i < 50 :
#                # Grab an image
#                if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
#                    # Display a pixel color
#                    zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
#                    err, center_rgb = image.get_value(image.get_width() / 2, image.get_height() / 2)
#                    if err == sl.ERROR_CODE.SUCCESS:
#                        print("Image ", i, " center pixel R:", int(center_rgb[0]), " G:", int(center_rgb[1]), " B:", int(center_rgb[2]))
#                    else:
#                        print("Image ", i, " error:", err)
#
#                    # Display a pixel depth
#                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Get the depth map
#                    err, center_depth = depth.get_value(depth.get_width() / 2, depth.get_height() /2)
#                    if err == sl.ERROR_CODE.SUCCESS:
#                        print("Image ", i," center depth:", center_depth)
#                    else:
#                        print("Image ", i, " error:", err)
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
    cdef c_Camera* camera

    def __cinit__(self):
        self.camera = new c_Camera()

    def __dealloc__(self):
        if self.camera != NULL:
            del self.camera

    ##
    # Close an opened camera.
    #
    # If \ref open() has been called, this method will close the connection to the camera (or the SVO file) and free the corresponding memory.
    #
    # If \ref open() wasn't called or failed, this method won't have any effect.
    #
    # \note If an asynchronous task is running within the \ref Camera object, like \ref save_area_map(), this method will wait for its completion.
    # \note To apply a new \ref InitParameters, you will need to close the camera first and then open it again with the new InitParameters values.
    # \warning If the CUDA context was created by \ref open(), this method will destroy it.
    # \warning Therefore you need to make sure to delete your GPU \ref sl.Mat objects before the context is destroyed.
    def close(self) -> None:
        self.camera.close()

    ##
    # Opens the ZED camera from the provided InitParameters.
    # The method will also check the hardware requirements and run a self-calibration.
    # \param py_init : A structure containing all the initial parameters. Default: a preset of InitParameters.
    # \return An error code giving information about the internal process. If \ref ERROR_CODE "ERROR_CODE.SUCCESS" is returned, the camera is ready to use. Every other code indicates an error and the program should be stopped.
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
    # \note If you are having issues opening a camera, the diagnostic tool provided in the SDK can help you identify to problems.
    #   - <b>Windows:</b> <i>C:\\Program Files (x86)\\ZED SDK\\tools\\ZED Diagnostic.exe</i>
    #   - <b>Linux:</b> <i>/usr/local/zed/tools/ZED Diagnostic</i>
    # \note If this method is called on an already opened camera, \ref close() will be called.
    def open(self, InitParameters py_init = None) -> ERROR_CODE:
        if py_init is None:
            py_init = InitParameters()
        return _error_code_cache.get(<int>self.camera.open(deref((<InitParameters>py_init).init)), ERROR_CODE.FAILURE)

    ##
    # Reports if the camera has been successfully opened.
    # It has the same behavior as checking if \ref open() returns \ref ERROR_CODE "ERROR_CODE.SUCCESS".
    # \return True if the ZED camera is already setup, otherwise false.
    def is_opened(self) -> bool:
        return self.camera.isOpened()

    ##
    # \brief Read the latest images and IMU from the camera and rectify the images.
    #
    # This method is meant to be called frequently in the main loop of your application.
    # 
    # \note If no new frames is available until timeout is reached, read() will return \ref ERROR_CODE "ERROR_CODE::CAMERA_NOT_DETECTED" since the camera has probably been disconnected.
    # \note Returned errors can be displayed using toString().
    # 
    # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" means that no problem was encountered.    
    def read(self) -> ERROR_CODE:
        cdef c_ERROR_CODE err
        with nogil:
            err = self.camera.read()
        return _error_code_cache.get(<int>err, ERROR_CODE.FAILURE)

    ##
    # This method will grab the latest images from the camera, rectify them, and compute the \ref retrieve_measure() "measurements" based on the \ref RuntimeParameters provided (depth, point cloud, tracking, etc.)
    #
    # As measures are created in this method, its execution can last a few milliseconds, depending on your parameters and your hardware.
    # \n The exact duration will mostly depend on the following parameters:
    # 
    #   - \ref InitParameters.enable_right_side_measure : Activating this parameter increases computation time.
    #   - \ref InitParameters.camera_resolution : Lower resolutions are faster to compute.
    #   - \ref enable_positional_tracking() : Activating the tracking is an additional load.
    #   - \ref RuntimeParameters.enable_depth : Avoiding the depth computation must be faster. However, it is required by most SDK features (tracking, spatial mapping, plane estimation, etc.)
    #   - \ref InitParameters.depth_mode : \ref DEPTH_MODE "DEPTH_MODE.PERFORMANCE" will run faster than \ref DEPTH_MODE "DEPTH_MODE.ULTRA".
    #   - \ref InitParameters.depth_stabilization : Stabilizing the depth requires an additional computation load as it enables tracking.
    #
    # This method is meant to be called frequently in the main loop of your application.
    # \note Since ZED SDK 3.0, this method is blocking. It means that grab() will wait until a new frame is detected and available.
    # \note If no new frames is available until timeout is reached, grab() will return \ref ERROR_CODE "ERROR_CODE.CAMERA_NOT_DETECTED" since the camera has probably been disconnected.
    # 
    # \param py_runtime : A structure containing all the runtime parameters. Default: a preset of \ref RuntimeParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" means that no problem was encountered.
    # \note Returned errors can be displayed using <code>str()</code>.
    #
    # \code
    # # Set runtime parameters after opening the camera
    # runtime_param = sl.RuntimeParameters()
    #
    # image = sl.Mat()
    # while True:
    #       # Grab an image
    #       if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
    #           zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image     
    #           # Use the image for your application
    # \endcode
    def grab(self, RuntimeParameters py_runtime = None) -> ERROR_CODE:
        if py_runtime is None:
            py_runtime = RuntimeParameters()
        cdef c_ERROR_CODE err
        with nogil:
            err = self.camera.grab(deref(py_runtime.runtime))
        return _error_code_cache.get(<int>err, ERROR_CODE.FAILURE)

    ##
    # Retrieves images from the camera (or SVO file).
    #
    # Multiple images are available along with a view of various measures for display purposes.
    # \n Available images and views are listed \ref VIEW "here".
    # \n As an example, \ref VIEW "VIEW.DEPTH" can be used to get a gray-scale version of the depth map, but the actual depth values can be retrieved using \ref retrieve_measure() .
    # \n
    # \n <b>Pixels</b>
    # \n Most VIEW modes output image with 4 channels as BGRA (Blue, Green, Red, Alpha), for more information see enum \ref VIEW
    # \n
    # \n <b>Memory</b>
    # \n By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
    # \n If your application can use GPU images, using the <b>type</b> parameter can increase performance by avoiding this copy.
    # \n If the provided sl.Mat object is already allocated  and matches the requested image format, memory won't be re-allocated.
    # \n
    # \n <b>Image size</b>
    # \n By default, images are returned in the resolution provided by \ref Resolution "get_camera_information().camera_configuration.resolution".
    # \n However, you can request custom resolutions. For example, requesting a smaller image can help you speed up your application.
    # \warning A sl.Mat resolution higher than the camera resolution <b>cannot</b> be requested.
    # 
    # \param py_mat[out] : The \ref sl.Mat to store the image.
    # \param view[in] : Defines the image you want (see \ref VIEW). Default: \ref VIEW "VIEW.LEFT".
    # \param mem_type[in] : Defines on which memory the image should be allocated. Default: \ref MEM "MEM.CPU" (you cannot change this default value).
    # \param resolution[in] : If specified, defines the \ref Resolution of the output sl.Mat. If set to \ref Resolution "Resolution(0,0)", the camera resolution will be taken. Default: (0,0).
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the method succeeded.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" if the view mode requires a module not enabled (\ref VIEW "VIEW.DEPTH" with \ref DEPTH_MODE "DEPTH_MODE.NONE" for example).
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_RESOLUTION" if the resolution is higher than one provided by \ref Resolution "get_camera_information().camera_configuration.resolution".
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if another error occurred.
    # 
    # \note As this method retrieves the images grabbed by the \ref grab() method, it should be called afterward.
    #
    # \code
    # # create sl.Mat objects to store the images
    # left_image = sl.Mat()
    # while True:
    #       # Grab an image
    #       if zed.grab() == sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
    #           zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
    #
    #           # Display the center pixel colors
    #           err, left_center = left_image.get_value(left_image.get_width() / 2, left_image.get_height() / 2)
    #           if err == sl.ERROR_CODE.SUCCESS:
    #               print("left_image center pixel R:", int(left_center[0]), " G:", int(left_center[1]), " B:", int(left_center[2]))
    #           else:
    #               print("error:", err)
    # \endcode
    def retrieve_image(self, Mat py_mat, view: VIEW = VIEW.LEFT, mem_type: MEM = MEM.CPU, Resolution resolution = None) -> ERROR_CODE:
        if resolution is None:
            resolution = Resolution(0,0)
        cdef c_ERROR_CODE err
        cdef c_VIEW c_view = <c_VIEW>(<int>view.value)
        cdef c_MEM c_type = <c_MEM>(<int>mem_type.value)
        with nogil:
            err = self.camera.retrieveImage(py_mat.mat, c_view, c_type, resolution.resolution)
        return _error_code_cache.get(<int>err, ERROR_CODE.FAILURE)

    ##
    # Computed measures, like depth, point cloud, or normals, can be retrieved using this method.
    # 
    # Multiple measures are available after a \ref grab() call. A full list is available \ref MEASURE "here".
    #
    # \n <b>Memory</b>
    # \n By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
    # \n If your application can use GPU images, using the \b type parameter can increase performance by avoiding this copy.
    # \n If the provided \ref Mat object is already allocated and matches the requested image format, memory won't be re-allocated.
    # 
    # \n <b>Measure size</b>
    # \n By default, measures are returned in the resolution provided by \ref get_camera_information() in \ref CameraInformations.camera_resolution .
    # \n However, custom resolutions can be requested. For example, requesting a smaller measure can help you speed up your application.
    # \warning A sl.Mat resolution higher than the camera resolution <b>cannot</b> be requested.
    #
    # \param py_mat[out] : The sl.Mat to store the measures.
    # \param measure[in] : Defines the measure you want (see \ref MEASURE). Default: \ref MEASURE "MEASURE.DEPTH".
    # \param mem_type[in] : Defines on which memory the image should be allocated. Default: \ref MEM "MEM.CPU" (you cannot change this default value).
    # \param resolution[in] : If specified, defines the \ref Resolution of the output sl.Mat. If set to \ref Resolution "Resolution(0,0)", the camera resolution will be taken. Default: (0,0).
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the method succeeded.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" if the view mode requires a module not enabled (\ref VIEW "VIEW.DEPTH" with \ref DEPTH_MODE "DEPTH_MODE.NONE" for example).
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_RESOLUTION" if the resolution is higher than one provided by \ref Resolution "get_camera_information().camera_configuration.resolution".
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if another error occured.
    #
    # \note As this method retrieves the images grabbed by the \ref grab() method, it should be called afterward.
    #
    # \code
    # depth_map = sl.Mat()
    # point_cloud = sl.Mat()
    # resolution = zed.get_camera_information().camera_configuration.resolution
    # x = int(resolution.width / 2) # Center coordinates
    # y = int(resolution.height / 2)
    #
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS: # Grab an image
    #
    #         zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Get the depth map
    #
    #         # Read a depth value
    #         err, center_depth = depth_map.get_value(x, y) # each depth map pixel is a float value
    #         if err == sl.ERROR_CODE.SUCCESS: # + Inf is "too far", -Inf is "too close", Nan is "unknown/occlusion"
    #             print("Depth value at center:", center_depth, init_params.coordinate_units)
    #         zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Get the point cloud
    #
    #         # Read a point cloud value
    #         err, pc_value = point_cloud.get_value(x, y) # each point cloud pixel contains 4 floats, so we are using a numpy array
    #         
    #         # Get 3D coordinates
    #         if err == sl.ERROR_CODE.SUCCESS:
    #             print("Point cloud coordinates at center: X=", pc_value[0], ", Y=", pc_value[1], ", Z=", pc_value[2])
    #         
    #        # Get color information using Python struct package to unpack the unsigned char array containing RGBA values
    #        import struct
    #        packed = struct.pack('f', pc_value[3])
    #        char_array = struct.unpack('BBBB', packed)
    #        print("Color values at center: R=", char_array[0], ", G=", char_array[1], ", B=", char_array[2], ", A=", char_array[3])
    #     
    # \endcode
    def retrieve_measure(self, Mat py_mat, measure: MEASURE = MEASURE.DEPTH, mem_type: MEM = MEM.CPU, Resolution resolution = None) -> ERROR_CODE:
        if resolution is None:
            resolution = Resolution(0, 0)
        cdef c_ERROR_CODE err
        cdef c_MEASURE c_measure = <c_MEASURE>(<int>measure.value)
        cdef c_MEM c_mem = <c_MEM>(<int>mem_type.value)
        with nogil:
            err = self.camera.retrieveMeasure(py_mat.mat, c_measure, c_mem, resolution.resolution)
        return _error_code_cache.get(<int>err, ERROR_CODE.FAILURE)

    ##
    # Defines a region of interest to focus on for all the SDK, discarding other parts.
    # \param roi_mask : The \ref Mat defining the requested region of interest, pixels lower than 127 will be discarded from all modules: depth, positional tracking, etc.
    # If empty, set all pixels as valid. The mask can be either at lower or higher resolution than the current images.
    # \return An ERROR_CODE if something went wrong.
    # \note The method support \ref MAT_TYPE "U8_C1/U8_C3/U8_C4" images type.
    def set_region_of_interest(self, Mat py_mat, list modules = [MODULE.ALL]) -> ERROR_CODE:
        cdef unordered_set[c_MODULE] modules_set
        for v in modules:
            modules_set.insert(<c_MODULE>(<int>v.value))
        return _error_code_cache.get(<int>self.camera.setRegionOfInterest(py_mat.mat, modules_set), ERROR_CODE.FAILURE)

    ##
    # Get the previously set or computed region of interest
    # \param roi_mask: The \ref Mat returned
    # \param image_size: The optional size of the returned mask
    # \return An \ref ERROR_CODE if something went wrong.
    def get_region_of_interest(self, Mat py_mat, Resolution resolution = None, module: MODULE = MODULE.ALL) -> ERROR_CODE:
        if resolution is None:
            resolution = Resolution(0, 0)
        return _error_code_cache.get(
            <int>self.camera.getRegionOfInterest(py_mat.mat, (<Resolution>resolution).resolution, (<c_MODULE>(<int>module.value))),
            ERROR_CODE.FAILURE)

    ##
    # Start the auto detection of a region of interest to focus on for all the SDK, discarding other parts.
    # This detection is based on the general motion of the camera combined with the motion in the scene. 
    # The camera must move for this process, an internal motion detector is used, based on the Positional Tracking module. 
    # It requires a few hundreds frames of motion to compute the mask.
    # \param roi_param: The \ref RegionOfInterestParameters defining parameters for the detection
    # 
    # \note This module is expecting a static portion, typically a fairly close vehicle hood at the bottom of the image.
    # This module may not work correctly or detect incorrect background area, especially with slow motion, if there's no static element.
    # This module work asynchronously, the status can be obtained using \ref get_region_of_interest_auto_detection_status(), the result is either auto applied, 
    # or can be retrieve using \ref get_region_of_interest function.
    # \return An \ref ERROR_CODE if something went wrong.
    def start_region_of_interest_auto_detection(self, RegionOfInterestParameters roi_param = None) -> ERROR_CODE:
        if roi_param is None:
            roi_param = RegionOfInterestParameters()
        return _error_code_cache.get(<int>self.camera.startRegionOfInterestAutoDetection(deref((<RegionOfInterestParameters>roi_param).roi_params)), ERROR_CODE.FAILURE)

    ##
    # Return the status of the automatic Region of Interest Detection
    # The automatic Region of Interest Detection is enabled by using \ref startRegionOfInterestAutoDetection
    # \return \ref REGION_OF_INTEREST_AUTO_DETECTION_STATE the status
    def get_region_of_interest_auto_detection_status(self) -> REGION_OF_INTEREST_AUTO_DETECTION_STATE:
        return REGION_OF_INTEREST_AUTO_DETECTION_STATE(<int>self.camera.getRegionOfInterestAutoDetectionStatus())

    ##
    # Set this camera as a data provider for the Fusion module. 
    # 
    # Metadata is exchanged with the Fusion.
    # \param communication_parameters : A structure containing all the initial parameters. Default: a preset of CommunicationParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    def start_publishing(self, CommunicationParameters communication_parameters) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.startPublishing(communication_parameters.communicationParameters), ERROR_CODE.FAILURE)

    ##
    # Set this camera as normal camera (without data providing).
    # 
    # Stop to send camera data to fusion.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    def stop_publishing(self) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.stopPublishing(), ERROR_CODE.FAILURE)

    ##
    # Sets the playback cursor to the desired frame number in the SVO file.
    #
    # This method allows you to move around within a played-back SVO file. After calling, the next call to \ref grab() will read the provided frame number.
    #
    # \param frame_number : The number of the desired frame to be decoded.
    # 
    # \note The method works only if the camera is open in SVO playback mode.
    #
    # \code
    #
    # import pyzed.sl as sl
    #
    # def main():
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #
    #     # Set configuration parameters
    #     init_params = sl.InitParameters()
    #     init_params.set_from_svo_file("path/to/my/file.svo")
    #
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Loop between frames 0 and 50
    #     left_image = sl.Mat()
    #     while zed.get_svo_position() < zed.get_svo_number_of_frames() - 1:
    #
    #         print("Current frame: ", zed.get_svo_position())
    #
    #         # Loop if we reached frame 50
    #         if zed.get_svo_position() == 50:
    #             zed.set_svo_position(0)
    #
    #         # Grab an image
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #             zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
    #
    #         # Use the image in your application
    #
    #     # Close the Camera
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    #
    # \endcode
    def set_svo_position(self, int frame_number) -> None:
        self.camera.setSVOPosition(frame_number)

    ##
    # Pauses or resumes SVO reading when using SVO Real time mode
    #  \param status : If true, the reading is paused. If false, the reading is resumed.
    #  \note This is only relevant for SVO \ref InitParameters::svo_real_time_mode
    def pause_svo_reading(self, bool status) -> None:
        self.camera.pauseSVOReading(status)

    ##
    # Returns the current playback position in the SVO file.
    #
    # The position corresponds to the number of frames already read from the SVO file, starting from 0 to n.
    # 
    # Each \ref grab() call increases this value by one (except when using \ref InitParameters.svo_real_time_mode).
    # \return The current frame position in the SVO file. -1 if the SDK is not reading an SVO.
    # 
    # \note The method works only if the camera is open in SVO playback mode.
    #
    # See \ref set_svo_position() for an example.
    def get_svo_position(self) -> int:
        return self.camera.getSVOPosition()

    ##
    # Returns the number of frames in the SVO file.
    #
    # \return The total number of frames in the SVO file. -1 if the SDK is not reading a SVO.
    #
    # The method works only if the camera is open in SVO playback mode.
    def get_svo_number_of_frames(self) -> int:
        return self.camera.getSVONumberOfFrames()

    ##
    # ingest a SVOData in the SVO file.
    #
    # \return An error code stating the success, or not.
    #
    # The method works only if the camera is open in SVO recording mode.
    def ingest_data_into_svo(self, SVOData data) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.ingestDataIntoSVO(data.svo_data), ERROR_CODE.FAILURE)

    ##
    # Get the external channels that can be retrieved from the SVO file.
    #
    # \return a list of keys
    #
    # The method works only if the camera is open in SVO playback mode.
    def get_svo_data_keys(self) -> list:
        vect_ = self.camera.getSVODataKeys()
        vect_python = []
        for i in range(vect_.size()):
            vect_python.append(vect_[i].decode())

        return vect_python

    ##
    # retrieve SVO datas from the SVO file at the given channel key and in the given timestamp range.
    #
    # \return An error code stating the success, or not.
    # \param key : The channel key.
    # \param data : The dict to be filled with SVOData objects, with timestamps as keys.
    # \param ts_begin : The beginning of the range.
    # \param ts_end : The end of the range.
    #
    # The method works only if the camera is open in SVO playback mode.
    def retrieve_svo_data(self, str key, dict data, Timestamp ts_begin, Timestamp ts_end) -> ERROR_CODE:
        cdef map[c_Timestamp, c_SVOData] data_c
        cdef map[c_Timestamp, c_SVOData].iterator it

        res = _error_code_cache.get(<int>self.camera.retrieveSVOData(key.encode('utf-8'), data_c, ts_begin.timestamp, ts_end.timestamp), ERROR_CODE.FAILURE)
        it = data_c.begin()

        while(it != data_c.end()):
            ts = Timestamp()
            ts.timestamp = deref(it).first
            content_c = SVOData()
            content_c.svo_data = deref(it).second
            data[ts] = content_c

            postincrement(it) # Increment the iterator to the net element

        return res

    # Sets the value of the requested \ref VIDEO_SETTINGS "camera setting" (gain, brightness, hue, exposure, etc.).
    #
    # This method only applies for \ref VIDEO_SETTINGS that require a single value.
    #
    # Possible values (range) of each settings are available \ref VIDEO_SETTINGS "here".
    #
    # \param settings : The setting to be set.
    # \param value : The value to set. Default: auto mode
    # \return \ref ERROR_CODE to indicate if the method was successful.
    #
    # \warning Setting [VIDEO_SETTINGS.EXPOSURE](\ref VIDEO_SETTINGS) or [VIDEO_SETTINGS.GAIN](\ref VIDEO_SETTINGS) to default will automatically sets the other to default.
    #
    # \note The method works only if the camera is open in LIVE or STREAM mode.
    #
    # \code
    # # Set the gain to 50
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
    # \endcode
    def set_camera_settings(self, settings: VIDEO_SETTINGS, int value=-1) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), value), ERROR_CODE.FAILURE)

    ##
    # Sets the value of the requested \ref VIDEO_SETTINGS "camera setting" that supports two values (min/max).
    #
    # This method only works with the following \ref VIDEO_SETTINGS:
    # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE"
    # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE"
    # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE"
    #
    # \param settings : The setting to be set.
    # \param min : The minimum value that can be reached (-1 or 0 gives full range).
    # \param max : The maximum value that can be reached (-1 or 0 gives full range).
    # \return \ref ERROR_CODE to indicate if the method was successful.
    #
    # \warning If \ref VIDEO_SETTINGS settings is not supported or min >= max, it will return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS".
    # \note The method works only if the camera is open in LIVE or STREAM mode.
    #
    # \code
    # # For ZED X based product, set the automatic exposure from 2ms to 5ms. Expected exposure time cannot go beyond those values
    # zed.set_camera_settings_range(sl.VIDEO_SETTINGS.AEC_RANGE, 2000, 5000);
    # \endcode
    def set_camera_settings_range(self, settings: VIDEO_SETTINGS, int mini=-1, int maxi=-1) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), mini, maxi), ERROR_CODE.FAILURE)

    ##
    # Overloaded method for \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI" which takes a Rect as parameter.
    #
    # \param settings : Must be set at \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI", otherwise the method will have no impact.
    # \param roi : Rect that defines the target to be applied for AEC/AGC computation. Must be given according to camera resolution.
    # \param eye : \ref SIDE on which to be applied for AEC/AGC computation. Default: \ref SIDE "SIDE.BOTH"
    # \param reset : Cancel the manual ROI and reset it to the full image. Default: False
    # 
    # \note The method works only if the camera is open in LIVE or STREAM mode.
    # 
    # \code
    #   roi = sl.Rect(42, 56, 120, 15)
    #   zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
    # \endcode
    #
    def set_camera_settings_roi(self, settings: VIDEO_SETTINGS, Rect roi, eye: SIDE = SIDE.BOTH, bool reset = False) -> ERROR_CODE:
        return _error_code_cache.get(
            <int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), roi.rect, <c_SIDE>(<int>eye.value), reset),
            ERROR_CODE.FAILURE)

    ##
    # Returns the current value of the requested \ref VIDEO_SETTINGS "camera setting" (gain, brightness, hue, exposure, etc.).
    # 
    # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
    # 
    # \param setting : The requested setting.
    # \return \ref ERROR_CODE to indicate if the method was successful.
    # \return The current value for the corresponding setting.
    #
    # \code
    # err, gain = zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
    # if err == sl.ERROR_CODE.SUCCESS:
    #       print("Current gain value:", gain)
    # else:
    #       print("error:", err)
    # \endcode
    #
    # \note The method works only if the camera is open in LIVE or STREAM mode.
    # \note Settings are not exported in the SVO file format.
    def get_camera_settings(self, setting: VIDEO_SETTINGS) -> tuple[ERROR_CODE, int]:
        cdef int value = 0
        error_code = _error_code_cache.get(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), value), ERROR_CODE.FAILURE)
        return error_code, value

    ##
    # Returns the values of the requested \ref VIDEO_SETTINGS "settings" for \ref VIDEO_SETTINGS that supports two values (min/max).
    # 
    # This method only works with the following VIDEO_SETTINGS:
    #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE"
    #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE"
    #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE"
    # 
    # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
    # \param setting : The requested setting.
    # \return \ref ERROR_CODE to indicate if the method was successful.
    # \return The current value of the minimum for the corresponding setting.
    # \return The current value of the maximum for the corresponding setting.
    #
    # \code
    # err, aec_range_min, aec_range_max = zed.get_camera_settings(sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE)
    # if err == sl.ERROR_CODE.SUCCESS:
    #       print("Current AUTO_EXPOSURE_TIME_RANGE range values ==> min:", aec_range_min, "max:", aec_range_max)
    # else:
    #       print("error:", err)
    # \endcode
    #
    # \note Works only with ZED X that supports low-level controls
    def get_camera_settings_range(self, setting: VIDEO_SETTINGS) -> tuple[ERROR_CODE, int, int]:
        cdef int mini = 0
        cdef int maxi = 0
        error_code = _error_code_cache.get(
            <int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), <int&>mini, <int&>maxi),
            ERROR_CODE.FAILURE)
        return error_code, mini, maxi

    ##
    # Returns the current value of the currently used ROI for the camera setting \ref VIDEO_SETTINGS "AEC_AGC_ROI".
    # 
    # \param setting[in] : Must be set at \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI", otherwise the method will have no impact.
    # \param roi[out] : Roi that will be filled.
    # \param eye[in] : The requested side. Default: \ref SIDE "SIDE.BOTH"
    # \return \ref ERROR_CODE to indicate if the method was successful.
    #
    # \code
    # roi = sl.Rect()
    # err = zed.get_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
    # print("Current ROI for AEC_AGC: " + str(roi.x) + " " + str(roi.y)+ " " + str(roi.width) + " " + str(roi.height))
    # \endcode
    #
    # \note Works only if the camera is open in LIVE or STREAM mode with \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI".
    # \note It will return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" or \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" otherwise.
    def get_camera_settings_roi(self, setting: VIDEO_SETTINGS, Rect roi, eye: SIDE = SIDE.BOTH) -> ERROR_CODE:
        return _error_code_cache.get(
            <int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), roi.rect, <c_SIDE>(<int>eye.value)),
            ERROR_CODE.FAILURE)

    ##
    # Returns if the video setting is supported by the camera or not
    #
    # \param setting[in] : the video setting to test
    # \return True if the \ref VIDEO_SETTINGS is supported by the camera, False otherwise
    #
    def is_camera_setting_supported(self, setting: VIDEO_SETTINGS) -> bool:
        if not isinstance(setting, VIDEO_SETTINGS):
            raise TypeError("Argument is not of VIDEO_SETTINGS type.")

        return self.camera.isCameraSettingSupported(<c_VIDEO_SETTINGS>(<int>setting.value))

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
    def get_current_fps(self) -> float:
        return self.camera.getCurrentFPS()

    ##
    # Returns the timestamp in the requested \ref TIME_REFERENCE.
    #
    # - When requesting the \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" timestamp, the UNIX nanosecond timestamp of the latest \ref grab() "grabbed" image will be returned.
    # \n This value corresponds to the time at which the entire image was available in the PC memory. As such, it ignores the communication time that corresponds to 2 or 3 frame-time based on the fps (ex: 33.3ms to 50ms at 60fps).
    #
    # - When requesting the [TIME_REFERENCE.CURRENT](\ref TIME_REFERENCE) timestamp, the current UNIX nanosecond timestamp is returned.
    #
    # This function can also be used when playing back an SVO file.
    #
    # \param time_reference : The selected \ref TIME_REFERENCE.
    # \return The \ref Timestamp in nanosecond. 0 if not available (SVO file without compression).
    #
    # \note As this function returns UNIX timestamps, the reference it uses is common across several \ref Camera instances.
    # \n This can help to organized the grabbed images in a multi-camera application.
    # 
    # \code
    # last_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
    # current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
    # print("Latest image timestamp: ", last_image_timestamp.get_nanoseconds(), "ns from Epoch.")
    # print("Current timestamp: ", current_timestamp.get_nanoseconds(), "ns from Epoch.")
    # \endcode 
    def get_timestamp(self, time_reference: TIME_REFERENCE) -> Timestamp:
        ts = Timestamp()
        cdef c_TIME_REFERENCE c_time_reference = <c_TIME_REFERENCE>(<int>time_reference.value)
        with nogil:
            ts.timestamp = self.camera.getTimestamp(c_time_reference)
        return ts

    ##
    # Returns the number of frames dropped since \ref grab() was called for the first time.
    #
    # A dropped frame corresponds to a frame that never made it to the grab method.
    # \n This can happen if two frames were extracted from the camera when grab() is called. The older frame will be dropped so as to always use the latest (which minimizes latency).
    #
    # \return The number of frames dropped since the first \ref grab() call.
    def get_frame_dropped_count(self) -> int:
        return self.camera.getFrameDroppedCount()


    ##
    # Gets the current range of perceived depth.
    # \param min[out] : Minimum depth detected (in selected sl.UNIT).
    # \param max[out] : Maximum depth detected (in selected sl.UNIT).
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if values can be extracted, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    def get_current_min_max_depth(self) -> tuple(ERROR_CODE, float, float):
        cdef float mini = 0
        cdef float maxi = 0
        error_code = _error_code_cache.get(<int>self.camera.getCurrentMinMaxDepth(<float&>mini, <float&>maxi), ERROR_CODE.FAILURE)
        return error_code, mini, maxi

    ##
    # Returns the CameraInformation associated the camera being used.
    #
    # To ensure accurate calibration, it is possible to specify a custom resolution as a parameter when obtaining scaled information, as calibration parameters are resolution-dependent.
    # \n When reading an SVO file, the parameters will correspond to the camera used for recording.
    # 
    # \param resizer : You can specify a size different from the default image size to get the scaled camera information.
    # Default = (0,0) meaning original image size (given by \ref CameraConfiguration.resolution "get_camera_information().camera_configuration.resolution").
    # \return \ref CameraInformation containing the calibration parameters of the ZED, as well as serial number and firmware version.
    #
    # \warning The returned parameters might vary between two execution due to the \ref InitParameters.camera_disable_self_calib "self-calibration" being run in the \ref open() method.
    # \note The calibration file SNXXXX.conf can be found in:
    # - <b>Windows:</b> <i>C:/ProgramData/Stereolabs/settings/</i>
    # - <b>Linux:</b> <i>/usr/local/zed/settings/</i>
    def get_camera_information(self, Resolution resizer = None) -> CameraInformation:
        if resizer is None:
            resizer = Resolution(0, 0)
        return CameraInformation(self, resizer)

    ##
    # Returns the RuntimeParameters used.
    # It corresponds to the structure given as argument to the \ref grab() method.
    #
    # \return \ref RuntimeParameters containing the parameters that define the behavior of the \ref grab method.
    def get_runtime_parameters(self) -> RuntimeParameters:
        runtime = RuntimeParameters()
        runtime.runtime.measure3D_reference_frame = self.camera.getRuntimeParameters().measure3D_reference_frame
        runtime.runtime.enable_depth = self.camera.getRuntimeParameters().enable_depth
        runtime.runtime.confidence_threshold = self.camera.getRuntimeParameters().confidence_threshold
        runtime.runtime.texture_confidence_threshold = self.camera.getRuntimeParameters().texture_confidence_threshold
        runtime.runtime.remove_saturated_areas = self.camera.getRuntimeParameters().remove_saturated_areas
        return runtime

    ##
    # Returns the InitParameters associated with the Camera object.
    # It corresponds to the structure given as argument to \ref open() method.
    #
    # \return InitParameters containing the parameters used to initialize the Camera object.
    def get_init_parameters(self) -> InitParameters:
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
        init.init.enable_image_validity_check = self.camera.getInitParameters().enable_image_validity_check
        init.init.maximum_working_resolution = self.camera.getInitParameters().maximum_working_resolution
        return init

    ##
    # Returns the PositionalTrackingParameters used.
    # 
    # It corresponds to the structure given as argument to the \ref enable_positional_tracking() method.
    #
    # \return \ref PositionalTrackingParameters containing the parameters used for positional tracking initialization.
    def get_positional_tracking_parameters(self) -> PositionalTrackingParameters:
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
        tracking.tracking.mode = self.camera.getPositionalTrackingParameters().mode
        return tracking

    ## 
    # Returns the SpatialMappingParameters used.
    #
    # It corresponds to the structure given as argument to the enable_spatial_mapping() method.
    #
    # \return \ref SpatialMappingParameters containing the parameters used for spatial mapping initialization.
    def get_spatial_mapping_parameters(self) -> SpatialMappingParameters:
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
    # Returns the ObjectDetectionParameters used.
    #
    # It corresponds to the structure given as argument to the enable_object_detection() method.
    # \return \ref ObjectDetectionParameters containing the parameters used for object detection initialization.
    def get_object_detection_parameters(self, unsigned int instance_module_id=0) -> ObjectDetectionParameters:
        object_detection = ObjectDetectionParameters()
        object_detection.object_detection.enable_tracking = self.camera.getObjectDetectionParameters(instance_module_id).enable_tracking
        object_detection.object_detection.max_range = self.camera.getObjectDetectionParameters(instance_module_id).max_range
        object_detection.object_detection.prediction_timeout_s = self.camera.getObjectDetectionParameters(instance_module_id).prediction_timeout_s
        object_detection.object_detection.instance_module_id = instance_module_id
        object_detection.object_detection.enable_segmentation = self.camera.getObjectDetectionParameters(instance_module_id).enable_segmentation
        return object_detection

    ##
    # Returns the BodyTrackingParameters used.
    #
    # It corresponds to the structure given as argument to the enable_body_tracking() method.
    #
    # \return \ref BodyTrackingParameters containing the parameters used for body tracking initialization.
    def get_body_tracking_parameters(self, unsigned int instance_id = 0) -> BodyTrackingParameters:
        body_params = BodyTrackingParameters()
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
    # Returns the StreamingParameters used.
    #
    #  It corresponds to the structure given as argument to the enable_streaming() method.
    # 
    # \return \ref StreamingParameters containing the parameters used for streaming initialization.
    def get_streaming_parameters(self) -> StreamingParameters:
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
    # This method allows you to enable the position estimation of the SDK. It only has to be called once in the camera's lifetime.
    # \n When enabled, the \ref get_position "position" will be update at each grab() call.
    # \n Tracking-specific parameters can be set by providing \ref PositionalTrackingParameters to this method.
    #
    # \param py_tracking : A structure containing all the specific parameters for the positional tracking. Default: a preset of \ref PositionalTrackingParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if the \ref PositionalTrackingParameters.area_file_path file wasn't found, \ref ERROR_CODE "ERROR_CODE.SUCCESS" otherwise.
    #
    # \warning The positional tracking feature benefits from a high framerate. We found HD720@60fps to be the best compromise between image quality and framerate.
    #
    # \code
    #
    # import pyzed.sl as sl
    # 
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
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Set tracking parameters
    #     track_params = sl.PositionalTrackingParameters()
    #
    #     # Enable positional tracking
    #     err = zed.enable_positional_tracking(track_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Tracking error: ", repr(err))
    #         exit(-1)
    #
    #     # --- Main loop
    #     while True:
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS: # Grab an image and computes the tracking
    #             camera_pose = sl.Pose()
    #             zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
    #             translation = camera_pose.get_translation().get()
    #             print("Camera position: X=", translation[0], " Y=", translation[1], " Z=", translation[2])
    #
    #     # --- Close the Camera
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    #
    # \endcode
    def enable_positional_tracking(self, PositionalTrackingParameters py_tracking = None) -> ERROR_CODE:
        if py_tracking is None:
            py_tracking = PositionalTrackingParameters()

        return _error_code_cache.get(
            <int>self.camera.enablePositionalTracking(deref((<PositionalTrackingParameters>py_tracking).tracking)),
            ERROR_CODE.FAILURE)

    ##
    # Performs a new self-calibration process.
    # In some cases, due to temperature changes or strong vibrations, the stereo calibration becomes less accurate.
    # \n Use this method to update the self-calibration data and get more reliable depth values.
    # \note The self-calibration will occur at the next \ref grab() call.
    # \note This method is similar to the previous reset_self_calibration() used in 2.X SDK versions.
    # \warning New values will then be available in \ref get_camera_information(), be sure to get them to still have consistent 2D <-> 3D conversion.
    def update_self_calibration(self) -> None:
        self.camera.updateSelfCalibration()

    ##
    # Initializes and starts the body tracking module.
    #
    # The body tracking module currently supports multiple classes of human skeleton detection with the \ref BODY_TRACKING_MODEL "BODY_TRACKING_MODEL.HUMAN_BODY_FAST",
    # \ref BODY_TRACKING_MODEL "BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM" or \ref BODY_TRACKING_MODEL "BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE".
    # \n This model only detects humans but provides a full skeleton map for each person.
    #
    # \n Detected objects can be retrieved using the \ref retrieve_bodies() method.
    #
    # \note - <b>This Deep Learning detection module is not available for \ref MODEL "MODEL.ZED" cameras (first generation ZED cameras).</b>
    # \note - This feature uses AI to locate objects and requires a powerful GPU. A GPU with at least 3GB of memory is recommended.
    #
    # \param body_tracking_parameters : A structure containing all the specific parameters for the object detection. Default: a preset of BodyTrackingParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine.
    # \return \ref ERROR_CODE "ERROR_CODE.OBJECT_DETECTION_NOT_AVAILABLE" if the AI model is missing or corrupted. In this case, the SDK needs to be reinstalled
    # \return \ref ERROR_CODE "ERROR_CODE.OBJECT_DETECTION_MODULE_NOT_COMPATIBLE_WITH_CAMERA" if the camera used does not have an IMU (\ref MODEL "MODEL.ZED").
    # \return \ref ERROR_CODE "ERROR_CODE.SENSORS_NOT_DETECTED" if the camera model is correct (not \ref MODEL "MODEL.ZED") but the IMU is missing. It probably happens because \ref InitParameters.sensors_required was set to False and that IMU has not been found.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" if one of the <b>body_tracking_parameters</b> parameter is not compatible with other modules parameters (for example, <b>depth_mode</b> has been set to \ref DEPTH_MODE "DEPTH_MODE.NONE").
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \code
    # import pyzed.sl as sl
    # 
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    # 
    #     # Open the camera
    #     err = zed.open()
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Opening camera error:", repr(err))
    #         exit(-1)
    #
    #     # Enable position tracking (mandatory for object detection)
    #     tracking_params = sl.PositionalTrackingParameters()
    #     err = zed.enable_positional_tracking(tracking_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Enabling Positional Tracking error:", repr(err))
    #         exit(-1)
    #
    #     # Set the body tracking parameters
    #     body_tracking_params = sl.BodyTrackingParameters()
    #
    #     # Enable the body tracking
    #     err = zed.enable_body_tracking(body_tracking_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Enabling Body Tracking error:", repr(err))
    #         exit(-1)
    # 
    #     # Grab an image and detect bodies on it
    #     bodies = sl.Bodies()
    #     while True :
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #             zed.retrieve_bodies(bodies)
    #             print(len(bodies.body_list), "bodies detected")
    #             # Use the bodies in your application
    # 
    #     # Close the camera
    #     zed.disable_body_tracking()
    #     zed.close()
    #
    # if __name__ == "__main__":
    #     main()
    # \endcode
    def enable_body_tracking(self, BodyTrackingParameters body_tracking_parameters = None) -> ERROR_CODE:
        if body_tracking_parameters is None:
            body_tracking_parameters = BodyTrackingParameters()
        return _error_code_cache.get(
            <int>self.camera.enableBodyTracking(deref(body_tracking_parameters.bodyTrackingParameters)),
            ERROR_CODE.FAILURE)

    ##
    # Disables the body tracking process.
    #
    # The body tracking module immediately stops and frees its memory allocations.
    #
    # \param instance_id : Id of the body tracking instance. Used when multiple instances of the body tracking module are enabled at the same time.
    # \param force_disable_all_instances : Should disable all instances of the body tracking module or just <b>instance_module_id</b>.
    #
    # \note If the body tracking has been enabled, this method will automatically be called by \ref close().
    def disable_body_tracking(self, unsigned int instance_id = 0, bool force_disable_all_instances = False) -> None:
        return self.camera.disableBodyTracking(instance_id, force_disable_all_instances)

    ##
    # Retrieves body tracking data from the body tracking module.
    #
    # This method returns the result of the body tracking, whether the module is running synchronously or asynchronously.
    #
    # - <b>Asynchronous:</b> this method immediately returns the last bodies tracked. If the current tracking isn't done, the bodies from the last tracking will be returned, and \ref Bodies.is_new will be set to False.
    # - <b>Synchronous:</b> this method executes tracking and waits for it to finish before returning the detected objects.
    #
    # It is recommended to keep the same \ref Bodies object as the input of all calls to this method. This will enable the identification and the tracking of every detected object.
    #
    # \param bodies : The detected bodies will be saved into this object. If the object already contains data from a previous tracking, it will be updated, keeping a unique ID for the same person.
    # \param body_tracking_runtime_parameters : Body tracking runtime settings, can be changed at each tracking. In async mode, the parameters update is applied on the next iteration. If None, the previously used parameters will be used.
    # \param instance_id : Id of the body tracking instance. Used when multiple instances of the body tracking module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \code
    # bodies = sl.Bodies() # Unique Bodies to be updated after each grab
    # # Main loop
    # while True:
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS: # Grab an image from the camera
    #         zed.retrieve_bodies(bodies)
    #         print(len(bodies.body_list), "bodies detected")
    # \endcode
    def retrieve_bodies(self, Bodies bodies, BodyTrackingRuntimeParameters body_tracking_runtime_parameters = None, unsigned int instance_id = 0) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            if body_tracking_runtime_parameters is None:
                ret = self.camera.retrieveBodies(bodies.bodies, instance_id)
            else:
                ret = self.camera.retrieveBodiesAndSetRuntimeParameters(bodies.bodies, deref(body_tracking_runtime_parameters.body_tracking_rt), instance_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Set the body tracking runtime parameters
    #
    def set_body_tracking_runtime_parameters(self, BodyTrackingRuntimeParameters body_tracking_runtime_parameters, unsigned int instance_module_id=0) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            ret = self.camera.setBodyTrackingRuntimeParameters(deref(body_tracking_runtime_parameters.body_tracking_rt), instance_module_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Tells if the body tracking module is enabled.
    def is_body_tracking_enabled(self, unsigned int instance_id = 0) -> bool:
        return self.camera.isBodyTrackingEnabled(instance_id)

    ##
    # Retrieves the SensorsData (IMU, magnetometer, barometer) at a specific time reference.
    # 
    # - Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" gives you the latest sensors data received. Getting all the data requires to call this method at 800Hz in a thread.
    # - Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" gives you the sensors data at the time of the latest image \ref grab() "grabbed".
    #
    # \ref SensorsData object contains the previous \ref IMUData structure that was used in ZED SDK v2.X:
    # \n For IMU data, the values are provided in 2 ways :
    # <ul>
    #   <li><b>Time-fused</b> pose estimation that can be accessed using:
    #       <ul><li>\ref IMUData.get_pose "data.get_imu_data().get_pose()"</li></ul>
    #   </li>
    #   <li><b>Raw values</b> from the IMU sensor:
    #       <ul>
    #           <li>\ref IMUData.get_angular_velocity "data.get_imu_data().get_angular_velocity()", corresponding to the gyroscope</li>
    #           <li>\ref IMUData.get_linear_acceleration "data.get_imu_data().get_linear_acceleration()", corresponding to the accelerometer</li>
    #       </ul> both the gyroscope and accelerometer are synchronized.
    #   </li>
    # </ul>
    # 
    # The delta time between previous and current values can be calculated using \ref data.imu.timestamp
    #
    # \note The IMU quaternion (fused data) is given in the specified \ref COORDINATE_SYSTEM of \ref InitParameters.
    #   
    # \param data[out] : The SensorsData variable to store the data.
    # \param reference_frame[in]: Defines the reference from which you want the data to be expressed. Default: \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD".
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if sensors data have been extracted.
    # \return \ref ERROR_CODE "ERROR_CODE.SENSORS_NOT_AVAILABLE" if the camera model is a \ref MODEL "MODEL.ZED".
    # \return \ref ERROR_CODE "ERROR_CODE.MOTION_SENSORS_REQUIRED" if the camera model is correct but the sensors module is not opened.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" if the <b>reference_time</b> is not valid. See Warning.
    #
    # \warning In SVO reading mode, the \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" is currently not available (yielding \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS".
    # \warning Only the quaternion data and barometer data (if available) at \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" are available. Other values will be set to 0.
    #
    def get_sensors_data(self, SensorsData py_sensors_data, time_reference = TIME_REFERENCE.CURRENT) -> ERROR_CODE:
        return _error_code_cache.get(
            <int>self.camera.getSensorsData(py_sensors_data.sensorsData, <c_TIME_REFERENCE>(<int>time_reference.value)),
            ERROR_CODE.FAILURE)

    ##
    # Set an optional IMU orientation hint that will be used to assist the tracking during the next \ref grab().
    # 
    # This method can be used to assist the positional tracking rotation.
    # 
    # \note This method is only effective if the camera has a model other than a \ref MODEL "MODEL.ZED", which does not contains internal sensors.
    # \warning It needs to be called before the \ref grab() method.
    # \param transform : \ref Transform to be ingested into IMU fusion. Note that only the rotation is used.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS"  if the transform has been passed, \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" otherwise (e.g. when used with a ZED camera which doesn't have IMU data).
    def set_imu_prior(self, Transform transfom) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.setIMUPrior(transfom.transform[0]), ERROR_CODE.FAILURE)

    ##
    # Retrieves the estimated position and orientation of the camera in the specified \ref REFERENCE_FRAME "reference frame".
    #
    # - Using \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD", the returned pose relates to the initial position of the camera (\ref PositionalTrackingParameters.initial_world_transform ).
    # - Using \ref REFERENCE_FRAME "REFERENCE_FRAME.CAMERA", the returned pose relates to the previous position of the camera.
    #
    # If the tracking has been initialized with \ref PositionalTrackingParameters.enable_area_memory to True (default), this method can return \ref POSITIONAL_TRACKING_STATE "POSITIONAL_TRACKING_STATE.SEARCHING".
    # This means that the tracking lost its link to the initial referential and is currently trying to relocate the camera. However, it will keep on providing position estimations.
    # 
    # \param camera_pose[out]: The pose containing the position of the camera and other information (timestamp, confidence).
    # \param reference_frame[in] : Defines the reference from which you want the pose to be expressed. Default: \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD".
    # \return The current \ref POSITIONAL_TRACKING_STATE "state" of the tracking process.
    #
    # \note Extract Rotation Matrix: Pose.get_rotation_matrix()
    # \note Extract Translation Vector: Pose.get_translation()
    # \note Extract Orientation / Quaternion: Pose.get_orientation()
    # 
    # \warning This method requires the tracking to be enabled. \ref enablePositionalTracking() .
    # 
    # \note The position is provided in the \ref InitParameters.coordinate_system . See \ref COORDINATE_SYSTEM for its physical origin.
    #
    # \code
    # while True:
    #       if zed.grab() == sl.ERROR_CODE.SUCCESS: # Grab an image and computes the tracking
    #           camera_pose = sl.Pose()
    #           zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
    #
    #           translation = camera_pose.get_translation().get()
    #           print("Camera position: X=", translation[0], " Y=", translation[1], " Z=", translation[2])
    #           print("Camera Euler rotation: X=", camera_pose.get_euler_angles()[0], " Y=", camera_pose.get_euler_angles()[1], " Z=", camera_pose.get_euler_angles()[2])
    #           print("Camera Rodrigues rotation: X=", camera_pose.get_rotation_vector()[0], " Y=", camera_pose.get_rotation_vector()[1], " Z=", camera_pose.get_rotation_vector()[2])
    #           orientation = camera_pose.get_orientation().get()
    #           print("Camera quaternion orientation: X=", orientation[0], " Y=", orientation[1], " Z=", orientation[2], " W=", orientation[3])
    # \endcode
    def get_position(self, Pose py_pose, reference_frame: REFERENCE_FRAME = REFERENCE_FRAME.WORLD) -> POSITIONAL_TRACKING_STATE:
        if not isinstance(reference_frame, REFERENCE_FRAME):
            raise TypeError("Invalid reference_frame type. Expected sl.REFERENCE_FRAME, got " + str(type(reference_frame)))
        return POSITIONAL_TRACKING_STATE(<int>self.camera.getPosition(py_pose.pose, <c_REFERENCE_FRAME>(<int>reference_frame.value)))

    ##
    # \brief Get the current positional tracking landmarks.
    # \param landmarks : The dictionary of landmarks_id and landmark.
    # \return ERROR_CODE that indicate if the function succeed or not.
    #
    def get_positional_tracking_landmarks(self, dict landmarks) -> ERROR_CODE:
        cdef map[uint64_t, c_Landmark] landmarks_map
        error = self.camera.getPositionalTrackingLandmarks(landmarks_map)

        cdef map[uint64_t, c_Landmark].iterator landmarks_map_it = landmarks_map.begin()
        while landmarks_map_it != landmarks_map.end():
            landmark_id = dereference(landmarks_map_it).first
            l = Landmark()
            l.landmark = dereference(landmarks_map_it).second
            landmarks[landmark_id] = l           
            postincrement(landmarks_map_it)
        return _error_code_cache.get(<int>error, ERROR_CODE.FAILURE)

    ##
    # \brief Get the current positional tracking landmark.
    # \param landmark : The landmark.
    # \return ERROR_CODE that indicate if the function succeed or not.
    #
    def get_positional_tracking_landmarks2d(self, list landmark2d) -> ERROR_CODE:
        cdef vector[c_Landmark2D] all_landmark2d
        error = self.camera.getPositionalTrackingLandmarks2D(all_landmark2d)

        # Clear existing contents if any
        if hasattr(landmark2d, 'clear'):
            landmark2d.clear()

        # Add landmarks to the list
        for l in all_landmark2d:
            l2d = Landmark2D()
            l2d.landmark2d = l
            landmark2d.append(l2d)

        return _error_code_cache.get(<int>error, ERROR_CODE.FAILURE)

    ## 
    # \brief Return the current status of positional tracking module.
    # 
    # \return sl::PositionalTrackingStatus current status of positional tracking module. 
    # 
    def get_positional_tracking_status(self) -> PositionalTrackingStatus:
        status = PositionalTrackingStatus()
        status.odometry_status = self.camera.getPositionalTrackingStatus().odometry_status
        status.spatial_memory_status = self.camera.getPositionalTrackingStatus().spatial_memory_status
        status.tracking_fusion_status = self.camera.getPositionalTrackingStatus().tracking_fusion_status
        return status

    

    ##
    # Returns the state of the spatial memory export process.
    #
    # As \ref Camera.save_area_map() only starts the exportation, this method allows you to know when the exportation finished or if it failed.
    # \return The current \ref AREA_EXPORTING_STATE "state" of the spatial memory export process.
    def get_area_export_state(self) -> AREA_EXPORTING_STATE:
        return AREA_EXPORTING_STATE(<int>self.camera.getAreaExportState())

    ##
    # Saves the current area learning file. The file will contain spatial memory data generated by the tracking.
    #
    # If the tracking has been initialized with \ref PositionalTrackingParameters.enable_area_memory to True (default), the method allows you to export the spatial memory.
    # \n Reloading the exported file in a future session with \ref PositionalTrackingParameters.area_file_path initializes the tracking within the same referential.
    # \n This method is asynchronous, and only triggers the file generation. You can use \ref get_area_export_state() to get the export state.
    # The positional tracking keeps running while exporting.
    #
    # \param area_file_path : Path of an '.area' file to save the spatial memory database in.
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if the <b>area_file_path</b> file wasn't found, \ref ERROR_CODE "ERROR_CODE.SUCCESS" otherwise.
    # 
    # See \ref get_area_export_state()
    #
    # \note Please note that this method will also flush the area database that was built/loaded.
    # 
    # \warning If the camera wasn't moved during the tracking session, or not enough, the spatial memory won't be usable and the file won't be exported.
    # \warning The \ref get_area_export_state() will return \ref AREA_EXPORTING_STATE "AREA_EXPORTING_STATE.FILE_EMPTY".
    # \warning A few meters (~3m) of translation or a full rotation should be enough to get usable spatial memory.
    # \warning However, as it should be used for relocation purposes, visiting a significant portion of the environment is recommended before exporting.
    #
    # \code
    # while True :
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS: # Grab an image and computes the tracking
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
    def save_area_map(self, area_file_path="") -> ERROR_CODE:
        filename = (<str>area_file_path).encode()
        return _error_code_cache.get(<int>self.camera.saveAreaMap(String(<char*>filename)), ERROR_CODE.FAILURE)

    ##
    # Disables the positional tracking.
    #
    # The positional tracking is immediately stopped. If a file path is given, \ref save_area_map() will be called asynchronously. See \ref get_area_export_state() to get the exportation state.
    # If the tracking has been enabled, this function will automatically be called by \ref close() .
    # 
    # \param area_file_path : If set, saves the spatial memory into an '.area' file. Default: (empty)
    # \n <b>area_file_path</b> is the name and path of the database, e.g. <i>path/to/file/myArea1.area"</i>.
    #
    def disable_positional_tracking(self, area_file_path="") -> None:
        filename = (<str>area_file_path).encode()
        self.camera.disablePositionalTracking(String(<char*> filename))
    
    ##
    # Tells if the tracking module is enabled
    def is_positional_tracking_enabled(self) -> bool:
        return self.camera.isPositionalTrackingEnabled()

    ##
    # Resets the tracking, and re-initializes the position with the given transformation matrix.
    # \param path : Position of the camera in the world frame when the method is called.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the tracking has been reset, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \note Please note that this method will also flush the accumulated or loaded spatial memory.
    def reset_positional_tracking(self, Transform path) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.resetPositionalTracking(path.transform[0]), ERROR_CODE.FAILURE)

    ##
    # Initializes and starts the spatial mapping processes.
    #
    # The spatial mapping will create a geometric representation of the scene based on both tracking data and 3D point clouds.
    # The resulting output can be a \ref Mesh or a \ref FusedPointCloud. It can be be obtained by calling \ref extract_whole_spatial_map() or \ref retrieve_spatial_map_async().
    # Note that \ref retrieve_spatial_map_async should be called after \ref request_spatial_map_async().
    # 
    # \param py_spatial : A structure containing all the specific parameters for the spatial mapping.
    # Default: a balanced parameter preset between geometric fidelity and output file size. For more information, see the \ref SpatialMappingParameters documentation.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \warning The tracking (\ref enable_positional_tracking() ) and the depth (\ref RuntimeParameters.enable_depth ) needs to be enabled to use the spatial mapping.
    # \warning The performance greatly depends on the <b>py_spatial</b>.
    # \warning Lower SpatialMappingParameters.range_meter and SpatialMappingParameters.resolution_meter for higher performance.
    # If the mapping framerate is too slow in live mode, consider using an SVO file, or choose a lower mesh resolution.
    #
    # \note This feature uses host memory (RAM) to store the 3D map. The maximum amount of available memory allowed can be tweaked using the SpatialMappingParameters.
    # \n Exceeding the maximum memory allowed immediately stops the mapping.
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
    #     init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system (The OpenGL one)
    #     init_params.coordinate_units = sl.UNIT.METER # Set units in meters
    #
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         exit(-1)
    #
    #     # Positional tracking needs to be enabled before using spatial mapping
    #     tracking_parameters = sl.PositionalTrackingParameters()
    #     err = zed.enable_positional_tracking(tracking_parameters)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         exit(-1)
    #
    #     # Enable spatial mapping
    #     mapping_parameters = sl.SpatialMappingParameters()
    #     err = zed.enable_spatial_mapping(mapping_parameters)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         exit(-1)
    #
    #     # Grab data during 500 frames
    #     i = 0
    #     mesh = sl.Mesh() # Create a mesh object
    #     while i < 500 :
    #           # For each new grab, mesh data is updated
    #           if zed.grab() == sl.ERROR_CODE.SUCCESS :
    #               # In the background, the spatial mapping will use newly retrieved images, depth and pose to update the mesh
    #               mapping_state = zed.get_spatial_mapping_state()
    #
    #               # Print spatial mapping state
    #               print("Images captured: ", i, "/ 500  ||  Spatial mapping state: ", repr(mapping_state))
    #           i = i + 1
    #
    #     # Extract, filter and save the mesh in a .obj file
    #     print("Extracting Mesh ...")
    #     zed.extract_whole_spatial_map(mesh) # Extract the whole mesh
    #     print("Filtering Mesh ...")
    #     mesh.filter(sl.MESH_FILTER.LOW) # Filter the mesh (remove unnecessary vertices and faces)
    #     print("Saving Mesh in mesh.obj ...")
    #     mesh.save("mesh.obj") # Save the mesh in an obj file
    #
    #     # Disable tracking and mapping and close the camera
    #     zed.disable_spatial_mapping()
    #     zed.disable_positional_tracking()
    #     zed.close()
    #     return 0
    #
    # if __name__ == "__main__" :
    #     main()
    # \endcode
    def enable_spatial_mapping(self, SpatialMappingParameters py_spatial = None) -> ERROR_CODE:
        if py_spatial is None:
            py_spatial = SpatialMappingParameters()
        return _error_code_cache.get(<int>self.camera.enableSpatialMapping(deref((<SpatialMappingParameters>py_spatial).spatial)), ERROR_CODE.FAILURE)

    ##
    # Pauses or resumes the spatial mapping processes.
    # 
    # As spatial mapping runs asynchronously, using this method can pause its computation to free some processing power, and resume it again later.
    # \n For example, it can be used to avoid mapping a specific area or to pause the mapping when the camera is static.
    # \param status : If True, the integration is paused. If False, the spatial mapping is resumed.
    def pause_spatial_mapping(self, bool status) -> None:
        self.camera.pauseSpatialMapping(status)

    ##
    #  Returns the current spatial mapping state.
    #
    # As the spatial mapping runs asynchronously, this method allows you to get reported errors or status info.
    # \return The current state of the spatial mapping process.
    # 
    # See also \ref SPATIAL_MAPPING_STATE
    def get_spatial_mapping_state(self) -> SPATIAL_MAPPING_STATE:
        return SPATIAL_MAPPING_STATE(<int>self.camera.getSpatialMappingState())

    ##
    # Starts the spatial map generation process in a non-blocking thread from the spatial mapping process.
    # 
    # The spatial map generation can take a long time depending on the mapping resolution and covered area. This function will trigger the generation of a mesh without blocking the program.
    # You can get info about the current generation using \ref get_spatial_map_request_status_async(), and retrieve the mesh using \ref retrieve_spatial_map_async().
    #
    # \note Only one mesh can be generated at a time. If the previous mesh generation is not over, new calls of the function will be ignored.
    def request_spatial_map_async(self) -> None:
        self.camera.requestSpatialMapAsync()

    ##
    # Returns the spatial map generation status.
    #
    # This status allows you to know if the mesh can be retrieved by calling \ref retrieve_spatial_map_async().
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the mesh is ready and not yet retrieved, otherwise \ref ERROR_CODE "ERROR_CODE.FAILURE".
    def get_spatial_map_request_status_async(self) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.getSpatialMapRequestStatusAsync(), ERROR_CODE.FAILURE)

    ##
    # Retrieves the current generated spatial map.
    # 
    # After calling \ref request_spatial_map_async(), this method allows you to retrieve the generated mesh or fused point cloud.
    # \n The \ref Mesh or \ref FusedPointCloud will only be available when \ref get_spatial_map_request_status_async() returns \ref ERROR_CODE "ERROR_CODE.SUCCESS".
    #
    # \param py_mesh[out] : The \ref Mesh or \ref FusedPointCloud to be filled with the generated spatial map.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the mesh is retrieved, otherwise \ref ERROR_CODE "ERROR_CODE.FAILURE".
    # 
    # \note This method only updates the necessary chunks and adds the new ones in order to improve update speed.
    # \warning You should not modify the mesh / fused point cloud between two calls of this method, otherwise it can lead to a corrupted mesh / fused point cloud.
    # See \ref request_spatial_map_async() for an example.
    def retrieve_spatial_map_async(self, py_mesh) -> ERROR_CODE:
        if isinstance(py_mesh, Mesh) :
            return _error_code_cache.get(<int>self.camera.retrieveSpatialMapAsync(deref((<Mesh>py_mesh).mesh)), ERROR_CODE.FAILURE)
        elif isinstance(py_mesh, FusedPointCloud) :
            py_mesh = <FusedPointCloud> py_mesh
            return _error_code_cache.get(<int>self.camera.retrieveSpatialMapAsync(deref((<FusedPointCloud>py_mesh).fpc)), ERROR_CODE.FAILURE)
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    ##
    # Extract the current spatial map from the spatial mapping process.
    #
    # If the object to be filled already contains a previous version of the mesh / fused point cloud, only changes will be updated, optimizing performance.
    #
    # \param py_mesh[out] : The \ref Mesh or \ref FusedPointCloud to be filled with the generated spatial map.
    #
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the mesh is filled and available, otherwise \ref ERROR_CODE "ERROR_CODE.FAILURE".
    #
    # \warning This is a blocking function. You should either call it in a thread or at the end of the mapping process.
    # The extraction can be long, calling this function in the grab loop will block the depth and tracking computation giving bad results.
    def extract_whole_spatial_map(self, py_mesh) -> ERROR_CODE:
        if isinstance(py_mesh, Mesh) :
            return _error_code_cache.get(<int>self.camera.extractWholeSpatialMap(deref((<Mesh>py_mesh).mesh)), ERROR_CODE.FAILURE)
        elif isinstance(py_mesh, FusedPointCloud) :
            return _error_code_cache.get(<int>self.camera.extractWholeSpatialMap(deref((<FusedPointCloud>py_mesh).fpc)), ERROR_CODE.FAILURE)
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.") 

    ##
    # Checks the plane at the given left image coordinates.
    # 
    # This method gives the 3D plane corresponding to a given pixel in the latest left image \ref grab() "grabbed".
    # \n The pixel coordinates are expected to be contained x=[0;width-1] and y=[0;height-1], where width/height are defined by the input resolution.
    # 
    # \param coord[in] : The image coordinate. The coordinate must be taken from the full-size image
    # \param plane[out] : The detected plane if the method succeeded.
    # \param parameters[in] :  A structure containing all the specific parameters for the plane detection. Default: a preset of PlaneDetectionParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if a plane is found otherwise \ref ERROR_CODE "ERROR_CODE.PLANE_NOT_FOUND".
    #
    # \note The reference frame is defined by the \ref RuntimeParameters.measure3D_reference_frame given to the \ref grab() method.
    def find_plane_at_hit(self, coord, py_plane: Plane, parameters=PlaneDetectionParameters()) -> ERROR_CODE:
        cdef Vector2[uint] vec = Vector2[uint](coord[0], coord[1])
        return _error_code_cache.get(<int>self.camera.findPlaneAtHit(vec, py_plane.plane, deref((<PlaneDetectionParameters>parameters).plane_detection_params)), ERROR_CODE.FAILURE)

    ##
    # Detect the floor plane of the scene.
    # 
    # This method analyses the latest image and depth to estimate the floor plane of the scene.
    # \n It expects the floor plane to be visible and bigger than other candidate planes, like a table.
    # 
    # \param py_plane[out] : The detected floor plane if the method succeeded.
    # \param reset_tracking_floor_frame[out] : The transform to align the tracking with the floor plane.
    # \n The initial position will then be at ground height, with the axis align with the gravity.
    # \n The positional tracking needs to be reset/enabled with this transform as a parameter (PositionalTrackingParameters.initial_world_transform).
    # \param floor_height_prior[in] : Prior set to locate the floor plane depending on the known camera distance to the ground, expressed in the same unit as the ZED.
    # \n If the prior is too far from the detected floor plane, the method will return \ref ERROR_CODE "ERROR_CODE.PLANE_NOT_FOUND".
    # \param world_orientation_prior[in] : Prior set to locate the floor plane depending on the known camera orientation to the ground.
    # \n If the prior is too far from the detected floor plane, the method will return \ref ERROR_CODE "ERROR_CODE.PLANE_NOT_FOUND.
    # \param floor_height_prior_tolerance[in] : Prior height tolerance, absolute value.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the floor plane is found and matches the priors (if defined), otherwise \ref ERROR_CODE "ERROR_CODE.PLANE_NOT_FOUND".
    #
    # \note The reference frame is defined by the sl.RuntimeParameters (measure3D_reference_frame) given to the grab() method.
    # \note The length unit is defined by sl.InitParameters (coordinate_units).
    # \note With the ZED, the assumption is made that the floor plane is the dominant plane in the scene. The ZED Mini uses gravity as prior.
    #
    def find_floor_plane(self,
                         Plane py_plane,
                         Transform reset_tracking_floor_frame,
                         float floor_height_prior = float('nan'),
                         Rotation world_orientation_prior = Rotation(Matrix3f().zeros()),
                         float floor_height_prior_tolerance = float('nan')
    ) -> ERROR_CODE:
        return _error_code_cache.get(
            <int>self.camera.findFloorPlane(py_plane.plane, reset_tracking_floor_frame.transform[0], floor_height_prior, (<Rotation>world_orientation_prior).rotation[0], floor_height_prior_tolerance),
            ERROR_CODE.FAILURE)

    ##
    # Disables the spatial mapping process.
    #
    # The spatial mapping is immediately stopped.
    # \n If the mapping has been enabled, this method will automatically be called by \ref close().
    # \note This method frees the memory allocated for the spatial mapping, consequently, meshes and fused point clouds cannot be retrieved after this call.
    def disable_spatial_mapping(self) -> None:
        self.camera.disableSpatialMapping()

    ##
    # Creates a streaming pipeline.
    #
    # \param streaming_parameters : A structure containing all the specific parameters for the streaming. Default: a reset of StreamingParameters .
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the streaming was successfully started.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" if open() was not successfully called before.
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if streaming RTSP protocol was not able to start.
    # \return \ref ERROR_CODE "ERROR_CODE.NO_GPU_COMPATIBLE" if the streaming codec is not supported (in this case, use H264 codec which is supported on all NVIDIA GPU the ZED SDK supports).
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
    def enable_streaming(self, StreamingParameters streaming_parameters = None) -> ERROR_CODE:
        if streaming_parameters is None:
            streaming_parameters = StreamingParameters()
        return _error_code_cache.get(
            <int>self.camera.enableStreaming(deref((<StreamingParameters>streaming_parameters).streaming)),
            ERROR_CODE.FAILURE)

    ##
    # Disables the streaming initiated by \ref enable_streaming().
    # \note This method will automatically be called by \ref close() if enable_streaming() was called.
    #
    # See \ref enable_streaming() for an example.
    def disable_streaming(self) -> None:
        self.camera.disableStreaming()

    ##
    # Tells if the streaming is running.
    # \return True if the stream is running, False otherwise.
    def is_streaming_enabled(self) -> bool:
        return self.camera.isStreamingEnabled()

    ##
    # Creates an SVO file to be filled by enable_recording() and disable_recording().
    # 
    # \n SVO files are custom video files containing the un-rectified images from the camera along with some meta-data like timestamps or IMU orientation (if applicable).
    # \n They can be used to simulate a live ZED and test a sequence with various SDK parameters.
    # \n Depending on the application, various compression modes are available. See \ref SVO_COMPRESSION_MODE.
    # 
    # \param record : A structure containing all the specific parameters for the recording such as filename and compression mode. Default: a reset of RecordingParameters .
    # \return An \ref ERROR_CODE that defines if the SVO file was successfully created and can be filled with images.
    # 
    # \warning This method can be called multiple times during a camera lifetime, but if <b>video_filename</b> is already existing, the file will be erased.
    #
    # 
    # \code
    # import pyzed.sl as sl
    # 
    # def main() :
    #     # Create a ZED camera object
    #     zed = sl.Camera()
    #     # Set initial parameters
    #     init_params = sl.InitParameters()
    #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
    #     init_params.coordinate_units = sl.UNIT.METER # Set units in meters
    #     # Open the camera
    #     err = zed.open(init_params)
    #     if (err != sl.ERROR_CODE.SUCCESS):
    #         print(repr(err))
    #         exit(-1)
    #
    #     # Enable video recording
    #     record_params = sl.RecordingParameters("myVideoFile.svo")
    #     err = zed.enable_recording(record_params)
    #     if (err != sl.ERROR_CODE.SUCCESS):
    #         print(repr(err))
    #         exit(-1)
    # 
    #     # Grab data during 500 frames
    #     i = 0
    #     while i < 500 :
    #         # Grab a new frame
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
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
    def enable_recording(self, RecordingParameters record) -> ERROR_CODE:
        return _error_code_cache.get(<int>self.camera.enableRecording(deref(record.record)), ERROR_CODE.FAILURE)

    ##
    # Disables the recording initiated by \ref enable_recording() and closes the generated file.
    #
    # \note This method will automatically be called by \ref close() if \ref enable_recording() was called.
    # 
    # See \ref enable_recording() for an example.
    def disable_recording(self) -> None:
        self.camera.disableRecording()

    ##
    # Get the recording information.
    # \return The recording state structure. For more details, see \ref RecordingStatus.
    def get_recording_status(self) -> RecordingStatus:
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
    # \param status : If True, the recording is paused. If False, the recording is resumed.
    def pause_recording(self, bool value=True) -> None:
        self.camera.pauseRecording(value)

    ##
    # Returns the RecordingParameters used.
    #
    # It corresponds to the structure given as argument to the enable_recording() method.
    # \return \ref RecordingParameters containing the parameters used for recording initialization.
    def get_recording_parameters(self) -> RecordingParameters:
        param = RecordingParameters()
        param.record.video_filename = self.camera.getRecordingParameters().video_filename
        param.record.compression_mode = self.camera.getRecordingParameters().compression_mode
        param.record.target_framerate = self.camera.getRecordingParameters().target_framerate
        param.record.bitrate = self.camera.getRecordingParameters().bitrate
        param.record.transcode_streaming_input = self.camera.getRecordingParameters().transcode_streaming_input
        return param

    ##
    # Get the Health information.
    # \return The health state structure. For more details, see \ref HealthStatus.
    def get_health_status(self) -> HealthStatus:
        state = HealthStatus()
        state.enabled = self.camera.getHealthStatus().enabled
        state.low_image_quality = self.camera.getHealthStatus().low_image_quality
        state.low_lighting = self.camera.getHealthStatus().low_lighting
        state.low_depth_reliability = self.camera.getHealthStatus().low_depth_reliability
        state.low_motion_sensors_reliability = self.camera.getHealthStatus().low_motion_sensors_reliability
        return state

    def get_retrieve_image_resolution(self, Resolution resolution = None) -> Resolution:
        if resolution is None:
            resolution = Resolution(0, 0)
        image_res = Resolution()
        image_res.resolution = self.camera.getRetrieveImageResolution((<Resolution>resolution).resolution)
        return image_res

    def get_retrieve_measure_resolution(self, Resolution resolution = None) -> Resolution:
        if resolution is None:
            resolution = Resolution(-1, -1)
        measure_res = Resolution()
        measure_res.resolution = self.camera.getRetrieveMeasureResolution((<Resolution>resolution).resolution)
        return measure_res

    ##
    # Initializes and starts object detection module.
    #
    # The object detection module currently supports multiple class of objects with the \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX" or \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE".
    # \n The full list of detectable objects is available through \ref OBJECT_CLASS and \ref OBJECT_SUBCLASS.
    #
    # \n Detected objects can be retrieved using the \ref retrieve_objects() method.
    #  the \ref retrieve_objects() method will be blocking during the detection.
    #
    # \n Alternatively, the object detection module supports custom class of objects with the \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS" (see \ref ingestCustomBoxObjects or \ref ingestCustomMaskObjects)
    # or \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS" (see \ref ObjectDetectionParameters.custom_onnx_file).
    #
    # \n Detected custom objects can be retrieved using the \ref retrieve_custom_objects() method.
    #
    # \note - <b>This Depth Learning detection module is not available \ref MODEL "MODEL.ZED" cameras.</b>
    # \note - This feature uses AI to locate objects and requires a powerful GPU. A GPU with at least 3GB of memory is recommended.
    # 
    # \param object_detection_parameters : A structure containing all the specific parameters for the object detection. Default: a preset of ObjectDetectionParameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine.
    # \return \ref ERROR_CODE "ERROR_CODE.OBJECT_DETECTION_NOT_AVAILABLE" if the AI model is missing or corrupted. In this case, the SDK needs to be reinstalled
    # \return \ref ERROR_CODE "ERROR_CODE.OBJECT_DETECTION_MODULE_NOT_COMPATIBLE_WITH_CAMERA" if the camera used does not have an IMU (\ref MODEL "MODEL.ZED").
    # \return \ref ERROR_CODE "ERROR_CODE.SENSORS_NOT_DETECTED" if the camera model is correct (not \ref MODEL "MODEL.ZED") but the IMU is missing. It probably happens because \ref InitParameters.sensors_required was set to False and that IMU has not been found.
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" if one of the <b>object_detection_parameters</b> parameter is not compatible with other modules parameters (for example, <b>depth_mode</b> has been set to \ref DEPTH_MODE "DEPTH_MODE.NONE").
    # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \note The IMU gives the gravity vector that helps in the 3D box localization. Therefore the object detection module is not available for the \ref MODEL "MODEL.ZED" models.
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
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Opening camera error:", repr(err))
    #         exit(-1)
    #
    #     # Enable position tracking (mandatory for object detection)
    #     tracking_params = sl.PositionalTrackingParameters()
    #     err = zed.enable_positional_tracking(tracking_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Enabling Positional Tracking error:", repr(err))
    #         exit(-1)
    #
    #     # Set the object detection parameters
    #     object_detection_params = sl.ObjectDetectionParameters()
    #
    #     # Enable the object detection
    #     err = zed.enable_object_detection(object_detection_params)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         print("Enabling Object Detection error:", repr(err))
    #         exit(-1)
    #
    #     # Grab an image and detect objects on it
    #     objects = sl.Objects()
    #     while True:
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #             zed.retrieve_objects(objects)
    #             print(len(objects.object_list), "objects detected")
    #             # Use the objects in your application
    # 
    #     # Close the camera
    #     zed.disable_object_detection()
    #     zed.close()
    #
    # if __name__ == "__main__":
    #     main()
    # \endcode
    def enable_object_detection(self, ObjectDetectionParameters object_detection_parameters = None) -> ERROR_CODE:
        if object_detection_parameters is None:
            object_detection_parameters = ObjectDetectionParameters()
        return _error_code_cache.get(
            <int>self.camera.enableObjectDetection(deref((<ObjectDetectionParameters>object_detection_parameters).object_detection)),
            ERROR_CODE.FAILURE)

    ##
    # Disables the object detection process.
    #
    # The object detection module immediately stops and frees its memory allocations.
    #
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \param force_disable_all_instances : Should disable all instances of the object detection module or just <b>instance_module_id</b>.
    #
    # \note If the object detection has been enabled, this method will automatically be called by \ref close().
    def disable_object_detection(self, unsigned int instance_module_id=0, bool force_disable_all_instances=False) -> None:
        self.camera.disableObjectDetection(instance_module_id, force_disable_all_instances)

    ##
    # Set the object detection runtime parameters
    #
    def set_object_detection_runtime_parameters(self, ObjectDetectionRuntimeParameters object_detection_parameters, unsigned int instance_module_id=0) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            ret = self.camera.setObjectDetectionRuntimeParameters(deref(object_detection_parameters.object_detection_rt), instance_module_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Set the custom object detection runtime parameters
    #
    def set_custom_object_detection_runtime_parameters(self, CustomObjectDetectionRuntimeParameters custom_object_detection_parameters, unsigned int instance_module_id=0) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            ret = self.camera.setCustomObjectDetectionRuntimeParameters(deref(custom_object_detection_parameters.custom_object_detection_rt), instance_module_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Retrieve objects detected by the object detection module.
    #
    # This method returns the result of the object detection, whether the module is running synchronously or asynchronously.
    #
    # - <b>Asynchronous:</b> this method immediately returns the last objects detected. If the current detection isn't done, the objects from the last detection will be returned, and \ref Objects.is_new will be set to False.
    # - <b>Synchronous:</b> this method executes detection and waits for it to finish before returning the detected objects.
    #
    # It is recommended to keep the same \ref Objects object as the input of all calls to this method. This will enable the identification and tracking of every object detected.
    # 
    # \param py_objects[out] : The detected objects will be saved into this object. If the object already contains data from a previous detection, it will be updated, keeping a unique ID for the same person.
    # \param py_object_detection_parameters[in] : Object detection runtime settings, can be changed at each detection. In async mode, the parameters update is applied on the next iteration. If None, use the previously passed parameters.
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE.FAILURE" otherwise.
    #
    # \code
    # objects = sl.Objects()
    # while True:
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #         zed.retrieve_objects(objects)
    #         object_list = objects.object_list
    #         for i in range(len(object_list)):
    #             print(repr(object_list[i].label))
    # \endcode
    def retrieve_objects(self,
                         Objects py_objects,
                         ObjectDetectionRuntimeParameters py_object_detection_parameters = None,
                         unsigned int instance_module_id = 0
    ) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            if py_object_detection_parameters is None:
                ret = self.camera.retrieveObjects(py_objects.objects, instance_module_id)
            else:
                ret = self.camera.retrieveObjectsAndSetRuntimeParameters(py_objects.objects,
                                                                         deref(py_object_detection_parameters.object_detection_rt),
                                                                         instance_module_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Retrieve custom objects detected by the object detection module.
    #
    # If the object detection module is initialized with \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS", the objects retrieved will be the ones from \ref ingest_custom_box_objects or \ref ingest_custom_mask_objects.
    # If the object detection module is initialized with \ref OBJECT_DETECTION_MODEL "OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS", the objects retrieved will be the ones detected using the optimized \ref ObjectDetectionParameters.custom_onnx_file model.
    #
    # When running the detection internally, this method returns the result of the object detection, whether the module is running synchronously or asynchronously.
    #
    # - <b>Asynchronous:</b> this method immediately returns the last objects detected. If the current detection isn't done, the objects from the last detection will be returned, and \ref Objects::is_new will be set to false.
    # - <b>Synchronous:</b> this method executes detection and waits for it to finish before returning the detected objects.
    #
    # It is recommended to keep the same \ref Objects object as the input of all calls to this method. This will enable the identification and tracking of every object detected.
    #
    # \param py_objects : The detected objects will be saved into this object. If the object already contains data from a previous detection, it will be updated, keeping a unique ID for the same person.
    # \param custom_object_detection_parameters : Custom object detection runtime settings, can be changed at each detection. In async mode, the parameters update is applied on the next iteration. If None, use the previously passed parameters.
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" if everything went fine, \ref ERROR_CODE "ERROR_CODE::FAILURE" otherwise.
    #
    # \ref set_custom_object_detection_runtime_parameters and \ref retrieve_objects methods should be used instead.
    #
    # \code
    # objects = sl.Objects()
    # while True:
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #         zed.retrieve_custom_objects(objects)
    #         object_list = objects.object_list
    #         for i in range(len(object_list)):
    #             print(repr(object_list[i].label))
    # \endcode
    def retrieve_custom_objects(self,
                                Objects py_objects,
                                CustomObjectDetectionRuntimeParameters custom_object_detection_parameters = None,
                                unsigned int instance_module_id = 0
    ) -> ERROR_CODE:
        cdef c_ERROR_CODE ret
        with nogil:
            if custom_object_detection_parameters is None:
                ret = self.camera.retrieveObjects(py_objects.objects, instance_module_id)
            else:
                custom_object_detection_parameters.custom_object_detection_rt.object_detection_properties = deref(custom_object_detection_parameters._object_detection_properties.custom_object_detection_props)
                ret = self.camera.retrieveCustomObjectsAndSetRuntimeParameters(py_objects.objects,
                                                                               deref(custom_object_detection_parameters.custom_object_detection_rt),
                                                                               instance_module_id)
        return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

    ##
    # Get a batch of detected objects.
    # \warning This method needs to be called after \ref retrieve_objects, otherwise trajectories will be empty.
    # \n It is the \ref retrieve_objects method that ingest the current/live objects into the batching queue.
    # 
    # \param trajectories : list of \ref sl.ObjectsBatch that will be filled by the batching queue process. An empty list should be passed to the function
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine
    # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" if batching module is not available (TensorRT!=7.1) or if object tracking was not enabled.
    # 
    # \note Most of the time, the vector will be empty and will be filled every \ref BatchParameters::latency.
    # 
    # \code
    # objects = sl.Objects()                                        # Unique Objects to be updated after each grab
    # while True:                                                   # Main loop
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:                   # Grab an image from the camera
    #         zed.retrieve_objects(objects)                         # Call retrieve_objects so that objects are ingested in the batching system
    #         trajectories = []                                     # Create an empty list of trajectories 
    #         zed.get_objects_batch(trajectories)                   # Get batch of objects
    #         print("Size of batch: {}".format(len(trajectories)))
    # \endcode
    def get_objects_batch(self, list trajectories, unsigned int instance_module_id=0) -> ERROR_CODE:
        cdef vector[c_ObjectsBatch] output_trajectories
        cdef c_ERROR_CODE status = self.camera.getObjectsBatch(output_trajectories, instance_module_id)
        for trajectory in output_trajectories:
            curr = ObjectsBatch()
            curr.objects_batch = trajectory
            trajectories.append(curr)
        return _error_code_cache.get(status, ERROR_CODE.FAILURE)

    ##
    # Feed the 3D Object tracking function with your own 2D bounding boxes from your own detection algorithm.
    # \param objects_in : List of \ref CustomBoxObjectData to feed the object detection.
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine.
    # \note The detection should be done on the current grabbed left image as the internal process will use all currently available data to extract 3D information and perform object tracking.
    def ingest_custom_box_objects(self, list objects_in, unsigned int instance_module_id=0) -> ERROR_CODE:
        cdef vector[c_CustomBoxObjectData] custom_obj
        # Convert input list into C vector
        for i in range(len(objects_in)):
            custom_obj.push_back((<CustomBoxObjectData>objects_in[i]).custom_box_object_data)
        return _error_code_cache.get(<int>self.camera.ingestCustomBoxObjects(custom_obj, instance_module_id), ERROR_CODE.FAILURE)

    ##
    # Feed the 3D Object tracking function with your own 2D bounding boxes with masks from your own detection algorithm.
    # \param objects_in : List of \ref CustomMaskObjectData to feed the object detection.
    # \param instance_module_id : Id of the object detection instance. Used when multiple instances of the object detection module are enabled at the same time.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if everything went fine.
    # \note The detection should be done on the current grabbed left image as the internal process will use all currently available data to extract 3D information and perform object tracking.
    def ingest_custom_mask_objects(self, list objects_in, unsigned int instance_module_id = 0) -> ERROR_CODE:
        cdef vector[c_CustomMaskObjectData] custom_obj
        # Convert input list into C vector
        for i in range(len(objects_in)):
            custom_obj.push_back((<CustomMaskObjectData>objects_in[i]).custom_mask_object_data)
        return _error_code_cache.get(<int>self.camera.ingestCustomMaskObjects(custom_obj, instance_module_id), ERROR_CODE.FAILURE)

    ##
    # Tells if the object detection module is enabled.
    def is_object_detection_enabled(self, unsigned int instance_id = 0) -> bool:
        return self.camera.isObjectDetectionEnabled(instance_id)

    ##
    # Returns the version of the currently installed ZED SDK.
    # \return The ZED SDK version as a string with the following format: MAJOR.MINOR.PATCH
    # 
    # \code
    # print(sl.Camera.get_sdk_version())
    # \endcode
    @staticmethod
    def get_sdk_version() -> str:
        cam = Camera()
        return to_str(cam.camera.getSDKVersion()).decode()

    ##
    # List all the connected devices with their associated information.
    # 
    # This method lists all the cameras available and provides their serial number, models and other information.
    # \return The device properties for each connected camera.
    @staticmethod
    def get_device_list() -> list[DeviceProperties]:
        cam = Camera()
        vect_ = cam.camera.getDeviceList()
        vect_python = []
        for i in range(vect_.size()):
            prop = DeviceProperties()
            prop.camera_state = CAMERA_STATE(<int> vect_[i].camera_state)
            prop.id = vect_[i].id
            if not vect_[i].path.empty():
                prop.path = vect_[i].path.get().decode()
            prop.camera_model = MODEL(<int>vect_[i].camera_model)
            prop.serial_number = vect_[i].serial_number
            vect_python.append(prop)
        return vect_python

    ##
    # Lists all the streaming devices with their associated information.
    # 
    # \return The streaming properties for each connected camera.
    # \warning This method takes around 2 seconds to make sure all network informations has been captured. Make sure to run this method in a thread.
    @staticmethod
    def get_streaming_device_list() -> list[StreamingProperties]:
        cam = Camera()
        vect_ = cam.camera.getStreamingDeviceList()
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
    # Performs a hardware reset of the ZED 2 and the ZED 2i.
    # 
    # \param sn : Serial number of the camera to reset, or 0 to reset the first camera detected.
    # \param full_reboot : Perform a full reboot (sensors and video modules) if True, otherwise only the video module will be rebooted.
    # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" if everything went fine.
    # \return \ref ERROR_CODE "ERROR_CODE::CAMERA_NOT_DETECTED" if no camera was detected.
    # \return \ref ERROR_CODE "ERROR_CODE::FAILURE"  otherwise.
    #
    # \note This method only works for ZED 2, ZED 2i, and newer camera models.
    # 
    # \warning This method will invalidate any sl.Camera object, since the device is rebooting.
    @staticmethod
    def reboot(sn : int, full_reboot: bool =True) -> ERROR_CODE:
        cam = Camera()
        return _error_code_cache.get(<int>cam.camera.reboot(sn, full_reboot), ERROR_CODE.FAILURE)

    ##
    # Performs a hardware reset of all devices matching the InputType.
    # 
    # \param input_type : Input type of the devices to reset.
    # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" if everything went fine.
    # \return \ref ERROR_CODE "ERROR_CODE::CAMERA_NOT_DETECTED" if no camera was detected.
    # \return \ref ERROR_CODE "ERROR_CODE::FAILURE" otherwise.
    # \return \ref ERROR_CODE "ERROR_CODE::INVALID_FUNCTION_PARAMETERS" for SVOs and streams.
    # 
    # \warning This method will invalidate any sl.Camera object, since the device is rebooting.
    @staticmethod
    def reboot_from_input(input_type: INPUT_TYPE) -> ERROR_CODE:
        if not isinstance(input_type, INPUT_TYPE):
            raise TypeError("Argument is not of INPUT_TYPE type.")
        cam = Camera()
        return _error_code_cache.get(<int>cam.camera.reboot_from_type(<c_INPUT_TYPE>(<int>input_type.value)), ERROR_CODE.FAILURE)

##
# Lists the different types of communications available for Fusion module.
# \ingroup Fusion_group
# 
# | Enumerator     |                  |
# |----------------|------------------|
# | LOCAL_NETWORK  | The sender and receiver are on the same local network and communicate by RTP.\n The communication can be affected by the local network load. |
# | INTRA_PROCESS  | Both sender and receiver are declared by the same process and can be in different threads.\n This type of communication is optimized. |
class COMM_TYPE(enum.Enum):
    LOCAL_NETWORK = <int>c_COMM_TYPE.LOCAL_NETWORK
    INTRA_PROCESS = <int>c_COMM_TYPE.INTRA_PROCESS
    LAST = <int>c_COMM_TYPE.LAST

##
# Lists the types of error that can be raised by the Fusion.
#
# \ingroup Fusion_group
# 
# | Enumerator     |                  |
# |----------------|------------------|
# | GNSS_DATA_NEED_FIX | GNSS Data need fix status in order to run fusion. |
# | GNSS_DATA_COVARIANCE_MUST_VARY | Ingested covariance data must vary between ingest. |
# | BODY_FORMAT_MISMATCH | The senders are using different body formats.\n Consider changing them. |
# | NOT_ENABLED | The following module was not enabled. |
# | SOURCE_MISMATCH | Some sources are provided by SVO and others by LIVE stream. |
# | CONNECTION_TIMED_OUT | Connection timed out. Unable to reach the sender.\n Verify the sender's IP/port. |
# | SHARED_MEMORY_LEAK | Intra-process shared memory allocation issue.\n Multiple connections to the same data. |
# | INVALID_IP_ADDRESS | The provided IP address format is incorrect.\n Please provide the IP in the format 'a.b.c.d', where (a, b, c, d) are numbers between 0 and 255. |
# | CONNECTION_ERROR | Something goes bad in the connection between sender and receiver. |
# | FAILURE | Standard code for unsuccessful behavior. |
# | SUCCESS | Standard code for successful behavior. |
# | FUSION_INCONSISTENT_FPS | Significant differences observed between sender's FPS. |
# | FUSION_FPS_TOO_LOW | At least one sender has an FPS lower than 10 FPS. |
# | INVALID_TIMESTAMP | Problem detected with the ingested timestamp.\n Sample data will be ignored. |
# | INVALID_COVARIANCE | Problem detected with the ingested covariance.\n Sample data will be ignored. |
# | NO_NEW_DATA_AVAILABLE | All data from all sources has been consumed.\n No new data is available for processing. |
class FUSION_ERROR_CODE(enum.Enum):
    GNSS_DATA_NEED_FIX = <int>c_FUSION_ERROR_CODE.GNSS_DATA_NEED_FIX
    GNSS_DATA_COVARIANCE_MUST_VARY = <int>c_FUSION_ERROR_CODE.GNSS_DATA_COVARIANCE_MUST_VARY
    BODY_FORMAT_MISMATCH = <int>c_FUSION_ERROR_CODE.BODY_FORMAT_MISMATCH
    MODULE_NOT_ENABLED = <int>c_FUSION_ERROR_CODE.MODULE_NOT_ENABLED
    SOURCE_MISMATCH = <int>c_FUSION_ERROR_CODE.SOURCE_MISMATCH
    CONNECTION_TIMED_OUT = <int>c_FUSION_ERROR_CODE.CONNECTION_TIMED_OUT
    MEMORY_ALREADY_USED = <int>c_FUSION_ERROR_CODE.MEMORY_ALREADY_USED
    INVALID_IP_ADDRESS = <int>c_FUSION_ERROR_CODE.INVALID_IP_ADDRESS
    FAILURE = <int>c_FUSION_ERROR_CODE.FAILURE
    SUCCESS = <int>c_FUSION_ERROR_CODE.SUCCESS
    FUSION_INCONSISTENT_FPS = <int>c_FUSION_ERROR_CODE.FUSION_INCONSISTENT_FPS
    FUSION_FPS_TOO_LOW = <int>c_FUSION_ERROR_CODE.FUSION_FPS_TOO_LOW
    INVALID_TIMESTAMP = <int>c_FUSION_ERROR_CODE.INVALID_TIMESTAMP
    INVALID_COVARIANCE = <int>c_FUSION_ERROR_CODE.INVALID_COVARIANCE
    NO_NEW_DATA_AVAILABLE = <int>c_FUSION_ERROR_CODE.NO_NEW_DATA_AVAILABLE
    
    def __str__(self):
        return to_str(toString(<c_FUSION_ERROR_CODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_FUSION_ERROR_CODE>(<int>self.value))).decode()


cdef dict _fusion_error_code_cache = {}
def _initialize_fusion_error_codes():
    global _fusion_error_code_cache
    if not _fusion_error_code_cache:  # Only initialize if not already done
        for fusion_error_code in FUSION_ERROR_CODE:  # Iterate through the existing Python enum
            _fusion_error_code_cache[fusion_error_code.value] = fusion_error_code
_initialize_fusion_error_codes()

##
# Lists the types of error that can be raised during the Fusion by senders.
#
# \ingroup Fusion_group
# 
# | Enumerator     |                  |
# |----------------|------------------|
# | DISCONNECTED | The sender has been disconnected. |
# | SUCCESS | Standard code for successful behavior. |
# | GRAB_ERROR | The sender encountered a grab error. |
# | INCONSISTENT_FPS | The sender does not run with a constant frame rate. |
# | FPS_TOO_LOW | The frame rate of the sender is lower than 10 FPS. |
class SENDER_ERROR_CODE(enum.Enum):
    DISCONNECTED = <int>c_SENDER_ERROR_CODE.DISCONNECTED
    SUCCESS = <int>c_SENDER_ERROR_CODE.SUCCESS
    GRAB_ERROR = <int>c_SENDER_ERROR_CODE.GRAB_ERROR
    INCONSISTENT_FPS = <int>c_SENDER_ERROR_CODE.INCONSISTENT_FPS
    FPS_TOO_LOW = <int>c_SENDER_ERROR_CODE.FPS_TOO_LOW

    def __str__(self):
        return to_str(toString(<c_SENDER_ERROR_CODE>(<int>self.value))).decode()

    def __repr__(self):
        return to_str(toString(<c_SENDER_ERROR_CODE>(<int>self.value))).decode()

##
# Lists the types of possible position outputs.
#
# \ingroup Fusion_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | RAW            | The output position will be the raw position data. |
# | FUSION         | The output position will be the fused position projected into the requested camera repository. |
class POSITION_TYPE(enum.Enum):
        RAW  = <int>c_POSITION_TYPE.RAW
        FUSION  = <int>c_POSITION_TYPE.FUSION
        LAST  = <int>c_POSITION_TYPE.LAST

##
# Enum to define the reference frame of the fusion SDK.
#
# \ingroup Fusion_group
#
# | Enumerator     |                  |
# |----------------|------------------|
# | WORLD            | The world frame is the reference frame of the world according to the fused positional Tracking. |
# | BASELINK         | The base link frame is the reference frame where camera calibration is given. |
class FUSION_REFERENCE_FRAME(enum.Enum):
        WORLD  = <int>c_FUSION_REFERENCE_FRAME.WORLD
        BASELINK  = <int>c_FUSION_REFERENCE_FRAME.BASELINK

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
    def port(self) -> int:
        return self.communicationParameters.getPort()

    ##
    # The IP address of the sender
    @property
    def ip_address(self) -> str:
        return self.communicationParameters.getIpAddress().decode()

    ##
    # The type of the used communication
    @property
    def comm_type(self) -> COMM_TYPE:
        return COMM_TYPE(<int>self.communicationParameters.getType())

##
# Useful struct to store the Fusion configuration, can be read from /write to a JSON file.
# \ingroup Fusion_group
cdef class FusionConfiguration:
    cdef c_FusionConfiguration fusionConfiguration
    cdef Transform pose

    def __cinit__(self):
        self.pose = Transform()

    ##
    # The serial number of the used ZED camera.
    @property
    def serial_number(self) -> int:
        return self.fusionConfiguration.serial_number

    @serial_number.setter
    def serial_number(self, value: int):
        self.fusionConfiguration.serial_number = value

    ##
    # The communication parameters to connect this camera to the Fusion.
    @property
    def communication_parameters(self) -> CommunicationParameters:
        cp = CommunicationParameters()
        cp.communicationParameters = self.fusionConfiguration.communication_parameters
        return cp

    @communication_parameters.setter
    def communication_parameters(self, communication_parameters : CommunicationParameters):
        self.fusionConfiguration.communication_parameters = communication_parameters.communicationParameters

    ##
    # The WORLD Pose of the camera for Fusion in the unit and coordinate system defined by the user in the InitFusionParameters.
    @property
    def pose(self) -> Transform:
        for i in range(16):
            self.pose.transform.m[i] = self.fusionConfiguration.pose.m[i]
        return self.pose

    @pose.setter
    def pose(self, transform : Transform):
        self.fusionConfiguration.pose = deref(transform.transform)

    ##
    # The input type for the current camera.
    @property
    def input_type(self) -> InputType:
        inp = InputType()
        inp.input = self.fusionConfiguration.input_type
        return inp

    @input_type.setter
    def input_type(self, input_type : InputType):
        self.fusionConfiguration.input_type = input_type.input

##
# Read a configuration JSON file to configure a fusion process.
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON file containing the configuration.
# \param serial_number : The serial number of the ZED Camera you want to retrieve.
# \param coord_system : The COORDINATE_SYSTEM in which you want the World Pose to be in.
# \param unit : The UNIT in which you want the World Pose to be in.
#
# \return A \ref FusionConfiguration for the requested camera.
# \note Empty if no data were found for the requested camera.
def read_fusion_configuration_file_from_serial(self, json_config_filename : str, serial_number : int, coord_system : COORDINATE_SYSTEM, unit: UNIT) -> FusionConfiguration:
    fusion_configuration = FusionConfiguration()
    fusion_configuration.fusionConfiguration = c_readFusionConfigurationFile(json_config_filename.encode('utf-8'), serial_number, <c_COORDINATE_SYSTEM>(<int>coord_system.value), <c_UNIT>(<int>unit.value))
    return fusion_configuration

##
# Read a Configuration JSON file to configure a fusion process.
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON file containing the configuration.
# \param coord_sys : The COORDINATE_SYSTEM in which you want the World Pose to be in.
# \param unit : The UNIT in which you want the World Pose to be in.
#
# \return A list of \ref FusionConfiguration for all the camera present in the file.
# \note Empty if no data were found for the requested camera.
def read_fusion_configuration_file(json_config_filename : str, coord_system : COORDINATE_SYSTEM, unit: UNIT) -> list[FusionConfiguration]:
    cdef vector[c_FusionConfiguration] fusion_configurations = c_readFusionConfigurationFile2(json_config_filename.encode('utf-8'), <c_COORDINATE_SYSTEM>(<int>coord_system.value), <c_UNIT>(<int>unit.value))
    return_list = []
    for item in fusion_configurations:
        fc = FusionConfiguration()
        fc.fusionConfiguration = item
        return_list.append(fc)
    return return_list

##
# Read a Configuration JSON to configure a fusion process.
# \ingroup Fusion_group
# \param fusion_configuration : The JSON containing the configuration.
# \param coord_sys : The COORDINATE_SYSTEM in which you want the World Pose to be in.
# \param unit : The UNIT in which you want the World Pose to be in.
#
# \return A list of \ref FusionConfiguration for all the camera present in the file.
# \note Empty if no data were found for the requested camera.
def read_fusion_configuration_json(fusion_configuration : dict, coord_system : COORDINATE_SYSTEM, unit: UNIT) -> list[FusionConfiguration]:
    cdef vector[c_FusionConfiguration] fusion_configurations = c_readFusionConfigurationFile2(json.dumps(fusion_configuration).encode('utf-8'), <c_COORDINATE_SYSTEM>(<int>coord_system.value), <c_UNIT>(<int>unit.value))
    return_list = []
    for item in fusion_configurations:
        fc = FusionConfiguration()
        fc.fusionConfiguration = item
        return_list.append(fc)
    return return_list

##
# Write a Configuration JSON file to configure a fusion process.
# \ingroup Fusion_group
# \param json_config_filename : The name of the JSON that will contain the information.
# \param conf: A list of \ref FusionConfiguration listing all the camera configurations.
# \param coord_sys : The COORDINATE_SYSTEM in which the World Pose is.
# \param unit : The UNIT in which the World Pose is.
def write_configuration_file(json_config_filename : str, fusion_configurations : list, coord_sys : COORDINATE_SYSTEM, unit: UNIT):
    cdef vector[c_FusionConfiguration] confs
    for fusion_configuration in fusion_configurations:
        cast_conf = <FusionConfiguration>fusion_configuration
        confs.push_back(cast_conf.fusionConfiguration)

    c_writeConfigurationFile(json_config_filename.encode('utf-8'), confs, <c_COORDINATE_SYSTEM>(<int>coord_sys.value), <c_UNIT>(<int>unit.value))

##
# Holds the options used for calibrating GNSS / VIO.
# \ingroup Fusion_group
cdef class GNSSCalibrationParameters:
    cdef c_GNSSCalibrationParameters gnssCalibrationParameters

    ##
    # This parameter defines the target yaw uncertainty at which the calibration process between GNSS and VIO concludes.
    # The unit of this parameter is in radian.
    #
    # Default: 0.1 radians
    ##
    @property
    def target_yaw_uncertainty(self) -> float:
        return self.target_yaw_uncertainty

    @target_yaw_uncertainty.setter
    def target_yaw_uncertainty(self, value:float):
        self.gnssCalibrationParameters.target_yaw_uncertainty = value
    
    ##
    # When this parameter is enabled (set to true), the calibration process between GNSS and VIO accounts for the uncertainty in the determined translation, thereby facilitating the calibration termination.
    # The maximum allowable uncertainty is controlled by the 'target_translation_uncertainty' parameter.
    #
    # Default: False
    ##
    @property
    def enable_translation_uncertainty_target(self) -> bool:
        return self.gnssCalibrationParameters.enable_translation_uncertainty_target

    @enable_translation_uncertainty_target.setter
    def enable_translation_uncertainty_target(self, value:bool):
        self.gnssCalibrationParameters.enable_translation_uncertainty_target = value

    ##
    # This parameter defines the target translation uncertainty at which the calibration process between GNSS and VIO concludes.
    #
    # Default: 10e-2 (10 centimeters)
    ##
    @property
    def target_translation_uncertainty(self) -> float:
        return self.gnssCalibrationParameters.target_translation_uncertainty
        
    @target_translation_uncertainty.setter
    def target_translation_uncertainty(self, value:float):
        self.gnssCalibrationParameters.target_translation_uncertainty = value
        
    ##
    # This parameter determines whether reinitialization should be performed between GNSS and VIO fusion when a significant disparity is detected between GNSS data and the current fusion data.
    # It becomes particularly crucial during prolonged GNSS signal loss scenarios.
    #
    # Default: True
    ##
    @property
    def enable_reinitialization(self) -> bool:
        return self.gnssCalibrationParameters.enable_reinitialization

    @enable_reinitialization.setter
    def enable_reinitialization(self, value:bool):
        self.gnssCalibrationParameters.enable_reinitialization = value
        
    ##
    # This parameter determines the threshold for GNSS/VIO reinitialization.
    # If the fused position deviates beyond out of the region defined by the product of the GNSS covariance and the gnss_vio_reinit_threshold, a reinitialization will be triggered.
    #
    # Default: 5
    ##
    @property
    def gnss_vio_reinit_threshold(self) -> float:
        return self.gnssCalibrationParameters.gnss_vio_reinit_threshold
        
    @gnss_vio_reinit_threshold.setter
    def gnss_vio_reinit_threshold(self, value:float):
        self.gnssCalibrationParameters.gnss_vio_reinit_threshold = value

    ##
    # If this parameter is set to true, the fusion algorithm will used a rough VIO / GNSS calibration at first and then refine it. This allow you to quickly get a fused position.
    #
    # Default: True
    ##
    @property
    def enable_rolling_calibration(self) -> bool:
        return self.gnssCalibrationParameters.enable_rolling_calibration
        
    @enable_rolling_calibration.setter
    def enable_rolling_calibration(self, value:bool):
        self.gnssCalibrationParameters.enable_rolling_calibration = value

    ##
    # Define a transform between the GNSS antenna and the camera system for the VIO / GNSS calibration.
    #
    # Default value is [0,0,0], this position can be refined by the calibration if enabled
    ##
    @property
    def gnss_antenna_position(self) -> np.array[float]:
        cdef np.ndarray gnss_antenna_position = np.zeros(3)
        for i in range(3):
            gnss_antenna_position[i] = self.gnssCalibrationParameters.gnss_antenna_position[i]
        return gnss_antenna_position

    @gnss_antenna_position.setter
    def gnss_antenna_position(self, np.ndarray gnss_antenna_position):
        for i in range(3):
            self.gnssCalibrationParameters.gnss_antenna_position[i] = gnss_antenna_position[i]


##
# Holds the options used for initializing the positional tracking fusion module.
# \ingroup Fusion_group
cdef class PositionalTrackingFusionParameters:
    cdef c_PositionalTrackingFusionParameters positionalTrackingFusionParameters

    ##
    # This attribute is responsible for enabling or not GNSS positional tracking fusion.
    #
    # Default: False
    @property
    def enable_GNSS_fusion(self) -> bool:
        return self.positionalTrackingFusionParameters.enable_GNSS_fusion

    @enable_GNSS_fusion.setter
    def enable_GNSS_fusion(self, value: bool):
        self.positionalTrackingFusionParameters.enable_GNSS_fusion = value

    ##
    # Control the VIO / GNSS calibration process.
    #
    @property
    def gnss_calibration_parameters(self) -> GNSSCalibrationParameters:
        tmp = GNSSCalibrationParameters()
        tmp.gnssCalibrationParameters = self.positionalTrackingFusionParameters.gnss_calibration_parameters
        return tmp

    @gnss_calibration_parameters.setter
    def gnss_calibration_parameters(self, value: GNSSCalibrationParameters):
        self.positionalTrackingFusionParameters.gnss_calibration_parameters = value.gnssCalibrationParameters

    ##
    # Position and orientation of the base footprint with respect to the user world.
    # This transform represents a basis change from base footprint coordinate frame to user world coordinate frame
    # 
    @property
    def base_footprint_to_world_transform(self) -> Transform:
        tr = Transform()
        for i in range(16):
            tr.transform.m[i] = self.positionalTrackingFusionParameters.base_footprint_to_world_transform.m[i]
        return tr

    @base_footprint_to_world_transform.setter
    def base_footprint_to_world_transform(self, transform : Transform):
        self.positionalTrackingFusionParameters.base_footprint_to_world_transform = deref(transform.transform)

    ##
    # Position and orientation of the base footprint with respect to the baselink.
    # This transform represents a basis change from base footprint coordinate frame to baselink coordinate frame
    # 
    @property
    def base_footprint_to_baselink_transform(self) -> Transform:
        tr = Transform()
        for i in range(16):
            tr.transform.m[i] = self.positionalTrackingFusionParameters.base_footprint_to_baselink_transform.m[i]
        return tr

    @base_footprint_to_baselink_transform.setter
    def base_footprint_to_baselink_transform(self, transform : Transform):
        self.positionalTrackingFusionParameters.base_footprint_to_baselink_transform = deref(transform.transform)

    ##
    # Whether to override 2 of the 3 rotations from base_footprint_to_world_transform using the IMU gravity.
    #
    # Default: False
    @property
    def set_gravity_as_origin(self) -> bool:
        return self.positionalTrackingFusionParameters.set_gravity_as_origin

    @set_gravity_as_origin.setter
    def set_gravity_as_origin(self, value: bool):
        self.positionalTrackingFusionParameters.set_gravity_as_origin = value

    ##
    # ID of the camera used for positional tracking. If not specified, will use the first camera called with the subscribe() method.
    @property
    def tracking_camera_id(self) -> CameraIdentifier:
        tmp = CameraIdentifier()
        tmp.cameraIdentifier = self.positionalTrackingFusionParameters.tracking_camera_id
        return tmp

    @tracking_camera_id.setter
    def tracking_camera_id(self, CameraIdentifier value):
        self.positionalTrackingFusionParameters.tracking_camera_id = value.cameraIdentifier

##
# Holds the options used to initialize the body tracking module of the \ref Fusion.
# \ingroup Fusion_group
cdef class BodyTrackingFusionParameters:
    cdef c_BodyTrackingFusionParameters bodyTrackingFusionParameters

    ##
    # Defines if the object detection will track objects across images flow.
    #
    # Default: True
    @property
    def enable_tracking(self) -> bool:
        return self.bodyTrackingFusionParameters.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, value: bool):
        self.bodyTrackingFusionParameters.enable_tracking = value

    ##
    # Defines if the body fitting will be applied.
    #
    # Default: False
    # \note If you enable it and the camera provides data as BODY_18 the fused body format will be BODY_34.
    @property
    def enable_body_fitting(self) -> bool:
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
    # If the fused skeleton has less than skeleton_minimum_allowed_keypoints keypoints, it will be discarded.
    #
    # Default: -1.
    @property
    def skeleton_minimum_allowed_keypoints(self) -> int:
        return self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_keypoints

    @skeleton_minimum_allowed_keypoints.setter
    def skeleton_minimum_allowed_keypoints(self, value: int):
        self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_keypoints = value

    ##
    # If a skeleton was detected in less than skeleton_minimum_allowed_camera cameras, it will be discarded.
    #
    # Default: -1.
    @property
    def skeleton_minimum_allowed_camera(self) -> int:
        return self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_camera

    @skeleton_minimum_allowed_camera.setter
    def skeleton_minimum_allowed_camera(self, value: int):
        self.bodyTrackingFusionRuntimeParameters.skeleton_minimum_allowed_camera = value

    ##
    # This value controls the smoothing of the tracked or fitted fused skeleton.
    #
    # It is ranged from 0 (low smoothing) and 1 (high smoothing).
    # \n Default: 0.
    @property
    def skeleton_smoothing(self) -> float:
        return self.bodyTrackingFusionRuntimeParameters.skeleton_smoothing

    @skeleton_smoothing.setter
    def skeleton_smoothing(self, value: float):
        self.bodyTrackingFusionRuntimeParameters.skeleton_smoothing = value

##
# Holds the options used to initialize the object detection module of the Fusion
# \ingroup Fusion_group
cdef class ObjectDetectionFusionParameters:
    cdef c_ObjectDetectionFusionParameters objectDetectionFusionParameters

    ##
    # Defines if the object detection will track objects across images flow.
    #
    # Default: True.
    @property
    def enable_tracking(self) -> bool:
        return self.objectDetectionFusionParameters.enable_tracking

    @enable_tracking.setter
    def enable_tracking(self, value: bool):
        self.objectDetectionFusionParameters.enable_tracking = value

##
# Holds the metrics of a sender in the fusion process.
# \ingroup Fusion_group
cdef class CameraMetrics :
    cdef c_CameraMetrics cameraMetrics

    ##
    # FPS of the received data.
    @property
    def received_fps(self) -> float:
        return self.cameraMetrics.received_fps

    @received_fps.setter
    def received_fps(self, value: float):
        self.cameraMetrics.received_fps = value

    ##
    # Latency (in second) of the received data.
    # Timestamp difference between the time when the data are sent and the time they are received (mostly introduced when using the local network workflow).
    @property
    def received_latency(self) -> float:
        return self.cameraMetrics.received_latency

    @received_latency.setter
    def received_latency(self, value: float):
        self.cameraMetrics.received_latency = value

    ##
    # Latency (in seconds) after Fusion synchronization.
    # Difference between the timestamp of the data received and the timestamp at the end of the Fusion synchronization.
    @property
    def synced_latency(self) -> float:
        return self.cameraMetrics.synced_latency

    @synced_latency.setter
    def synced_latency(self, value: float):
        self.cameraMetrics.synced_latency = value

    ##
    # Is set to false if no data in this batch of metrics.
    @property
    def is_present(self) -> bool:
        return self.cameraMetrics.is_present

    @is_present.setter
    def is_present(self, value: bool):
        self.cameraMetrics.is_present = value

    ##
    # Skeleton detection percent during the last second.
    # Number of frames with at least one detection / number of frames, over the last second.
    # A low value means few detections occured lately for this sender.
    @property
    def ratio_detection(self) -> float:
        return self.cameraMetrics.ratio_detection

    @ratio_detection.setter
    def ratio_detection(self, value: float):
        self.cameraMetrics.ratio_detection = value

    ##
    # Average data acquisition timestamp difference.
    # Average standard deviation of sender's period since the start.
    @property
    def delta_ts(self) -> float:
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
    # Reset the current metrics.
    def reset(self):
        return self.fusionMetrics.reset()
    
    ##
    # Mean number of camera that provides data during the past second.
    @property
    def mean_camera_fused(self) -> float:
        return self.fusionMetrics.mean_camera_fused

    @mean_camera_fused.setter
    def mean_camera_fused(self, value: float):
        self.fusionMetrics.mean_camera_fused = value

    ##
    # Standard deviation of the data timestamp fused, the lower the better.
    @property
    def mean_stdev_between_camera(self) -> float:
        return self.fusionMetrics.mean_stdev_between_camera

    @mean_stdev_between_camera.setter
    def mean_stdev_between_camera(self, value: float):
        self.fusionMetrics.mean_stdev_between_camera = value

    ##
    # Sender metrics.
    @property
    def camera_individual_stats(self) -> dict:
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
        self.cameraIdentifier = c_CameraIdentifier(<int>serial_number)

    @property
    def serial_number(self) -> int:
        return self.cameraIdentifier.sn

    @serial_number.setter
    def serial_number(self, value: int):
        self.cameraIdentifier.sn = value

##
# Represents a world position in ECEF format.
# \ingroup Fusion_group
cdef class ECEF:
    cdef c_ECEF ecef

    ##
    # x coordinate of ECEF.
    @property
    def x(self) -> double:
        return self.ecef.x

    @x.setter
    def x(self, value: double):
        self.ecef.x = value

    ##
    # y coordinate of ECEF.
    @property
    def y(self) -> double:
        return self.ecef.y

    @y.setter
    def y(self, value: double):
        self.ecef.y = value

    ##
    # z coordinate of ECEF.
    @property
    def z(self) -> double:
        return self.ecef.z

    @z.setter
    def z(self, value: double):
        self.ecef.z = value

##
# Represents a world position in LatLng format.
# \ingroup Fusion_group
cdef class LatLng:
    cdef c_LatLng latLng

    ##
    # Get the latitude coordinate
    #
    # \param in_radian: Is the output should be in radian or degree.
    # \return Latitude in radian or in degree depending \ref in_radian parameter.
    def get_latitude(self, in_radian : bool = True):
        return self.latLng.getLatitude(in_radian)

    ##
    # Get the longitude coordinate
    #
    # \param in_radian: Is the output should be in radian or degree.
    # \return Longitude in radian or in degree depending \ref in_radian parameter.
    def get_longitude(self, in_radian=True):
        return self.latLng.getLongitude(in_radian)

    ##
    # Get the altitude coordinate
    #
    # \return Altitude coordinate in meters.
    def get_altitude(self):
        return self.latLng.getAltitude()
    
    ##
    # Get the coordinates in radians (default) or in degrees.
    #
    # \param latitude: Latitude coordinate.
    # \param longitude: Longitude coordinate.
    # \param altitude: Altitude coordinate.
    # \param in_radian: Should the output be expressed in radians or degrees.
    def get_coordinates(self, in_radian=True):
        cdef double lat = 0, lng = 0, alt = 0
        self.latLng.getCoordinates(lat, lng, alt, in_radian)
        return lat, lng , alt
    
    ##
    # Set the coordinates in radians (default) or in degrees.
    #
    # \param latitude: Latitude coordinate.
    # \param longitude: Longitude coordinate.
    # \param altitude: Altitude coordinate.
    # \@param in_radian: Is input are in radians or in degrees.
    def set_coordinates(self, latitude: double, longitude: double, altitude: double, in_radian=True):
        self.latLng.setCoordinates(latitude, longitude, altitude, in_radian)

##
# Represents a world position in UTM format.
# \ingroup Fusion_group
cdef class UTM:
    cdef c_UTM utm

    ##
    # Northing coordinate.
    @property
    def northing(self) -> double:
        return self.utm.northing

    @northing.setter
    def northing(self, value: double):
        self.utm.northing = value

    ##
    # Easting coordinate.
    @property
    def easting(self) -> double:
        return self.utm.easting

    @easting.setter
    def easting(self, value: double):
        self.utm.easting = value

    ##
    # Gamma coordinate.
    @property
    def gamma(self) -> double:
        return self.utm.gamma

    @gamma.setter
    def gamma(self, value: double):
        self.utm.gamma = value

    ##
    # UTMZone of the coordinate.
    @property
    def UTM_zone(self) -> str:
        return self.utm.UTMZone.decode()

    @UTM_zone.setter
    def UTM_zone(self, value: str):
        self.utm.UTMZone = value.encode('utf-8')

##
# Represent a world position in ENU format.
# \ingroup Fusion_group
cdef class ENU:
    cdef c_ENU enu
    
    ##
    # East parameter
    @property 
    def east(self) -> double:
        return self.enu.east
    @east.setter
    def east(self, value: double):
        self.enu.east = value
    
    ##
    # North parameter
    #
    @property
    def north(self) -> double:
        return self.enu.north
    @north.setter
    def north(self, value: double):
        self.enu.north = value
    
    ##
    # Up parameter
    #
    @property
    def up(self) -> double:
        return self.enu.up
    @up.setter
    def up(self, value: double):
        self.enu.up = value

##
# Purely static class for Geo functions.
# \ingroup Fusion_group
cdef class GeoConverter:
    ##
    # Convert ECEF coordinates to Lat/Long coordinates.
    @staticmethod
    def ecef2latlng(input: ECEF) -> LatLng:
        cdef c_LatLng temp
        c_GeoConverter.ecef2latlng(input.ecef, temp)
        result = LatLng()
        result.latLng = temp
        return result

    ##
    # Convert ECEF coordinates to UTM coordinates.
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
    # Convert Lat/Long coordinates to ECEF coordinates.
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
    # Convert Lat/Long coordinates to UTM coordinates.
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
    # Convert UTM coordinates to ECEF coordinates.
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
    # Convert UTM coordinates to Lat/Long coordinates.
    @staticmethod
    def utm2latlng(input: UTM) -> LatLng:
        cdef c_LatLng temp
        c_GeoConverter.utm2latlng(input.utm, temp)
        result = LatLng()
        result.latLng = temp
        return result

##
# Holds Geo reference position.
# \ingroup Fusion_group
# \brief Holds geographic reference position information.
#
# This class represents a geographic pose, including position, orientation, and accuracy information.
# It is used for storing and manipulating geographic data, such as latitude, longitude, altitude,
# pose matrices, covariances, and timestamps.
# 
# The pose data is defined in the East-North-Up (ENU) reference frame. The ENU frame is a local
# Cartesian coordinate system commonly used in geodetic applications. In this frame, the X-axis
# points towards the East, the Y-axis points towards the North, and the Z-axis points upwards.
cdef class GeoPose:
    cdef c_GeoPose geopose
    cdef Transform pose_data

    ##
    # Default constructor
    def __cinit__(self):
        self.geopose = c_GeoPose()
        self.pose_data = Transform()

    ##
    # The 4x4 matrix defining the pose in the East-North-Up (ENU) coordinate system.
    @property
    def pose_data(self) -> Transform:
        for i in range(16):
            self.pose_data.transform.m[i] = self.geopose.pose_data.m[i]

        return self.pose_data

    @pose_data.setter
    def pose_data(self, transform : Transform):
        self.geopose.pose_data = deref(transform.transform)

    ##
    # The pose covariance matrix in ENU.
    @property
    def pose_covariance(self) -> np.array[float]:
        cdef np.ndarray arr = np.zeros(36)
        for i in range(36) :
            arr[i] = self.geopose.pose_covariance[i]
        return arr

    @pose_covariance.setter
    def pose_covariance(self, np.ndarray pose_covariance_):
        for i in range(36) :
            self.geopose.pose_covariance[i] = pose_covariance_[i]

    ##
    # The horizontal accuracy of the pose in meters.
    @property
    def horizontal_accuracy(self) -> double:
        return self.geopose.horizontal_accuracy

    @horizontal_accuracy.setter
    def horizontal_accuracy(self, value: double):
        self.geopose.horizontal_accuracy = value

    ##
    # The vertical accuracy of the pose in meters.
    @property
    def vertical_accuracy(self) -> double:
        return self.geopose.vertical_accuracy

    @vertical_accuracy.setter
    def vertical_accuracy(self, value: double):
        self.geopose.vertical_accuracy = value

    ##
    # The latitude, longitude, and altitude coordinates of the pose.
    @property
    def latlng_coordinates(self) -> LatLng:
        result = LatLng()
        result.latLng = self.geopose.latlng_coordinates
        return result

    @latlng_coordinates.setter
    def latlng_coordinates(self, value: LatLng):
        self.geopose.latlng_coordinates = value.latLng

    ##
    # The heading (orientation) of the pose in radians (rad). It indicates the direction in which the object or observer is facing, with 0 degrees corresponding to North and increasing in a counter-clockwise direction.
    @property
    def heading(self) -> double:
        return self.geopose.heading

    @heading.setter
    def heading(self, value: double):
        self.geopose.heading = value

    ##
    # The timestamp associated with the GeoPose.
    @property
    def timestamp(self) -> Timestamp:
        timestamp = Timestamp()
        timestamp.timestamp = self.geopose.timestamp
        return  timestamp

    @timestamp.setter
    def timestamp(self, value: Timestamp):
        self.geopose.timestamp = value.timestamp

##
# Class containing GNSS data to be used for positional tracking as prior.
# \ingroup Sensors_group
cdef class GNSSData:

    cdef c_GNSSData gnss_data

    ##
    # Get the coordinates of the sl.GNSSData.
    # The sl.LatLng coordinates could be expressed in degrees or radians.
    # \param latitude: Latitude coordinate.
    # \param longitude: Longitude coordinate.
    # \param altitude: Altitude coordinate.
    # \param is_radian: Should the output be expressed in radians or degrees.
    def get_coordinates(self, in_radian=True) -> tuple(float, float, float):
        cdef double lat = 0, lng = 0, alt = 0
        self.gnss_data.getCoordinates(lat, lng, alt, in_radian)
        return lat, lng , alt
    
    ##
    # Set the sl.LatLng coordinates of sl.GNSSData.
    # The sl.LatLng coordinates could be expressed in degrees or radians.
    # \param latitude: Latitude coordinate.
    # \param longitude: Longitude coordinate.
    # \param altitude: Altitude coordinate.
    # \param is_radian: Are the inputs expressed in radians or in degrees.
    def set_coordinates(self, latitude: double, longitude: double, altitude: double, in_radian=True) -> None:
        self.gnss_data.setCoordinates(latitude, longitude, altitude, in_radian)

    ##
    # Latitude standard deviation.
    @property
    def latitude_std(self) -> float:
        return self.gnss_data.latitude_std

    @latitude_std.setter
    def latitude_std(self, value: double):
        self.gnss_data.latitude_std = value

    ##
    # Longitude standard deviation.
    @property
    def longitude_std(self) -> float:
        return self.gnss_data.longitude_std

    @longitude_std.setter
    def longitude_std(self, value: double):
        self.gnss_data.longitude_std = value

    ##
    # Altitude standard deviation
    @property
    def altitude_std(self) -> float:
        return self.gnss_data.altitude_std

    @altitude_std.setter
    def altitude_std(self, value: double):
        self.gnss_data.altitude_std = value

    ##
    # Timestamp of the GNSS position (must be aligned with the camera time reference).
    @property
    def ts(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp = self.gnss_data.ts
        return  ts

    @ts.setter
    def ts(self, value: Timestamp):
        self.gnss_data.ts = value.timestamp

    ##
    # Represents the current status of GNSS. 
    @property
    def gnss_status(self) -> GNSS_STATUS:
        return GNSS_STATUS(<int>self.gnss_data.gnss_status)

    @gnss_status.setter
    def gnss_status(self, gnss_status):
        self.gnss_data.gnss_status = (<c_GNSS_STATUS> (<int>gnss_status))

    ##
    # Represents the current mode of GNSS.
    @property
    def gnss_mode(self) -> GNSS_MODE:
        return GNSS_STATUS(<int>self.gnss_data.gnss_mode)

    @gnss_mode.setter
    def gnss_mode(self, gnss_mode):
        self.gnss_data.gnss_mode = (<c_GNSS_MODE> (<int>gnss_mode))

    ##
    # Covariance of the position in meter (must be expressed in the ENU coordinate system).
    # For eph, epv GNSS sensors, set it as follow: ```{eph*eph, 0, 0, 0, eph*eph, 0, 0, 0, epv*epv}```.
    @property
    def position_covariances(self) -> list[float]:
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


# @brief Configuration parameters for data synchronization.
#
# The SynchronizationParameter struct represents the configuration parameters used by the synchronizer. It allows customization
# of the synchronization process based on specific requirements.
cdef class SynchronizationParameter:
    cdef c_SynchronizationParameter synchronization_parameters

    #
    # @brief Default constructor
    #
    def __cinit__(self, windows_size : double = 0, data_source_timeout : double = 50, keep_last_data : bool = False, maximum_lateness : double = 50):
        self.synchronization_parameters.windows_size = windows_size
        self.synchronization_parameters.data_source_timeout = data_source_timeout
        self.synchronization_parameters.keep_last_data = keep_last_data
        self.synchronization_parameters.maximum_lateness = maximum_lateness

    # @brief Size of synchronization windows in milliseconds.
    #
    # The synchronization window is used by the synchronizer to return all data present inside the current
    # synchronization window. For efficient fusion, the synchronization window size is expected to be equal
    # to the period of the lowest source of data. If not provided, the fusion SDK will compute it from the
    # data's sources.
    @property
    def windows_size(self) -> double:
        return self.synchronization_parameters.windows_size

    @windows_size.setter
    def windows_size(self, value: double):
        self.synchronization_parameters.windows_size = value

    # @brief Duration in milliseconds before considering a data source inactive if no more data is present.
    #
    # The data_source_timeout parameter specifies the duration to wait before considering a data source as inactive
    # if no new data is received within the specified time frame.
    @property
    def data_source_timeout(self) -> double:
        return self.synchronization_parameters.data_source_timeout

    @data_source_timeout.setter
    def data_source_timeout(self, value:double):
        self.synchronization_parameters.data_source_timeout = value

    # @brief Determines whether to include the last data returned by a source in the final synchronized data.
    #
    # If the keep_last_data parameter is set to true and no data is present in the current synchronization window,
    # the last data returned by the source will be included in the final synchronized data. This ensures continuity
    # even in the absence of fresh data.
    @property
    def keep_last_data(self) -> bool:
        return self.synchronization_parameters.keep_last_data

    @keep_last_data.setter
    def keep_last_data(self, value: bool):
        self.synchronization_parameters.keep_last_data = value

    # @brief Maximum duration in milliseconds allowed for data to be considered as the last data.
    #
    # The maximum_lateness parameter sets the maximum duration within which data can be considered as the last
    # available data. If the duration between the last received data and the current synchronization window exceeds
    # this value, the data will not be included as the last data in the final synchronized output.
    @property
    def maximum_lateness(self) -> double:
        return self.synchronization_parameters.maximum_lateness

    @maximum_lateness.setter
    def maximum_lateness(self, value : double):
        self.synchronization_parameters.maximum_lateness = value


##
# Holds the options used to initialize the \ref Fusion object.
# \ingroup Fusion_group
cdef class InitFusionParameters:
    cdef c_InitFusionParameters* initFusionParameters

    def __cinit__(self,
                  coordinate_unit : UNIT = UNIT.MILLIMETER,
                  coordinate_system : COORDINATE_SYSTEM = COORDINATE_SYSTEM.IMAGE,
                  output_performance_metrics : bool = False,
                  verbose_ : bool = False,
                  timeout_period_number : int = 5,
                  sdk_gpu_id : int = -1,
                  synchronization_parameters : SynchronizationParameter = SynchronizationParameter(),
                  maximum_working_resolution : Resolution = Resolution(-1, -1)
    ):
        self.initFusionParameters = new c_InitFusionParameters(
            <c_UNIT>(<int>coordinate_unit.value), 
            <c_COORDINATE_SYSTEM>(<int>coordinate_system.value),
            output_performance_metrics, verbose_,
            timeout_period_number,
            sdk_gpu_id,
            <CUcontext>0,
            <c_SynchronizationParameter>(synchronization_parameters.synchronization_parameters),
            <c_Resolution>(maximum_working_resolution.resolution)
        )

    def __dealloc__(self):
        del self.initFusionParameters

    ##
    # This parameter allows you to select the unit to be used for all metric values of the SDK (depth, point cloud, tracking, mesh, and others).
    # Default : \ref UNIT "UNIT::MILLIMETER"
    @property
    def coordinate_units(self) -> UNIT:
        return UNIT(<int>self.initFusionParameters.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value: UNIT):
        self.initFusionParameters.coordinate_units = <c_UNIT>(<int>value.value)

    ##
    # Positional tracking, point clouds and many other features require a given \ref COORDINATE_SYSTEM to be used as reference.
    # This parameter allows you to select the \ref COORDINATE_SYSTEM used by the \ref Camera to return its measures.
    # \n This defines the order and the direction of the axis of the coordinate system.
    # \n Default : \ref COORDINATE_SYSTEM "COORDINATE_SYSTEM::IMAGE"
    @property
    def coordinate_system(self) -> COORDINATE_SYSTEM:
        return COORDINATE_SYSTEM(<int>self.initFusionParameters.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value: COORDINATE_SYSTEM):
        self.initFusionParameters.coordinate_system = <c_COORDINATE_SYSTEM>(<int>value.value)

    ##
    # It allows users to extract some stats of the Fusion API like drop frame of each camera, latency, etc...
    @property
    def output_performance_metrics(self) -> bool:
        return self.initFusionParameters.output_performance_metrics

    @output_performance_metrics.setter
    def output_performance_metrics(self, value: bool):
        self.initFusionParameters.output_performance_metrics = value

    ##
    # Enable the verbosity mode of the SDK.
    #
    @property
    def verbose(self) -> bool:
        return self.initFusionParameters.verbose

    @verbose.setter
    def verbose(self, value: bool):
        self.initFusionParameters.verbose = value

    ##
    # If specified change the number of period necessary for a source to go in timeout without data. For example, if you set this to 5 then, if any source do not receive data during 5 period, these sources will go to timeout and will be ignored.
    #
    @property
    def timeout_period_number(self) -> int:
        return self.initFusionParameters.timeout_period_number

    @timeout_period_number.setter
    def timeout_period_number(self, value: int):
        self.initFusionParameters.timeout_period_number = value

    ##
    # NVIDIA graphics card id to use.
    #
    # By default the SDK will use the most powerful NVIDIA graphics card found.
    # \n However, when running several applications, or using several cameras at the same time, splitting the load over available GPUs can be useful.
    # \n This parameter allows you to select the GPU used by the sl.Camera using an ID from 0 to n-1 GPUs in your PC.
    # \n Default: -1
    # \note A non-positive value will search for all CUDA capable devices and select the most powerful.
    @property
    def sdk_gpu_id(self) -> int:
        return self.init.sdk_gpu_id

    @sdk_gpu_id.setter
    def sdk_gpu_id(self, value: int):
        self.init.sdk_gpu_id = value

    ##
    # Specifies the parameters used for data synchronization during fusion.
    #
    # The SynchronizationParameter struct encapsulates the synchronization parameters that control the data fusion process.
    #
    @property
    def synchronization_parameters(self) -> SynchronizationParameter:
        sp = SynchronizationParameter()
        sp.synchronization_parameters = self.initFusionParameters.synchronization_parameters
        return sp

    @synchronization_parameters.setter
    def synchronization_parameters(self, value: SynchronizationParameter):
        self.initFusionParameters.synchronization_parameters = value.synchronization_parameters

    ##
    #  Sets the maximum resolution for all Fusion outputs, such as images and measures.
    # 
    # The default value is (-1, -1), which allows the Fusion to automatically select the optimal resolution for the best quality/runtime ratio.
    # 
    # - For images, the output resolution can be up to the native resolution of the camera.
    # - For measures involving depth, the output resolution can be up to the maximum working resolution.
    # 
    # Setting this parameter to (-1, -1) will ensure the best balance between quality and performance for depth measures.
    #
    @property
    def maximum_working_resolution(self) -> Resolution:
        return Resolution(self.initFusionParameters.maximum_working_resolution.width, self.initFusionParameters.maximum_working_resolution.height)

    @maximum_working_resolution.setter
    def maximum_working_resolution(self, Resolution value):
        self.initFusionParameters.maximum_working_resolution = c_Resolution(value.width, value.height)

##
# Holds Fusion process data and functions
# \ingroup Fusion_group
cdef class Fusion:
    cdef c_Fusion* fusion

    def __cinit__(self):
        self.fusion = new c_Fusion()

    def __dealloc__(self):
        if self.fusion != NULL:
            del self.fusion

    ##
    # Initialize the fusion module with the requested parameters.
    # \param init_parameters: Initialization parameters.
    # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an ERROR_CODE.
    def init(self, init_fusion_parameters : InitFusionParameters) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.init(deref(init_fusion_parameters.initFusionParameters)))

    ##
    # Will deactivate all the fusion modules and free internal data.
    def close(self) -> None:
        self.fusion.close()

    ##
    # Set the specified camera as a data provider.
    # \param uuid: The requested camera identifier.
    # \param communication_parameters: The communication parameters to connect to the camera.
    # \param pose: The World position of the camera, regarding the other camera of the setup.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def subscribe(self, uuid : CameraIdentifier, communication_parameters: CommunicationParameters, pose: Transform) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.subscribe(uuid.cameraIdentifier, communication_parameters.communicationParameters, deref(pose.transform)))

    ##
    # Remove the specified camera from data provider.
    # \param uuid: The requested camera identifier.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def unsubscribe(self, uuid : CameraIdentifier) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.unsubscribe(uuid.cameraIdentifier))

    ##
    # Updates the specified camera position inside fusion WORLD.
    # \param uuid: The requested camera identifier.
    # \param pose: The World position of the camera, regarding the other camera of the setup.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def update_pose(self, uuid : CameraIdentifier, pose: Transform) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.updatePose(uuid.cameraIdentifier, deref(pose.transform)))

    ##
    # Get the metrics of the Fusion process, for the fused data as well as individual camera provider data.
    # \param metrics: The process metrics.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    # \return The process metrics.
    def get_process_metrics(self) -> tuple(FUSION_ERROR_CODE, FusionMetrics):
        cdef c_FusionMetrics temp_fusion_metrics
        err = FUSION_ERROR_CODE(<int>self.fusion.getProcessMetrics(temp_fusion_metrics))
        metrics = FusionMetrics()
        metrics.fusionMetrics = temp_fusion_metrics
        return err, metrics

    ##
    # Returns the state of each connected data senders.
    # \return The individual state of each connected senders.
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
    # Runs the main function of the Fusion, this trigger the retrieve and synchronization of all connected senders and updates the enabled modules.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def process(self) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        with nogil:
            ret = self.fusion.process()
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Enables the body tracking fusion module.
    # \param params: Structure containing all specific parameters for body tracking fusion.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def enable_body_tracking(self, params : BodyTrackingFusionParameters) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.enableBodyTracking(params.bodyTrackingFusionParameters))

    ##
    # Retrieves the body data, can be the fused data (default), or the raw data provided by a specific sender.
    # \param bodies: The fused bodies will be saved into this objects.
    # \param parameters: Body detection runtime settings, can be changed at each detection.
    # \param uuid: The id of the sender.
    # \param reference_frame: The reference frame in which the objects will be expressed. Default: \ref FUSION_REFERENCE_FRAME "FUSION_REFERENCE_FRAME::BASELINK".
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def retrieve_bodies(self, bodies : Bodies, parameters : BodyTrackingFusionRuntimeParameters, uuid : CameraIdentifier = CameraIdentifier(0), reference_frame: FUSION_REFERENCE_FRAME = FUSION_REFERENCE_FRAME.BASELINK) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        cdef c_FUSION_REFERENCE_FRAME c_reference_frame = <c_FUSION_REFERENCE_FRAME>(<int>reference_frame.value)
        with nogil:
            ret = self.fusion.retrieveBodies(bodies.bodies, parameters.bodyTrackingFusionRuntimeParameters, uuid.cameraIdentifier, c_reference_frame)
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Enables the object detection fusion module.
    # \param params: Structure containing all specific parameters for object detection fusion.
    # \n For more information, see the \ref ObjectDetectionFusionParameters documentation.
    # \return \ref FUSION_ERROR_CODE "SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def enable_object_detection(self, ObjectDetectionFusionParameters params = ObjectDetectionFusionParameters()) -> FUSION_ERROR_CODE:
        return _fusion_error_code_cache.get(<int>self.fusion.enableObjectDetection(params.objectDetectionFusionParameters), FUSION_ERROR_CODE.FAILURE)

    ##
    # Retrieves all the fused objects data.
    # \param objs: The fused objects will be saved into this dictionary of objects.
    # \param reference_frame: The reference frame in which the objects will be expressed. Default: \ref FUSION_REFERENCE_FRAME "FUSION_REFERENCE_FRAME::BASELINK".
    # \return \ref FUSION_ERROR_CODE "SUCCESS" if it goes as it should, otherwise it returns a FUSION_ERROR_CODE.
    def retrieve_objects_all_od_groups(self, dict objs, reference_frame: FUSION_REFERENCE_FRAME = FUSION_REFERENCE_FRAME.BASELINK) -> FUSION_ERROR_CODE:
        cdef unordered_map[String, c_Objects] map_obj
        cdef c_FUSION_ERROR_CODE ret
        cdef c_FUSION_REFERENCE_FRAME c_reference_frame = <c_FUSION_REFERENCE_FRAME>(<int>reference_frame.value)
        with nogil:
            ret = self.fusion.retrieveObjectsAllODGroups(map_obj, c_reference_frame)
        py_ret = _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)
        if py_ret == FUSION_ERROR_CODE.SUCCESS:
            objs.clear()
            for item in map_obj:
                py_objects = Objects()
                py_objects.objects = item.second
                key_string = to_str(item.first).decode()
                objs[key_string] = py_objects
        return py_ret

    ##
    # Retrieves the fused objects of a given fused OD group.
    # \param objs: The fused objects will be saved into this objects.
    # \param fused_od_group_name: The name of the fused objects group to retrieve.
    # \param reference_frame: The reference frame in which the objects will be expressed. Default: \ref FUSION_REFERENCE_FRAME "FUSION_REFERENCE_FRAME::BASELINK".
    # \return \ref FUSION_ERROR_CODE "SUCCESS" if it goes as it should, otherwise it returns a FUSION_ERROR_CODE.
    def retrieve_objects_one_od_group(self, Objects objs, str fused_od_group_name, reference_frame: FUSION_REFERENCE_FRAME = FUSION_REFERENCE_FRAME.BASELINK) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        cdef c_FUSION_REFERENCE_FRAME c_reference_frame = <c_FUSION_REFERENCE_FRAME>(<int>reference_frame.value)
        cdef String c_fused_od_group_name_str = String(fused_od_group_name.encode())
        with nogil:
            ret = self.fusion.retrieveObjectsOneODGroup(objs.objects, c_fused_od_group_name_str, c_reference_frame)
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Retrieves all the raw objects data provided by a specific sender.
    # \param objs: The fused objects will be saved into this dictionary of objects.
    # \param uuid: Retrieve the raw data provided by this sender.
    def retrieve_raw_objects_all_ids(self, dict objs, CameraIdentifier uuid) -> FUSION_ERROR_CODE:
        cdef unordered_map[unsigned int, c_Objects] umap
        cdef c_FUSION_ERROR_CODE ret
        with nogil:
            ret = self.fusion.retrieveObjectsAllIds(umap, uuid.cameraIdentifier)
        py_ret = _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)
        if py_ret == FUSION_ERROR_CODE.SUCCESS:
            objs.clear()
            for item in umap:
                py_objects = Objects()
                py_objects.objects = item.second
                objs[item.first] = py_objects
        return py_ret

    ##
    # Retrieves the raw objects data provided by a specific sender and a specific instance id.
    # \param objs: The fused objects will be saved into this objects.
    # \param uuid: Retrieve the raw data provided by this sender.
    # \param instance_id: Retrieve the objects inferred by the model with this ID only.
    # \return \ref FUSION_ERROR_CODE "SUCCESS" if it goes as it should, otherwise it returns a FUSION_ERROR_CODE.
    def retrieve_raw_objects_one_id(self, Objects py_objects, CameraIdentifier uuid, uint instance_id) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        with nogil:
            ret = self.fusion.retrieveObjectsOneId(py_objects.objects, uuid.cameraIdentifier, instance_id)
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Disable the body fusion tracking module.
    def disable_objects_detection(self) -> None:
        self.disableObjectDetection()

    ##
    # Returns the current sl.VIEW.LEFT of the specified camera, the data is synchronized.
    # \param mat: the CPU BGRA image of the requested camera.
    # \param resolution: the requested resolution of the output image, can be lower or equal (default) to the original image resolution.
    # \param uuid: If set to a sender serial number (different from 0), this will retrieve the raw data provided by this sender.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def retrieve_image(self, Mat mat, CameraIdentifier uuid, Resolution resolution = Resolution(0,0)) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        with nogil:
            ret = self.fusion.retrieveImage(mat.mat, uuid.cameraIdentifier, resolution.resolution)
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Returns the current measure of the specified camera, the data is synchronized.
    # \param mat: the CPU data of the requested camera.
    # \param uuid: The id of the sender.
    # \param measure: measure: the requested measure type, by default DEPTH (F32_C1).
    # \param resolution: the requested resolution of the output image, can be lower or equal (default) to the original image resolution.
    # \param reference_frame: The reference frame in which the objects will be expressed. Default: \ref FUSION_REFERENCE_FRAME "FUSION_REFERENCE_FRAME::BASELINK".
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def retrieve_measure(self, Mat mat, CameraIdentifier uuid, measure: MEASURE, Resolution resolution = Resolution(0,0), reference_frame: FUSION_REFERENCE_FRAME = FUSION_REFERENCE_FRAME.BASELINK) -> FUSION_ERROR_CODE:
        cdef c_FUSION_ERROR_CODE ret
        cdef c_MEASURE c_measure = <c_MEASURE>(<int>measure.value)
        cdef c_FUSION_REFERENCE_FRAME c_reference_frame = <c_FUSION_REFERENCE_FRAME>(<int>reference_frame.value)
        with nogil:
            ret = self.fusion.retrieveMeasure(mat.mat, uuid.cameraIdentifier, c_measure, resolution.resolution, c_reference_frame)
        return _fusion_error_code_cache.get(<int>ret, FUSION_ERROR_CODE.FAILURE)

    ##
    # Disable the body fusion tracking module.
    def disable_body_tracking(self) -> None:
        self.fusion.disableBodyTracking()

    ##
    # Enables positional tracking fusion module.
    # \param parameters: A structure containing all the \ref PositionalTrackingFusionParameters that define positional tracking fusion module.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def enable_positionnal_tracking(self, parameters : PositionalTrackingFusionParameters) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.enablePositionalTracking(parameters.positionalTrackingFusionParameters))
    
    ##
    # Ingest GNSS data from an external sensor into the fusion module.
    # \param gnss_data: The current GNSS data to combine with the current positional tracking data.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if it goes as it should, otherwise it returns an FUSION_ERROR_CODE.
    def ingest_gnss_data(self, gnss_data : GNSSData) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.ingestGNSSData(gnss_data.gnss_data))

    ##
    # Get the Fused Position referenced to the first camera subscribed. If \ref uuid is specified then project position on the referenced camera.
    # \param camera_pose: Will contain the fused position referenced by default in world (world is given by the calibration of the cameras system).
    # \param reference_frame: Defines the reference from which you want the pose to be expressed. Default : \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD".
    # \param uuid: If set to a sender serial number (different from 0), this will retrieve position projected on the requested camera if \ref position_type is equal to \ref POSITION_TYPE "POSITION_TYPE.FUSION" or raw sender position if \ref position_type is equal to \ref POSITION_TYPE "POSITION_TYPE.RAW".
    # \param position_type: Select if the position should the fused position re-projected in the camera with uuid or if the position should be the raw position (without fusion) of camera with uui.
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process.
    def get_position(self, camera_pose : Pose, reference_frame : REFERENCE_FRAME = REFERENCE_FRAME.WORLD, uuid: CameraIdentifier = CameraIdentifier(), position_type : POSITION_TYPE = POSITION_TYPE.FUSION) -> POSITIONAL_TRACKING_STATE:
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.getPosition(camera_pose.pose, <c_REFERENCE_FRAME>(<int>reference_frame.value), uuid.cameraIdentifier, <c_POSITION_TYPE>(<int>position_type.value)))

    def get_fused_positional_tracking_status(self) -> FusedPositionalTrackingStatus:
        status = FusedPositionalTrackingStatus()
        status.odometry_status = self.fusion.getFusedPositionalTrackingStatus().odometry_status
        status.spatial_memory_status = self.fusion.getFusedPositionalTrackingStatus().spatial_memory_status
        status.gnss_status = self.fusion.getFusedPositionalTrackingStatus().gnss_status
        status.gnss_mode = self.fusion.getFusedPositionalTrackingStatus().gnss_mode
        status.gnss_fusion_status = self.fusion.getFusedPositionalTrackingStatus().gnss_fusion_status
        return status

    ##
    # Returns the last synchronized gnss data.
    # \param out [out]: Last synchronized gnss data.
    # \return POSITIONAL_TRACKING_STATE is the current state of the tracking process.
    def get_current_gnss_data(self, gnss_data : GNSSData) -> POSITIONAL_TRACKING_STATE:
        return POSITIONAL_TRACKING_STATE(<int>self.fusion.getCurrentGNSSData(gnss_data.gnss_data))

    ##
    # Returns the current GeoPose.
    # \param pose [out]: The current GeoPose.
    # \return GNSS_FUSION_STATUS is the current state of the tracking process.
    def get_geo_pose(self, pose : GeoPose) -> GNSS_FUSION_STATUS:
        return GNSS_FUSION_STATUS(<int>self.fusion.getGeoPose(pose.geopose))

    ##
    # Convert latitude / longitude into position in sl::Fusion coordinate system.
    # \param input  [in]: The latitude / longitude to be converted in sl::Fusion coordinate system.
    # \param out [out]: Converted position in sl.Fusion coordinate system.
    # \return GNSS_FUSION_STATUS is the current state of the tracking process.
    def geo_to_camera(self, input : LatLng, output : Pose) -> GNSS_FUSION_STATUS:
        return GNSS_FUSION_STATUS(<int>self.fusion.Geo2Camera(input.latLng, output.pose))

    ##
    # Convert a position in sl.Fusion coordinate system in global world coordinate.
    # \param pose [in]: Position to convert in global world coordinate. 
    # \param pose [out]: Converted position in global world coordinate. 
    # \return GNSS_FUSION_STATUS is the current state of the tracking process.
    def camera_to_geo(self, input : Pose, output : GeoPose) -> GNSS_FUSION_STATUS:
        return GNSS_FUSION_STATUS(<int>self.fusion.Camera2Geo(input.pose, output.geopose))

    ##
    # Disable the fusion positional tracking module.
    #
    # The positional tracking is immediately stopped. If a file path is given, saveAreaMap(area_file_path) will be called asynchronously. See getAreaExportState() to get the exportation state.
    def disable_positionnal_tracking(self) -> None:
        self.fusion.disablePositionalTracking()

    ##
    # Convert ENU to LatLng
    #
    # Concert an ENU position into LatLng
    def ENU_to_geo(self, input: ENU, output: LatLng) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.ENU2Geo(input.enu, output.latLng))

    ##
    # Convert LatLng to ENU
    #
    # Convert am LatLng to ENU
    def geo_to_ENU(self, input : LatLng, out : ENU) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int> self.fusion.Geo2ENU(input.latLng, out.enu))


    ##
    # Get the current calibration uncertainty obtained during calibration process.
    # \return sl.GNSS_FUSION_STATUS representing current initialisation status.
    # \return Output yaw uncertainty.
    # \return Output position uncertainty.
    ##
    def get_current_gnss_calibration_std(self) -> tuple[GNSS_FUSION_STATUS, float, np.array]:
        cdef float3 position_std = float3(0, 0, 0)
        cdef float yaw_std = 0
        gnss_fusion_status = GNSS_FUSION_STATUS(<int>self.fusion.getCurrentGNSSCalibrationSTD(yaw_std, position_std))
        position_std_out = np.array([0,0,0], dtype=np.float64)
        position_std_out[0] = position_std[0]
        position_std_out[1] = position_std[1]
        position_std_out[2] = position_std[2]
        return gnss_fusion_status, yaw_std, position_std_out

    ##
    # Get the calibration found between VIO and GNSS.
    #
    # \return sl.Transform is the calibration found between VIO and GNSS during calibration process.
    ## 
    def get_geo_tracking_calibration(self) -> Transform:
        # cdef c_Transform tmp
        tf_out = Transform()
        tmp = <c_Transform>(self.fusion.getGeoTrackingCalibration())
        for i in range(16):
            tf_out.transform.m[i] = tmp.m[i]
        return tf_out

    ##
    # Starts the spatial map generation process in a non blocking thread from the spatial mapping process.
    #
    # The spatial map generation can take a long time depending on the mapping resolution and covered area. This function will trigger the generation of a mesh without blocking the program.
    # You can get info about the current generation using \ref get_spatial_map_request_status_async(), and retrieve the mesh using \ref request_spatial_map_async(...) .
    #
    # \note Only one mesh can be generated at a time. If the previous mesh generation is not over, new calls of the function will be ignored.
    ##
    def request_spatial_map_async(self) -> None:
        self.camera.requestSpatialMapAsync()

    ##
    # Returns the spatial map generation status. This status allows to know if the mesh can be retrieved by calling \ref retrieve_spatial_map_async().
    # \return \ref FUSION_ERROR_CODE "SUCCESS" if the mesh is ready and not yet retrieved, otherwise \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE::FAILURE".
    #
    # \n See \ref request_spatial_map_async() for an example.
    ##
    def get_spatial_map_request_status_async(self) -> FUSION_ERROR_CODE:
        return FUSION_ERROR_CODE(<int>self.fusion.getSpatialMapRequestStatusAsync())

    ##
    # Retrieves the current generated spatial map.
    #
    # After calling \ref request_spatial_map_async(), this method allows you to retrieve the generated mesh or fused point cloud.
    # \n The \ref Mesh or \ref FusedPointCloud will only be available when \ref get_spatial_map_request_status_async() returns \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS".
    #
    # \param py_mesh[out] : The \ref Mesh or \ref FusedPointCloud to be filled with the generated spatial map.
    # \return \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.SUCCESS" if the mesh is retrieved, otherwise \ref FUSION_ERROR_CODE "FUSION_ERROR_CODE.FAILURE".
    #
    # \note This method only updates the necessary chunks and adds the new ones in order to improve update speed.
    # \warning You should not modify the mesh / fused point cloud between two calls of this method, otherwise it can lead to a corrupted mesh / fused point cloud.
    # See \ref request_spatial_map_async() for an example.
    def retrieve_spatial_map_async(self, py_mesh) -> FUSION_ERROR_CODE:
        if isinstance(py_mesh, Mesh) :
            return _error_code_cache.get(<int>self.fusion.retrieveSpatialMapAsync(deref((<Mesh>py_mesh).mesh)), FUSION_ERROR_CODE.FAILURE)
        elif isinstance(py_mesh, FusedPointCloud) :
            py_mesh = <FusedPointCloud> py_mesh
            return _error_code_cache.get(<int>self.fusion.retrieveSpatialMapAsync(deref((<FusedPointCloud>py_mesh).fpc)), FUSION_ERROR_CODE.FAILURE)
        else :
           raise TypeError("Argument is not of Mesh or FusedPointCloud type.")

##
# Class containing SVO data to be ingested/retrieved to/from SVO.
# \ingroup Core_group
cdef class SVOData:

    cdef c_SVOData svo_data

    ##
    # Get the content of the sl.SVOData as a string.
    #
    # \return The content of the sl.SVOData as a string.
    def get_content_as_string(self) -> str:
        cdef string input
        self.svo_data.getContent(input)
        return input.decode()

    ##
    # Set the content of the sl.SVOData as a string.
    #
    # \param data The string data content to set.
    def set_string_content(self, data: str) -> str:
        self.svo_data.setContent(data.encode('utf-8'))

    ##
    # Timestamp of the data.
    @property
    def timestamp_ns(self) -> Timestamp:
        ts = Timestamp()
        ts.timestamp=self.svo_data.timestamp_ns
        return ts

    @timestamp_ns.setter
    def timestamp_ns(self, timestamp : Timestamp):
        self.svo_data.timestamp_ns.data_ns = timestamp.get_nanoseconds()

    ##
    # Key of the data.
    @property
    def key(self) -> str:
        return self.svo_data.key.decode()

    @key.setter
    def key(self, key_value: str):
        self.svo_data.key = key_value.encode('utf-8')

IF UNAME_SYSNAME == u"Linux":

    ##
    # Structure containing information about the camera sensor. 
    # \ingroup Core_group
    # 
    # Information about the camera is available in the sl.CameraInformation struct returned by sl.Camera.get_camera_information().
    # \note This object is meant to be used as a read-only container, editing any of its field won't impact the SDK.
    # \warning sl.CalibrationOneParameters are returned in sl.COORDINATE_SYSTEM.IMAGE, they are not impacted by the sl.InitParametersOne.coordinate_system.
    cdef class CameraOneConfiguration:
        cdef CameraParameters py_calib
        cdef CameraParameters py_calib_raw
        cdef unsigned int firmware_version
        cdef c_Resolution py_res
        cdef float camera_fps

        def __cinit__(self, CameraOne py_camera, Resolution resizer=Resolution(0,0), int firmware_version_=0, int fps_=0, CameraParameters py_calib_= CameraParameters(), CameraParameters py_calib_raw_= CameraParameters()):
            res = c_Resolution(resizer.width, resizer.height)
            self.py_calib = CameraParameters()
            caminfo = py_camera.camera.getCameraInformation(res)
            camconfig = caminfo.camera_configuration
            self.py_calib.camera_params = camconfig.calibration_parameters
            self.py_calib_raw = CameraParameters()
            self.py_calib_raw.camera_params = camconfig.calibration_parameters_raw
            self.firmware_version = camconfig.firmware_version
            self.py_res = camconfig.resolution
            self.camera_fps = camconfig.fps

        ##
        # Resolution of the camera.
        @property
        def resolution(self) -> Resolution:
            return Resolution(self.py_res.width, self.py_res.height)

        ##
        # FPS of the camera.
        @property
        def fps(self) -> float:
            return self.camera_fps

        ##
        # Intrinsics and extrinsic stereo parameters for rectified/undistorted images.
        @property
        def calibration_parameters(self) -> CameraParameters:
            return self.py_calib

        ##
        # Intrinsics and extrinsic stereo parameters for unrectified/distorted images.
        @property
        def calibration_parameters_raw(self) -> CameraParameters:
            return self.py_calib_raw

        ##
        # Internal firmware version of the camera.
        @property
        def firmware_version(self) -> int:
            return self.firmware_version


    ##
    # Structure containing information of a single camera (serial number, model, calibration, etc.)
    # \ingroup Core_group
    # That information about the camera will be returned by \ref CameraOne.get_camera_information()
    # \note This object is meant to be used as a read-only container, editing any of its fields won't impact the SDK.
    # \warning \ref CalibrationParameters are returned in \ref COORDINATE_SYSTEM.IMAGE , they are not impacted by the \ref InitParametersOne.coordinate_system
    cdef class CameraOneInformation:
        cdef unsigned int serial_number
        cdef c_MODEL camera_model
        cdef c_INPUT_TYPE input_type
        cdef CameraOneConfiguration py_camera_configuration
        cdef SensorsConfiguration py_sensors_configuration
        
        ##
        # Default constructor.
        # Gets the sl.CameraParameters from a sl.CameraOne object.
        # \param py_camera : sl.CameraOne object.
        # \param resizer : You can specify a sl.Resolution different from default image size to get the scaled camera information. Default: (0, 0) (original image size)
        #
        # \code
        # cam = sl.CameraOne()
        # res = sl.Resolution(0,0)
        # cam_info = sl.CameraInformation(cam, res)
        # \endcode
        def __cinit__(self, py_camera: CameraOne, resizer=Resolution(0,0)) -> CameraOneInformation:
            res = c_Resolution(resizer.width, resizer.height)
            caminfo = py_camera.camera.getCameraInformation(res)

            self.serial_number = caminfo.serial_number
            self.camera_model = caminfo.camera_model
            self.py_camera_configuration = CameraOneConfiguration(py_camera, resizer)
            self.py_sensors_configuration = SensorsConfiguration(py_camera, resizer)
            self.input_type = caminfo.input_type

        ##
        # Sensors configuration parameters stored in a sl.SensorsConfiguration.
        @property
        def sensors_configuration(self) -> SensorsConfiguration:
            return self.py_sensors_configuration

        ##
        # Camera configuration parameters stored in a sl.CameraOneConfiguration.
        @property
        def camera_configuration(self) -> CameraOneConfiguration:
            return self.py_camera_configuration

        ##
        # Input type used in the ZED SDK.
        @property
        def input_type(self) -> INPUT_TYPE:
            return INPUT_TYPE(<int>self.input_type)

        ##
        # Model of the camera (see sl.MODEL).
        @property
        def camera_model(self) -> MODEL:
            return MODEL(<int>self.camera_model)

        ##
        # Serial number of the camera.
        @property
        def serial_number(self) -> int:
            return self.serial_number

    ##
    # Class containing the options used to initialize the sl.CameraOne object.
    # \ingroup Video_group
    # 
    # This class allows you to select multiple parameters for the sl.Camera such as the selected camera, resolution, depth mode, coordinate system, and units of measurement.
    # \n Once filled with the desired options, it should be passed to the sl.Camera.open() method.
    #
    # \code
    #
    #        import pyzed.sl as sl
    #
    #        def main() :
    #            zed = sl.CameraOne() # Create a ZED camera object
    #
    #            init_params = sl.InitParametersOne()   # Set initial parameters
    #            init_params.sdk_verbose = 0  # Disable verbose mode
    #
    #            # Use the camera in LIVE mode
    #            init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD1080 video mode
    #            init_params.camera_fps = 30 # Set fps at 30
    #
    #            # Or use the camera in SVO (offline) mode
    #            #init_params.set_from_svo_file("xxxx.svo")
    #
    #            # Or use the camera in STREAM mode
    #            #init_params.set_from_stream("192.168.1.12", 30000)
    #
    #            # Other parameters are left to their default values
    #
    #            # Open the camera
    #            err = zed.open(init_params)
    #            if err != sl.ERROR_CODE.SUCCESS:
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
    # With its default values, it opens the camera in live mode at \ref RESOLUTION "sl.RESOLUTION.HD720"
    # \n You can customize it to fit your application.
    # \note The parameters can also be saved and reloaded using its \ref save() and \ref load() methods.
    cdef class InitParametersOne:
        cdef c_InitParametersOne init
        ##
        # Default constructor.
        #
        # All the parameters are set to their default and optimized values.
        # \param camera_resolution : Chosen \ref camera_resolution
        # \param camera_fps : Chosen \ref camera_fps
        # \param svo_real_time_mode : Activates \ref svo_real_time_mode
        # \param coordinate_units : Chosen \ref coordinate_units
        # \param coordinate_system : Chosen \ref coordinate_system
        # \param sdk_verbose : Sets \ref sdk_verbose
        # \param sdk_verbose_log_file : Chosen \ref sdk_verbose_log_file
        # \param input_t : Chosen input_t (\ref InputType )
        # \param optional_settings_path : Chosen \ref optional_settings_path
        # \param sensors_required : Activates \ref sensors_required
        # \param optional_opencv_calibration_file : Sets \ref optional_opencv_calibration_file
        # \param async_grab_camera_recovery : Sets \ref async_grab_camera_recovery
        # \param grab_compute_capping_fps : Sets \ref grab_compute_capping_fps
        # \param enable_image_validity_check : Sets \ref enable_image_validity_check
    
        ##
        # Desired camera resolution.
        # \note Small resolutions offer higher framerate and lower computation time.
        # \note In most situations, \ref RESOLUTION "sl.RESOLUTION.HD720" at 60 FPS is the best balance between image quality and framerate.
        #
        # Default: <ul>
        # <li>ZED X/X Mini: \ref RESOLUTION "sl.RESOLUTION.HD1200"</li>
        # <li>other cameras: \ref RESOLUTION "sl.RESOLUTION.HD720"</li></ul>
        # \note Available resolutions are listed here: sl.RESOLUTION.
        @property
        def camera_resolution(self) -> RESOLUTION:
            return RESOLUTION(<int>self.init.camera_resolution)

        @camera_resolution.setter
        def camera_resolution(self, value):
            if isinstance(value, RESOLUTION):
                self.init.camera_resolution = <c_RESOLUTION>(<int>value.value)
            else:
                raise TypeError("Argument must be of RESOLUTION type.")

        ##
        # Requested camera frame rate.
        #
        # If set to 0, the highest FPS of the specified \ref camera_resolution will be used.
        # \n Default: 0
        # \n\n See sl.RESOLUTION for a list of supported frame rates.
        # \note If the requested \ref camera_fps is unsupported, the closest available FPS will be used.
        @property
        def camera_fps(self) -> int:
            return self.init.camera_fps

        @camera_fps.setter
        def camera_fps(self, int value):
            self.init.camera_fps = value

        ##
        # Defines if sl.Camera object return the frame in real time mode.
        #
        # When playing back an SVO file, each call to sl.Camera.grab() will extract a new frame and use it.
        # \n However, it ignores the real capture rate of the images saved in the SVO file.
        # \n Enabling this parameter will bring the SDK closer to a real simulation when playing back a file by using the images' timestamps.
        # \n Default: False
        # \note sl.Camera.grab() will return an error when trying to play too fast, and frames will be dropped when playing too slowly.
        @property
        def svo_real_time_mode(self) -> bool:
            return self.init.svo_real_time_mode

        @svo_real_time_mode.setter
        def svo_real_time_mode(self, value: bool):
            self.init.svo_real_time_mode = value

        ##
        # Unit of spatial data (depth, point cloud, tracking, mesh, etc.) for retrieval.
        #
        # Default: \ref UNIT "sl.UNIT.MILLIMETER"
        @property
        def coordinate_units(self) -> UNIT:
            return UNIT(<int>self.init.coordinate_units)

        @coordinate_units.setter
        def coordinate_units(self, value):
            if isinstance(value, UNIT):
                self.init.coordinate_units = <c_UNIT>(<int>value.value)
            else:
                raise TypeError("Argument must be of UNIT type.")

        ##
        # sl.COORDINATE_SYSTEM to be used as reference for positional tracking, mesh, point clouds, etc.
        #
        # This parameter allows you to select the sl.COORDINATE_SYSTEM used by the sl.Camera object to return its measures.
        # \n This defines the order and the direction of the axis of the coordinate system.
        # \n Default: \ref COORDINATE_SYSTEM "sl.COORDINATE_SYSTEM.IMAGE"
        @property
        def coordinate_system(self) -> COORDINATE_SYSTEM:
            return COORDINATE_SYSTEM(<int>self.init.coordinate_system)

        @coordinate_system.setter
        def coordinate_system(self, value):
            if isinstance(value, COORDINATE_SYSTEM):
                self.init.coordinate_system = <c_COORDINATE_SYSTEM>(<int>value.value)
            else:
                raise TypeError("Argument must be of COORDINATE_SYSTEM type.")

        ##
        # Enable the ZED SDK verbose mode.
        #
        # This parameter allows you to enable the verbosity of the ZED SDK to get a variety of runtime information in the console.
        # \n When developing an application, enabling verbose (<code>\ref sdk_verbose >= 1</code>) mode can help you understand the current ZED SDK behavior.
        # \n However, this might not be desirable in a shipped version.
        # \n Default: 0 (no verbose message)
        # \note The verbose messages can also be exported into a log file.
        # \note See \ref sdk_verbose_log_file for more.
        @property
        def sdk_verbose(self) -> int:
            return self.init.sdk_verbose

        @sdk_verbose.setter
        def sdk_verbose(self, value: int):
            self.init.sdk_verbose = value

        ##
        # File path to store the ZED SDK logs (if \ref sdk_verbose is enabled).
        #
        # The file will be created if it does not exist.
        # \n Default: ""
        #
        # \note Setting this parameter to any value will redirect all standard output print calls of the entire program.
        # \note This means that your own standard output print calls will be redirected to the log file.
        # \warning The log file won't be cleared after successive executions of the application.
        # \warning This means that it can grow indefinitely if not cleared. 
        @property
        def sdk_verbose_log_file(self) -> str:
            if not self.init.sdk_verbose_log_file.empty():
                return self.init.sdk_verbose_log_file.get().decode()
            else:
                return ""

        @sdk_verbose_log_file.setter
        def sdk_verbose_log_file(self, value: str):
            value_filename = value.encode()
            self.init.sdk_verbose_log_file.set(<char*>value_filename)

        ##
        # The SDK can handle different input types:
        #   - Select a camera by its ID (/dev/video<i>X</i> on Linux, and 0 to N cameras connected on Windows)
        #   - Select a camera by its serial number
        #   - Open a recorded sequence in the SVO file format
        #   - Open a streaming camera from its IP address and port
        #
        # This parameter allows you to select to desired input. It should be used like this:
        # \code
        # init_params = sl.InitParametersOne()    # Set initial parameters
        # init_params.sdk_verbose = 1 # Enable verbose mode
        # input_t = sl.InputType()
        # input_t.set_from_camera_id(0) # Selects the camera with ID = 0
        # init_params.input = input_t
        # init_params.set_from_camera_id(0) # You can also use this
        # \endcode
        #
        # \code
        # init_params = sl.InitParametersOne()    # Set initial parameters
        # init_params.sdk_verbose = 1 # Enable verbose mode
        # input_t = sl.InputType()
        # input_t.set_from_serial_number(1010) # Selects the camera with serial number = 101
        # init_params.input = input_t
        # init_params.set_from_serial_number(1010) # You can also use this
        # \endcode
        #
        # \code
        # init_params = sl.InitParametersOne()    # Set initial parameters
        # init_params.sdk_verbose = 1 # Enable verbose mode
        # input_t = sl.InputType()
        # input_t.set_from_svo_file("/path/to/file.svo") # Selects the and SVO file to be read
        # init_params.input = input_t
        # init_params.set_from_svo_file("/path/to/file.svo")  # You can also use this
        # \endcode
        # 
        # \code
        # init_params = sl.InitParametersOne()   # Set initial parameters
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
        # def input(self) -> InputType:
        #    input_t = InputType()
        #    input_t.input = self.init.input
        #    return input_t

        # @input.setter
        def input(self, input_t: InputType):
            self.init.input = input_t.input

        input = property(None, input)

        ##
        # Optional path where the ZED SDK has to search for the settings file (<i>SN<XXXX>.conf</i> file).
        #
        # This file contains the calibration information of the camera.
        # \n Default: ""
        #
        # \note The settings file will be searched in the default directory: <ul>
        # <li><b>Linux</b>: <i>/usr/local/zed/settings/</i></li> 
        # <li><b>Windows</b>: <i>C:/ProgramData/stereolabs/settings</i></li></ul>
        # 
        # \note If a path is specified and no file has been found, the ZED SDK will search the settings file in the default directory.
        # \note An automatic download of the settings file (through <b>ZED Explorer</b> or the installer) will still download the files on the default path.
        #
        # \code
        # init_params = sl.InitParametersOne()    # Set initial parameters
        # home = "/path/to/home"
        # path = home + "/Documents/settings/" # assuming /path/to/home/Documents/settings/SNXXXX.conf exists. Otherwise, it will be searched in /usr/local/zed/settings/
        # init_params.optional_settings_path = path
        # \endcode
        @property
        def optional_settings_path(self) -> str:
            if not self.init.optional_settings_path.empty():
                return self.init.optional_settings_path.get().decode()
            else:
                return ""

        @optional_settings_path.setter
        def optional_settings_path(self, value: str):
            value_filename = value.encode()
            self.init.optional_settings_path.set(<char*>value_filename)

        ##
        # Define the behavior of the automatic camera recovery during sl.Camera.grab() method call.
        #
        # When async is enabled and there's an issue with the communication with the sl.Camera object,
        # sl.Camera.grab() will exit after a short period and return the \ref ERROR_CODE "sl.ERROR_CODE.CAMERA_REBOOTING" warning.
        # \n The recovery will run in the background until the correct communication is restored.
        # \n When \ref async_grab_camera_recovery is false, the sl.Camera.grab() method is blocking and will return
        # only once the camera communication is restored or the timeout is reached. 
        # \n Default: False
        @property
        def async_grab_camera_recovery(self) -> bool:
            return self.init.async_grab_camera_recovery

        @async_grab_camera_recovery.setter
        def async_grab_camera_recovery(self, value: bool):
            self.init.async_grab_camera_recovery = value
        
        ##
        # Defines the input source with a camera id to initialize and open an sl.Camera object from.
        # \param id : Id of the desired camera to open.
        # \param bus_type : sl.BUS_TYPE of the desired camera to open.
        def set_from_camera_id(self, id: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
            self.init.input.setFromCameraID(id, <c_BUS_TYPE>(<int>(bus_type.value)))

        ##
        # Defines the input source with a serial number to initialize and open an sl.Camera object from.
        # \param serial_number : Serial number of the desired camera to open.
        # \param bus_type : sl.BUS_TYPE of the desired camera to open.
        def set_from_serial_number(self, serial_number: uint, bus_type : BUS_TYPE = BUS_TYPE.AUTO) -> None:
            self.init.input.setFromSerialNumber(serial_number, <c_BUS_TYPE>(<int>(bus_type.value)))

        ##
        # Defines the input source with an SVO file to initialize and open an sl.Camera object from.
        # \param svo_input_filename : Path to the desired SVO file to open.
        def set_from_svo_file(self, svo_input_filename: str) -> None:
            filename = svo_input_filename.encode()
            self.init.input.setFromSVOFile(String(<char*> filename))

        ##
        # Defines the input source from a stream to initialize and open an sl.Camera object from.
        # \param sender_ip : IP address of the streaming sender.
        # \param port : Port on which to listen. Default: 30000
        def set_from_stream(self, sender_ip: str, port=30000) -> None:
            sender_ip_ = sender_ip.encode()
            self.init.input.setFromStream(String(<char*>sender_ip_), port)

    cdef class CameraOne:
        cdef c_CameraOne camera

        ##
        # Close an opened camera.
        #
        # If \ref open() has been called, this method will close the connection to the camera (or the SVO file) and free the corresponding memory.
        #
        # If \ref open() wasn't called or failed, this method won't have any effect.
        #
        # \note If an asynchronous task is running within the \ref Camera object, like \ref save_area_map(), this method will wait for its completion.
        # \note To apply a new \ref InitParametersOne, you will need to close the camera first and then open it again with the new InitParameters values.
        # \warning Therefore you need to make sure to delete your GPU \ref sl.Mat objects before the context is destroyed.
        def close(self) -> None:
            self.camera.close()

        ##
        # Opens the ZED camera from the provided InitParametersOne.
        # The method will also check the hardware requirements and run a self-calibration.
        # \param py_init : A structure containing all the initial parameters. Default: a preset of InitParametersOne.
        # \return An error code giving information about the internal process. If \ref ERROR_CODE "ERROR_CODE.SUCCESS" is returned, the camera is ready to use. Every other code indicates an error and the program should be stopped.
        #
        # Here is the proper way to call this function:
        #
        # \code
        # zed = sl.CameraOne() # Create a ZED camera object
        #
        # init_params = sl.InitParametersOne()    # Set configuration parameters
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
        # \note If you are having issues opening a camera, the diagnostic tool provided in the SDK can help you identify to problems.
        #   - <b>Windows:</b> <i>C:\\Program Files (x86)\\ZED SDK\\tools\\ZED Diagnostic.exe</i>
        #   - <b>Linux:</b> <i>/usr/local/zed/tools/ZED Diagnostic</i>
        # \note If this method is called on an already opened camera, \ref close() will be called.
        def open(self, py_init : InitParametersOne = InitParametersOne()) -> ERROR_CODE:
            cdef c_InitParametersOne ini = py_init.init
            return _error_code_cache.get(<int>self.camera.open(ini), ERROR_CODE.FAILURE)

        ##
        # Reports if the camera has been successfully opened.
        # It has the same behavior as checking if \ref open() returns \ref ERROR_CODE "ERROR_CODE.SUCCESS".
        # \return True if the ZED camera is already setup, otherwise false.
        def is_opened(self) -> bool:
            return self.camera.isOpened()

        ##
        # This method will grab the latest images from the camera, rectify them, and compute the \ref retrieve_measure() "measurements" based on the \ref RuntimeParameters provided (depth, point cloud, tracking, etc.)
        #
        # As measures are created in this method, its execution can last a few milliseconds, depending on your parameters and your hardware.
        # \n The exact duration will mostly depend on the following parameters:
        # 
        #   - \ref InitParametersOne.camera_resolution : Lower resolutions are faster to compute.
        #
        # This method is meant to be called frequently in the main loop of your application.
        # \note Since ZED SDK 3.0, this method is blocking. It means that grab() will wait until a new frame is detected and available.
        # \note If no new frames is available until timeout is reached, grab() will return \ref ERROR_CODE "ERROR_CODE.CAMERA_NOT_DETECTED" since the camera has probably been disconnected.
        # 
        # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" means that no problem was encountered.
        # \note Returned errors can be displayed using <code>str()</code>.
        #
        # \code
        # image = sl.Mat()
        # while True:
        #       # Grab an image
        #       if zed.grab() == sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
        #           zed.retrieve_image(image) # Get the left image     
        #           # Use the image for your application
        # \endcode
        def grab(self) -> ERROR_CODE:
            cdef c_ERROR_CODE ret
            with nogil:
                ret = self.camera.grab()
            return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

        ##
        # Retrieves images from the camera (or SVO file).
        #
        # Multiple images are available along with a view of various measures for display purposes.
        # \n Available images and views are listed \ref VIEW "here".
        # \n As an example, \ref VIEW "VIEW.DEPTH" can be used to get a gray-scale version of the depth map, but the actual depth values can be retrieved using \ref retrieve_measure() .
        # \n
        # \n <b>Pixels</b>
        # \n Most VIEW modes output image with 4 channels as BGRA (Blue, Green, Red, Alpha), for more information see enum \ref VIEW
        # \n
        # \n <b>Memory</b>
        # \n By default, images are copied from GPU memory to CPU memory (RAM) when this function is called.
        # \n If your application can use GPU images, using the <b>type</b> parameter can increase performance by avoiding this copy.
        # \n If the provided sl.Mat object is already allocated  and matches the requested image format, memory won't be re-allocated.
        # \n
        # \n <b>Image size</b>
        # \n By default, images are returned in the resolution provided by \ref Resolution "get_camera_information().camera_configuration.resolution".
        # \n However, you can request custom resolutions. For example, requesting a smaller image can help you speed up your application.
        # \warning A sl.Mat resolution higher than the camera resolution <b>cannot</b> be requested.
        # 
        # \param py_mat[out] : The \ref sl.Mat to store the image.
        # \param view[in] : Defines the image you want (see \ref VIEW). Default: \ref VIEW "VIEW.LEFT".
        # \param mem_type[in] : Defines on which memory the image should be allocated. Default: \ref MEM "MEM.CPU" (you cannot change this default value).
        # \param resolution[in] : If specified, defines the \ref Resolution of the output sl.Mat. If set to \ref Resolution "Resolution(0,0)", the camera resolution will be taken. Default: (0,0).
        # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the method succeeded.
        # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" if the view mode requires a module not enabled (\ref VIEW "VIEW.DEPTH" with \ref DEPTH_MODE "DEPTH_MODE.NONE" for example).
        # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if another error occurred.
        # 
        # \note As this method retrieves the images grabbed by the \ref grab() method, it should be called afterward.
        #
        # \code
        # # create sl.Mat objects to store the images
        # left_image = sl.Mat()
        # while True:
        #       # Grab an image
        #       if zed.grab() == sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
        #           zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
        #
        #           # Display the center pixel colors
        #           err, left_center = left_image.get_value(left_image.get_width() / 2, left_image.get_height() / 2)
        #           if err == sl.ERROR_CODE.SUCCESS:
        #               print("left_image center pixel R:", int(left_center[0]), " G:", int(left_center[1]), " B:", int(left_center[2]))
        #           else:
        #               print("error:", err)
        # \endcode
        def retrieve_image(self, Mat py_mat, view=VIEW.LEFT, mem_type=MEM.CPU, Resolution resolution=Resolution(0,0)) -> ERROR_CODE:
            cdef c_ERROR_CODE ret
            cdef c_VIEW c_view = <c_VIEW>(<int>view.value)
            cdef c_MEM c_mem_type = <c_MEM>(<int>mem_type.value)
            with nogil:
                ret = self.camera.retrieveImage(py_mat.mat, c_view, c_mem_type, resolution.resolution)
            return _error_code_cache.get(<int>ret, ERROR_CODE.FAILURE)

        ##
        # Sets the playback cursor to the desired frame number in the SVO file.
        #
        # This method allows you to move around within a played-back SVO file. After calling, the next call to \ref grab() will read the provided frame number.
        #
        # \param frame_number : The number of the desired frame to be decoded.
        # 
        # \note The method works only if the camera is open in SVO playback mode.
        #
        # \code
        #
        # import pyzed.sl as sl
        #
        # def main():
        #     # Create a ZED camera object
        #     zed = sl.CameraOne()
        #
        #     # Set configuration parameters
        #     init_params = sl.InitParametersOne()   
        #     init_params.set_from_svo_file("path/to/my/file.svo")
        #
        #     # Open the camera
        #     err = zed.open(init_params)
        #     if err != sl.ERROR_CODE.SUCCESS:
        #         print(repr(err))
        #         exit(-1)
        #
        #     # Loop between frames 0 and 50
        #     left_image = sl.Mat()
        #     while zed.get_svo_position() < zed.get_svo_number_of_frames() - 1:
        #
        #         print("Current frame: ", zed.get_svo_position())
        #
        #         # Loop if we reached frame 50
        #         if zed.get_svo_position() == 50:
        #             zed.set_svo_position(0)
        #
        #         # Grab an image
        #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
        #             zed.retrieve_image(left_image, sl.VIEW.LEFT) # Get the rectified left image
        #
        #         # Use the image in your application
        #
        #     # Close the Camera
        #     zed.close()
        #     return 0
        #
        # if __name__ == "__main__" :
        #     main()
        #
        # \endcode
        def set_svo_position(self, frame_number: int) -> None:
            self.camera.setSVOPosition(frame_number)

        ##
        # Returns the current playback position in the SVO file.
        #
        # The position corresponds to the number of frames already read from the SVO file, starting from 0 to n.
        # 
        # Each \ref grab() call increases this value by one (except when using \ref InitParametersOne.svo_real_time_mode).
        # \return The current frame position in the SVO file. -1 if the SDK is not reading an SVO.
        # 
        # \note The method works only if the camera is open in SVO playback mode.
        #
        # See \ref set_svo_position() for an example.
        def get_svo_position(self) -> int:
            return self.camera.getSVOPosition()

        ##
        # Returns the number of frames in the SVO file.
        #
        # \return The total number of frames in the SVO file. -1 if the SDK is not reading a SVO.
        #
        # The method works only if the camera is open in SVO playback mode.
        def get_svo_number_of_frames(self) -> int:
            return self.camera.getSVONumberOfFrames()
        
        ##
        # ingest a SVOData in the SVO file.
        #
        # \return An error code stating the success, or not.
        #
        # The method works only if the camera is open in SVO recording mode.
        def ingest_data_into_svo(self, data: SVOData) -> ERROR_CODE:
            if isinstance(data, SVOData) :
                return _error_code_cache.get(<int>self.camera.ingestDataIntoSVO(data.svo_data), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Arguments must be of SVOData.") 

        ##
        # Get the external channels that can be retrieved from the SVO file.
        #
        # \return a list of keys
        #
        # The method works only if the camera is open in SVO playback mode.
        def get_svo_data_keys(self) -> list:
            vect_ = self.camera.getSVODataKeys()
            vect_python = []
            for i in range(vect_.size()):
                vect_python.append(vect_[i].decode())

            return vect_python

        ##
        # retrieve SVO datas from the SVO file at the given channel key and in the given timestamp range.
        #
        # \return An error code stating the success, or not.
        # \param key : The channel key.
        # \param data : The dict to be filled with SVOData objects, with timestamps as keys.
        # \param ts_begin : The beginning of the range.
        # \param ts_end : The end of the range.
        #
        # The method works only if the camera is open in SVO playback mode.
        def retrieve_svo_data(self, key: str, data: dict, ts_begin: Timestamp, ts_end: Timestamp) -> ERROR_CODE:
            cdef map[c_Timestamp, c_SVOData] data_c
            cdef map[c_Timestamp, c_SVOData].iterator it

            if isinstance(key, str) :
                if isinstance(data, dict) :
                    res = _error_code_cache.get(<int>self.camera.retrieveSVOData(key.encode('utf-8'), data_c, ts_begin.timestamp, ts_end.timestamp), ERROR_CODE.FAILURE)
                    it = data_c.begin()

                    while(it != data_c.end()):
                        # let's pretend here I just want to print the key and the value
                        # print(deref(it).first) # print the key        
                        # print(deref(it).second) # print the associated value

                        ts = Timestamp()
                        ts.timestamp = deref(it).first
                        content_c = SVOData()
                        content_c.svo_data = deref(it).second
                        data[ts] = content_c

                        postincrement(it) # Increment the iterator to the net element

                    return res

                else:
                    raise TypeError("The content must be a dict.") 
            else:
                raise TypeError("The key must be a string.") 


        # Sets the value of the requested \ref VIDEO_SETTINGS "camera setting" (gain, brightness, hue, exposure, etc.).
        #
        # This method only applies for \ref VIDEO_SETTINGS that require a single value.
        #
        # Possible values (range) of each settings are available \ref VIDEO_SETTINGS "here".
        #
        # \param settings : The setting to be set.
        # \param value : The value to set. Default: auto mode
        # \return \ref ERROR_CODE to indicate if the method was successful.
        #
        # \warning Setting [VIDEO_SETTINGS.EXPOSURE](\ref VIDEO_SETTINGS) or [VIDEO_SETTINGS.GAIN](\ref VIDEO_SETTINGS) to default will automatically sets the other to default.
        #
        # \note The method works only if the camera is open in LIVE or STREAM mode.
        #
        # \code
        # # Set the gain to 50
        # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
        # \endcode
        def set_camera_settings(self, settings: VIDEO_SETTINGS, value=-1) -> ERROR_CODE:
            if isinstance(settings, VIDEO_SETTINGS) :
                return _error_code_cache.get(<int>self.camera.setCameraSettings(<c_VIDEO_SETTINGS>(<int>settings.value), value), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")
        ##
        # Sets the value of the requested \ref VIDEO_SETTINGS "camera setting" that supports two values (min/max).
        #
        # This method only works with the following \ref VIDEO_SETTINGS:
        # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE"
        # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE"
        # - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE"
        #
        # \param settings : The setting to be set.
        # \param min : The minimum value that can be reached (-1 or 0 gives full range).
        # \param max : The maximum value that can be reached (-1 or 0 gives full range).
        # \return \ref ERROR_CODE to indicate if the method was successful.
        #
        # \warning If \ref VIDEO_SETTINGS settings is not supported or min >= max, it will return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS".
        # \note The method works only if the camera is open in LIVE or STREAM mode.
        #
        # \code
        # # For ZED X based product, set the automatic exposure from 2ms to 5ms. Expected exposure time cannot go beyond those values
        # zed.set_camera_settings_range(sl.VIDEO_SETTINGS.AEC_RANGE, 2000, 5000);
        # \endcode
        def set_camera_settings_range(self, settings: VIDEO_SETTINGS, value_min=-1, value_max=-1) -> ERROR_CODE:
            if isinstance(settings, VIDEO_SETTINGS) :
                return _error_code_cache.get(<int>self.camera.setCameraSettingsRange(<c_VIDEO_SETTINGS>(<int>settings.value), value_min, value_max), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")

        ##
        # Overloaded method for \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI" which takes a Rect as parameter.
        #
        # \param settings : Must be set at \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI", otherwise the method will have no impact.
        # \param roi : Rect that defines the target to be applied for AEC/AGC computation. Must be given according to camera resolution.
        # \param eye : \ref SIDE on which to be applied for AEC/AGC computation. Default: \ref SIDE "SIDE.BOTH"
        # \param reset : Cancel the manual ROI and reset it to the full image. Default: False
        # 
        # \note The method works only if the camera is open in LIVE or STREAM mode.
        # 
        # \code
        #   roi = sl.Rect(42, 56, 120, 15)
        #   zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
        # \endcode
        #
        def set_camera_settings_roi(self, settings: VIDEO_SETTINGS, roi: Rect, reset = False) -> ERROR_CODE:
            if isinstance(settings, VIDEO_SETTINGS) :
                return _error_code_cache.get(<int>self.camera.setCameraSettingsROI(<c_VIDEO_SETTINGS>(<int>settings.value), roi.rect, reset), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Arguments must be of VIDEO_SETTINGS and boolean types.")
        
        ##
        # Returns the current value of the requested \ref VIDEO_SETTINGS "camera setting" (gain, brightness, hue, exposure, etc.).
        # 
        # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
        # 
        # \param setting : The requested setting.
        # \return \ref ERROR_CODE to indicate if the method was successful.
        # \return The current value for the corresponding setting.
        #
        # \code
        # err, gain = zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
        # if err == sl.ERROR_CODE.SUCCESS:
        #       print("Current gain value:", gain)
        # else:
        #       print("error:", err)
        # \endcode
        #
        # \note The method works only if the camera is open in LIVE or STREAM mode.
        # \note Settings are not exported in the SVO file format.
        def get_camera_settings(self, setting: VIDEO_SETTINGS) -> tuple(ERROR_CODE, int):
            cdef int value = 0
            if isinstance(setting, VIDEO_SETTINGS):
                error_code = _error_code_cache.get(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), value), ERROR_CODE.FAILURE)
                return error_code, value
            else:
                raise TypeError("Argument is not of VIDEO_SETTINGS type.")

        ##
        # Returns the values of the requested \ref VIDEO_SETTINGS "settings" for \ref VIDEO_SETTINGS that supports two values (min/max).
        # 
        # This method only works with the following VIDEO_SETTINGS:
        #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE"
        #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_ANALOG_GAIN_RANGE"
        #   - \ref VIDEO_SETTINGS "sl.VIDEO_SETTINGS.AUTO_DIGITAL_GAIN_RANGE"
        # 
        # Possible values (range) of each setting are available \ref VIDEO_SETTINGS "here".
        # \param setting : The requested setting.
        # \return \ref ERROR_CODE to indicate if the method was successful.
        # \return The current value of the minimum for the corresponding setting.
        # \return The current value of the maximum for the corresponding setting.
        #
        # \code
        # err, aec_range_min, aec_range_max = zed.get_camera_settings(sl.VIDEO_SETTINGS.AUTO_EXPOSURE_TIME_RANGE)
        # if err == sl.ERROR_CODE.SUCCESS:
        #       print("Current AUTO_EXPOSURE_TIME_RANGE range values ==> min:", aec_range_min, "max:", aec_range_max)
        # else:
        #       print("error:", err)
        # \endcode
        #
        # \note Works only with ZED X that supports low-level controls
        def get_camera_settings_range(self, setting: VIDEO_SETTINGS) -> tuple(ERROR_CODE, int, int):
            cdef int mini = 0
            cdef int maxi = 0
            if isinstance(setting, VIDEO_SETTINGS):
                error_code = _error_code_cache.get(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), <int&>mini, <int&>maxi), ERROR_CODE.FAILURE)
                return error_code, mini, maxi
            else:
                raise TypeError("Argument is not of VIDEO_SETTINGS type.")

        ##
        # Returns the current value of the currently used ROI for the camera setting \ref VIDEO_SETTINGS "AEC_AGC_ROI".
        # 
        # \param setting[in] : Must be set at \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI", otherwise the method will have no impact.
        # \param roi[out] : Roi that will be filled.
        # \param eye[in] : The requested side. Default: \ref SIDE "SIDE.BOTH"
        # \return \ref ERROR_CODE to indicate if the method was successful.
        #
        # \code
        # roi = sl.Rect()
        # err = zed.get_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
        # print("Current ROI for AEC_AGC: " + str(roi.x) + " " + str(roi.y)+ " " + str(roi.width) + " " + str(roi.height))
        # \endcode
        #
        # \note Works only if the camera is open in LIVE or STREAM mode with \ref VIDEO_SETTINGS "VIDEO_SETTINGS.AEC_AGC_ROI".
        # \note It will return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" or \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" otherwise.
        def get_camera_settings_roi(self, setting: VIDEO_SETTINGS, roi: Rect) -> ERROR_CODE:
            if isinstance(setting, VIDEO_SETTINGS):
                return _error_code_cache.get(<int>self.camera.getCameraSettings(<c_VIDEO_SETTINGS>(<int>setting.value), roi.rect), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Argument is not of SIDE type.")

        ##
        # Returns if the video setting is supported by the camera or not
        #
        # \param setting[in] : the video setting to test
        # \return True if the \ref VIDEO_SETTINGS is supported by the camera, False otherwise
        #
        def is_camera_setting_supported(self, setting: VIDEO_SETTINGS) -> bool:
            if not isinstance(setting, VIDEO_SETTINGS):
                raise TypeError("Argument is not of VIDEO_SETTINGS type.")

            return self.camera.isCameraSettingSupported(<c_VIDEO_SETTINGS>(<int>setting.value))

        ##
        # Returns the current framerate at which the \ref grab() method is successfully called.
        #
        # The returned value is based on the difference of camera \ref get_timestamp() "timestamps" between two successful grab() calls.
        #
        # \return The current SDK framerate
        #
        # \warning The returned framerate (number of images grabbed per second) can be lower than \ref InitParametersOne.camera_fps if the \ref grab() function runs slower than the image stream or is called too often.
        #
        # \code
        # current_fps = zed.get_current_fps()
        # print("Current framerate: ", current_fps)
        # \endcode
        def get_current_fps(self) -> float:
            return self.camera.getCurrentFPS()

        ##
        # Returns the timestamp in the requested \ref TIME_REFERENCE.
        #
        # - When requesting the \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" timestamp, the UNIX nanosecond timestamp of the latest \ref grab() "grabbed" image will be returned.
        # \n This value corresponds to the time at which the entire image was available in the PC memory. As such, it ignores the communication time that corresponds to 2 or 3 frame-time based on the fps (ex: 33.3ms to 50ms at 60fps).
        #
        # - When requesting the [TIME_REFERENCE.CURRENT](\ref TIME_REFERENCE) timestamp, the current UNIX nanosecond timestamp is returned.
        #
        # This function can also be used when playing back an SVO file.
        #
        # \param time_reference : The selected \ref TIME_REFERENCE.
        # \return The \ref Timestamp in nanosecond. 0 if not available (SVO file without compression).
        #
        # \note As this function returns UNIX timestamps, the reference it uses is common across several \ref Camera instances.
        # \n This can help to organized the grabbed images in a multi-camera application.
        # 
        # \code
        # last_image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        # current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
        # print("Latest image timestamp: ", last_image_timestamp.get_nanoseconds(), "ns from Epoch.")
        # print("Current timestamp: ", current_timestamp.get_nanoseconds(), "ns from Epoch.")
        # \endcode 
        def get_timestamp(self, time_reference: TIME_REFERENCE) -> Timestamp:
            if isinstance(time_reference, TIME_REFERENCE):
                ts = Timestamp()
                ts.timestamp = self.camera.getTimestamp(<c_TIME_REFERENCE>(<int>time_reference.value))
                return ts
            else:
                raise TypeError("Argument is not of TIME_REFERENCE type.")

        ##
        # Returns the number of frames dropped since \ref grab() was called for the first time.
        #
        # A dropped frame corresponds to a frame that never made it to the grab method.
        # \n This can happen if two frames were extracted from the camera when grab() is called. The older frame will be dropped so as to always use the latest (which minimizes latency).
        #
        # \return The number of frames dropped since the first \ref grab() call.
        def get_frame_dropped_count(self) -> int:
            return self.camera.getFrameDroppedCount()

        # #
        # Not implemented. Returns the CameraInformation associated the camera being used.
        
        # To ensure accurate calibration, it is possible to specify a custom resolution as a parameter when obtaining scaled information, as calibration parameters are resolution-dependent.
        # \n When reading an SVO file, the parameters will correspond to the camera used for recording.
        
        # \param resizer : You can specify a size different from the default image size to get the scaled camera information.
        # Default = (0,0) meaning original image size (given by \ref CameraConfiguration.resolution "get_camera_information().camera_configuration.resolution").
        # \return \ref CameraInformation containing the calibration parameters of the ZED, as well as serial number and firmware version.
        
        # \warning The returned parameters might vary between two execution due to the \ref InitParametersOne.camera_disable_self_calib "self-calibration" being run in the \ref open() method.
        # \note The calibration file SNXXXX.conf can be found in:
        # - <b>Windows:</b> <i>C:/ProgramData/Stereolabs/settings/</i>
        # - <b>Linux:</b> <i>/usr/local/zed/settings/</i>.
        def get_camera_information(self, resizer = Resolution(0, 0)) -> CameraOneInformation:
            return CameraOneInformation(self, resizer)

        ##
        # Returns the InitParametersOne associated with the Camera object.
        # It corresponds to the structure given as argument to \ref open() method.
        #
        # \return InitParametersOne containing the parameters used to initialize the Camera object.
        def get_init_parameters(self) -> InitParametersOne:
            init = InitParametersOne()
            init.init = self.camera.getInitParameters()
            return init

        ##
        # Returns the StreamingParameters used.
        #
        #  It corresponds to the structure given as argument to the enable_streaming() method.
        # 
        # \return \ref StreamingParameters containing the parameters used for streaming initialization.
        def get_streaming_parameters(self) -> StreamingParameters:
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
        # Retrieves the SensorsData (IMU, magnetometer, barometer) at a specific time reference.
        # 
        # - Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" gives you the latest sensors data received. Getting all the data requires to call this method at 800Hz in a thread.
        # - Calling \ref get_sensors_data with \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" gives you the sensors data at the time of the latest image \ref grab() "grabbed".
        #
        # \ref SensorsData object contains the previous \ref IMUData structure that was used in ZED SDK v2.X:
        # \n For IMU data, the values are provided in 2 ways :
        # <ul>
        #   <li><b>Time-fused</b> pose estimation that can be accessed using:
        #       <ul><li>\ref IMUData.get_pose "data.get_imu_data().get_pose()"</li></ul>
        #   </li>
        #   <li><b>Raw values</b> from the IMU sensor:
        #       <ul>
        #           <li>\ref IMUData.get_angular_velocity "data.get_imu_data().get_angular_velocity()", corresponding to the gyroscope</li>
        #           <li>\ref IMUData.get_linear_acceleration "data.get_imu_data().get_linear_acceleration()", corresponding to the accelerometer</li>
        #       </ul> both the gyroscope and accelerometer are synchronized.
        #   </li>
        # </ul>
        # 
        # The delta time between previous and current values can be calculated using \ref data.imu.timestamp
        #
        # \note The IMU quaternion (fused data) is given in the specified \ref COORDINATE_SYSTEM of \ref InitParametersOne.
        #   
        # \param data[out] : The SensorsData variable to store the data.
        # \param reference_frame[in]: Defines the reference from which you want the data to be expressed. Default: \ref REFERENCE_FRAME "REFERENCE_FRAME.WORLD".
        # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if sensors data have been extracted.
        # \return \ref ERROR_CODE "ERROR_CODE.SENSORS_NOT_AVAILABLE" if the camera model is a \ref MODEL "MODEL.ZED".
        # \return \ref ERROR_CODE "ERROR_CODE.MOTION_SENSORS_REQUIRED" if the camera model is correct but the sensors module is not opened.
        # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS" if the <b>reference_time</b> is not valid. See Warning.
        #
        # \warning In SVO reading mode, the \ref TIME_REFERENCE "TIME_REFERENCE.CURRENT" is currently not available (yielding \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_PARAMETERS".
        # \warning Only the quaternion data and barometer data (if available) at \ref TIME_REFERENCE "TIME_REFERENCE.IMAGE" are available. Other values will be set to 0.
        #
        def get_sensors_data(self, py_sensors_data: SensorsData, time_reference = TIME_REFERENCE.CURRENT) -> ERROR_CODE:
            if isinstance(time_reference, TIME_REFERENCE):
                return _error_code_cache.get(<int>self.camera.getSensorsData(py_sensors_data.sensorsData, <c_TIME_REFERENCE>(<int>time_reference.value)), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Argument is not of TIME_REFERENCE type.")

        ##
        # Creates a streaming pipeline.
        #
        # \param streaming_parameters : A structure containing all the specific parameters for the streaming. Default: a reset of StreamingParameters .
        # \return \ref ERROR_CODE "ERROR_CODE.SUCCESS" if the streaming was successfully started.
        # \return \ref ERROR_CODE "ERROR_CODE.INVALID_FUNCTION_CALL" if open() was not successfully called before.
        # \return \ref ERROR_CODE "ERROR_CODE.FAILURE" if streaming RTSP protocol was not able to start.
        # \return \ref ERROR_CODE "ERROR_CODE.NO_GPU_COMPATIBLE" if the streaming codec is not supported (in this case, use H264 codec which is supported on all NVIDIA GPU the ZED SDK supports).
        #
        # \code
        # import pyzed.sl as sl
        #
        # def main() :
        #     # Create a ZED camera object
        #     zed = sl.CameraOneOne()
        #
        #     # Set initial parameters
        #     init_params = sl.InitParametersOne()   
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
        def enable_streaming(self, streaming_parameters = StreamingParameters()) -> ERROR_CODE:
            return _error_code_cache.get(<int>self.camera.enableStreaming(deref((<StreamingParameters>streaming_parameters).streaming)), ERROR_CODE.FAILURE)

        ##
        # Disables the streaming initiated by \ref enable_streaming().
        # \note This method will automatically be called by \ref close() if enable_streaming() was called.
        #
        # See \ref enable_streaming() for an example.
        def disable_streaming(self) -> None:
            self.camera.disableStreaming()

        ##
        # Tells if the streaming is running.
        # \return True if the stream is running, False otherwise.
        def is_streaming_enabled(self) -> bool:
            return self.camera.isStreamingEnabled()


        ##
        # Creates an SVO file to be filled by enable_recording() and disable_recording().
        # 
        # \n SVO files are custom video files containing the un-rectified images from the camera along with some meta-data like timestamps or IMU orientation (if applicable).
        # \n They can be used to simulate a live ZED and test a sequence with various SDK parameters.
        # \n Depending on the application, various compression modes are available. See \ref SVO_COMPRESSION_MODE.
        # 
        # \param record : A structure containing all the specific parameters for the recording such as filename and compression mode. Default: a reset of RecordingParameters .
        # \return An \ref ERROR_CODE that defines if the SVO file was successfully created and can be filled with images.
        # 
        # \warning This method can be called multiple times during a camera lifetime, but if <b>video_filename</b> is already existing, the file will be erased.
        #
        # 
        # \code
        # import pyzed.sl as sl
        # 
        # def main() :
        #     # Create a ZED camera object
        #     zed = sl.CameraOneOne()
        #     # Set initial parameters
        #     init_params = sl.InitParametersOne()
        #     init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
        #     init_params.coordinate_units = sl.UNIT.METER # Set units in meters
        #     # Open the camera
        #     err = zed.open(init_params)
        #     if (err != sl.ERROR_CODE.SUCCESS):
        #         print(repr(err))
        #         exit(-1)
        #
        #     # Enable video recording
        #     record_params = sl.RecordingParameters("myVideoFile.svo")
        #     err = zed.enable_recording(record_params)
        #     if (err != sl.ERROR_CODE.SUCCESS):
        #         print(repr(err))
        #         exit(-1)
        # 
        #     # Grab data during 500 frames
        #     i = 0
        #     while i < 500 :
        #         # Grab a new frame
        #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
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
        def enable_recording(self, record: RecordingParameters) -> ERROR_CODE:
            if isinstance(record, RecordingParameters):
                return _error_code_cache.get(<int>self.camera.enableRecording(deref(record.record)), ERROR_CODE.FAILURE)
            else:
                raise TypeError("Argument is not of RecordingParameters type.")

        ##
        # Disables the recording initiated by \ref enable_recording() and closes the generated file.
        #
        # \note This method will automatically be called by \ref close() if \ref enable_recording() was called.
        # 
        # See \ref enable_recording() for an example.
        def disable_recording(self) -> None:
            self.camera.disableRecording()

        ##
        # Get the recording information.
        # \return The recording state structure. For more details, see \ref RecordingStatus.
        def get_recording_status(self) -> RecordingStatus:
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
        # \param status : If True, the recording is paused. If False, the recording is resumed.
        def pause_recording(self, value=True) -> None:
            self.camera.pauseRecording(value)

        ##
        # List all the connected devices with their associated information.
        # 
        # This method lists all the cameras available and provides their serial number, models and other information.
        # \return The device properties for each connected camera.
        @staticmethod
        def get_device_list() -> list[DeviceProperties]:
            cls = CameraOne()
            vect_ = cls.camera.getDeviceList()
            vect_python = []
            for i in range(vect_.size()):
                prop = DeviceProperties()
                prop.camera_state = CAMERA_STATE(<int> vect_[i].camera_state)
                prop.id = vect_[i].id
                if not vect_[i].path.empty():
                    prop.path = vect_[i].path.get().decode()
                prop.camera_model = MODEL(<int>vect_[i].camera_model)
                prop.serial_number = vect_[i].serial_number
                vect_python.append(prop)
            return vect_python

        ##
        # Performs a hardware reset of the ZED 2 and the ZED 2i.
        # 
        # \param sn : Serial number of the camera to reset, or 0 to reset the first camera detected.
        # \param full_reboot : Perform a full reboot (sensors and video modules) if True, otherwise only the video module will be rebooted.
        # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" if everything went fine.
        # \return \ref ERROR_CODE "ERROR_CODE::CAMERA_NOT_DETECTED" if no camera was detected.
        # \return \ref ERROR_CODE "ERROR_CODE::FAILURE"  otherwise.
        #
        # \note This method only works for ZED 2, ZED 2i, and newer camera models.
        # 
        # \warning This method will invalidate any sl.Camera object, since the device is rebooting.
        @staticmethod
        def reboot(sn : int, full_reboot: bool =True) -> ERROR_CODE:
            cls = Camera()
            return _error_code_cache.get(<int>cls.camera.reboot(sn, full_reboot), ERROR_CODE.FAILURE)

        ##
        # Performs a hardware reset of all devices matching the InputType.
        # 
        # \param input_type : Input type of the devices to reset.
        # \return \ref ERROR_CODE "ERROR_CODE::SUCCESS" if everything went fine.
        # \return \ref ERROR_CODE "ERROR_CODE::CAMERA_NOT_DETECTED" if no camera was detected.
        # \return \ref ERROR_CODE "ERROR_CODE::FAILURE" otherwise.
        # \return \ref ERROR_CODE "ERROR_CODE::INVALID_FUNCTION_PARAMETERS" for SVOs and streams.
        # 
        # \warning This method will invalidate any sl.Camera object, since the device is rebooting.
        @staticmethod
        def reboot_from_input(input_type: INPUT_TYPE) -> ERROR_CODE:
            if not isinstance(input_type, INPUT_TYPE):
                raise TypeError("Argument is not of INPUT_TYPE type.")
            cls = Camera()
            return _error_code_cache.get(<int>cls.camera.reboot_from_type(<c_INPUT_TYPE>(<int>input_type.value)), ERROR_CODE.FAILURE)
