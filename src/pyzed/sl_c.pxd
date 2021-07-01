########################################################################
#
# Copyright (c) 2021, STEREOLABS.
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

# File containing the Cython declarations to use the sl functions.

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.string cimport const_char
from libcpp.map cimport map

cdef extern from "<array>" namespace "std" nogil:
  cdef cppclass array6 "std::array<float, 6>":
    array6() except+
    int& operator[](size_t)

cdef extern from "Utils.cpp" namespace "sl":
    string to_str(String sl_str)

cdef extern from "sl/Camera.hpp" namespace "sl":

    cdef cppclass Timestamp 'sl::Timestamp':
        unsigned long long data_ns

        Timestamp()
        Timestamp(unsigned long long _data_ns)
        unsigned long long getNanoseconds()
        unsigned long long getMicroseconds()
        unsigned long long getMilliseconds()
        unsigned long long getSeconds()

        void setNanoseconds(unsigned long long t_ns)
        void setMicroseconds(unsigned long long t_us)
        void setMilliseconds(unsigned long long t_ms)
        void setSeconds(unsigned long long t_s)


    ctypedef enum ERROR_CODE "sl::ERROR_CODE" :
        SUCCESS 'sl::ERROR_CODE::SUCCESS',
        FAILURE 'sl::ERROR_CODE::FAILURE',
        NO_GPU_COMPATIBLE 'sl::ERROR_CODE::NO_GPU_COMPATIBLE',
        NOT_ENOUGH_GPU_MEMORY 'sl::ERROR_CODE::NOT_ENOUGH_GPU_MEMORY',
        CAMERA_NOT_DETECTED 'sl::ERROR_CODE::CAMERA_NOT_DETECTED',
        SENSORS_NOT_AVAILABLE 'sl::ERROR_CODE::SENSORS_NOT_AVAILABLE',
        INVALID_RESOLUTION 'sl::ERROR_CODE::INVALID_RESOLUTION',
        LOW_USB_BANDWIDTH 'sl::ERROR_CODE::LOW_USB_BANDWIDTH',
        CALIBRATION_FILE_NOT_AVAILABLE 'sl::ERROR_CODE::CALIBRATION_FILE_NOT_AVAILABLE',
        INVALID_CALIBRATION_FILE 'sl::ERROR_CODE::INVALID_CALIBRATION_FILE',
        INVALID_SVO_FILE 'sl::ERROR_CODE::INVALID_SVO_FILE',
        SVO_RECORDING_ERROR 'sl::ERROR_CODE::SVO_RECORDING_ERROR',
        SVO_UNSUPPORTED_COMPRESSION 'sl::ERROR_CODE::SVO_UNSUPPORTED_COMPRESSION',
        END_OF_SVOFILE_REACHED 'sl::ERROR_CODE::END_OF_SVOFILE_REACHED',
        INVALID_COORDINATE_SYSTEM 'sl::ERROR_CODE::INVALID_COORDINATE_SYSTEM',
        INVALID_FIRMWARE 'sl::ERROR_CODE::INVALID_FIRMWARE',
        INVALID_FUNCTION_PARAMETERS 'sl::ERROR_CODE::INVALID_FUNCTION_PARAMETERS',
        CUDA_ERROR 'sl::ERROR_CODE::CUDA_ERROR',
        CAMERA_NOT_INITIALIZED 'sl::ERROR_CODE::CAMERA_NOT_INITIALIZED',
        NVIDIA_DRIVER_OUT_OF_DATE 'sl::ERROR_CODE::NVIDIA_DRIVER_OUT_OF_DATE',
        INVALID_FUNCTION_CALL 'sl::ERROR_CODE::INVALID_FUNCTION_CALL',
        CORRUPTED_SDK_INSTALLATION 'sl::ERROR_CODE::CORRUPTED_SDK_INSTALLATION',
        INCOMPATIBLE_SDK_VERSION 'sl::ERROR_CODE::INCOMPATIBLE_SDK_VERSION',
        INVALID_AREA_FILE 'sl::ERROR_CODE::INVALID_AREA_FILE',
        INCOMPATIBLE_AREA_FILE 'sl::ERROR_CODE::INCOMPATIBLE_AREA_FILE',
        CAMERA_FAILED_TO_SETUP 'sl::ERROR_CODE::CAMERA_FAILED_TO_SETUP',
        CAMERA_DETECTION_ISSUE 'sl::ERROR_CODE::CAMERA_DETECTION_ISSUE',
        CANNOT_START_CAMERA_STREAM 'sl::ERROR_CODE::CANNOT_START_CAMERA_STREAM',
        NO_GPU_DETECTED 'sl::ERROR_CODE::NO_GPU_DETECTED',
        PLANE_NOT_FOUND 'sl::ERROR_CODE::PLANE_NOT_FOUND',
        MODULE_NOT_COMPATIBLE_WITH_CAMERA 'sl::ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_CAMERA',
        MOTION_SENSORS_REQUIRED 'sl::ERROR_CODE::MOTION_SENSORS_REQUIRED',
        LAST 'sl::ERROR_CODE::LAST'

    String toString(ERROR_CODE o)

    void sleep_ms(int time)

    void sleep_us(int time)

    ctypedef enum MODEL "sl::MODEL":
        ZED 'sl::MODEL::ZED',
        ZED_M 'sl::MODEL::ZED_M',
        ZED2 'sl::MODEL::ZED2',
        ZED2i 'sl::MODEL::ZED2i',
        MODEL_LAST 'sl::MODEL::LAST'

    String toString(MODEL o)

    ctypedef enum CAMERA_STATE:
        AVAILABLE 'sl::CAMERA_STATE::AVAILABLE',
        NOT_AVAILABLE 'sl::CAMERA_STATE::NOT_AVAILABLE',
        CAMERA_STATE_LAST 'sl::CAMERA_STATE::LAST'

    String toString(CAMERA_STATE o)

    cdef cppclass String 'sl::String':
        String()
        String(const char *data)
        void set(const char *data)
        const char *get() const
        bool empty() const
        string std_str() const

    cdef cppclass DeviceProperties:
        DeviceProperties()
        CAMERA_STATE camera_state
        int id
        String path
        MODEL camera_model
        unsigned int serial_number

    String toString(DeviceProperties o)

    cdef cppclass Vector2[T]:
        int size()
        Vector2()
        Vector2(const T v0, const T v1)
        T *ptr()
        T &operator[](int i)


    cdef cppclass Vector3[T]:
        int size()
        Vector3()
        Vector3(const T v0, const T v1, const T v2)
        T *ptr()
        T &operator[](int i)


    cdef cppclass Vector4[T]:
        int size()
        Vector4()
        Vector4(const T v0, const T v1, const T v2, const T v3)
        T *ptr()
        T &operator[](int i)

    cdef cppclass Matrix3f:
        float r[]
        Matrix3f() except +
        Matrix3f(float data[]) except +
        Matrix3f(const Matrix3f &mat) except +
        Matrix3f operator*(const Matrix3f &mat) const
        Matrix3f operator*(const double &scalar) const
        bool operator==(const Matrix3f &mat) const
        bool operator!=(const Matrix3f &mat) const
        void inverse()
        Matrix3f inverse(const Matrix3f &rotation)
        void transpose()
        Matrix3f transpose(const Matrix3f &rotation)
        void setIdentity()
        Matrix3f identity()
        void setZeros()
        Matrix3f zeros()
        String getInfos()
        String matrix_name

    cdef cppclass Matrix4f 'sl::Matrix4f':
        float m[]
        Matrix4f() except +
        Matrix4f(float data[]) except +
        Matrix4f(const Matrix4f &mat) except +
        Matrix4f operator*(const Matrix4f &mat) const
        Matrix4f operator*(const double &scalar) const
        bool operator==(const Matrix4f  &mat) const
        bool operator!=(const Matrix4f &mat) const
        ERROR_CODE inverse()

        Matrix4f inverse(const Matrix4f &mat)
        void transpose()
        @staticmethod
        Matrix4f transpose(const Matrix4f &mat)

        void setIdentity()

        @staticmethod
        Matrix4f identity()

        void setZeros()

        @staticmethod
        Matrix4f zeros()

        ERROR_CODE setSubMatrix3f(Matrix3f input, int row, int column)
        ERROR_CODE setSubVector3f(Vector3[float] input, int column)
        ERROR_CODE setSubVector4f(Vector4[float] input, int column)


        String getInfos()
        String matrix_name

    ctypedef enum UNIT 'sl::UNIT':
        MILLIMETER 'sl::UNIT::MILLIMETER'
        CENTIMETER 'sl::UNIT::CENTIMETER'
        METER 'sl::UNIT::METER'
        INCH 'sl::UNIT::INCH'
        FOOT 'sl::UNIT::FOOT'
        UNIT_LAST 'sl::UNIT::LAST'

    String toString(UNIT o)

    ctypedef enum COORDINATE_SYSTEM 'sl::COORDINATE_SYSTEM':
        IMAGE 'sl::COORDINATE_SYSTEM::IMAGE'
        LEFT_HANDED_Y_UP 'sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP'
        RIGHT_HANDED_Y_UP 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP'
        RIGHT_HANDED_Z_UP 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP'
        LEFT_HANDED_Z_UP 'sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP'
        RIGHT_HANDED_Z_UP_X_FWD 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD'
        COORDINATE_SYSTEM_LAST 'sl::COORDINATE_SYSTEM::LAST'

    String toString(COORDINATE_SYSTEM o)

    ctypedef enum SIDE 'sl::SIDE':
        LEFT 'sl::SIDE::LEFT'
        RIGHT 'sl::SIDE::RIGHT'
        BOTH 'sl::SIDE::BOTH'

    ctypedef enum RESOLUTION 'sl::RESOLUTION':
        HD2K 'sl::RESOLUTION::HD2K'
        HD1080 'sl::RESOLUTION::HD1080'
        HD720 'sl::RESOLUTION::HD720'
        VGA 'sl::RESOLUTION::VGA'
        RESOLUTION_LAST 'sl::RESOLUTION::LAST'

    String toString(RESOLUTION o)

    ctypedef enum VIDEO_SETTINGS 'sl::VIDEO_SETTINGS':
        BRIGHTNESS 'sl::VIDEO_SETTINGS::BRIGHTNESS'
        CONTRAST 'sl::VIDEO_SETTINGS::CONTRAST'
        HUE 'sl::VIDEO_SETTINGS::HUE'
        SATURATION 'sl::VIDEO_SETTINGS::SATURATION'
        SHARPNESS 'sl::VIDEO_SETTINGS::SHARPNESS'
        GAMMA 'sl::VIDEO_SETTINGS::GAMMA'
        GAIN 'sl::VIDEO_SETTINGS::GAIN'
        EXPOSURE 'sl::VIDEO_SETTINGS::EXPOSURE'
        AEC_AGC 'sl::VIDEO_SETTINGS::AEC_AGC'
        AEC_AGC_ROI 'sl::VIDEO_SETTINGS::AEC_AGC_ROI'
        WHITEBALANCE_TEMPERATURE 'sl::VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE'
        WHITEBALANCE_AUTO 'sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO'
        LED_STATUS 'sl::VIDEO_SETTINGS::LED_STATUS'
        VIDEO_SETTINGS_LAST 'sl::VIDEO_SETTINGS::LAST'

    String toString(VIDEO_SETTINGS o)

    ctypedef enum DEPTH_MODE 'sl::DEPTH_MODE':
        NONE 'sl::DEPTH_MODE::NONE'
        PERFORMANCE 'sl::DEPTH_MODE::PERFORMANCE'
        QUALITY 'sl::DEPTH_MODE::QUALITY'
        ULTRA 'sl::DEPTH_MODE::ULTRA'
        DEPTH_MODE_LAST 'sl::DEPTH_MODE::LAST'

    String toString(DEPTH_MODE o)

    ctypedef enum SENSING_MODE 'sl::SENSING_MODE':
        STANDARD 'sl::SENSING_MODE::STANDARD'
        FILL 'sl::SENSING_MODE::FILL'
        SENSING_MODE_LAST 'sl::SENSING_MODE::LAST'

    String toString(SENSING_MODE o)

    ctypedef enum MEASURE 'sl::MEASURE':
        DISPARITY 'sl::MEASURE::DISPARITY'
        DEPTH 'sl::MEASURE::DEPTH'
        CONFIDENCE 'sl::MEASURE::CONFIDENCE'
        XYZ 'sl::MEASURE::XYZ'
        XYZRGBA 'sl::MEASURE::XYZRGBA'
        XYZBGRA 'sl::MEASURE::XYZBGRA'
        XYZARGB 'sl::MEASURE::XYZARGB'
        XYZABGR 'sl::MEASURE::XYZABGR'
        NORMALS 'sl::MEASURE::NORMALS'
        DISPARITY_RIGHT 'sl::MEASURE::DISPARITY_RIGHT'
        DEPTH_RIGHT 'sl::MEASURE::DEPTH_RIGHT'
        XYZ_RIGHT 'sl::MEASURE::XYZ_RIGHT'
        XYZRGBA_RIGHT 'sl::MEASURE::XYZRGBA_RIGHT'
        XYZBGRA_RIGHT 'sl::MEASURE::XYZBGRA_RIGHT'
        XYZARGB_RIGHT 'sl::MEASURE::XYZARGB_RIGHT'
        XYZABGR_RIGHT 'sl::MEASURE::XYZABGR_RIGHT'
        NORMALS_RIGHT 'sl::MEASURE::NORMALS_RIGHT'
        DEPTH_U16_MM 'sl::MEASURE::DEPTH_U16_MM'
        DEPTH_U16_MM_RIGHT 'sl::MEASURE::DEPTH_U16_MM_RIGHT'
        MEASURE_LAST 'sl::MEASURE::LAST'

    String toString(MEASURE o)

    ctypedef enum VIEW 'sl::VIEW':
        LEFT 'sl::VIEW::LEFT'
        RIGHT 'sl::VIEW::RIGHT'
        LEFT_GRAY 'sl::VIEW::LEFT_GRAY'
        RIGHT_GRAY 'sl::VIEW::RIGHT_GRAY'
        LEFT_UNRECTIFIED 'sl::VIEW::LEFT_UNRECTIFIED'
        RIGHT_UNRECTIFIED 'sl::VIEW::RIGHT_UNRECTIFIED'
        LEFT_UNRECTIFIED_GRAY 'sl::VIEW::LEFT_UNRECTIFIED_GRAY'
        RIGHT_UNRECTIFIED_GRAY 'sl::VIEW::RIGHT_UNRECTIFIED_GRAY'
        SIDE_BY_SIDE 'sl::VIEW::SIDE_BY_SIDE'
        VIEW_DEPTH 'sl::VIEW::DEPTH'
        VIEW_CONFIDENCE 'sl::VIEW::CONFIDENCE'
        VIEW_NORMALS 'sl::VIEW::NORMALS'
        VIEW_DEPTH_RIGHT 'sl::VIEW::DEPTH_RIGHT'
        VIEW_NORMALS_RIGHT 'sl::VIEW::NORMALS_RIGHT'
        VIEW_LAST 'sl::VIEW::LAST'

    String toString(VIEW o)

    ctypedef enum TIME_REFERENCE 'sl::TIME_REFERENCE':
        TIME_REFERENCE_IMAGE 'sl::TIME_REFERENCE::IMAGE'
        CURRENT 'sl::TIME_REFERENCE::CURRENT'
        TIME_REFERENCE_LAST 'sl::TIME_REFERENCE::LAST'

    String toString(TIME_REFERENCE o)

    ctypedef enum POSITIONAL_TRACKING_STATE 'sl::POSITIONAL_TRACKING_STATE':
        SEARCHING 'sl::POSITIONAL_TRACKING_STATE::SEARCHING'
        OK 'sl::POSITIONAL_TRACKING_STATE::OK'
        OFF 'sl::POSITIONAL_TRACKING_STATE::OFF'
        FPS_TOO_LOW 'sl::POSITIONAL_TRACKING_STATE::FPS_TOO_LOW'
        POSITIONAL_TRACKING_STATE_LAST 'sl::POSITIONAL_TRACKING_STATE::LAST'

    String toString(POSITIONAL_TRACKING_STATE o)

    ctypedef enum AREA_EXPORTING_STATE 'sl::AREA_EXPORTING_STATE':
        AREA_EXPORTING_STATE_SUCCESS 'sl::AREA_EXPORTING_STATE::SUCCESS'
        RUNNING 'sl::AREA_EXPORTING_STATE::RUNNING'
        NOT_STARTED 'sl::AREA_EXPORTING_STATE::NOT_STARTED'
        FILE_EMPTY 'sl::AREA_EXPORTING_STATE::FILE_EMPTY'
        FILE_ERROR 'sl::AREA_EXPORTING_STATE::FILE_ERROR'
        SPATIAL_MEMORY_DISABLED 'sl::AREA_EXPORTING_STATE::SPATIAL_MEMORY_DISABLED'
        AREA_EXPORTING_STATE_LAST 'sl::AREA_EXPORTING_STATE::LAST'

    String toString(AREA_EXPORTING_STATE o)

    ctypedef enum REFERENCE_FRAME 'sl::REFERENCE_FRAME':
        WORLD 'sl::REFERENCE_FRAME::WORLD'
        CAMERA 'sl::REFERENCE_FRAME::CAMERA'
        REFERENCE_FRAME_LAST 'sl::REFERENCE_FRAME::LAST'

    String toString(REFERENCE_FRAME o)

    ctypedef enum SPATIAL_MAPPING_STATE 'sl::SPATIAL_MAPPING_STATE':
        INITIALIZING 'sl::SPATIAL_MAPPING_STATE::INITIALIZING'
        SPATIAL_MAPPING_STATE_OK 'sl::SPATIAL_MAPPING_STATE::OK'
        NOT_ENOUGH_MEMORY 'sl::SPATIAL_MAPPING_STATE::NOT_ENOUGH_MEMORY'
        NOT_ENABLED 'sl::SPATIAL_MAPPING_STATE::NOT_ENABLED'
        SPATIAL_MAPPING_STATE_FPS_TOO_LOW 'sl::SPATIAL_MAPPING_STATE::FPS_TOO_LOW'
        SPATIAL_MAPPING_STATE_LAST 'sl::SPATIAL_MAPPING_STATE::LAST'

    String toString(SPATIAL_MAPPING_STATE o)

    ctypedef enum SVO_COMPRESSION_MODE 'sl::SVO_COMPRESSION_MODE':
        LOSSLESS 'sl::SVO_COMPRESSION_MODE::LOSSLESS'
        H264 'sl::SVO_COMPRESSION_MODE::H264'
        H265 'sl::SVO_COMPRESSION_MODE::H265'
        SVO_COMPRESSION_MODE_LAST 'sl::SVO_COMPRESSION_MODE::LAST'

    String toString(SVO_COMPRESSION_MODE o)

    ctypedef enum SENSOR_TYPE 'sl::SENSOR_TYPE':
        ACCELEROMETER 'sl::SENSOR_TYPE::ACCELEROMETER'
        GYROSCOPE 'sl::SENSOR_TYPE::GYROSCOPE'
        MAGNETOMETER 'sl::SENSOR_TYPE::MAGNETOMETER'
        BAROMETER 'sl::SENSOR_TYPE::BAROMETER'

    ctypedef enum SENSORS_UNIT 'sl::SENSORS_UNIT':
        M_SEC_2 'sl::SENSORS_UNIT::M_SEC_2'
        DEG_SEC 'sl::SENSORS_UNIT::DEG_SEC'
        U_T 'sl::SENSORS_UNIT::U_T'
        HPA 'sl::SENSORS_UNIT::HPA'
        CELSIUS 'sl::SENSORS_UNIT::CELSIUS'
        HERTZ 'sl::SENSORS_UNIT::HERTZ'

    ctypedef enum INPUT_TYPE 'sl::INPUT_TYPE':
        USB 'sl::INPUT_TYPE::USB'
        SVO 'sl::INPUT_TYPE::SVO'
        STREAM 'sl::INPUT_TYPE::STREAM'

    ctypedef enum DETECTION_MODEL 'sl::DETECTION_MODEL':
        MULTI_CLASS_BOX 'sl::DETECTION_MODEL::MULTI_CLASS_BOX'
        HUMAN_BODY_FAST 'sl::DETECTION_MODEL::HUMAN_BODY_FAST'
        HUMAN_BODY_ACCURATE 'sl::DETECTION_MODEL::HUMAN_BODY_ACCURATE'
        MULTI_CLASS_BOX_ACCURATE 'sl::DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE'
        MULTI_CLASS_BOX_MEDIUM 'sl::DETECTION_MODEL::MULTI_CLASS_BOX_MEDIUM'
        HUMAN_BODY_MEDIUM 'sl::DETECTION_MODEL::HUMAN_BODY_MEDIUM'
        LAST 'sl::DETECTION_MODEL::LAST'

    cdef struct RecordingStatus:
        bool is_recording
        bool is_paused
        bool status
        double current_compression_time
        double current_compression_ratio
        double average_compression_time
        double average_compression_ratio


    Timestamp getCurrentTimeStamp()

    cdef struct Resolution:
        size_t width
        size_t height

    cdef cppclass Rect 'sl::Rect':
        size_t x
        size_t y
        size_t width
        size_t height
        bool contains(const Rect target, bool proper) const
        bool isContained(Rect target, bool proper) const

    cdef struct CameraParameters:
        float fx
        float fy
        float cx
        float cy
        double disto[5]
        float v_fov
        float h_fov
        float d_fov
        Resolution image_size

        void SetUp(float focal_x, float focal_y, float center_x, float center_y)


    cdef struct CalibrationParameters:
        CameraParameters left_cam
        CameraParameters right_cam
        Transform stereo_transform

        float getCameraBaseline()
        
        # Deprecated
        Vector3[float] R
        Vector3[float] T


    cdef struct SensorParameters:
        SENSOR_TYPE type
        float resolution
        float sampling_rate
        Vector2[float] range
        float noise_density
        float random_walk
        SENSORS_UNIT sensor_unit
        bool isAvailable


    cdef struct SensorsConfiguration:
        unsigned int firmware_version
        Transform camera_imu_transform

        SensorParameters accelerometer_parameters
        SensorParameters gyroscope_parameters
        SensorParameters magnetometer_parameters
        SensorParameters barometer_parameters


    cdef struct CameraConfiguration:
        CalibrationParameters calibration_parameters
        CalibrationParameters calibration_parameters_raw
        unsigned int firmware_version
        float camera_fps
        Resolution camera_resolution


    cdef struct CameraInformation:
        unsigned int serial_number
        MODEL camera_model
        INPUT_TYPE input_type
        CameraConfiguration camera_configuration
        SensorsConfiguration sensors_configuration

        # Deprecated
        CalibrationParameters calibration_parameters
        CalibrationParameters calibration_parameters_raw
        Transform camera_imu_transform
        unsigned int camera_firmware_version
        unsigned int sensors_firmware_version
        float camera_fps
        Resolution camera_resolution


    ctypedef enum MEM 'sl::MEM':
        CPU 'sl::MEM::CPU'
        GPU 'sl::MEM::GPU'

    MEM operator|(MEM a, MEM b)


    ctypedef enum COPY_TYPE 'sl::COPY_TYPE':
        CPU_CPU 'sl::COPY_TYPE::CPU_CPU'
        CPU_GPU 'sl::COPY_TYPE::CPU_GPU'
        GPU_GPU 'sl::COPY_TYPE::GPU_GPU'
        GPU_CPU 'sl::COPY_TYPE::GPU_CPU'


    ctypedef enum MAT_TYPE 'sl::MAT_TYPE':
        F32_C1 'sl::MAT_TYPE::F32_C1'
        F32_C2 'sl::MAT_TYPE::F32_C2'
        F32_C3 'sl::MAT_TYPE::F32_C3'
        F32_C4 'sl::MAT_TYPE::F32_C4'
        U8_C1 'sl::MAT_TYPE::U8_C1'
        U8_C2 'sl::MAT_TYPE::U8_C2'
        U8_C3 'sl::MAT_TYPE::U8_C3'
        U8_C4 'sl::MAT_TYPE::U8_C4'
        U16_C1 'sl::MAT_TYPE::U16_C1'

    ctypedef enum OBJECT_CLASS 'sl::OBJECT_CLASS':
        PERSON 'sl::OBJECT_CLASS::PERSON' = 0
        VEHICLE 'sl::OBJECT_CLASS::VEHICLE' = 1
        BAG 'sl::OBJECT_CLASS::BAG' = 2
        ANIMAL 'sl::OBJECT_CLASS::ANIMAL' = 3
        ELECTRONICS 'sl::OBJECT_CLASS::ELECTRONICS' = 4
        FRUIT_VEGETABLE 'sl::OBJECT_CLASS::FRUIT_VEGETABLE' = 5
        OBJECT_CLASS_LAST 'sl::OBJECT_CLASS::LAST' = 6

    String toString(OBJECT_CLASS o)

    ctypedef enum OBJECT_SUBCLASS 'sl::OBJECT_SUBCLASS':
        PERSON 'sl::OBJECT_SUBCLASS::PERSON' = 0
        BICYCLE 'sl::OBJECT_SUBCLASS::BICYCLE' = 1
        CAR 'sl::OBJECT_SUBCLASS::CAR' = 2
        MOTORBIKE 'sl::OBJECT_SUBCLASS::MOTORBIKE' = 3
        BUS 'sl::OBJECT_SUBCLASS::BUS' = 4
        TRUCK 'sl::OBJECT_SUBCLASS::TRUCK' = 5
        BOAT 'sl::OBJECT_SUBCLASS::BOAT' = 6
        BACKPACK 'sl::OBJECT_SUBCLASS::BACKPACK' = 7
        HANDBAG 'sl::OBJECT_SUBCLASS::HANDBAG' = 8
        SUITCASE 'sl::OBJECT_SUBCLASS::SUITCASE' = 9
        BIRD 'sl::OBJECT_SUBCLASS::BIRD' = 10
        CAT 'sl::OBJECT_SUBCLASS::CAT' = 11
        DOG 'sl::OBJECT_SUBCLASS::DOG' = 12
        HORSE 'sl::OBJECT_SUBCLASS::HORSE' = 13
        SHEEP 'sl::OBJECT_SUBCLASS::SHEEP' = 14
        COW 'sl::OBJECT_SUBCLASS::COW' = 15
        CELLPHONE 'sl::OBJECT_SUBCLASS::CELLPHONE' = 16
        LAPTOP 'sl::OBJECT_SUBCLASS::LAPTOP' = 17
        BANANA 'sl::OBJECT_SUBCLASS::BANANA' = 18
        APPLE 'sl::OBJECT_SUBCLASS::APPLE' = 19
        ORANGE 'sl::OBJECT_SUBCLASS::ORANGE' = 20
        CARROT 'sl::OBJECT_SUBCLASS::CARROT' = 21
        OBJECT_SUBCLASS_LAST 'sl::OBJECT_SUBCLASS::LAST' = 22

    String toString(OBJECT_SUBCLASS o)

    ctypedef enum OBJECT_TRACKING_STATE 'sl::OBJECT_TRACKING_STATE':
        OBJECT_TRACKING_STATE_OFF 'sl::OBJECT_TRACKING_STATE::OFF'
        OBJECT_TRACKING_STATE_OK 'sl::OBJECT_TRACKING_STATE::OK'
        OBJECT_TRACKING_STATE_SEARCHING 'sl::OBJECT_TRACKING_STATE::SEARCHING'
        TERMINATE 'sl::OBJECT_TRACKING_STATE::TERMINATE'
        OBJECT_TRACKING_STATE_LAST 'sl::OBJECT_TRACKING_STATE::LAST'

    String toString(OBJECT_TRACKING_STATE o)

    ctypedef enum OBJECT_ACTION_STATE 'sl::OBJECT_ACTION_STATE':
        IDLE 'sl::OBJECT_ACTION_STATE::IDLE'
        OBJECT_ACTION_STATE_MOVING 'sl::OBJECT_ACTION_STATE::MOVING'
        OBJECT_ACTION_STATE_LAST 'sl::OBJECT_ACTION_STATE::LAST'

    String toString(OBJECT_ACTION_STATE o)

    cdef cppclass ObjectData 'sl::ObjectData':
        unsigned int id
        OBJECT_CLASS label
        OBJECT_SUBCLASS sublabel
        OBJECT_TRACKING_STATE tracking_state
        OBJECT_ACTION_STATE action_state
        Mat mask
        Vector3[float] position
        Vector3[float] velocity
        Vector3[float] dimensions
        vector[Vector3[float]] bounding_box
        vector[Vector2[uint]] bounding_box_2d
        float confidence
        vector[Vector3[float]] keypoint
        vector[Vector2[float]] keypoint_2d
        vector[Vector3[float]] head_bounding_box
        vector[Vector2[uint]] head_bounding_box_2d
        Vector3[float] head_position
        vector[float] keypoint_confidence

    cdef cppclass ObjectsBatch 'sl::ObjectsBatch':
        int id
        OBJECT_CLASS label
        OBJECT_SUBCLASS sublabel
        OBJECT_TRACKING_STATE tracking_state
        vector[Vector3[float]] positions
        vector[array6] position_covariances
        vector[Vector3[float]] velocities
        vector[Timestamp] timestamps
        vector[vector[Vector3[float]]] bounding_boxes
        vector[vector[Vector2[uint]]] bounding_boxes_2d
        vector[float] confidences
        vector[OBJECT_ACTION_STATE] action_states
        vector[vector[Vector2[float]]] keypoints_2d
        vector[vector[Vector3[float]]] keypoints
        vector[vector[Vector2[uint]]] head_bounding_boxes_2d
        vector[vector[Vector3[float]]] head_bounding_boxes
        vector[Vector3[float]] head_positions
        vector[vector[float]] keypoint_confidences

    cdef cppclass Objects 'sl::Objects':
        Timestamp timestamp
        vector[ObjectData] object_list
        bool is_new
        bool is_tracked
        bool getObjectDataFromId(ObjectData &objectData, int objectDataId)


    ctypedef enum BODY_PARTS 'sl::BODY_PARTS':
        NOSE 'sl::BODY_PARTS::NOSE'
        NECK 'sl::BODY_PARTS::NECK'
        RIGHT_SHOULDER 'sl::BODY_PARTS::RIGHT_SHOULDER'
        RIGHT_ELBOW 'sl::BODY_PARTS::RIGHT_ELBOW'
        RIGHT_WRIST 'sl::BODY_PARTS::RIGHT_WRIST'
        LEFT_SHOULDER 'sl::BODY_PARTS::LEFT_SHOULDER'
        LEFT_ELBOW 'sl::BODY_PARTS::LEFT_ELBOW'
        LEFT_WRIST 'sl::BODY_PARTS::LEFT_WRIST'
        RIGHT_HIP 'sl::BODY_PARTS::RIGHT_HIP'
        RIGHT_KNEE 'sl::BODY_PARTS::RIGHT_KNEE'
        RIGHT_ANKLE 'sl::BODY_PARTS::RIGHT_ANKLE'
        LEFT_HIP 'sl::BODY_PARTS::LEFT_HIP'
        LEFT_KNEE 'sl::BODY_PARTS::LEFT_KNEE'
        LEFT_ANKLE 'sl::BODY_PARTS::LEFT_ANKLE'
        RIGHT_EYE 'sl::BODY_PARTS::RIGHT_EYE'
        LEFT_EYE 'sl::BODY_PARTS::LEFT_EYE'
        RIGHT_EAR 'sl::BODY_PARTS::RIGHT_EAR'
        LEFT_EAR 'sl::BODY_PARTS::LEFT_EAR'
        LAST 'sl::BODY_PARTS::LAST'

    int getIdx(BODY_PARTS part)

    cdef cppclass Mat 'sl::Mat':
        String name
        bool verbose

        Mat()
        Mat(size_t width, size_t height, MAT_TYPE mat_type, MEM memory_type)
        Mat(size_t width, size_t height, MAT_TYPE mat_type, unsigned char *ptr, size_t step, MEM memory_type)
        Mat(size_t width, size_t height, MAT_TYPE mat_type, unsigned char *ptr_cpu, size_t step_cpu,
            unsigned char *ptr_gpu, size_t step_gpu)
        Mat(Resolution resolution, MAT_TYPE mat_type, MEM memory_type)
        Mat(Resolution resolution, MAT_TYPE mat_type, unsigned char *ptr, size_t step, MEM memory_type)
        Mat(const Mat &mat)

        void alloc(size_t width, size_t height, MAT_TYPE mat_type, MEM memory_type)
        void alloc(Resolution resolution, MAT_TYPE mat_type, MEM memory_type)
        void free(MEM memory_type)
        Mat &operator=(const Mat &that)
        ERROR_CODE updateCPUfromGPU()
        ERROR_CODE updateGPUfromCPU()
        ERROR_CODE copyTo(Mat &dst, COPY_TYPE cpyType) const
        ERROR_CODE setFrom(const Mat &src, COPY_TYPE cpyType) const
        ERROR_CODE read(const char* filePath)
        ERROR_CODE write(const char* filePath, MEM memory_type, int compression_level)
        size_t getWidth() const
        size_t getHeight() const
        Resolution getResolution() const
        size_t getChannels() const
        MAT_TYPE getDataType() const
        MEM getMemoryType() const
        size_t getStepBytes(MEM memory_type)
        size_t getStep(MEM memory_type)
        size_t getPixelBytes()
        size_t getWidthBytes()
        String getInfos()
        bool isInit()
        bool isMemoryOwner()
        ERROR_CODE clone(const Mat &src)
        ERROR_CODE move(Mat &dst)

        @staticmethod
        void swap(Mat &mat1, Mat &mat2)

    cdef cppclass Rotation(Matrix3f):
        Rotation() except +
        Rotation(const Rotation &rotation) except +
        Rotation(const Matrix3f &mat) except +
        Rotation(const Orientation &orientation) except +
        Rotation(const float angle, const Translation &axis) except +
        void setOrientation(const Orientation &orientation)
        Orientation getOrientation() const
        Vector3[float] getRotationVector()
        void setRotationVector(const Vector3[float] &vec_rot)
        Vector3[float] getEulerAngles(bool radian) const
        void setEulerAngles(const Vector3[float] &euler_angles, bool radian)


    cdef cppclass Translation(Vector3):
        Translation()
        Translation(const Translation &translation)
        Translation(float t1, float t2, float t3)
        Translation operator*(const Orientation &mat) const
        void normalize()

        @staticmethod
        Translation normalize(const Translation &tr)
        float dot(const Translation &tr1,const Translation &tr2)
        float &operator()(int x)


    cdef cppclass Orientation(Vector4):
        Orientation()
        Orientation(const Orientation &orientation)
        Orientation(const Vector4[float] &input)
        Orientation(const Rotation &rotation)
        Orientation(const Translation &tr1, const Translation &tr2)
        float operator()(int x)
        Orientation operator*(const Orientation &orientation) const
        void setRotationMatrix(const Rotation &rotation)
        Rotation getRotationMatrix() const
        void setIdentity()
        @staticmethod
        Orientation identity()
        void setZeros()
        @staticmethod
        Orientation zeros()
        void normalise()

        @staticmethod
        Orientation normalise(const Orientation &orient)


    cdef cppclass Transform(Matrix4f):
        Transform()
        Transform(Transform &motion)
        Transform(const Matrix4f &mat)
        Transform(const Rotation &rotation, const Translation &translation)
        Transform(const Orientation &orientation, const Translation &translation)
        void setRotationMatrix(const Rotation &rotation)
        Rotation getRotationMatrix() const
        void setTranslation(const Translation &translation)
        Translation getTranslation() const
        void setOrientation(const Orientation &orientation)
        Orientation getOrientation() const
        Vector3[float] getRotationVector()
        void setRotationVector(const Vector3[float] &vec_rot)
        Vector3[float] getEulerAngles(bool radian) const
        void setEulerAngles(const Vector3[float] &euler_angles, bool radian)

ctypedef unsigned char uchar1
ctypedef Vector2[unsigned char] uchar2
ctypedef Vector3[unsigned char] uchar3
ctypedef Vector4[unsigned char] uchar4

ctypedef float float1
ctypedef Vector2[float] float2
ctypedef Vector3[float] float3
ctypedef Vector4[float] float4

ctypedef unsigned short ushort1

cdef extern from "Utils.cpp" namespace "sl":

    Mat matResolution(Resolution resolution, MAT_TYPE mat_type, uchar1 *ptr_cpu, size_t step_cpu,
                      uchar1 *ptr_gpu, size_t step_gpu)

    ERROR_CODE setToUchar1(Mat &mat, uchar1 value, MEM memory_type)
    ERROR_CODE setToUchar2(Mat &mat, uchar2 value, MEM memory_type)
    ERROR_CODE setToUchar3(Mat &mat, uchar3 value, MEM memory_type)
    ERROR_CODE setToUchar4(Mat &mat, uchar4 value, MEM memory_type)
    ERROR_CODE setToUshort1(Mat &mat, ushort1 value, MEM memory_type)

    ERROR_CODE setToFloat1(Mat &mat, float1 value, MEM memory_type)
    ERROR_CODE setToFloat2(Mat &mat, float2 value, MEM memory_type)
    ERROR_CODE setToFloat3(Mat &mat, float3 value, MEM memory_type)
    ERROR_CODE setToFloat4(Mat &mat, float4 value, MEM memory_type)

    ERROR_CODE setValueUchar1(Mat &mat, size_t x, size_t y, uchar1 value, MEM memory_type)
    ERROR_CODE setValueUchar2(Mat &mat, size_t x, size_t y, uchar2 value, MEM memory_type)
    ERROR_CODE setValueUchar3(Mat &mat, size_t x, size_t y, uchar3 value, MEM memory_type)
    ERROR_CODE setValueUchar4(Mat &mat, size_t x, size_t y, uchar4 value, MEM memory_type)
    ERROR_CODE setValueUshort1(Mat &mat, size_t x, size_t y, ushort1 value, MEM memory_type)

    ERROR_CODE setValueFloat1(Mat &mat, size_t x, size_t y, float1 value, MEM memory_type)
    ERROR_CODE setValueFloat2(Mat &mat, size_t x, size_t y, float2 value, MEM memory_type)
    ERROR_CODE setValueFloat3(Mat &mat, size_t x, size_t y, float3 value, MEM memory_type)
    ERROR_CODE setValueFloat4(Mat &mat, size_t x, size_t y, float4 value, MEM memory_type)

    ERROR_CODE getValueUchar1(Mat &mat, size_t x, size_t y, uchar1 *value, MEM memory_type)
    ERROR_CODE getValueUchar2(Mat &mat, size_t x, size_t y, Vector2[uchar1] *value, MEM memory_type)
    ERROR_CODE getValueUchar3(Mat &mat, size_t x, size_t y, Vector3[uchar1] *value, MEM memory_type)
    ERROR_CODE getValueUchar4(Mat &mat, size_t x, size_t y, Vector4[uchar1] *value, MEM memory_type)
    ERROR_CODE getValueUshort1(Mat &mat, size_t x, size_t y, ushort1 *value, MEM memory_type)

    ERROR_CODE getValueFloat1(Mat &mat, size_t x, size_t y, float1 *value, MEM memory_type)
    ERROR_CODE getValueFloat2(Mat &mat, size_t x, size_t y, Vector2[float1] *value, MEM memory_type)
    ERROR_CODE getValueFloat3(Mat &mat, size_t x, size_t y, Vector3[float1] *value, MEM memory_type)
    ERROR_CODE getValueFloat4(Mat &mat, size_t x, size_t y, Vector4[float1] *value, MEM memory_type)

    uchar1 *getPointerUchar1(Mat &mat, MEM memory_type)
    uchar2 *getPointerUchar2(Mat &mat, MEM memory_type)
    uchar3 *getPointerUchar3(Mat &mat, MEM memory_type)
    uchar4 *getPointerUchar4(Mat &mat, MEM memory_type)
    ushort1 *getPointerUshort1(Mat &mat, MEM memory_type)

    float1 *getPointerFloat1(Mat &mat, MEM memory_type)
    float2 *getPointerFloat2(Mat &mat, MEM memory_type)
    float3 *getPointerFloat3(Mat &mat, MEM memory_type)
    float4 *getPointerFloat4(Mat &mat, MEM memory_type)

ctypedef unsigned int uint

cdef extern from "sl/Camera.hpp" namespace "sl":

    ctypedef enum MESH_FILE_FORMAT 'sl::MESH_FILE_FORMAT':
        PLY 'sl::MESH_FILE_FORMAT::PLY'
        PLY_BIN 'sl::MESH_FILE_FORMAT::PLY_BIN'
        OBJ 'sl::MESH_FILE_FORMAT::OBJ'
        MESH_FILE_FORMAT_LAST 'sl::MESH_FILE_FORMAT::LAST'


    ctypedef enum MESH_TEXTURE_FORMAT 'sl::MESH_TEXTURE_FORMAT':
        RGB 'sl::MESH_TEXTURE_FORMAT::RGB'
        RGBA 'sl::MESH_TEXTURE_FORMAT::RGBA'
        MESH_TEXTURE_FORMAT_LAST 'sl::MESH_TEXTURE_FORMAT::LAST'


    ctypedef enum MESH_FILTER 'sl::MeshFilterParameters::MESH_FILTER':
        LOW 'sl::MeshFilterParameters::MESH_FILTER::LOW'
        MESH_FILTER_MEDIUM 'sl::MeshFilterParameters::MESH_FILTER::MEDIUM'
        HIGH 'sl::MeshFilterParameters::MESH_FILTER::HIGH'

    ctypedef enum PLANE_TYPE 'sl::PLANE_TYPE':
        HORIZONTAL 'sl::PLANE_TYPE::HORIZONTAL'
        VERTICAL 'sl::PLANE_TYPE::VERTICAL'
        UNKNOWN 'sl::PLANE_TYPE::UNKNOWN'
        PLANE_TYPE_LAST 'sl::PLANE_TYPE::LAST'

    cdef cppclass MeshFilterParameters 'sl::MeshFilterParameters':
        MeshFilterParameters(MESH_FILTER filtering_)
        void set(MESH_FILTER filtering_)
        bool save(String filename)
        bool load(String filename)

    cdef cppclass Chunk 'sl::Chunk':
        Chunk()
        vector[Vector4[float]] vertices
        vector[Vector3[uint]] triangles
        vector[Vector3[float]] normals
        vector[Vector2[float]] uv
        unsigned long long timestamp
        Vector3[float] barycenter
        bool has_been_updated
        void clear()

    cdef cppclass PointCloudChunk 'sl::PointCloudChunk':
        PointCloudChunk()
        vector[Vector4[float]] vertices
        vector[Vector3[float]] normals
        unsigned long long timestamp
        Vector3[float] barycenter
        bool has_been_updated
        void clear()

    cdef cppclass Mesh 'sl::Mesh':
        ctypedef vector[size_t] chunkList
        Mesh()
        vector[Chunk] chunks
        Chunk &operator[](int index)
        vector[Vector4[float]] vertices
        vector[Vector3[uint]] triangles
        vector[Vector3[float]] normals
        vector[Vector2[float]] uv
        Mat texture
        size_t getNumberOfTriangles()
        void mergeChunks(int faces_per_chunk)
        Vector3[float] getGravityEstimate()
        vector[int] getBoundaries()
        chunkList getVisibleList(Transform camera_pose)
        chunkList getSurroundingList(Transform camera_pose, float radius)
        void updateMeshFromChunkList(chunkList IDs)
        bool filter(MeshFilterParameters params, bool updateMesh)
        bool applyTexture(MESH_TEXTURE_FORMAT texture_format)
        bool save(String filename, MESH_FILE_FORMAT type, chunkList IDs)
        bool load(const String filename, bool updateMesh)
        void clear()

    cdef cppclass FusedPointCloud 'sl::FusedPointCloud':
        ctypedef vector[size_t] chunkList
        FusedPointCloud()
        vector[PointCloudChunk] chunks
        PointCloudChunk &operator[](int index)
        vector[Vector4[float]] vertices
        vector[Vector3[float]] normals
        size_t getNumberOfPoints()
        void updateFromChunkList(chunkList IDs)
        bool save(String filename, MESH_FILE_FORMAT type, chunkList IDs)
        bool load(const String filename, bool updateMesh)
        void clear()



    cdef cppclass Plane 'sl::Plane':
        Plane()
        PLANE_TYPE type
        Vector3[float] getNormal()
        Vector3[float] getCenter()
        Transform getPose()
        Vector2[float] getExtents()
        Vector4[float] getPlaneEquation()
        vector[Vector3[float]] getBounds()
        Mesh extractMesh()
        float getClosestDistance(Vector3[float] point)
        void clear()



cdef extern from 'cuda.h' :
    cdef struct CUctx_st :
        pass
    ctypedef CUctx_st* CUcontext

cdef extern from 'sl/Camera.hpp' namespace 'sl':

    ctypedef enum SPATIAL_MAP_TYPE 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE':
        MESH 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH'
        FUSED_POINT_CLOUD 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::FUSED_POINT_CLOUD'

    ctypedef enum MAPPING_RESOLUTION 'sl::SpatialMappingParameters::MAPPING_RESOLUTION':
        MAPPING_RESOLUTION_HIGH 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::HIGH'
        MAPPING_RESOLUTION_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::MEDIUM'
        MAPPING_RESOLUTION_LOW 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::LOW'


    ctypedef enum MAPPING_RANGE 'sl::SpatialMappingParameters::MAPPING_RANGE':
        SHORT 'sl::SpatialMappingParameters::MAPPING_RANGE::SHORT'
        MAPPING_RANGE_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RANGE::MEDIUM'
        LONG 'sl::SpatialMappingParameters::MAPPING_RANGE::LONG'
        AUTO 'sl::SpatialMappingParameters::MAPPING_RANGE::AUTO'

    cdef cppclass InputType 'sl::InputType':
        InputType()
        InputType(InputType &type)

        void setFromCameraID(unsigned int id)
        void setFromSerialNumber(unsigned int serial_number)
        void setFromSVOFile(String svo_input_filename)
        void setFromStream(String senderIP, unsigned short port)

    cdef cppclass InitParameters 'sl::InitParameters':
        RESOLUTION camera_resolution
        int camera_fps
        int camera_image_flip
        bool camera_disable_self_calib
        bool enable_right_side_measure
        bool svo_real_time_mode
        DEPTH_MODE depth_mode
        int depth_stabilization
        float depth_minimum_distance
        float depth_maximum_distance
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        int sdk_gpu_id
        bool sdk_verbose

        String sdk_verbose_log_file

        CUcontext sdk_cuda_ctx
        InputType input
        String optional_settings_path
        bool sensors_required
        bool enable_image_enhancement
        String optional_opencv_calibration_file

        InitParameters(RESOLUTION camera_resolution,
                       int camera_fps,
                       bool svo_real_time_mode,
                       DEPTH_MODE depth_mode,
                       UNIT coordinate_units,
                       COORDINATE_SYSTEM coordinate_system,
                       bool sdk_verbose,
                       int sdk_gpu_id,
                       float depth_minimum_distance,
                       float depth_maxium_distance,
                       bool camera_disable_self_calib,
                       bool camera_image_flip,
                       bool enable_right_side_measure,
                       String sdk_verbose_log_file,
                       int depth_stabilization,
                       CUcontext sdk_cuda_ctx,
                       InputType input,
                       String optional_settings_path,
                       bool sensors_required,
                       bool enable_image_enhancement,
                       String optional_opencv_calibration_file)

        bool save(String filename)
        bool load(String filename)

    cdef cppclass RecordingParameters 'sl::RecordingParameters':
        String video_filename
        SVO_COMPRESSION_MODE compression_mode
        unsigned int bitrate
        unsigned int target_framerate
        bool transcode_streaming_input

        RecordingParameters(String video_filename_, 
                            SVO_COMPRESSION_MODE compression_mode_,
                            unsigned int target_framerate,
                            unsigned int bitrate,
                            bool transcode_streaming_input
                            )

    cdef cppclass RuntimeParameters 'sl::RuntimeParameters':
        SENSING_MODE sensing_mode
        bool enable_depth
        int confidence_threshold
        int textureness_confidence_threshold
        int texture_confidence_threshold
        REFERENCE_FRAME measure3D_reference_frame

        RuntimeParameters(SENSING_MODE sensing_mode,
                          bool enable_depth,
                          int confidence_threshold,
                          int texture_confidence_threshold,
                          REFERENCE_FRAME measure3D_reference_frame)

        bool save(String filename)
        bool load(String filename)


    cdef cppclass PositionalTrackingParameters 'sl::PositionalTrackingParameters':
        Transform initial_world_transform
        bool enable_area_memory
        bool enable_pose_smoothing
        bool set_floor_as_origin
        String area_file_path
        bool enable_imu_fusion
        bool set_as_static

        PositionalTrackingParameters(Transform init_pos,
                           bool _enable_memory,
                           bool _enable_pose_smoothing,
                           String _area_path,
                           bool _set_floor_as_origin,
                           bool _enable_imu_fusion,
                           bool _set_as_static)

        bool save(String filename)
        bool load(String filename)

    cdef cppclass SpatialMappingParameters 'sl::SpatialMappingParameters':
        ctypedef pair[float, float] interval

        SpatialMappingParameters(MAPPING_RESOLUTION resolution,
                                 MAPPING_RANGE range,
                                 int max_memory_usage_,
                                 bool save_texture_,
                                 bool use_chunk_only_,
                                 bool reverse_vertex_order_,
                                 SPATIAL_MAP_TYPE map_type_)

        @staticmethod
        float get(MAPPING_RESOLUTION resolution)

        void set(MAPPING_RESOLUTION resolution)

        @staticmethod
        float get(MAPPING_RANGE range)

        @staticmethod
        float getRecommendedRange(MAPPING_RESOLUTION mapping_resolution, Camera &camera)

        @staticmethod
        float getRecommendedRange(float resolution_meters, Camera &camera)

        void set(MAPPING_RANGE range)

        int max_memory_usage
        bool save_texture
        bool use_chunk_only
        bool reverse_vertex_order

        SPATIAL_MAP_TYPE map_type

        const interval allowed_range
        float range_meter
        const interval allowed_resolution
        float resolution_meter

        bool save(String filename)
        bool load(String filename)


    ctypedef enum STREAMING_CODEC 'sl::STREAMING_CODEC':
        STREAMING_CODEC_H264 'sl::STREAMING_CODEC::H264'
        STREAMING_CODEC_H265 'sl::STREAMING_CODEC::H265'
        STREAMING_CODEC_LAST 'sl::STREAMING_CODEC::LAST'

    cdef struct StreamingProperties:
        String ip
        unsigned short port
        unsigned int serial_number
        int current_bitrate
        STREAMING_CODEC codec

    cdef cppclass StreamingParameters:
        STREAMING_CODEC codec
        unsigned short port
        unsigned int bitrate
        int gop_size
        bool adaptative_bitrate
        unsigned short chunk_size
        unsigned int target_framerate
        StreamingParameters(STREAMING_CODEC codec, unsigned short port_, unsigned int bitrate, int gop_size, bool adaptative_bitrate_, unsigned short chunk_size_, unsigned int target_framerate)

    cdef cppclass BatchParameters:
        bool enable
        float id_retention_time
        float latency
        BatchParameters(bool enable, float id_retention_time, float batch_duration)

    cdef cppclass ObjectDetectionParameters:
        bool image_sync
        bool enable_tracking
        bool enable_mask_output
        DETECTION_MODEL detection_model
        bool enable_body_fitting
        float max_range
        BatchParameters batch_parameters
        ObjectDetectionParameters(bool image_sync, bool enable_tracking, bool enable_mask_output, DETECTION_MODEL detection_model, bool enable_body_fitting, float max_range, BatchParameters batch_trajectories_parameters)

    cdef cppclass ObjectDetectionRuntimeParameters:
        float detection_confidence_threshold
        vector[OBJECT_CLASS] object_class_filter
        map[OBJECT_CLASS,float] object_class_detection_confidence_threshold
        ObjectDetectionRuntimeParameters(float detection_confidence_threshold, vector[OBJECT_CLASS] object_class_filter, map[OBJECT_CLASS,float] object_class_detection_confidence_threshold)

    cdef cppclass Pose:
        Pose()
        Pose(const Pose &pose)
        Pose(const Transform &pose_data, unsigned long long mtimestamp, int mconfidence)
        Translation getTranslation()
        Orientation getOrientation()
        Rotation getRotationMatrix()
        Vector3[float] getRotationVector()
        Vector3[float] getEulerAngles(bool radian)

        bool valid
        Timestamp timestamp

        Transform pose_data

        int pose_confidence
        float pose_covariance[36]

    ctypedef enum CAMERA_MOTION_STATE 'sl::SensorsData::CAMERA_MOTION_STATE':
        STATIC 'sl::SensorsData::CAMERA_MOTION_STATE::STATIC'
        MOVING 'sl::SensorsData::CAMERA_MOTION_STATE::MOVING'
        FALLING 'sl::SensorsData::CAMERA_MOTION_STATE::FALLING'
        CAMERA_MOTION_STATE_LAST 'sl::SensorsData::CAMERA_MOTION_STATE::LAST'

    cdef cppclass BarometerData 'sl::SensorsData::BarometerData':
        bool is_available
        float pressure
        float relative_altitude
        Timestamp timestamp
        float effective_rate

        BarometerData()

    ctypedef enum SENSOR_LOCATION 'sl::SensorsData::TemperatureData::SENSOR_LOCATION':
        IMU 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::IMU'
        BAROMETER 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::BAROMETER'
        ONBOARD_LEFT 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_LEFT'
        ONBOARD_RIGHT 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_RIGHT'
        SENSOR_LOCATION_LAST 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::LAST'

    cdef cppclass TemperatureData 'sl::SensorsData::TemperatureData':
        ERROR_CODE get(SENSOR_LOCATION location, float& temperature)
        map[SENSOR_LOCATION,float] temperature_map
        TemperatureData()

    cdef cppclass MagnetometerData 'sl::SensorsData::MagnetometerData':
        bool is_available
        Timestamp timestamp
        float effective_rate

        Vector3[float] magnetic_field_uncalibrated
        Vector3[float] magnetic_field_calibrated
        MagnetometerData()

    cdef cppclass IMUData 'sl::SensorsData::IMUData':
        bool is_available
        Timestamp timestamp
        Transform pose
        Matrix3f pose_covariance
        Vector3[float] angular_velocity
        Vector3[float] linear_acceleration
        Vector3[float] linear_acceleration_uncalibrated
        Matrix3f angular_velocity_covariance
        Matrix3f linear_acceleration_covariance
        IMUData()


    cdef cppclass SensorsData 'sl::SensorsData':
        SensorsData()
        SensorsData(const SensorsData &data)

        TemperatureData temperature
        BarometerData barometer
        MagnetometerData magnetometer
        IMUData imu

        CAMERA_MOTION_STATE camera_moving_state

        int image_sync_trigger

    ctypedef enum FLIP_MODE 'sl::FLIP_MODE':
        OFF 'sl::FLIP_MODE::OFF'
        ON 'sl::FLIP_MODE::ON'
        AUTO 'sl::FLIP_MODE::AUTO'
    
    String toString(FLIP_MODE o)

    Resolution getResolution(RESOLUTION resolution)

    ctypedef vector[ObjectsBatch] vectorObj
    cdef cppclass Camera 'sl::Camera':
        Camera()
        void close()
        ERROR_CODE open(InitParameters init_parameters)

        InitParameters getInitParameters()

        bool isOpened()
        ERROR_CODE grab(RuntimeParameters rt_parameters)

        RuntimeParameters getRuntimeParameters()
        

        ERROR_CODE retrieveImage(Mat &mat, VIEW view, MEM type, Resolution resolution)
        ERROR_CODE retrieveMeasure(Mat &mat, MEASURE measure, MEM type, Resolution resolution)

        void setSVOPosition(int frame_number)
        int getSVOPosition()
        int getSVONumberOfFrames()
        void setCameraSettings(VIDEO_SETTINGS settings, int value)
        ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, Rect roi, SIDE eye, bool reset)
        int getCameraSettings(VIDEO_SETTINGS setting)
        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, Rect &roi, SIDE eye)
        float getCurrentFPS()
        Timestamp getTimestamp(TIME_REFERENCE reference_time)
        unsigned int getFrameDroppedCount()
        CameraInformation getCameraInformation(Resolution resizer)

        ERROR_CODE enablePositionalTracking(PositionalTrackingParameters tracking_params)
        POSITIONAL_TRACKING_STATE getPosition(Pose &camera_pose, REFERENCE_FRAME reference_frame)
        ERROR_CODE saveAreaMap(String area_file_path)
        AREA_EXPORTING_STATE getAreaExportState()

        PositionalTrackingParameters getPositionalTrackingParameters()
        bool isPositionalTrackingEnabled()
        void disablePositionalTracking(String area_file_path)
        ERROR_CODE resetPositionalTracking(Transform &path)
        ERROR_CODE getSensorsData(SensorsData &imu_data, TIME_REFERENCE reference_time)
        ERROR_CODE setIMUPrior(Transform &transfom)

        ERROR_CODE enableSpatialMapping(SpatialMappingParameters spatial_mapping_parameters)
        void pauseSpatialMapping(bool status)
        SPATIAL_MAPPING_STATE getSpatialMappingState()

        SpatialMappingParameters getSpatialMappingParameters()

        void disableSpatialMapping()

        void requestSpatialMapAsync()
        ERROR_CODE getSpatialMapRequestStatusAsync()
        ERROR_CODE retrieveSpatialMapAsync(Mesh &mesh)
        ERROR_CODE retrieveSpatialMapAsync(FusedPointCloud &fpc)
        ERROR_CODE extractWholeSpatialMap(Mesh &mesh)
        ERROR_CODE extractWholeSpatialMap(FusedPointCloud &fpc)

        ERROR_CODE findPlaneAtHit(Vector2[uint] coord, Plane &plane)
        ERROR_CODE findFloorPlane(Plane &plane, Transform &resetTrackingFloorFrame, float floor_height_prior, Rotation world_orientation_prior, float floor_height_prior_tolerance)

        ERROR_CODE enableRecording(RecordingParameters recording_params)

        RecordingParameters getRecordingParameters()
        StreamingParameters getStreamingParameters()

        RecordingStatus getRecordingStatus()
        void pauseRecording(bool status)

        void disableRecording()

        ERROR_CODE enableStreaming(StreamingParameters streaming_parameters)
        void disableStreaming()
        bool isStreamingEnabled()

        ERROR_CODE enableObjectDetection(ObjectDetectionParameters object_detection_parameters)
        void disableObjectDetection()
        ERROR_CODE retrieveObjects(Objects &objects, ObjectDetectionRuntimeParameters parameters)
        ERROR_CODE getObjectsBatch(vector[ObjectsBatch] &trajectories)
        ObjectDetectionParameters getObjectDetectionParameters()
        void pauseObjectDetection(bool status)
        void updateSelfCalibration()

        @staticmethod
        String getSDKVersion()

        @staticmethod
        vector[DeviceProperties] getDeviceList()

        @staticmethod
        vector[StreamingProperties] getStreamingDeviceList()

        @staticmethod
        ERROR_CODE reboot(int sn)

cdef extern from "Utils.cpp" namespace "sl":
    ObjectDetectionRuntimeParameters* create_object_detection_runtime_parameters(float confidence_threshold, vector[int] object_vector, map[int,float] object_confidence_map)
