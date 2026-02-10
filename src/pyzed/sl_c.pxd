########################################################################
#
# Copyright (c) 2025, STEREOLABS.
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
from libc.stdint cimport uint8_t, uint64_t, int64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libc.string cimport const_char
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

ctypedef unsigned int uint

cdef extern from "<array>" namespace "std" nogil:
  cdef cppclass array6 "std::array<float, 6>":
    array6() except+
    float& operator[](size_t)

cdef extern from "<array>" namespace "std" nogil:
  cdef cppclass array9 "std::array<double, 9>":
    array9() except+
    double& operator[](size_t)

cdef extern from 'cuda_runtime.h':
    ctypedef void* cudaStream_t

cdef extern from "Utils.cpp" namespace "sl":
    string to_str(String sl_str)

cdef extern from "sl/Camera.hpp":
    cdef int getZEDSDKRuntimeVersion(int &major, int &minor, int &patch)

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

    bool operator<(const Timestamp &lhs, const Timestamp &rhs)
    bool operator>(const Timestamp &lhs, const Timestamp &rhs)
    bool operator<=(const Timestamp &lhs, const Timestamp &rhs)
    bool operator>=(const Timestamp &lhs, const Timestamp &rhs)
    bool operator==(const Timestamp &lhs, const Timestamp &rhs)
    bool operator!=(const Timestamp &lhs, const Timestamp &rhs)
    Timestamp operator+(Timestamp lhs, const Timestamp &rhs)
    Timestamp operator-(Timestamp lhs, const Timestamp &rhs)
    Timestamp operator/(Timestamp lhs, const Timestamp &rhs)
    Timestamp operator*(Timestamp lhs, const Timestamp &rhs)

    ctypedef enum ERROR_CODE "sl::ERROR_CODE" :
        ERROR_CODE_SENSOR_CONFIGURATION_CHANGED 'sl::ERROR_CODE::SENSOR_CONFIGURATION_CHANGED'
        ERROR_CODE_POTENTIAL_CALIBRATION_ISSUE 'sl::ERROR_CODE::POTENTIAL_CALIBRATION_ISSUE'
        ERROR_CODE_CONFIGURATION_FALLBACK 'sl::ERROR_CODE::CONFIGURATION_FALLBACK'
        ERROR_CODE_SENSORS_DATA_REQUIRED 'sl::ERROR_CODE::SENSORS_DATA_REQUIRED'
        ERROR_CODE_CORRUPTED_FRAME 'sl::ERROR_CODE::CORRUPTED_FRAME'
        ERROR_CODE_CAMERA_REBOOTING 'sl::ERROR_CODE::CAMERA_REBOOTING'
        ERROR_CODE_SUCCESS 'sl::ERROR_CODE::SUCCESS'
        ERROR_CODE_FAILURE 'sl::ERROR_CODE::FAILURE'
        ERROR_CODE_NO_GPU_COMPATIBLE 'sl::ERROR_CODE::NO_GPU_COMPATIBLE'
        ERROR_CODE_NOT_ENOUGH_GPU_MEMORY 'sl::ERROR_CODE::NOT_ENOUGH_GPU_MEMORY'
        ERROR_CODE_CAMERA_NOT_DETECTED 'sl::ERROR_CODE::CAMERA_NOT_DETECTED'
        ERROR_CODE_SENSORS_NOT_INITIALIZED 'sl::ERROR_CODE::SENSORS_NOT_INITIALIZED' 
        ERROR_CODE_SENSORS_NOT_AVAILABLE 'sl::ERROR_CODE::SENSORS_NOT_AVAILABLE'
        ERROR_CODE_INVALID_RESOLUTION 'sl::ERROR_CODE::INVALID_RESOLUTION'
        ERROR_CODE_LOW_USB_BANDWIDTH 'sl::ERROR_CODE::LOW_USB_BANDWIDTH'
        ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE 'sl::ERROR_CODE::CALIBRATION_FILE_NOT_AVAILABLE'
        ERROR_CODE_INVALID_CALIBRATION_FILE 'sl::ERROR_CODE::INVALID_CALIBRATION_FILE'
        ERROR_CODE_INVALID_SVO_FILE 'sl::ERROR_CODE::INVALID_SVO_FILE'
        ERROR_CODE_SVO_RECORDING_ERROR 'sl::ERROR_CODE::SVO_RECORDING_ERROR'
        ERROR_CODE_SVO_UNSUPPORTED_COMPRESSION 'sl::ERROR_CODE::SVO_UNSUPPORTED_COMPRESSION'
        ERROR_CODE_END_OF_SVOFILE_REACHED 'sl::ERROR_CODE::END_OF_SVOFILE_REACHED'
        ERROR_CODE_INVALID_COORDINATE_SYSTEM 'sl::ERROR_CODE::INVALID_COORDINATE_SYSTEM'
        ERROR_CODE_INVALID_FIRMWARE 'sl::ERROR_CODE::INVALID_FIRMWARE'
        ERROR_CODE_INVALID_FUNCTION_PARAMETERS 'sl::ERROR_CODE::INVALID_FUNCTION_PARAMETERS'
        ERROR_CODE_CUDA_ERROR 'sl::ERROR_CODE::CUDA_ERROR'
        ERROR_CODE_CAMERA_NOT_INITIALIZED 'sl::ERROR_CODE::CAMERA_NOT_INITIALIZED'
        ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE 'sl::ERROR_CODE::NVIDIA_DRIVER_OUT_OF_DATE'
        ERROR_CODE_INVALID_FUNCTION_CALL 'sl::ERROR_CODE::INVALID_FUNCTION_CALL'
        ERROR_CODE_CORRUPTED_SDK_INSTALLATION 'sl::ERROR_CODE::CORRUPTED_SDK_INSTALLATION'
        ERROR_CODE_INCOMPATIBLE_SDK_VERSION 'sl::ERROR_CODE::INCOMPATIBLE_SDK_VERSION'
        ERROR_CODE_INVALID_AREA_FILE 'sl::ERROR_CODE::INVALID_AREA_FILE'
        ERROR_CODE_INCOMPATIBLE_AREA_FILE 'sl::ERROR_CODE::INCOMPATIBLE_AREA_FILE'
        ERROR_CODE_CAMERA_FAILED_TO_SETUP 'sl::ERROR_CODE::CAMERA_FAILED_TO_SETUP'
        ERROR_CODE_CAMERA_DETECTION_ISSUE 'sl::ERROR_CODE::CAMERA_DETECTION_ISSUE'
        ERROR_CODE_CANNOT_START_CAMERA_STREAM 'sl::ERROR_CODE::CANNOT_START_CAMERA_STREAM'
        ERROR_CODE_NO_GPU_DETECTED 'sl::ERROR_CODE::NO_GPU_DETECTED'
        ERROR_CODE_PLANE_NOT_FOUND 'sl::ERROR_CODE::PLANE_NOT_FOUND'
        ERROR_CODE_MODULE_NOT_COMPATIBLE_WITH_CAMERA 'sl::ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_CAMERA'
        ERROR_CODE_MOTION_SENSORS_REQUIRED 'sl::ERROR_CODE::MOTION_SENSORS_REQUIRED'
        ERROR_CODE_MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION 'sl::ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION'
        ERROR_CODE_DRIVER_FAILURE 'sl::ERROR_CODE::DRIVER_FAILURE'
        ERROR_CODE_LAST 'sl::ERROR_CODE::LAST'

    cdef String toString(const ERROR_CODE &o)
    String toVerbose(const ERROR_CODE &o)

    cdef cppclass AI_Model_status 'sl::AI_Model_status':
        bool downloaded
        bool optimized

    void sleep_ms(int time)

    void sleep_us(int time)

    ctypedef enum MODEL "sl::MODEL":
        MODEL_ZED 'sl::MODEL::ZED'
        MODEL_ZED_M 'sl::MODEL::ZED_M'
        MODEL_ZED2 'sl::MODEL::ZED2'
        MODEL_ZED2i 'sl::MODEL::ZED2i'
        MODEL_ZED_X 'sl::MODEL::ZED_X'
        MODEL_ZED_XM 'sl::MODEL::ZED_XM'
        MODEL_ZED_X_HDR 'sl::MODEL::ZED_X_HDR'
        MODEL_ZED_X_HDR_MINI 'sl::MODEL::ZED_X_HDR_MINI'
        MODEL_ZED_X_HDR_MAX 'sl::MODEL::ZED_X_HDR_MAX'
        MODEL_VIRTUAL_ZED_X 'sl::MODEL::VIRTUAL_ZED_X'
        MODEL_ZED_XONE_GS 'sl::MODEL::ZED_XONE_GS'
        MODEL_ZED_XONE_UHD 'sl::MODEL::ZED_XONE_UHD'
        MODEL_ZED_XONE_HDR 'sl::MODEL::ZED_XONE_HDR'
        MODEL_LAST 'sl::MODEL::LAST'

    cdef String toString(const MODEL &o)

    ctypedef enum CAMERA_STATE 'sl::CAMERA_STATE':
        CAMERA_STATE_AVAILABLE 'sl::CAMERA_STATE::AVAILABLE'
        CAMERA_STATE_NOT_AVAILABLE 'sl::CAMERA_STATE::NOT_AVAILABLE'
        CAMERA_STATE_LAST 'sl::CAMERA_STATE::LAST'

    cdef String toString(const CAMERA_STATE &o)

    ctypedef enum SENSOR_STATE 'sl::SENSOR_STATE':
        SENSOR_STATE_AVAILABLE 'sl::SENSOR_STATE::AVAILABLE'
        SENSOR_STATE_NOT_AVAILABLE 'sl::SENSOR_STATE::NOT_AVAILABLE'
        SENSOR_STATE_LAST 'sl::SENSOR_STATE::LAST'

    cdef String toString(const SENSOR_STATE &o)

    cdef cppclass String 'sl::String':
        String()
        String(const char *data)
        void set(const char *data)
        const char *get() const
        bool empty() const
        string std_str() const

    cdef cppclass DeviceProperties 'sl::DeviceProperties':
        DeviceProperties()
        CAMERA_STATE camera_state
        int id
        String path
        int i2c_port
        MODEL camera_model
        uint serial_number
        int gmsl_port
        unsigned char identifier[]
        String camera_badge
        String camera_sensor_model
        String camera_name
        INPUT_TYPE input_type
        unsigned char sensor_address_left
        unsigned char sensor_address_right

    cdef String toString(const DeviceProperties &o)

    cdef cppclass SVOData 'sl::SVOData':
        SVOData()
        bool setContent(string &s)
        bool getContent(string &s)
        Timestamp timestamp_ns
        string key
        vector[uint8_t] content


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

    cdef cppclass Matrix3f 'sl::Matrix3f':
        float r[]
        Matrix3f()
        Matrix3f(float data[])
        Matrix3f(const Matrix3f &mat)
        Matrix3f operator*(const Matrix3f &mat) const
        Matrix3f operator*(const float &scalar) const
        Matrix3f operator+(const Matrix3f &mat) const
        Matrix3f operator+(const float &scalar) const
        Matrix3f operator-(const Matrix3f &mat) const
        Matrix3f operator-(const float &scalar) const
        bool operator==(const Matrix3f &mat) const
        bool operator!=(const Matrix3f &mat) const
        float& operator()(int row, int col)
        void inverse()
        Matrix3f inverse(const Matrix3f &rotation)
        void transpose()
        Matrix3f transpose(const Matrix3f &rotation)
        void setIdentity()
        @staticmethod
        Matrix3f identity()
        void setZeros()
        @staticmethod
        Matrix3f zeros()
        String getInfos()
        String matrix_name

    cdef cppclass Matrix4f 'sl::Matrix4f':
        float m[]
        Matrix4f()
        Matrix4f(float data[])
        Matrix4f(const Matrix4f &mat)
        Matrix4f operator*(const Matrix4f &mat) const
        Matrix4f operator*(const Vector4[float] &vect) const
        Matrix4f operator*(const float &scalar) const
        Matrix4f operator+(const Matrix4f &mat) const
        Matrix4f operator+(const float &scalar) const
        Matrix4f operator-(const Matrix4f &mat) const
        Matrix4f operator-(const float &scalar) const
        bool operator==(const Matrix4f  &mat) const
        bool operator!=(const Matrix4f &mat) const
        float& operator()(const int row, const int col)
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
        UNIT_MILLIMETER 'sl::UNIT::MILLIMETER'
        UNIT_CENTIMETER 'sl::UNIT::CENTIMETER'
        UNIT_METER 'sl::UNIT::METER'
        UNIT_INCH 'sl::UNIT::INCH'
        UNIT_FOOT 'sl::UNIT::FOOT'
        UNIT_LAST 'sl::UNIT::LAST'

    cdef String toString(const UNIT &o)
    cdef bool fromString(const String &s, UNIT &o)
    cdef UNIT str2unit(String unit_str)

    ctypedef enum COORDINATE_SYSTEM 'sl::COORDINATE_SYSTEM':
        COORDINATE_SYSTEM_IMAGE 'sl::COORDINATE_SYSTEM::IMAGE'
        COORDINATE_SYSTEM_LEFT_HANDED_Y_UP 'sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP'
        COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP'
        COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP'
        COORDINATE_SYSTEM_LEFT_HANDED_Z_UP 'sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP'
        COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD 'sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD'
        COORDINATE_SYSTEM_LAST 'sl::COORDINATE_SYSTEM::LAST'

    cdef String toString(const COORDINATE_SYSTEM &o)
    cdef bool fromString(const String &s, COORDINATE_SYSTEM &o)
    cdef Matrix3f getCoordinateTransformConversion3f(COORDINATE_SYSTEM src, COORDINATE_SYSTEM dst)
    cdef Matrix4f getCoordinateTransformConversion4f(COORDINATE_SYSTEM src, COORDINATE_SYSTEM dst)
    cdef ERROR_CODE convertCoordinateSystemTransform "convertCoordinateSystem"(Transform &transform, COORDINATE_SYSTEM src, COORDINATE_SYSTEM dst)
    cdef ERROR_CODE convertCoordinateSystemMat "convertCoordinateSystem"(Mat &mat, COORDINATE_SYSTEM src, COORDINATE_SYSTEM dst, MEM memory_type, cudaStream_t stream)
    cdef float getUnitScale(UNIT src, UNIT dst)
    cdef ERROR_CODE convertUnitTransform "convertUnit"(Transform &transform, UNIT src, UNIT dst)
    cdef ERROR_CODE convertUnitMat "convertUnit"(Mat &mat, UNIT src, UNIT dst, MEM memory_type, cudaStream_t stream)
    cdef ERROR_CODE applyTransform(Mat &mat, const Transform &transform, MEM memory_type, cudaStream_t stream)
    ctypedef enum SIDE 'sl::SIDE':
        SIDE_LEFT 'sl::SIDE::LEFT'
        SIDE_RIGHT 'sl::SIDE::RIGHT'
        SIDE_BOTH 'sl::SIDE::BOTH'

    ctypedef enum RESOLUTION 'sl::RESOLUTION':
        RESOLUTION_HD4K 'sl::RESOLUTION::HD4K'
        RESOLUTION_QHDPLUS 'sl::RESOLUTION::QHDPLUS'
        RESOLUTION_HD2K 'sl::RESOLUTION::HD2K'
        RESOLUTION_HD1080 'sl::RESOLUTION::HD1080'
        RESOLUTION_HD1200 'sl::RESOLUTION::HD1200'
        RESOLUTION_HD1536 'sl::RESOLUTION::HD1536'
        RESOLUTION_HD720 'sl::RESOLUTION::HD720'
        RESOLUTION_SVGA 'sl::RESOLUTION::SVGA'
        RESOLUTION_VGA 'sl::RESOLUTION::VGA'
        RESOLUTION_AUTO 'sl::RESOLUTION::AUTO'
        RESOLUTION_LAST 'sl::RESOLUTION::LAST'

    cdef String toString(const RESOLUTION &o)
    cdef bool fromString(const String &s, RESOLUTION &o)

    ctypedef enum VIDEO_SETTINGS 'sl::VIDEO_SETTINGS':
        VIDEO_SETTINGS_BRIGHTNESS 'sl::VIDEO_SETTINGS::BRIGHTNESS'
        VIDEO_SETTINGS_CONTRAST 'sl::VIDEO_SETTINGS::CONTRAST'
        VIDEO_SETTINGS_HUE 'sl::VIDEO_SETTINGS::HUE'
        VIDEO_SETTINGS_SATURATION 'sl::VIDEO_SETTINGS::SATURATION'
        VIDEO_SETTINGS_SHARPNESS 'sl::VIDEO_SETTINGS::SHARPNESS'
        VIDEO_SETTINGS_GAMMA 'sl::VIDEO_SETTINGS::GAMMA'
        VIDEO_SETTINGS_GAIN 'sl::VIDEO_SETTINGS::GAIN'
        VIDEO_SETTINGS_EXPOSURE 'sl::VIDEO_SETTINGS::EXPOSURE'
        VIDEO_SETTINGS_AEC_AGC 'sl::VIDEO_SETTINGS::AEC_AGC'
        VIDEO_SETTINGS_AEC_AGC_ROI 'sl::VIDEO_SETTINGS::AEC_AGC_ROI'
        VIDEO_SETTINGS_WHITEBALANCE_TEMPERATURE 'sl::VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE'
        VIDEO_SETTINGS_WHITEBALANCE_AUTO 'sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO'
        VIDEO_SETTINGS_LED_STATUS 'sl::VIDEO_SETTINGS::LED_STATUS'
        VIDEO_SETTINGS_EXPOSURE_TIME 'sl::VIDEO_SETTINGS::EXPOSURE_TIME'
        VIDEO_SETTINGS_ANALOG_GAIN 'sl::VIDEO_SETTINGS::ANALOG_GAIN'
        VIDEO_SETTINGS_DIGITAL_GAIN 'sl::VIDEO_SETTINGS::DIGITAL_GAIN'
        VIDEO_SETTINGS_AUTO_EXPOSURE_TIME_RANGE 'sl::VIDEO_SETTINGS::AUTO_EXPOSURE_TIME_RANGE'
        VIDEO_SETTINGS_AUTO_ANALOG_GAIN_RANGE 'sl::VIDEO_SETTINGS::AUTO_ANALOG_GAIN_RANGE'
        VIDEO_SETTINGS_AUTO_DIGITAL_GAIN_RANGE 'sl::VIDEO_SETTINGS::AUTO_DIGITAL_GAIN_RANGE'
        VIDEO_SETTINGS_EXPOSURE_COMPENSATION 'sl::VIDEO_SETTINGS::EXPOSURE_COMPENSATION'
        VIDEO_SETTINGS_DENOISING 'sl::VIDEO_SETTINGS::DENOISING'
        VIDEO_SETTINGS_SCENE_ILLUMINANCE 'sl::VIDEO_SETTINGS::SCENE_ILLUMINANCE'
        VIDEO_SETTINGS_LAST 'sl::VIDEO_SETTINGS::LAST'

    cdef String toString(const VIDEO_SETTINGS &o)

    ctypedef enum DEPTH_MODE 'sl::DEPTH_MODE':
        DEPTH_MODE_NONE 'sl::DEPTH_MODE::NONE'
        DEPTH_MODE_PERFORMANCE 'sl::DEPTH_MODE::PERFORMANCE'
        DEPTH_MODE_QUALITY 'sl::DEPTH_MODE::QUALITY'
        DEPTH_MODE_ULTRA 'sl::DEPTH_MODE::ULTRA'
        DEPTH_MODE_NEURAL_LIGHT 'sl::DEPTH_MODE::NEURAL_LIGHT'
        DEPTH_MODE_NEURAL 'sl::DEPTH_MODE::NEURAL'
        DEPTH_MODE_NEURAL_PLUS 'sl::DEPTH_MODE::NEURAL_PLUS'
        DEPTH_MODE_LAST 'sl::DEPTH_MODE::LAST'

    cdef String toString(const DEPTH_MODE &o)
    cdef bool fromString(const String &s, DEPTH_MODE &o)

    ctypedef enum MEASURE 'sl::MEASURE':
        MEASURE_DISPARITY 'sl::MEASURE::DISPARITY'
        MEASURE_DEPTH 'sl::MEASURE::DEPTH'
        MEASURE_CONFIDENCE 'sl::MEASURE::CONFIDENCE'
        MEASURE_XYZ 'sl::MEASURE::XYZ'
        MEASURE_XYZRGBA 'sl::MEASURE::XYZRGBA'
        MEASURE_XYZBGRA 'sl::MEASURE::XYZBGRA'
        MEASURE_XYZARGB 'sl::MEASURE::XYZARGB'
        MEASURE_XYZABGR 'sl::MEASURE::XYZABGR'
        MEASURE_NORMALS 'sl::MEASURE::NORMALS'
        MEASURE_DISPARITY_RIGHT 'sl::MEASURE::DISPARITY_RIGHT'
        MEASURE_DEPTH_RIGHT 'sl::MEASURE::DEPTH_RIGHT'
        MEASURE_XYZ_RIGHT 'sl::MEASURE::XYZ_RIGHT'
        MEASURE_XYZRGBA_RIGHT 'sl::MEASURE::XYZRGBA_RIGHT'
        MEASURE_XYZBGRA_RIGHT 'sl::MEASURE::XYZBGRA_RIGHT'
        MEASURE_XYZARGB_RIGHT 'sl::MEASURE::XYZARGB_RIGHT'
        MEASURE_XYZABGR_RIGHT 'sl::MEASURE::XYZABGR_RIGHT'
        MEASURE_NORMALS_RIGHT 'sl::MEASURE::NORMALS_RIGHT'
        MEASURE_DEPTH_U16_MM 'sl::MEASURE::DEPTH_U16_MM'
        MEASURE_DEPTH_U16_MM_RIGHT 'sl::MEASURE::DEPTH_U16_MM_RIGHT'
        MEASURE_LAST 'sl::MEASURE::LAST'

    cdef String toString(const MEASURE &o)

    ctypedef enum VIEW 'sl::VIEW':
        VIEW_LEFT 'sl::VIEW::LEFT'
        VIEW_RIGHT 'sl::VIEW::RIGHT'
        VIEW_LEFT_GRAY 'sl::VIEW::LEFT_GRAY'
        VIEW_RIGHT_GRAY 'sl::VIEW::RIGHT_GRAY'
        VIEW_LEFT_NV12_UNRECTIFIED 'sl::VIEW::LEFT_NV12_UNRECTIFIED'
        VIEW_RIGHT_NV12_UNRECTIFIED 'sl::VIEW::RIGHT_NV12_UNRECTIFIED'
        VIEW_LEFT_UNRECTIFIED 'sl::VIEW::LEFT_UNRECTIFIED'
        VIEW_RIGHT_UNRECTIFIED 'sl::VIEW::RIGHT_UNRECTIFIED'
        VIEW_LEFT_UNRECTIFIED_GRAY 'sl::VIEW::LEFT_UNRECTIFIED_GRAY'
        VIEW_RIGHT_UNRECTIFIED_GRAY 'sl::VIEW::RIGHT_UNRECTIFIED_GRAY'
        VIEW_SIDE_BY_SIDE 'sl::VIEW::SIDE_BY_SIDE'
        VIEW_DEPTH 'sl::VIEW::DEPTH'
        VIEW_CONFIDENCE 'sl::VIEW::CONFIDENCE'
        VIEW_NORMALS 'sl::VIEW::NORMALS'
        VIEW_DEPTH_RIGHT 'sl::VIEW::DEPTH_RIGHT'
        VIEW_NORMALS_RIGHT 'sl::VIEW::NORMALS_RIGHT'
        VIEW_LEFT_BGRA 'sl::VIEW::LEFT_BGRA'
        VIEW_LEFT_BGR 'sl::VIEW::LEFT_BGR'
        VIEW_RIGHT_BGRA 'sl::VIEW::RIGHT_BGRA'
        VIEW_RIGHT_BGR 'sl::VIEW::RIGHT_BGR'
        VIEW_LEFT_UNRECTIFIED_BGRA 'sl::VIEW::LEFT_UNRECTIFIED_BGRA'
        VIEW_LEFT_UNRECTIFIED_BGR 'sl::VIEW::LEFT_UNRECTIFIED_BGR'
        VIEW_RIGHT_UNRECTIFIED_BGRA 'sl::VIEW::RIGHT_UNRECTIFIED_BGRA'
        VIEW_RIGHT_UNRECTIFIED_BGR 'sl::VIEW::RIGHT_UNRECTIFIED_BGR'
        VIEW_SIDE_BY_SIDE_BGRA 'sl::VIEW::SIDE_BY_SIDE_BGRA'
        VIEW_SIDE_BY_SIDE_BGR 'sl::VIEW::SIDE_BY_SIDE_BGR'
        VIEW_SIDE_BY_SIDE_GRAY 'sl::VIEW::SIDE_BY_SIDE_GRAY'
        VIEW_SIDE_BY_SIDE_UNRECTIFIED_BGRA 'sl::VIEW::SIDE_BY_SIDE_UNRECTIFIED_BGRA'
        VIEW_SIDE_BY_SIDE_UNRECTIFIED_BGR 'sl::VIEW::SIDE_BY_SIDE_UNRECTIFIED_BGR'
        VIEW_SIDE_BY_SIDE_UNRECTIFIED_GRAY 'sl::VIEW::SIDE_BY_SIDE_UNRECTIFIED_GRAY'
        VIEW_DEPTH_BGRA 'sl::VIEW::DEPTH_BGRA'
        VIEW_DEPTH_BGR 'sl::VIEW::DEPTH_BGR'
        VIEW_DEPTH_GRAY 'sl::VIEW::DEPTH_GRAY'
        VIEW_CONFIDENCE_BGRA 'sl::VIEW::CONFIDENCE_BGRA'
        VIEW_CONFIDENCE_BGR 'sl::VIEW::CONFIDENCE_BGR'
        VIEW_CONFIDENCE_GRAY 'sl::VIEW::CONFIDENCE_GRAY'
        VIEW_NORMALS_BGRA 'sl::VIEW::NORMALS_BGRA'
        VIEW_NORMALS_BGR 'sl::VIEW::NORMALS_BGR'
        VIEW_NORMALS_GRAY 'sl::VIEW::NORMALS_GRAY'
        VIEW_DEPTH_RIGHT_BGRA 'sl::VIEW::DEPTH_RIGHT_BGRA'
        VIEW_DEPTH_RIGHT_BGR 'sl::VIEW::DEPTH_RIGHT_BGR'
        VIEW_DEPTH_RIGHT_GRAY 'sl::VIEW::DEPTH_RIGHT_GRAY'
        VIEW_NORMALS_RIGHT_BGRA 'sl::VIEW::NORMALS_RIGHT_BGRA'
        VIEW_NORMALS_RIGHT_BGR 'sl::VIEW::NORMALS_RIGHT_BGR'
        VIEW_NORMALS_RIGHT_GRAY 'sl::VIEW::NORMALS_RIGHT_GRAY'
        VIEW_LAST 'sl::VIEW::LAST'

    cdef String toString(const VIEW &o)

    ctypedef enum TIME_REFERENCE 'sl::TIME_REFERENCE':
        TIME_REFERENCE_IMAGE 'sl::TIME_REFERENCE::IMAGE'
        TIME_REFERENCE_CURRENT 'sl::TIME_REFERENCE::CURRENT'
        TIME_REFERENCE_LAST 'sl::TIME_REFERENCE::LAST'

    cdef String toString(const TIME_REFERENCE &o)

    ctypedef enum POSITIONAL_TRACKING_STATE 'sl::POSITIONAL_TRACKING_STATE':
        POSITIONAL_TRACKING_STATE_SEARCHING 'sl::POSITIONAL_TRACKING_STATE::SEARCHING'
        POSITIONAL_TRACKING_STATE_OK 'sl::POSITIONAL_TRACKING_STATE::OK'
        POSITIONAL_TRACKING_STATE_OFF 'sl::POSITIONAL_TRACKING_STATE::OFF'
        POSITIONAL_TRACKING_STATE_FPS_TOO_LOW 'sl::POSITIONAL_TRACKING_STATE::FPS_TOO_LOW'
        POSITIONAL_TRACKING_STATE_SEARCHING_FLOOR_PLANE 'sl::POSITIONAL_TRACKING_STATE::SEARCHING_FLOOR_PLANE'
        POSITIONAL_TRACKING_STATE_UNAVAILABLE 'sl::POSITIONAL_TRACKING_STATE::UNAVAILABLE'
        POSITIONAL_TRACKING_STATE_LAST 'sl::POSITIONAL_TRACKING_STATE::LAST'

    cdef String toString(const POSITIONAL_TRACKING_STATE &o)

    ctypedef enum GNSS_STATUS 'sl::GNSS_STATUS':
        GNSS_STATUS_UNKNOWN 'sl::GNSS_STATUS::UNKNOWN'
        GNSS_STATUS_SINGLE 'sl::GNSS_STATUS::SINGLE'
        GNSS_STATUS_DGNSS 'sl::GNSS_STATUS::DGNSS'
        GNSS_STATUS_PPS 'sl::GNSS_STATUS::PPS'
        GNSS_STATUS_RTK_FLOAT 'sl::GNSS_STATUS::RTK_FLOAT'
        GNSS_STATUS_RTK_FIX 'sl::GNSS_STATUS::RTK_FIX'
        GNSS_STATUS_LAST 'sl::GNSS_STATUS::LAST'

    cdef String toString(const GNSS_STATUS &o)

    ctypedef enum GNSS_MODE 'sl::GNSS_MODE':
        GNSS_MODE_UNKNOWN 'sl::GNSS_MODE::UNKNOWN'
        GNSS_MODE_NO_FIX 'sl::GNSS_MODE::NO_FIX'
        GNSS_MODE_FIX_2D 'sl::GNSS_MODE::FIX_2D'
        GNSS_MODE_FIX_3D 'sl::GNSS_MODE::FIX_3D'
        GNSS_MODE_LAST 'sl::GNSS_MODE::LAST'

    cdef String toString(const GNSS_MODE &o)

    ctypedef enum GNSS_FUSION_STATUS 'sl::GNSS_FUSION_STATUS':
        GNSS_FUSION_STATUS_OK 'sl::GNSS_FUSION_STATUS::OK'
        GNSS_FUSION_STATUS_OFF 'sl::GNSS_FUSION_STATUS::OFF'
        GNSS_FUSION_STATUS_CALIBRATION_IN_PROGRESS 'sl::GNSS_FUSION_STATUS::CALIBRATION_IN_PROGRESS'
        GNSS_FUSION_STATUS_RECALIBRATION_IN_PROGRESS 'sl::GNSS_FUSION_STATUS::RECALIBRATION_IN_PROGRESS'
        GNSS_FUSION_STATUS_LAST 'sl::GNSS_FUSION_STATUS::LAST'

    cdef String toString(const GNSS_FUSION_STATUS &o)

    ctypedef enum ODOMETRY_STATUS 'sl::ODOMETRY_STATUS':
        ODOMETRY_STATUS_OK 'sl::ODOMETRY_STATUS::OK'
        ODOMETRY_STATUS_UNAVAILABLE 'sl::ODOMETRY_STATUS::UNAVAILABLE'
        ODOMETRY_STATUS_INSUFFICIENT_FEATURES 'sl::ODOMETRY_STATUS::INSUFFICIENT_FEATURES'
        ODOMETRY_STATUS_LAST 'sl::ODOMETRY_STATUS::LAST'

    cdef String toString(const ODOMETRY_STATUS &o)

    ctypedef enum SPATIAL_MEMORY_STATUS 'sl::SPATIAL_MEMORY_STATUS':
        SPATIAL_MEMORY_STATUS_OK 'sl::SPATIAL_MEMORY_STATUS::OK'
        SPATIAL_MEMORY_STATUS_LOOP_CLOSED 'sl::SPATIAL_MEMORY_STATUS::LOOP_CLOSED'
        SPATIAL_MEMORY_STATUS_SEARCHING 'sl::SPATIAL_MEMORY_STATUS::SEARCHING'
        SPATIAL_MEMORY_STATUS_INITIALIZING 'sl::SPATIAL_MEMORY_STATUS::INITIALIZING'
        SPATIAL_MEMORY_STATUS_MAP_UPDATE 'sl::SPATIAL_MEMORY_STATUS::MAP_UPDATE'
        SPATIAL_MEMORY_STATUS_KNOWN_MAP 'sl::SPATIAL_MEMORY_STATUS::KNOWN_MAP'
        SPATIAL_MEMORY_STATUS_LOST 'sl::SPATIAL_MEMORY_STATUS::LOST'
        SPATIAL_MEMORY_STATUS_OFF 'sl::SPATIAL_MEMORY_STATUS::OFF'
        SPATIAL_MEMORY_STATUS_LAST 'sl::SPATIAL_MEMORY_STATUS::LAST'

    cdef String toString(const SPATIAL_MEMORY_STATUS &o)

    ctypedef enum POSITIONAL_TRACKING_FUSION_STATUS 'sl::POSITIONAL_TRACKING_FUSION_STATUS':
        POSITIONAL_TRACKING_FUSION_STATUS_VISUAL_INERTIAL 'sl::POSITIONAL_TRACKING_FUSION_STATUS::VISUAL_INERTIAL'
        POSITIONAL_TRACKING_FUSION_STATUS_VISUAL 'sl::POSITIONAL_TRACKING_FUSION_STATUS::VISUAL'
        POSITIONAL_TRACKING_FUSION_STATUS_INERTIAL 'sl::POSITIONAL_TRACKING_FUSION_STATUS::INERTIAL'
        POSITIONAL_TRACKING_FUSION_STATUS_GNSS 'sl::POSITIONAL_TRACKING_FUSION_STATUS::GNSS'
        POSITIONAL_TRACKING_FUSION_STATUS_VISUAL_INERTIAL_GNSS 'sl::POSITIONAL_TRACKING_FUSION_STATUS::VISUAL_INERTIAL_GNSS'
        POSITIONAL_TRACKING_FUSION_STATUS_VISUAL_GNSS 'sl::POSITIONAL_TRACKING_FUSION_STATUS::VISUAL_GNSS'
        POSITIONAL_TRACKING_FUSION_STATUS_INERTIAL_GNSS 'sl::POSITIONAL_TRACKING_FUSION_STATUS::INERTIAL_GNSS'
        POSITIONAL_TRACKING_FUSION_STATUS_UNAVAILABLE 'sl::POSITIONAL_TRACKING_FUSION_STATUS::UNAVAILABLE'
        POSITIONAL_TRACKING_FUSION_STATUS_LAST 'sl::POSITIONAL_TRACKING_FUSION_STATUS::LAST'

    cdef String toString(const POSITIONAL_TRACKING_FUSION_STATUS &o)

    cdef cppclass PositionalTrackingStatus 'sl::PositionalTrackingStatus':
        ODOMETRY_STATUS odometry_status
        SPATIAL_MEMORY_STATUS spatial_memory_status
        POSITIONAL_TRACKING_FUSION_STATUS tracking_fusion_status

    cdef String toString(const PositionalTrackingStatus &o)

    cdef cppclass FusedPositionalTrackingStatus 'sl::FusedPositionalTrackingStatus':
        ODOMETRY_STATUS odometry_status
        SPATIAL_MEMORY_STATUS spatial_memory_status
        GNSS_STATUS gnss_status
        GNSS_MODE gnss_mode
        GNSS_FUSION_STATUS gnss_fusion_status
        POSITIONAL_TRACKING_FUSION_STATUS tracking_fusion_status


    ctypedef enum POSITIONAL_TRACKING_MODE 'sl::POSITIONAL_TRACKING_MODE':
        POSITIONAL_TRACKING_MODE_GEN_1 'sl::POSITIONAL_TRACKING_MODE::GEN_1'
        POSITIONAL_TRACKING_MODE_GEN_2 'sl::POSITIONAL_TRACKING_MODE::GEN_2'
        POSITIONAL_TRACKING_MODE_GEN_3 'sl::POSITIONAL_TRACKING_MODE::GEN_3'
        POSITIONAL_TRACKING_MODE_LAST 'sl::POSITIONAL_TRACKING_MODE::LAST'

    cdef String toString(const POSITIONAL_TRACKING_MODE &o)
    cdef bool fromString(const String &s, POSITIONAL_TRACKING_MODE &o)

    ctypedef enum AREA_EXPORTING_STATE 'sl::AREA_EXPORTING_STATE':
        AREA_EXPORTING_STATE_SUCCESS 'sl::AREA_EXPORTING_STATE::SUCCESS'
        AREA_EXPORTING_STATE_RUNNING 'sl::AREA_EXPORTING_STATE::RUNNING'
        AREA_EXPORTING_STATE_NOT_STARTED 'sl::AREA_EXPORTING_STATE::NOT_STARTED'
        AREA_EXPORTING_STATE_FILE_EMPTY 'sl::AREA_EXPORTING_STATE::FILE_EMPTY'
        AREA_EXPORTING_STATE_FILE_ERROR 'sl::AREA_EXPORTING_STATE::FILE_ERROR'
        AREA_EXPORTING_STATE_SPATIAL_MEMORY_DISABLED 'sl::AREA_EXPORTING_STATE::SPATIAL_MEMORY_DISABLED'
        AREA_EXPORTING_STATE_LAST 'sl::AREA_EXPORTING_STATE::LAST'

    cdef String toString(const AREA_EXPORTING_STATE &o)

    ctypedef enum REFERENCE_FRAME 'sl::REFERENCE_FRAME':
        REFERENCE_FRAME_WORLD 'sl::REFERENCE_FRAME::WORLD'
        REFERENCE_FRAME_CAMERA 'sl::REFERENCE_FRAME::CAMERA'
        REFERENCE_FRAME_LAST 'sl::REFERENCE_FRAME::LAST'

    cdef String toString(const REFERENCE_FRAME &o)
    cdef bool fromString(const String &s, REFERENCE_FRAME &o)

    ctypedef enum SPATIAL_MAPPING_STATE 'sl::SPATIAL_MAPPING_STATE':
        SPATIAL_MAPPING_STATE_INITIALIZING 'sl::SPATIAL_MAPPING_STATE::INITIALIZING'
        SPATIAL_MAPPING_STATE_OK 'sl::SPATIAL_MAPPING_STATE::OK'
        SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY 'sl::SPATIAL_MAPPING_STATE::NOT_ENOUGH_MEMORY'
        SPATIAL_MAPPING_STATE_NOT_ENABLED 'sl::SPATIAL_MAPPING_STATE::NOT_ENABLED'
        SPATIAL_MAPPING_STATE_FPS_TOO_LOW 'sl::SPATIAL_MAPPING_STATE::FPS_TOO_LOW'
        SPATIAL_MAPPING_STATE_LAST 'sl::SPATIAL_MAPPING_STATE::LAST'

    cdef String toString(const SPATIAL_MAPPING_STATE &o)

    ctypedef enum REGION_OF_INTEREST_AUTO_DETECTION_STATE 'sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE':
        REGION_OF_INTEREST_AUTO_DETECTION_STATE_RUNNING 'sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::RUNNING'
        REGION_OF_INTEREST_AUTO_DETECTION_STATE_READY 'sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::READY'
        REGION_OF_INTEREST_AUTO_DETECTION_STATE_NOT_ENABLED 'sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::NOT_ENABLED'
        REGION_OF_INTEREST_AUTO_DETECTION_STATE_LAST 'sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::LAST'

    cdef String toString(const REGION_OF_INTEREST_AUTO_DETECTION_STATE &o)

    ctypedef enum SVO_COMPRESSION_MODE 'sl::SVO_COMPRESSION_MODE':
        SVO_COMPRESSION_MODE_LOSSLESS 'sl::SVO_COMPRESSION_MODE::LOSSLESS'
        SVO_COMPRESSION_MODE_H264 'sl::SVO_COMPRESSION_MODE::H264'
        SVO_COMPRESSION_MODE_H265 'sl::SVO_COMPRESSION_MODE::H265'
        SVO_COMPRESSION_MODE_H264_LOSSLESS 'sl::SVO_COMPRESSION_MODE::H264_LOSSLESS'
        SVO_COMPRESSION_MODE_H265_LOSSLESS 'sl::SVO_COMPRESSION_MODE::H265_LOSSLESS'
        SVO_COMPRESSION_MODE_LAST 'sl::SVO_COMPRESSION_MODE::LAST'

    cdef String toString(const SVO_COMPRESSION_MODE &o)
    cdef bool fromString(const String &s, SVO_COMPRESSION_MODE &o)

    ctypedef enum SENSOR_TYPE 'sl::SENSOR_TYPE':
        SENSOR_TYPE_ACCELEROMETER 'sl::SENSOR_TYPE::ACCELEROMETER'
        SENSOR_TYPE_GYROSCOPE 'sl::SENSOR_TYPE::GYROSCOPE'
        SENSOR_TYPE_MAGNETOMETER 'sl::SENSOR_TYPE::MAGNETOMETER'
        SENSOR_TYPE_BAROMETER 'sl::SENSOR_TYPE::BAROMETER'
        SENSOR_TYPE_LAST 'sl::SENSOR_TYPE::LAST'

    cdef String toString(const SENSOR_TYPE &o)

    ctypedef enum SENSORS_UNIT 'sl::SENSORS_UNIT':
        SENSORS_UNIT_M_SEC_2 'sl::SENSORS_UNIT::M_SEC_2'
        SENSORS_UNIT_DEG_SEC 'sl::SENSORS_UNIT::DEG_SEC'
        SENSORS_UNIT_U_T 'sl::SENSORS_UNIT::U_T'
        SENSORS_UNIT_HPA 'sl::SENSORS_UNIT::HPA'
        SENSORS_UNIT_CELSIUS 'sl::SENSORS_UNIT::CELSIUS'
        SENSORS_UNIT_HERTZ 'sl::SENSORS_UNIT::HERTZ'
        SENSORS_UNIT_LAST 'sl::SENSORS_UNIT::LAST'

    cdef String toString(const SENSORS_UNIT &o)

    ctypedef enum INPUT_TYPE 'sl::INPUT_TYPE':
        INPUT_TYPE_USB 'sl::INPUT_TYPE::USB'
        INPUT_TYPE_SVO 'sl::INPUT_TYPE::SVO'
        INPUT_TYPE_STREAM 'sl::INPUT_TYPE::STREAM'
        INPUT_TYPE_GMSL 'sl::INPUT_TYPE::GMSL'
        INPUT_TYPE_LAST 'sl::INPUT_TYPE::LAST'

    cdef String toString(const INPUT_TYPE &o)
    cdef bool fromString(const String &s, INPUT_TYPE &o)

    ctypedef enum AI_MODELS 'sl::AI_MODELS':
        AI_MODELS_MULTI_CLASS_DETECTION 'sl::AI_MODELS::MULTI_CLASS_DETECTION'
        AI_MODELS_MULTI_CLASS_MEDIUM_DETECTION 'sl::AI_MODELS::MULTI_CLASS_MEDIUM_DETECTION'
        AI_MODELS_MULTI_CLASS_ACCURATE_DETECTION 'sl::AI_MODELS::MULTI_CLASS_ACCURATE_DETECTION'
        AI_MODELS_HUMAN_BODY_FAST_DETECTION 'sl::AI_MODELS::HUMAN_BODY_FAST_DETECTION'
        AI_MODELS_HUMAN_BODY_MEDIUM_DETECTION 'sl::AI_MODELS::HUMAN_BODY_MEDIUM_DETECTION'
        AI_MODELS_HUMAN_BODY_ACCURATE_DETECTION 'sl::AI_MODELS::HUMAN_BODY_ACCURATE_DETECTION'
        AI_MODELS_HUMAN_BODY_38_FAST_DETECTION 'sl::AI_MODELS::HUMAN_BODY_38_FAST_DETECTION'
        AI_MODELS_HUMAN_BODY_38_MEDIUM_DETECTION 'sl::AI_MODELS::HUMAN_BODY_38_MEDIUM_DETECTION'
        AI_MODELS_HUMAN_BODY_38_ACCURATE_DETECTION 'sl::AI_MODELS::HUMAN_BODY_38_ACCURATE_DETECTION'
        AI_MODELS_PERSON_HEAD_DETECTION 'sl::AI_MODELS::PERSON_HEAD_DETECTION'
        AI_MODELS_PERSON_HEAD_ACCURATE_DETECTION 'sl::AI_MODELS::PERSON_HEAD_ACCURATE_DETECTION'
        AI_MODELS_REID_ASSOCIATION 'sl::AI_MODELS::REID_ASSOCIATION'
        AI_MODELS_NEURAL_LIGHT_DEPTH 'sl::AI_MODELS::NEURAL_LIGHT_DEPTH'
        AI_MODELS_NEURAL_DEPTH 'sl::AI_MODELS::NEURAL_DEPTH'
        AI_MODELS_NEURAL_PLUS_DEPTH 'sl::AI_MODELS::NEURAL_PLUS_DEPTH'
        AI_MODELS_LAST 'sl::AI_MODELS::LAST'

    cdef String toString(const AI_MODELS &o)

    ctypedef enum OBJECT_DETECTION_MODEL 'sl::OBJECT_DETECTION_MODEL':
        OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_FAST 'sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_FAST'
        OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_ACCURATE 'sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE'
        OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_MEDIUM 'sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_MEDIUM'
        OBJECT_DETECTION_MODEL_PERSON_HEAD_BOX_FAST 'sl::OBJECT_DETECTION_MODEL::PERSON_HEAD_BOX_FAST'
        OBJECT_DETECTION_MODEL_PERSON_HEAD_BOX_ACCURATE 'sl::OBJECT_DETECTION_MODEL::PERSON_HEAD_BOX_ACCURATE'
        OBJECT_DETECTION_MODEL_CUSTOM_BOX_OBJECTS 'sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS'
        OBJECT_DETECTION_MODEL_CUSTOM_YOLOLIKE_BOX_OBJECTS 'sl::OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS'
        OBJECT_DETECTION_MODEL_LAST 'sl::OBJECT_DETECTION_MODEL::LAST'

    cdef String toString(const OBJECT_DETECTION_MODEL &o)
    cdef bool fromString(const String &s, OBJECT_DETECTION_MODEL &o)

    ctypedef enum BODY_TRACKING_MODEL 'sl::BODY_TRACKING_MODEL':
        BODY_TRACKING_MODEL_HUMAN_BODY_FAST 'sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST'
        BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE 'sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE'
        BODY_TRACKING_MODEL_HUMAN_BODY_MEDIUM 'sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM'
        BODY_TRACKING_MODEL_LAST 'sl::BODY_TRACKING_MODEL::LAST'

    cdef String toString(const BODY_TRACKING_MODEL &o)
    cdef bool fromString(const String &s, BODY_TRACKING_MODEL &o)

    ctypedef enum OBJECT_FILTERING_MODE 'sl::OBJECT_FILTERING_MODE':
        OBJECT_FILTERING_MODE_NONE 'sl::OBJECT_FILTERING_MODE::NONE'
        OBJECT_FILTERING_MODE_NMS3D 'sl::OBJECT_FILTERING_MODE::NMS3D'
        OBJECT_FILTERING_MODE_NMS3D_PER_CLASS 'sl::OBJECT_FILTERING_MODE::NMS3D_PER_CLASS'
        OBJECT_FILTERING_MODE_LAST 'sl::OBJECT_FILTERING_MODE::LAST'

    cdef String toString(const OBJECT_FILTERING_MODE &o)
    cdef bool fromString(const String &s, OBJECT_FILTERING_MODE &o)

    ctypedef enum OBJECT_ACCELERATION_PRESET 'sl::OBJECT_ACCELERATION_PRESET':
        OBJECT_ACCELERATION_PRESET_DEFAULT 'sl::OBJECT_ACCELERATION_PRESET::DEFAULT'
        OBJECT_ACCELERATION_PRESET_LOW 'sl::OBJECT_ACCELERATION_PRESET::LOW'
        OBJECT_ACCELERATION_PRESET_MEDIUM 'sl::OBJECT_ACCELERATION_PRESET::MEDIUM'
        OBJECT_ACCELERATION_PRESET_HIGH 'sl::OBJECT_ACCELERATION_PRESET::HIGH'
        OBJECT_ACCELERATION_PRESET_LAST 'sl::OBJECT_ACCELERATION_PRESET::LAST'

    cdef String toString(const OBJECT_ACCELERATION_PRESET &o)
    cdef bool fromString(const String &s, OBJECT_ACCELERATION_PRESET &o)

    cdef struct RecordingStatus 'sl::RecordingStatus':
        bool is_recording
        bool is_paused
        bool status
        double current_compression_time
        double current_compression_ratio
        double average_compression_time
        double average_compression_ratio
        int number_frames_ingested
        int number_frames_encoded

    Timestamp getCurrentTimeStamp()

    bool saveJSONFile(const String filepath, const String &serialized_content, bool merge)
    bool loadJSONFile(const String filepath, String &serialized_content)

    cdef struct Resolution 'sl::Resolution':
        int width
        int height

    cdef cppclass Rect 'sl::Rect':
        size_t x
        size_t y
        size_t width
        size_t height
        Rect(size_t x = 0, size_t y = 0, size_t width = 0, size_t height = 0)
        size_t area() const
        bool operator==(const Rect& r) const
        bool operator!=(const Rect& r) const
        bool isEmpty() const
        bool contains(const Rect& target, bool proper) const
        bool isContained_Rect "isContained"(const Rect& target, bool proper) const
        bool isContained_Resolution "isContained"(const Resolution& resolution, bool proper) const

    cdef struct CameraParameters 'sl::CameraParameters':
        float fx
        float fy
        float cx
        float cy
        double disto[12]
        float v_fov
        float h_fov
        float d_fov
        Resolution image_size
        float focal_length_metric
        CameraParameters scale(Resolution output_resolution) 
        void SetUp(float focal_x, float focal_y, float center_x, float center_y)


    cdef struct CalibrationParameters 'sl::CalibrationParameters':
        CameraParameters left_cam
        CameraParameters right_cam
        Transform stereo_transform

        float getCameraBaseline()

    cdef struct SensorParameters 'sl::SensorParameters':
        SENSOR_TYPE type
        float resolution
        float sampling_rate
        Vector2[float] range
        float noise_density
        float random_walk
        SENSORS_UNIT sensor_unit
        bool isAvailable


    cdef struct SensorsConfiguration 'sl::SensorsConfiguration':
        uint firmware_version
        Transform camera_imu_transform
        Transform imu_magnetometer_transform

        SensorParameters accelerometer_parameters
        SensorParameters gyroscope_parameters
        SensorParameters magnetometer_parameters
        SensorParameters barometer_parameters


    cdef struct CameraConfiguration 'sl::CameraConfiguration':
        CalibrationParameters calibration_parameters
        CalibrationParameters calibration_parameters_raw
        uint firmware_version
        float fps
        Resolution resolution


    cdef struct CameraInformation 'sl::CameraInformation':
        uint serial_number
        MODEL camera_model
        INPUT_TYPE input_type
        CameraConfiguration camera_configuration
        SensorsConfiguration sensors_configuration


    cdef struct CameraOneConfiguration 'sl::CameraOneConfiguration':
        CameraParameters calibration_parameters
        CameraParameters calibration_parameters_raw
        uint firmware_version
        float fps
        Resolution resolution


    cdef struct CameraOneInformation 'sl::CameraOneInformation':
        uint serial_number
        MODEL camera_model
        INPUT_TYPE input_type
        CameraOneConfiguration camera_configuration
        SensorsConfiguration sensors_configuration

    void resetBenchmarkTimer()

    ctypedef enum MEM 'sl::MEM':
        MEM_CPU 'sl::MEM::CPU'
        MEM_GPU 'sl::MEM::GPU'
        MEM_BOTH 'sl::MEM::BOTH'
        MEM_LAST 'sl::MEM::LAST'

    MEM operator|(MEM a, MEM b)
    cdef String toString(const MEM &o)

    ctypedef enum COPY_TYPE 'sl::COPY_TYPE':
        COPY_TYPE_CPU_CPU 'sl::COPY_TYPE::CPU_CPU'
        COPY_TYPE_GPU_CPU 'sl::COPY_TYPE::GPU_CPU'
        COPY_TYPE_CPU_GPU 'sl::COPY_TYPE::CPU_GPU'
        COPY_TYPE_GPU_GPU 'sl::COPY_TYPE::GPU_GPU'

    cdef String toString(const COPY_TYPE &o)

    ctypedef enum MAT_TYPE 'sl::MAT_TYPE':
        MAT_TYPE_F32_C1 'sl::MAT_TYPE::F32_C1'
        MAT_TYPE_F32_C2 'sl::MAT_TYPE::F32_C2'
        MAT_TYPE_F32_C3 'sl::MAT_TYPE::F32_C3'
        MAT_TYPE_F32_C4 'sl::MAT_TYPE::F32_C4'
        MAT_TYPE_U8_C1 'sl::MAT_TYPE::U8_C1'
        MAT_TYPE_U8_C2 'sl::MAT_TYPE::U8_C2'
        MAT_TYPE_U8_C3 'sl::MAT_TYPE::U8_C3'
        MAT_TYPE_U8_C4 'sl::MAT_TYPE::U8_C4'
        MAT_TYPE_U16_C1 'sl::MAT_TYPE::U16_C1'
        MAT_TYPE_S8_C4 'sl::MAT_TYPE::S8_C4'
        MAT_TYPE_NV12 'sl::MAT_TYPE::NV12'

    cdef String toString(const MAT_TYPE &o)

    ctypedef enum MODULE 'sl::MODULE':
        MODULE_ALL 'sl::MODULE::ALL' = 0
        MODULE_DEPTH 'sl::MODULE::DEPTH' = 1
        MODULE_POSITIONAL_TRACKING 'sl::MODULE::POSITIONAL_TRACKING' = 2
        MODULE_OBJECT_DETECTION 'sl::MODULE::OBJECT_DETECTION' = 3
        MODULE_BODY_TRACKING 'sl::MODULE::BODY_TRACKING' = 4
        MODULE_SPATIAL_MAPPING 'sl::MODULE::SPATIAL_MAPPING' = 5
        MODULE_LAST 'sl::MODULE::LAST' = 6

    cdef String toString(const MODULE &o)

    ctypedef enum OBJECT_CLASS 'sl::OBJECT_CLASS':
        OBJECT_CLASS_PERSON 'sl::OBJECT_CLASS::PERSON' = 0
        OBJECT_CLASS_VEHICLE 'sl::OBJECT_CLASS::VEHICLE' = 1
        OBJECT_CLASS_BAG 'sl::OBJECT_CLASS::BAG' = 2
        OBJECT_CLASS_ANIMAL 'sl::OBJECT_CLASS::ANIMAL' = 3
        OBJECT_CLASS_ELECTRONICS 'sl::OBJECT_CLASS::ELECTRONICS' = 4
        OBJECT_CLASS_FRUIT_VEGETABLE 'sl::OBJECT_CLASS::FRUIT_VEGETABLE' = 5
        OBJECT_CLASS_SPORT 'sl::OBJECT_CLASS::SPORT' = 6
        OBJECT_CLASS_LAST 'sl::OBJECT_CLASS::LAST' = 7

    cdef String toString(const OBJECT_CLASS &o)
    cdef bool fromString(const String &s, OBJECT_CLASS &o)

    ctypedef enum OBJECT_SUBCLASS 'sl::OBJECT_SUBCLASS':
        OBJECT_SUBCLASS_PERSON 'sl::OBJECT_SUBCLASS::PERSON' = 0
        OBJECT_SUBCLASS_BICYCLE 'sl::OBJECT_SUBCLASS::BICYCLE' = 1
        OBJECT_SUBCLASS_CAR 'sl::OBJECT_SUBCLASS::CAR' = 2
        OBJECT_SUBCLASS_MOTORBIKE 'sl::OBJECT_SUBCLASS::MOTORBIKE' = 3
        OBJECT_SUBCLASS_BUS 'sl::OBJECT_SUBCLASS::BUS' = 4
        OBJECT_SUBCLASS_TRUCK 'sl::OBJECT_SUBCLASS::TRUCK' = 5
        OBJECT_SUBCLASS_BOAT 'sl::OBJECT_SUBCLASS::BOAT' = 6
        OBJECT_SUBCLASS_BACKPACK 'sl::OBJECT_SUBCLASS::BACKPACK' = 7
        OBJECT_SUBCLASS_HANDBAG 'sl::OBJECT_SUBCLASS::HANDBAG' = 8
        OBJECT_SUBCLASS_SUITCASE 'sl::OBJECT_SUBCLASS::SUITCASE' = 9
        OBJECT_SUBCLASS_BIRD 'sl::OBJECT_SUBCLASS::BIRD' = 10
        OBJECT_SUBCLASS_CAT 'sl::OBJECT_SUBCLASS::CAT' = 11
        OBJECT_SUBCLASS_DOG 'sl::OBJECT_SUBCLASS::DOG' = 12
        OBJECT_SUBCLASS_HORSE 'sl::OBJECT_SUBCLASS::HORSE' = 13
        OBJECT_SUBCLASS_SHEEP 'sl::OBJECT_SUBCLASS::SHEEP' = 14
        OBJECT_SUBCLASS_COW 'sl::OBJECT_SUBCLASS::COW' = 15
        OBJECT_SUBCLASS_CELLPHONE 'sl::OBJECT_SUBCLASS::CELLPHONE' = 16
        OBJECT_SUBCLASS_LAPTOP 'sl::OBJECT_SUBCLASS::LAPTOP' = 17
        OBJECT_SUBCLASS_BANANA 'sl::OBJECT_SUBCLASS::BANANA' = 18
        OBJECT_SUBCLASS_APPLE 'sl::OBJECT_SUBCLASS::APPLE' = 19
        OBJECT_SUBCLASS_ORANGE 'sl::OBJECT_SUBCLASS::ORANGE' = 20
        OBJECT_SUBCLASS_CARROT 'sl::OBJECT_SUBCLASS::CARROT' = 21
        OBJECT_SUBCLASS_PERSON_HEAD 'sl::OBJECT_SUBCLASS::PERSON_HEAD' = 22
        OBJECT_SUBCLASS_SPORTSBALL 'sl::OBJECT_SUBCLASS::SPORTSBALL' = 23
        OBJECT_SUBCLASS_MACHINERY 'sl::OBJECT_SUBCLASS::MACHINERY' = 24
        OBJECT_SUBCLASS_LAST 'sl::OBJECT_SUBCLASS::LAST' = 25

    cdef String toString(const OBJECT_SUBCLASS &o)
    cdef OBJECT_CLASS getObjectClass(OBJECT_SUBCLASS object_subclass)
    cdef vector[OBJECT_SUBCLASS] getObjectSubClasses(OBJECT_CLASS object_class)
    cdef int getObjectSubClassAsCOCO(const OBJECT_SUBCLASS object_subclass)

    ctypedef enum OBJECT_TRACKING_STATE 'sl::OBJECT_TRACKING_STATE':
        OBJECT_TRACKING_STATE_OFF 'sl::OBJECT_TRACKING_STATE::OFF'
        OBJECT_TRACKING_STATE_OK 'sl::OBJECT_TRACKING_STATE::OK'
        OBJECT_TRACKING_STATE_SEARCHING 'sl::OBJECT_TRACKING_STATE::SEARCHING'
        OBJECT_TRACKING_STATE_TERMINATE 'sl::OBJECT_TRACKING_STATE::TERMINATE'
        OBJECT_TRACKING_STATE_LAST 'sl::OBJECT_TRACKING_STATE::LAST'

    cdef String toString(const OBJECT_TRACKING_STATE &o)

    ctypedef enum OBJECT_ACTION_STATE 'sl::OBJECT_ACTION_STATE':
        OBJECT_ACTION_STATE_IDLE 'sl::OBJECT_ACTION_STATE::IDLE'
        OBJECT_ACTION_STATE_MOVING 'sl::OBJECT_ACTION_STATE::MOVING'
        OBJECT_ACTION_STATE_LAST 'sl::OBJECT_ACTION_STATE::LAST'

    cdef String toString(const OBJECT_ACTION_STATE &o)

    cdef cppclass ObjectData 'sl::ObjectData':
        int id
        String unique_object_id
        int raw_label
        OBJECT_CLASS label
        OBJECT_SUBCLASS sublabel
        OBJECT_TRACKING_STATE tracking_state
        OBJECT_ACTION_STATE action_state
        Vector3[float] position
        Vector3[float] velocity
        float position_covariance[6]
        vector[Vector2[uint]] bounding_box_2d
        Mat mask
        float confidence
        vector[Vector3[float]] bounding_box
        Vector3[float] dimensions
        vector[Vector2[uint]] head_bounding_box_2d
        vector[Vector3[float]] head_bounding_box
        Vector3[float] head_position

        ObjectData()

    cdef cppclass BodyData 'sl::BodyData':
        int id
        String unique_object_id
        OBJECT_TRACKING_STATE tracking_state
        OBJECT_ACTION_STATE action_state
        Vector3[float] position
        Vector3[float] velocity
        float position_covariance[6]
        vector[Vector2[uint]] bounding_box_2d
        Mat mask
        float confidence
        vector[Vector3[float]] bounding_box
        Vector3[float] dimensions
        vector[Vector2[float]] keypoint_2d
        vector[Vector3[float]] keypoint
        vector[Vector2[uint]] head_bounding_box_2d
        vector[Vector3[float]] head_bounding_box
        Vector3[float] head_position
        vector[float] keypoint_confidence
        vector[array6] keypoint_covariances
        vector[Vector3[float]] local_position_per_joint
        vector[Vector4[float]] local_orientation_per_joint
        Vector4[float] global_root_orientation

        BodyData()

    String generate_unique_id()

    cdef cppclass ObjectTrackingParameters 'sl::ObjectTrackingParameters':
        OBJECT_ACCELERATION_PRESET object_acceleration_preset
        float velocity_smoothing_factor
        float min_velocity_threshold
        float prediction_timeout_s
        float min_confirmation_time_s

    cdef cppclass CustomBoxObjectData 'sl::CustomBoxObjectData':
        String unique_object_id
        vector[Vector2[uint]] bounding_box_2d
        int label
        float probability
        bool is_grounded
        bool is_static
        float tracking_timeout
        float tracking_max_dist
        float max_box_width_meters
        float min_box_width_meters
        float max_box_height_meters
        float min_box_height_meters
        float max_allowed_acceleration
        float velocity_smoothing_factor
        float min_velocity_threshold
        float prediction_timeout_s
        float min_confirmation_time_s

        CustomBoxObjectData()

    cdef cppclass CustomMaskObjectData 'sl::CustomMaskObjectData':
        String unique_object_id
        vector[Vector2[uint]] bounding_box_2d
        int label
        float probability
        bool is_grounded
        bool is_static
        float tracking_timeout
        float tracking_max_dist
        float max_box_width_meters
        float min_box_width_meters
        float max_box_height_meters
        float min_box_height_meters
        float max_allowed_acceleration
        float velocity_smoothing_factor
        float min_velocity_threshold
        float prediction_timeout_s
        float min_confirmation_time_s
        Mat box_mask

        CustomMaskObjectData()

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
        vector[vector[Vector2[uint]]] head_bounding_boxes_2d
        vector[vector[Vector3[float]]] head_bounding_boxes
        vector[Vector3[float]] head_positions
        vector[vector[float]] keypoint_confidences

        ObjectsBatch()

    cdef cppclass BodiesBatch 'sl::BodiesBatch':
        int id
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

        BodiesBatch()

    cdef cppclass Objects 'sl::Objects':
        Timestamp timestamp
        vector[ObjectData] object_list
        bool is_new
        bool is_tracked
        bool getObjectDataFromId(ObjectData &objectData, int objectDataId)

        Objects()

    cdef cppclass Bodies 'sl::Bodies':
        Timestamp timestamp
        vector[BodyData] body_list
        bool is_new
        bool is_tracked
        BODY_FORMAT body_format
        INFERENCE_PRECISION inference_precision_mode
        bool getBodyDataFromId(BodyData &bodyData, int bodyDataId)

    ctypedef enum BODY_18_PARTS 'sl::BODY_18_PARTS':
        BODY_18_PARTS_NOSE 'sl::BODY_18_PARTS::NOSE'
        BODY_18_PARTS_NECK 'sl::BODY_18_PARTS::NECK'
        BODY_18_PARTS_RIGHT_SHOULDER 'sl::BODY_18_PARTS::RIGHT_SHOULDER'
        BODY_18_PARTS_RIGHT_ELBOW 'sl::BODY_18_PARTS::RIGHT_ELBOW'
        BODY_18_PARTS_RIGHT_WRIST 'sl::BODY_18_PARTS::RIGHT_WRIST'
        BODY_18_PARTS_LEFT_SHOULDER 'sl::BODY_18_PARTS::LEFT_SHOULDER'
        BODY_18_PARTS_LEFT_ELBOW 'sl::BODY_18_PARTS::LEFT_ELBOW'
        BODY_18_PARTS_LEFT_WRIST 'sl::BODY_18_PARTS::LEFT_WRIST'
        BODY_18_PARTS_RIGHT_HIP 'sl::BODY_18_PARTS::RIGHT_HIP'
        BODY_18_PARTS_RIGHT_KNEE 'sl::BODY_18_PARTS::RIGHT_KNEE'
        BODY_18_PARTS_RIGHT_ANKLE 'sl::BODY_18_PARTS::RIGHT_ANKLE'
        BODY_18_PARTS_LEFT_HIP 'sl::BODY_18_PARTS::LEFT_HIP'
        BODY_18_PARTS_LEFT_KNEE 'sl::BODY_18_PARTS::LEFT_KNEE'
        BODY_18_PARTS_LEFT_ANKLE 'sl::BODY_18_PARTS::LEFT_ANKLE'
        BODY_18_PARTS_RIGHT_EYE 'sl::BODY_18_PARTS::RIGHT_EYE'
        BODY_18_PARTS_LEFT_EYE 'sl::BODY_18_PARTS::LEFT_EYE'
        BODY_18_PARTS_RIGHT_EAR 'sl::BODY_18_PARTS::RIGHT_EAR'
        BODY_18_PARTS_LEFT_EAR 'sl::BODY_18_PARTS::LEFT_EAR'
        BODY_18_PARTS_LAST 'sl::BODY_18_PARTS::LAST'

    cdef String toString(const BODY_18_PARTS &o)

    ctypedef enum BODY_34_PARTS 'sl::BODY_34_PARTS':
        BODY_34_PARTS_PELVIS 'sl::BODY_34_PARTS::PELVIS' 
        BODY_34_PARTS_NAVAL_SPINE 'sl::BODY_34_PARTS::NAVAL_SPINE' 
        BODY_34_PARTS_CHEST_SPINE 'sl::BODY_34_PARTS::CHEST_SPINE' 
        BODY_34_PARTS_NECK 'sl::BODY_34_PARTS::NECK' 
        BODY_34_PARTS_LEFT_CLAVICLE 'sl::BODY_34_PARTS::LEFT_CLAVICLE' 
        BODY_34_PARTS_LEFT_SHOULDER 'sl::BODY_34_PARTS::LEFT_SHOULDER' 
        BODY_34_PARTS_LEFT_ELBOW 'sl::BODY_34_PARTS::LEFT_ELBOW' 
        BODY_34_PARTS_LEFT_WRIST 'sl::BODY_34_PARTS::LEFT_WRIST' 
        BODY_34_PARTS_LEFT_HAND 'sl::BODY_34_PARTS::LEFT_HAND' 
        BODY_34_PARTS_LEFT_HANDTIP 'sl::BODY_34_PARTS::LEFT_HANDTIP' 
        BODY_34_PARTS_LEFT_THUMB 'sl::BODY_34_PARTS::LEFT_THUMB' 
        BODY_34_PARTS_RIGHT_CLAVICLE 'sl::BODY_34_PARTS::RIGHT_CLAVICLE'  
        BODY_34_PARTS_RIGHT_SHOULDER 'sl::BODY_34_PARTS::RIGHT_SHOULDER' 
        BODY_34_PARTS_RIGHT_ELBOW 'sl::BODY_34_PARTS::RIGHT_ELBOW' 
        BODY_34_PARTS_RIGHT_WRIST 'sl::BODY_34_PARTS::RIGHT_WRIST' 
        BODY_34_PARTS_RIGHT_HAND 'sl::BODY_34_PARTS::RIGHT_HAND' 
        BODY_34_PARTS_RIGHT_HANDTIP 'sl::BODY_34_PARTS::RIGHT_HANDTIP' 
        BODY_34_PARTS_RIGHT_THUMB 'sl::BODY_34_PARTS::RIGHT_THUMB' 
        BODY_34_PARTS_LEFT_HIP 'sl::BODY_34_PARTS::LEFT_HIP' 
        BODY_34_PARTS_LEFT_KNEE 'sl::BODY_34_PARTS::LEFT_KNEE' 
        BODY_34_PARTS_LEFT_ANKLE 'sl::BODY_34_PARTS::LEFT_ANKLE' 
        BODY_34_PARTS_LEFT_FOOT 'sl::BODY_34_PARTS::LEFT_FOOT' 
        BODY_34_PARTS_RIGHT_HIP 'sl::BODY_34_PARTS::RIGHT_HIP' 
        BODY_34_PARTS_RIGHT_KNEE 'sl::BODY_34_PARTS::RIGHT_KNEE' 
        BODY_34_PARTS_RIGHT_ANKLE 'sl::BODY_34_PARTS::RIGHT_ANKLE' 
        BODY_34_PARTS_RIGHT_FOOT 'sl::BODY_34_PARTS::RIGHT_FOOT' 
        BODY_34_PARTS_HEAD 'sl::BODY_34_PARTS::HEAD' 
        BODY_34_PARTS_NOSE 'sl::BODY_34_PARTS::NOSE' 
        BODY_34_PARTS_LEFT_EYE 'sl::BODY_34_PARTS::LEFT_EYE' 
        BODY_34_PARTS_LEFT_EAR 'sl::BODY_34_PARTS::LEFT_EAR' 
        BODY_34_PARTS_RIGHT_EYE 'sl::BODY_34_PARTS::RIGHT_EYE' 
        BODY_34_PARTS_RIGHT_EAR 'sl::BODY_34_PARTS::RIGHT_EAR' 
        BODY_34_PARTS_LEFT_HEEL 'sl::BODY_34_PARTS::LEFT_HEEL' 
        BODY_34_PARTS_RIGHT_HEEL 'sl::BODY_34_PARTS::RIGHT_HEEL' 
        BODY_34_PARTS_LAST 'sl::BODY_34_PARTS::LAST'

    cdef String toString(const BODY_34_PARTS &o)

    ctypedef enum BODY_38_PARTS 'sl::BODY_38_PARTS':
        BODY_38_PARTS_PELVIS 'sl::BODY_38_PARTS::PELVIS' 
        BODY_38_PARTS_SPINE_1 'sl::BODY_38_PARTS::SPINE_1' 
        BODY_38_PARTS_SPINE_2 'sl::BODY_38_PARTS::SPINE_2' 
        BODY_38_PARTS_SPINE_3 'sl::BODY_38_PARTS::SPINE_3' 
        BODY_38_PARTS_NECK 'sl::BODY_38_PARTS::NECK' 
        BODY_38_PARTS_NOSE 'sl::BODY_38_PARTS::NOSE' 
        BODY_38_PARTS_LEFT_EYE 'sl::BODY_38_PARTS::LEFT_EYE' 
        BODY_38_PARTS_RIGHT_EYE 'sl::BODY_38_PARTS::RIGHT_EYE' 
        BODY_38_PARTS_LEFT_EAR 'sl::BODY_38_PARTS::LEFT_EAR'         
        BODY_38_PARTS_RIGHT_EAR 'sl::BODY_38_PARTS::RIGHT_EAR'         
        BODY_38_PARTS_LEFT_CLAVICLE 'sl::BODY_38_PARTS::LEFT_CLAVICLE' 
        BODY_38_PARTS_RIGHT_CLAVICLE 'sl::BODY_38_PARTS::RIGHT_CLAVICLE'  
        BODY_38_PARTS_LEFT_SHOULDER 'sl::BODY_38_PARTS::LEFT_SHOULDER' 
        BODY_38_PARTS_RIGHT_SHOULDER 'sl::BODY_38_PARTS::RIGHT_SHOULDER' 
        BODY_38_PARTS_LEFT_ELBOW 'sl::BODY_38_PARTS::LEFT_ELBOW' 
        BODY_38_PARTS_RIGHT_ELBOW 'sl::BODY_38_PARTS::RIGHT_ELBOW' 
        BODY_38_PARTS_LEFT_WRIST 'sl::BODY_38_PARTS::LEFT_WRIST' 
        BODY_38_PARTS_RIGHT_WRIST 'sl::BODY_38_PARTS::RIGHT_WRIST'
        BODY_38_PARTS_LEFT_HIP 'sl::BODY_38_PARTS::LEFT_HIP' 
        BODY_38_PARTS_RIGHT_HIP 'sl::BODY_38_PARTS::RIGHT_HIP' 
        BODY_38_PARTS_LEFT_KNEE 'sl::BODY_38_PARTS::LEFT_KNEE' 
        BODY_38_PARTS_RIGHT_KNEE 'sl::BODY_38_PARTS::RIGHT_KNEE' 
        BODY_38_PARTS_LEFT_ANKLE 'sl::BODY_38_PARTS::LEFT_ANKLE' 
        BODY_38_PARTS_RIGHT_ANKLE 'sl::BODY_38_PARTS::RIGHT_ANKLE' 
        BODY_38_PARTS_LEFT_BIG_TOE 'sl::BODY_38_PARTS::LEFT_BIG_TOE' 
        BODY_38_PARTS_RIGHT_BIG_TOE 'sl::BODY_38_PARTS::RIGHT_BIG_TOE' 
        BODY_38_PARTS_LEFT_SMALL_TOE 'sl::BODY_38_PARTS::LEFT_SMALL_TOE' 
        BODY_38_PARTS_RIGHT_SMALL_TOE 'sl::BODY_38_PARTS::RIGHT_SMALL_TOE' 
        BODY_38_PARTS_LEFT_HEEL 'sl::BODY_38_PARTS::LEFT_HEEL' 
        BODY_38_PARTS_RIGHT_HEEL 'sl::BODY_38_PARTS::RIGHT_HEEL'    
        BODY_38_PARTS_LEFT_HAND_THUMB_4 'sl::BODY_38_PARTS::LEFT_HAND_THUMB_4' 
        BODY_38_PARTS_RIGHT_HAND_THUMB_4 'sl::BODY_38_PARTS::RIGHT_HAND_THUMB_4' 
        BODY_38_PARTS_LEFT_HAND_INDEX_1 'sl::BODY_38_PARTS::LEFT_HAND_INDEX_1' 
        BODY_38_PARTS_RIGHT_HAND_INDEX_1 'sl::BODY_38_PARTS::RIGHT_HAND_INDEX_1' 
        BODY_38_PARTS_LEFT_HAND_MIDDLE_4 'sl::BODY_38_PARTS::LEFT_HAND_MIDDLE_4' 
        BODY_38_PARTS_RIGHT_HAND_MIDDLE_4 'sl::BODY_38_PARTS::RIGHT_HAND_MIDDLE_4' 
        BODY_38_PARTS_LEFT_HAND_PINKY_1 'sl::BODY_38_PARTS::LEFT_HAND_PINKY_1' 
        BODY_38_PARTS_RIGHT_HAND_PINKY_1 'sl::BODY_38_PARTS::RIGHT_HAND_PINKY_1' 
        BODY_38_PARTS_LAST 'sl::BODY_38_PARTS::LAST'

    cdef String toString(const BODY_38_PARTS &o)

    ctypedef enum INFERENCE_PRECISION 'sl::INFERENCE_PRECISION':
        INFERENCE_PRECISION_FP32 'sl::INFERENCE_PRECISION::FP32'
        INFERENCE_PRECISION_FP16 'sl::INFERENCE_PRECISION::FP16'
        INFERENCE_PRECISION_INT8 'sl::INFERENCE_PRECISION::INT8'
        INFERENCE_PRECISION_LAST 'sl::INFERENCE_PRECISION::LAST'

    cdef String toString(const INFERENCE_PRECISION &o)

    ctypedef enum BODY_FORMAT 'sl::BODY_FORMAT':
        BODY_FORMAT_BODY_18 'sl::BODY_FORMAT::BODY_18'
        BODY_FORMAT_BODY_34 'sl::BODY_FORMAT::BODY_34'
        BODY_FORMAT_BODY_38 'sl::BODY_FORMAT::BODY_38'
        BODY_FORMAT_LAST 'sl::BODY_FORMAT::LAST'

    cdef String toString(const BODY_FORMAT &o)
    cdef bool fromString(const String &s, BODY_FORMAT &o)

    ctypedef enum BODY_KEYPOINTS_SELECTION 'sl::BODY_KEYPOINTS_SELECTION':
        BODY_KEYPOINTS_SELECTION_FULL 'sl::BODY_KEYPOINTS_SELECTION::FULL'
        BODY_KEYPOINTS_SELECTION_UPPER_BODY 'sl::BODY_KEYPOINTS_SELECTION::UPPER_BODY'
        BODY_KEYPOINTS_SELECTION_LAST 'sl::BODY_KEYPOINTS_SELECTION::LAST'

    cdef String toString(const BODY_KEYPOINTS_SELECTION &o)
    cdef bool fromString(const String &s, BODY_KEYPOINTS_SELECTION &o)

    int getIdx(BODY_18_PARTS part)
    int getIdx(BODY_34_PARTS part)
    int getIdx(BODY_38_PARTS part)

    cdef cppclass Mat 'sl::Mat':
        String name
        bool verbose
        Timestamp timestamp

        Mat()
        Mat(size_t width, size_t height, MAT_TYPE mat_type, MEM memory_type)
        Mat(size_t width, size_t height, MAT_TYPE mat_type, unsigned char *ptr, size_t step, MEM memory_type)
        Mat(size_t width, size_t height, MAT_TYPE mat_type, unsigned char *ptr_cpu, size_t step_cpu, unsigned char *ptr_gpu, size_t step_gpu)
        Mat(const Mat &mat)

        void alloc(size_t width, size_t height, MAT_TYPE mat_type, MEM memory_type)
        void alloc(Resolution resolution, MAT_TYPE mat_type, MEM memory_type)
        void free(MEM memory_type)
        Mat &operator=(const Mat &that)
        ERROR_CODE copyTo(Mat &dst, COPY_TYPE cpyType, cudaStream_t stream = 0) const
        ERROR_CODE updateCPUfromGPU(cudaStream_t stream = 0)
        ERROR_CODE updateGPUfromCPU(cudaStream_t stream = 0, int gpu_id = 0)
        ERROR_CODE setFrom(const Mat &src, COPY_TYPE cpyType, cudaStream_t stream = 0) const
        ERROR_CODE read(const String &filePath)
        ERROR_CODE write(const String &filePath, MEM memory_type, int compression_level)
        size_t getWidth() const
        size_t getHeight() const
        const Resolution& getResolution() const
        size_t getChannels() const
        MAT_TYPE getDataType() const
        MEM getMemoryType() const
        size_t getStepBytes(const MEM memory_type)
        size_t getStep(MEM memory_type)
        size_t getPixelBytes()
        size_t getWidthBytes()
        String getInfos()
        bool isInit()
        bool isMemoryOwner()
        ERROR_CODE clone(const Mat &src)
        ERROR_CODE move(Mat &dst)
        ERROR_CODE convertColor(MEM memory_type, cudaStream_t stream = 0, bool swap_RB_channels = true)

        @staticmethod
        void swap(Mat &mat1, Mat &mat2)

        @staticmethod
        ERROR_CODE convertColor(Mat &mat1, Mat &mat2, bool swap_RB_channels, bool remove_alpha_channels, MEM memory_type, cudaStream_t stream)

    ctypedef enum TENSOR_COLOR_FORMAT 'sl::TensorParameters::COLOR_FORMAT':
        TENSOR_COLOR_FORMAT_GRAY 'sl::TensorParameters::COLOR_FORMAT::GRAY'
        TENSOR_COLOR_FORMAT_Y 'sl::TensorParameters::COLOR_FORMAT::Y'
        TENSOR_COLOR_FORMAT_RGB 'sl::TensorParameters::COLOR_FORMAT::RGB'
        TENSOR_COLOR_FORMAT_BGR 'sl::TensorParameters::COLOR_FORMAT::BGR'
        TENSOR_COLOR_FORMAT_RGBA 'sl::TensorParameters::COLOR_FORMAT::RGBA'
        TENSOR_COLOR_FORMAT_BGRA 'sl::TensorParameters::COLOR_FORMAT::BGRA'

    ctypedef enum TENSOR_LAYOUT 'sl::TensorParameters::LAYOUT':
        TENSOR_LAYOUT_NCHW 'sl::TensorParameters::LAYOUT::NCHW'
        TENSOR_LAYOUT_NHWC 'sl::TensorParameters::LAYOUT::NHWC'

    ctypedef enum TENSOR_PIXEL_TYPE 'sl::TensorParameters::PIXEL_TYPE':
        TENSOR_PIXEL_TYPE_FLOAT 'sl::TensorParameters::PIXEL_TYPE::FLOAT'
        TENSOR_PIXEL_TYPE_HALF 'sl::TensorParameters::PIXEL_TYPE::HALF'
        TENSOR_PIXEL_TYPE_UCHAR 'sl::TensorParameters::PIXEL_TYPE::UCHAR'

    cdef cppclass TensorParameters 'sl::TensorParameters':
        Resolution target_size
        TENSOR_PIXEL_TYPE pixel_type
        Vector3[float] mean
        Vector3[float] std
        bool stretch
        Vector3[float] scale
        TENSOR_COLOR_FORMAT color_format
        TENSOR_LAYOUT layout
        size_t batch_size
        MEM memory_type

        size_t getPixelSize() const
        int getRequestedChannels() const

    cdef cppclass Tensor 'sl::Tensor':
        Tensor()
        Tensor(const TensorParameters &params)
        Tensor(const Tensor &tensor)

        void alloc(const TensorParameters &params)
        vector[size_t] getDims() const
        size_t getElementCount() const
        TensorParameters getParameters() const


    cdef ERROR_CODE blobFromImage(Mat &mat1, Mat &mat2, Resolution resolution, float scale, Vector3[float] mean, Vector3[float] stdev, bool keep_aspect_ratio, bool swap_RB_channels, cudaStream_t stream)
    cdef ERROR_CODE blobFromImages(vector[Mat] &mats, Mat &mat2, Resolution resolution, float scale, Vector3[float] mean, Vector3[float] stdev, bool keep_aspect_ratio, bool swap_RB_channels, cudaStream_t stream)
    cdef ERROR_CODE convertImage(Mat &image_in, Mat &image_signed, cudaStream_t stream)
    cdef size_t getStep_internal(const size_t data_type, const size_t width)

    cdef bool isCameraOne(const MODEL camera_model)
    cdef bool isResolutionAvailable "isAvailable"(const RESOLUTION resolution, const MODEL camera_model)
    cdef bool isFPSAvailable "isAvailable"(const int fps, const RESOLUTION resolution, const MODEL camera_model)
    cdef bool supportHDR(const RESOLUTION resolution, const MODEL camera_model)
    cdef void setEnvironmentVariable(const String &name, const String &value)
    cdef const vector[int] &getAvailableCameraFPS()
    cdef AI_Model_status checkAIModelStatus(AI_MODELS model, int gpu_id)
    cdef ERROR_CODE downloadAIModel(AI_MODELS model, int gpu_id)
    cdef ERROR_CODE optimizeAIModel(AI_MODELS model, int gpu_id)
    cdef Rotation computeRotationMatrixFromGravity(Vector3[float] axis_to_align, Vector3[float] gravity)

    cdef cppclass Rotation(Matrix3f):
        Rotation()
        Rotation(const Rotation &rotation)
        Rotation(const Matrix3f &mat)
        Rotation(const Orientation &orientation)
        Rotation(const float angle, const Translation &axis)
        void setOrientation(const Orientation &orientation)
        Orientation getOrientation() const
        Vector3[float] getRotationVector()
        void setRotationVector(const Vector3[float] &vec_rot)
        Vector3[float] getEulerAngles(bool radian) const
        void setEulerAngles(const Vector3[float] &euler_angles, bool radian)


    cdef cppclass Translation(Vector3):
        Translation()
        Translation(const Translation &translation)
        Translation(Vector3[float] input)
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

cdef extern from "sl/Camera.hpp" namespace "sl":

    ctypedef enum MESH_FILE_FORMAT 'sl::MESH_FILE_FORMAT':
        MESH_FILE_FORMAT_PLY 'sl::MESH_FILE_FORMAT::PLY'
        MESH_FILE_FORMAT_PLY_BIN 'sl::MESH_FILE_FORMAT::PLY_BIN'
        MESH_FILE_FORMAT_OBJ 'sl::MESH_FILE_FORMAT::OBJ'
        MESH_FILE_FORMAT_LAST 'sl::MESH_FILE_FORMAT::LAST'

    cdef String toString(const MESH_FILE_FORMAT &o)

    ctypedef enum MESH_TEXTURE_FORMAT 'sl::MESH_TEXTURE_FORMAT':
        MESH_TEXTURE_FORMAT_RGB 'sl::MESH_TEXTURE_FORMAT::RGB'
        MESH_TEXTURE_FORMAT_RGBA 'sl::MESH_TEXTURE_FORMAT::RGBA'
        MESH_TEXTURE_FORMAT_LAST 'sl::MESH_TEXTURE_FORMAT::LAST'

    cdef String toString(const MESH_TEXTURE_FORMAT &o)

    ctypedef enum MESH_FILTER 'sl::MeshFilterParameters::MESH_FILTER':
        MESH_FILTER_LOW 'sl::MeshFilterParameters::MESH_FILTER::LOW'
        MESH_FILTER_MEDIUM 'sl::MeshFilterParameters::MESH_FILTER::MEDIUM'
        MESH_FILTER_HIGH 'sl::MeshFilterParameters::MESH_FILTER::HIGH'
        MESH_FILTER_LAST 'sl::MeshFilterParameters::MESH_FILTER::LAST'

    cdef String toString(const MESH_FILTER &o)

    ctypedef enum PLANE_TYPE 'sl::PLANE_TYPE':
        PLANE_TYPE_HORIZONTAL 'sl::PLANE_TYPE::HORIZONTAL'
        PLANE_TYPE_VERTICAL 'sl::PLANE_TYPE::VERTICAL'
        PLANE_TYPE_UNKNOWN 'sl::PLANE_TYPE::UNKNOWN'
        PLANE_TYPE_LAST 'sl::PLANE_TYPE::LAST'

    cdef String toString(const PLANE_TYPE &o)

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
        vector[Vector3[unsigned char]] colors
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

    ctypedef enum MESH_CREATION 'sl::MESH_CREATION':
        MESH_CREATION_NONE 'sl::MESH_CREATION::NONE'
        MESH_CREATION_LIVE 'sl::MESH_CREATION::LIVE'
        MESH_CREATION_LOAD 'sl::MESH_CREATION::LOAD'

    cdef cppclass Mesh 'sl::Mesh':
        ctypedef vector[size_t] chunkList
        Mesh()
        vector[Chunk] chunks
        Chunk &operator[](int index)
        vector[Vector4[float]] vertices
        vector[Vector3[uint]] triangles
        vector[Vector3[float]] normals
        vector[Vector3[unsigned char]] colors
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
        bool load(String filename, bool update_chunk_only)
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
        bool load(String filename, bool update_chunk_only)
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
        SPATIAL_MAP_TYPE_MESH 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH'
        SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::FUSED_POINT_CLOUD'
        SPATIAL_MAP_TYPE_LAST 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::LAST'

    cdef String toString(const SPATIAL_MAP_TYPE &o)
    cdef bool fromString(const String &s, SPATIAL_MAP_TYPE &o)

    ctypedef enum MAPPING_RESOLUTION 'sl::SpatialMappingParameters::MAPPING_RESOLUTION':
        MAPPING_RESOLUTION_HIGH 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::HIGH'
        MAPPING_RESOLUTION_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::MEDIUM'
        MAPPING_RESOLUTION_LOW 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::LOW'
        MAPPING_RESOLUTION_LAST 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::LAST'

    cdef String toString(const MAPPING_RESOLUTION &o)

    ctypedef enum MAPPING_RANGE 'sl::SpatialMappingParameters::MAPPING_RANGE':
        MAPPING_RANGE_SHORT 'sl::SpatialMappingParameters::MAPPING_RANGE::SHORT'
        MAPPING_RANGE_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RANGE::MEDIUM'
        MAPPING_RANGE_LONG 'sl::SpatialMappingParameters::MAPPING_RANGE::LONG'
        MAPPING_RANGE_AUTO 'sl::SpatialMappingParameters::MAPPING_RANGE::AUTO'
        MAPPING_RANGE_LAST 'sl::SpatialMappingParameters::MAPPING_RANGE::LAST'

    cdef String toString(const MAPPING_RANGE &o)

    ctypedef enum BUS_TYPE 'sl::BUS_TYPE':
        BUS_TYPE_USB 'sl::BUS_TYPE::USB'
        BUS_TYPE_GMSL 'sl::BUS_TYPE::GMSL'
        BUS_TYPE_AUTO 'sl::BUS_TYPE::AUTO'
        BUS_TYPE_LAST 'sl::BUS_TYPE::LAST'

    cdef String toString(const BUS_TYPE &o)

    uint generateVirtualStereoSerialNumber(uint serial_left, uint serial_right)

    ctypedef enum TYPE_OF_INPUT_TYPE 'sl::InputType::INPUT_TYPE':
        TYPE_OF_INPUT_TYPE_USB_ID 'sl::InputType::INPUT_TYPE::USB_ID'
        TYPE_OF_INPUT_TYPE_USB_SERIAL 'sl::InputType::INPUT_TYPE::USB_SERIAL'
        TYPE_OF_INPUT_TYPE_SVO_FILE 'sl::InputType::INPUT_TYPE::SVO_FILE'
        TYPE_OF_INPUT_TYPE_STREAM 'sl::InputType::INPUT_TYPE::STREAM'
        TYPE_OF_INPUT_TYPE_GMSL_ID 'sl::InputType::INPUT_TYPE::GMSL_ID'
        TYPE_OF_INPUT_TYPE_GMSL_SERIAL 'sl::InputType::INPUT_TYPE::GMSL_SERIAL'
        TYPE_OF_INPUT_TYPE_GMSL_PORT 'sl::InputType::INPUT_TYPE::GMSL_PORT'
        TYPE_OF_INPUT_TYPE_AUTO 'sl::InputType::INPUT_TYPE::AUTO'
        TYPE_OF_INPUT_TYPE_LAST 'sl::InputType::INPUT_TYPE::LAST'

    cdef String toString(const TYPE_OF_INPUT_TYPE &o)
    cdef bool fromString(const String &s, TYPE_OF_INPUT_TYPE &o)

    cdef cppclass InputType 'sl::InputType':
        InputType()
        InputType(const InputType &type)

        void setFromCameraID(int cam_id, BUS_TYPE bus_type)
        void setFromSerialNumber(uint serial_number)
        void setFromSerialNumber(String serial_number)
        void setFromGMSLPort(int gmsl_port)
        bool setVirtualStereoFromCameraIDs(int id_left, int id_right, uint virtual_serial_number)
        bool setVirtualStereoFromSerialNumbers(uint camera_left_serial_number, uint camera_right_serial_number, uint virtual_serial_number)
        void setFromSVOFile(String svo_input_filename)
        void setFromStream(String senderIP, unsigned short port)
        TYPE_OF_INPUT_TYPE getType() const
        String getConfiguration() const
        bool isInit() const
        bool operator==(const InputType &that) const
        bool operator!=(const InputType &that) const

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
        int sdk_verbose

        String sdk_verbose_log_file

        CUcontext sdk_cuda_ctx
        InputType input
        String optional_settings_path
        bool sensors_required
        bool enable_image_enhancement
        String optional_opencv_calibration_file
        float open_timeout_sec
        bool async_grab_camera_recovery
        float grab_compute_capping_fps
        bool async_image_retrieval
        int enable_image_validity_check
        Resolution maximum_working_resolution

        InitParameters(RESOLUTION camera_resolution,
                       int camera_fps,
                       bool svo_real_time_mode,
                       DEPTH_MODE depth_mode,
                       UNIT coordinate_units,
                       COORDINATE_SYSTEM coordinate_system,
                       int sdk_verbose,
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
                       String optional_opencv_calibration_file,
                       float open_timeout_sec,
                       bool async_grab_camera_recovery,
                       float grab_compute_capping_fps,
                       bool async_image_retrieval,
                       int enable_image_validity_check,
                       Resolution maximum_working_resolution)

        bool save(String filename)
        bool load(String filename)

    cdef cppclass RecordingParameters 'sl::RecordingParameters':
        String video_filename
        SVO_COMPRESSION_MODE compression_mode
        uint bitrate
        uint target_framerate
        bool transcode_streaming_input

        RecordingParameters(String video_filename_, 
                            SVO_COMPRESSION_MODE compression_mode_,
                            uint target_framerate,
                            uint bitrate,
                            bool transcode_streaming_input
                            )

    cdef cppclass RuntimeParameters 'sl::RuntimeParameters':
        bool enable_depth
        bool enable_fill_mode
        int confidence_threshold
        int texture_confidence_threshold
        REFERENCE_FRAME measure3D_reference_frame
        bool remove_saturated_areas

        RuntimeParameters(bool enable_depth,
                          bool enable_fill_mode,
                          int confidence_threshold,
                          int texture_confidence_threshold,
                          REFERENCE_FRAME measure3D_reference_frame,
                          bool remove_saturated_areas)

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
        float depth_min_range
        bool set_gravity_as_origin
        POSITIONAL_TRACKING_MODE mode
        bool enable_localization_only
        bool enable_2d_ground_mode

        PositionalTrackingParameters(Transform init_pos,
                           bool _enable_memory,
                           bool _enable_pose_smoothing,
                           String _area_path,
                           bool _set_floor_as_origin,
                           bool _enable_imu_fusion,
                           bool _set_as_static,
                           float _depth_min_range,
                           bool _set_gravity_as_origin,
                           POSITIONAL_TRACKING_MODE _mode,
                           bool enable_localization_only,
                           bool enable_2d_ground_mode)

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
        float stability_counter

        bool save(String filename)
        bool load(String filename)


    ctypedef enum STREAMING_CODEC 'sl::STREAMING_CODEC':
        STREAMING_CODEC_H264 'sl::STREAMING_CODEC::H264'
        STREAMING_CODEC_H265 'sl::STREAMING_CODEC::H265'
        STREAMING_CODEC_LAST 'sl::STREAMING_CODEC::LAST'

    cdef String toString(const STREAMING_CODEC &o)
    cdef bool fromString(const String &s, STREAMING_CODEC &o)

    cdef struct StreamingProperties 'sl::StreamingProperties':
        String ip
        unsigned short port
        uint serial_number
        int current_bitrate
        STREAMING_CODEC codec
        MODEL camera_model

    cdef String toString(const StreamingProperties &o)

    cdef cppclass StreamingParameters 'sl::StreamingParameters':
        STREAMING_CODEC codec
        unsigned short port
        uint bitrate
        int gop_size
        bool adaptative_bitrate
        unsigned short chunk_size
        uint target_framerate
        StreamingParameters(STREAMING_CODEC codec, unsigned short port_, uint bitrate, int gop_size, bool adaptative_bitrate_, unsigned short chunk_size_, uint target_framerate)

    cdef cppclass BatchParameters 'sl::BatchParameters':
        bool enable
        float id_retention_time
        float latency
        BatchParameters(bool enable, float id_retention_time, float batch_duration)

    cdef cppclass ObjectDetectionParameters 'sl::ObjectDetectionParameters':
        bool enable_tracking
        bool enable_segmentation
        OBJECT_DETECTION_MODEL detection_model
        float max_range
        BatchParameters batch_parameters
        OBJECT_FILTERING_MODE filtering_mode
        float prediction_timeout_s
        bool allow_reduced_precision_inference
        uint instance_module_id
        String fused_objects_group_name
        String custom_onnx_file
        Resolution custom_onnx_dynamic_input_shape

        ObjectDetectionParameters(bool enable_tracking, 
                bool enable_segmentation, 
                OBJECT_DETECTION_MODEL detection_model, 
                float max_range, 
                BatchParameters batch_trajectories_parameters, 
                OBJECT_FILTERING_MODE filtering_mode, 
                float prediction_timeout_s, 
                bool allow_reduced_precision_inference,
                uint instance_module_id,
                const String& fused_objects_group_name,
                const String& custom_onnx_file_,
                const Resolution& custom_onnx_dynamic_input_shape_
            )



    cdef cppclass ObjectDetectionRuntimeParameters 'sl::ObjectDetectionRuntimeParameters':
        float detection_confidence_threshold
        vector[OBJECT_CLASS] object_class_filter
        map[OBJECT_CLASS,float] object_class_detection_confidence_threshold
        ObjectTrackingParameters object_tracking_parameters
        map[OBJECT_CLASS, ObjectTrackingParameters] object_class_tracking_parameters
        ObjectDetectionRuntimeParameters(float detection_confidence_threshold, vector[OBJECT_CLASS] object_class_filter, map[OBJECT_CLASS,float] object_class_detection_confidence_threshold, ObjectTrackingParameters object_tracking_parameters, map[OBJECT_CLASS, ObjectTrackingParameters] object_class_tracking_parameters)

    cdef cppclass CustomObjectDetectionProperties 'sl::CustomObjectDetectionProperties':
        bool enabled
        float detection_confidence_threshold
        bool is_grounded
        bool is_static
        float tracking_timeout
        float tracking_max_dist
        float max_box_width_normalized
        float min_box_width_normalized
        float max_box_height_normalized
        float min_box_height_normalized
        float max_box_width_meters
        float min_box_width_meters
        float max_box_height_meters
        float min_box_height_meters
        OBJECT_SUBCLASS native_mapped_class
        OBJECT_ACCELERATION_PRESET object_acceleration_preset
        float max_allowed_acceleration
        ObjectTrackingParameters object_tracking_parameters
        CustomObjectDetectionProperties(bool enabled,
                                        float detection_confidence_threshold,
                                        bool is_grounded,
                                        bool is_static,
                                        float tracking_timeout,
                                        float tracking_max_dist,
                                        float max_box_width_normalized,
                                        float min_box_width_normalized,
                                        float max_box_height_normalized,
                                        float min_box_height_normalized,
                                        float max_box_width_meters,
                                        float min_box_width_meters,
                                        float max_box_height_meters,
                                        float min_box_height_meters,
                                        OBJECT_SUBCLASS native_mapped_class,
                                        OBJECT_ACCELERATION_PRESET object_acceleration_preset,
                                        float max_allowed_acceleration,
                                        ObjectTrackingParameters object_tracking_parameters)

    cdef cppclass CustomObjectDetectionRuntimeParameters 'sl::CustomObjectDetectionRuntimeParameters':
        CustomObjectDetectionProperties object_detection_properties
        unordered_map[int, CustomObjectDetectionProperties] object_class_detection_properties
        CustomObjectDetectionRuntimeParameters(CustomObjectDetectionProperties object_detection_properties,
                                               unordered_map[int, CustomObjectDetectionProperties] object_class_detection_properties)

    cdef cppclass BodyTrackingParameters 'sl::BodyTrackingParameters':
        bool enable_tracking
        bool enable_segmentation
        BODY_TRACKING_MODEL detection_model
        bool enable_body_fitting
        BODY_FORMAT body_format
        BODY_KEYPOINTS_SELECTION body_selection
        float max_range
        float prediction_timeout_s
        bool allow_reduced_precision_inference
        uint instance_module_id
        BodyTrackingParameters(bool enable_tracking, 
                    bool enable_segmentation, 
                    BODY_TRACKING_MODEL detection_model, 
                    bool enable_body_fitting, 
                    float max_range, 
                    BODY_FORMAT body_format, 
                    BODY_KEYPOINTS_SELECTION body_selection, 
                    float prediction_timeout_s, 
                    bool allow_reduced_precision_inference, 
                    uint instance_module_id
            )

    cdef cppclass BodyTrackingRuntimeParameters 'sl::BodyTrackingRuntimeParameters':
        float detection_confidence_threshold
        int minimum_keypoints_threshold
        float skeleton_smoothing
        BodyTrackingRuntimeParameters(float detection_confidence_threshold, int minimum_keypoints_threshold, float skeleton_smoothing)
    
    cdef cppclass PlaneDetectionParameters 'sl::PlaneDetectionParameters':
        float max_distance_threshold
        float normal_similarity_threshold
        PlaneDetectionParameters()

    cdef cppclass RegionOfInterestParameters 'sl::RegionOfInterestParameters':
        float depth_far_threshold_meters
        float image_height_ratio_cutoff
        unordered_set[MODULE] auto_apply_module
    
    cdef cppclass Pose 'sl::Pose':
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

        float twist[6]
        float twist_covariance[36]

    cdef cppclass Landmark 'sl::Landmark':
        unsigned long long id
        Vector3[float] position

    cdef cppclass Landmark2D 'sl::Landmark2D':
        unsigned long long id
        Vector2[uint] image_position
        float dynamic_confidence

    cdef cppclass KeyFrame 'sl::KeyFrame':
        Transform pose
        Timestamp timestamp
        bool is_loaded

    ctypedef enum CAMERA_MOTION_STATE 'sl::SensorsData::CAMERA_MOTION_STATE':
        CAMERA_MOTION_STATE_STATIC 'sl::SensorsData::CAMERA_MOTION_STATE::STATIC'
        CAMERA_MOTION_STATE_MOVING 'sl::SensorsData::CAMERA_MOTION_STATE::MOVING'
        CAMERA_MOTION_STATE_FALLING 'sl::SensorsData::CAMERA_MOTION_STATE::FALLING'
        CAMERA_MOTION_STATE_LAST 'sl::SensorsData::CAMERA_MOTION_STATE::LAST'

    cdef String toString(const CAMERA_MOTION_STATE &o)

    cdef cppclass BarometerData 'sl::SensorsData::BarometerData':
        bool is_available
        float pressure
        float relative_altitude
        Timestamp timestamp
        float effective_rate

        BarometerData()

    ctypedef enum SENSOR_LOCATION 'sl::SensorsData::TemperatureData::SENSOR_LOCATION':
        SENSOR_LOCATION_IMU 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::IMU'
        SENSOR_LOCATION_BAROMETER 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::BAROMETER'
        SENSOR_LOCATION_ONBOARD_LEFT 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_LEFT'
        SENSOR_LOCATION_ONBOARD_RIGHT 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_RIGHT'
        SENSOR_LOCATION_LAST 'sl::SensorsData::TemperatureData::SENSOR_LOCATION::LAST'

    cdef String toString(const SENSOR_LOCATION &o)

    cdef cppclass TemperatureData 'sl::SensorsData::TemperatureData':
        ERROR_CODE get(SENSOR_LOCATION location, float& temperature)
        map[SENSOR_LOCATION,float] temperature_map
        TemperatureData()

    ctypedef enum HEADING_STATE 'sl::SensorsData::MagnetometerData::HEADING_STATE':
        HEADING_STATE_GOOD 'sl::SensorsData::MagnetometerData::HEADING_STATE::GOOD'
        HEADING_STATE_OK 'sl::SensorsData::MagnetometerData::HEADING_STATE::OK'
        HEADING_STATE_NOT_GOOD 'sl::SensorsData::MagnetometerData::HEADING_STATE::NOT_GOOD'
        HEADING_STATE_NOT_CALIBRATED 'sl::SensorsData::MagnetometerData::HEADING_STATE::NOT_CALIBRATED'
        HEADING_STATE_MAG_NOT_AVAILABLE 'sl::SensorsData::MagnetometerData::HEADING_STATE::MAG_NOT_AVAILABLE'
        HEADING_STATE_LAST 'sl::SensorsData::MagnetometerData::HEADING_STATE::LAST'

    cdef String toString(const HEADING_STATE &o)

    cdef cppclass MagnetometerData 'sl::SensorsData::MagnetometerData':
        bool is_available
        Timestamp timestamp
        float effective_rate
        float magnetic_heading
        float magnetic_heading_accuracy
        HEADING_STATE magnetic_heading_state

        Vector3[float] magnetic_field_uncalibrated
        Vector3[float] magnetic_field_calibrated
        MagnetometerData()

    cdef cppclass IMUData 'sl::SensorsData::IMUData':
        bool is_available
        Timestamp timestamp
        float effective_rate
        Transform pose
        Matrix3f pose_covariance
        Vector3[float] angular_velocity
        Vector3[float] angular_velocity_uncalibrated
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
        FLIP_MODE_OFF 'sl::FLIP_MODE::OFF'
        FLIP_MODE_ON 'sl::FLIP_MODE::ON'
        FLIP_MODE_AUTO 'sl::FLIP_MODE::AUTO'
        FLIP_MODE_LAST 'sl::FLIP_MODE::LAST'
    
    cdef String toString(const FLIP_MODE &o)

    Resolution getResolution(RESOLUTION resolution)

    cdef cppclass HealthStatus 'sl::HealthStatus':
        bool enabled
        bool low_image_quality
        bool low_lighting
        bool low_depth_reliability
        bool low_motion_sensors_reliability

    cdef String toString(const HealthStatus &status)

    cdef cppclass Camera 'sl::Camera':
        Camera()
        void close()
        ERROR_CODE open(InitParameters init_parameters)

        InitParameters getInitParameters()

        bool isOpened()
        ERROR_CODE read() nogil
        ERROR_CODE grab(RuntimeParameters rt_parameters) nogil

        RuntimeParameters getRuntimeParameters()

        ERROR_CODE retrieveImage(Mat &mat, VIEW view, MEM type, Resolution resolution, cudaStream_t stream = 0) nogil
        ERROR_CODE retrieveMeasure(Mat &mat, MEASURE measure, MEM type, Resolution resolution, cudaStream_t stream = 0) nogil
        ERROR_CODE retrieveTensor(Tensor &dest, const TensorParameters& params, cudaStream_t stream = 0) nogil
        ERROR_CODE getCurrentMinMaxDepth(float& min, float& max)

        ERROR_CODE setRegionOfInterest(Mat &mat, unordered_set[MODULE] modules)
        ERROR_CODE getRegionOfInterest(Mat &roi_mask, Resolution image_size, MODULE module)
        ERROR_CODE startRegionOfInterestAutoDetection(RegionOfInterestParameters roi_param)
        REGION_OF_INTEREST_AUTO_DETECTION_STATE getRegionOfInterestAutoDetectionStatus()

        ERROR_CODE startPublishing(CommunicationParameters parameters)
        ERROR_CODE stopPublishing()

        ERROR_CODE setSVOPosition(int position)
        void pauseSVOReading(bool status)
        int getSVOPosition()
        int getSVONumberOfFrames()

        ERROR_CODE ingestDataIntoSVO(const SVOData& data)
        ERROR_CODE retrieveSVOData(const string &key, map[Timestamp, SVOData] &data, Timestamp ts_begin, Timestamp ts_end)
        vector[string] getSVODataKeys()

        ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, int value)
        ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, int mini, int maxi)
        ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, Rect roi, SIDE eye, bool reset)

        int getSVOPositionAtTimestamp(const Timestamp &timestamp)

        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, int &settings)
        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, int &aec_min_val, int &aec_max_val)
        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, Rect &roi, SIDE eye)

        ERROR_CODE getCameraSettingsRange(VIDEO_SETTINGS settings, int &min, int &max)

        bool isCameraSettingSupported(VIDEO_SETTINGS setting)

        float getCurrentFPS()
        Timestamp getTimestamp(TIME_REFERENCE reference_time) nogil
        uint getFrameDroppedCount()
        CameraInformation getCameraInformation(Resolution resizer)

        ERROR_CODE enablePositionalTracking(PositionalTrackingParameters tracking_params)
        POSITIONAL_TRACKING_STATE getPosition(Pose &camera_pose, REFERENCE_FRAME reference_frame)
        PositionalTrackingStatus getPositionalTrackingStatus()
        ERROR_CODE getPositionalTrackingLandmarks(map[uint64_t, Landmark] & landmarks)
        ERROR_CODE getPositionalTrackingLandmarks2D(vector[Landmark2D] & current_2d_landmarks)
        ERROR_CODE getPositionalTrackingKeyframes(map[uint64_t, KeyFrame] & keyframes)
        ERROR_CODE saveAreaMap(String area_file_path)
        AREA_EXPORTING_STATE getAreaExportState()

        PositionalTrackingParameters getPositionalTrackingParameters()
        bool isPositionalTrackingEnabled()
        void disablePositionalTracking(String area_file_path)
        ERROR_CODE resetPositionalTracking(const Transform &path)
        ERROR_CODE getSensorsData(SensorsData &sensor_data, TIME_REFERENCE reference_time) nogil
        ERROR_CODE getSensorsDataBatch(vector[SensorsData] &sensor_data) nogil
        ERROR_CODE setIMUPrior(const Transform &transfom)

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

        ERROR_CODE findPlaneAtHit(Vector2[uint] coord, Plane &plane, PlaneDetectionParameters plane_detection_parameters)
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
        CommunicationParameters getCommunicationParameters()

        ERROR_CODE enableObjectDetection(ObjectDetectionParameters object_detection_parameters)
        void disableObjectDetection(uint instance_module_id, bool force_disable_all_instances)
        ERROR_CODE setObjectDetectionRuntimeParameters(const ObjectDetectionRuntimeParameters &parameters, const uint instance_id) nogil
        ERROR_CODE setCustomObjectDetectionRuntimeParameters(const CustomObjectDetectionRuntimeParameters &parameters, const uint instance_id) nogil
        ERROR_CODE retrieveObjects "retrieveObjects"(Objects &objects, const uint instance_module_id) nogil
        ERROR_CODE retrieveObjectsAndSetRuntimeParameters "retrieveObjects"(Objects &objects, ObjectDetectionRuntimeParameters parameters, const uint instance_module_id) nogil
        ERROR_CODE retrieveCustomObjectsAndSetRuntimeParameters "retrieveCustomObjects"(Objects &objects, CustomObjectDetectionRuntimeParameters parameters, const uint instance_module_id) nogil
        ERROR_CODE getObjectsBatch(vector[ObjectsBatch] &trajectories, uint instance_module_id)
        bool isObjectDetectionEnabled(uint instance_id)
        ERROR_CODE ingestCustomBoxObjects(const vector[CustomBoxObjectData] &objects_in, const uint instance_module_id)
        ERROR_CODE ingestCustomMaskObjects(const vector[CustomMaskObjectData] &objects_in, const uint instance_module_id)
        ObjectDetectionParameters getObjectDetectionParameters(uint instance_module_id)
        void updateSelfCalibration()

        ERROR_CODE enableBodyTracking(BodyTrackingParameters object_detection_parameters)
        void disableBodyTracking(uint instance_id, bool force_disable_all_instances)
        ERROR_CODE setBodyTrackingRuntimeParameters(const BodyTrackingRuntimeParameters &parameters, const uint instance_id) nogil
        ERROR_CODE retrieveBodies "retrieveBodies"(Bodies &objects, uint instance_id) nogil
        ERROR_CODE retrieveBodiesAndSetRuntimeParameters "retrieveBodies"(Bodies &objects, BodyTrackingRuntimeParameters parameters, uint instance_id) nogil
        bool isBodyTrackingEnabled(uint instance_id)
        BodyTrackingParameters getBodyTrackingParameters(uint instance_id)

        HealthStatus getHealthStatus()

        CUcontext getCUDAContext()
        void getSDKVersion(int &major, int &minor, int &patch)

        Resolution getRetrieveImageResolution(Resolution res)
        Resolution getRetrieveMeasureResolution(Resolution res)

        @staticmethod
        String getSDKVersion()

        @staticmethod
        vector[DeviceProperties] getDeviceList()

        @staticmethod
        vector[StreamingProperties] getStreamingDeviceList()

        @staticmethod
        ERROR_CODE reboot(int sn, bool fullReboot)

        @staticmethod
        ERROR_CODE reboot_from_type "reboot" (INPUT_TYPE input)

cdef extern from "Utils.cpp" namespace "sl":
    ObjectDetectionRuntimeParameters* create_object_detection_runtime_parameters(float confidence_threshold, vector[int] object_vector, map[int,float] object_confidence_map)

cdef extern from "sl/Fusion.hpp" namespace "sl":

    cdef cppclass FusionConfiguration 'sl::FusionConfiguration':
        int serial_number
        CommunicationParameters communication_parameters
        Transform pose
        InputType input_type
        bool override_gravity

    cdef cppclass CommunicationParameters 'sl::CommunicationParameters':
        CommunicationParameters()
        void setForSharedMemory()
        void setForLocalNetwork(int port)
        void setForLocalNetwork(string ip_address, int port)
        int getPort()
        string getIpAddress()
        COMM_TYPE getType()

        bool operator==(const CommunicationParameters &that) const
        bool operator!=(const CommunicationParameters &that) const

    ctypedef enum COMM_TYPE "sl::CommunicationParameters::COMM_TYPE":
        COMM_TYPE_LOCAL_NETWORK 'sl::CommunicationParameters::COMM_TYPE::LOCAL_NETWORK'
        COMM_TYPE_INTRA_PROCESS 'sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS'
        COMM_TYPE_LAST 'sl::CommunicationParameters::COMM_TYPE::LAST'

    cdef String toString(const COMM_TYPE &o)
    cdef bool fromString(const String &s, COMM_TYPE &o)

    FusionConfiguration readFusionConfigurationFile(const string& json_config_filename, const int serial_number, const COORDINATE_SYSTEM coord_system, const UNIT unit)    
    vector[FusionConfiguration] readFusionConfigurationFile2 "readFusionConfigurationFile"(const string& json_config_filename, const COORDINATE_SYSTEM coord_sys, const UNIT unit)
    vector[FusionConfiguration] readFusionConfiguration (const string& fusion_configuration, const COORDINATE_SYSTEM coord_sys, const UNIT unit)
    void writeConfigurationFile(const string& json_config_filename, vector[FusionConfiguration] &conf, const COORDINATE_SYSTEM coord_sys, const UNIT unit)

    cdef struct SynchronizationParameter 'sl::SynchronizationParameter':
        double windows_size
        double data_source_timeout
        bool keep_last_data
        double maximum_lateness

    cdef cppclass InitFusionParameters 'sl::InitFusionParameters':
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        bool output_performance_metrics
        bool verbose
        unsigned timeout_period_number
        int sdk_gpu_id
        CUcontext sdk_cuda_ctx
        SynchronizationParameter synchronization_parameters
        Resolution maximum_working_resolution

        InitFusionParameters(
            UNIT coordinate_units_,
            COORDINATE_SYSTEM coordinate_system_,
            bool output_performance_metrics, 
            bool verbose_,
            unsigned timeout_period_number,
            int sdk_gpu_id,
            CUcontext sdk_cuda_ctx,
            SynchronizationParameter synchronization_parameters_,
            Resolution maximum_working_resolution_
            )

    cdef cppclass CameraIdentifier 'sl::CameraIdentifier':
        CameraIdentifier()
        unsigned long long sn
        CameraIdentifier(unsigned long long sn_)
        String get()

    bool operator<(const CameraIdentifier &lhs, const CameraIdentifier &rhs)
    bool operator==(const CameraIdentifier &lhs, const CameraIdentifier &rhs)
    bool operator!=(const CameraIdentifier &lhs, const CameraIdentifier &rhs)

    ctypedef enum FUSION_ERROR_CODE "sl::FUSION_ERROR_CODE" :
        FUSION_ERROR_CODE_GNSS_DATA_NEED_FIX 'sl::FUSION_ERROR_CODE::GNSS_DATA_NEED_FIX'
        FUSION_ERROR_CODE_GNSS_DATA_COVARIANCE_MUST_VARY 'sl::FUSION_ERROR_CODE::GNSS_DATA_COVARIANCE_MUST_VARY'
        FUSION_ERROR_CODE_BODY_FORMAT_MISMATCH 'sl::FUSION_ERROR_CODE::BODY_FORMAT_MISMATCH'
        FUSION_ERROR_CODE_MODULE_NOT_ENABLED 'sl::FUSION_ERROR_CODE::MODULE_NOT_ENABLED'
        FUSION_ERROR_CODE_SOURCE_MISMATCH 'sl::FUSION_ERROR_CODE::SOURCE_MISMATCH'
        FUSION_ERROR_CODE_CONNECTION_TIMED_OUT 'sl::FUSION_ERROR_CODE::CONNECTION_TIMED_OUT'
        FUSION_ERROR_CODE_MEMORY_ALREADY_USED 'sl::FUSION_ERROR_CODE::MEMORY_ALREADY_USED'
        FUSION_ERROR_CODE_INVALID_IP_ADDRESS 'sl::FUSION_ERROR_CODE::INVALID_IP_ADDRESS'
        FUSION_ERROR_CODE_FAILURE 'sl::FUSION_ERROR_CODE::FAILURE'
        FUSION_ERROR_CODE_SUCCESS 'sl::FUSION_ERROR_CODE::SUCCESS'
        FUSION_ERROR_CODE_FUSION_INCONSISTENT_FPS 'sl::FUSION_ERROR_CODE::FUSION_INCONSISTENT_FPS'
        FUSION_ERROR_CODE_FUSION_FPS_TOO_LOW 'sl::FUSION_ERROR_CODE::FUSION_FPS_TOO_LOW'
        FUSION_ERROR_CODE_NO_NEW_DATA_AVAILABLE 'sl::FUSION_ERROR_CODE::NO_NEW_DATA_AVAILABLE'
        FUSION_ERROR_CODE_INVALID_TIMESTAMP 'sl::FUSION_ERROR_CODE::INVALID_TIMESTAMP'
        FUSION_ERROR_CODE_INVALID_COVARIANCE 'sl::FUSION_ERROR_CODE::INVALID_COVARIANCE'

    cdef String toString(const FUSION_ERROR_CODE &o)

    ctypedef enum SENDER_ERROR_CODE "sl::SENDER_ERROR_CODE":
        SENDER_ERROR_CODE_DISCONNECTED 'sl::SENDER_ERROR_CODE::DISCONNECTED'
        SENDER_ERROR_CODE_SUCCESS 'sl::SENDER_ERROR_CODE::SUCCESS'
        SENDER_ERROR_CODE_GRAB_ERROR 'sl::SENDER_ERROR_CODE::GRAB_ERROR'
        SENDER_ERROR_CODE_INCONSISTENT_FPS 'sl::SENDER_ERROR_CODE::INCONSISTENT_FPS'
        SENDER_ERROR_CODE_FPS_TOO_LOW 'sl::SENDER_ERROR_CODE::FPS_TOO_LOW'

    cdef String toString(const SENDER_ERROR_CODE &o)

    ctypedef enum POSITION_TYPE 'sl::POSITION_TYPE':
        POSITION_TYPE_RAW 'sl::POSITION_TYPE::RAW'
        POSITION_TYPE_FUSION 'sl::POSITION_TYPE::FUSION'
        POSITION_TYPE_LAST 'sl::POSITION_TYPE::LAST'

    cdef String toString(const POSITION_TYPE &o)

    ctypedef enum FUSION_REFERENCE_FRAME 'sl::FUSION_REFERENCE_FRAME':
        FUSION_REFERENCE_FRAME_WORLD 'sl::FUSION_REFERENCE_FRAME::WORLD'
        FUSION_REFERENCE_FRAME_BASELINK 'sl::FUSION_REFERENCE_FRAME::BASELINK'

    cdef struct GNSSCalibrationParameters 'sl::GNSSCalibrationParameters':
        float target_yaw_uncertainty
        bool enable_translation_uncertainty_target
        float target_translation_uncertainty
        bool enable_reinitialization
        float gnss_vio_reinit_threshold
        bool enable_rolling_calibration
        Vector3[float] gnss_antenna_position

    cdef struct PositionalTrackingFusionParameters 'sl::PositionalTrackingFusionParameters':
        bool enable_GNSS_fusion
        GNSSCalibrationParameters gnss_calibration_parameters
        Transform base_footprint_to_world_transform
        Transform base_footprint_to_baselink_transform
        bool set_gravity_as_origin
        CameraIdentifier tracking_camera_id

    cdef struct SpatialMappingFusionParameters 'sl::SpatialMappingFusionParameters':
        float resolution_meter
        float range_meter
        bool use_chunk_only
        int max_memory_usage
        float disparity_std
        float decay
        bool enable_forget_past
        int stability_counter
        SPATIAL_MAP_TYPE map_type

    cdef struct BodyTrackingFusionParameters 'sl::BodyTrackingFusionParameters':
        bool enable_tracking
        bool enable_body_fitting

    cdef struct BodyTrackingFusionRuntimeParameters 'sl::BodyTrackingFusionRuntimeParameters':
        int skeleton_minimum_allowed_keypoints
        int skeleton_minimum_allowed_camera
        float skeleton_smoothing

    cdef struct ObjectDetectionFusionParameters 'sl::ObjectDetectionFusionParameters':
        bool enable_tracking

    cdef struct CameraMetrics 'sl::CameraMetrics':
        CameraMetrics()
        float received_fps
        float received_latency
        float synced_latency
        bool is_present
        float ratio_detection
        float delta_ts

    cdef struct FusionMetrics 'sl::FusionMetrics':
        FusionMetrics()
        void reset()
        float mean_camera_fused
        float mean_stdev_between_camera
        map[CameraIdentifier, CameraMetrics] camera_individual_stats

    cdef cppclass ECEF 'sl::ECEF':
        double x
        double y
        double z

    cdef cppclass ENU 'sl::ENU':
        double east
        double north
        double up

    cdef cppclass LatLng 'sl::LatLng':
        void getCoordinates(double & latitude, double & longitude, double & altitude, bool in_radian)
        void setCoordinates(double latitude, double longitude, double altitude, bool in_radian)
        double getLatitude(bool in_radian)
        double getLongitude(bool in_radian)
        double getAltitude()

    cdef cppclass UTM 'sl::UTM':
        double northing
        double easting
        double gamma
        string UTMZone

    cdef cppclass GeoConverter 'sl::GeoConverter':

        @staticmethod
        void ecef2latlng(ECEF &input, LatLng& out)

        @staticmethod
        void ecef2utm(ECEF &input, UTM &out)

        @staticmethod
        void latlng2ecef(LatLng &input, ECEF &out)

        @staticmethod
        void latlng2utm(LatLng &input, UTM &out)

        @staticmethod
        void utm2ecef(UTM &input, ECEF &out)

        @staticmethod
        void utm2latlng(UTM &input, LatLng &out)

    cdef cppclass GeoPose 'sl::GeoPose':
        GeoPose()
        GeoPose(GeoPose &geopose)

        double getLatitude()
        double getLongitude()
        double getAltitude()

        Transform pose_data
        Timestamp timestamp
        float pose_covariance[36]
        double horizontal_accuracy
        double vertical_accuracy
        LatLng latlng_coordinates
        double heading

    cdef struct GNSSData 'sl::GNSSData':
        void setCoordinates(double latitude, double longitude, double altitude, bool is_radian);
        void getCoordinates(double &latitude, double &longitude, double &altitude, bool in_radian);
        Timestamp ts
        array9 position_covariance
        double longitude_std
        double latitude_std
        double altitude_std
        GNSS_STATUS gnss_status
        GNSS_MODE gnss_mode

        GNSSData()

    cdef cppclass Fusion 'sl::Fusion':
        Fusion()
        FUSION_ERROR_CODE init(InitFusionParameters init_parameters)
        void close()
        FUSION_ERROR_CODE subscribe(const FusionConfiguration &fusion_config)
        FUSION_ERROR_CODE subscribe(CameraIdentifier uuid, CommunicationParameters param, Transform pose, bool override_gravity)
        FUSION_ERROR_CODE unsubscribe(CameraIdentifier uuid)
        FUSION_ERROR_CODE updatePose(CameraIdentifier uuid, Transform pose)
        FUSION_ERROR_CODE getProcessMetrics(FusionMetrics &metrics)
        map[CameraIdentifier, SENDER_ERROR_CODE] getSenderState()
        FUSION_ERROR_CODE process() nogil

        FUSION_ERROR_CODE retrieveImage(Mat &mat, CameraIdentifier uuid, Resolution resolution) nogil
        FUSION_ERROR_CODE retrieveMeasure(Mat &mat, CameraIdentifier uuid, MEASURE measure, Resolution resolution, FUSION_REFERENCE_FRAME reference_frame) nogil

        FUSION_ERROR_CODE enableBodyTracking(const BodyTrackingFusionParameters &params)
        FUSION_ERROR_CODE retrieveBodies(Bodies &objs, BodyTrackingFusionRuntimeParameters parameters, CameraIdentifier uuid, FUSION_REFERENCE_FRAME reference_frame) nogil
        void disableBodyTracking()

        FUSION_ERROR_CODE enableObjectDetection(const ObjectDetectionFusionParameters& params)
        FUSION_ERROR_CODE retrieveObjectsAllODGroups "retrieveObjects"(unordered_map[String, Objects] &objs, FUSION_REFERENCE_FRAME reference_frame) nogil
        FUSION_ERROR_CODE retrieveObjectsOneODGroup "retrieveObjects"(Objects &objs, const String& fused_od_group_name, FUSION_REFERENCE_FRAME reference_frame) nogil
        FUSION_ERROR_CODE retrieveObjectsAllIds "retrieveObjects"(unordered_map[uint, Objects] &objs, const CameraIdentifier& uuid) nogil
        FUSION_ERROR_CODE retrieveObjectsOneId "retrieveObjects"(Objects &objs, const CameraIdentifier& uuid, const uint instance_id) nogil
        void disableObjectDetection()

        FUSION_ERROR_CODE getPose(CameraIdentifier uuid, Transform &pose)

        FUSION_ERROR_CODE enablePositionalTracking(PositionalTrackingFusionParameters parameters)
        FUSION_ERROR_CODE ingestGNSSData(GNSSData _gnss_data)
        POSITIONAL_TRACKING_STATE getPosition(Pose &camera_pose, REFERENCE_FRAME reference_frame, CameraIdentifier uuid, POSITION_TYPE position_type)
        FusedPositionalTrackingStatus getFusedPositionalTrackingStatus()
        POSITIONAL_TRACKING_STATE getCurrentGNSSData(GNSSData &out)
        GNSS_FUSION_STATUS getGeoPose(GeoPose &pose)
        GNSS_FUSION_STATUS Geo2Camera(LatLng &input_, Pose &out)
        GNSS_FUSION_STATUS Camera2Geo(const Pose &input_, GeoPose &out)
        GNSS_FUSION_STATUS getCurrentGNSSCalibrationSTD(float & yaw_std, float3 & position_std)
        Transform getGeoTrackingCalibration()
        void disablePositionalTracking()
        Timestamp getCurrentTimeStamp()
        FUSION_ERROR_CODE ENU2Geo(const ENU &input_, LatLng &out)
        FUSION_ERROR_CODE Geo2ENU(const LatLng &input_, ENU &out)

        FUSION_ERROR_CODE enableSpatialMapping(SpatialMappingFusionParameters parameters)
        void requestSpatialMapAsync()
        FUSION_ERROR_CODE getSpatialMapRequestStatusAsync()
        FUSION_ERROR_CODE retrieveSpatialMapAsync(Mesh &mesh)
        FUSION_ERROR_CODE retrieveSpatialMapAsync(FusedPointCloud &fpc)
        void disableSpatialMapping()

cdef extern from "sl/CameraOne.hpp" namespace "sl":
    cdef cppclass InitParametersOne 'sl::InitParametersOne':
        RESOLUTION camera_resolution
        int camera_fps
        int camera_image_flip
        bool svo_real_time_mode
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        int sdk_gpu_id
        int sdk_verbose
        String sdk_verbose_log_file
        InputType input
        String optional_settings_path
        bool async_grab_camera_recovery
        bool enable_hdr

    cdef cppclass CameraOne 'sl::CameraOne':
        CameraOne()
        ERROR_CODE close()
        ERROR_CODE open(const InitParametersOne init_parameters_one)

        InitParametersOne getInitParameters() const

        bool isOpened() const
        ERROR_CODE read() nogil
        ERROR_CODE grab() nogil
        ERROR_CODE retrieveImage(Mat &mat, VIEW view, MEM type, Resolution resolution) nogil
        ERROR_CODE retrieveTensor(Tensor &dest, const TensorParameters& params, cudaStream_t stream = 0) nogil
        ERROR_CODE getSensorsData(SensorsData &sensor_data, const TIME_REFERENCE reference_time) nogil
        ERROR_CODE getSensorsDataBatch(vector[SensorsData] &sensor_data) nogil

        ERROR_CODE setSVOPosition(int position)
        int getSVOPosition()
        int getSVONumberOfFrames()

        ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, int value)
        ERROR_CODE setCameraSettingsRange "setCameraSettings"(VIDEO_SETTINGS settings, int mini, int maxi)
        ERROR_CODE setCameraSettingsROI "setCameraSettings"(VIDEO_SETTINGS settings, Rect roi, bool reset)

        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, int &settings)
        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, int &aec_min_val, int &aec_max_val)
        ERROR_CODE getCameraSettings(VIDEO_SETTINGS setting, Rect &roi)

        ERROR_CODE getCameraSettingsRange(VIDEO_SETTINGS settings, int &mini, int &maxi)

        bool isCameraSettingSupported(VIDEO_SETTINGS setting)

        float getCurrentFPS()
        Timestamp getTimestamp(TIME_REFERENCE reference_time)
        uint getFrameDroppedCount()
        CameraOneInformation getCameraInformation(Resolution resizer) const

        ERROR_CODE enableRecording(RecordingParameters recording_params)

        RecordingParameters getRecordingParameters()
        StreamingParameters getStreamingParameters()

        RecordingStatus getRecordingStatus()
        void pauseRecording(bool status)
        void disableRecording()
        ERROR_CODE ingestDataIntoSVO(const SVOData& data)
        ERROR_CODE retrieveSVOData(const string &key, map[Timestamp, SVOData] &data, Timestamp ts_begin, Timestamp ts_end)
        vector[string] getSVODataKeys()

        int getSVOPositionAtTimestamp(const Timestamp &timestamp)

        ERROR_CODE enableStreaming(StreamingParameters streaming_parameters)
        void disableStreaming()
        bool isStreamingEnabled()

        @staticmethod
        vector[DeviceProperties] getDeviceList()

        @staticmethod
        vector[StreamingProperties] getStreamingDeviceList()

        @staticmethod
        ERROR_CODE reboot()

# LIDAR definitions
cdef extern from "sl/Lidar.hpp" namespace "sl":

    ctypedef enum LIDAR_MEASURE "sl::LIDAR_MEASURE":
        LIDAR_MEASURE_RANGE "sl::LIDAR_MEASURE::RANGE"
        LIDAR_MEASURE_RANGE2 "sl::LIDAR_MEASURE::RANGE2"
        LIDAR_MEASURE_SIGNAL "sl::LIDAR_MEASURE::SIGNAL"
        LIDAR_MEASURE_SIGNAL2 "sl::LIDAR_MEASURE::SIGNAL2"
        LIDAR_MEASURE_INTENSITY "sl::LIDAR_MEASURE::INTENSITY"
        LIDAR_MEASURE_REFLECTIVITY "sl::LIDAR_MEASURE::REFLECTIVITY"
        LIDAR_MEASURE_REFLECTIVITY2 "sl::LIDAR_MEASURE::REFLECTIVITY2"
        LIDAR_MEASURE_NEAR_IR "sl::LIDAR_MEASURE::NEAR_IR"
        LIDAR_MEASURE_XYZ "sl::LIDAR_MEASURE::XYZ"
        LIDAR_MEASURE_XYZ_RANGE "sl::LIDAR_MEASURE::XYZ_RANGE"
        LIDAR_MEASURE_XYZ_RANGE2 "sl::LIDAR_MEASURE::XYZ_RANGE2"
        LIDAR_MEASURE_XYZ_SIGNAL "sl::LIDAR_MEASURE::XYZ_SIGNAL"
        LIDAR_MEASURE_XYZ_SIGNAL2 "sl::LIDAR_MEASURE::XYZ_SIGNAL2"
        LIDAR_MEASURE_XYZ_INTENSITY "sl::LIDAR_MEASURE::XYZ_INTENSITY"
        LIDAR_MEASURE_XYZ_REFLECTIVITY "sl::LIDAR_MEASURE::XYZ_REFLECTIVITY"
        LIDAR_MEASURE_XYZ_REFLECTIVITY2 "sl::LIDAR_MEASURE::XYZ_REFLECTIVITY2"
        LIDAR_MEASURE_XYZ_NEAR_IR "sl::LIDAR_MEASURE::XYZ_NEAR_IR"
        LIDAR_MEASURE_XYZ_RANGE_VIEW "sl::LIDAR_MEASURE::XYZ_RANGE_VIEW"
        LIDAR_MEASURE_XYZ_RANGE2_VIEW "sl::LIDAR_MEASURE::XYZ_RANGE2_VIEW"
        LIDAR_MEASURE_XYZ_SIGNAL_VIEW "sl::LIDAR_MEASURE::XYZ_SIGNAL_VIEW"
        LIDAR_MEASURE_XYZ_SIGNAL2_VIEW "sl::LIDAR_MEASURE::XYZ_SIGNAL2_VIEW"
        LIDAR_MEASURE_XYZ_INTENSITY_VIEW "sl::LIDAR_MEASURE::XYZ_INTENSITY_VIEW"
        LIDAR_MEASURE_XYZ_REFLECTIVITY_VIEW "sl::LIDAR_MEASURE::XYZ_REFLECTIVITY_VIEW"
        LIDAR_MEASURE_XYZ_REFLECTIVITY2_VIEW "sl::LIDAR_MEASURE::XYZ_REFLECTIVITY2_VIEW"
        LIDAR_MEASURE_XYZ_NEAR_IR_VIEW "sl::LIDAR_MEASURE::XYZ_NEAR_IR_VIEW"
        # RAW staggered data
        LIDAR_MEASURE_RANGE_RAW "sl::LIDAR_MEASURE::RANGE_RAW"
        LIDAR_MEASURE_RANGE2_RAW "sl::LIDAR_MEASURE::RANGE2_RAW"
        LIDAR_MEASURE_SIGNAL_RAW "sl::LIDAR_MEASURE::SIGNAL_RAW"
        LIDAR_MEASURE_SIGNAL2_RAW "sl::LIDAR_MEASURE::SIGNAL2_RAW"
        LIDAR_MEASURE_REFLECTIVITY_RAW "sl::LIDAR_MEASURE::REFLECTIVITY_RAW"
        LIDAR_MEASURE_REFLECTIVITY2_RAW "sl::LIDAR_MEASURE::REFLECTIVITY2_RAW"
        LIDAR_MEASURE_NEAR_IR_RAW "sl::LIDAR_MEASURE::NEAR_IR_RAW"

    ctypedef enum LIDAR_VIEW "sl::LIDAR_VIEW":
        LIDAR_VIEW_INTENSITY "sl::LIDAR_VIEW::INTENSITY"
        LIDAR_VIEW_REFLECTIVITY "sl::LIDAR_VIEW::REFLECTIVITY"
        LIDAR_VIEW_DEPTH "sl::LIDAR_VIEW::DEPTH"
        LIDAR_VIEW_RANGE "sl::LIDAR_VIEW::RANGE"
        LIDAR_VIEW_SIGNAL "sl::LIDAR_VIEW::SIGNAL"
        LIDAR_VIEW_NEAR_IR "sl::LIDAR_VIEW::NEAR_IR"
        LIDAR_VIEW_RANGE2 "sl::LIDAR_VIEW::RANGE2"
        LIDAR_VIEW_SIGNAL2 "sl::LIDAR_VIEW::SIGNAL2"
        LIDAR_VIEW_REFLECTIVITY2 "sl::LIDAR_VIEW::REFLECTIVITY2"

    ctypedef enum VIEW_COLOR_MAP "sl::VIEW_COLOR_MAP":
        VIEW_COLOR_MAP_AUTO "sl::VIEW_COLOR_MAP::AUTO"
        VIEW_COLOR_MAP_GREY "sl::VIEW_COLOR_MAP::GREY"
        VIEW_COLOR_MAP_MAGMA "sl::VIEW_COLOR_MAP::MAGMA"
        VIEW_COLOR_MAP_INFERNO "sl::VIEW_COLOR_MAP::INFERNO"
        VIEW_COLOR_MAP_PLASMA "sl::VIEW_COLOR_MAP::PLASMA"
        VIEW_COLOR_MAP_VIRIDIS "sl::VIEW_COLOR_MAP::VIRIDIS"
        VIEW_COLOR_MAP_SPEZIA "sl::VIEW_COLOR_MAP::SPEZIA"
        VIEW_COLOR_MAP_TURBO "sl::VIEW_COLOR_MAP::TURBO"
        VIEW_COLOR_MAP_JET "sl::VIEW_COLOR_MAP::JET"
        VIEW_COLOR_MAP_BONE "sl::VIEW_COLOR_MAP::BONE"
        VIEW_COLOR_MAP_GNUPLOT2 "sl::VIEW_COLOR_MAP::GNUPLOT2"

    ctypedef enum LIDAR_MODE "sl::LIDAR_MODE":
        LIDAR_MODE_AUTO "sl::LIDAR_MODE::AUTO"
        LIDAR_MODE_512x10 "sl::LIDAR_MODE::MODE_512x10"
        LIDAR_MODE_1024x10 "sl::LIDAR_MODE::MODE_1024x10"
        LIDAR_MODE_2048x10 "sl::LIDAR_MODE::MODE_2048x10"
        LIDAR_MODE_512x20 "sl::LIDAR_MODE::MODE_512x20"
        LIDAR_MODE_1024x20 "sl::LIDAR_MODE::MODE_1024x20"

    ctypedef enum LIDAR_MULTICAST_MODE "sl::LIDAR_MULTICAST_MODE":
        LIDAR_MULTICAST_MODE_OFF "sl::LIDAR_MULTICAST_MODE::OFF"
        LIDAR_MULTICAST_MODE_AUTO "sl::LIDAR_MULTICAST_MODE::AUTO"
        LIDAR_MULTICAST_MODE_ON "sl::LIDAR_MULTICAST_MODE::ON"

    ctypedef enum RECORDING_FORMAT "sl::RECORDING_FORMAT":
        RECORDING_FORMAT_OSF "sl::RECORDING_FORMAT::OSF"
        RECORDING_FORMAT_PCAP "sl::RECORDING_FORMAT::PCAP"
        RECORDING_FORMAT_PCD "sl::RECORDING_FORMAT::PCD"
        RECORDING_FORMAT_AUTO "sl::RECORDING_FORMAT::AUTO"

    cdef cppclass LidarCalibration "sl::LidarCalibration":
        vector[double] beam_altitude_angles
        vector[double] beam_azimuth_angles
        vector[int] pixel_shift_by_row
        double lidar_origin_to_beam_origin_mm
        Matrix4f beam_to_lidar_transform

    cdef cppclass LidarSensorConfiguration "sl::LidarSensorConfiguration":
        int vertical_resolution
        int horizontal_resolution
        float fps
        float vertical_fov
        float horizontal_fov
        LidarCalibration calibration
        Transform imu_to_lidar_transform

    cdef cppclass LidarInformation "sl::LidarInformation":
        String serial_number
        String firmware_version
        String product_line
        String product_part_number
        String build_date
        String image_rev
        String status
        LidarSensorConfiguration sensor_configuration

    cdef cppclass LidarDeviceProperties "sl::LidarDeviceProperties":
        String name
        String ip_address
        int port
        int imu_port
        SENSOR_STATE sensor_state
        LidarInformation lidar_information

    cdef cppclass InitLidarParameters "sl::InitLidarParameters":
        InputType input
        LIDAR_MODE mode
        float depth_minimum_distance
        float depth_maximum_distance
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        bool svo_real_time_mode
        bool auto_recovery_on_config_change
        LIDAR_MULTICAST_MODE multicast_mode
        string multicast_group
        int sdk_verbose
        String sdk_verbose_log_file
        InitLidarParameters()

    cdef cppclass PositionalTrackingLidarParameters "sl::PositionalTrackingLidarParameters":
        Transform initial_world_transform
        bool enable_real_time_computation
        float voxel_size
        PositionalTrackingLidarParameters()

    cdef cppclass RecordingLidarParameters "sl::RecordingLidarParameters":
        string output_filename
        RECORDING_FORMAT format
        RecordingLidarParameters()

    cdef cppclass Lidar "sl::Lidar":
        Lidar()
        
        @staticmethod
        vector[LidarDeviceProperties] getDeviceList()
        
        @staticmethod
        vector[LidarDeviceProperties] getStreamingDeviceList()
        
        ERROR_CODE open(InitLidarParameters params)
        bool isOpened()
        InitLidarParameters getInitParameters()
        LidarInformation getLidarInformation()
        ERROR_CODE read()
        ERROR_CODE grab()
        CUcontext getCUDAContext()
        ERROR_CODE retrieveMeasure(Mat& mat, LIDAR_MEASURE measure, Resolution image_size, MEM mem_type, VIEW_COLOR_MAP color_map, cudaStream_t stream)
        ERROR_CODE retrieveView(Mat& mat, LIDAR_VIEW view, Resolution image_size, VIEW_COLOR_MAP color_map, cudaStream_t stream)
        Timestamp getTimestamp(TIME_REFERENCE reference_time)
        float getCurrentFPS()
        ERROR_CODE getSensorsData(SensorsData& data, TIME_REFERENCE reference_time)
        ERROR_CODE enablePositionalTracking(PositionalTrackingLidarParameters tracking_parameters)
        void disablePositionalTracking()
        POSITIONAL_TRACKING_STATE getPosition(Pose& camera_pose, REFERENCE_FRAME reference_frame)
        ERROR_CODE enableRecording(RecordingLidarParameters recording_parameters)
        void disableRecording()
        int getSVOPosition()
        ERROR_CODE setSVOPosition(int position)
        int getSVONumberOfFrames()
        int getSVOPositionAtTimestamp(Timestamp timestamp)
        void setTimestampBaseOffset(int64_t offset_ns)
        int64_t getTimestampBaseOffset()
        void pauseSVOReading(bool pause)
        void close()

cdef extern from "sl/Sensors.hpp" namespace "sl":

    ctypedef enum SENSORS_TYPE "sl::SENSORS_TYPE":
        SENSORS_TYPE_CAMERA "sl::SENSORS_TYPE::CAMERA"
        SENSORS_TYPE_CAMERA_ONE "sl::SENSORS_TYPE::CAMERA_ONE"
        SENSORS_TYPE_LIDAR "sl::SENSORS_TYPE::LIDAR"

    cdef String toString_sensors_type "sl::toString"(const SENSORS_TYPE &o)

    ctypedef enum SENSORS_REFERENCE_FRAME "sl::SENSORS_REFERENCE_FRAME":
        SENSORS_REFERENCE_FRAME_DEFAULT "sl::SENSORS_REFERENCE_FRAME::DEFAULT"
        SENSORS_REFERENCE_FRAME_SENSOR "sl::SENSORS_REFERENCE_FRAME::SENSOR"
        SENSORS_REFERENCE_FRAME_BASELINK "sl::SENSORS_REFERENCE_FRAME::BASELINK"
        SENSORS_REFERENCE_FRAME_WORLD "sl::SENSORS_REFERENCE_FRAME::WORLD"

    ctypedef enum SENSORS_EXECUTION_MODE "sl::SENSORS_EXECUTION_MODE":
        SENSORS_EXECUTION_MODE_EAGER "sl::SENSORS_EXECUTION_MODE::EAGER"
        SENSORS_EXECUTION_MODE_PIPELINED "sl::SENSORS_EXECUTION_MODE::PIPELINED"

    ctypedef enum SENSORS_SYNC_POLICY "sl::SENSORS_SYNC_POLICY":
        SENSORS_SYNC_POLICY_NONE "sl::SENSORS_SYNC_POLICY::NONE"
        SENSORS_SYNC_POLICY_CLOSEST "sl::SENSORS_SYNC_POLICY::CLOSEST"
        SENSORS_SYNC_POLICY_DROP_STALE "sl::SENSORS_SYNC_POLICY::DROP_STALE"

    ctypedef string SensorID

    cdef cppclass SensorDeviceIdentifier "sl::SensorDeviceIdentifier":
        SensorDeviceIdentifier()
        SensorDeviceIdentifier(string sensor_name, SENSORS_TYPE sensor_type)
        SensorID getID() const
        uint64_t getSerialNumber() const
        string sensor_name
        SENSORS_TYPE type
        bool operator<(const SensorDeviceIdentifier& other) const
        bool operator==(const SensorDeviceIdentifier& other) const

    SensorDeviceIdentifier defaultSensorDeviceIdentifier()

    ctypedef enum SENSORS_ERROR_CODE "sl::SENSORS_ERROR_CODE":
        SENSORS_ERROR_CODE_STALE_DATA "sl::SENSORS_ERROR_CODE::STALE_DATA"
        SENSORS_ERROR_CODE_GNSS_DATA_NEED_FIX "sl::SENSORS_ERROR_CODE::GNSS_DATA_NEED_FIX"
        SENSORS_ERROR_CODE_INVALID_COVARIANCE "sl::SENSORS_ERROR_CODE::INVALID_COVARIANCE"
        SENSORS_ERROR_CODE_INVALID_TIMESTAMP "sl::SENSORS_ERROR_CODE::INVALID_TIMESTAMP"
        SENSORS_ERROR_CODE_CONFIGURATION_FALLBACK "sl::SENSORS_ERROR_CODE::CONFIGURATION_FALLBACK"
        SENSORS_ERROR_CODE_MOTION_SENSORS_DATA_REQUIRED "sl::SENSORS_ERROR_CODE::MOTION_SENSORS_DATA_REQUIRED"
        SENSORS_ERROR_CODE_CORRUPTED_FRAME "sl::SENSORS_ERROR_CODE::CORRUPTED_FRAME"
        SENSORS_ERROR_CODE_SENSOR_REBOOTING "sl::SENSORS_ERROR_CODE::SENSOR_REBOOTING"
        SENSORS_ERROR_CODE_SUCCESS "sl::SENSORS_ERROR_CODE::SUCCESS"
        SENSORS_ERROR_CODE_FAILURE "sl::SENSORS_ERROR_CODE::FAILURE"
        SENSORS_ERROR_CODE_NO_GPU_COMPATIBLE "sl::SENSORS_ERROR_CODE::NO_GPU_COMPATIBLE"
        SENSORS_ERROR_CODE_NOT_ENOUGH_GPU_MEMORY "sl::SENSORS_ERROR_CODE::NOT_ENOUGH_GPU_MEMORY"
        SENSORS_ERROR_CODE_SENSOR_NOT_DETECTED "sl::SENSORS_ERROR_CODE::SENSOR_NOT_DETECTED"
        SENSORS_ERROR_CODE_MOTION_SENSORS_NOT_INITIALIZED "sl::SENSORS_ERROR_CODE::MOTION_SENSORS_NOT_INITIALIZED"
        SENSORS_ERROR_CODE_MOTION_SENSORS_NOT_AVAILABLE "sl::SENSORS_ERROR_CODE::MOTION_SENSORS_NOT_AVAILABLE"
        SENSORS_ERROR_CODE_INVALID_RESOLUTION "sl::SENSORS_ERROR_CODE::INVALID_RESOLUTION"
        SENSORS_ERROR_CODE_LOW_USB_BANDWIDTH "sl::SENSORS_ERROR_CODE::LOW_USB_BANDWIDTH"
        SENSORS_ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE "sl::SENSORS_ERROR_CODE::CALIBRATION_FILE_NOT_AVAILABLE"
        SENSORS_ERROR_CODE_INVALID_CALIBRATION_FILE "sl::SENSORS_ERROR_CODE::INVALID_CALIBRATION_FILE"
        SENSORS_ERROR_CODE_INVALID_SVO_FILE "sl::SENSORS_ERROR_CODE::INVALID_SVO_FILE"
        SENSORS_ERROR_CODE_SVO_RECORDING_ERROR "sl::SENSORS_ERROR_CODE::SVO_RECORDING_ERROR"
        SENSORS_ERROR_CODE_SVO_UNSUPPORTED_COMPRESSION "sl::SENSORS_ERROR_CODE::SVO_UNSUPPORTED_COMPRESSION"
        SENSORS_ERROR_CODE_END_OF_SVOFILE_REACHED "sl::SENSORS_ERROR_CODE::END_OF_SVOFILE_REACHED"
        SENSORS_ERROR_CODE_INVALID_COORDINATE_SYSTEM "sl::SENSORS_ERROR_CODE::INVALID_COORDINATE_SYSTEM"
        SENSORS_ERROR_CODE_INVALID_FIRMWARE "sl::SENSORS_ERROR_CODE::INVALID_FIRMWARE"
        SENSORS_ERROR_CODE_INVALID_FUNCTION_PARAMETERS "sl::SENSORS_ERROR_CODE::INVALID_FUNCTION_PARAMETERS"
        SENSORS_ERROR_CODE_CUDA_ERROR "sl::SENSORS_ERROR_CODE::CUDA_ERROR"
        SENSORS_ERROR_CODE_SENSOR_NOT_INITIALIZED "sl::SENSORS_ERROR_CODE::SENSOR_NOT_INITIALIZED"
        SENSORS_ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE "sl::SENSORS_ERROR_CODE::NVIDIA_DRIVER_OUT_OF_DATE"
        SENSORS_ERROR_CODE_INVALID_FUNCTION_CALL "sl::SENSORS_ERROR_CODE::INVALID_FUNCTION_CALL"
        SENSORS_ERROR_CODE_CORRUPTED_SDK_INSTALLATION "sl::SENSORS_ERROR_CODE::CORRUPTED_SDK_INSTALLATION"
        SENSORS_ERROR_CODE_INCOMPATIBLE_SDK_VERSION "sl::SENSORS_ERROR_CODE::INCOMPATIBLE_SDK_VERSION"
        SENSORS_ERROR_CODE_INVALID_AREA_FILE "sl::SENSORS_ERROR_CODE::INVALID_AREA_FILE"
        SENSORS_ERROR_CODE_INCOMPATIBLE_AREA_FILE "sl::SENSORS_ERROR_CODE::INCOMPATIBLE_AREA_FILE"
        SENSORS_ERROR_CODE_SENSOR_FAILED_TO_SETUP "sl::SENSORS_ERROR_CODE::SENSOR_FAILED_TO_SETUP"
        SENSORS_ERROR_CODE_SENSOR_DETECTION_ISSUE "sl::SENSORS_ERROR_CODE::SENSOR_DETECTION_ISSUE"
        SENSORS_ERROR_CODE_CANNOT_START_SENSOR_STREAM "sl::SENSORS_ERROR_CODE::CANNOT_START_SENSOR_STREAM"
        SENSORS_ERROR_CODE_NO_GPU_DETECTED "sl::SENSORS_ERROR_CODE::NO_GPU_DETECTED"
        SENSORS_ERROR_CODE_PLANE_NOT_FOUND "sl::SENSORS_ERROR_CODE::PLANE_NOT_FOUND"
        SENSORS_ERROR_CODE_MODULE_NOT_COMPATIBLE_WITH_SENSOR "sl::SENSORS_ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_SENSOR"
        SENSORS_ERROR_CODE_MOTION_SENSORS_REQUIRED "sl::SENSORS_ERROR_CODE::MOTION_SENSORS_REQUIRED"
        SENSORS_ERROR_CODE_MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION "sl::SENSORS_ERROR_CODE::MODULE_NOT_COMPATIBLE_WITH_CUDA_VERSION"
        SENSORS_ERROR_CODE_LAST "sl::SENSORS_ERROR_CODE::LAST"

    cdef String toString_sensors "sl::toString"(const SENSORS_ERROR_CODE& errorCode)

    cdef cppclass InitSensorsParameters 'sl::InitSensorsParameters':
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        bool output_performance_metrics
        int verbose
        int sdk_gpu_id
        CUcontext sdk_cuda_ctx
        Resolution maximum_working_resolution
        SENSORS_EXECUTION_MODE execution_mode
        SENSORS_SYNC_POLICY sync_policy
        Timestamp sync_tolerance

    cdef cppclass RecordingSensorsParameters 'sl::RecordingSensorsParameters':
        set[SensorDeviceIdentifier] sensors_ids
        unordered_map[SensorDeviceIdentifier, String] video_filenames
        SVO_COMPRESSION_MODE compression_mode
        unsigned int bitrate
        unsigned int target_framerate
        bool transcode_streaming_input

    cdef cppclass StreamingSensorsParameters 'sl::StreamingSensorsParameters':
        set[SensorDeviceIdentifier] sensors_ids
        STREAMING_CODEC codec
        unsigned short port
        unsigned int bitrate
        int gop_size
        bool adaptative_bitrate
        unsigned short chunk_size
        unsigned int target_framerate

    cdef cppclass ObjectDetectionSensorsParameters 'sl::ObjectDetectionSensorsParameters':
        unsigned int instance_module_id
        set[SensorDeviceIdentifier] sensors_ids
        String fused_objects_group_name
        bool enable_tracking
        bool enable_segmentation
        OBJECT_DETECTION_MODEL detection_model
        String custom_onnx_file
        Resolution custom_onnx_dynamic_input_shape
        float max_range
        OBJECT_FILTERING_MODE filtering_mode
        float prediction_timeout_s

    # Body Tracking Parameters for Sensors API
    cdef cppclass BodyTrackingSensorsParameters 'sl::BodyTrackingSensorsParameters':
        unsigned int instance_module_id
        set[SensorDeviceIdentifier] sensors_ids
        bool enable_tracking
        bool enable_segmentation
        BODY_TRACKING_MODEL detection_model
        bool enable_body_fitting
        BODY_FORMAT body_format
        BODY_KEYPOINTS_SELECTION body_selection
        float max_range
        float prediction_timeout_s
        bool allow_reduced_precision_inference

    cdef cppclass BodyTrackingSensorsRuntimeParameters 'sl::BodyTrackingSensorsRuntimeParameters':
        float detection_confidence_threshold
        int minimum_keypoints_threshold
        float skeleton_smoothing

    # Positional Tracking Parameters for Sensors API
    cdef cppclass PositionalTrackingSensorsParameters 'sl::PositionalTrackingSensorsParameters':
        bool enable_area_memory
        bool enable_pose_smoothing
        bool set_floor_as_origin
        String area_file_path
        bool enable_imu_fusion
        bool set_as_static
        float depth_min_range
        bool set_gravity_as_origin
        POSITIONAL_TRACKING_MODE mode

    # BatchedData is a type alias for unordered_map<SensorDeviceIdentifier, T>
    # We use ctypedef alias to match the C++ type alias
    ctypedef unordered_map[SensorDeviceIdentifier, ERROR_CODE] BatchedDataErrorCode "sl::BatchedData<sl::ERROR_CODE>"
    ctypedef unordered_map[SensorDeviceIdentifier, Mat] BatchedDataMat "sl::BatchedData<sl::Mat>"
    ctypedef unordered_map[SensorDeviceIdentifier, SensorsData] BatchedDataSensorsData "sl::BatchedData<sl::SensorsData>"
    ctypedef unordered_map[SensorDeviceIdentifier, float] BatchedDataFloat "sl::BatchedData<float>"
    ctypedef unordered_map[SensorDeviceIdentifier, unsigned int] BatchedDataUInt "sl::BatchedData<unsigned int>"
    ctypedef unordered_map[SensorDeviceIdentifier, Timestamp] BatchedDataTimestamp "sl::BatchedData<sl::Timestamp>"
    ctypedef unordered_map[SensorDeviceIdentifier, HealthStatus] BatchedDataHealthStatus "sl::BatchedData<sl::HealthStatus>"
    ctypedef unordered_map[SensorDeviceIdentifier, int] BatchedDataInt "sl::BatchedData<int>"
    ctypedef unordered_map[SensorDeviceIdentifier, pair[Pose, POSITIONAL_TRACKING_STATE]] BatchedDataPose "sl::BatchedData<std::pair<sl::Pose, sl::POSITIONAL_TRACKING_STATE>>"

    # Runtime Sensors Parameters
    cdef cppclass RuntimeSensorsParameters 'sl::RuntimeSensorsParameters':
        BatchedDataInt depth_confidence_threshold
        SENSORS_REFERENCE_FRAME reference_frame

    cdef cppclass SensorList 'sl::SensorList':
        vector[DeviceProperties] cameras
        vector[DeviceProperties] cameras_one
        vector[LidarDeviceProperties] lidars

    cdef cppclass StreamingSensorList 'sl::StreamingSensorList':
        vector[StreamingProperties] cameras
        vector[StreamingProperties] cameras_one
        vector[LidarDeviceProperties] lidars

    cdef cppclass Sensors 'sl::Sensors':
        Sensors() except +
        @staticmethod
        SensorList getSensorList() nogil
        @staticmethod
        StreamingSensorList getStreamingSensorList() nogil
        SENSORS_ERROR_CODE init(const InitSensorsParameters& init_parameters) nogil
        SENSORS_ERROR_CODE addCamera "add"(const InitParameters& init_parameters, SensorDeviceIdentifier& camera_identifier_out) nogil
        SENSORS_ERROR_CODE addCameraOne "add"(const InitParametersOne& init_parameters, SensorDeviceIdentifier& camera_identifier_out) nogil
        SENSORS_ERROR_CODE addLidar "add"(const InitLidarParameters& init_parameters, SensorDeviceIdentifier& lidar_identifier_out) nogil
        SENSORS_ERROR_CODE remove(SensorDeviceIdentifier& unique_identifier) nogil
        SENSORS_ERROR_CODE setSensorPose(const Transform& pose, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getSensorPose(Transform& pose, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getSensorsID(set[SensorDeviceIdentifier]& sensors_identifier) nogil
        SENSORS_ERROR_CODE read() nogil
        SENSORS_ERROR_CODE process() nogil
        # Error codes
        SENSORS_ERROR_CODE getErrorCodes(BatchedDataErrorCode& errs) nogil
        SENSORS_ERROR_CODE getProcessErrorCodes(BatchedDataErrorCode& errs) nogil
        SENSORS_ERROR_CODE setRuntimeSensorsParameters(const RuntimeSensorsParameters& params) nogil
        SENSORS_ERROR_CODE setSVOPositionAtTimestamp(const Timestamp& timestamp) nogil
        # Image retrieval
        SENSORS_ERROR_CODE retrieveSingleImage "retrieveImage"(Mat& mat, const VIEW view, const MEM mem_type, const Resolution image_size, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE retrieveBatchedImages "retrieveImage"(BatchedDataMat& mat_list, const VIEW view, const MEM mem_type, const Resolution image_size, set[SensorDeviceIdentifier] sensors_ids) nogil
        # Measure retrieval
        SENSORS_ERROR_CODE retrieveSingleMeasure "retrieveMeasure"(Mat& mat, const MEASURE measure, const MEM mem_type, const Resolution image_size, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE retrieveBatchedMeasures "retrieveMeasure"(BatchedDataMat& mat_list, const MEASURE measure, const MEM mem_type, const Resolution image_size, set[SensorDeviceIdentifier] sensors_ids) nogil
        SENSORS_ERROR_CODE retrieveBatchedLidarMeasures "retrieveMeasure"(BatchedDataMat& mat_list, const LIDAR_MEASURE measure, const MEM mem_type, const Resolution image_size, set[SensorDeviceIdentifier] sensors_ids) nogil
        # Motion sensors data
        SENSORS_ERROR_CODE getMotionSensorsData(SensorsData& data, TIME_REFERENCE reference_time, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getBatchedMotionSensorsData "getMotionSensorsData"(BatchedDataSensorsData& data, TIME_REFERENCE reference_time, set[SensorDeviceIdentifier] sensors_ids) nogil
        void updateSelfCalibration(SensorDeviceIdentifier unique_identifier) nogil
        # FPS and frame count
        SENSORS_ERROR_CODE getCurrentFPS(float& fps, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getBatchedCurrentFPS "getCurrentFPS"(BatchedDataFloat& data, set[SensorDeviceIdentifier] sensors_ids) nogil
        SENSORS_ERROR_CODE getFrameDroppedCount(unsigned int& count, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getBatchedFrameDroppedCount "getFrameDroppedCount"(BatchedDataUInt& data, set[SensorDeviceIdentifier] sensors_ids) nogil
        # Timestamp and health
        SENSORS_ERROR_CODE getTimestamp(BatchedDataTimestamp& data, TIME_REFERENCE reference_time, set[SensorDeviceIdentifier] sensors_ids) nogil
        SENSORS_ERROR_CODE getHealthStatus(BatchedDataHealthStatus& data, set[SensorDeviceIdentifier] sensors_ids) nogil
        CameraInformation getCameraInformation(Resolution image_size, SensorDeviceIdentifier unique_identifier) nogil
        # Region of Interest
        SENSORS_ERROR_CODE setRegionOfInterest(Mat& roi_mask, unordered_set[MODULE] module, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getRegionOfInterest(Mat& roi_mask, Resolution image_size, MODULE module, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE startRegionOfInterestAutoDetection(RegionOfInterestParameters roi_param, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getRegionOfInterestAutoDetectionStatus(REGION_OF_INTEREST_AUTO_DETECTION_STATE& status, SensorDeviceIdentifier unique_identifier) nogil
        # Camera Settings
        SENSORS_ERROR_CODE getCameraSettings(VIDEO_SETTINGS settings, int& setting, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE getCameraSettingsMinMax "getCameraSettings"(VIDEO_SETTINGS settings, int& min_val, int& max_val, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE getCameraSettingsRect "getCameraSettings"(VIDEO_SETTINGS settings, Rect& roi, SIDE side, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE getCameraSettingsRange(VIDEO_SETTINGS settings, int& min_val, int& max_val, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE isCameraSettingSupported(VIDEO_SETTINGS setting, bool& is_supported, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE setCameraSettings(VIDEO_SETTINGS settings, int value, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE setCameraSettingsMinMax "setCameraSettings"(VIDEO_SETTINGS settings, int min_val, int max_val, SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE setCameraSettingsRect "setCameraSettings"(VIDEO_SETTINGS settings, Rect roi, SIDE side, bool reset, SensorDeviceIdentifier sensors_id) nogil
        # Recording
        SENSORS_ERROR_CODE enableRecording(const RecordingSensorsParameters& recording_parameters) nogil
        RecordingStatus getRecordingStatus(const SensorDeviceIdentifier& sensors_id) nogil
        void pauseRecording(bool status, const SensorDeviceIdentifier& sensors_id) nogil
        void disableRecording(const set[SensorDeviceIdentifier]& sensors_ids) nogil
        # Streaming
        SENSORS_ERROR_CODE enableStreaming(const StreamingSensorsParameters& streaming_parameters) nogil
        void disableStreaming(const set[SensorDeviceIdentifier]& sensors_ids) nogil
        # SVO
        SENSORS_ERROR_CODE getSVOPosition(int& svo_position, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getSVOPositionAtTimestamp(int& svo_position, const Timestamp& timestamp, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE setSVOPosition(int svo_position, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getSVONumberOfFrames(int& number_of_frames, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE syncSVO(Timestamp t) nogil
        SENSORS_ERROR_CODE ingestDataIntoSVO(const SVOData& data, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE retrieveSVOData(const string& key, map[Timestamp, SVOData]& data, Timestamp ts_begin, Timestamp ts_end, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getSVODataKeys(vector[string]& keys, SensorDeviceIdentifier unique_identifier) nogil
        # Object Detection
        SENSORS_ERROR_CODE enableObjectDetection(const ObjectDetectionSensorsParameters& param) nogil
        SENSORS_ERROR_CODE setObjectDetectionRuntimeParameters(const ObjectDetectionRuntimeParameters& params, const unsigned int instance_id, set[SensorDeviceIdentifier] cam_identifiers) nogil
        SENSORS_ERROR_CODE setCustomObjectDetectionRuntimeParameters(const CustomObjectDetectionRuntimeParameters& params, const unsigned int instance_id, set[SensorDeviceIdentifier] cam_identifiers) nogil
        SENSORS_ERROR_CODE ingestCustomBoxObjects(const vector[CustomBoxObjectData]& objects_in, const unsigned int instance_id, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE ingestCustomMaskObjects(const vector[CustomMaskObjectData]& objects_in, const unsigned int instance_id, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE retrieveObjects(Objects& objects, SensorDeviceIdentifier sensors_id, unsigned int instance_module_id) nogil
        SENSORS_ERROR_CODE disableObjectDetection(unsigned int instance_module_id) nogil
        # Body Tracking
        SENSORS_ERROR_CODE enableBodyTracking(const BodyTrackingSensorsParameters& param) nogil
        SENSORS_ERROR_CODE setBodyTrackingRuntimeParameters(const BodyTrackingSensorsRuntimeParameters& params, const unsigned int instance_id, set[SensorDeviceIdentifier] cam_identifiers) nogil
        SENSORS_ERROR_CODE retrieveBodies(Bodies& bodies, SensorDeviceIdentifier sensors_id, unsigned int instance_module_id) nogil
        SENSORS_ERROR_CODE disableBodyTracking(unsigned int instance_module_id) nogil
        # Positional Tracking
        SENSORS_ERROR_CODE enablePositionalTracking(const PositionalTrackingSensorsParameters& tracking_parameters, set[SensorDeviceIdentifier] sensors_id) nogil
        POSITIONAL_TRACKING_STATE getPosition(Pose& camera_pose, SENSORS_REFERENCE_FRAME reference_frame, SensorDeviceIdentifier unique_identifier) nogil
        SENSORS_ERROR_CODE getBatchedPosition "getPosition"(BatchedDataPose& camera_poses, SENSORS_REFERENCE_FRAME reference_frame, set[SensorDeviceIdentifier] sensors_id) nogil
        SENSORS_ERROR_CODE resetPositionalTracking(const Transform& path, set[SensorDeviceIdentifier] sensors_ids) nogil
        PositionalTrackingStatus getPositionalTrackingStatus(SensorDeviceIdentifier sensors_id) nogil
        SENSORS_ERROR_CODE saveAreaMap(String area_file_path, SensorDeviceIdentifier sensors_id) nogil
        AREA_EXPORTING_STATE getAreaExportState(SensorDeviceIdentifier sensors_id) nogil
        void disablePositionalTracking(String area_file_path) nogil
        bool isPositionalTrackingEnabled() nogil
        SENSORS_ERROR_CODE close() nogil
