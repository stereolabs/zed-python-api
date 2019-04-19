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

# File containing the Cython declarations to use the sl functions.

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.string cimport const_char

cdef extern from "Utils.cpp" namespace "sl":
    string to_str(String sl_str)

cdef extern from "sl/types.hpp" namespace "sl":

    ctypedef unsigned long long timeStamp

    ctypedef enum ERROR_CODE:
        SUCCESS,
        ERROR_CODE_FAILURE,
        ERROR_CODE_NO_GPU_COMPATIBLE,
        ERROR_CODE_NOT_ENOUGH_GPUMEM,
        ERROR_CODE_CAMERA_NOT_DETECTED,
        ERROR_CODE_SENSOR_NOT_DETECTED,
        ERROR_CODE_INVALID_RESOLUTION,
        ERROR_CODE_LOW_USB_BANDWIDTH,
        ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE,
        ERROR_CODE_INVALID_CALIBRATION_FILE,
        ERROR_CODE_INVALID_SVO_FILE,
        ERROR_CODE_SVO_RECORDING_ERROR,
        ERROR_CODE_SVO_UNSUPPORTED_COMPRESSION,
        ERROR_CODE_INVALID_COORDINATE_SYSTEM,
        ERROR_CODE_INVALID_FIRMWARE,
        ERROR_CODE_INVALID_FUNCTION_PARAMETERS,
        ERROR_CODE_NOT_A_NEW_FRAME,
        ERROR_CODE_CUDA_ERROR,
        ERROR_CODE_CAMERA_NOT_INITIALIZED,
        ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE,
        ERROR_CODE_INVALID_FUNCTION_CALL,
        ERROR_CODE_CORRUPTED_SDK_INSTALLATION,
        ERROR_CODE_INCOMPATIBLE_SDK_VERSION,
        ERROR_CODE_INVALID_AREA_FILE,
        ERROR_CODE_INCOMPATIBLE_AREA_FILE,
        ERROR_CODE_CAMERA_FAILED_TO_SETUP,
        ERROR_CODE_CAMERA_DETECTION_ISSUE,
        ERROR_CODE_CAMERA_ALREADY_IN_USE,
        ERROR_CODE_NO_GPU_DETECTED,
        ERROR_CODE_PLANE_NOT_FOUND,
        ERROR_CODE_LAST

    String toString(ERROR_CODE o)

    void sleep_ms(int time)

    ctypedef enum MODEL:
        MODEL_ZED,
        MODEL_ZED_M,
        MODEL_LAST

    String model2str(MODEL model)
    String toString(MODEL o)

    ctypedef enum CAMERA_STATE:
        CAMERA_STATE_AVAILABLE,
        CAMERA_STATE_NOT_AVAILABLE,
        CAMERA_STATE_LAST

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


    cdef cppclass Matrix3f 'sl::Matrix3f':
        int nbElem
        float r00
        float r01
        float r02
        float r10
        float r11
        float r12
        float r20
        float r21
        float r22
        float r[]
        Matrix3f()
        Matrix3f(float data[])
        Matrix3f(const Matrix3f &mat)
        Matrix3f operator*(const Matrix3f &mat) const
        Matrix3f operator*(const double &scalar) const
        bool operator==(const Matrix3f &mat) const
        bool operator!=(const Matrix3f &mat) const
        void inverse()

        @staticmethod
        Matrix3f inverse(const Matrix3f &rotation)

        void transpose()

        @staticmethod
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
        int nbElem
        float r00
        float r01
        float r02
        float tx
        float r10
        float r11
        float r12
        float ty
        float r20
        float r21
        float r22
        float tz
        float m30
        float m31
        float m32
        float m33

        float m[]
        Matrix4f()
        Matrix4f(float data[])
        Matrix4f(const Matrix4f &mat)
        Matrix4f operator*(const Matrix4f &mat) const
        Matrix4f operator*(const double &scalar) const
        bool operator==(const Matrix4f  &mat) const
        bool operator!=(const Matrix4f &mat) const
        ERROR_CODE inverse()

        @staticmethod
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

    ctypedef enum UNIT:
        UNIT_MILLIMETER
        UNIT_CENTIMETER
        UNIT_METER
        UNIT_INCH
        UNIT_FOOT
        UNIT_LAST

    String toString(UNIT o)

    ctypedef enum COORDINATE_SYSTEM:
        COORDINATE_SYSTEM_IMAGE
        COORDINATE_SYSTEM_LEFT_HANDED_Y_UP
        COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
        COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP
        COORDINATE_SYSTEM_LEFT_HANDED_Z_UP
        COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD
        COORDINATE_SYSTEM_LAST

    String toString(COORDINATE_SYSTEM o)


cdef extern from "sl/defines.hpp" namespace "sl":

    ctypedef enum RESOLUTION:
        RESOLUTION_HD2K
        RESOLUTION_HD1080
        RESOLUTION_HD720
        RESOLUTION_VGA
        RESOLUTION_LAST

    String toString(RESOLUTION o)

    ctypedef enum CAMERA_SETTINGS:
        CAMERA_SETTINGS_BRIGHTNESS
        CAMERA_SETTINGS_CONTRAST
        CAMERA_SETTINGS_HUE
        CAMERA_SETTINGS_SATURATION
        CAMERA_SETTINGS_GAIN
        CAMERA_SETTINGS_EXPOSURE
        CAMERA_SETTINGS_WHITEBALANCE
        CAMERA_SETTINGS_AUTO_WHITEBALANCE
        CAMERA_SETTINGS_LED_STATUS
        CAMERA_SETTINGS_LAST

    String toString(CAMERA_SETTINGS o)

    ctypedef enum SELF_CALIBRATION_STATE:
        SELF_CALIBRATION_STATE_NOT_STARTED
        SELF_CALIBRATION_STATE_RUNNING
        SELF_CALIBRATION_STATE_FAILED
        SELF_CALIBRATION_STATE_SUCCESS
        SELF_CALIBRATION_STATE_LAST

    String toString(SELF_CALIBRATION_STATE o)

    ctypedef enum DEPTH_MODE:
        DEPTH_MODE_NONE
        DEPTH_MODE_PERFORMANCE
        DEPTH_MODE_MEDIUM
        DEPTH_MODE_QUALITY
        DEPTH_MODE_ULTRA
        DEPTH_MODE_LAST

    String toString(DEPTH_MODE o)

    ctypedef enum SENSING_MODE:
        SENSING_MODE_STANDARD
        SENSING_MODE_FILL
        SENSING_MODE_LAST

    String toString(SENSING_MODE o)

    ctypedef enum MEASURE:
        MEASURE_DISPARITY
        MEASURE_DEPTH
        MEASURE_CONFIDENCE
        MEASURE_XYZ
        MEASURE_XYZRGBA
        MEASURE_XYZBGRA
        MEASURE_XYZARGB
        MEASURE_XYZABGR
        MEASURE_NORMALS
        MEASURE_DISPARITY_RIGHT
        MEASURE_DEPTH_RIGHT
        MEASURE_XYZ_RIGHT
        MEASURE_XYZRGBA_RIGHT
        MEASURE_XYZBGRA_RIGHT
        MEASURE_XYZARGB_RIGHT
        MEASURE_XYZABGR_RIGHT
        MEASURE_NORMALS_RIGHT
        MEASURE_LAST

    String toString(MEASURE o)

    ctypedef enum VIEW:
        VIEW_LEFT
        VIEW_RIGHT
        VIEW_LEFT_GRAY
        VIEW_RIGHT_GRAY
        VIEW_LEFT_UNRECTIFIED
        VIEW_RIGHT_UNRECTIFIED
        VIEW_LEFT_UNRECTIFIED_GRAY
        VIEW_RIGHT_UNRECTIFIED_GRAY
        VIEW_SIDE_BY_SIDE
        VIEW_DEPTH
        VIEW_CONFIDENCE
        VIEW_NORMALS
        VIEW_DEPTH_RIGHT
        VIEW_NORMALS_RIGHT
        VIEW_LAST

    String toString(VIEW o)

    ctypedef enum TIME_REFERENCE:
        TIME_REFERENCE_IMAGE
        TIME_REFERENCE_CURRENT
        TIME_REFERENCE_LAST

    String toString(TIME_REFERENCE o)

    ctypedef enum DEPTH_FORMAT:
        DEPTH_FORMAT_PNG
        DEPTH_FORMAT_PFM
        DEPTH_FORMAT_PGM
        DEPTH_FORMAT_LAST

    String toString(DEPTH_FORMAT o)

    ctypedef enum POINT_CLOUD_FORMAT:
        POINT_CLOUD_FORMAT_XYZ_ASCII
        POINT_CLOUD_FORMAT_PCD_ASCII
        POINT_CLOUD_FORMAT_PLY_ASCII
        POINT_CLOUD_FORMAT_VTK_ASCII
        POINT_CLOUD_FORMAT_LAST

    String toString(POINT_CLOUD_FORMAT o)

    ctypedef enum TRACKING_STATE:
        TRACKING_STATE_SEARCHING
        TRACKING_STATE_OK
        TRACKING_STATE_OFF
        TRACKING_STATE_FPS_TOO_LOW
        TRACKING_STATE_LAST

    String toString(TRACKING_STATE o)

    ctypedef enum AREA_EXPORT_STATE:
        AREA_EXPORT_STATE_SUCCESS
        AREA_EXPORT_STATE_RUNNING
        AREA_EXPORT_STATE_NOT_STARTED
        AREA_EXPORT_STATE_FILE_EMPTY
        AREA_EXPORT_STATE_FILE_ERROR
        AREA_EXPORT_STATE_SPATIAL_MEMORY_DISABLED
        AREA_EXPORT_STATE_LAST

    String toString(AREA_EXPORT_STATE o)

    ctypedef enum REFERENCE_FRAME:
        REFERENCE_FRAME_WORLD
        REFERENCE_FRAME_CAMERA
        REFERENCE_FRAME_LAST

    String toString(REFERENCE_FRAME o)

    ctypedef enum SPATIAL_MAPPING_STATE:
        SPATIAL_MAPPING_STATE_INITIALIZING
        SPATIAL_MAPPING_STATE_OK
        SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY
        SPATIAL_MAPPING_STATE_NOT_ENABLED
        SPATIAL_MAPPING_STATE_FPS_TOO_LOW
        SPATIAL_MAPPING_STATE_LAST

    String toString(SPATIAL_MAPPING_STATE o)

    ctypedef enum SVO_COMPRESSION_MODE:
        SVO_COMPRESSION_MODE_RAW
        SVO_COMPRESSION_MODE_LOSSLESS
        SVO_COMPRESSION_MODE_LOSSY
        SVO_COMPRESSION_MODE_AVCHD
        SVO_COMPRESSION_MODE_HEVC
        SVO_COMPRESSION_MODE_LAST

    String toString(SVO_COMPRESSION_MODE o)

    cdef struct RecordingState:
        bool status
        double current_compression_time
        double current_compression_ratio
        double average_compression_time
        double average_compression_ratio


    @staticmethod
    cdef vector[pair[int, int]] cameraResolution

    @staticmethod
    cdef const_char* resolution2str(RESOLUTION res)

    @staticmethod
    cdef const_char* statusCode2str(SELF_CALIBRATION_STATE state)
    @staticmethod
    cdef DEPTH_MODE str2mode(const_char* mode)

    @staticmethod
    cdef const_char* depthMode2str(DEPTH_MODE mode)

    @staticmethod
    cdef const_char* sensingMode2str(SENSING_MODE mode)

    @staticmethod
    cdef const_char* unit2str(UNIT unit)

    @staticmethod
    cdef UNIT str2unit(const_char* unit)

    @staticmethod
    cdef const_char* trackingState2str(TRACKING_STATE state)

    @staticmethod
    cdef const_char* spatialMappingState2str(SPATIAL_MAPPING_STATE state)


cdef extern from "sl/Core.hpp" namespace "sl":

    timeStamp getCurrentTimeStamp()

    cdef struct Resolution:
        size_t width
        size_t height


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
        Vector3[float] R
        Vector3[float] T
        CameraParameters left_cam
        CameraParameters right_cam


    cdef struct CameraInformation:
        CalibrationParameters calibration_parameters
        CalibrationParameters calibration_parameters_raw
        Transform camera_imu_transform
        unsigned int serial_number
        unsigned int firmware_version
        MODEL camera_model

    cdef enum MEM:
        MEM_CPU
        MEM_GPU

    MEM operator|(MEM a, MEM b)


    cdef enum COPY_TYPE:
        COPY_TYPE_CPU_CPU
        COPY_TYPE_CPU_GPU
        COPY_TYPE_GPU_GPU
        COPY_TYPE_GPU_CPU


    cdef enum MAT_TYPE:
        MAT_TYPE_32F_C1
        MAT_TYPE_32F_C2
        MAT_TYPE_32F_C3
        MAT_TYPE_32F_C4
        MAT_TYPE_8U_C1
        MAT_TYPE_8U_C2
        MAT_TYPE_8U_C3
        MAT_TYPE_8U_C4


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
        ERROR_CODE write(const char* filePath)
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
        Translation(float t1, float t2, float t3)
        Translation operator*(const Orientation &mat) const
        void normalize()

        @staticmethod
        Translation normalize(const Translation &tr)
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
        Transform(const Transform &motion)
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

cdef extern from "Utils.cpp" namespace "sl":

    Mat matResolution(Resolution resolution, MAT_TYPE mat_type, uchar1 *ptr_cpu, size_t step_cpu,
                      uchar1 *ptr_gpu, size_t step_gpu)

    ERROR_CODE setToUchar1(Mat &mat, uchar1 value, MEM memory_type)
    ERROR_CODE setToUchar2(Mat &mat, uchar2 value, MEM memory_type)
    ERROR_CODE setToUchar3(Mat &mat, uchar3 value, MEM memory_type)
    ERROR_CODE setToUchar4(Mat &mat, uchar4 value, MEM memory_type)

    ERROR_CODE setToFloat1(Mat &mat, float1 value, MEM memory_type)
    ERROR_CODE setToFloat2(Mat &mat, float2 value, MEM memory_type)
    ERROR_CODE setToFloat3(Mat &mat, float3 value, MEM memory_type)
    ERROR_CODE setToFloat4(Mat &mat, float4 value, MEM memory_type)

    ERROR_CODE setValueUchar1(Mat &mat, size_t x, size_t y, uchar1 value, MEM memory_type)
    ERROR_CODE setValueUchar2(Mat &mat, size_t x, size_t y, uchar2 value, MEM memory_type)
    ERROR_CODE setValueUchar3(Mat &mat, size_t x, size_t y, uchar3 value, MEM memory_type)
    ERROR_CODE setValueUchar4(Mat &mat, size_t x, size_t y, uchar4 value, MEM memory_type)

    ERROR_CODE setValueFloat1(Mat &mat, size_t x, size_t y, float1 value, MEM memory_type)
    ERROR_CODE setValueFloat2(Mat &mat, size_t x, size_t y, float2 value, MEM memory_type)
    ERROR_CODE setValueFloat3(Mat &mat, size_t x, size_t y, float3 value, MEM memory_type)
    ERROR_CODE setValueFloat4(Mat &mat, size_t x, size_t y, float4 value, MEM memory_type)

    ERROR_CODE getValueUchar1(Mat &mat, size_t x, size_t y, uchar1 *value, MEM memory_type)
    ERROR_CODE getValueUchar2(Mat &mat, size_t x, size_t y, Vector2[uchar1] *value, MEM memory_type)
    ERROR_CODE getValueUchar3(Mat &mat, size_t x, size_t y, Vector3[uchar1] *value, MEM memory_type)
    ERROR_CODE getValueUchar4(Mat &mat, size_t x, size_t y, Vector4[uchar1] *value, MEM memory_type)

    ERROR_CODE getValueFloat1(Mat &mat, size_t x, size_t y, float1 *value, MEM memory_type)
    ERROR_CODE getValueFloat2(Mat &mat, size_t x, size_t y, Vector2[float1] *value, MEM memory_type)
    ERROR_CODE getValueFloat3(Mat &mat, size_t x, size_t y, Vector3[float1] *value, MEM memory_type)
    ERROR_CODE getValueFloat4(Mat &mat, size_t x, size_t y, Vector4[float1] *value, MEM memory_type)

    uchar1 *getPointerUchar1(Mat &mat, MEM memory_type)
    uchar2 *getPointerUchar2(Mat &mat, MEM memory_type)
    uchar3 *getPointerUchar3(Mat &mat, MEM memory_type)
    uchar4 *getPointerUchar4(Mat &mat, MEM memory_type)

    float1 *getPointerFloat1(Mat &mat, MEM memory_type)
    float2 *getPointerFloat2(Mat &mat, MEM memory_type)
    float3 *getPointerFloat3(Mat &mat, MEM memory_type)
    float4 *getPointerFloat4(Mat &mat, MEM memory_type)


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

    ctypedef enum PLANE_TYPE:
        PLANE_TYPE_HORIZONTAL
        PLANE_TYPE_VERTICAL
        PLANE_TYPE_UNKNOWN
        PLANE_TYPE_LAST

    cdef cppclass MeshFilterParameters 'sl::MeshFilterParameters':
        MeshFilterParameters(MESH_FILTER filtering_)
        void set(MESH_FILTER filtering_)
        bool save(String filename)
        bool load(String filename)

    cdef cppclass Texture 'sl::Texture':
        Texture()
        String name
        Mat data
        unsigned int indice_gl
        void clear()

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
        Texture texture
        size_t getNumberOfTriangles()
        void mergeChunks(int faces_per_chunk)
        Vector3[float] getGravityEstimate()
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



cdef extern from 'cuda.h' :
    cdef struct CUctx_st :
        pass
    ctypedef CUctx_st* CUcontext

cdef extern from 'sl/Camera.hpp' namespace 'sl':

    ctypedef enum SPATIAL_MAP_TYPE 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE':
        SPATIAL_MAP_TYPE_MESH 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::SPATIAL_MAP_TYPE_MESH'
        SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD 'sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD'

    ctypedef enum MAPPING_RESOLUTION 'sl::SpatialMappingParameters::MAPPING_RESOLUTION':
        MAPPING_RESOLUTION_HIGH 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::MAPPING_RESOLUTION_HIGH'
        MAPPING_RESOLUTION_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::MAPPING_RESOLUTION_MEDIUM'
        MAPPING_RESOLUTION_LOW 'sl::SpatialMappingParameters::MAPPING_RESOLUTION::MAPPING_RESOLUTION_LOW'


    ctypedef enum MAPPING_RANGE 'sl::SpatialMappingParameters::MAPPING_RANGE':
        MAPPING_RANGE_NEAR 'sl::SpatialMappingParameters::MAPPING_RANGE::MAPPING_RANGE_NEAR'
        MAPPING_RANGE_MEDIUM 'sl::SpatialMappingParameters::MAPPING_RANGE::MAPPING_RANGE_MEDIUM'
        MAPPING_RANGE_FAR 'sl::SpatialMappingParameters::MAPPING_RANGE::MAPPING_RANGE_FAR'

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
        int camera_linux_id
        String svo_input_filename
        bool svo_real_time_mode
        UNIT coordinate_units
        COORDINATE_SYSTEM coordinate_system
        DEPTH_MODE depth_mode
        float depth_minimum_distance
        int camera_image_flip
        bool enable_right_side_measure
        bool camera_disable_self_calib
        int camera_buffer_count_linux
        bool sdk_verbose
        int sdk_gpu_id
        bool depth_stabilization

        String sdk_verbose_log_file

        CUcontext sdk_cuda_ctx
        InputType input
        String optional_settings_path

        InitParameters(RESOLUTION camera_resolution,
                       int camera_fps,
                       int camera_linux_id,
                       String svo_input_filename,
                       bool svo_real_time_mode,
                       DEPTH_MODE depth_mode,
                       UNIT coordinate_units,
                       COORDINATE_SYSTEM coordinate_system,
                       bool sdk_verbose,
                       int sdk_gpu_id,
                       float depth_minimum_distance,
                       bool camera_disable_self_calib,
                       bool camera_image_flip,
                       bool enable_right_side_measure,
                       int camera_buffer_count_linux,
                       String sdk_verbose_log_file,
                       bool depth_stabilization,
                       CUcontext sdk_cuda_ctx,
                       InputType input,
                       String optional_settings_path)

        bool save(String filename)
        bool load(String filename)


    cdef cppclass RuntimeParameters 'sl::RuntimeParameters':
        SENSING_MODE sensing_mode
        bool enable_depth
        bool enable_point_cloud
        REFERENCE_FRAME measure3D_reference_frame

        RuntimeParameters(SENSING_MODE sensing_mode,
                          bool enable_depth,
                          bool enable_point_cloud,
                          REFERENCE_FRAME measure3D_reference_frame)

        bool save(String filename)
        bool load(String filename)


    cdef cppclass TrackingParameters 'sl::TrackingParameters':
        Transform initial_world_transform
        bool enable_spatial_memory
        bool enable_pose_smoothing
        bool set_floor_as_origin
        String area_file_path
        bool enable_imu_fusion

        TrackingParameters(Transform init_pos,
                           bool _enable_memory,
                           String _area_path)

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


    cdef enum STREAMING_CODEC:
        STREAMING_CODEC_AVCHD
        STREAMING_CODEC_HEVC
        STREAMING_CODEC_LAST

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
        StreamingParameters(STREAMING_CODEC codec, unsigned short port_, unsigned int bitrate, int gop_size, bool adaptative_bitrate_)

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
        unsigned long long timestamp

        Transform pose_data

        int pose_confidence
        float pose_covariance[36]

    cdef cppclass IMUData(Pose):
        IMUData()
        IMUData(const IMUData &pose)
        IMUData(const Transform &pose_data, unsigned long long mtimestamp, int mconfidence)

        Matrix3f orientation_covariance
        Vector3[float] angular_velocity
        Vector3[float] linear_acceleration
        Matrix3f angular_velocity_convariance
        Matrix3f linear_acceleration_convariance


    cdef cppclass Camera 'sl::Camera':
        Camera()
        void close()
        ERROR_CODE open(InitParameters init_parameters)
        bool isOpened()
        ERROR_CODE grab(RuntimeParameters rt_parameters)
        ERROR_CODE retrieveImage(Mat &mat, VIEW view, MEM type, int width, int height)
        ERROR_CODE retrieveMeasure(Mat &mat, MEASURE measure, MEM type, int width, int height)
        void setConfidenceThreshold(int conf_threshold_value)
        int getConfidenceThreshold()

        Resolution getResolution()
        void setDepthMaxRangeValue(float depth_max_range)
        float getDepthMaxRangeValue()
        float getDepthMinRangeValue()
        void setSVOPosition(int frame_number)
        int getSVOPosition()
        int getSVONumberOfFrames()
        void setCameraSettings(CAMERA_SETTINGS settings, int value, bool use_default)
        int getCameraSettings(CAMERA_SETTINGS setting)
        float getCameraFPS()
        void setCameraFPS(int desired_fps)
        float getCurrentFPS()
        timeStamp getCameraTimestamp() # deprecated
        timeStamp getCurrentTimestamp() # deprecated
        timeStamp getTimestamp(TIME_REFERENCE reference_time)
        unsigned int getFrameDroppedCount()
        CameraInformation getCameraInformation(Resolution resizer);
        SELF_CALIBRATION_STATE getSelfCalibrationState()
        void resetSelfCalibration()

        ERROR_CODE enableTracking(TrackingParameters tracking_params)
        TRACKING_STATE getPosition(Pose &camera_pose, REFERENCE_FRAME reference_frame)
        ERROR_CODE saveCurrentArea(String area_file_path);
        AREA_EXPORT_STATE getAreaExportState()
        void disableTracking(String area_file_path)
        ERROR_CODE resetTracking(Transform &path)
        ERROR_CODE getIMUData(IMUData &imu_data, TIME_REFERENCE reference_time)
        ERROR_CODE setIMUPrior(Transform &transfom)

        ERROR_CODE enableSpatialMapping(SpatialMappingParameters spatial_mapping_parameters)
        void pauseSpatialMapping(bool status)
        SPATIAL_MAPPING_STATE getSpatialMappingState()
        ERROR_CODE extractWholeMesh(Mesh &mesh)
        void requestMeshAsync()
        ERROR_CODE getMeshRequestStatusAsync()
        ERROR_CODE retrieveMeshAsync(Mesh &mesh)
        void disableSpatialMapping()

        void requestSpatialMapAsync()
        ERROR_CODE getSpatialMapRequestStatusAsync()
        ERROR_CODE retrieveSpatialMapAsync(Mesh &mesh)
        ERROR_CODE retrieveSpatialMapAsync(FusedPointCloud &fpc)
        ERROR_CODE extractWholeSpatialMap(Mesh &mesh)
        ERROR_CODE extractWholeSpatialMap(FusedPointCloud &fpc)

        ERROR_CODE findPlaneAtHit(Vector2[uint] coord, Plane &plane)
        ERROR_CODE findFloorPlane(Plane &plane, Transform &resetTrackingFloorFrame, float floor_height_prior, Rotation world_orientation_prior, float floor_height_prior_tolerance)

        ERROR_CODE enableRecording(String video_filename, SVO_COMPRESSION_MODE compression_mode)
        RecordingState record()
        void disableRecording()

        ERROR_CODE enableStreaming(StreamingParameters streaming_parameters)
        void disableStreaming()
        bool isStreamingEnabled()


        @staticmethod
        String getSDKVersion()

        @staticmethod
        int isZEDconnected()

        @staticmethod
        ERROR_CODE sticktoCPUCore(int cpu_core)

        @staticmethod
        vector[DeviceProperties] getDeviceList()

        @staticmethod
        vector[StreamingProperties] getStreamingDeviceList()

    bool saveDepthAs(Camera &zed, DEPTH_FORMAT format, String name, float factor)
    bool savePointCloudAs(Camera &zed, POINT_CLOUD_FORMAT format, String name,
                          bool with_color)


cdef extern from "Utils.cpp" namespace "sl":
    bool saveMatDepthAs(Mat &depth, DEPTH_FORMAT format, String name, float factor)
    bool saveMatPointCloudAs(Mat &cloud, POINT_CLOUD_FORMAT format, String name,
                          bool with_color)

