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

# File containing the Cython declarations to use the Core.hpp functions.

from libcpp cimport bool
from libcpp.vector cimport vector
cimport pyzed.types as types


cdef extern from "sl/Core.hpp" namespace "sl":

    types.timeStamp getCurrentTimeStamp()

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
        types.Vector3[float] R
        types.Vector3[float] T
        CameraParameters left_cam
        CameraParameters right_cam


    cdef struct CameraInformation:
        CalibrationParameters calibration_parameters
        CalibrationParameters calibration_parameters_raw
        unsigned int serial_number
        unsigned int firmware_version
        types.MODEL camera_model

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
        types.String name
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
        types.ERROR_CODE updateCPUfromGPU()
        types.ERROR_CODE updateGPUfromCPU()
        types.ERROR_CODE copyTo(Mat &dst, COPY_TYPE cpyType) const
        types.ERROR_CODE setFrom(const Mat &src, COPY_TYPE cpyType) const
        types.ERROR_CODE read(const char* filePath)
        types.ERROR_CODE write(const char* filePath)
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
        types.String getInfos()
        bool isInit()
        bool isMemoryOwner()
        types.ERROR_CODE clone(const Mat &src)
        types.ERROR_CODE move(Mat &dst)

        @staticmethod
        void swap(Mat &mat1, Mat &mat2)

    cdef cppclass Rotation(types.Matrix3f):
        Rotation()
        Rotation(const Rotation &rotation)
        Rotation(const types.Matrix3f &mat)
        Rotation(const Orientation &orientation)
        Rotation(const float angle, const Translation &axis)
        void setOrientation(const Orientation &orientation)
        Orientation getOrientation() const
        types.Vector3[float] getRotationVector()
        void setRotationVector(const types.Vector3[float] &vec_rot)
        types.Vector3[float] getEulerAngles(bool radian) const
        void setEulerAngles(const types.Vector3[float] &euler_angles, bool radian)


    cdef cppclass Translation(types.Vector3):
        Translation()
        Translation(const Translation &translation)
        Translation(float t1, float t2, float t3)
        Translation operator*(const Orientation &mat) const
        void normalize()

        @staticmethod
        Translation normalize(const Translation &tr)
        float &operator()(int x)


    cdef cppclass Orientation(types.Vector4):
        Orientation()
        Orientation(const Orientation &orientation)
        Orientation(const types.Vector4[float] &input)
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


    cdef cppclass Transform(types.Matrix4f):
        Transform()
        Transform(const Transform &motion)
        Transform(const types.Matrix4f &mat)
        Transform(const Rotation &rotation, const Translation &translation)
        Transform(const Orientation &orientation, const Translation &translation)
        void setRotationMatrix(const Rotation &rotation)
        Rotation getRotationMatrix() const
        void setTranslation(const Translation &translation)
        Translation getTranslation() const
        void setOrientation(const Orientation &orientation)
        Orientation getOrientation() const
        types.Vector3[float] getRotationVector()
        void setRotationVector(const types.Vector3[float] &vec_rot)
        types.Vector3[float] getEulerAngles(bool radian) const
        void setEulerAngles(const types.Vector3[float] &euler_angles, bool radian)


ctypedef unsigned char uchar1
ctypedef types.Vector2[unsigned char] uchar2
ctypedef types.Vector3[unsigned char] uchar3
ctypedef types.Vector4[unsigned char] uchar4

ctypedef float float1
ctypedef types.Vector2[float] float2
ctypedef types.Vector3[float] float3
ctypedef types.Vector4[float] float4


cdef extern from "Utils.cpp" namespace "sl":

    Mat matResolution(Resolution resolution, MAT_TYPE mat_type, uchar1 *ptr_cpu, size_t step_cpu,
                      uchar1 *ptr_gpu, size_t step_gpu)

    types.ERROR_CODE setToUchar1(Mat &mat, uchar1 value, MEM memory_type)
    types.ERROR_CODE setToUchar2(Mat &mat, uchar2 value, MEM memory_type)
    types.ERROR_CODE setToUchar3(Mat &mat, uchar3 value, MEM memory_type)
    types.ERROR_CODE setToUchar4(Mat &mat, uchar4 value, MEM memory_type)

    types.ERROR_CODE setToFloat1(Mat &mat, float1 value, MEM memory_type)
    types.ERROR_CODE setToFloat2(Mat &mat, float2 value, MEM memory_type)
    types.ERROR_CODE setToFloat3(Mat &mat, float3 value, MEM memory_type)
    types.ERROR_CODE setToFloat4(Mat &mat, float4 value, MEM memory_type)

    types.ERROR_CODE setValueUchar1(Mat &mat, size_t x, size_t y, uchar1 value, MEM memory_type)
    types.ERROR_CODE setValueUchar2(Mat &mat, size_t x, size_t y, uchar2 value, MEM memory_type)
    types.ERROR_CODE setValueUchar3(Mat &mat, size_t x, size_t y, uchar3 value, MEM memory_type)
    types.ERROR_CODE setValueUchar4(Mat &mat, size_t x, size_t y, uchar4 value, MEM memory_type)

    types.ERROR_CODE setValueFloat1(Mat &mat, size_t x, size_t y, float1 value, MEM memory_type)
    types.ERROR_CODE setValueFloat2(Mat &mat, size_t x, size_t y, float2 value, MEM memory_type)
    types.ERROR_CODE setValueFloat3(Mat &mat, size_t x, size_t y, float3 value, MEM memory_type)
    types.ERROR_CODE setValueFloat4(Mat &mat, size_t x, size_t y, float4 value, MEM memory_type)

    types.ERROR_CODE getValueUchar1(Mat &mat, size_t x, size_t y, uchar1 *value, MEM memory_type)
    types.ERROR_CODE getValueUchar2(Mat &mat, size_t x, size_t y, types.Vector2[uchar1] *value, MEM memory_type)
    types.ERROR_CODE getValueUchar3(Mat &mat, size_t x, size_t y, types.Vector3[uchar1] *value, MEM memory_type)
    types.ERROR_CODE getValueUchar4(Mat &mat, size_t x, size_t y, types.Vector4[uchar1] *value, MEM memory_type)

    types.ERROR_CODE getValueFloat1(Mat &mat, size_t x, size_t y, float1 *value, MEM memory_type)
    types.ERROR_CODE getValueFloat2(Mat &mat, size_t x, size_t y, types.Vector2[float1] *value, MEM memory_type)
    types.ERROR_CODE getValueFloat3(Mat &mat, size_t x, size_t y, types.Vector3[float1] *value, MEM memory_type)
    types.ERROR_CODE getValueFloat4(Mat &mat, size_t x, size_t y, types.Vector4[float1] *value, MEM memory_type)

    uchar1 *getPointerUchar1(Mat &mat, MEM memory_type)
    uchar2 *getPointerUchar2(Mat &mat, MEM memory_type)
    uchar3 *getPointerUchar3(Mat &mat, MEM memory_type)
    uchar4 *getPointerUchar4(Mat &mat, MEM memory_type)

    float1 *getPointerFloat1(Mat &mat, MEM memory_type)
    float2 *getPointerFloat2(Mat &mat, MEM memory_type)
    float3 *getPointerFloat3(Mat &mat, MEM memory_type)
    float4 *getPointerFloat4(Mat &mat, MEM memory_type)




cdef class PyMat:
    cdef Mat mat

cdef class PyRotation(types.PyMatrix3f):
    cdef Rotation rotation


cdef class PyTranslation:
    cdef Translation translation


cdef class PyOrientation:
    cdef Orientation orientation


cdef class PyTransform(types.PyMatrix4f):
    cdef Transform transform


cdef class PyCameraParameters:
    cdef CameraParameters camera_params


cdef class PyCalibrationParameters:
    cdef CalibrationParameters calibration
    cdef PyCameraParameters py_left_cam
    cdef PyCameraParameters py_right_cam
    cdef types.Vector3[float] R
    cdef types.Vector3[float] T


cdef class PyCameraInformation:
    cdef PyCalibrationParameters py_calib
    cdef PyCalibrationParameters py_calib_raw
    cdef unsigned int serial_number
    cdef unsigned int firmware_version
    cdef types.MODEL camera_model
