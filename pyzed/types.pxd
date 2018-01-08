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

# File containing the Cython declarations to use the types.hpp functions.

from libcpp.string cimport string
from libcpp cimport bool

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
        ERROR_CODE_CAMERA_DETECTION_ISSUE,
        ERROR_CODE_CAMERA_ALREADY_IN_USE,
        ERROR_CODE_NO_GPU_DETECTED,
        ERROR_CODE_LAST

    String errorCode2str(ERROR_CODE err)

    void sleep_ms(int time)

    ctypedef enum MODEL:
        MODEL_ZED,
        MODEL_ZED_M,
        MODEL_LAST

    String model2str(MODEL model)

    cdef cppclass String 'sl::String':

        String()
        String(const char *data)
        void set(const char *data)
        const char *get() const
        bool empty() const
        string std_str() const


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
        const T &operator[](int i) const


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


cdef class PyMatrix3f:
    cdef Matrix3f mat


cdef class PyMatrix4f:
    cdef Matrix4f mat
