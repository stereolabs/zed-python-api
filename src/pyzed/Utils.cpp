///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/**
 *
 * File containing functions from Core.hpp and Camera.hpp to use them with Cython.
 * It contains a Mat constructor and Camera saving functions which can't be used directly with Cython because of the
 * same number of arguments of other functions with different argument types.
 * There are other functions from Mat converting values or pointers from the functions of Core.hpp using templates.
 *
 */

#include <stdio.h>
#include <string.h>

#include <sl/Camera.hpp>

namespace sl {

    Mat matResolution(sl::Resolution resolution, MAT_TYPE mat_type, uchar1 *ptr_cpu, size_t step_cpu, uchar1 *ptr_gpu, size_t step_gpu) {
        return Mat(resolution, mat_type, ptr_cpu, step_cpu, ptr_gpu, step_gpu);
    }

    ERROR_CODE setToUchar1(sl::Mat &mat, uchar1 value, MEM memory_type) {
        return mat.setTo<uchar1>(value, memory_type);
    }

    ERROR_CODE setToUchar2(sl::Mat &mat, uchar2 value, MEM memory_type) {
        return mat.setTo<uchar2>(value, memory_type);
    }

    ERROR_CODE setToUchar3(sl::Mat &mat, uchar3 value, MEM memory_type) {
        return mat.setTo<uchar3>(value, memory_type);
    }

    ERROR_CODE setToUchar4(sl::Mat &mat, uchar4 value, MEM memory_type) {
        return mat.setTo<uchar4>(value, memory_type);
    }

    ERROR_CODE setToUshort1(sl::Mat &mat, ushort1 value, MEM memory_type) {
        return mat.setTo<ushort1>(value, memory_type);
    }

    ERROR_CODE setToFloat1(sl::Mat &mat, float1 value, MEM memory_type) {
        return mat.setTo<float1>(value, memory_type);
    }

    ERROR_CODE setToFloat2(sl::Mat &mat, float2 value, MEM memory_type) {
        return mat.setTo<float2>(value, memory_type);
    }

    ERROR_CODE setToFloat3(sl::Mat &mat, float3 value, MEM memory_type) {
        return mat.setTo<float3>(value, memory_type);
    }

    ERROR_CODE setToFloat4(sl::Mat &mat, float4 value, MEM memory_type) {
        return mat.setTo<float4>(value, memory_type);
    }

    ERROR_CODE setValueUchar1(sl::Mat &mat, size_t x, size_t y, uchar1 value, MEM memory_type) {
        return mat.setValue<uchar1>(x, y, value, memory_type);
    }

    ERROR_CODE setValueUchar2(sl::Mat &mat, size_t x, size_t y, uchar2 value, MEM memory_type) {
        return mat.setValue<uchar2>(x, y, value, memory_type);
    }

    ERROR_CODE setValueUchar3(sl::Mat &mat, size_t x, size_t y, uchar3 value, MEM memory_type) {
        return mat.setValue<uchar3>(x, y, value, memory_type);
    }

    ERROR_CODE setValueUchar4(sl::Mat &mat, size_t x, size_t y, uchar4 value, MEM memory_type) {
        return mat.setValue<uchar4>(x, y, value, memory_type);
    }

    ERROR_CODE setValueUshort1(sl::Mat &mat, size_t x, size_t y, ushort1 value, MEM memory_type) {
        return mat.setValue<ushort1>(x, y, value, memory_type);
    }

    ERROR_CODE setValueFloat1(sl::Mat &mat, size_t x, size_t y, float1 value, MEM memory_type) {
        return mat.setValue<float1>(x, y, value, memory_type);
    }

    ERROR_CODE setValueFloat2(sl::Mat &mat, size_t x, size_t y, float2 value, MEM memory_type) {
        return mat.setValue<float2>(x, y, value, memory_type);
    }

    ERROR_CODE setValueFloat3(sl::Mat &mat, size_t x, size_t y, float3 value, MEM memory_type) {
        return mat.setValue<float3>(x, y, value, memory_type);
    }

    ERROR_CODE setValueFloat4(sl::Mat &mat, size_t x, size_t y, float4 value, MEM memory_type) {
        return mat.setValue<float4>(x, y, value, memory_type);
    }

    ERROR_CODE getValueUchar1(sl::Mat &mat, size_t x, size_t y, uchar1 *value, MEM memory_type) {
        return mat.getValue<uchar1>(x, y, value, memory_type);
    }

    ERROR_CODE getValueUchar2(sl::Mat &mat, size_t x, size_t y, uchar2 *value, MEM memory_type) {
        return mat.getValue<uchar2>(x, y, value, memory_type);
    }

    ERROR_CODE getValueUchar3(sl::Mat &mat, size_t x, size_t y, uchar3 *value, MEM memory_type) {
        return mat.getValue<uchar3>(x, y, value, memory_type);
    }

    ERROR_CODE getValueUchar4(sl::Mat &mat, size_t x, size_t y, uchar4 *value, MEM memory_type) {
        return mat.getValue<uchar4>(x, y, value, memory_type);
    }

    ERROR_CODE getValueUshort1(sl::Mat &mat, size_t x, size_t y, ushort1 *value, MEM memory_type) {
        return mat.getValue<ushort1>(x, y, value, memory_type);
    }

    ERROR_CODE getValueFloat1(sl::Mat &mat, size_t x, size_t y, float1 *value, MEM memory_type) {
        return mat.getValue<float1>(x, y, value, memory_type);
    }

    ERROR_CODE getValueFloat2(sl::Mat &mat, size_t x, size_t y, float2 *value, MEM memory_type) {
        return mat.getValue<float2>(x, y, value, memory_type);
    }

    ERROR_CODE getValueFloat3(sl::Mat &mat, size_t x, size_t y, float3 *value, MEM memory_type) {
        return mat.getValue<float3>(x, y, value, memory_type);
    }

    ERROR_CODE getValueFloat4(sl::Mat &mat, size_t x, size_t y, float4 *value, MEM memory_type) {
        return mat.getValue<float4>(x, y, value, memory_type);
    }

    uchar1 *getPointerUchar1(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<uchar1>(memory_type);
    }

    uchar2 *getPointerUchar2(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<uchar2>(memory_type);
    }

    uchar3 *getPointerUchar3(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<uchar3>(memory_type);
    }

    uchar4 *getPointerUchar4(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<uchar4>(memory_type);
    }

    ushort1 *getPointerUshort1(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<ushort1>(memory_type);
    }

    float1 *getPointerFloat1(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<float1>(memory_type);
    }

    float2 *getPointerFloat2(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<float2>(memory_type);
    }

    float3 *getPointerFloat3(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<float3>(memory_type);
    }

    float4 *getPointerFloat4(sl::Mat &mat, MEM memory_type) {
        return mat.getPtr<float4>(memory_type);
    }

/*    bool saveMatDepthAs(sl::Mat &depth, sl::DEPTH_FORMAT format, sl::String name, float factor = 1.) {
        return saveDepthAs(depth, format, name, factor);
    }

    bool saveMatPointCloudAs(sl::Mat &cloud, sl::POINT_CLOUD_FORMAT format, sl::String name, bool with_color = false) {
        return savePointCloudAs(cloud, format, name, with_color);
    }
*/
    std::string to_str(sl::String sl_str) {
        return std::string(sl_str.c_str());
    }

    sl::ObjectDetectionRuntimeParameters* create_object_detection_runtime_parameters(float confidence_threshold, 
                                        std::vector<int> object_vector,
                                        std::map<int,float> object_class_confidence_map) {

        std::vector<sl::OBJECT_CLASS> object_vector_cpy;
        for (unsigned int i = 0; i < object_vector.size(); i++)
            object_vector_cpy.push_back(static_cast<sl::OBJECT_CLASS>(object_vector[i]));
        
        std::map<sl::OBJECT_CLASS,float> object_class_confidence_map_cpy = std::map<sl::OBJECT_CLASS,float>();
        if (object_class_confidence_map.size()>0) {
            for (const auto& map_elem: object_class_confidence_map) {
                object_class_confidence_map_cpy[static_cast<sl::OBJECT_CLASS>(map_elem.first)] = map_elem.second;
            }
        }
        return new ObjectDetectionRuntimeParameters(confidence_threshold, object_vector_cpy, object_class_confidence_map_cpy);
    }
}
