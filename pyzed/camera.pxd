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

# File containing the Cython declarations to use the Camera.hpp functions.

from libcpp cimport bool
from libcpp.pair cimport pair

cimport pyzed.defines as defines
cimport pyzed.core as core
cimport pyzed.types as types
cimport pyzed.mesh as mesh


cdef extern from 'sl/Camera.hpp' namespace 'sl':

    ctypedef enum RESOLUTION 'sl::SpatialMappingParameters::RESOLUTION':
        RESOLUTION_HIGH 'sl::SpatialMappingParameters::RESOLUTION::RESOLUTION_HIGH'
        RESOLUTION_MEDIUM 'sl::SpatialMappingParameters::RESOLUTION::RESOLUTION_MEDIUM'
        RESOLUTION_LOW 'sl::SpatialMappingParameters::RESOLUTION::RESOLUTION_LOW'


    ctypedef enum RANGE 'sl::SpatialMappingParameters::RANGE':
        RANGE_NEAR 'sl::SpatialMappingParameters::RANGE::RANGE_NEAR'
        RANGE_MEDIUM 'sl::SpatialMappingParameters::RANGE::RANGE_MEDIUM'
        RANGE_FAR 'sl::SpatialMappingParameters::RANGE::RANGE_FAR'


    cdef cppclass InitParameters 'sl::InitParameters':
        defines.RESOLUTION camera_resolution
        int camera_fps
        int camera_linux_id
        types.String svo_input_filename
        bool svo_real_time_mode
        defines.UNIT coordinate_units
        defines.COORDINATE_SYSTEM coordinate_system
        defines.DEPTH_MODE depth_mode
        float depth_minimum_distance
        int camera_image_flip
        bool enable_right_side_measure
        bool camera_disable_self_calib
        int camera_buffer_count_linux
        bool sdk_verbose
        int sdk_gpu_id
        bool depth_stabilization

        types.String sdk_verbose_log_file

        InitParameters(defines.RESOLUTION camera_resolution,
                       int camera_fps,
                       int camera_linux_id,
                       types.String svo_input_filename,
                       bool svo_real_time_mode,
                       defines.DEPTH_MODE depth_mode,
                       defines.UNIT coordinate_units,
                       defines.COORDINATE_SYSTEM coordinate_system,
                       bool sdk_verbose,
                       int sdk_gpu_id,
                       float depth_minimum_distance,
                       bool camera_disable_self_calib,
                       bool camera_image_flip,
                       bool enable_right_side_measure,
                       int camera_buffer_count_linux,
                       types.String sdk_verbose_log_file,
                       bool depth_stabilization)

        bool save(types.String filename)
        bool load(types.String filename)


    cdef cppclass RuntimeParameters 'sl::RuntimeParameters':
        defines.SENSING_MODE sensing_mode
        bool enable_depth
        bool enable_point_cloud
        defines.REFERENCE_FRAME measure3D_reference_frame

        RuntimeParameters(defines.SENSING_MODE sensing_mode,
                          bool enable_depth,
                          bool enable_point_cloud,
                          defines.REFERENCE_FRAME measure3D_reference_frame)

        bool save(types.String filename)
        bool load(types.String filename)


    cdef cppclass TrackingParameters 'sl::TrackingParameters':
        core.Transform initial_world_transform
        bool enable_spatial_memory
        types.String area_file_path

        TrackingParameters(core.Transform init_pos,
                           bool _enable_memory,
                           types.String _area_path)

        bool save(types.String filename)
        bool load(types.String filename)


    cdef cppclass SpatialMappingParameters 'sl::SpatialMappingParameters':
        ctypedef pair[float, float] interval

        SpatialMappingParameters(RESOLUTION resolution,
                                 RANGE range,
                                 int max_memory_usage_,
                                 bool save_texture_,
                                 bool keep_mesh_consistent_,
                                 bool inverse_triangle_vertices_order_)

        @staticmethod
        float get(RESOLUTION resolution)

        void set(RESOLUTION resolution)

        @staticmethod
        float get(RANGE range)

        void set(RANGE range)

        int max_memory_usage
        bool save_texture
        bool keep_mesh_consistent
        bool inverse_triangle_vertices_order

        const interval allowed_min
        const interval allowed_max
        interval range_meter
        const interval allowed_resolution
        float resolution_meter

        bool save(types.String filename)
        bool load(types.String filename)


    cdef cppclass Pose 'sl::Pose':
        Pose()
        Pose(const Pose &pose)
        Pose(const core.Transform &pose_data, unsigned long long mtimestamp, int mconfidence)
        core.Translation getTranslation()
        core.Orientation getOrientation()
        core.Rotation getRotationMatrix()
        types.Vector3[float] getRotationVector()
        types.Vector3[float] getEulerAngles(bool radian)

        bool valid
        unsigned long long timestamp

        core.Transform pose_data

        int pose_confidence


    cdef cppclass Camera 'sl::Camera':
        Camera()
        void close()
        types.ERROR_CODE open(InitParameters init_parameters)
        bool isOpened()
        types.ERROR_CODE grab(RuntimeParameters rt_parameters)
        types.ERROR_CODE retrieveImage(core.Mat &mat, defines.VIEW view, core.MEM type, int width, int height)
        types.ERROR_CODE retrieveMeasure(core.Mat &mat, defines.MEASURE measure, core.MEM type, int width, int height)
        void setConfidenceThreshold(int conf_threshold_value)
        int getConfidenceThreshold()

        core.Resolution getResolution()
        void setDepthMaxRangeValue(float depth_max_range)
        float getDepthMaxRangeValue()
        float getDepthMinRangeValue()
        void setSVOPosition(int frame_number)
        int getSVOPosition()
        int getSVONumberOfFrames()
        void setCameraSettings(defines.CAMERA_SETTINGS settings, int value, bool use_default)
        int getCameraSettings(defines.CAMERA_SETTINGS setting)
        float getCameraFPS()
        void setCameraFPS(int desired_fps)
        float getCurrentFPS()
        types.timeStamp getCameraTimestamp()
        types.timeStamp getCurrentTimestamp()
        unsigned int getFrameDroppedCount()
        core.CameraInformation getCameraInformation(core.Resolution resizer);
        defines.SELF_CALIBRATION_STATE getSelfCalibrationState()
        void resetSelfCalibration()

        types.ERROR_CODE enableTracking(TrackingParameters tracking_params)
        defines.TRACKING_STATE getPosition(Pose &camera_pose, defines.REFERENCE_FRAME reference_frame)
        defines.AREA_EXPORT_STATE getAreaExportState()
        void disableTracking(types.String area_file_path)
        void resetTracking(core.Transform &path)

        types.ERROR_CODE enableSpatialMapping(SpatialMappingParameters spatial_mapping_parameters)
        void pauseSpatialMapping(bool status)
        defines.SPATIAL_MAPPING_STATE getSpatialMappingState()
        types.ERROR_CODE extractWholeMesh(mesh.Mesh &mesh)
        void requestMeshAsync()
        types.ERROR_CODE getMeshRequestStatusAsync()
        types.ERROR_CODE retrieveMeshAsync(mesh.Mesh &mesh)
        void disableSpatialMapping()

        types.ERROR_CODE enableRecording(types.String video_filename, defines.SVO_COMPRESSION_MODE compression_mode)
        defines.RecordingState record()
        void disableRecording()

        @staticmethod
        types.String getSDKVersion()

        @staticmethod
        int isZEDconnected()

        @staticmethod
        types.ERROR_CODE sticktoCPUCore(int cpu_core)

    bool saveDepthAs(Camera &zed, defines.DEPTH_FORMAT format, types.String name, float factor)
    bool savePointCloudAs(Camera &zed, defines.POINT_CLOUD_FORMAT format, types.String name,
                          bool with_color, bool keep_occluded_point)


cdef extern from "Utils.cpp" namespace "sl":
    bool saveMatDepthAs(core.Mat &depth, defines.DEPTH_FORMAT format, types.String name, float factor)
    bool saveMatPointCloudAs(core.Mat &cloud, defines.POINT_CLOUD_FORMAT format, types.String name,
                          bool with_color, bool keep_occluded_point)


cdef class PyZEDCamera:
    cdef Camera camera
