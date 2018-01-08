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

# Source file of the camera Python module.

from cython.operator cimport dereference as deref
from cpython cimport bool

import enum
import numpy as np
cimport numpy as np

import pyzed.core as core
import pyzed.mesh as mesh
import pyzed.defines as defines
import pyzed.types as types


class PyRESOLUTION(enum.Enum):
    PyRESOLUTION_HIGH = MAPPING_RESOLUTION_HIGH
    PyRESOLUTION_MEDIUM  = MAPPING_RESOLUTION_MEDIUM
    PyRESOLUTION_LOW = MAPPING_RESOLUTION_LOW


class PyRANGE(enum.Enum):
    PyRANGE_NEAR = MAPPING_RANGE_NEAR
    PyRANGE_MEDIUM = MAPPING_RANGE_MEDIUM
    PyRANGE_FAR = MAPPING_RANGE_FAR


cdef class PyInitParameters:
    cdef InitParameters* init
    def __cinit__(self, camera_resolution=defines.PyRESOLUTION.PyRESOLUTION_HD720, camera_fps=0,
                  camera_linux_id=0, svo_input_filename="", svo_real_time_mode=False,
                  depth_mode=defines.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE,
                  coordinate_units=defines.PyUNIT.PyUNIT_MILLIMETER,
                  coordinate_system=defines.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_IMAGE,
                  sdk_verbose=False, sdk_gpu_id=-1, depth_minimum_distance=-1.0, camera_disable_self_calib=False,
                  camera_image_flip=False, enable_right_side_measure=False, camera_buffer_count_linux=4,
                  sdk_verbose_log_file="", depth_stabilization=True):
        if (isinstance(camera_resolution, defines.PyRESOLUTION) and isinstance(camera_fps, int) and
            isinstance(camera_linux_id, int) and isinstance(svo_input_filename, str) and
            isinstance(svo_real_time_mode, bool) and isinstance(depth_mode, defines.PyDEPTH_MODE) and
            isinstance(coordinate_units, defines.PyUNIT) and
            isinstance(coordinate_system, defines.PyCOORDINATE_SYSTEM) and isinstance(sdk_verbose, bool) and
            isinstance(sdk_gpu_id, int) and isinstance(depth_minimum_distance, float) and
            isinstance(camera_disable_self_calib, bool) and isinstance(camera_image_flip, bool) and
            isinstance(enable_right_side_measure, bool) and isinstance(camera_buffer_count_linux, int) and
            isinstance(sdk_verbose_log_file, str) and isinstance(depth_stabilization, bool)):

            filename = svo_input_filename.encode()
            filelog = sdk_verbose_log_file.encode()
            self.init = new InitParameters(camera_resolution.value, camera_fps, camera_linux_id,
                                           types.String(<char*> filename), svo_real_time_mode, depth_mode.value,
                                           coordinate_units.value, coordinate_system.value, sdk_verbose, sdk_gpu_id,
                                           depth_minimum_distance, camera_disable_self_calib, camera_image_flip,
                                           enable_right_side_measure, camera_buffer_count_linux,
                                           types.String(<char*> filelog), depth_stabilization)
        else:
            raise TypeError("Argument is not of right type.")

    def save(self, str filename):
        filename_save = filename.encode()
        return self.init.save(types.String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.init.load(types.String(<char*> filename_load))

    @property
    def camera_resolution(self):
        return defines.PyRESOLUTION(self.init.camera_resolution)

    @camera_resolution.setter
    def camera_resolution(self, value):
        if isinstance(value, defines.PyRESOLUTION):
            self.init.camera_resolution = value.value
        else:
            raise TypeError("Argument must be of PyRESOLUTION type.")

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
        return defines.PyDEPTH_MODE(self.init.depth_mode)

    @depth_mode.setter
    def depth_mode(self, value):
        if isinstance(value, defines.PyDEPTH_MODE):
            self.init.depth_mode = value.value
        else:
            raise TypeError("Argument must be of PyDEPTH_MODE type.")

    @property
    def coordinate_units(self):
        return defines.PyUNIT(self.init.coordinate_units)

    @coordinate_units.setter
    def coordinate_units(self, value):
        if isinstance(value, defines.PyUNIT):
            self.init.coordinate_units = value.value
        else:
            raise TypeError("Argument must be of PyUNIT type.")

    @property
    def coordinate_system(self):
        return defines.PyCOORDINATE_SYSTEM(self.init.coordinate_system)

    @coordinate_system.setter
    def coordinate_system(self, value):
        if isinstance(value, defines.PyCOORDINATE_SYSTEM):
            self.init.coordinate_system = value.value
        else:
            raise TypeError("Argument must be of PyCOORDINATE_SYSTEM type.")

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


cdef class PyRuntimeParameters:
    cdef RuntimeParameters* runtime
    def __cinit__(self, sensing_mode=defines.PySENSING_MODE.PySENSING_MODE_STANDARD, enable_depth=True,
                  enable_point_cloud=True,
                  measure3D_reference_frame=defines.PyREFERENCE_FRAME.PyREFERENCE_FRAME_CAMERA):
        if (isinstance(sensing_mode, defines.PySENSING_MODE) and isinstance(enable_depth, bool)
            and isinstance(enable_point_cloud, bool) and
            isinstance(measure3D_reference_frame, defines.PyREFERENCE_FRAME)):

            self.runtime = new RuntimeParameters(sensing_mode.value, enable_depth, enable_point_cloud,
                                                 measure3D_reference_frame.value)
        else:
            raise TypeError()

    def save(self, str filename):
        filename_save = filename.encode()
        return self.runtime.save(types.String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.runtime.load(types.String(<char*> filename_load))

    @property
    def sensing_mode(self):
        return defines.PySENSING_MODE(self.runtime.sensing_mode)

    @sensing_mode.setter
    def sensing_mode(self, value):
        if isinstance(value, defines.PySENSING_MODE):
            self.runtime.sensing_mode = value.value
        else:
            raise TypeError("Argument must be of PySENSING_MODE type.")

    @property
    def enable_depth(self):
        return self.runtime.enable_depth

    @enable_depth.setter
    def enable_depth(self, bool value):
        self.runtime.enable_depth = value

    @property
    def measure3D_reference_frame(self):
        return defines.PyREFERENCE_FRAME(self.runtime.measure3D_reference_frame)

    @measure3D_reference_frame.setter
    def measure3D_reference_frame(self, value):
        if isinstance(value, defines.PyREFERENCE_FRAME):
            self.runtime.measure3D_reference_frame = value.value
        else:
            raise TypeError("Argument must be of PyREFERENCE type.")


cdef class PyTrackingParameters:
    cdef TrackingParameters* tracking
    def __cinit__(self, core.PyTransform init_pos, _enable_memory=True, _area_path=None):
        if _area_path is None:
            self.tracking = new TrackingParameters(init_pos.transform, _enable_memory, types.String())
        else:
            raise TypeError("Argument init_pos must be initialized first with PyTransform().")

    def save(self, str filename):
        filename_save = filename.encode()
        return self.tracking.save(types.String(<char*> filename_save))

    def load(self, str filename):
        filename_load = filename.encode()
        return self.tracking.load(types.String(<char*> filename_load))

    def initial_world_transform(self, core.PyTransform init_pos):
        init_pos.transform = self.tracking.initial_world_transform
        return init_pos

    def set_initial_world_transform(self, core.PyTransform value):
        self.tracking.initial_world_transform = value.transform

    @property
    def enable_spatial_memory(self):
        return self.tracking.enable_spatial_memory

    @enable_spatial_memory.setter
    def enable_spatial_memory(self, bool value):
        self.tracking.enable_spatial_memory = value

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


cdef class PySpatialMappingParameters:
    cdef SpatialMappingParameters* spatial
    def __cinit__(self, resolution=PyRESOLUTION.PyRESOLUTION_HIGH, range=PyRANGE.PyRANGE_MEDIUM,
                  max_memory_usage=2048, save_texture=True, use_chunk_only=True,
                  reverse_vertex_order=False):
        if (isinstance(resolution, PyRESOLUTION) and isinstance(range, PyRANGE) and
            isinstance(use_chunk_only, bool) and isinstance(reverse_vertex_order, bool)):
            self.spatial = new SpatialMappingParameters(resolution.value, range.value, max_memory_usage, save_texture,
                                                        use_chunk_only, reverse_vertex_order)
        else:
            raise TypeError()

    def get_resolution(self, resolution=PyRESOLUTION.PyRESOLUTION_HIGH):
        if isinstance(resolution, PyRESOLUTION):
            return self.spatial.get(<MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of PyRESOLUTION type.")

    def set_resolution(self, resolution=PyRESOLUTION.PyRESOLUTION_HIGH):
        if isinstance(resolution, PyRESOLUTION):
            self.spatial.set(<MAPPING_RESOLUTION> resolution.value)
        else:
            raise TypeError("Argument is not of PyRESOLUTION type.")

    def get_range(self, range=PyRANGE.PyRANGE_MEDIUM):
        if isinstance(range, PyRANGE):
            return self.spatial.get(<MAPPING_RANGE> range.value)
        else:
            raise TypeError("Argument is not of PyRANGE type.")

    def set_range(self, range=PyRANGE.PyRANGE_MEDIUM):
        if isinstance(range, PyRANGE):
            self.spatial.set(<MAPPING_RANGE> range.value)
        else:
            raise TypeError("Argument is not of PyRANGE type.")

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


cdef class PyPose:
    cdef Pose pose
    def __cinit__(self):
        self.pose = Pose()

    def init_pose(self, PyPose pose):
        self.pose = Pose(pose.pose)

    def init_transform(self, core.PyTransform pose_data, mtimestamp=0, mconfidence=0):
        self.pose = Pose(pose_data.transform, mtimestamp, mconfidence)

    def get_translation(self, core.PyTranslation py_translation):
        py_translation.translation = self.pose.getTranslation()
        return py_translation

    def get_orientation(self, core.PyOrientation py_orientation):
        py_orientation.orientation = self.pose.getOrientation()
        return py_orientation

    def get_rotation_matrix(self, core.PyRotation py_rotation):
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

    @property
    def timestamp(self):
        return self.pose.timestamp

    def pose_data(self, core.PyTransform pose_data):
        pose_data.transform = self.pose.pose_data
        pose_data.mat = self.pose.pose_data
        return pose_data

    @property
    def pose_confidence(self):
        return self.pose.pose_confidence


cdef class PyIMUData:
    cdef IMUData imuData

    def __cinit__(self):
        self.imuData = IMUData()
        
    def init_imuData(self, PyIMUData imuData):
        self.imuData = IMUData(imuData.imuData)

    def init_transform(self, core.PyTransform pose_data, mtimestamp=0, mconfidence=0):
        self.imuData = IMUData(pose_data.transform, mtimestamp, mconfidence)

    def get_orientation_covariance(self, types.PyMatrix3f orientation_covariance):
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

    def get_angular_velocity_convariance(self, types.PyMatrix3f angular_velocity_convariance):
        angular_velocity_convariance.mat = self.imuData.angular_velocity_convariance
        return angular_velocity_convariance

    def get_linear_acceleration_convariance(self, types.PyMatrix3f linear_acceleration_convariance):
        linear_acceleration_convariance.mat = self.imuData.linear_acceleration_convariance
        return linear_acceleration_convariance



cdef class PyZEDCamera:
    def __cinit__(self):
        self.camera = Camera()

    def close(self):
        self.camera.close()

    def open(self, PyInitParameters py_init):
        if py_init:
            return types.PyERROR_CODE(self.camera.open(deref(py_init.init)))
        else:
            print("InitParameters must be initialized first with PyInitParameters().")

    def is_opened(self):
        return self.camera.isOpened()

    def grab(self, PyRuntimeParameters py_runtime):
        if py_runtime:
            return types.PyERROR_CODE(self.camera.grab(deref(py_runtime.runtime)))
        else:
            print("RuntimeParameters must be initialized first with PyRuntimeParameters().")

    def retrieve_image(self, core.PyMat py_mat, view=defines.PyVIEW.PyVIEW_LEFT, type=core.PyMEM.PyMEM_CPU, width=0,
                       height=0):
        if (isinstance(view, defines.PyVIEW) and isinstance(type, core.PyMEM) and isinstance(width, int) and
           isinstance(height, int)):
            return types.PyERROR_CODE(self.camera.retrieveImage(py_mat.mat, view.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of PyVIEW, PyMEM and integer types.")

    def retrieve_measure(self, core.PyMat py_mat, measure=defines.PyMEASURE.PyMEASURE_DEPTH, type=core.PyMEM.PyMEM_CPU,
                         width=0, height=0):
        if (isinstance(measure, defines.PyMEASURE) and isinstance(type, core.PyMEM) and isinstance(width, int) and
           isinstance(height, int)):
            return types.PyERROR_CODE(self.camera.retrieveMeasure(py_mat.mat, measure.value, type.value, width, height))
        else:
            raise TypeError("Arguments must be of PyMEASURE, PyMEM and integer types.")

    def set_confidence_threshold(self, int conf_treshold_value):
        self.camera.setConfidenceThreshold(conf_treshold_value)

    def get_confidence_threshold(self):
        return self.camera.getConfidenceThreshold()

    def get_resolution(self):
        return core.PyResolution(self.camera.getResolution().width, self.camera.getResolution().height)

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
        if isinstance(settings, defines.PyCAMERA_SETTINGS) and isinstance(use_default, bool):
            self.camera.setCameraSettings(settings.value, value, use_default)
        else:
            raise TypeError("Arguments must be of PyCAMERA_SETTINGS and boolean types.")

    def get_camera_settings(self, setting):
        if isinstance(setting, defines.PyCAMERA_SETTINGS):
            return self.camera.getCameraSettings(setting.value)
        else:
            raise TypeError("Argument is not of PyCAMERA_SETTINGS type.")

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
        if isinstance(time_reference, defines.PyTIME_REFERENCE):
            return self.camera.getTimestamp(time_reference.value)
        else:
            raise TypeError("Argument is not of PyTIME_REFERENCE type.")

    def get_frame_dropped_count(self):
        return self.camera.getFrameDroppedCount()

    def get_camera_information(self, resizer = core.PyResolution(0, 0)):
        return core.PyCameraInformation(self, resizer)

    def get_self_calibration_state(self):
        return defines.PySELF_CALIBRATION_STATE(self.camera.getSelfCalibrationState())

    def reset_self_calibration(self):
        self.camera.resetSelfCalibration()

    def enable_tracking(self, PyTrackingParameters py_tracking):
        if py_tracking:
            return types.PyERROR_CODE(self.camera.enableTracking(deref(py_tracking.tracking)))
        else:
            print("TrackingParameters must be initialized first with PyTrackingParameters().")
   
    def get_imu_data(self, PyIMUData py_imu_data, time_reference = defines.PyTIME_REFERENCE.PyTIME_REFERENCE_CURRENT):
        if isinstance(time_reference, defines.PyTIME_REFERENCE):
            return types.PyERROR_CODE(self.camera.getIMUData(py_imu_data.imuData, time_reference.value))
        else:
            raise TypeError("Argument is not of PyTIME_REFERENCE type.")
    
    def set_imu_prior(self, core.PyTransform transfom):
        return types.PyERROR_CODE(self.camera.setIMUPrior(transfom.transform))

    def get_position(self, PyPose py_pose, reference_frame = defines.PyREFERENCE_FRAME.PyREFERENCE_FRAME_WORLD):
        if isinstance(reference_frame, defines.PyREFERENCE_FRAME):
            return defines.PyTRACKING_STATE(self.camera.getPosition(py_pose.pose, reference_frame.value))
        else:
            raise TypeError("Argument is not of PyREFERENCE_FRAME type.")

    def get_area_export_state(self):
        return defines.PyAREA_EXPORT_STATE(self.camera.getAreaExportState())
   
    def save_current_area(self, str area_file_path):
        filename = area_file_path.encode()
        return types.PyERROR_CODE(self.camera.saveCurrentArea(types.String(<char*> filename)))

    def disable_tracking(self, str area_file_path):
        filename = area_file_path.encode()
        self.camera.disableTracking(types.String(<char*> filename))


    def reset_tracking(self, core.PyTransform path):
        return types.PyERROR_CODE(self.camera.resetTracking(path.transform))

    def enable_spatial_mapping(self, PySpatialMappingParameters py_spatial):
        if py_spatial:
            return types.PyERROR_CODE(self.camera.enableSpatialMapping(deref(py_spatial.spatial)))
        else:
            print("SpatialMappingParameters must be initialized first with PySpatialMappingParameters()")

    def pause_spatial_mapping(self, status):
        if isinstance(status, bool):
            self.camera.pauseSpatialMapping(status)
        else:
            raise TypeError("Argument is not of boolean type.")

    def get_spatial_mapping_state(self):
        return defines.PySPATIAL_MAPPING_STATE(self.camera.getSpatialMappingState())

    def extract_whole_mesh(self, mesh.PyMesh py_mesh):
        return types.PyERROR_CODE(self.camera.extractWholeMesh(deref(py_mesh.mesh)))

    def request_mesh_async(self):
        self.camera.requestMeshAsync()

    def get_mesh_request_status_async(self):
        return types.PyERROR_CODE(self.camera.getMeshRequestStatusAsync())

    def retrieve_mesh_async(self, mesh.PyMesh py_mesh):
        return types.PyERROR_CODE(self.camera.retrieveMeshAsync(deref(py_mesh.mesh)))

    def disable_spatial_mapping(self):
        self.camera.disableSpatialMapping()

    def enable_recording(self, str video_filename,
                          compression_mode=defines.PySVO_COMPRESSION_MODE.PySVO_COMPRESSION_MODE_LOSSLESS):
        if isinstance(compression_mode, defines.PySVO_COMPRESSION_MODE):
            filename = video_filename.encode()
            return types.PyERROR_CODE(self.camera.enableRecording(types.String(<char*> filename),
                                      compression_mode.value))
        else:
            raise TypeError("Argument is not of PySVO_COMPRESSION_MODE type.")

    def record(self):
        return self.camera.record()

    def disable_recording(self):
        self.camera.disableRecording()

    def get_sdk_version(self):
        return self.camera.getSDKVersion().get().decode()

    def is_zed_connected(self):
        return self.camera.isZEDconnected()

    def stickto_cpu_core(self, int cpu_core):
        return types.PyERROR_CODE(self.camera.sticktoCPUCore(cpu_core))


def save_camera_depth_as(PyZEDCamera zed, format, str name, factor=1):
    if isinstance(format, defines.PyDEPTH_FORMAT) and factor <= 65536:
        name_save = name.encode()
        return saveDepthAs(zed.camera, format.value, types.String(<char*>name_save), factor)
    else:
        raise TypeError("Arguments must be of PyDEPTH_FORMAT type and factor not over 65536.")


def save_camera_point_cloud_as(PyZEDCamera zed, format, str name, with_color=False):
    if isinstance(format, defines.PyPOINT_CLOUD_FORMAT):
        name_save = name.encode()
        return savePointCloudAs(zed.camera, format.value, types.String(<char*>name_save),
                                with_color)
    else:
        raise TypeError("Argument is not of PyPOINT_CLOUD_FORMAT type.")

def save_mat_depth_as(core.PyMat py_mat, format, str name, factor=1):
    if isinstance(format, defines.PyDEPTH_FORMAT) and factor <= 65536:
        name_save = name.encode()
        return saveMatDepthAs(py_mat.mat, format.value, types.String(<char*>name_save), factor)
    else:
        raise TypeError("Arguments must be of PyDEPTH_FORMAT type and factor not over 65536.")


def save_mat_point_cloud_as(core.PyMat py_mat, format, str name, with_color=False):
    if isinstance(format, defines.PyPOINT_CLOUD_FORMAT):
        name_save = name.encode()
        return saveMatPointCloudAs(py_mat.mat, format.value, types.String(<char*>name_save),
                                with_color)
    else:
        raise TypeError("Argument is not of PyPOINT_CLOUD_FORMAT type.")
