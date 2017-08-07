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

"""
    Position sample shows the position of the ZED camera in a OpenGL window.
"""
from OpenGL.GLUT import *
import positional_tracking.tracking_viewer as tv
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import threading


def main():

    init = zcam.PyInitParameters(camera_resolution=sl.PyRESOLUTION.PyRESOLUTION_HD720,
                                 depth_mode=sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE,
                                 coordinate_units=sl.PyUNIT.PyUNIT_METER,
                                 coordinate_system=sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_RIGHT_HANDED_Y_UP,
                                 sdk_verbose=True)
    cam = zcam.PyZEDCamera()
    status = cam.open(init)
    if status != tp.PyERROR_CODE.PySUCCESS:
        print(repr(status))
        exit()

    transform = core.PyTransform()
    tracking_params = zcam.PyTrackingParameters(transform)
    cam.enable_tracking(tracking_params)

    runtime = zcam.PyRuntimeParameters()
    camera_pose = zcam.PyPose()

    viewer = tv.PyTrackingViewer()
    viewer.init()

    py_translation = core.PyTranslation()

    start_zed(cam, runtime, camera_pose, viewer, py_translation)

    viewer.exit()
    glutMainLoop()


def start_zed(cam, runtime, camera_pose, viewer, py_translation):
    zed_callback = threading.Thread(target=run, args=(cam, runtime, camera_pose, viewer, py_translation))
    zed_callback.start()


def run(cam, runtime, camera_pose, viewer, py_translation):
    while True:
        if cam.grab(runtime) == tp.PyERROR_CODE.PySUCCESS:
            tracking_state = cam.get_position(camera_pose)
            text_translation = ""
            text_rotation = ""
            if tracking_state == sl.PyTRACKING_STATE.PyTRACKING_STATE_OK:
                rotation = camera_pose.get_rotation_vector()
                rx = round(rotation[0], 2)
                ry = round(rotation[1], 2)
                rz = round(rotation[2], 2)

                translation = camera_pose.get_translation(py_translation)
                tx = round(translation.get()[0], 2)
                ty = round(translation.get()[1], 2)
                tz = round(translation.get()[2], 2)

                text_translation = str((tx, ty, tz))
                text_rotation = str((rx, ry, rz))
                pose_data = camera_pose.pose_data(core.PyTransform())
                viewer.update_zed_position(pose_data)

            viewer.update_text(text_translation, text_rotation, tracking_state)
        else:
            tp.c_sleep_ms(1)


if __name__ == "__main__":
    main()
