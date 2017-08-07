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

import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720  # Use HD720 video mode (default fps: 60)
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_METER  # Set units in meters

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Enable positional tracking with default parameters
    py_transform = core.PyTransform()  # First create a PyTransform object for PyTrackingParameters object
    tracking_parameters = zcam.PyTrackingParameters(init_pos=py_transform)
    err = zed.enable_tracking(tracking_parameters)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Track the camera position during 1000 frames
    i = 0
    zed_pose = zcam.PyPose()
    while i < 1000:
        if zed.grab(zcam.PyRuntimeParameters()) == tp.PyERROR_CODE.PySUCCESS:
            # Get the pose of the left eye of the camera with reference to the world frame
            zed.get_position(zed_pose, sl.PyREFERENCE_FRAME.PyREFERENCE_FRAME_WORLD)

            # Display the translation and timestamp
            py_translation = core.PyTranslation()
            tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
            ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
            tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
            print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp))

            # Display the orientation quaternion
            py_orientation = core.PyOrientation()
            ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
            oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
            print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

            i = i + 1

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
