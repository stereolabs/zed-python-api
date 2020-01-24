########################################################################
#
# Copyright (c) 2018, STEREOLABS.
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
    Multi cameras sample showing how to open multiple ZED in one program
"""

import cv2
import pyzed.sl as sl


def main():
    print("Running...")

    input_type = sl.InputType()
    input_type.set_from_camera_id(0)

    init = sl.InitParameters()
    init.input = input_type
    init.camera_resolution = sl.RESOLUTION.HD720
    #init.camera_linux_id = 0
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera 1...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    input_type.set_from_camera_id(1)
    init.input = input_type
    cam2 = sl.Camera()
    if not cam2.is_opened():
        print("Opening ZED Camera 2...")
    status = cam2.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    mat2 = sl.Mat()

    print_camera_information(cam)
    print_camera_information(cam2)

    key = ''
    while key != 113:  # for 'q' key
        # The computation could also be done in a thread, one for each camera
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            cv2.imshow("ZED 1", mat.get_data())

        err = cam2.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam2.retrieve_image(mat2, sl.VIEW.LEFT)
            cv2.imshow("ZED 2", mat2.get_data())

        key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(
        round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(
        cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(
        cam.get_camera_information().serial_number))


if __name__ == "__main__":
    main()
