#!/usr/bin/env python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2

def main():
    
    if not sys.argv or len(sys.argv) != 2:
        print("Only the path of the output SVO file should be passed as argument.")
        exit(1)

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_NONE

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    path_output = sys.argv[1]
    err = cam.enable_recording(path_output, sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_AVCHD)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    print("SVO is Recording, use Ctrl-C to stop.")
    frames_recorded = 0

    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            # Each new frame is added to the SVO file
            state = cam.record()
            if state["status"]:
                frames_recorded += 1
            print("Frame count: " + str(frames_recorded), end="\r")

    # Stop recording
    cam.disable_recording()
    cam.close()

if __name__ == "__main__":
    main()