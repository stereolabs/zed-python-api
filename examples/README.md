# Run the examples

Some Python samples require OpenCV and OpenGL, you can install them via pip with **opencv-python** and **PyOpenGL** packages.

## Camera control

Live camera sample showing the camera information and video in real time and allows to control the different settings.

```
python examples/camera_control/camera_control.py
```

## Positional tracking
 
Position sample shows how to get the position of the camera and uses OpenGL.

```
python examples/positional_tracking/positional_tracking.py
```

## Spatial mapping

Mesh sample shows mesh information after filtering and applying texture on frames. The mesh and its filter parameters can be saved.

```
python examples/spatial_mapping/spatial_mapping.py svo_file.svo
```

## Plane detection

Plane sample is searching for the floor in a video and extracts it into a mesh if it found it.

```
python examples/plane_detection/plane_detection.py svo_file.svo
```

## SVO examples

### SVO recording

This sample shows how to record video in Stereolabs SVO format. SVO video files can be played with the ZED API and used with its different modules.

```
python examples/svo_recording/svo_recording.py svo_file.svo
```

### SVO playback

This sample demonstrates how to read a SVO video file.

```
python examples/svo_recording/svo_playback.py svo_file.svo
```

### SVO export

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH_VIEW).

It can also convert a SVO in the following png image sequences: LEFT+RIGHT, LEFT+DEPTH_VIEW, and LEFT+DEPTH_16Bit.

```
python examples/svo_recording/svo_export.py svo_file.svo
```

## Streaming

These 2 samples show the local network streaming capabilities of the SDK. The sender is opening the camera and transmitting the images.
The receiver opens the network image stream and display the images.

```
python examples/camera_streaming/sender/streaming_sender.py
```

```
python examples/camera_streaming/receiver/streaming_receiver.py 127.0.0.1
```