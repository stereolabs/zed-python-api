## Run the examples

Some Python samples require OpenCV and OpenGL, you can install them via pip with **opencv-python** and **PyOpenGL** packages.

### Live camera

Live camera sample showing the camera information and video in real time and allows to control the different settings.
    
```
python examples/live_camera.py
```

### Read SVO

Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.

```
python examples/read_svo.py svo_file.svo
```

### Position   
 
Position sample shows that the position of the camera can be get and used with OpenGL.

```
python examples/position_example.py
```

### Mesh

Mesh sample shows mesh information after filtering and applying texture on frames. The mesh and its filter parameters can be saved.

```
python examples/mesh_example.py svo_file.svo
```

### Object

Object sample shows the objects detected and tracked by the AI module with their bouding boxes and their 3D positions

```
python examples/object_example.py
```

### Plane

Plane sample is searching for the floor in a video and extracts it into a mesh if it found it.

```
python examples/plane_example.py svo_file.svo
```
