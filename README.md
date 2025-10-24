# Stereolabs ZED - Python API

This package lets you use the ZED stereo camera in Python 3. The Python API is a wrapper around the ZED SDK which is written in C++ optimized code. We make the ZED SDK accessible from external Python code using Cython.

## Table of Contents

- [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
- [Installing the Python API](#installing-the-python-api)
    - [Running the install script](#running-the-install-script)
- [Use the plugin](#use-the-plugin)
    - [Code](#code)
    - [Run the tutorials](#run-the-tutorials)
    - [Run the examples](#run-the-examples)
- [GPU data retrieval using CuPy](#GPU-data-retrieval-using-CuPy)
    - [Validate your CuPy setup](#validate-your-CuPy-setup)
- [Troubleshooting](#troubleshooting)
    - ["Numpy binary incompatibility"](#numpy-binary-incompatibility)
    - ["PyTorch on Jetson requiring NumPy 1.x while pyzed requires NumPy 2.x"](#pytorch-on-jetson-requiring-numpy-1x-while-pyzed-requires-numpy-2x)
    - ["CuPy failed to load libnvrtc.so.1x"](#cupy-failed-to-load-libnvrtc-so1x)
- [Compiling the Python API from source (only for developers of the python wrapper)](#compiling-the-python-api-from-source-only-for-developers-of-the-python-wrapper)
- [Support](#support)

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com/developers)
- For more information, read the ZED [Documentation](https://www.stereolabs.com/docs/app-development/python/install/) and [API documentation](https://www.stereolabs.com/docs/api/python/) or our [Community page](https://community.stereolabs.com)

### Prerequisites

To start using the ZED SDK in Python, you will need to install the following dependencies on your system:

- To use pyzed
    - [ZED SDK 5.1](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)
        - For the ZED SDK 5.0 compatible version, use the [zedsdk_5.X branch](https://github.com/stereolabs/zed-python-api/tree/zedsdk_5.X) or the [5.0.7 release tag](https://github.com/stereolabs/zed-python-api/releases/tag/v5.0.7)
        - For the ZED SDK 4.2 compatible version, use the [zedsdk_4.X branch](https://github.com/stereolabs/zed-python-api/tree/zedsdk_4.X) or the [4.2 release tag](https://github.com/stereolabs/zed-python-api/releases/tag/v4.2)
    - Python 3.8 to Python 3.14
    - [Cython >= 3.0.0](http://cython.org/#download)
    - [Numpy >= 2.0](https://numpy.org/install/)
- To use most of our samples (optional)
    - OpenCV Python
    - PyOpenGL
- To profit from the GPU acceleration and getting the data on the GPU (optional)
    - [CuPy](https://cupy.dev/)

Please check your python version with the following command. The result should be between 3.8 and 3.14.

```
python --version
```

Cython and Numpy can be installed via pip.
```
python -m pip install cython numpy
```

The sample dependencies can also be installed via pip
```
python -m pip install opencv-python pyopengl
```

**Note:** On Linux, it is advised to use the `python3` command instead of `python` which by default may point to python 2.7. To do so, the following packages `python3-dev` and `python3-pip` need to be installed.

## Installing the Python API

A Python script is available in the ZED SDK installation folder and can automatically detect your platform, CUDA and Python version and download the corresponding pre-compiled Python API package.

### Running the install script

**Windows**

The Python install script is located in: `C:\Program Files (x86)\ZED SDK\`

:warning: *Make sure you have admin access to run it in the Program Files folder, otherwise, you will have a `Permission denied` error. You can still copy the file into another location to run it without permissions.*

**Linux**

The Python install script is located in: `/usr/local/zed/`


Run the script:

```bash
$ cd "/usr/local/zed/"
$ python get_python_api.py
    # The script displays the detected platform versions
    Detected platform: 
        linux_x86_64
        Python 3.11
        ZED SDK 5.1
    # Downloads and install the whl package
    -> Checking if https://download.stereolabs.com/zedsdk/5.1/whl/linux_x86_64/pyzed-5.1-cp311-cp311-linux_x86_64.whl exists and is available
    -> Found ! Downloading python package into /home/user/pyzed-5.1-cp311-cp311-linux_x86_64.whl
    -> Installing necessary dependencies
    ...
    Successfully installed pyzed-5.1
```

To install it later or on a different environment run : 

```bash
$ python -m pip install --ignore-installed /home/user/pyzed-5.1-cp311-cp311-linux_x86_64.wh
```

That's it ! The Python API is now installed.

## Use the plugin

### Code

Import the packages in your Python terminal or file like this:
```
import pyzed.sl as sl
```

Vectors operations like norm, sum, square, dot, cross, distance but also simple operations can be done with numpy package on CPU or CuPy package on GPU.

### Run the tutorials

The [tutorials](https://github.com/stereolabs/zed-examples/tree/master/tutorials) provide simple projects to show how to use each module of the ZED SDK.

### Run the examples

Please refer to the [examples](https://github.com/stereolabs/zed-examples) README for more informations.

## GPU data retrieval using CuPy

`CuPy` is a NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python (see https://cupy.dev/) and the ZED Python API support getting data in its format.

Calling `mat.get_data(sl.MEM_GPU, deep_copy=False)` on a `sl.Mat` object will give you a `cupy.ndarray` object.

### Validate your CuPy setup

**Prerequisites**:
- A plugged camera
- pyzed
- [Numpy >= 2.0](https://numpy.org/install/)
- [CuPy](https://cupy.dev/) (corresponding to your CUDA version)

You can use script the script `cupy/test_cupy_integration.py`. The script will:
 - Open a connected ZED camera.
 - Retrieve an image.
 - Run some operations and benchmark on the retrieved image.
 - Display on the terminal the results of the tests.

Without deep-diving into the script content, you can just look at what it prints to validate everything is fine with your setup.

For example, on an Orin NX16 with a ZED X

```
> python test_cupy_integration.py
✅ CuPy detected - GPU acceleration available
   CuPy version: 13.5.1
   CUDA version: 12080
ZED SDK CuPy Integration Test
========================================
Opening ZED camera...
[2025-07-31 12:54:15 UTC][ZED][INFO] Logging level INFO
[2025-07-31 12:54:16 UTC][ZED][INFO] Using GMSL input... Switched to default resolution HD1200
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Camera FW version: 2001
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Video mode: HD1200@30
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Serial Number: S/N 48922857
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Depth mode: NEURAL
ZED camera opened successfully.
Retrieving image data...
Retrieved image on GPU: 1920x1200

🧪 Testing GPU image processing (basic grayscale conversion)...
   Input image: (1200, 1920, 4)
   Processed image: (1200, 1920)
✅ GPU processing test passed!
========================================

💾 Testing memory allocation strategies...
   CPU allocation: (480, 640, 4), float32
   GPU allocation: (480, 640, 4), float32
   CPU->GPU transfer: (480, 640, 4)
   GPU->CPU transfer: (480, 640, 4)
✅ Memory allocation test passed!
========================================

🔍 Testing GPU memory usage...
   Initial GPU memory usage: 0.0 MB
   After allocation: 15.3 MB
   After cleanup: 0.0 MB
✅ GPU memory test passed!
========================================

🔬 Testing data integrity...
   Data integrity verified: (2, 2, 4)
✅ Data integrity test passed!
========================================

⚡ Running performance benchmark...
   Benchmark image size: 1920x1200
   CPU processing (10 iterations): 538.658 milliseconds
   GPU processing (10 iterations): 93.961 milliseconds
   Speedup: 5.7x
🚀 GPU processing is faster!
========================================

🎉 All tests completed!
   Your system is ready for GPU-accelerated ZED processing with the Python API!
```

## Troubleshooting

###  "Numpy binary incompatibility"

```
Traceback (most recent call last):
    ...
    File "__init__.pxd", line 918, in init pyzed.sl
ValueError: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216 from C header, got 192 from PyObject
```

This error usually means numpy isn't installed. To install it, simply run these commands : 

```bash
# On Jetson (aarch64) cython needs to be installed first since numpy needs to be compiled.
python3 -m pip install cython
python3 -m pip install numpy
```

### "PyTorch on Jetson requiring NumPy 1.x while pyzed requires NumPy 2.x"

By default, PyTorch on Jetson requires NumPy 1.x (see NVidia's doc [Install PyTorch on Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)). Since pyzed requires NumPy 2.x, it can lead to:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
```

To fix this, until NVidia releases its wheels for PyTorch on Jetson with NumPy 2.x, you can either:
- compile PyTorch from source with NumPy 2.x: see [this post](https://forums.developer.nvidia.com/t/pytorch-environment-and-zed-sdk-conflicts-on-jetpack-6-2/334056/5) for more information.
- compile pyzed from source with NumPy 1.x: see [zed-python-api/src](https://github.com/stereolabs/zed-python-api/tree/master/src) for more information.

### "CuPy failed to load libnvrtc.so.1x"

If an __error__ like `CuPy failed to load libnvrtc.so.12` happens when using GPU retrieval, it indicates a mismatch between the installed CuPy version (e.g. cupy-cuda11x) and the CUDA runtime libraries (e.g. 12.x)

## Compiling the Python API from source (only for developers of the python wrapper)

To compile the ZED SDK Python wrapper go to [src folder](./src) to get the cython sources and instructions.

Note : This step is not useful for *users* of the wrapper, it is only meant to be used to extend the wrapper for advanced uses.

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
