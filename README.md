# Stereolabs ZED - Python API

This package lets you use the ZED stereo camera in Python 3. The Python API is a wrapper around the ZED SDK which is written in C++ optimized code. We make the ZED SDK accessible from external Python code using Cython.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com/developers)
- For more information, read the ZED [Documentation](https://www.stereolabs.com/docs/app-development/python/install/) and [API documentation](https://www.stereolabs.com/docs/api/python/) or our [Community page](https://community.stereolabs.com)

### Prerequisites

To start using the ZED SDK in Python, you will need to install the following dependencies on your system:  

- [ZED SDK 5.0](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)

For the ZED SDK 4.2 compatible version, use the [zedsdk_4.X branch](https://github.com/stereolabs/zed-python-api/tree/zedsdk_4.X) or the [4.2 release tag](https://github.com/stereolabs/zed-python-api/releases/tag/v4.2)

- Python 3.8+ x64
- [Cython >= 3.0.0](http://cython.org/#download)
- [Numpy >= 2.0](https://numpy.org/install/)
- OpenCV Python (optional)
- PyOpenGL (optional)

Please check your python version with the following command. The result should be 3.8 or higher.

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
        ZED SDK 5.0
    # Downloads and install the whl package
    -> Checking if https://download.stereolabs.com/zedsdk/5.0/whl/linux_x86_64/pyzed-5.0-cp311-cp311-linux_x86_64.whl exists and is available
    -> Found ! Downloading python package into /home/user/pyzed-5.0-cp311-cp311-linux_x86_64.whl
    -> Installing necessary dependencies
    ...
    Successfully installed pyzed-5.0
```

To install it later or on a different environment run : 

```bash
$ python -m pip install --ignore-installed /home/user/pyzed-5.0-cp311-cp311-linux_x86_64.wh
```

That's it ! The Python API is now installed.

## Use the plugin

### Code

Import the packages in your Python terminal or file like this:
```
import pyzed.sl as sl
```

Vectors operations like norm, sum, square, dot, cross, distance but also simple operations can be done with
Numpy package.

### Run the tutorials

The [tutorials](https://github.com/stereolabs/zed-examples/tree/master/tutorials) provide simple projects to show how to use each module of the ZED SDK.

### Run the examples

Please refer to the [examples](https://github.com/stereolabs/zed-examples) README for more informations.


## Troubleshooting

###  "Numpy binary incompatiblity"

```
Traceback (most recent call last):
    ...
    File "__init__.pxd", line 918, in init pyzed.sl
ValueError: numpy.ufunc size changed, may indicate binary incompatiblity. Expected 216 from C header, got 192 from PyObject
```

This error usually means numpy isn't installed. To install it, simply run these commands : 

```bash
# On Jetson (aarch64) cython needs to be installed first since numpy needs to be compiled.
python3 -m pip install cython
python3 -m pip install numpy
```


## Compiling the Python API from source (only for developers of the python wrapper)

To compile the ZED SDK Python wrapper go to [src folder](./src) to get the cython sources and instructions.

Note : This step is not useful for *users* of the wrapper, it is only meant to be used to extend the wrapper for advanced uses.

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
