# Stereolabs ZED - Python API

This package lets you use the ZED stereo camera in Python 3. The Python API is a wrapper around the ZED SDK which is written in C++ optimized code. We make the ZED SDK accessible from external Python code using Cython.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com/developers)
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/)

### Prerequisites

To start using the ZED SDK in Python, you will need to install the following dependencies on your system:  

- [ZED SDK 3.1](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python 3.5+ x64  (3.7 recommended, [Windows installer](https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe))
- [Cython 0.28](http://cython.org/#download)
- [Numpy 1.13](https://www.scipy.org/scipylib/download.html)
- OpenCV Python (optional)
- PyOpenGL (optional)

Please check your python version with the following command. The result should be 3.5 or higher.

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

**Note:** On Linux, it is advised to use the `python3` command instead of `python` which by default point to python 2.7. To do so, the following packages `python3-dev` and `python3-pip` need to be installed.

## Installing the Python API

A Python script is available in the ZED SDK installation folder and can automatically detect your platform, CUDA and Python version and download the corresponding pre-compiled Python API package.

### Running the install script

The Python install script is located on Windows in `C:\Program Files (x86)\ZED SDK\` (make sure you have admin access to run it in the Program Files folder). On Linux it is located in `/usr/local/zed/`.

Run the script:

```bash
$ cd "/usr/local/zed/"
$ python get_python_api.py

    # The script displays the detected platform versions
    CUDA 10.0
    Platform ubuntu18
    ZED 3.1
    Python 3.7
    # Downloads the whl package
    Downloading python package from https://download.stereolabs.com/zedsdk/3.1/ubuntu18/cu100/py37 ...

    # Gives instruction on how to install the downloaded package
    File saved into pyzed-3.1-cp37-cp37m-linux_x86_64.whl
    To install it run :
      python3 -m pip install pyzed-3.1-cp37-cp37m-linux_x86_64.whl
```

Now install the downloaded package with pip:

```bash
$ python3 -m pip install pyzed-3.1-cp37-cp37m-linux_x86_64.whl

    Processing ./pyzed-3.1-cp37-cp37m-linux_x86_64.whl
    Installing collected packages: pyzed
    Successfully installed pyzed-3.1
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

