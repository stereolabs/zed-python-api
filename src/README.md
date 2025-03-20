# Stereolabs ZED - Python API Sources (Advanced)

**To install and use the Python wrapper please refer to [this page instead](https://github.com/stereolabs/zed-python-api)**

## Getting started

### Prerequisites

- [ZED SDK 5.0](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python 3.8+ x64
- C++ compiler
- [Cython >= 3.0.0](http://cython.org/#download)
- [Numpy >= 2.0](https://numpy.org/install/)
- CuPy (optional)

The ZED SDK 4.X compatible API can be found in the [zedsdk_4.X branch](https://github.com/stereolabs/zed-python-api/tree/zedsdk_4.X).

Please check your python version with the following command. The result should be 3.8 or higher.

```
python --version
```

Cython and Numpy can be installed via pip.
```
python -m pip install cython numpy
```

**Note:** On Linux, it is advised to use the `python3` command instead of `python` which by default point to python 2.7. To do so, the following packages `python3-dev` and `python3-pip` need to be installed.

```
python3 --version
pip3 install -r requirements.txt
```
  
### Build the plugin

```
python setup.py build
python setup.py install
python -m pip wheel .
python -m pip install {WHEEL_FILE}.whl --force-reinstall
```

or on Linux

```
python3 setup.py build
python3 setup.py install
python3 -m pip wheel .
python3 -m pip install {WHEEL_FILE}.whl --force-reinstall
```

#### Tips

##### Error on compilation

If an __error__ occurs during the compilation, make sure that you're using the latest [ZED SDK](https://www.stereolabs.com/developers/) and that you installed an x64 version of python. `python -c "import platform; print(platform.architecture())"`

The packages *.pyd* for Windows or *.so* for Linux will be generated and installed.

You can use `python setup.py cleanall` to remove every cpp files generated and build directory.

> Make sure to be **out** of the plugin driectory when using it. It will prevent Python from considering the `pyzed` folder of the plugin as the **pyzed** package.

## Contributing

Feel free to open an issue if you find a bug, or a pull request for bug fixes, features or other improvements.
