# Stereolabs ZED - Python Integration (beta)

This package lets you use the ZED stereo camera in Python 3.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com/developers)
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/)

### Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies 
([CUDA](https://developer.nvidia.com/cuda-downloads))
- Python 3.5+ (x64).  ([Windows installer](https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe))
-  C++ compiler (VS2015 recommended)
- [Cython 0.26](http://cython.org/#download)
- [Numpy 1.13.1](https://www.scipy.org/scipylib/download.html)

Cython and Numpy can be installed via pip.
```
python -m pip install cython numpy
```

  
### Build the plugin

```
python setup.py build
python setup.py install
```

If an __error__ occurs during the compilation, make sure that you're using the latest [ZED SDK](https://www.stereolabs.com/developers/) and that you installed an x64 version of python. `python -c "import platform; print(platform.architecture())"`

The packages *.pyd* for Windows or *.so* for Linux will be generated and installed.

You can use `python setup.py cleanall` to remove every cpp files generated and build directory.

## Use the plugin

### Code

Import the packages in your Python terminal or file like this:
```
import pyzed.camera as zcam
import pyzed.core as mat
import pyzed.defines as sl
import pyzed.types as types
import pyzed.mesh as mesh
import numpy as np
```
Vectors operations like norm, sum, square, dot, cross, distance but also simple operations can be done with
Numpy package.

**N.B.:** **pyzed.camera* is linked with *pyzed.core* and *pyzed.mesh* packages so you must 
import *pyzed.camera* before *pyzed.core* and *pyzed.mesh* to avoid import errors.

### Run the tutorials

The [tutorials](tutorials) provide simple projects to show how to use each module of the ZED SDK. For a similar version using the C++ API checkout the [Cpp tutorials](https://github.com/stereolabs/zed-examples/tree/master/tutorials).

### Run the examples

Please refer to the [examples](examples) README for more informations.

## Contributing

This is a beta version of the wrapper. Feel free to open an issue if you find a bug, or a pull request for bug fixes, features or other improvements.
