# Stereolabs ZED - Python API Sources (Advanced)

**To install and use the Python wrapper please refer to [this page instead](https://github.com/stereolabs/zed-python-api)**

## Getting started

### Prerequisites

Please check your python version with the following command. The result should be 3.8 or higher.

```
python --version
```

- [ZED SDK 5.1](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python 3.8+ x64
- C++ compiler

**Python Dependencies:**
- Use `requirements.txt` for Python 3.9+ with modern dependencies
- Use `requirements_legacy.txt` for older Python versions or legacy setups

Install dependencies with:
```bash
# For modern Python environments (recommended)
pip install -r requirements.txt

# For legacy Python environments
pip install -r requirements_legacy.txt
```

**Note:** On Linux, it is advised to use the `python3` command instead of `python` which by default point to python 2.7. To do so, the following packages `python3-dev` and `python3-pip` need to be installed.

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
