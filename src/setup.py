# !/usr/bin/env python
########################################################################
#
# Copyright (c) 2022, STEREOLABS.
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
    Setup file to build, install, clean the pyzed package.
"""
import os
import sys
import shutil
import re
# import numpy # Import moved to after setup_requires might install it

from setuptools import setup, Extension
from Cython.Build import cythonize

# It's good practice to try importing numpy after setup_requires had a chance
# However, numpy.get_include() is needed early for incDirs.
# oldest-supported-numpy will ensure a suitable version is available for the build.
import numpy


incDirs = ""
libDirs = ""
libs = ""
cflags = ""

ZED_SDK_MAJOR = "5"
ZED_SDK_MINOR = "0"

cuda_path = "/usr/local/cuda"

def check_zed_sdk_version_private(file_path):

    global ZED_SDK_MAJOR
    global ZED_SDK_MINOR

    with open(file_path, "r", encoding="utf-8") as myfile:
        data = myfile.read()

    p = re.compile("ZED_SDK_MAJOR_VERSION (.*)")
    major = p.search(data).group(1)

    p = re.compile("ZED_SDK_MINOR_VERSION (.*)")
    minor = p.search(data).group(1)

    p = re.compile("ZED_SDK_PATCH_VERSION (.*)")
    patch = p.search(data).group(1)

    if major == ZED_SDK_MAJOR and minor >= ZED_SDK_MINOR:
        print("ZED SDK Version: OK")
        ZED_SDK_MAJOR = major
        ZED_SDK_MINOR = minor
    else:
        print("WARNING ! Required ZED SDK version: " + ZED_SDK_MAJOR + "." + ZED_SDK_MINOR)
        print("ZED SDK detected: " + major + "." + minor + "." + patch)
        print("Aborting")
        sys.exit(1)

def check_zed_sdk_version(file_path_):
    dev_file_path = file_path_ + "/sl_zed/defines.hpp"
    file_path = file_path_ + "/sl/Camera.hpp"
    if os.path.isfile(dev_file_path):
        # internal dev mode
        check_zed_sdk_version_private(dev_file_path)
    else:
        check_zed_sdk_version_private(file_path)

def clean_cpp():
    files_to_remove = [
        "pyzed/camera.cpp",
        "pyzed/core.cpp",
        "pyzed/defines.cpp",
        "pyzed/mesh.cpp",
        "pyzed/types.cpp"
    ]
    for f_path in files_to_remove:
        if os.path.isfile(f_path):
            os.remove(f_path)

if "clean" in "".join(sys.argv[1:]):
    target = "clean"
else:
    clean_cpp()
    target = "build"

if "cleanall" in "".join(sys.argv[1:]):
    target = "clean"
    print("Deleting Cython files ..")
    clean_cpp()
    sys.argv[1] = "clean"
    if os.path.isdir("build"):
        shutil.rmtree("build")
    sys.exit()

# numpy.get_include() will be called here.
# If oldest-supported-numpy is in setup_requires,
# setuptools attempts to install it before this script fully runs,
# making the 'oldest' numpy available for get_include().
numpy_include_dir = numpy.get_include()

if sys.platform == "win32":
    if os.getenv("ZED_SDK_ROOT_DIR") is None:
        print("Error: ZED_SDK_ROOT_DIR is not set. You must install the ZED SDK and set this environment variable.")
        sys.exit(1)
    elif os.getenv("CUDA_PATH") is None:
        print("Error: CUDA_PATH is not set. You must install Cuda and set this environment variable.")
        sys.exit(1)
    else:
        check_zed_sdk_version(os.getenv("ZED_SDK_ROOT_DIR")+"/include")
        incDirs = [numpy_include_dir, os.getenv("ZED_SDK_ROOT_DIR")+"/include",
                   os.getenv("CUDA_PATH") + "/include"]

        libDirs = [os.getenv("ZED_SDK_ROOT_DIR")+"/lib",
                   os.getenv("CUDA_PATH")+"/lib/x64"]

        libs = ["sl_zed64"]
elif "linux" in sys.platform:
    zed_path = "/usr/local/zed"
    if not os.path.isdir(zed_path):
        print(f"Error: ZED SDK not found at {zed_path}. You must install the ZED SDK.")
        sys.exit(1)
    elif not os.path.isdir(cuda_path):
        print(f"Error: CUDA not found at {cuda_path}. You must install Cuda.")
        sys.exit(1)
    else:
        check_zed_sdk_version(zed_path+"/include")
        incDirs = [numpy_include_dir,
                   zed_path + "/include",
                   cuda_path + "/include"]

        libDirs = [zed_path + "/lib",
                   cuda_path + "/lib64"]

        libs = ["sl_zed", "usb-1.0"]

        cflags = ["-std=c++14", "-Wno-reorder", "-Wno-deprecated-declarations", "-Wno-cpp", "-O3"]
else:
    print ("Unknown system.platform: %s  Installation failed, see setup.py." % sys.platform)
    sys.exit(1)

print ("compilation flags:", cflags)
print ("include dirs:", incDirs)
print ("library dirs:", libDirs)
print ("libraries:", libs)

cython_directives = {"embedsignature": True, "language_level": "3"}

def create_extension(name, sources):
    global incDirs
    global libDirs
    global libs
    global cflags

    if sys.platform == "win32":
        ext = Extension(name,
                        sources=sources,
                        include_dirs=incDirs,
                        library_dirs=libDirs,
                        libraries=libs,
                        language="c++"
                        )
        return ext
    elif "linux" in sys.platform:
        ext = Extension(name,
                        sources=sources,
                        include_dirs=incDirs,
                        library_dirs=libDirs,
                        libraries=libs,
                        runtime_library_dirs=libDirs,
                        language="c++",
                        extra_compile_args=cflags
                        )
        return ext
    else:
        print ("Unknown system.platform: %s" % sys.platform)
        return None

extensions = list()
py_packages = ["pyzed"]

GPUmodulesTable = [("pyzed.sl", ["pyzed/sl.pyx"])]

for mod in GPUmodulesTable:
    print ("Building module:", mod)
    extension = create_extension(mod[0], mod[1])
    if extension is None:
        print(f"ERROR: Failed to create extension for {mod[0]}. Platform {sys.platform} might not be supported or configured correctly.")
        sys.exit(1)
    extList = cythonize(extension, compiler_directives=cython_directives)
    extensions.extend(extList)

# Runtime requirements
install_requires = [
    'cython>=3.0.0'
    # NumPy requirement is added below based on Python version
]

# Numpy versioning based on Python version
# Numpy 2.0 dropped support for Python 3.8 and introduced ABI break
if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
    install_requires.append('numpy>=1.15,<2.0') # oldest-supported-numpy might build against ~1.15 for py38
else:
    install_requires.append('numpy>=2.0,<3.0') # For py > 3.8, target numpy 2.x

print("Install requires:", install_requires)

# Build-time requirements
setup_requires = [
    'setuptools>=42', # Good to have a recent setuptools for pyproject.toml compatibility (even if not used here)
    'cython>=3.0.0',
    'oldest-supported-numpy' # Ensures compilation against an ABI-stable NumPy
]

# For Python versions that will use numpy < 2.0 at runtime,
# ensure oldest-supported-numpy doesn't try to pull numpy 2.0 at build time.
# oldest-supported-numpy itself handles Python compatibility, but this adds an explicit cap for older Pythons.
if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
    setup_requires.append('numpy<2.0')


print("Setup requires:", setup_requires)

setup(name="pyzed",
      version= ZED_SDK_MAJOR + "." + ZED_SDK_MINOR,
      author_email="developers@stereolabs.com",
      description="Use the ZED SDK with Python",
      url='https://github.com/stereolabs/zed-python-api',
      packages=py_packages,
      ext_modules=extensions,
      python_requires='>=3.8',
      setup_requires=setup_requires, # ADDED: For build-time dependencies
      install_requires=install_requires,
      extras_require={
        'sample': [
            'opencv-python',
            'pyopengl',
        ]
      },
      package_data={
          'pyzed': ['*.pyi'],
      }
)