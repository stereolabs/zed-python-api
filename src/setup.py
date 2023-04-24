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
import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize

incDirs = ""
libDirs = ""
libs = ""
cflags = ""

ZED_SDK_MAJOR = "4"
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
        sys.exit(0)

def check_zed_sdk_version(file_path_):
    dev_file_path = file_path_ + "/sl_zed/defines.hpp"
    file_path = file_path_ + "/sl/Camera.hpp"
    if os.path.isfile(dev_file_path):
         # internal dev mode
        check_zed_sdk_version_private(dev_file_path)
    else:
        check_zed_sdk_version_private(file_path)

def clean_cpp():
    if os.path.isfile("pyzed/camera.cpp"):
        os.remove("pyzed/camera.cpp")
    if os.path.isfile("pyzed/core.cpp"):
        os.remove("pyzed/core.cpp")
    if os.path.isfile("pyzed/defines.cpp"):
        os.remove("pyzed/defines.cpp")
    if os.path.isfile("pyzed/mesh.cpp"):
        os.remove("pyzed/mesh.cpp")
    if os.path.isfile("pyzed/types.cpp"):
        os.remove("pyzed/types.cpp")

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

if sys.platform == "win32":
    if os.getenv("ZED_SDK_ROOT_DIR") is None:
        print(" you must install the ZED SDK.")
    elif os.getenv("CUDA_PATH") is None:
        print("Error: you must install Cuda.")
    else:
        check_zed_sdk_version(os.getenv("ZED_SDK_ROOT_DIR")+"/include")
        incDirs = [numpy.get_include(), os.getenv("ZED_SDK_ROOT_DIR")+"/include",
                   os.getenv("CUDA_PATH") + "/include"]

        libDirs = [numpy.get_include(), os.getenv("ZED_SDK_ROOT_DIR")+"/lib",
                   os.getenv("CUDA_PATH")+"/lib/x64"]

        libs = ["sl_zed64"]
elif "linux" in sys.platform:
    zed_path = "/usr/local/zed"
    if not os.path.isdir(zed_path):
        print("Error: you must install the ZED SDK.")
    elif not os.path.isdir(cuda_path):
        print("Error: you must install Cuda.")
    else:
        check_zed_sdk_version(zed_path+"/include")
        incDirs = [numpy.get_include(),
                   zed_path + "/include",
                   cuda_path + "/include"]

        libDirs = [numpy.get_include(), zed_path + "/lib",
                   cuda_path + "/lib64"]

        libs = ["sl_zed", "usb-1.0"]

        cflags = ["-std=c++14", "-Wno-reorder", "-Wno-deprecated-declarations", "-Wno-cpp", "-O3"]
else:
    print ("Unknown system.platform: %s  Installation failed, see setup.py." % sys.platform)
    exit(1)    

print ("compilation flags:", cflags)
print ("include dirs:", incDirs)
print ("library dirs:", libDirs)
print ("libraries:", libs)

#cython_directives = {"embedsignature": True, 'language_level' : "2"}
cython_directives = {"embedsignature": True}

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

GPUmodulesTable = [("pyzed.sl", ["pyzed/sl.pyx"])]      # Provide source files

for mod in GPUmodulesTable:
    print ("Building module:", mod)
    extension = create_extension(mod[0], mod[1])
    if extension == None:
        print ("WARNING: extension is None, see setup.py:", mod)
    extList = cythonize(extension, compiler_directives=cython_directives)
    extensions.extend(extList)

setup(name="pyzed",
      version= ZED_SDK_MAJOR + "." + ZED_SDK_MINOR,
      author_email="developers@stereolabs.com",
      description="Use the ZED SDK with Python",
      url='https://github.com/stereolabs/zed-python-api',
      packages=py_packages,
      ext_modules=extensions,
      python_requires='>=3.6',
      install_requires=[
        'numpy>=1.13',
        'cython>=0.28'],
      extras_require={
        'sample': [
            'opencv-python',
            'pyopengl',
        ]
      }
)
