# !/usr/bin/env python
########################################################################
#
# Copyright (c) 2017, STEREOLABS.
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
from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy
import os
import sys
import shutil

incDirs = ""
libDirs = ""
libs = ""
cflags = ""


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
    if os.getenv("ZED_INCLUDE_DIRS") is None:
        print("Error: you must install the ZED SDK.")
    elif os.getenv("CUDA_PATH") is None:
        print("Error: you must install Cuda.")
    else:
        incDirs = [numpy.get_include(), os.getenv("ZED_INCLUDE_DIRS"),
                   os.getenv("CUDA_PATH") + "/include"]

        libDirs = [numpy.get_include(), os.getenv("ZED_LIBRARY_DIR"),
                   os.getenv("CUDA_PATH")+"/lib/x64"]

        libs = ["sl_core64", "sl_scanning64", "sl_zed64"]

cuda_path = "/usr/local/cuda"

if sys.platform == "linux":

    zed_path = "/usr/local/zed"
    if not os.path.isdir(zed_path):
        print("Error: you must install the ZED SDK.")
    elif not os.path.isdir(cuda_path):
        print("Error: you must install Cuda.")
    else:

        incDirs = [numpy.get_include(),
                   zed_path + "/include",
                   cuda_path + "/include"]

        libDirs = [numpy.get_include(), zed_path + "/lib",
                   cuda_path + "/lib64"]

        libs = ["sl_core", "sl_scanning", "sl_zed"]

        cflags = ["-std=c++11"]

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

    if sys.platform == "linux":
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

extensions = list()

py_packages = ["pyzed"]

package_data = {"pyzed": ["*.pxd"]}

GPUmodulesTable = [("pyzed.defines", ["pyzed/defines.pyx"]),
                   ("pyzed.types", ["pyzed/types.pyx"]),
                   ("pyzed.core", ["pyzed/core.pyx"]),
                   ("pyzed.mesh", ["pyzed/mesh.pyx"]),
                   ("pyzed.camera", ["pyzed/camera.pyx"])
                   ]

for mod in GPUmodulesTable:
    extList = cythonize(create_extension(mod[0], mod[1]), compiler_directives=cython_directives)
    extensions.extend(extList)

setup(name="pyzed",
      version="2.1",
      author_email="developers@stereolabs.com",
      description="Use the ZED SDK with Python",
      packages=py_packages,
      ext_modules=extensions,
      package_data=package_data)
