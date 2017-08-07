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

import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import pyzed.mesh as mesh


def main():
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720  # Use HD720 video mode (default fps: 60)
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_METER  # Set units in meters

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Enable positional tracking with default parameters.
    # Positional tracking needs to be enabled before using spatial mapping
    py_transform = core.PyTransform()
    tracking_parameters = zcam.PyTrackingParameters(init_pos=py_transform)
    err = zed.enable_tracking(tracking_parameters)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Enable spatial mapping
    mapping_parameters = zcam.PySpatialMappingParameters()
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Grab data during 500 frames
    i = 0
    py_mesh = mesh.PyMesh()  # Create a PyMesh object

    while i < 500:
        # For each new grab, mesh data is updated
        if zed.grab(zcam.PyRuntimeParameters()) == tp.PyERROR_CODE.PySUCCESS:
            # In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
            mapping_state = zed.get_spatial_mapping_state()

            print("\rImages captured: {0} / 500 || {1}".format(i, mapping_state))

            i = i + 1

    print("\n")

    # Extract, filter and save the mesh in an obj file
    print("Extracting Mesh...\n")
    zed.extract_whole_mesh(py_mesh)
    print("Filtering Mesh...\n")
    py_mesh.filter(mesh.PyMeshFilterParameters())  # Filter the mesh (remove unnecessary vertices and faces)
    print("Saving Mesh...\n")
    py_mesh.save("mesh.obj")

    # Disable tracking and mapping and close the camera
    zed.disable_spatial_mapping()
    zed.disable_tracking()
    zed.close()

if __name__ == "__main__":
    main()
