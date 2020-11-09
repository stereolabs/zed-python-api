# Stereolabs ZED - Python Wrapper
How to maintain the Python wrapper with Cython.

## Documentation

The **Cython** documentation: http://cython.org/

The **Cython** wiki: https://github.com/cython/cython/wiki

The **ZED SDK** documentation:
https://www.stereolabs.com/developers/documentation/API/latest

## Configuration

New packages can simply be added in the GPUmodulesTable with the package name and its *.pyx* file.

The version of the SDK and the wrapper is the same and should be indicated in the setup version.

Check the new Cython versions, they may need to update the wrapper.

## Notes

### General

  * The wrapper keeps the same structure as the ZED SDK.
  
  * The classes of the wrapper contain a maximum of Python functions like str, repr, gettitem, settitem
  and properties.
  
  * *.pxd* files correspond to C++ headers and contain all the C++ methods and functions that can be used in
  the wrapper. *.pyx* files contain methods, functions and their bodies.
  
  * Python and C++ functions must be named differently.
  
  * Warning about deprecated Numpy API is showing during building but we can ignore it.
  
### Enum

  * Enum must be redefined in order to use them with Python.
  They have their own representation so you must redefine str and repr if there is a string
  conversion function.
  
  * Enum arguments can't be typed directly with classes defined in the wrapper.
  They must be tested inside the functions.
  
### Types

  * Each representation of matrix or vectors are done with Numpy.
  
  * Template can't be used with Cython. 
  Each function with a different type must be defined in a file like Utils.cpp
  
  * typedef made inside classes is not working in Cython. We can create it outside of the class and
  link it to its C++ declaration directly. See MeshFilterParameters FILTER in *mesh.pxd*
  for example.
  
  * Ctypes can be used, for example with *bool* type or *memcpy*.

  * Vector2, Vector3, Vector4; float1, float2, float3, float4; uchar1, uchar2, uchar3, uchar4 
  are not available in Python but can be used in the wrapper.
  
### Code in *.pyx*
  
  * Other classes attributes must be declared in *.pxd* file instead of *.pyx* file if you want
  to have access to them when the package is imported.

  * Default values of function arguments must be given in the Python function in *.pyx* file but not
  in the declaration in the *.pxd*.

  * Python does not support constructor overloading but Cython does. However, this is not working
  when the number of arguments is the same but the types are different. The other constructors are
  defined in a C++ file with another name and called as a function after the creation of the Python object.
  
  * When there is an object to get defined in another file, it must be given in parameters 
  if not Python will interpret it as a C++ type trying to be converted in Python object and 
  won't work. 

  * Subclasses of Matrix3f and Matrix4f have both Matrix and their own type as attributes.
  For example: PyTransform has a sl::Transform and sl::Matrix4f attributes initialized.
  
  * Python will crash if you return a C++ object not initialized (like an empty sl::String).
 
  * *memcpy* is used when there is not any access to pointer data directly.
  
## Limitations

  * PyMat/Mat constructor with resolution and pointer not working directly when Mat
    constructors with width and height are defined because of the same number of arguments,
    replaced by matResolution redefined in Utils.cpp.
  
  * ```Normalise(const Orientation  orient)```
 is not working, replaced by normalizing the Orientation argument with normalise().
  
  * Overloading operators like *=, /=, += and -= (for Vector3 and Vector4) are not supported by Cython.
  
