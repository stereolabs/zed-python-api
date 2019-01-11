=============
Depth Sensing
=============

Classes
=======

.. currentmodule:: pyzed.sl

+-----------------------------------------+-----------------------------------------------------------------------------+
|:class:`~pyzed.sl.RuntimeParameters`     |Parameters that defines the behavior of the grab.                            |
+-----------------------------------------+-----------------------------------------------------------------------------+
|:class:`~pyzed.sl.CameraParameters`      |Intrinsic parameters of a camera                                             |
+-----------------------------------------+-----------------------------------------------------------------------------+
|:class:`~pyzed.sl.CalibrationParameters` |Intrinsic and Extrinsic parameters of the camera (translation and rotation). |
+-----------------------------------------+-----------------------------------------------------------------------------+

Functions
=========

.. currentmodule:: pyzed.sl

+-----------------------------------+-------------------------------------------+
|:func:`~save_camera_depth_as`      |Writes the current depth map into a file.  |
+-----------------------------------+-------------------------------------------+
|:func:`~save_camera_point_cloud_as`|Writes the current point cloud into a file.|
+-----------------------------------+-------------------------------------------+

Enumerations
============

+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.DEPTH_MODE`          |Lists available depth computation modes.                                                                                                 |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.SENSING_MODE`        |Lists available depth sensing modes.                                                                                                     |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.MEASURE`             |Lists retrievable measures.                                                                                                              |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.DEPTH_FORMAT`        |Lists available file formats for saving depth maps.                                                                                      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.POINT_CLOUD_FORMAT`  |Lists available file formats for saving point clouds. Stores the spatial coordinates (x,y,z) of each pixel and optionally its RGB color. |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+

Function Documentation
======================

.. autofunction:: save_camera_depth_as

.. autofunction:: save_camera_point_cloud_as

Enumeration Type Documentation
==============================

.. autoclass:: DEPTH_MODE

.. autoclass:: SENSING_MODE

.. autoclass:: MEASURE

.. autoclass:: DEPTH_FORMAT

.. autoclass:: POINT_CLOUD_FORMAT
