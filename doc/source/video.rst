=====
Video
=====

Classes
=======

.. currentmodule:: pyzed.sl

+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                     |Holds the options used to initialize the :class:`~pyzed.sl.Camera` object.                                                                                   |
|:class:`~pyzed.sl.InitParameters`    |                                                                                                                                                             |
|                                     |Once passed to the :meth:`~pyzed.sl.Camera.open()` function, these settings will be set for the entire execution life time of the :class:`~pyzed.sl.Camera`. |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Camera`            |This class is the main interface with the camera and the SDK features, suche as: video, depth, tracking, mapping, and more.                                  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.CameraInformation` |Structure containing information of a signle camera (serial number, model, calibration, etc.)                                                                |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+

Enumerations
============

+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.RESOLUTION`             |Represents the available resolutions.                |
+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.CAMERA_SETTINGS`        |Lists available camera settings for the ZED camera.  |
+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.SELF_CALIBRATION_STATE` |Status for asynchrnous self-calibration.             |
+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.VIEW`                   |Lists available views.                               |
+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.TIME_REFERENCE`         |Lists specific and particular timestamps.            |
+------------------------------------------+-----------------------------------------------------+
|:class:`~pyzed.sl.SVO_COMPRESSION_MODE`   |Lists available compression modes for SVO recording. |
+------------------------------------------+-----------------------------------------------------+

Enumeration Type Documentation
==============================

.. autoclass:: RESOLUTION

.. autoclass:: CAMERA_SETTINGS

.. autoclass:: SELF_CALIBRATION_STATE

.. autoclass:: VIEW

.. autoclass:: TIME_REFERENCE

.. autoclass:: SVO_COMPRESSION_MODE
