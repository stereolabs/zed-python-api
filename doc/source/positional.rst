===================
Positional Tracking
===================

Classes
=======

.. currentmodule:: pyzed.sl

+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.TrackingParameters` |Parameters for positional tracking initialization.                                                                      |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Rotation`           |Designed to contain rotation data of the positional tracking. It inherits from the generic :class:`~pyzed.sl.Matrix3f`. |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Translation`        |Designed to contain translation data of the positional tracking.                                                        |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Orientation`        |Designed to contain orientation (quaternion) data of the positional tracking.                                           |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Transform`          |Designed to contain translation and rotation data of the positional tracking.                                           |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Pose`               |Contains positional tracking data which gives the position and orientation of the ZED in 3D space.                      |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.IMUData`            |Contains inertial positional tracking data which gives the orientation of the ZED-M.                                    |
+--------------------------------------+------------------------------------------------------------------------------------------------------------------------+

Enumerations
============

+----------------------------------+-----------------------------------------------------------------------------+
|:class:`~pyzed.sl.TRACKING_STATE` |Lists the different states of positional tracking.                           |
+----------------------------------+-----------------------------------------------------------------------------+
|:class:`~pyzed.sl.REFERENCE_FRAME`|Defines which type of position matrix is used to store camera path and pose. |
+----------------------------------+-----------------------------------------------------------------------------+

Enumeration Type Documentation
==============================

.. autoclass:: TRACKING_STATE

.. autoclass:: REFERENCE_FRAME
