===============
Spatial Mapping
===============

Classes
=======

.. currentmodule:: pyzed.sl

+--------------------------------------------+------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.MeshFilterParameters`     |Defines the behavior of the :meth:`~pyzed.sl.Mesh.filter()` function.                                 |
+--------------------------------------------+------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Texture`                  |Contains information about texture image associated to a :class:`~pyzed.sl.Mesh`                      |
+--------------------------------------------+------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Chunk`                    |Represents a sub-mesh, it contains local vertices and triangles.                                      |
+--------------------------------------------+------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.Mesh`                     |A mesh contains the geometric (and optionally texture) data of the scene captured by spatial mapping. |
+--------------------------------------------+------------------------------------------------------------------------------------------------------+
|:class:`~pyzed.sl.SpatialMappingParameters` |Sets the spatial mapping parameters.                                                                  |
+--------------------------------------------+------------------------------------------------------------------------------------------------------+

Enumerations
============

+---------------------------------------+----------------------------------------------------------+
|:class:`~pyzed.sl.MESH_FILE_FORMAT`    |Lists available mesh file formats.                        |
+---------------------------------------+----------------------------------------------------------+
|:class:`~pyzed.sl.MESH_TEXUTRE_FORMAT` |Lists available mesh texture formats.                     |
+---------------------------------------+----------------------------------------------------------+
|:class:`~pyzed.sl.MAPPING_RESOLUTION`  |List the spatial mapping resolution presets.              |
+---------------------------------------+----------------------------------------------------------+
|:class:`~pyzed.sl.MAPPING_RANGE`       |List the spatial mapping depth range presets.             |
+---------------------------------------+----------------------------------------------------------+
|:class:`~pyzed.sl.AREA_EXPORT_STATE`   |Lists the different states of spatial memory area export. |
+---------------------------------------+----------------------------------------------------------+

Enumeration Type Documentation
==============================

.. autoclass:: MESH_FILE_FORMAT

.. autoclass:: MESH_TEXTURE_FORMAT

.. autoclass:: MAPPING_RESOLUTION

.. autoclass:: MAPPING_RANGE

.. autoclass:: AREA_EXPORT_STATE

.. autoclass:: SPATIAL_MAPPING_STATE
