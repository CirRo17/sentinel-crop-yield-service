"""地块多数投票工具的兼容导入入口。"""

from importlib import import_module

_parcel_majority = import_module("pipeline.crop_classification.06_parcel_majority")
NODATA_CLASS = _parcel_majority.NODATA_CLASS
attach_raster_majority_to_parcels = _parcel_majority.attach_raster_majority_to_parcels

__all__ = ["NODATA_CLASS", "attach_raster_majority_to_parcels"]
