from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import geopandas as gpd
from pyproj import CRS, Geod, Transformer
from shapely.geometry import GeometryCollection, mapping, shape
from shapely.ops import transform as shapely_transform


WGS84 = CRS.from_epsg(4326)
GEOD = Geod(ellps="WGS84")


def normalize_geojson(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("type") == "FeatureCollection":
        features = obj.get("features", [])
    elif obj.get("type") == "Feature":
        features = [obj]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": obj}]

    normalized = []
    for feature in features:
        geom_obj = feature.get("geometry")
        if not geom_obj:
            continue
        geom = shape(geom_obj)
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        normalized.append(
            {
                "type": "Feature",
                "properties": feature.get("properties") or {},
                "geometry": mapping(geom),
            }
        )
    if not normalized:
        raise ValueError("No valid geometry was supplied.")
    return {"type": "FeatureCollection", "features": normalized}


def feature_collection_geometry(fc: Dict[str, Any]):
    geometries = [shape(f["geometry"]) for f in normalize_geojson(fc)["features"]]
    if len(geometries) == 1:
        return geometries[0]
    return GeometryCollection(geometries).buffer(0)


def shapes_for_crs(fc: Dict[str, Any], dst_crs) -> List[Dict[str, Any]]:
    dst = CRS.from_user_input(dst_crs)
    if dst == WGS84:
        return [f["geometry"] for f in normalize_geojson(fc)["features"]]
    transformer = Transformer.from_crs(WGS84, dst, always_xy=True)
    out = []
    for feature in normalize_geojson(fc)["features"]:
        geom = shape(feature["geometry"])
        out.append(mapping(shapely_transform(transformer.transform, geom)))
    return out


def transform_geom(geom, src_crs, dst_crs):
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    if src == dst:
        return geom
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def bounds(fc: Dict[str, Any]) -> List[float]:
    geom = feature_collection_geometry(fc)
    return [float(v) for v in geom.bounds]


def area_ha(fc: Dict[str, Any]) -> float:
    area_m2 = 0.0
    for feature in normalize_geojson(fc)["features"]:
        geom = shape(feature["geometry"])
        signed_area, _ = GEOD.geometry_area_perimeter(geom)
        area_m2 += abs(signed_area)
    return area_m2 / 10000.0


def read_vector_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return normalize_geojson(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".zip":
        gdf = gpd.read_file(f"zip://{path}")
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Uploaded vector file is empty.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=4326)
    return normalize_geojson(json.loads(gdf.to_json()))


def feature_collection_from_geometries(geometries: Iterable, properties=None) -> Dict[str, Any]:
    features = []
    for geom in geometries:
        if geom.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": properties or {},
                "geometry": mapping(geom),
            }
        )
    return normalize_geojson({"type": "FeatureCollection", "features": features})
