# Data directories

`data/` is the default runtime workspace used by scripts and the API. Keep
case-specific raw inputs under `input/<case>/`; generated case outputs
belong in explicit output paths or local archives.

- `input/`: AOI, crop samples, cropland masks, and other source vectors/rasters.
  - `aoi_caobuhu.geojson`: default Caobuhu AOI for the main deliverable workflow.
  - `aoi_dangyang.geojson`: Dangyang AOI for the 2023-08 case workflow.
  - `caobuhu/`: Caobuhu case raw inputs such as AOI vectors, original imagery, and parcel vectors.
  - `tuanlinpu/`: Tuanlinpu case raw inputs such as AOI vectors, original imagery, and parcel vectors.
  - `uav_multispectral/`: future UAV multispectral orthomosaics or flight-date rasters.
- `exported/`: intermediate products organized by research area and function.
  - `shared/manifests/`: shared scene manifests.
  - `shared/training/`: shared training arrays.
  - `dangyang/feature_stack/`: Dangyang feature stacks and metadata.
  - `tuanlinpu/feature_stack/`: Tuanlinpu feature stacks and metadata.
  - `cache/`: temporary feature-building cache.
- `output/`: outputs organized by research area and function.
  - `caobuhu/crop_classification/`: Caobuhu crop classification rasters and metadata.
  - `caobuhu/parcel_postprocess/`: Caobuhu parcel-level post-processing outputs.
  - `tuanlinpu/crop_classification/`: Tuanlinpu crop classification metadata.
  - `tuanlinpu/yield_estimation/`: Tuanlinpu yield rasters and reports.
  - `runtime/api_predictions/`: API-generated inference and yield task artifacts.
- `source/`: downloaded or materialized source imagery caches.
  - `dangyang/sentinel2/2023_08/`: Dangyang Sentinel-2 source rasters.
  - `tuanlinpu/sentinel2/2025_07/aws_local/`: Tuanlinpu 2025-07 Sentinel-2 source rasters.
  - `tuanlinpu/sentinel2/2026_06/aws_local/`: Tuanlinpu 2026-06 Sentinel-2 source rasters from AWS/STAC.
  - `tuanlinpu/sentinel2/2026_06/local_tif/`: Tuanlinpu 2026-06 source rasters from local TIFF inputs.
  - `runtime/aws_local/`: default source cache for new generic Sentinel runs.
- `uploads/`: API-uploaded files and extracted parcel archives.

When UAV data is available, record its band order or file-to-band mapping in
`configs/default.yaml` under `uav_multispectral.semantic_band_mapping`.

See `docs/DATA_LAYOUT.md` for the current cleanup contract and migration notes.
