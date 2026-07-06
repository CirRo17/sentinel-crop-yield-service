# Data directories

- `input/`: AOI, crop samples, cropland masks, and other source vectors/rasters.
  - `aoi_caobuhu.geojson`: default Caobuhu AOI for the main deliverable workflow.
  - `aoi_dangyang.geojson`: Dangyang AOI for the 2023-08 case workflow.
  - `uav_multispectral/`: future UAV multispectral orthomosaics or flight-date rasters.
- `exported/`: prepared feature stacks and training arrays.
- `output/`: classification GeoTIFFs, confidence maps, area statistics, and reports.

When UAV data is available, record its band order or file-to-band mapping in
`configs/default.yaml` under `uav_multispectral.semantic_band_mapping`.
