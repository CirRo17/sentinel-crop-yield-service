# Data layout

This project currently has these local data areas:

- `data/`: default runtime workspace used by scripts, the API, and examples.
- `data/input/<case>/`: case-specific raw inputs used for demos, regression checks, and manual verification.
- `data/_archive_<date>/`: local archives for historical outputs moved out of active paths.

Keep raw inputs, generated outputs, and runtime state separate. Do not place generated case deliverables under `data/input/`.

## Current state

Observed on 2026-07-09:

| Path | Files | Size | Role |
| --- | ---: | ---: | --- |
| `data/input/caobuhu/` | 14 | ~0.1 GB | Caobuhu raw input dataset |
| `data/input/tuanlinpu/` | 15 | ~0.2 GB | Tuanlinpu raw input dataset |
| `data/_archive_20260709/` | 248 | ~86.8 GB | Historical generated outputs and caches |

Large generated outputs from the former `test_data_caobuhu/` and `test_data_tuanlinpu/` roots have been moved to the archive. Their raw inputs now live under `data/input/<case>/`.

## Recommended contract

Use this contract for new work:

```text
data/
  input/       shared reusable inputs and case raw inputs
    caobuhu/
      aoi/
      original_tif/
      parcels/
    tuanlinpu/
      aoi/
      original_tif/
      parcels/
  source/      downloaded or materialized source imagery caches
    dangyang/
      sentinel2/2023_08/
    tuanlinpu/
      sentinel2/2025_07/aws_local/
      sentinel2/2026_06/aws_local/
      sentinel2/2026_06/local_tif/
    runtime/
      aws_local/
  exported/    default intermediate products for scripts
    shared/
      manifests/
      training/
    dangyang/
      feature_stack/
    tuanlinpu/
      feature_stack/
    cache/
  output/      outputs organized by research area and function
    caobuhu/
      crop_classification/
      parcel_postprocess/
    tuanlinpu/
      crop_classification/
      yield_estimation/
    runtime/
      api_predictions/
  uploads/     API-uploaded files
```

## Cleanup direction

Recommended non-breaking cleanup order:

1. Keep `data/uploads/` and `data/output/runtime/api_predictions/` under `data/`; they are API runtime state.
2. Keep shared label maps under `data/input/lables/` for now because existing sample metadata references them.
3. Keep case-specific intermediate products under `data/exported/<case>/<function>/`.
4. Prefer explicit CLI arguments such as `--input`, `--output`, `--metadata`, `--parcels`, and `--feature-stack` when running a case workflow.
5. Avoid introducing new default paths that point to a specific case.

## Local archive

`data/_archive_20260709/` contains historical outputs and caches moved out of active paths on 2026-07-09. It includes an `archive_manifest.csv` file with original and archive paths.

## Likely future structure

If we want a cleaner long-term layout, split runtime state from durable shared inputs:

```text
data/
  runtime/
    uploads/
    api_predictions/
  shared/
    labels/
    cases/
  cache/
```

That migration should be done with path updates and a validation run, not by manually dragging files around.
