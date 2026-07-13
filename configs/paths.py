"""项目路径构建器。

从配置文件名自动生成项目前缀，并提供统一的路径构建方法。
换研究区只需更换 --config 参数，代码零改动。

用法:
    from configs.paths import ProjectPaths
    paths = ProjectPaths("configs/tuanlinpu_2026_06.yaml")
    print(paths.feature_stack)
    # data/exported/feature_stack/tuanlinpu_2026_06_feature_stack.tif
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ProjectPaths:
    """从配置 YAML 推导所有输入/输出路径。

    目录层级:
        data/
          input/          — 共享输入（AOI、标签等），不按项目分
          exported/       — 中间产物（特征栈、训练数据、缓存）
            feature_stack/
            shared/
            cache/
          output/         — 最终输出，按功能分目录，文件带项目前缀
            crop_classification/
            yield_estimation/
            parcel_postprocess/
            growth_monitoring/
            pest_detect/
            harvest_window/
          source/         — 源影像缓存
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.prefix = self.config_path.stem  # tuanlinpu_2026_06
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with open(self.config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def project_name(self) -> str:
        return self.config.get("project", {}).get("name", self.prefix)

    @property
    def geometry(self) -> Path:
        """AOI 文件路径。"""
        geo = self.config.get("project", {}).get("geometry", "")
        return Path(str(geo)) if geo else Path(f"data/input/aoi/{self.prefix}_aoi.shp")

    # -- 数据源 ------------------------------------------------------------------

    @property
    def source_dir(self) -> Path:
        return Path("data/source")

    # -- 中间产物 ----------------------------------------------------------------

    @property
    def exported_dir(self) -> Path:
        return Path("data/exported")

    @property
    def feature_stack(self) -> Path:
        return self.exported_dir / "feature_stack" / f"{self.prefix}_feature_stack.tif"

    @property
    def feature_stack_metadata(self) -> Path:
        return self.exported_dir / "feature_stack" / f"{self.prefix}_feature_stack_metadata.json"

    @property
    def training_dir(self) -> Path:
        return self.exported_dir / "shared" / "training"

    @property
    def training_data(self) -> Path:
        return self.training_dir / "pixel_training_data.npz"

    @property
    def training_report(self) -> Path:
        return self.training_dir / "pixel_training_data_report.json"

    @property
    def cache_dir(self) -> Path:
        return self.exported_dir / "cache"

    @property
    def parcels(self) -> Path:
        """地块 Shapefile 路径。"""
        parcels = self.config.get("project", {}).get("parcels", "")
        return Path(str(parcels)) if parcels else Path(f"data/input/parcels/{self.prefix}_parcel.shp")

    @property
    def labels_dir(self) -> Path:
        return Path("data/input/lables")

    # -- 最终输出 ----------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        return Path("data/output")

    @property
    def crop_classification_dir(self) -> Path:
        return self.output_dir / "crop_classification"

    @property
    def classification(self) -> Path:
        return self.crop_classification_dir / f"{self.prefix}_classification.tif"

    @property
    def classification_confidence(self) -> Path:
        return self.crop_classification_dir / f"{self.prefix}_confidence.tif"

    @property
    def classification_clean(self) -> Path:
        return self.crop_classification_dir / f"{self.prefix}_classification_clean.tif"

    @property
    def classification_info(self) -> Path:
        return self.crop_classification_dir / f"{self.prefix}_prediction_info.json"

    @property
    def postprocess_info(self) -> Path:
        return self.crop_classification_dir / f"{self.prefix}_postprocess_info.json"

    @property
    def parcel_postprocess_dir(self) -> Path:
        return self.output_dir / "parcel_postprocess"

    @property
    def parcel_majority_shp(self) -> Path:
        return self.parcel_postprocess_dir / f"{self.prefix}_parcel_majority.shp"

    @property
    def parcel_majority_summary(self) -> Path:
        return self.parcel_postprocess_dir / f"{self.prefix}_parcel_majority_summary.json"

    @property
    def accuracy_dir(self) -> Path:
        return self.output_dir / "accuracy_eval"

    @property
    def accuracy_report(self) -> Path:
        return self.accuracy_dir / f"{self.prefix}_accuracy_report.json"

    @property
    def confusion_matrix(self) -> Path:
        return self.accuracy_dir / f"{self.prefix}_confusion_matrix.csv"

    @property
    def class_accuracy(self) -> Path:
        return self.accuracy_dir / f"{self.prefix}_class_accuracy.csv"

    @property
    def yield_dir(self) -> Path:
        return self.output_dir / "yield_estimation"

    @property
    def yield_stats(self) -> Path:
        return self.yield_dir / f"{self.prefix}_yield_stats.json"

    @property
    def yield_summary(self) -> Path:
        return self.yield_dir / f"{self.prefix}_yield_summary.json"

    @property
    def yield_report_html(self) -> Path:
        return self.yield_dir / f"{self.prefix}_yield_report.html"

    @property
    def yield_raster(self) -> Path:
        return self.yield_dir / f"{self.prefix}_yield_all.tif"

    @property
    def growth_monitoring_dir(self) -> Path:
        return self.output_dir / "growth_monitoring"

    @property
    def growth_baseline_dir(self) -> Path:
        return self.growth_monitoring_dir / "baseline"

    @property
    def growth_baseline_manifest(self) -> Path:
        return self.growth_baseline_dir / f"{self.prefix}_baseline_manifest.json"

    @property
    def pest_detect_dir(self) -> Path:
        return self.output_dir / "pest_detect"

    @property
    def pest_inputs_dir(self) -> Path:
        return self.pest_detect_dir / "inputs"

    @property
    def pest_inputs_manifest(self) -> Path:
        return self.pest_inputs_dir / f"{self.prefix}_pest_inputs_manifest.json"

    @property
    def pest_pixel_dir(self) -> Path:
        return self.pest_detect_dir / "pixel"

    @property
    def pest_pixel_stats(self) -> Path:
        return self.pest_pixel_dir / f"{self.prefix}_pest_step2_stats.json"

    @property
    def pest_parcel_dir(self) -> Path:
        return self.pest_detect_dir / "parcel"

    @property
    def pest_parcel_output(self) -> Path:
        return self.pest_parcel_dir / f"{self.prefix}_parcel_pest_stress_grade.gpkg"

    @property
    def harvest_window_dir(self) -> Path:
        return self.output_dir / "harvest_window"

    # -- API --------------------------------------------------------------------

    @property
    def api_predictions_dir(self) -> Path:
        return self.output_dir / "runtime" / "api_predictions"

    @property
    def uploads_dir(self) -> Path:
        return Path("data/uploads")

    # -- 模型 --------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        return Path("models")

    @property
    def model_file(self) -> Path:
        return self.model_dir / "crop_classification_classifier.joblib"

    @property
    def model_info(self) -> Path:
        return self.model_dir / "crop_classification_model_info.json"
