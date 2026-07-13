# Sentinel Crop Service

基于 Sentinel-2 遥感影像的农作物类型识别、长势监测、病虫害检测与估产服务。
包含离线处理流水线、FastAPI 推理服务和完整的像素级→地块级分析链路。

## 功能概览

- **农作物类型识别**：支持水稻、小麦、玉米、油菜和其他类别，基于 Random Forest 像素级分类。
- **长势监测**：基于多年同期 NDVI 基准的 Z-Score 距平分析，输出像元级和地块级长势等级。
- **病虫害检测**：多时相植被指数对比（当前 vs 前期 vs 历史同期），综合评分输出地块级胁迫等级。
- **产量估算**：基于 NDVI/LAI 与经验回归模型估算作物产量。
- **收获窗口**：结合 Sentinel-1/2 和 ERA5 气象数据的收获期检测。
- **API 推理服务**：支持影像上传、多时相自动分组、分类推理、任务状态查询、结果下载。

分类编码：

```text
0  Others  / 其他
1  Rice    / 水稻
2  Wheat   / 小麦
3  Maize   / 玉米
4  Rapeseed / 油菜
```

## 项目结构

```text
SentinelCropService/
  crop_domain/                    作物领域公共定义
    labels.py                     类别编码、标签、映射
  crop_service_api/               FastAPI 服务
    api.py                        主 API 应用
    parcel_stats.py               地块统计
    home.html                     Web 门户页面
  data_sources/                   数据源适配（多平台）
    common/                       通用配置、数据结构、STAC 工具、传感器约定
    copernicus/                   Copernicus Data Space 数据源
      auth.py                     OAuth2 认证
      search.py                   STAC 场景检索
      download.py                 SAFE 格式下载
      extract.py                  波段提取
      build_features.py           特征栈构建
    aws_element84/                AWS Element84 Earth Search 数据源
      aws_open_data.py            STAC 检索与场景清单
      build_features_aws.py       特征栈构建
      cache_source_rasters.py     源影像缓存
      config.py                   波段映射与 SCL 云掩膜
    local_raster/                 本地 GeoTIFF 数据源适配
    uav/                          无人机多光谱数据源适配
  image_core/                     影像处理核心
    spectral.py                   光谱指数（NDVI、EVI、NDWI、NDRE、NBR）
    feature_schema.py             特征栈 schema 校验与波段匹配
    build_features_from_multiband.py  从本地多波段 GeoTIFF 构建特征栈
  pipeline/                       离线处理流水线
    crop_classification/          作物分类
      01_prepare_samples.py       准备训练样本
      02_train_rf.py              训练 Random Forest 模型
      03_predict_classify.py      像素级分类预测
      04_postprocess.py           后处理（置信度过滤、小斑块筛除）
      05_accuracy_eval.py         精度评估
      06_parcel_majority.py       地块级多数投票
    growth_monitoring/            长势监测
      01_prepare_baseline.py      准备多年同期基准
      02_pixel_zscore.py          像元级 Z-Score 长势分级
      03_parcel_grade.py          地块级长势定级
    pest_detect/                  病虫害检测
      01_prepare_inputs.py        准备多期历史对比数据
      02_pixel_stress_score.py    像元级胁迫异常评分
      03_parcel_pest_stress_grade.py  地块级病虫害等级
    yield_estimation/             产量估算
      01_yield_estimation.py      逐作物产量估算
      02_yield_summary.py         产量汇总与报告
    crop_harvest_window/          收获窗口检测
      01_harvest_window.py        收获期判定
  configs/                        配置文件
    paths.py                      ProjectPaths 统一路径管理（换研究区只换 --config）
    default.yaml                  默认配置（当前为团林铺 2026）
    tuanlinpu_2026_06.yaml        团林铺研究区配置
    caobuhu.yaml                  草埠湖研究区配置
    dangyang_2023_08.yaml         当阳研究区配置
    class_mapping.yaml            类别映射
    harvest_window.yaml           收获窗口配置
  docs/
    API_TEST_GUIDE.md             API 测试指南
    DATA_LAYOUT.md                数据目录布局说明
  models/
    crop_classification_classifier.joblib   已训练分类模型
    crop_classification_model_info.json     模型元数据
  data/                           运行数据（gitignore 排除）
    input/                        输入数据（AOI、地块、样本）
    exported/                     中间产物（特征栈、训练数据）
    output/                       最终输出（分类、长势、病虫害、估产）
    source/                       源影像缓存
    uploads/                      API 上传文件
  run_api.py                      API 服务启动入口
  requirements.txt                Python 依赖
```

## 环境安装

Python 3.10 / 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 配置驱动设计

项目的核心设计理念：**换研究区只换 `--config`，不改代码**。

`configs/paths.py` 中的 `ProjectPaths` 类从 YAML 配置文件自动推导所有输入/输出路径，
文件名统一带 `{prefix}` 前缀（取自配置文件名），实现多研究区数据隔离。

```powershell
# 切换研究区只需指定不同配置文件
python -m pipeline.crop_classification.03_predict_classify --config configs/tuanlinpu_2026_06.yaml
python -m pipeline.crop_classification.06_parcel_majority  --config configs/dangyang_2023_08.yaml
```

配置文件关键字段：

```yaml
project:
  name: CropClassifier-Tuanlinpu-2026-05-06
  target_crs: EPSG:4326
  geometry: data/input/aoi/tuanlinpu_aoi.shp
  parcels: data/input/parcels/tuanlinpu_parcel.shp

season:
  year: 2026
  start_date: "2026-05-01"
  end_date: "2026-06-30"
```

## 离线流水线

### 通用数据流

```text
配置文件 (*.yaml)
      ↓
ProjectPaths 推导所有路径
      ↓
data_sources/  下载/检索影像 → 构建特征栈 → data/exported/feature_stack/
      ↓
pipeline/  消费特征栈 + 模型 → 输出结果 → data/output/
```

### 作物分类

```powershell
# 1. 准备训练样本
python -m pipeline.crop_classification.01_prepare_samples

# 2. 训练模型
python -m pipeline.crop_classification.02_train_rf

# 3. 像素级分类
python -m pipeline.crop_classification.03_predict_classify

# 4. 后处理（过滤低置信度、筛除小斑块）
python -m pipeline.crop_classification.04_postprocess

# 5. 精度评估（需要独立验证数据）
python -m pipeline.crop_classification.05_accuracy_eval --reference-csv validation_points.csv

# 6. 地块级多数投票
python -m pipeline.crop_classification.06_parcel_majority
```

### 长势监测

```powershell
# 1. 准备多年同期 NDVI 基准
python -m pipeline.growth_monitoring.01_prepare_baseline

# 2. 像元级 Z-Score 分级
python -m pipeline.growth_monitoring.02_pixel_zscore

# 3. 地块级长势定级
python -m pipeline.growth_monitoring.03_parcel_grade
```

### 病虫害检测

```powershell
# 1. 准备多期历史对比数据（Copernicus 下载 + 特征栈构建）
python -m pipeline.pest_detect.01_prepare_inputs \
  --current-start 2026-06-01 --current-end 2026-06-30 \
  --baseline-start-year 2023 --baseline-end-year 2025

# 2. 像元级胁迫异常评分
python -m pipeline.pest_detect.02_pixel_stress_score

# 3. 地块级病虫害等级
python -m pipeline.pest_detect.03_parcel_pest_stress_grade
```

### 产量估算

```powershell
python -m pipeline.yield_estimation.01_yield_estimation
python -m pipeline.yield_estimation.02_yield_summary
```

## 数据源

项目支持两种 Sentinel-2 数据获取方式：

| 数据源 | 目录 | 认证 | 适用场景 |
|---|---|---|---|
| Copernicus Data Space | `data_sources/copernicus/` | 需要账号密码 | 全球范围，SAFE 格式 |
| AWS Element84 | `data_sources/aws_element84/` | 无需认证 | 公开数据，COG 格式 |

Copernicus 认证配置（任选其一）：

```text
1. 环境变量 COPERNICUS_USERNAME / COPERNICUS_PASSWORD
2. ~/.copernicus/credentials.json 中的 username/password
3. 配置文件 copernicus.username / copernicus.password
```

## 启动 API 服务

```powershell
python run_api.py
```

访问：

```text
http://127.0.0.1:8000        Web 门户
http://127.0.0.1:8000/docs   Swagger 文档
```

### 核心 API

```text
GET  /api/health                              健康检查
GET  /classes                                 类别映射
GET  /artifacts                               产物列表
GET  /artifacts/{name}/download               下载产物

POST /api/data/upload                         上传影像
POST /api/data/upload-parcels                 上传地块 Shapefile ZIP

POST /api/infer/start                         启动作物识别
GET  /api/infer/status/{task_id}              查询状态
GET  /api/infer/download/{task_id}            下载结果
WS   /ws/infer/{task_id}                      WebSocket 状态推送

POST /api/yield/estimate                      启动产量估算
GET  /api/yield/status/{yield_task_id}        查询估产状态
GET  /api/yield/download/{yield_task_id}      下载估产结果

POST /api/growth/start                        启动长势监测
GET  /api/growth/status/{task_id}             查询长势状态
GET  /api/growth/tasks                        列出长势任务
GET  /api/growth/download/{task_id}           下载长势结果

POST /api/pest/start                          启动病虫害检测
GET  /api/pest/status/{task_id}               查询病虫害状态
GET  /api/pest/tasks                          列出病虫害任务
GET  /api/pest/download/{task_id}             下载病虫害结果
```

## 数据说明

仓库不包含遥感影像、训练数据和运行输出。本地数据目录：

```text
data/
  input/         AOI、地块矢量、样本、本地 GeoTIFF
  exported/      特征栈、训练数据、场景清单等中间产物
  output/        分类结果、长势等级、病虫害等级、估产报告
  source/        下载的源影像缓存
  uploads/       API 上传文件
```

详见 [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)。

## 注意事项

- 当前服务用于本地或局域网运行，非公网生产部署。
- 模型精度依赖训练样本质量、影像时相覆盖和区域代表性。
- Copernicus 数据下载需要能访问 `identity.dataspace.copernicus.eu`（国内网络可能需要代理）。
- 估产模型基于经验回归关系，建议结合区域实测数据校准。
