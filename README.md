# Sentinel Crop Service

基于遥感影像的农作物类型识别与估产服务。项目包含离线处理流水线、FastAPI 推理服务、作物类型分类模型、产量估算和基于植被指数的估产等模块。

## 功能概览

- **农作物类型识别**：支持水稻、小麦、玉米、油菜和其他类别，基于 Random Forest 像素级分类。
- **遥感特征构建**：从 Sentinel-2 L2A 或本地多波段 GeoTIFF 构建 NDVI、EVI、NDWI、NDRE、NBR 等光谱指数。
- **API 推理服务**：支持单景/多景影像上传、多时相自动分组合成、分类推理、任务状态查询（含 WebSocket 推送）、结果下载。
- **地块级统计**：可结合 Shapefile 输出地块级多数投票结果。
- **产量估算**：基于 NDVI/LAI 与经验回归模型估算水稻、小麦、玉米产量，支持多种函数类型。


分类编码：

```text
0  Others / 其他
1  Rice / 水稻
2  Wheat / 小麦
3  Maize / 玉米
4  Rapeseed / 油菜
```

## 项目结构

```text
SentinelCropService/
  crop_domain/               作物领域公共定义
    labels.py                作物类别编码、标签、训练标签到目标标签的映射
  crop_service_api/          FastAPI 服务与接口逻辑
    api.py                   主 API 应用（v0.6.0），涵盖分类推理、估产、产物管理
    parcel_stats.py          地块多数投票兼容导入入口
    home.html                Web 门户页面模板
  data_sources/              数据源适配
    common/                  数据源通用配置、场景数据结构、传感器能力约定、STAC 工具
    sentinel/                Sentinel 场景检索、本地缓存、特征栈构建
    local_raster/            本地 GeoTIFF / COG 数据源适配预留
    uav/                     无人机多光谱数据源适配预留
  image_core/                本地影像处理
    spectral.py              标准光谱指数（NDVI、EVI、NDWI、NDRE、NBR）
    feature_schema.py        特征栈 schema 校验、波段名提取、多时相特征匹配
    build_features_from_multiband.py  从本地多波段 GeoTIFF 构建特征栈
  pipeline/                  功能型业务流程
    crop_classification/     作物分类：样本准备、RF 训练、预测、后处理、精度评价、地块多数投票
    yield_estimation/        基于分类结果的产量估算与汇总报告
    growth_monitoring/       长势监测功能预留
    pest_detect/             病虫害识别功能预留
  configs/                   项目配置（默认参数、分类映射、各研究区配置）
  docs/                      API 测试指南
  models/                    已训练模型（crop_classifier.joblib）和模型元数据
  data/README.md             数据目录说明
  run_api.py                 服务启动入口
  requirements.txt           Python 依赖
```

以下目录通常包含本地大数据、运行结果或临时文件，已通过 `.gitignore` 排除，不会上传到 GitHub：

```text
data/input/
data/output/
data/exported/
data/source/
data/uploads/
CropGlobe_data/
tiff_Files/
shp_Files/
.venv/
```

## 环境安装

建议使用 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

部分依赖如 `geopandas`、`rasterio` 在 Windows 上可能需要预编译 wheel 或 conda 环境。

## 启动 API 服务

```powershell
python run_api.py
```

服务启动后访问：

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

局域网访问时，将 `127.0.0.1` 替换为运行服务电脑的局域网 IP。

## 核心 API

### 系统与参考

```text
GET  /api/health                             健康检查
GET  /classes                                 获取类别映射
GET  /artifacts                               列出可用产物（模型、配置、输出文件等）
GET  /artifacts/{name}                        获取产物信息
GET  /artifacts/{name}/download               下载产物
```

### 数据上传

```text
POST /api/data/upload                         上传多光谱 GeoTIFF 影像
POST /api/data/upload-parcels                 上传地块 Shapefile ZIP
```

### 分类推理

```text
POST /api/infer/start                         启动分类推理任务
GET  /api/infer/status/{task_id}              查询任务状态
GET  /api/infer/tasks                         列出所有分类推理任务
GET  /api/infer/download/{task_id}            下载结果（classification/confidence/metadata/shp）
WS   /ws/infer/{task_id}                      WebSocket 实时推送任务状态
```

### 产量估算

```text
POST /api/yield/estimate                      启动产量估算（依赖已完成分类推理任务）
GET  /api/yield/status/{yield_task_id}        查询估产状态
GET  /api/yield/tasks                         列出所有估产任务
GET  /api/yield/download/{yield_task_id}      下载估产元数据
```

### 历史产物与报告

```text
GET  /api-predictions/{job_id}/classification  下载分类图
GET  /api-predictions/{job_id}/confidence      下载置信度图
GET  /api-predictions/{job_id}/shp             下载地块级 Shapefile ZIP
GET  /api-predictions/{job_id}/metadata        查看预测元数据
GET  /reports/prediction                       查看已有预测报告
GET  /reports/postprocess                      查看已有后处理报告
GET  /reports/accuracy                         查看已有精度报告
GET  /maps/summary                             查看分类图面积统计
```

更多测试方式见 [docs/API_TEST_GUIDE.md](docs/API_TEST_GUIDE.md)。

## API 调用示例

上传影像：

```powershell
$base = "http://127.0.0.1:8000"
$file = "C:\path\to\feature_stack.tif"

$upload = curl.exe -X POST "$base/api/data/upload" -F "file=@$file"
$fileId = ($upload | ConvertFrom-Json).file_id
```

启动分类推理（支持多文件上传，服务端按文件名中的日期自动分组为多时相）：

```powershell
$start = curl.exe -X POST "$base/api/infer/start" `
  -H "Content-Type: application/json" `
  -d "{`"file_ids`": [`"$fileId`"], `"reflectance_scale`": 1.0, `"top_k`": 3}"

$taskId = ($start | ConvertFrom-Json).task_id
```

查询状态并下载结果：

```powershell
curl.exe "$base/api/infer/status/$taskId"
curl.exe -o "${taskId}_classification.tif" "$base/api/infer/download/${taskId}?format=classification"
curl.exe -o "${taskId}_confidence.tif" "$base/api/infer/download/${taskId}?format=confidence"
```

启动估产（依赖已完成的分类推理任务）：

```powershell
$yieldStart = curl.exe -X POST "$base/api/yield/estimate" `
  -H "Content-Type: application/json" `
  -d "{`"task_id`": `"$taskId`", `"index`": `"ndvi`"}"

$yieldTaskId = ($yieldStart | ConvertFrom-Json).yield_task_id
curl.exe "$base/api/yield/status/$yieldTaskId"
```

## 离线流水线

项目按以下原则组织离线流程：

```text
外部数据源 / 本地上传
        ↓
data_sources/ 下载、导入或登记为本地数据
        ↓
image_core/ 或 data_sources/sentinel/ 构建标准特征栈
        ↓
pipeline/ 执行业务功能（分类 / 估产 / 长势 / 病虫害）
```

功能型 pipeline 只消费本地影像、特征栈、样本、模型和配置；不直接依赖公开平台在线读取。

### 脚本职责

#### 数据源通用层

```text
data_sources/common/config.py                 数据源通用配置（STAC 地址、默认检索限制）
data_sources/common/schemas.py                场景检索请求/响应数据结构
data_sources/common/stac.py                   STAC 场景检索通用工具
data_sources/common/sensor_contracts.py       传感器能力约定（统一语义波段、指数依赖）
```

#### Sentinel 数据源

```text
data_sources/sentinel/config.py               Sentinel-2 波段资产名、缩放系数、云掩膜配置
data_sources/sentinel/01_preprocess_s2.py      检索 Sentinel-2 场景清单
data_sources/sentinel/02_preprocess_s1.py      检索 Sentinel-1 场景清单
data_sources/sentinel/aws_open_data.py         Sentinel AWS Open Data / STAC 工具
data_sources/sentinel/cache_source_rasters.py  缓存或落盘 Sentinel 源影像
data_sources/sentinel/build_features.py        从 Sentinel assets 构建特征栈
```

#### 影像处理 / 本地多波段输入

```text
image_core/spectral.py                        NDVI、EVI、NDWI、NDRE、NBR 标准光谱指数
image_core/feature_schema.py                  特征栈 band 名称校验与模型特征顺序匹配
image_core/build_features_from_multiband.py   从本地多波段 GeoTIFF 构建特征栈
```

#### 作物领域公共定义

```text
crop_domain/labels.py                         作物类别编码、标签、训练标签到目标标签映射
```

#### 作物分类

```text
pipeline/crop_classification/model_defaults.py  历史模型路径和默认特征名配置
pipeline/crop_classification/01_prepare_samples.py  准备训练样本
pipeline/crop_classification/02_train_rf.py         训练 Random Forest 分类模型
pipeline/crop_classification/03_predict_classify.py 生成分类图和置信度图
pipeline/crop_classification/04_postprocess.py      后处理分类结果
pipeline/crop_classification/05_accuracy_eval.py    精度评价
pipeline/crop_classification/06_parcel_majority.py  地块级多数投票统计
```

#### 产量估算（基于分类结果）

```text
pipeline/yield_estimation/01_yield_estimation.py  逐作物产量估算与栅格输出
pipeline/yield_estimation/02_yield_summary.py     产量汇总 JSON 与 HTML 报告
```

### Sentinel 路线示例

```powershell
python -m data_sources.sentinel.01_preprocess_s2
python -m data_sources.sentinel.02_preprocess_s1

python -m data_sources.sentinel.cache_source_rasters `
  --materialize-local-assets

python -m data_sources.sentinel.build_features `
  --s2-manifest data\exported\sentinel2_scenes_local.json `
  --s1-manifest data\exported\sentinel1_scenes_local.json `
  --include-s1 `
  --output data\exported\feature_stack.tif `
  --metadata data\exported\feature_stack_metadata.json
```

### 本地多波段 GeoTIFF 路线示例

```powershell
python -m image_core.build_features_from_multiband `
  --input data\input\local_multiband.tif `
  --band-map blue=1,green=2,red=3,rededge=4,nir=5,swir=6 `
  --reflectance-scale 10000 `
  --output data\exported\feature_stack_local.tif `
  --metadata data\exported\feature_stack_local_metadata.json
```

### 作物分类路线示例

```powershell
python -m pipeline.crop_classification.01_prepare_samples `
  --feature-stack data\exported\feature_stack.tif `
  --metadata data\exported\feature_stack_metadata.json

python -m pipeline.crop_classification.02_train_rf

python -m pipeline.crop_classification.03_predict_classify `
  --feature-stack data\exported\feature_stack.tif `
  --metadata data\exported\feature_stack_metadata.json

python -m pipeline.crop_classification.04_postprocess
```

### 产量估算路线示例（基于分类结果）

```powershell
python -m pipeline.yield_estimation.01_yield_estimation `
  --classification data\output\crop_classification_clean.tif `
  --feature-stack data\exported\feature_stack.tif `
  --metadata data\exported\feature_stack_metadata.json `
  --index ndvi

python -m pipeline.yield_estimation.02_yield_summary `
  --yield-stats data\output\yield_stats.json
```

## 多时相支持

API 推理支持多时相分类。上传多个文件时，服务端按文件名中的日期（如 `202504_xxx.tif`、`2025-04_xxx.tif`、`xxx_20250415_xxx.tif`）自动分组为不同时相：

- 同组内多景影像先做中值合成
- 各组之间按时间顺序映射为 `t1`、`t2`…时相标签
- 模型特征（如 `t1_ndvi`、`t2_ndvi`）从对应时相的特征栈中按名匹配

对于已预构建的多时相特征栈 GeoTIFF（band description 包含 `t1_blue`、`t2_ndvi` 等），上传单文件即可直接推理。

## 数据说明

仓库不包含原始遥感影像、训练栅格、运行输出和上传文件。使用前请在本地准备数据目录：

```text
data/input/       输入 AOI、样本、专题图、本地 GeoTIFF 等
data/source/      下载或落库后的本地源影像
data/exported/    场景清单、特征栈、训练数据等中间产物
data/output/      分类、估产、地块统计、报告等输出
data/uploads/     API 上传文件
```

详细说明见 [data/README.md](data/README.md)。

## 注意事项

- 当前服务默认用于本地或局域网运行，不是公网生产部署。
- 模型精度依赖训练样本、影像质量、时相覆盖和区域差异。
- 估产模型基于经验回归关系，实际使用时建议结合区域实测数据校准。
- 如果要长期部署，建议进一步封装为 Windows 服务、Docker 服务或云端 API。
