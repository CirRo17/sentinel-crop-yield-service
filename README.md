# 作物分类与估产服务

本项目提供端到端的农作物遥感监测能力，包含三部分：

- **离线处理流程**：基于 Sentinel 影像构建特征、训练模型、生成分类图、估算产量、后处理和精度评价。
- **Web API 服务**：使用 FastAPI 对外提供影像上传、分类推理、产量估算、状态查询和结果下载接口。
- **估产模型**：基于 NDVI/LAI 植被指数，对水稻、小麦、玉米进行像素级产量回归与不确定性评估。

当前分类模型输出类别如下：

```text
0   Others / 非耕地、其他作物、无效、不确定
1   Rice / 水稻
2   Wheat / 小麦
3   Maize / 玉米
4   Rapeseed / 油菜
```

估产模型覆盖 3 种主要作物（水稻、小麦、玉米），支持多种回归函数（二次多项式、线性、指数、幂、对数）。

## 项目结构

```text
SentinelCropService/
  crop_classifier_api/        # FastAPI 服务入口和展示页
  crop_classifier_core/       # 公共配置、类别映射、光谱指数、估产模型
  crop_yield/                 # 估产模块（独立可运行）
  pipeline/                   # 离线处理流程脚本（01~11）
  configs/                    # 配置文件和类别编码表
  models/                     # 已训练模型和模型说明
  docs/                       # 接口测试文档
  examples/                   # 示例请求和示例 AOI
  data/
    input/                    # AOI、样例影像等输入数据
    exported/                 # 特征栈、训练数据、场景清单等中间成果
    output/                   # 分类图、产量图、报告和后处理结果
    uploads/                  # API 上传影像的运行时目录
```

当前仓库内置了草埠湖、当阳等离线案例配置，主要用于本地调试和复现实验：

```text
configs/default.yaml
configs/caobuhu.yaml
data/input/aoi_caobuhu.geojson
```

当阳 2023 年 8 月是独立案例配置：

```text
configs/dangyang_2023_08.yaml
data/input/aoi_dangyang.geojson
```

## 启动 API 服务

在 PowerShell 中执行：

```powershell
cd D:\CirRou\CropClassifier\SentinelCropService
.\.venv\Scripts\python.exe run_api.py
```

看到下面这行说明服务启动成功：

```text
Uvicorn running on http://0.0.0.0:8000
```

启动服务的 PowerShell 窗口需要保持打开。关闭窗口、按 `Ctrl+C`、电脑休眠或断网，服务都会停止。

本机访问：

```text
http://127.0.0.1:8000
```

同一局域网内其他电脑访问时，使用服务机器的局域网 IP，例如：

```text
http://192.168.110.53:8000
```

Swagger 接口文档：

```text
http://192.168.110.53:8000/docs
```

## API 主流程

当前 API 不需要登录认证，调用方可以直接请求接口。

主链路如下：

```text
1. 上传影像                       → file_id
2. 上传地块 Shapefile ZIP（可选）  → parcel_file_id
3. 发起分类推理                   → task_id
4. 查询任务状态
5. 下载分类图 / 地块级 Shapefile
6. 发起估产                       → yield_task_id
7. 查询估产状态并下载结果
```

核心接口：

```text
GET  /api/health                         服务健康检查
GET  /classes                            查看分类编码表
POST /api/data/upload                    上传 GeoTIFF 影像
POST /api/data/upload-parcels            上传地块 Shapefile ZIP
POST /api/infer/start                    发起分类推理任务（波段自动识别）
GET  /api/infer/status/{task_id}         查询任务状态和主要类别结果
GET  /api/infer/download/{task_id}       下载分类图、置信度图、地块级 Shapefile 或元数据
POST /api/yield/estimate                 发起估产任务
GET  /api/yield/status/{yield_task_id}   查询估产状态与结果
GET  /api/yield/download/{yield_task_id} 下载估产元数据 JSON
```

辅助接口：

```text
GET /api/infer/tasks
GET /api/yield/tasks
GET /api-predictions/{job_id}/classification
GET /api-predictions/{job_id}/confidence
GET /api-predictions/{job_id}/shp
GET /api-predictions/{job_id}/metadata
GET /reports/prediction
GET /reports/postprocess
GET /reports/accuracy
GET /maps/summary
GET /artifacts
GET /artifacts/{name}
GET /artifacts/{name}/download
```

## 调用示例

### cURL

```bash
BASE="http://192.168.110.53:8000/api"

# 1. 上传多时相特征栈（pipeline 03b/03d 输出的多波段 GeoTIFF）
curl -X POST $BASE/data/upload -F "file=@feature_stack.tif"

# 2. 上传地块数据（可选，ZIP 格式含 .shp/.shx/.dbf）
curl -X POST $BASE/data/upload-parcels -F "file=@your_parcels.zip"

# 3. 启动推理（波段自动识别）
curl -X POST $BASE/infer/start \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id", "parcel_file_id": "your-parcel-id"}'

# 4. 查询状态
curl $BASE/infer/status/{task_id}

# 5. 下载结果
curl "$BASE/infer/download/{task_id}?format=classification" -o classification.tif
curl "$BASE/infer/download/{task_id}?format=shp" -o parcels.zip

# 6. 估产
curl -X POST $BASE/yield/estimate \
  -H "Content-Type: application/json" \
  -d '{"task_id": "{task_id}", "index": "ndvi"}'

# 7. 查询估产状态
curl $BASE/yield/status/yield_{task_id}

# 8. 下载估产结果
curl "$BASE/yield/download/yield_{task_id}?format=metadata" -o yield.json
```

### PowerShell

输入通常是 pipeline 03b/03d 输出的多时相特征栈（含 t1_ndvi、t2_ndvi 等多波段）。若只有单景多光谱影像也可以直接上传，服务会自动构建指数，但分类精度会受限于单时相信息。

```powershell
$base = "http://192.168.110.53:8000"
$file = "C:\path\to\feature_stack.tif"

# 上传多时相特征栈
$upload = curl.exe -X POST "$base/api/data/upload" -F "file=@$file"
$fileId = ($upload | ConvertFrom-Json).file_id

# 上传地块（可选）
$parcelZip = "C:\path\to\parcels.zip"
$parcelUpload = curl.exe -X POST "$base/api/data/upload-parcels" -F "file=@$parcelZip"
$parcelFileId = ($parcelUpload | ConvertFrom-Json).parcel_file_id

# 启动推理（波段自动识别，只需 file_id）
$start = curl.exe -X POST "$base/api/infer/start" `
  -H "Content-Type: application/json" `
  -d "{`"file_id`": `"$fileId`", `"parcel_file_id`": `"$parcelFileId`"}"

$taskId = ($start | ConvertFrom-Json).task_id

# 查询状态
curl.exe "$base/api/infer/status/$taskId"

# 下载结果
curl.exe -o "${taskId}_classification.tif" "$base/api/infer/download/${taskId}?format=classification"
curl.exe -o "${taskId}_parcels.zip" "$base/api/infer/download/${taskId}?format=shp"

# 估产
$yieldStart = curl.exe -X POST "$base/api/yield/estimate" `
  -H "Content-Type: application/json" `
  -d "{`"task_id`": `"$taskId`", `"index`": `"ndvi`"}"
$yieldTaskId = ($yieldStart | ConvertFrom-Json).yield_task_id

# 查询估产状态
curl.exe "$base/api/yield/status/$yieldTaskId"

# 下载估产结果
curl.exe -o "${yieldTaskId}_yield.json" "$base/api/yield/download/${yieldTaskId}?format=metadata"
```

如果不传 `parcel_file_id`，任务只生成分类 TIFF、置信度 TIFF 和元数据，不生成 `format=shp`。

### 输入数据说明

作物分类依赖长时序植被指数（通常跨越 4~8 月），单个 Sentinel-2 景无法同时满足空间覆盖和时间跨度的要求：

- **空间覆盖**：研究区可能跨越多个 S2 轨道，同一时段需要多景影像才能拼满整个 AOI。
- **时间跨度**：作物物候区分需要数月内的多期影像（如 4 月拔节期、7 月抽穗期、8 月成熟期）。

管线 01~03b 负责处理这个复杂度：

```text
原始数据                           管线处理                     API 输入
─────────────────────────────────────────────────────────────────────
4月: scene_A, scene_B    ─┐
5月: scene_C, scene_D     │  03a 缓存 + 对齐
7月: scene_E, scene_F     │  03b 按月做中值合成       每时相一个完整
8月: scene_G              ─┘  + 计算光谱指数           GeoTIFF → API
```

API 接收的是各时相合成后的影像（每时相一个 GeoTIFF，完整覆盖研究区），通过 `file_ids` 列表传入。服务端自动按 `time_labels` 构建多时相特征栈后推理。

如果只有单期影像，也可以直接上传一个多波段 GeoTIFF（Sentinel-2、Landsat、无人机等），服务端现场做指数构建。分类精度受限于单时相信息，适合快速测试。

### 波段说明

上传多时相特征栈时，波段名已内置在文件中，无需手动指定。，波段编号由服务端自动识别，不需要手动指定。非标影像（如 Landsat、无人机多光谱）可通过以下参数手动指定：

```json
{
  "blue_band": 1,
  "green_band": 2,
  "red_band": 3,
  "rededge_band": 4,
  "nir_band": 5,
  "swir_band": 0,
  "reflectance_scale": 1.0
}
```

`reflectance_scale`：Sentinel-2 用 10000（整型反射率），0~1 浮点反射率用 1.0。

## 离线处理流程

`pipeline/` 目录下的编号脚本对应完整处理流程：

```text
01_preprocess_s2.py                检索 Sentinel-2 场景清单
02_preprocess_s1.py                检索 Sentinel-1 场景清单
03a_cache_source_rasters.py        缓存或落库 AOI 对齐后的源影像
03b_build_features.py              从 AWS/本地单波段 assets 构建标准特征栈
03d_build_features_from_multiband.py 从本地多波段 GeoTIFF 构建标准特征栈
04_prepare_samples.py              从水稻/玉米/小麦/油菜专项图准备训练样本
05_train_rf.py                     训练 Random Forest 分类模型
06_predict_classify.py             生成像素级分类图和置信度图
07_postprocess.py                  后处理分类结果
08_accuracy_eval.py                精度评价
09_parcel_majority.py              将分类图按地块多数投票写入 Shapefile
10_yield_estimation.py             基于分类图和植被指数估算作物产量
11_yield_summary.py                汇总产量统计，生成地块级产量报告
```

### 估产流程（Step 10~11）

Step 10 读取分类栅格和特征栈，对每种作物逐像素计算产量：

```powershell
# 基础用法（自动选择 NDVI 最高的时相）
python pipeline/10_yield_estimation.py `
  --classification data/output/crop_classification_clean.tif `
  --feature-stack data/exported/feature_stack.tif `
  --metadata data/exported/feature_stack_metadata.json

# 指定时相和指数类型
python pipeline/10_yield_estimation.py `
  --timepoint t2 `
  --index lai
```

Step 11 汇总产量统计，支持地块级聚合：

```powershell
# 仅汇总报告
python pipeline/11_yield_summary.py `
  --yield-stats data/output/yield_stats.json

# 地块级汇总
python pipeline/11_yield_summary.py `
  --yield-raster data/output/yield_all.tif `
  --classification data/output/crop_classification_clean.tif `
  --parcels shp_Files/Caobuhu_Parcel_shp/草埠湖镇修改.shp
```

输出文件：

```text
data/output/yield_rice.tif          # 水稻产量栅格 (kg/ha)
data/output/yield_wheat.tif         # 小麦产量栅格 (kg/ha)
data/output/yield_maize.tif         # 玉米产量栅格 (kg/ha)
data/output/yield_all.tif           # 综合产量栅格
data/output/yield_stats.json        # 产量统计（面积/单产/总产/不确定性）
data/output/yield_summary.json      # 汇总报告
data/output/yield_report.html       # HTML 报告
```

常用命令示例：

草埠湖离线示例流程：

```powershell
python -m pipeline.01_preprocess_s2 `
  --config configs\caobuhu.yaml `
  --geometry data\input\aoi_caobuhu.geojson `
  --output data\exported\sentinel2_scenes.json `
  --limit 30

python -m pipeline.02_preprocess_s1 `
  --config configs\caobuhu.yaml `
  --geometry data\input\aoi_caobuhu.geojson `
  --output data\exported\sentinel1_scenes.json `
  --limit 30

python -m pipeline.03a_cache_source_rasters `
  --s2-manifest data\exported\sentinel2_scenes.json `
  --s1-manifest data\exported\sentinel1_scenes.json `
  --include-s1 `
  --materialize-local-assets `
  --local-source-dir data\source\aws_local `
  --local-s2-manifest data\exported\sentinel2_scenes_local.json `
  --local-s1-manifest data\exported\sentinel1_scenes_local.json `
  --timepoints 2025-07

python -m pipeline.03b_build_features `
  --s2-manifest data\exported\sentinel2_scenes_local.json `
  --s1-manifest data\exported\sentinel1_scenes_local.json `
  --include-s1 `
  --timepoints 2025-07 `
  --output data\exported\feature_stack_2025_07_full.tif `
  --metadata data\exported\feature_stack_2025_07_full_metadata.json

python -m pipeline.04_prepare_samples `
  --feature-stack data\exported\feature_stack_2025_07_s2_test.tif `
  --metadata data\exported\feature_stack_2025_07_s2_test_metadata.json `
  --rice-map data\input\lables\rice `
  --maize-map data\input\lables\maize `
  --wheat-map data\input\lables\wheat `
  --rapeseed-map data\input\lables\rapeseed `
  --sample-region data\input\aoi_caobuhu.geojson `
  --max-per-class 5000 `
  --erode-pixels 0 `
  --min-per-class 20

python -m pipeline.05_train_rf

python -m pipeline.06_predict_classify `
  --feature-stack data\exported\feature_stack_2025_07_full.tif `
  --metadata data\exported\feature_stack_2025_07_full_metadata.json
```

`03b_build_features` 默认使用稳定时相槽命名特征，例如 `t1_blue`、`t1_ndvi`。`--timepoints 2025-07`
只用于选择输入影像时间窗，日期不会进入模型特征名。只有复现实验旧模型时才使用
`--timepoint-name-mode label` 保留 `2025_07_blue` 这类历史命名。

本地多波段 GeoTIFF 可以走独立输入适配入口，最终同样输出标准 `t1_*` 特征栈：

```powershell
python -m pipeline.03d_build_features_from_multiband `
  --input data\input\local_multiband_t1.tif `
  --band-map blue=1,green=2,red=3,rededge=4,nir=5,swir=6 `
  --reflectance-scale 10000 `
  --output data\exported\feature_stack_local_t1.tif `
  --metadata data\exported\feature_stack_local_t1_metadata.json
```

多时相本地输入则重复 `--input`，并可显式指定稳定时相槽：

```powershell
python -m pipeline.03d_build_features_from_multiband `
  --input data\input\local_multiband_early.tif `
  --input data\input\local_multiband_late.tif `
  --time-slots t1 t2 `
  --band-map blue=1,green=2,red=3,rededge=4,nir=5,swir=6 `
  --reflectance-scale 10000 `
  --output data\exported\feature_stack_local_t1_t2.tif `
  --metadata data\exported\feature_stack_local_t1_t2_metadata.json
```

两条输入路线可以共存：AWS Sentinel-2 由 `03b` 读取单波段 assets，本地多波段影像由 `03d` 读取一张或多张 GeoTIFF。后续 `04/05/06` 和 API 只读取统一后的标准特征栈，不关心原始数据来源。

生产交付建议使用上面的 `03a --materialize-local-assets` 模式：AWS/Earth Search 只负责检索和首次落库，后续 `03b`、预测、后处理都读取本地 GeoTIFF，不再运行时访问 AWS COG。若只是研发快速试跑，也可以跳过 `--materialize-local-assets`，让 `03b` 直接按远程清单读取并使用 `data/exported/cache/*.npy` 缓存。

当阳 2023 年 8 月案例：

```powershell
python -m pipeline.01_preprocess_s2 `
  --config configs\dangyang_2023_08.yaml `
  --geometry data\input\aoi_dangyang.geojson `
  --output data\exported\sentinel2_dangyang_2023_08.json `
  --limit 30

python -m pipeline.03b_build_features `
  --s2-manifest data\exported\sentinel2_dangyang_2023_08_local.json `
  --timepoints 2023-08 `
  --resolution 100 `
  --output data\exported\feature_stack_dangyang_2023_08_100m.tif `
  --metadata data\exported\feature_stack_dangyang_2023_08_100m_metadata.json
```

地块级多数投票示例：

```powershell
python -m pipeline.09_parcel_majority `
  --classification data\output\crop_classification.tif `
  --parcels shp_Files\Caobuhu_Parcel_shp\草埠湖镇修改.shp `
  --output-shp data\output\parcel_postprocess\parcel_majority.shp `
  --output-zip data\output\parcel_postprocess\parcel_majority.zip `
  --summary data\output\parcel_postprocess\parcel_majority_summary.json `
  --include-all
```

## 产量模型说明

估产基于植被指数（NDVI 或 LAI）与产量的统计回归关系。默认使用二次多项式模型 `y = a·x² + b·x + c`。

作物模型参数：

| 作物 | a | b | c | RMSE (kg/ha) |
|---|---|---|---|---|
| 水稻 (Rice) | -10,128.83 | 13,475.88 | 4,032.55 | 380 |
| 小麦 (Wheat) | 92,997.92 | -114,110.35 | 38,604.30 | 520 |
| 玉米 (Maize) | 77,763.10 | -80,886.71 | 24,626.40 | 450 |

LAI 计算采用半经验幂函数 `LAI = k · CI^m`，其中 `CI = B8/B5 - 1`（叶绿素指数）。

支持的回归函数类型：`default`（二次多项式）、`linear`、`exponential`、`power`、`logarithmic`、`polynomial`。

## 当前已有成果

草埠湖像素级分类结果：

```text
data/output/crop_classification.tif
data/output/crop_confidence.tif
```

草埠湖地块级多数投票示例结果：

```text
data/output/parcel_postprocess/caobuhu_parcel_majority.shp
data/output/parcel_postprocess/caobuhu_parcel_majority.tif
data/output/parcel_postprocess/caobuhu_parcel_majority_summary.json
```

当阳 2023 年 8 月 100 米快速分类结果：

```text
data/output/dangyang_2023_08_100m_classification_v2.tif
data/output/dangyang_2023_08_100m_confidence_v2.tif
data/output/dangyang_2023_08_100m_prediction_info_v2.json
```

## 注意事项

- 当前 API 服务是本机启动的局域网服务，不是长期部署的公网服务。
- `127.0.0.1` 只代表调用者自己的电脑，发给别人时应使用服务机器的局域网 IP。
- `data/uploads/` 和 `data/output/api_predictions/` 属于 API 运行时目录，会随着测试调用逐渐增加文件。
- 当前模型精度依赖训练样本质量，输出图应结合业务场景和人工核查使用。
- 产量模型基于统计回归经验公式，系数可根据区域和品种调整。`crop_classifier_core/config.py` 中的 `CROP_MODELS` 和 `LAI_DEFAULTS` 可按需修改。
- 如果要长期稳定提供接口，建议后续部署为 Windows 服务、Docker 服务或服务器服务。
