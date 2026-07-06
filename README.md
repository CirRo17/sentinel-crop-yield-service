# Sentinel Crop Yield Service

基于遥感影像的农作物类型识别与估产服务。项目包含离线处理流水线、FastAPI 推理服务、作物类型分类模型和基于植被指数的估产模块。

## 功能概览

- 农作物类型识别：支持水稻、小麦、玉米、油菜和其他类别。
- 遥感特征构建：从 Sentinel 或本地多波段 GeoTIFF 构建 NDVI、EVI、NDWI、LAI 等特征。
- API 推理服务：支持影像上传、分类推理、任务状态查询、结果下载。
- 地块级统计：可结合 Shapefile 输出地块级多数投票结果。
- 产量估算：基于 NDVI/LAI 与经验回归模型估算水稻、小麦、玉米产量。

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
sentinel-crop-yield-service/
  crop_classifier_api/      FastAPI 服务与接口逻辑
  crop_classifier_core/     公共配置、特征计算、数据结构
  crop_yield/               估产相关模块
  pipeline/                 离线处理流水线脚本
  configs/                  项目配置和分类映射
  docs/                     API 测试说明
  examples/                 示例请求和 AOI
  models/                   已训练模型和模型说明
  data/README.md            数据目录说明
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

```text
GET  /api/health
GET  /classes
POST /api/data/upload
POST /api/data/upload-parcels
POST /api/infer/start
GET  /api/infer/status/{task_id}
GET  /api/infer/download/{task_id}
POST /api/yield/estimate
GET  /api/yield/status/{yield_task_id}
GET  /api/yield/download/{yield_task_id}
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

启动分类推理：

```powershell
$start = curl.exe -X POST "$base/api/infer/start" `
  -H "Content-Type: application/json" `
  -d "{`"file_id`": `"$fileId`"}"

$taskId = ($start | ConvertFrom-Json).task_id
```

查询状态并下载结果：

```powershell
curl.exe "$base/api/infer/status/$taskId"
curl.exe -o "${taskId}_classification.tif" "$base/api/infer/download/${taskId}?format=classification"
curl.exe -o "${taskId}_confidence.tif" "$base/api/infer/download/${taskId}?format=confidence"
```

启动估产：

```powershell
$yieldStart = curl.exe -X POST "$base/api/yield/estimate" `
  -H "Content-Type: application/json" `
  -d "{`"task_id`": `"$taskId`", `"index`": `"ndvi`"}"

$yieldTaskId = ($yieldStart | ConvertFrom-Json).yield_task_id
curl.exe "$base/api/yield/status/$yieldTaskId"
```

## 离线流水线

`pipeline/` 中的脚本按编号组织：

```text
01_preprocess_s2.py                  检索 Sentinel-2 场景
02_preprocess_s1.py                  检索 Sentinel-1 场景
03a_cache_source_rasters.py          缓存或落盘源影像
03b_build_features.py                从 Sentinel assets 构建特征栈
03c_build_features_from_multiband.py 从本地多波段 GeoTIFF 构建特征栈
04_prepare_samples.py                准备训练样本
05_train_rf.py                       训练 Random Forest 分类模型
06_predict_classify.py               生成分类图和置信度图
07_postprocess.py                    后处理分类结果
08_accuracy_eval.py                  精度评价
09_parcel_majority.py                地块级多数投票统计
10_yield_estimation.py               作物产量估算
11_yield_summary.py                  产量汇总报告
```

本地多波段 GeoTIFF 构建特征栈示例：

```powershell
python -m pipeline.03c_build_features_from_multiband `
  --input data\input\local_multiband.tif `
  --band-map blue=1,green=2,red=3,rededge=4,nir=5,swir=6 `
  --reflectance-scale 10000 `
  --output data\exported\feature_stack_local.tif `
  --metadata data\exported\feature_stack_local_metadata.json
```

## 数据说明

仓库不包含原始遥感影像、训练栅格、运行输出和上传文件。使用前请在本地准备数据目录：

```text
data/input/       输入 AOI、样本、GeoTIFF 等
data/exported/    中间特征栈和训练数据
data/output/      分类、估产、报告等输出
data/uploads/     API 上传文件
```

详细说明见 [data/README.md](data/README.md)。

## 注意事项

- 当前服务默认用于本地或局域网运行，不是公网生产部署。
- 模型精度依赖训练样本、影像质量、时相覆盖和区域差异。
- 估产模型基于经验回归关系，实际使用时建议结合区域实测数据校准。
- 如果要长期部署，建议进一步封装为 Windows 服务、Docker 服务或云端 API。
