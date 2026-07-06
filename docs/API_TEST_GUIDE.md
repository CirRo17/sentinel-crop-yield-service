# 农作物类别区分在线服务接口测试说明

本文档用于给对接方或测试人员快速验证当前 API 服务是否可用。

## 1. 服务地址

当前服务部署示例地址：

```text
http://192.168.110.53:8000
```

如果服务后续部署到其他机器，请把下面命令里的 `http://192.168.110.53:8000` 替换成实际地址。

Swagger 文档地址：

```text
http://192.168.110.53:8000/docs
```

注意：

- `127.0.0.1` 只表示调用者自己的电脑，不能发给别人用。
- `192.168.110.53` 是当前这台机器的局域网地址，只适用于同一局域网内测试。

## 2. 当前可交付接口

按调用流程排序如下：

| 接口 | 方法 | 用途 |
| --- | --- | --- |
| `/api/health` | GET | 检查服务是否在线 |
| `/classes` | GET | 查看分类编码表 |
| `/api/data/upload` | POST | 上传待分类的 GeoTIFF 影像 |
| `/api/data/upload-parcels` | POST | 可选，上传地块 Shapefile ZIP |
| `/api/infer/start` | POST | 发起一次分类推理任务 |
| `/api/infer/status/{task_id}` | GET | 查询任务状态和主要类别结果 |
| `/api/infer/download/{task_id}` | GET | 下载本次任务输出文件 |
| `/api/infer/tasks` | GET | 查看近期推理任务列表 |
| `/api-predictions/{job_id}/classification` | GET | 下载某次任务的分类图 |
| `/api-predictions/{job_id}/confidence` | GET | 下载某次任务的置信度图 |
| `/api-predictions/{job_id}/shp` | GET | 下载某次任务的地块级 Shapefile ZIP |
| `/api-predictions/{job_id}/metadata` | GET | 查看某次任务的元数据 |
| `/reports/prediction` | GET | 查看已有预测报告 |
| `/reports/postprocess` | GET | 查看已有后处理报告 |
| `/reports/accuracy` | GET | 查看已有精度报告 |
| `/maps/summary` | GET | 查看分类图面积统计 |

## 3. 推荐测试流程

建议按下面步骤测试主链路：

1. 先调 `/api/health`，确认服务在线
2. 上传一个 `.tif` 或 `.tiff` 影像到 `/api/data/upload`
3. 如需地块级 Shapefile，上传调用方自己研究区的地块 Shapefile ZIP 到 `/api/data/upload-parcels`
4. 用返回的 `file_id` 和可选 `parcel_file_id` 调 `/api/infer/start`
5. 轮询 `/api/infer/status/{task_id}`，完成后下载结果

## 4. 快速测试命令

以下示例使用 `curl`。Windows PowerShell、Git Bash、Linux、macOS 都可以参考。

### 4.1 健康检查

```bash
curl http://192.168.110.53:8000/api/health
```

期望返回示例：

```json
{
  "status": "ok",
  "service": "CropClassifier API",
  "version": "0.5.0"
}
```

### 4.2 查看分类编码表

```bash
curl http://192.168.110.53:8000/classes
```

当前类别语义如下：

| 编码 | 类别 |
| --- | --- |
| `0` | Others / 非耕地、其他作物、无效、不确定 |
| `1` | Rice |
| `2` | Wheat |
| `3` | Maize |
| `4` | Rapeseed |

### 4.3 上传影像

```bash
curl -X POST "http://192.168.110.53:8000/api/data/upload" \
  -F "file=@sample_upload_multispectral_5band.tif"
```

期望返回示例：

```json
{
  "file_id": "c2f7f61300904800b034bf0637d7b895",
  "filename": "sample_upload_multispectral_5band.tif",
  "size_bytes": 10584759
}
```

说明：

- 仅支持 `.tif` 或 `.tiff`
- 这里拿到的 `file_id` 是后续发起推理时要用的上传文件编号

### 4.3.1 可选：上传地块 Shapefile ZIP

如果需要输出地块级 Shapefile，需要上传调用方自己区域的地块 SHP 压缩包。ZIP 中必须包含同一套 Shapefile 的 `.shp`、`.shx`、`.dbf`，建议同时包含 `.prj` 和 `.cpg`。

```bash
curl -X POST "http://192.168.110.53:8000/api/data/upload-parcels" \
  -F "file=@parcels.zip"
```

返回示例：

```json
{
  "parcel_file_id": "9a0a7efc8eb5493e9d64de0c972f721a",
  "filename": "parcels.zip",
  "shapefile": "data\\uploads\\9a0a7efc8eb5493e9d64de0c972f721a_parcels\\parcels.shp",
  "size_bytes": 1234567
}
```

### 4.4 发起推理

方式 A：JSON 请求体

```bash
curl -X POST "http://192.168.110.53:8000/api/infer/start" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "c2f7f61300904800b034bf0637d7b895",
    "parcel_file_id": "9a0a7efc8eb5493e9d64de0c972f721a",
    "blue_band": 1,
    "green_band": 2,
    "red_band": 3,
    "rededge_band": 4,
    "nir_band": 5,
    "swir_band": 0,
    "reflectance_scale": 1.0,
    "top_k": 1
  }'
```

方式 B：表单字段

这个方式更适合 Windows PowerShell 下直接用 `curl`，不容易被 JSON 引号转义问题影响。

```bash
curl -X POST "http://192.168.110.53:8000/api/infer/start" \
  -F "file_id=c2f7f61300904800b034bf0637d7b895" \
  -F "parcel_file_id=9a0a7efc8eb5493e9d64de0c972f721a" \
  -F "blue_band=1" \
  -F "green_band=2" \
  -F "red_band=3" \
  -F "rededge_band=4" \
  -F "nir_band=5" \
  -F "swir_band=0" \
  -F "reflectance_scale=1.0" \
  -F "top_k=1"
```

期望返回示例：

```json
{
  "task_id": "20260625_170806_a4a6d4ee",
  "status": "queued"
}
```

参数说明：

- `file_id`：上传成功后返回的文件编号
- `parcel_file_id`：可选。上传地块 Shapefile ZIP 后返回；传入后才会生成 `format=shp`
- `blue_band` / `green_band` / `red_band` / `rededge_band` / `nir_band` / `swir_band`：影像波段编号，按 1 开始计数
- `swir_band = 0` 表示当前影像没有 SWIR 波段
- `reflectance_scale = 10000` 适用于整型反射率数据
- `top_k` 表示返回整幅影像主要类别结果的数量，常用 `1` 或 `5`
- 多时相模型建议上传已由离线流程构建好的特征栈 GeoTIFF，band description 需与模型特征名一致，例如 `t1_blue`、`t2_ndvi`。
- 单张原始影像上传只适用于单时相模型；如果模型包含多个时相槽，API 会要求上传多时相特征栈。

当前默认测试影像的波段顺序是：

```text
1 Blue
2 Green
3 Red
4 Red Edge
5 NIR
```

### 4.5 轮询任务状态

```bash
curl http://192.168.110.53:8000/api/infer/status/20260625_170806_a4a6d4ee
```

完成后的返回示例：

```json
{
  "task_id": "20260625_170806_a4a6d4ee",
  "status": "completed",
  "progress": 100.0,
  "message": "Completed.",
  "valid_pixel_count": 953581,
  "model_features": [
    "blue",
    "green",
    "red",
    "rededge",
    "nir",
    "ndvi",
    "ndwi",
    "evi",
    "ndre"
  ],
  "top_predictions": [
    {
      "class_code": 0,
      "label": "Others",
      "confidence": 0.7076045211251011
    }
  ],
  "downloads": {
    "classification": "/api/infer/download/20260625_170806_a4a6d4ee?format=classification",
    "confidence": "/api/infer/download/20260625_170806_a4a6d4ee?format=confidence",
    "shp": "/api/infer/download/20260625_170806_a4a6d4ee?format=shp",
    "metadata": "/api/infer/download/20260625_170806_a4a6d4ee?format=metadata"
  }
}
```

字段解释：

- `status`：`queued`、`running`、`completed`、`failed`
- `valid_pixel_count`：参与分类的有效像素数量
- `top_predictions`：整幅影像的主要类别结果
- `downloads`：本次任务结果文件的下载地址

### 4.6 下载结果文件

下载分类图：

```bash
curl -OJ "http://192.168.110.53:8000/api/infer/download/20260625_170806_a4a6d4ee?format=classification"
```

下载置信度图：

```bash
curl -OJ "http://192.168.110.53:8000/api/infer/download/20260625_170806_a4a6d4ee?format=confidence"
```

下载地块级 Shapefile ZIP，仅当启动任务时传入了 `parcel_file_id` 才会生成：

```bash
curl -OJ "http://192.168.110.53:8000/api/infer/download/20260625_170806_a4a6d4ee?format=shp"
```

下载元数据：

```bash
curl -OJ "http://192.168.110.53:8000/api/infer/download/20260625_170806_a4a6d4ee?format=metadata"
```

也可以直接调用固定结果接口：

```bash
curl -OJ "http://192.168.110.53:8000/api-predictions/20260625_170806_a4a6d4ee/classification"
curl -OJ "http://192.168.110.53:8000/api-predictions/20260625_170806_a4a6d4ee/confidence"
curl -OJ "http://192.168.110.53:8000/api-predictions/20260625_170806_a4a6d4ee/shp"
curl "http://192.168.110.53:8000/api-predictions/20260625_170806_a4a6d4ee/metadata"
```

## 5. Python 测试示例

```python
import time
import requests

BASE_URL = "http://192.168.110.53:8000"
IMAGE_PATH = "sample_upload_multispectral_5band.tif"

health = requests.get(f"{BASE_URL}/api/health", timeout=10)
print("health:", health.json())

with open(IMAGE_PATH, "rb") as f:
    upload = requests.post(
        f"{BASE_URL}/api/data/upload",
        files={"file": (IMAGE_PATH, f, "image/tiff")},
        timeout=300,
    )
upload_json = upload.json()
print("upload:", upload_json)

start = requests.post(
    f"{BASE_URL}/api/infer/start",
    json={
        "file_id": upload_json["file_id"],
        "blue_band": 1,
        "green_band": 2,
        "red_band": 3,
        "rededge_band": 4,
        "nir_band": 5,
        "swir_band": 0,
        "reflectance_scale": 1.0,
        "top_k": 1,
    },
    timeout=30,
)
task_id = start.json()["task_id"]
print("task_id:", task_id)

while True:
    status = requests.get(f"{BASE_URL}/api/infer/status/{task_id}", timeout=30).json()
    print("status:", status["status"], "progress:", status["progress"])
    if status["status"] in {"completed", "failed"}:
        break
    time.sleep(1)

if status["status"] == "completed":
    r = requests.get(
        f"{BASE_URL}/api/infer/download/{task_id}",
        params={"format": "classification"},
        timeout=300,
    )
    with open(f"{task_id}_classification.tif", "wb") as f:
        f.write(r.content)
    print("downloaded:", f"{task_id}_classification.tif")
else:
    print(status)
```

## 6. 本项目当前已验证通过的一次真实测试

本服务已使用以下样例完成过一次主链路验证：

- 输入文件：`data/input/sample_upload_multispectral_5band.tif`
- 上传返回 `file_id`：`c2f7f61300904800b034bf0637d7b895`
- 推理返回 `task_id`：`20260625_170806_a4a6d4ee`
- 最终状态：`completed`
- 整幅影像主要类别结果：
  - `Others`
  - 置信度 `0.7076045211251011`

对应输出文件位于：

```text
data/output/api_predictions/20260625_170806_a4a6d4ee_classification.tif
data/output/api_predictions/20260625_170806_a4a6d4ee_confidence.tif
data/output/api_predictions/20260625_170806_a4a6d4ee_metadata.json
```

## 7. 常见问题

### 7.1 为什么我能打开页面，但接口调用失败？

常见原因有：

- 服务只在本机启动，未监听外部可访问地址
- 对方和服务机器不在同一局域网
- Windows 防火墙未放行 `8000` 端口
- 调用了 `127.0.0.1`，而不是服务机器的真实 IP

### 7.2 为什么上传成功了，但推理失败？

常见原因有：

- 上传的不是 GeoTIFF
- 波段编号设置错误
- 影像缺少模型所需波段
- 模型文件或输出目录缺失

### 7.3 `top_k` 是什么意思？

它表示返回整幅影像的前 `k` 个主要类别结果。

例如：

- `top_k = 1`：只返回最主要的一个类别
- `top_k = 5`：返回前 5 个主要类别及各自置信度

对接展示时，可以直接理解为“整幅影像主要类别结果数量”。
