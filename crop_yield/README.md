本文档用于协助作物种植分布算法模块理解现有估产流程，并将估产功能集成到种植分类API中，实现从作物分类到产量估算的完整端到端服务。

一、整体流程概述
<TEXT>
用户请求（经纬度/区域/时间）
↓
【种植分类API】得到作物种类栅格（TIF/掩膜）
↓
【调用估产模块】输入：影像路径 + 作物掩膜 → 输出：各作物产量统计
↓
返回JSON结果（总产量/面积/单产等）
现有估产模块代码结构（5个文件）：

文件 功能
sentinel.py 主流程：读取卫星/无人机影像 → 计算NDVI/LAI → 套作物掩膜 → 估产 → 生成统计结果
yield_model.py 估产核心函数：calculate_lai() 和 estimate_yield()
geometry.py 矢量处理：范围裁剪、坐标转换、面积计算
crop_distribution.py 作物分布数据读取：支持本地TIF、远程TIF、SHP/GeoJSON/ZIP
config.py 全局参数：作物光谱系数、模型斜率截距、默认路径、外部API地址
二、各文件关键接口及调用方式

1. crop_distribution.py（作物分布读取）
   用途：将您生成的作物分类结果（栅格或矢量）加载为numpy数组或掩膜，供后续估产使用。

主要函数：

<PYTHON>
def read_crop_distribution(source, source_type='tif', band_index=0,
                           crs=None, extent=None):
    """
    输入：
        source       : 文件路径（本地TIF）或URL（外部API返回的TIF）或SHP/GeoJSON路径
        source_type  : 'tif' | 'shp' | 'geojson' | 'zip'
        band_index   : 若多波段栅格，指定作物类别所在波段
        crs          : 目标坐标系（如'EPSG:4326'），若不指定则保持原始
        extent       : 可选，裁剪范围（[minx, miny, maxx, maxy]）
    返回：
        crop_mask    : 二维numpy数组，每个像素值为作物类别ID（0=非作物）
        transform    : geotransform对象（用于坐标转换）
        crs          : 栅格坐标系
    """
集成建议：

您可在分类API的输出中直接生成GeoTIFF，然后调用此函数读取即可获取掩膜。
若分类结果以GeoJSON/SHP形式返回，建议先栅格化（利用geometry.py中的shp_to_raster辅助函数）再读入。2. geometry.py（空间几何工具）
用途：处理区域裁剪、面积计算、坐标投影转换，保证估产区域与影像空间对齐。

关键函数：

<PYTHON>
def clip_image_to_extent(image_path, extent, output_path=None):
    """用矢量范围裁剪卫星影像，返回裁剪后的numpy数组和affine变换"""
def polygon_area(polygon, crs='EPSG:4326'):
    """计算多边形面积（平方米/亩），支持WGS84自动转投影"""
def transform_coords(coords, source_crs, target_crs):
    """坐标批量转换"""
def shp_to_raster(shp_path, reference_raster_path, output_tif_path,
                  attribute_column='class_id', fill_value=0):
    """将矢量分类结果栅格化为与参考影像对齐的TIF，供crop_distribution读取"""
集成注意点：

当用户请求的区域是GeoJSON时，先用geometry.py将矢量转为栅格或直接用于裁剪影像。
面积计算在最终统计中要用到（如总产量/总面积 → 单产）。3. sentinel.py（核心估产流程）
用途：串联整个估产步骤。该模块假设已获得作物掩膜，并下载了对应时段的卫星/无人机影像。

主函数：

<PYTHON>
def estimate_yield_pipeline(image_path, crop_mask, model_params,
                            resolution=10, output_format='json'):
    """
    输入：
        image_path   : 多光谱栅格路径（含4个波段：B1蓝, B2绿, B3红, B4近红外）
        crop_mask    : 与影像同size的二维数组（0=背景，正整数=作物类别）
        model_params : 从config.py读取的模型字典，包含各作物LAI→产量系数
        resolution   : 像素分辨率（米），用于面积计算
        output_format: 'json' 或 'dict'
    内部流程：
        1. 读取影像，计算NDVI = (NIR - Red) / (NIR + Red)
        2. 应用作物掩膜，只处理有作物区域
        3. 调用yield_model.calculate_lai(ndvi) 得到LAI
        4. 按作物类别调用yield_model.estimate_yield(lai, category) 得到每个像素产量
        5. 积分（求和/均值）并生成统计结果
    返回：
        result_json : {
            "total_area_mu": 123.45,       # 总种植面积（亩）
            "crops": [
                {"class_id": 1, "class_name": "水稻",
                 "area_mu": 100, "yield_kg": 50000, "yield_per_mu_kg": 500},
                ...
            ],
            "timestamp": "..."
        }
    """
调用示例：

<PYTHON>
from sentinel import estimate_yield_pipeline
from crop_distribution import read_crop_distribution
from config import MODEL_PARAMS
# 1. 获得作物分类掩膜（假设您已生成）
crop_mask, transform, crs = read_crop_distribution('crop_classification.tif')
# 2. 调用估产（影像应与掩膜空间范围完全一致）
result = estimate_yield_pipeline('sentinel2_2023_08_15.tif', crop_mask, MODEL_PARAMS)
print(result)
4. yield_model.py（估产模型）
用途：提供LAI计算和产量估算的数学公式。

关键函数：

<PYTHON>
def calculate_lai(ndvi, method='default'):
    """
    NDVI → LAI (Leaf Area Index)
    支持多种经验模型（method可选：'default', 'clumped'）
    默认公式：LAI = 0.57 * exp(2.33 * NDVI)  [校正参数存储在config.py]
    """
    return lai_array
def estimate_yield(lai, category, params_dict):
    """
    输入：
        lai         : 像素级LAI数组
        category    : 作物类别ID（1=水稻, 2=小麦, ...）
        params_dict : 从config.py读取的系数，如
                      {'rice': {'a': 0.45, 'b': -0.1, 'max_lai': 6.0}, ...}
    公式：yield = a * lai + b   (或更复杂的非线性模型)
    返回像素级产量数组（kg/像素）
    """
    return yield_array
注意：当前模型为统计回归经验模型，若您后续有更先进的AI模型（如深度学习），可直接替换此文件中的函数，保持接口一致即可。

5. config.py（全局配置）
   用途：集中存放所有可变参数，便于调试和更换区域。

<PYTHON>
# 作物类别名称映射
CROP_CLASS_NAMES = {
    1: "水稻", 2: "小麦", 3: "玉米", 4: "大豆", ...
}
# LAI→产量模型系数 (对应每个作物)
LAI_YIELD_PARAMS = {
    1: {"a": 0.45, "b": -0.08},   # a * lai + b  (kg/m²)
    2: {"a": 0.32, "b": -0.05},
    ...
}
# 面积转换常数
MU_PER_SQKM = 1500.0   # 每亩面积（平方米）
# 默认影像源（卫星/无人机）
DEFAULT_SATELLITE_SOURCE = {
    'sentinel2': {'bands': ['B2','B3','B4','B8'], 'resolution': 10}
}
# 外部种植分布API地址（若您是调用远程服务）
EXTERNAL_DISTRIBUTION_API = "http://your-service/crop-classification"
集成建议：

您可以在自己的分类API中导入此config，并确保作物类别ID与分类模型输出的ID一致。
外部API地址可留空或改为您的分类服务地址，供后续直接请求。
三、集成步骤（推荐方案）
假设您的种植分类API已经能返回作物分类栅格（GeoTIFF），需要添加估产端点。建议如下：

模块整合

将上述5个文件复制到您的API项目目录（例如estimator/），确保requirements.txt包含gdal, rasterio, shapely, numpy, flask等。

扩展API路由

增加一个POST端点，例如 /v1/estimate-yield，接收参数：

image_url (或image_data)：卫星影像（自动下载或上传）
area_of_interest（GeoJSON多边形）
date_range（可选，用于选择影像时间）
内部处理流程

<PYTHON>
from estimator.sentinel import estimate_yield_pipeline
from estimator.geometry import clip_image_to_extent
@app.route('/v1/estimate-yield', methods=['POST'])
def yield_estimate():
    # 1. 获取参数
    aoi = request.json['area_of_interest']
    # 2. 调用作物分类(您已有的函数)得到作物掩膜TIF
    crop_mask_path = classify_area(aoi)
    # 3. 下载或获取对应影像（sentinel2或其他源）
    image_path = download_satellite_image(aoi, date)
    # 4. 裁剪影像至与掩膜相同范围
    clipped_image = clip_image_to_extent(image_path, aoi)
    # 5. 读取掩膜
    crop_mask, _, _ = read_crop_distribution(crop_mask_path)
    # 6. 估产
    result = estimate_yield_pipeline(clipped_image, crop_mask, config.LAI_YIELD_PARAMS)
    return jsonify(result)
测试验证

准备一块已知产量区域，输入测试数据，对比实际统计。
检查不同作物类别是否正确映射到config中的系数。
四、注意事项
空间对齐：影像与作物掩膜必须使用相同的坐标参考系（CRS）和像素分辨率。建议统一采用EPSG:4326或UTM投影。geometry.py中的clip_image_to_extent会自动重投影对齐。

内存管理：大范围影像可能占用大量内存。可在estimate_yield_pipeline中添加分块处理（rasterio.windows），当前代码已内置分块逻辑。

模型参数更新：config.py中的系数可根据区域和作物品种调整。例如水稻在不同省份的a、b值不同，建议通过外部配置或数据库管理。

错误处理：当影像缺失或掩膜空时，返回友好的错误信息（如{"error": "No crop detected in area"}）。

外部依赖：如果您的分类API返回的是ZIP压缩包（内含SHP），crop_distribution.py支持直接读取。
