from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("crop_classifier_api.api:app", host="0.0.0.0", port=8000)
"""
启动服务：
cd D:\CirRou\CropClassifier\SentinelCropService
.\.venv\Scripts\python.exe run_api.py
"""