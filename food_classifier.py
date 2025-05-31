# food_classifier.py

import os
import httpx
import pathlib
import json
from typing import Optional

# 從 .env 或環境變數讀 Spoonacular API Key
API_KEY = os.getenv("SPOONACULAR_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("Missing SPOONACULAR_API_KEY in environment.")

# Spoonacular Image Classification Endpoint
SPOONACULAR_URL = "https://api.spoonacular.com/food/images/classify"

def classify_image(img_path: str) -> str:
    """
    使用 Spoonacular 的 Food Image Classification API  
    參數 img_path: 本地暫存圖檔路徑
    回傳: 最高機率條目的 food name (若 API 失敗或拿不到則回空字串)
    """
    # 讀取圖片二進位
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # 給 httpx 一個 multipart/form-data 請求 body
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    params = {"apiKey": API_KEY}

    try:
        # 非同步用 httpx.AsyncClient or 同步用 httpx.post
        r = httpx.post(SPOONACULAR_URL, params=params, files=files, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        # data 看起來像：
        # {
        #   "status": "success",
        #   "classified": [
        #       {"name": "pizza", "probability": 0.95},
        #       {"name": "pasta", "probability": 0.03},
        #       ...
        #   ]
        # }
        classes = data.get("classified", [])
        if classes:
            # 回傳最高機率的那個 food name
            top = max(classes, key=lambda x: x.get("probability", 0))
            return top.get("name", "").lower()
        else:
            return ""
    except Exception as e:
        print("Spoonacular classify_image error:", e)
        return ""
