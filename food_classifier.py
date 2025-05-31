import os, torch, pathlib, json
from PIL import Image
from transformers import AutoProcessor, ViTForImageClassification

# ----- 模型 ID -----
# 1) Processor：公開可抓的 ID  (加上 google/)
# 2) Classifier：已 fine-tune 的 Food101 權重
PROCESSOR_ID  = "google/vit-base-patch16-224-in21k"
CLASSIFIER_ID = "nateraw/food101-vit-base-patch16-224"

device = "cpu"   # Render 免費方案只有 CPU

# 下載 / 載入
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
model     = ViTForImageClassification.from_pretrained(CLASSIFIER_ID).to(device)

# 英文→中文菜名對照
try:
    with open(pathlib.Path(__file__).parent / "label_zh.json", encoding="utf8") as f:
        LABEL_ZH = json.load(f)
except FileNotFoundError:
    LABEL_ZH = {}

def classify_image(img_path: str) -> str:
    """給圖片路徑，回傳食物名稱（盡量中文）。"""
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    en_name = model.config.id2label[pred_id]
    return LABEL_ZH.get(en_name, en_name)
