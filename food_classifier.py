import os, torch, pathlib, json
from PIL import Image
from transformers import AutoProcessor, ViTForImageClassification

# 讀環境變數，可用 .env 覆寫
MODEL_NAME = os.getenv("MODEL_NAME", "vit-base-patch16-224-in21k")

# Render 免費／Starter 只有 CPU，把 tensor 固定放 CPU
device = "cpu"

# 下載（或由 cache 載入）Processor & Fine-tune 後的 Food101 ViT
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    "nateraw/food101-vit-base-patch16-224"
).to(device)

# 英文 → 中文菜名對照；若找不到 json 檔就退回英文
try:
    with open(pathlib.Path(__file__).parent / "label_zh.json",
              encoding="utf8") as f:
        LABEL_ZH = json.load(f)
except FileNotFoundError:
    LABEL_ZH = {}

def classify_image(img_path: str) -> str:
    """回傳食物名稱（盡量中文）。"""
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    en_label = model.config.id2label[pred_id]
    return LABEL_ZH.get(en_label, en_label)
