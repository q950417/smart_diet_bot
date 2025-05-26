import torch, pathlib, json
from PIL import Image
from transformers import AutoProcessor, ViTForImageClassification

MODEL_NAME = os.getenv("MODEL_NAME","vit-base-patch16-224-in21k")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    "nateraw/food101-vit-base-patch16-224",  # 已 fine-tune
).to(device)
# 英文 → 中文菜名對照
with open(pathlib.Path(__file__).parent/"label_zh.json","r",encoding="utf8") as f:
    LABEL_ZH = json.load(f)

def classify_image(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]
    return LABEL_ZH.get(label,label)   # 盡量回傳中文
