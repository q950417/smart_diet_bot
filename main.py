import os, tempfile
from fastapi import FastAPI, Request, HTTPException

# -------- LINE Bot SDK v3 --------
from linebot.v3.webhook import WebhookParser, WebhookHandler
from linebot.v3.messaging import (
    AsyncMessagingApi, ReplyMessageRequest, TextMessage, ImageMessage
)
from linebot.v3.exceptions import ApiException        # ← 正確名稱

# -------- 你的功能模組 --------
from food_classifier import classify_image
from nutrition_db import lookup_food
from chat import generate_reply
from dotenv import load_dotenv; load_dotenv()

app = FastAPI()

parser   = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))
handler  = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
line_api = AsyncMessagingApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))

# ---------- Webhook 入口 ----------
@app.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")
    try:
        events = parser.parse(body.decode(), signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    for event in events:
        if isinstance(event.message, ImageMessage):
            await handle_image(event)
        elif isinstance(event.message, TextMessage):
            await handle_text(event)
    return "OK"

# ---------- 文字 ----------
async def handle_text(event):
    text = event.message.text.strip()
    info = lookup_food(text)
    if info:
        reply = (
            f"{info['name']} ≈ {info['kcal']} kcal\n"
            f"蛋白質 {info['protein']} g、脂肪 {info['fat']} g\n"
            f"建議：{info['advice']}"
        )
    else:
        reply = generate_reply(text)
    await safe_reply(event.reply_token, reply)

# ---------- 圖片 ----------
async def handle_image(event):
    msg_id = event.message.id
    content = await line_api.get_message_content(msg_id)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        async for chunk in content.iter_content():
            fp.write(chunk)
        tmp_path = fp.name

    food = classify_image(tmp_path)
    info = lookup_food(food)
    if info:
        reply = (
            f"{info['name']} ≈ {info['kcal']} kcal\n"
            f"蛋白質 {info['protein']} g、脂肪 {info['fat']} g\n"
            f"建議：{info['advice']}"
        )
    else:
        reply = f"辨識到「{food}」，但暫時沒有營養資料 QQ"

    await safe_reply(event.reply_token, reply)

# ---------- 安全回覆 ----------
async def safe_reply(token: str, message: str):
    try:
        await line_api.reply_message(
            ReplyMessageRequest(
                reply_token=token,
                messages=[TextMessage(text=message)]
            )
        )
    except ApiException as e:           # ← v3 通用例外
        print("ApiException:", e.status_code, e.headers, e.body)

# ---------- 健康檢查 ----------
@app.get("/healthz")
def health():
    return {"ok": True}
