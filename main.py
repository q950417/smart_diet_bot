import os, tempfile, asyncio, json
from fastapi import FastAPI, Request, HTTPException

# ✅ v3 正確匯入路徑（單數 webhook）
from linebot.v3.webhook import WebhookParser, WebhookHandler
from linebot.v3.messaging import (
    AsyncMessagingApi, ReplyMessageRequest, TextMessage, ImageMessage
)
from linebot.v3.exceptions import MessagingApiError   # v3 的例外類別

from food_classifier import classify_image
from nutrition_db import lookup_food
from chat import generate_reply
from dotenv import load_dotenv; load_dotenv()

app = FastAPI()

# ---- LINE SDK 初始化 ----
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
line_api = AsyncMessagingApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))


# 入口：LINE POST /callback
@app.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")
    try:
        events = parser.parse(body.decode(), signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 逐一處理訊息事件
    for event in events:
        if isinstance(event.message, ImageMessage):
            await handle_image(event)
        elif isinstance(event.message, TextMessage):
            await handle_text(event)
    return "OK"


# ----- 處理圖片 -----
async def handle_image(event):
    # 下載暫存圖片
    msg_id = event.message.id
    content = await line_api.get_message_content(msg_id)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        async for chunk in content.iter_content():
            fp.write(chunk)
        tmp_path = fp.name

    # 推論 → 查營養
    food = classify_image(tmp_path)            # e.g. "fried_rice"
    info = lookup_food(food)                   # dict 或 None

    if info:
        reply = (
            f"{info['name']} ≈ {info['kcal']} kcal\n"
            f"蛋白質 {info['protein']} g、脂肪 {info['fat']} g\n"
            f"建議：{info['advice']}"
        )
    else:
        reply = f"辨識到「{food}」，但暫時沒有營養資料 QQ"

    # 回覆
    await safe_reply(event.reply_token, reply)


# ----- 處理文字 -----
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
        reply = generate_reply(text)           # GPT-4o 聊天

    await safe_reply(event.reply_token, reply)


# ---- 共用安全回覆 ----
async def safe_reply(token: str, message: str):
    try:
        await line_api.reply_message(
            ReplyMessageRequest(
                reply_token=token,
                messages=[TextMessage(text=message)]
            )
        )
    except MessagingApiError as e:
        # 記錄錯誤避免程式崩潰
        print("LINE MessagingApiError",
              e.status_code, e.headers, e.body)


# (可選) 健康檢查，方便 Render / UptimeRobot ping
@app.get("/healthz")
def health():
    return {"ok": True}
