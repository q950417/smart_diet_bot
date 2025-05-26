import os, tempfile, asyncio, json
from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import (
    WebhookParser, WebhookHandler
)
from linebot.v3.messaging import (
    AsyncMessagingApi, ReplyMessageRequest, TextMessage, ImageMessage
)
from linebot.v3.exceptions import LineBotApiError
from food_classifier import classify_image
from nutrition_db import lookup_food
from chat import generate_reply
from dotenv import load_dotenv; load_dotenv()

app  = FastAPI()
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
line_api = AsyncMessagingApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))

# 入口──LINE 將 POST 打到 /callback
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
        if event.message.type == "image":
            await handle_image(event)
        elif event.message.type == "text":
            await handle_text(event)
    return "OK"

async def handle_image(event):
    # 下載圖片檔
    msg_id = event.message.id
    content = await line_api.get_message_content(msg_id)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        for chunk in content.iter_content():
            fp.write(chunk)
        tmp_path = fp.name

    food = classify_image(tmp_path)         # ex. "fried_rice"
    info = lookup_food(food)                # dict {name, kcal, protein, fat ...}
    if info:
        reply = f"{info['name']} ≈ {info['kcal']} kcal\n" \
                f"蛋白質 {info['protein']} g、脂肪 {info['fat']} g\n" \
                f"建議：{info['advice']}"
    else:
        reply = f"辨識到「{food}」，但暫時找不到營養資料 QQ"

    await line_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply)]
        )
    )

async def handle_text(event):
    text = event.message.text.strip()
    info = lookup_food(text)
    if info:
        reply = f"{info['name']} ≈ {info['kcal']} kcal\n" \
                f"蛋白質 {info['protein']} g、脂肪 {info['fat']} g\n" \
                f"建議：{info['advice']}"
    else:
        # 非食物關鍵字 → 丟給 GPT-4o 當陪聊
        reply = generate_reply(text)
    await line_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply)]
        )
    )
