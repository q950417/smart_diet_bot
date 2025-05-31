import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "你是一個溫暖的飲食小幫手。"
    "回答時先簡短回應使用者，再附一句飲食小提醒。"
)

def generate_reply(user_msg: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",             # 若成本考量，可改 gpt-3.5-turbo
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()
