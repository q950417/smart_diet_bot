import pandas as pd, pathlib, re

CSV = pathlib.Path(__file__).parent/"nutrition.csv"
df  = pd.read_csv(CSV)          # 欄位: name,kcal,protein,fat,carb,advice

def _normalize(s: str):
    return re.sub(r"\s+","",s.lower())

def lookup_food(query: str):
    q = _normalize(query)
    hit = df[df["name_norm"]==q]
    if hit.empty:
        # 簡易模糊搜尋
        hit = df[df["name_norm"].str.contains(q)]
    return hit.iloc[0].to_dict() if not hit.empty else None
