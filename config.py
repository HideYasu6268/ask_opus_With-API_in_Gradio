"""設定。単価は公式料金ページで確認して更新すること（仕様書8章）。"""
import os

# 単価: USD / 100万トークン（2026-10時点の公開情報。Opus 5.5のキャッシュ読み取りは0.05倍で$0.20）
MODELS = {
    "Sonnet": dict(id="claude-sonnet-5-5", inp=2.0, out=10.0, w5m=2.5, w1h=4.0, read=0.20),
    "Opus": dict(id="claude-opus-5-5", inp=4.0, out=20.0, w5m=5.0, w1h=8.0, read=0.20),
}
USD_JPY = float(os.getenv("USD_JPY", "150"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
CACHE_TTL = os.getenv("CACHE_TTL", "1h")  # "1h"（既定）または "5m"

# 回答の長さ: 出力上限（思考分は別枠で自動加算）
LENGTH = {"短め": 2000, "標準": 4000, "長め": 8000}
# 思考の深さ（claude.aiと同じ 低/中/高）: 5.5系は budget_tokens が400エラーのため adaptive thinking + effort で指定する
THINK = {"低": "low", "中": "medium", "高": "high"}
THINK_HEADROOM = {"low": 2000, "medium": 4000, "high": 8000}  # 思考ぶんとしてmax_tokensに足す量

# 履歴圧縮（文字数ベースの簡易推定）
COMPRESS_CHARS = int(os.getenv("COMPRESS_CHARS", "16000"))
KEEP_TURNS = int(os.getenv("KEEP_TURNS", "3"))  # 直近の残すやり取り数

BASE_SYSTEM = """あなたは本人専用の個人相談アドバイザーです。深く丁寧に推論し、日本語で答えてください。
資料（handoff package・要約・全文のいずれか）が与えられた場合は、原文抜粋を根拠にし、どのファイルのどの箇所に基づくかを示してください。
要約しか与えられていない場合、細部（数値・文言）は原文で確認できていないことを踏まえてください。
資料にないことは推測だと明示し、uncertainties（不足・不確かな点）は率直に指摘してください。
資料が足りず結論が変わり得るときは、何が必要かを伝えてください。"""

HANDOFF_HEADER = ("【資料: Geminiが作成したhandoff package（JSON）。原文抜粋が主。落ちている情報がある可能性はuncertaintiesを参照。\n"
                  "この抜粋は資料追加時の下書きの相談文に基づいて抽出したもので、実際の質問と一致しない場合がある。"
                  "足りない情報があれば、uncertaintiesや全文読み込みの必要性を踏まえて、何が必要かを伝えること】\n")
FULLTEXT_HEADER = "【資料: ファイル全文（テキスト化したもの）】\n"