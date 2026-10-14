"""設定。単価は公式料金ページで確認して更新すること（仕様書8章）。"""
import os

# 単価: USD / 100万トークン（2026-10時点の公開情報。Opus 5.5のキャッシュ読み取りは0.05倍で$0.20）
MODELS = {
    "Sonnet": dict(id="claude-sonnet-5-5", inp=2.0, out=10.0, w5m=2.5, w1h=4.0, read=0.20),
    "Opus": dict(id="claude-opus-5-5", inp=4.0, out=20.0, w5m=5.0, w1h=8.0, read=0.20),
}
USD_JPY = float(os.getenv("USD_JPY", "150"))
# Geminiモデル指定。未設定時はAPIから一覧を取得してflash（新しい順）→liteの順にフォールバック試行
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
CACHE_TTL = os.getenv("CACHE_TTL", "1h")  # "1h"（既定）または "5m"

# 回答の長さ: 出力上限（思考分は別枠で自動加算）
LENGTH = {"短め": 2000, "標準": 4000, "長め": 8000}
# 思考の深さ（claude.aiと同じ 低/中/高）: 5.5系は budget_tokens が400エラーのため adaptive thinking + effort で指定する
THINK = {"低": "low", "中": "medium", "高": "high"}
THINK_HEADROOM = {"low": 2000, "medium": 4000, "high": 8000}  # 思考ぶんとしてmax_tokensに足す量
# 思考オフの指定。5.5系は {"type": "disabled"} が400になるため between_tools を使う
THINKING_OFF = {"type": "between_tools"}

# 履歴圧縮（文字数ベースの簡易推定）
COMPRESS_CHARS = int(os.getenv("COMPRESS_CHARS", "16000"))
KEEP_TURNS = int(os.getenv("KEEP_TURNS", "3"))  # 直近の残すやり取り数

BASE_SYSTEM = ""  # 空にすると素のClaudeに近い（空でなければ資料の前にsystemとして入る）

HANDOFF_HEADER = ("【資料: Geminiが作成したhandoff package（JSON）。原文抜粋が主。落ちている情報がある可能性はuncertaintiesを参照。\n"
                  "この抜粋は資料追加時の下書きの相談文に基づいて抽出したもので、実際の質問と一致しない場合がある。"
                  "足りない情報があれば、uncertaintiesや全文読み込みの必要性を踏まえて、何が必要かを伝えること】\n")
FULLTEXT_HEADER = "【資料: ファイル全文（テキスト化したもの）】\n"
