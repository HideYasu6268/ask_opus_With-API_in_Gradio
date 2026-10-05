"""Claude / Gemini / Google Drive まわりの処理。"""
import json
import os
import re
from functools import cache
from typing import Literal

import anthropic
import google.auth
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydantic import BaseModel

import config as C


@cache
def claude():
    return anthropic.Anthropic()  # ANTHROPIC_API_KEY（環境変数 / Secret Manager）


@cache
def gemini():
    return genai.Client()  # GEMINI_API_KEY。有料枠のキーを使うこと


# ---------------- Drive ----------------
@cache
def _drive():
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # ADCでスコープが使えない場合の代替
    if info:
        creds = service_account.Credentials.from_service_account_info(json.loads(info), scopes=scopes)
    else:
        creds, _ = google.auth.default(scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


EXPORT = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",  # 先頭シートのみ
}


def file_id(url: str) -> str:
    m = re.search(r"/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if not m:
        raise ValueError(f"URLからファイルIDを取り出せません: {url}")
    return m.group(1)


# メッセージ欄からDriveのURLを拾う。ASCII文字だけに限定し、URL直後の日本語や句読点（。、）を巻き込まない
DRIVE_URL = re.compile(
    r"https?://(?:drive|docs)\.google\.com/[A-Za-z0-9_\-./?=&%#+~:@]*[A-Za-z0-9_\-/=&%#+~@]")


def drive_links(text: str) -> tuple[list[str], str]:
    """メッセージから(DriveのURL一覧, URLを除いた本文)を返す。"""
    text = text or ""
    urls = list(dict.fromkeys(DRIVE_URL.findall(text)))
    rest = DRIVE_URL.sub("", text)
    rest = re.sub(r"[ \t]+\n", "\n", rest)
    rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
    return urls, rest


def fetch_drive_file(url: str) -> dict:
    """共有リンクはファイルIDの入口としてのみ使い、内容はDrive APIで取得する。"""
    svc, fid = _drive(), file_id(url)
    meta = svc.files().get(fileId=fid, fields="id,name,mimeType", supportsAllDrives=True).execute()
    mime = meta["mimeType"]
    if mime in EXPORT:
        data = svc.files().export(fileId=fid, mimeType=EXPORT[mime]).execute()
    else:
        data = svc.files().get_media(fileId=fid, supportsAllDrives=True).execute()
    f = {"name": meta["name"], "mime": mime, "text": None, "pdf": None}
    if mime == "application/pdf":
        f["pdf"] = data
    elif mime in EXPORT or mime.startswith("text/") or mime == "application/json":
        f["text"] = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    else:
        raise ValueError(f"未対応の形式です: {meta['name']} ({mime})")
    return f


# ---------------- Gemini（資料係） ----------------
class Evidence(BaseModel):
    text: str  # 原文抜粋（改変しない）
    location: str  # 章・見出し・ページなど


class RelevantFile(BaseModel):
    file: str
    relevance: Literal["high", "medium", "low"]
    evidence: list[Evidence]


class Handoff(BaseModel):
    relevant_files: list[RelevantFile]
    retrieval_summary: str
    uncertainties: list[str]
    full_read_required: bool


HANDOFF_PROMPT = """あなたは資料係です。指定ファイルから、ユーザーの相談に関連する箇所を抽出してhandoff packageを作ります。
- 要約より原文の抜粋（verbatim）を主にする。抜粋は改変・言い換えしない。locationに見出し・章・ページを書く。
- 重要な箇所を落とさない。不足している情報・不確かな点は必ずuncertaintiesに書く（例: 2024年の記録が不足）。
- 全文を読まないと判断できないと考えたらfull_read_required=trueにする。
- 相談が未入力・曖昧なら、資料全体の重要事項（年表・決定事項・数値・固有名詞）を広めに抜粋する。"""


def _gen(contents, **cfg):
    return gemini().models.generate_content(
        model=C.GEMINI_MODEL, contents=contents, config=types.GenerateContentConfig(**cfg)
    )


def _file_parts(f):
    body = f["text"] if f["text"] is not None else types.Part.from_bytes(data=f["pdf"], mime_type="application/pdf")
    return [f"=== ファイル: {f['name']} ===", body]


def make_handoff(files: list[dict], hint: str) -> str:
    parts = [p for f in files for p in _file_parts(f)]
    parts.append(f"=== ユーザーの相談（下書き。空のこともある） ===\n{hint or '（未入力）'}")
    r = _gen(parts, system_instruction=HANDOFF_PROMPT, response_mime_type="application/json", response_schema=Handoff)
    data = r.parsed.model_dump() if r.parsed else json.loads(r.text)
    return C.HANDOFF_HEADER + json.dumps(data, ensure_ascii=False, indent=1)


SUMMARY_PROMPT = """あなたは資料係です。指定ファイルを、後続の相談で使えるよう要約してください。
- 数値・日付・固有名詞・決定事項は落とさず、正確に残す。
- ファイルごとに見出しを分け、元の章立てが分かるようにする。
- 原文にない推測は書かない。読み取れなかった箇所は末尾に「不明点」として書く。
- 相談文の下書きがあれば関連の深い部分を厚めにする。ただし無関係に見える部分も1〜2行は残す。"""

SUMMARY_HEADER = ("以下は資料をGeminiが要約したもの。原文ではなく、細部は省略されている可能性がある。"
                  "正確な数値や文言が必要な場合は、要約では確認できないことをユーザーに伝えること。\n\n")


def make_summary(files: list[dict], hint: str) -> str:
    parts = [p for f in files for p in _file_parts(f)]
    parts.append(f"=== ユーザーの相談（下書き。空のこともある） ===\n{hint or '（未入力）'}")
    return SUMMARY_HEADER + _gen(parts, system_instruction=SUMMARY_PROMPT).text


def make_fulltext(files: list[dict]) -> str:
    out = []
    for f in files:
        text = f["text"]
        if text is None:  # PDFは忠実にテキスト化（出力が長大な場合は途中で切れ得る）
            text = _gen(
                [types.Part.from_bytes(data=f["pdf"], mime_type="application/pdf"),
                 "このPDFの全文を、要約・改変せず忠実にテキスト化してください。図表は[図: 内容]の形で簡潔に。"]
            ).text
        out.append(f'<file name="{f["name"]}">\n{text}\n</file>')
    return C.FULLTEXT_HEADER + "\n".join(out)


def summarize_history(turns: list[tuple[str, str]]) -> str:
    log = "\n".join(f"{'ユーザー' if r == 'user' else 'アシスタント'}: {c}" for r, c in turns)
    return _gen(
        f"次の会話ログを、後続の相談に必要な事実・決定・論点・ユーザーの意向を落とさず簡潔に要約してください。\n\n{log}"
    ).text


# ---------------- Claude（相談・推論） ----------------
def cc(ttl: str | None) -> dict:
    return {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}


def build_system(docs: list[str] | None, ttl: str) -> list[dict]:
    """system → 資料1 → 資料2 … の順。資料は追加方式で、古い資料の並びと中身は変えない。
    キャッシュ境界は新しい側の2つの資料に置く（上限4個のうち、履歴用に1個を残す）。"""
    blocks = [{"type": "text", "text": C.BASE_SYSTEM}]
    docs = docs or []
    for i, d in enumerate(docs):
        b = {"type": "text", "text": f"【資料セット{i + 1}】\n{d}"}
        if i >= len(docs) - 2:
            b["cache_control"] = cc(ttl)
        blocks.append(b)
    return blocks


def build_messages(history, question: str, ttl: str | None, cache_last: bool = False) -> list[dict]:
    """cache_last=True: 今回のユーザー発言（添付資料入り）にもキャッシュ境界を置き、書き込みを1回で済ませる。
    境界の合計は 資料2 + 履歴末尾1 + 今回1 = 4（上限）。"""
    msgs = [{"role": r, "content": c} for r, c in history]
    if msgs and ttl:  # 資料ありの場合のみ、履歴末尾にもキャッシュ境界を置く
        msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"], "cache_control": cc(ttl)}]
    msgs.append({"role": "user", "content": question})
    if cache_last and ttl:
        msgs[-1]["content"] = [{"type": "text", "text": question, "cache_control": cc(ttl)}]
    return msgs


def request_kwargs(model: str, system, messages, length: str, think: str) -> dict:
    effort = C.THINK[think]
    max_tokens = C.LENGTH[length] + (C.THINK_HEADROOM[effort] if effort else 0)  # 思考はmax_tokensに含まれる
    kw = dict(model=C.MODELS[model]["id"], system=system, messages=messages, max_tokens=max_tokens)
    if effort:
        kw["thinking"] = {"type": "adaptive"}
        kw["extra_body"] = {"output_config": {"effort": effort}}
    else:
        kw["thinking"] = {"type": "disabled"}
    return kw


def warm_cache(model: str, docs: list[str], history, ttl: str):
    """出力を最小にしたリクエストで、資料と履歴をキャッシュへ書き込む（資料を追加した直後に使う）。"""
    r = claude().messages.create(
        model=C.MODELS[model]["id"], max_tokens=1, system=build_system(docs, ttl),
        messages=build_messages(history, "資料の読み込み確認", ttl), thinking={"type": "disabled"},
    )
    return r.usage


def _count(model_id, system, messages) -> int:
    return claude().messages.count_tokens(model=model_id, system=system, messages=messages).input_tokens


def estimate(model: str, system, messages, max_tokens: int, ttl: str) -> dict:
    m = C.MODELS[model]
    total = _count(m["id"], system, messages)
    prefix = _count(m["id"], system, [{"role": "user", "content": "."}]) if len(system) > 1 else 0
    rest = max(total - prefix, 0)
    write = m["w1h"] if ttl == "1h" else m["w5m"]
    return dict(
        tokens=total, prefix=prefix, max_tokens=max_tokens,
        miss=(prefix * write + rest * m["inp"]) / 1e6,
        hit=(prefix * m["read"] + rest * m["inp"]) / 1e6,
        out=max_tokens * m["out"] / 1e6,
    )


def actual_cost(usage, model: str, ttl: str) -> float:
    m = C.MODELS[model]
    write = m["w1h"] if ttl == "1h" else m["w5m"]
    return (
        usage.input_tokens * m["inp"]
        + (usage.cache_creation_input_tokens or 0) * write
        + (usage.cache_read_input_tokens or 0) * m["read"]
        + usage.output_tokens * m["out"]
    ) / 1e6


def yen(usd: float) -> str:
    return f"${usd:.4f}（約{usd * C.USD_JPY:.1f}円）"