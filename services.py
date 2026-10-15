"""Claude / Gemini / Google Drive まわりの処理。"""
import json
import logging
import os
import time
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Literal

import anthropic
import google.auth
from google import genai
from google.genai import errors, types
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gradio as gr
from pydantic import BaseModel

logger = logging.getLogger(__name__)

import config as C


@cache
def claude():
    return anthropic.Anthropic()  # ANTHROPIC_API_KEY（環境変数 / Secret Manager）


def gemini_keys() -> list[str]:
    """GEMINI_API_KEY（先頭）と GEMINI_API_KEYS（カンマ/空白/改行区切り）を、重複なしで順に集める。
    別アカウントのキーを並べておくと、アカウント/プロジェクト単位の失敗（429・404など）のときに次のキーで再試行する。
    どのキーも【有料枠】にすること（無料枠は、送った内容がGoogleの製品改善に使われ得る）。"""
    raw = [os.getenv("GEMINI_API_KEY", "")] + re.split(r"[,\s]+", os.getenv("GEMINI_API_KEYS", ""))
    return list(dict.fromkeys(k.strip() for k in raw if k.strip()))


@cache
def _clients() -> list:
    keys = gemini_keys()
    if len(keys) > 1:
        logger.info("Gemini: %d 個のAPIキーを使います", len(keys))
    return [genai.Client(api_key=k) for k in keys] or [genai.Client()]  # キーが環境変数に無ければSDKの既定（GOOGLE_API_KEY等）


def gemini():
    return _clients()[0]  # 題名生成など、キーの切り替えが要らない用途は先頭のキー


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
    elif (mime in EXPORT or mime.startswith("text/") or mime == "application/json"
          or meta["name"].lower().endswith((".txt", ".md", ".markdown"))):  # .mdはDrive上で汎用バイナリ扱いのことがある
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


# 使うモデルを固定: 3.5-flash → ダメなら 3.5-flash-lite（2.5系は新規ユーザーには提供終了: 404 NOT_FOUND）
PREFERRED_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


def get_candidate_models() -> list[str]:
    """候補モデルを返す。環境変数 GEMINI_MODEL があれば先頭に追加する。"""
    if C.GEMINI_MODEL:
        pref = C.GEMINI_MODEL.removeprefix("models/")
        return [pref] + [m for m in PREFERRED_MODELS if m != pref]
    return list(PREFERRED_MODELS)


def _warn(msg: str):
    try:
        gr.Warning(msg)
    except Exception:  # noqa: BLE001  画面の文脈が無いスレッドでは出さない
        pass


def _info(msg: str):
    try:
        gr.Info(msg)
    except Exception:  # noqa: BLE001
        pass


def _gen_stream(contents, *, models=None, **cfg):
    """(status_msg, response, model_name) を yield するジェネレータ。
    試行中・フォールバック時は (msg, None, None) を yield し、成功時は (f"使用モデル: {model}", response, model) を yield する。
    モデルごとに、キーを順に試す。401/403（キーの不備）・404・429（アカウント/プロジェクト単位の失敗）のときだけ次のキーへ。
    503（Google側の混雑）などはキーを替えても同じなので、次のモデルへ進む。全キーが401/403ならそのまま投げる（設定の問題）。"""
    models = list(models) if models is not None else get_candidate_models()
    clients = _clients()
    many = len(clients) > 1
    last_err = None
    for i, model in enumerate(models):
        status = f"{model} を試行中…" if i == 0 else f"前モデルでエラーのため {model} に切り替えて試行中…"
        logger.info("Gemini: %s", status)
        _info(f"Gemini: {status}")
        yield status, None, None

        auth_only = True  # このモデルでの失敗が、すべて401/403か
        for ki, cl in enumerate(clients):
            tag = f"（キー{ki + 1}/{len(clients)}）" if many else ""
            if ki:
                msg = f"{model} をキー{ki + 1}で再試行中…"
                logger.info("Gemini: %s", msg)
                _info(f"Gemini: {msg}")
                yield msg, None, None
            try:
                res = None
                for attempt in range(3):  # 503(混雑)などは2秒・4秒待って同じモデル・同じキーを再試行
                    try:
                        res = cl.models.generate_content(
                            model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
                        break
                    except errors.ServerError:
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
                if ki:
                    _info(f"Gemini: {model} はキー{ki + 1}で成功しました")
                yield f"使用モデル: {model}{tag}", res, model
                return
            except errors.ClientError as e:
                code = getattr(e, "code", None)
                last_err = e
                if code in (401, 403):
                    logger.warning("Gemini: %s キー%d は使えません (code=%s)", model, ki + 1, code)
                    _warn(f"Gemini: キー{ki + 1} が使えません (code={code})")
                    continue  # このキーはだめ。次のキーへ（最後のキーなら下のelseへ）
                auth_only = False
                logger.warning("%s%s でエラー (code=%s): %s", model, tag, code, e)
                if code in (404, 429) and ki + 1 < len(clients):
                    _warn(f"Gemini: {model}{tag} でエラー (code={code})。次のキーで試します")
                    continue
                _warn(f"Gemini: {model}{tag} でエラー (code={code})")
                break  # 次のモデルへ
            except (errors.ServerError, errors.APIError) as e:
                auth_only = False
                last_err = e
                logger.warning("%s%s でサーバーエラー (%s): %s", model, tag, type(e).__name__, e)
                _warn(f"Gemini: {model} でサーバーエラー (503等)。次の候補へ切り替えます")
                break
            except Exception as e:  # noqa: BLE001
                auth_only = False
                last_err = e
                logger.warning("Gemini: %s%s で予期せぬエラー: %s", model, tag, e)
                _warn(f"Gemini: {model} でエラー発生。次の候補へ切り替えます")
                break
        else:  # breakせずにキーを使い切った（最後のキーが401/403）
            if auth_only:
                raise last_err  # 全キーが認証エラー: 設定の問題なので、他のモデルを試しても無駄

    raise RuntimeError(
        f"すべてのGemini候補モデルの試行に失敗しました（最後のエラー: {last_err}）"
    ) from last_err


def _gen(contents, **cfg):
    last_res = None
    for _, res, _ in _gen_stream(contents, **cfg):
        if res is not None:
            last_res = res
    return last_res


def _file_parts(f):
    body = f["text"] if f["text"] is not None else types.Part.from_bytes(data=f["pdf"], mime_type="application/pdf")
    return [f"=== ファイル: {f['name']} ===", body]


# ---------------- 資料係（抜粋・要約）: 本命が使えないときは2案を作って人が選ぶ ----------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_WHO = "GPT-OSS（Groq）"
GROQ_MAX_CHARS = int(os.getenv("GROQ_MAX_CHARS", "80000"))  # 131Kトークンに余裕を残した、資料の文字数の目安


def groq_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@cache
def groq_client():
    from groq import Groq  # Groqを使わない環境ではimport不要
    return Groq(timeout=120)  # GROQ_API_KEY


def _groq_chat(system: str, user: str, schema=None) -> str:
    cl = groq_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = dict(model=GROQ_MODEL, messages=msgs, max_completion_tokens=16000)
    if schema is None:
        r = cl.chat.completions.create(**kw)
    else:
        sch = schema.model_json_schema()
        try:
            r = cl.chat.completions.create(
                **kw, response_format={"type": "json_schema", "json_schema": {"name": "handoff", "schema": sch}})
        except Exception as e:  # noqa: BLE001  スキーマ指定が通らない場合は、プロンプトにスキーマを書いてjson_objectで
            logger.warning("Groq: json_schemaが使えないためjson_objectで再試行: %s", e)
            msgs[0]["content"] = system + "\n\n出力は、次のJSON Schemaに従うJSONだけにしてください:\n" + json.dumps(sch, ensure_ascii=False)
            r = cl.chat.completions.create(**kw, response_format={"type": "json_object"})
    text = (r.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("GPT-OSSの応答が空でした")
    return text


def _user_text(files: list[dict], hint: str) -> str:
    body = "\n".join(f"=== ファイル: {f['name']} ===\n{f['text']}" for f in files)
    return f"{body}\n=== ユーザーの相談（下書き。空のこともある） ===\n{hint or '（未入力）'}"


def _groq_doc(kind: str, files: list[dict], hint: str, source: str) -> tuple[str, str]:
    """Groq GPT-OSSで資料を作る。(doc, model)。使えない条件では理由つきでRuntimeError。"""
    if any(f["text"] is None for f in files):
        raise RuntimeError("PDFなどテキストでない資料はGPT-OSSの対象外です（txt/mdのみ）")
    if len(source) > GROQ_MAX_CHARS:
        raise RuntimeError(f"資料が長いためGPT-OSSは対象外です（{len(source):,}文字 > 目安{GROQ_MAX_CHARS:,}文字）")
    user = _user_text(files, hint)
    if kind == "handoff":
        data = Handoff.model_validate_json(_groq_chat(HANDOFF_PROMPT, user, Handoff))
        header = C.HANDOFF_HEADER.replace("Gemini", GROQ_WHO)  # 作ったのはGeminiではないので、ヘッダーの表記を合わせる
        return header + json.dumps(data.model_dump(), ensure_ascii=False, indent=1), GROQ_MODEL
    return SUMMARY_HEADER.replace("Gemini", GROQ_WHO) + _groq_chat(SUMMARY_PROMPT, user), GROQ_MODEL


def _gemini_stream(kind: str, files: list[dict], hint: str, models=None):
    """(status, doc, model)。kindは 'handoff'（抜粋）か 'summary'（要約）。"""
    parts = [p for f in files for p in _file_parts(f)]
    parts.append(f"=== ユーザーの相談（下書き。空のこともある） ===\n{hint or '（未入力）'}")
    if kind == "handoff":
        cfg = dict(system_instruction=HANDOFF_PROMPT, response_mime_type="application/json", response_schema=Handoff)
    else:
        cfg = dict(system_instruction=SUMMARY_PROMPT)
    for status, res, model in _gen_stream(parts, models=models, **cfg):
        if res is None:
            yield status, None, None
        elif kind == "handoff":
            data = res.parsed.model_dump() if res.parsed else json.loads(res.text)
            yield status, C.HANDOFF_HEADER + json.dumps(data, ensure_ascii=False, indent=1), model
        else:
            yield status, SUMMARY_HEADER + res.text, model


def _last_doc(gen) -> tuple[str, str]:
    doc = model = None
    for _, d, m in gen:
        if d is not None:
            doc, model = d, m
    return doc, model


def _with_compare(kind: str, files: list[dict], hint: str):
    """本命（先頭のGeminiモデル）が使えればその1案。使えなければ、次のGeminiモデルとGroq GPT-OSSの2案を作り、
    選択用の dict（{"choose": True, ...}）を結果として返す。どちらか片方しか作れなければ、その1案を使う。
    Groqが未設定なら、従来どおり残りのGeminiモデルで続ける。"""
    models = get_candidate_models()
    try:
        yield from _gemini_stream(kind, files, hint, models=models)
        return
    except RuntimeError as e:  # 401/403（キーの不備）は_gen_streamがそのまま投げるので、ここには来ない
        primary_err = e
    rest = models[1:]
    if not groq_enabled():
        if not rest:
            raise
        yield from _gemini_stream(kind, files, hint, models=rest)
        return

    yield f"{models[0]} が使えないため、{' / '.join(rest) or 'Gemini'} と {GROQ_WHO} の2案を作ります…", None, None
    source = "\n\n".join(f["text"] or "" for f in files)
    errors_: list[str] = []
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        jobs = []
        if rest:
            jobs.append(("Gemini", ex.submit(lambda: _last_doc(_gemini_stream(kind, files, hint, models=rest)))))
        jobs.append((GROQ_WHO, ex.submit(_groq_doc, kind, files, hint, source)))
        for who, fut in jobs:
            try:
                doc, model = fut.result()
                if not doc:
                    raise RuntimeError("空の結果でした")
                out.append(dict(who=who, model=model, label=f"{who}（{model}）", doc=doc))
            except Exception as e:  # noqa: BLE001
                logger.warning("2案作成: %s が失敗: %s", who, e)
                errors_.append(f"{who}: {e}")
    if len(out) == 2:
        yield "2案ができました", dict(choose=True, kind=kind, options=out, source=source, errors=errors_), "compare"
    elif len(out) == 1:
        o = out[0]
        yield f"{errors_[0]} のため、{o['label']} だけを使います", o["doc"], o["model"]
    else:
        raise RuntimeError(f"{models[0]} も、代わりの2案も作れませんでした（{primary_err} / {' / '.join(errors_)}）")


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", t))


def _numbers(t: str) -> set[str]:
    t = unicodedata.normalize("NFKC", t).replace(",", "")
    return set(re.findall(r"\d+(?:\.\d+)?", t))


def compare_stats(kind: str, doc: str, source: str) -> dict:
    """2案を見比べるための客観的な目安。内容の良し悪しそのものは人が読んで判断する。"""
    st: dict = {"chars": len(doc)}
    if kind == "handoff":
        m = re.search(r'\{\s*"relevant_files"', doc)
        try:
            data = json.loads(doc[m.start():]) if m else {}
        except json.JSONDecodeError:
            data = {}
        evs = [e.get("text", "") for rf in data.get("relevant_files", []) for e in rf.get("evidence", [])]
        src = _norm(source)
        st.update(evidence=len(evs), verbatim=sum(1 for e in evs if _norm(e) and _norm(e) in src),
                  uncertainties=len(data.get("uncertainties", [])), full_read=bool(data.get("full_read_required")))
    else:
        body = doc.split("\n\n", 1)[-1]
        st["unknown_numbers"] = sorted(_numbers(body) - _numbers(source), key=float)[:15]
    return st


def make_handoff_stream(files: list[dict], hint: str):
    yield from _with_compare("handoff", files, hint)


def make_handoff(files: list[dict], hint: str) -> str:
    return _last_doc(_gemini_stream("handoff", files, hint))[0]


SUMMARY_PROMPT = """あなたは資料係です。指定ファイルを、後続の相談で使えるよう要約してください。
- 数値・日付・固有名詞・決定事項は落とさず、正確に残す。
- ファイルごとに見出しを分け、元の章立てが分かるようにする。
- 原文にない推測は書かない。読み取れなかった箇所は末尾に「不明点」として書く。
- 相談文の下書きがあれば関連の深い部分を厚めにする。ただし無関係に見える部分も1〜2行は残す。"""

SUMMARY_HEADER = ("以下は資料をGeminiが要約したもの。原文ではなく、細部は省略されている可能性がある。"
                  "正確な数値や文言が必要な場合は、要約では確認できないことをユーザーに伝えること。\n\n")


def make_summary_stream(files: list[dict], hint: str):
    yield from _with_compare("summary", files, hint)


def make_summary(files: list[dict], hint: str) -> str:
    return _last_doc(_gemini_stream("summary", files, hint))[0]


def make_fulltext_stream(files: list[dict]):
    out = []
    last_model = None
    for f in files:
        text = f["text"]
        if text is None:  # PDFは忠実にテキスト化（出力が長大な場合は途中で切れ得る）
            for status, res, model in _gen_stream(
                [types.Part.from_bytes(data=f["pdf"], mime_type="application/pdf"),
                 "このPDFの全文を、要約・改変せず忠実にテキスト化してください。図表は[図: 内容]の形で簡潔に。"]
            ):
                if res is None:
                    yield f"{f['name']}: {status}", None, None
                else:
                    text = res.text
                    last_model = model
        out.append(f'<file name="{f["name"]}">\n{text}\n</file>')
    doc = C.FULLTEXT_HEADER + "\n".join(out)
    yield "完了", doc, (last_model or "PDFテキスト抽出なし")


def make_fulltext(files: list[dict]) -> str:
    for _, doc, _ in make_fulltext_stream(files):
        if doc is not None:
            return doc


def summarize_history_stream(turns: list[tuple[str, str]]):
    log = "\n".join(f"{'ユーザー' if r == 'user' else 'アシスタント'}: {c}" for r, c in turns)
    for status, res, model in _gen_stream(
        f"次の会話ログを、後続の相談に必要な事実・決定・論点・ユーザーの意向を落とさず簡潔に要約してください。\n\n{log}"
    ):
        if res is None:
            yield status, None, None
        else:
            yield status, res.text, model


# スレッド名の自動生成。軽く速いliteモデルを順に試す（環境変数 GEMINI_TITLE_MODEL を指定すればそれを最優先）
TITLE_MODELS = ([os.environ["GEMINI_TITLE_MODEL"]] if os.getenv("GEMINI_TITLE_MODEL") else []) + [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TITLE_PROMPT = (
    "相談内容の題名を日本語で付けてください。\n"
    "- 20文字以内。名詞中心で、後から一覧で見て何の相談か分かるようにする。\n"
    "- 題名だけを1行で出力する。記号の飾り、かぎ括弧、説明は付けない。"
)


def make_title(question: str, doc_names: list[str] | None = None) -> str | None:
    """相談文からスレッド名を作る。全候補が失敗・空の場合はNoneを返す（呼び出し側が先頭切り出しで代用する）。
    Gemini 3系ではthinking_budgetもtemperatureも使わない（3.5 Flash-Liteは既定で最小の思考）。"""
    body = (question or "").strip()[:600]
    if doc_names:
        body += f"\n（添付資料: {' / '.join(doc_names)[:200]}）"
    if not body:
        return None
    for model in TITLE_MODELS:
        try:
            res = gemini().models.generate_content(
                model=model, contents=body,
                config=types.GenerateContentConfig(system_instruction=TITLE_PROMPT, max_output_tokens=200),
            )
            lines = (res.text or "").strip().splitlines()
            title = lines[0].strip(" 「」『』\"'\u3000#*") if lines else ""
            if title:
                return title[:30]
        except Exception as e:  # noqa: BLE001  題名のために相談を止めない。次の候補へ
            logger.warning("スレッド名の生成に失敗（%s）: %s", model, e)
    return None


def summarize_history(turns: list[tuple[str, str]]) -> str:
    for _, text, _ in summarize_history_stream(turns):
        if text is not None:
            return text


# ---------------- Claude（相談・推論） ----------------
def cc(ttl: str | None) -> dict:
    return {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}


def build_system(docs: list[str] | None, ttl: str) -> list[dict]:
    """system → 資料1 → 資料2 … の順。資料は追加方式で、古い資料の並びと中身は変えない。
    キャッシュ境界は新しい側の2つの資料に置く（上限4個のうち、履歴用に1個を残す）。"""
    blocks = [{"type": "text", "text": C.BASE_SYSTEM}] if C.BASE_SYSTEM else []  # 空のテキストブロックはAPIが拒否する
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
    kw = dict(model=C.MODELS[model]["id"], messages=messages, max_tokens=max_tokens)
    if system:  # 資料なし＋BASE_SYSTEM空のときはsystemを送らない
        kw["system"] = system
    if effort:
        kw["thinking"] = {"type": "adaptive"}
        kw["extra_body"] = {"output_config": {"effort": effort}}
    else:
        kw["thinking"] = C.THINKING_OFF
    return kw


def warm_cache(model: str, docs: list[str], history, ttl: str):
    """出力を最小にしたリクエストで、資料と履歴をキャッシュへ書き込む（資料を追加した直後に使う）。"""
    r = claude().messages.create(
        model=C.MODELS[model]["id"], max_tokens=1, system=build_system(docs, ttl),
        messages=build_messages(history, "資料の読み込み確認", ttl), thinking=C.THINKING_OFF,
    )
    return r.usage


def _count(model_id, system, messages) -> int:
    kw = dict(model=model_id, messages=messages)
    if system:
        kw["system"] = system
    return claude().messages.count_tokens(**kw).input_tokens


def estimate(model: str, system, messages, max_tokens: int, ttl: str) -> dict:
    m = C.MODELS[model]
    total = _count(m["id"], system, messages)
    prefix = _count(m["id"], system, [{"role": "user", "content": "."}]) if system else 0
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
