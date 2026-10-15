"""スレッド（相談ごとの資料・履歴・設定）の保存。

- THREAD_BUCKET があればGCSに保存（Cloud Run用）。1スレッド = threads/<id>.json
- なければローカルの THREAD_DIR（既定 ./threads）に保存（ローカル開発用。.gitignoreに入れること）

一覧に使うタイトルはGCSのカスタムメタデータに持たせるので、一覧表示のたびに
資料入りの大きなJSONを読み込むことはない。
"""
import json
import os
import re
import time
import uuid
from functools import cache
from pathlib import Path

BUCKET = os.getenv("THREAD_BUCKET")
LOCAL_DIR = Path(os.getenv("THREAD_DIR", "threads"))
PREFIX = "threads/"
DEFAULT_TITLE = "新しい相談"
_ID = re.compile(r"[\w-]+")


def new_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:6]}"  # 時刻順に並ぶ


def make_title(text: str, n: int = 24) -> str:
    t = " ".join((text or "").split())
    return (t[:n] + "…") if len(t) > n else (t or DEFAULT_TITLE)


def _check(tid: str) -> str:
    if not tid or not _ID.fullmatch(tid):  # パス区切り等を弾く
        raise ValueError(f"不正なスレッドID: {tid!r}")
    return tid


@cache
def _bucket():
    from google.cloud import storage  # ローカルでGCSを使わないときはimport不要
    return storage.Client().bucket(BUCKET)


def _local(tid: str) -> Path:
    return LOCAL_DIR / f"{_check(tid)}.json"


def save(state: dict) -> None:
    state["updated"] = time.time()
    data = json.dumps({k: v for k, v in state.items() if k != "pending"}, ensure_ascii=False, default=str)  # 選びかけの2案は保存しない
    if BUCKET:
        blob = _bucket().blob(f"{PREFIX}{_check(state['id'])}.json")
        blob.metadata = {"title": state["title"]}
        blob.upload_from_string(data, content_type="application/json")
    else:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        p = _local(state["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(data, "utf-8")
        tmp.replace(p)  # 書き込み途中で壊れたファイルを残さない


def save_safe(state: dict) -> str | None:
    """保存に失敗しても相談自体は止めない。失敗時はエラー文を返す。"""
    try:
        save(state)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def load(tid: str) -> dict:
    if BUCKET:
        return json.loads(_bucket().blob(f"{PREFIX}{_check(tid)}.json").download_as_text("utf-8"))
    return json.loads(_local(tid).read_text("utf-8"))


def delete(tid: str) -> None:
    if BUCKET:
        _bucket().blob(f"{PREFIX}{_check(tid)}.json").delete()
    else:
        _local(tid).unlink(missing_ok=True)


def list_threads(limit: int = 50) -> list[dict]:
    """新しい順。各要素は {id, title, updated}。"""
    rows = []
    if BUCKET:
        for b in _bucket().client.list_blobs(BUCKET, prefix=PREFIX):
            if b.name.endswith(".json"):
                rows.append(dict(id=b.name[len(PREFIX):-5], title=(b.metadata or {}).get("title") or "（無題）",
                                 updated=b.updated.timestamp()))
    elif LOCAL_DIR.exists():
        for p in LOCAL_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text("utf-8"))
                rows.append(dict(id=p.stem, title=d.get("title") or "（無題）", updated=p.stat().st_mtime))
            except Exception:  # noqa: BLE001
                continue
    rows.sort(key=lambda r: r["updated"], reverse=True)
    return rows[:limit]
