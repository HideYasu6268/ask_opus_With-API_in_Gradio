"""ask_opus_With_API_in_Gradio（Gradio）。仕様書 v1 のフェーズ1〜4を実装。"""
import os
import time
from datetime import datetime, timedelta, timezone

import gradio as gr

import config as C
import services as S
import threads as T

SEARCH, SUMMARY, FULL = "関連箇所を抜粋", "要約", "ファイル全体を読む"
BUSY = {SEARCH: "Geminiが関連箇所を抽出中…", SUMMARY: "Geminiが要約中…", FULL: "全文をテキスト化中…"}
MAKE_STREAM = {
    SEARCH: S.make_handoff_stream,
    SUMMARY: S.make_summary_stream,
    FULL: S.make_fulltext_stream,
}
MAKE = {SEARCH: lambda f, h: S.make_handoff(f, h),
        SUMMARY: lambda f, h: S.make_summary(f, h),
        FULL: lambda f, h: S.make_fulltext(f)}
TTL = C.CACHE_TTL


def new_state():
    """1スレッド分の状態。そのままJSONでGCS/ローカルに保存される（tupleはlistになるが、履歴の扱いは同じ）。"""
    return dict(docs=[], doc_names=[], prep_model=None, history=[], last_think=None, cache_ts=None, ids=[], has_attach=False,
                id=T.new_id(), title=T.DEFAULT_TITLE, titled=False, chat=[], model=None,
                pending=None)  # pending: 2案から選ぶ前の資料（保存しない）


JST = timezone(timedelta(hours=9))  # Cloud RunはUTCなので、一覧の日時は日本時間に固定


def thread_choices():
    try:
        rows = T.list_threads()
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"スレッド一覧を取得できません: {e}")
        return []
    return [(f"{r['title']}（{datetime.fromtimestamp(r['updated'], JST):%m/%d %H:%M}）", r["id"]) for r in rows]


def thread_update(state):
    """一覧を最新にし、保存済みの現スレッドなら選択状態にする。"""
    ch = thread_choices()
    sid = state["id"] if isinstance(state, dict) else None  # ページ読み込み直後はStateの初期化前のことがあるので、dict以外は未選択扱い
    return gr.update(choices=ch, value=sid if sid in {v for _, v in ch} else None)


def save_note(state):
    err = T.save_safe(state)
    return f"\n\n⚠️ スレッドの保存に失敗しました（この相談は画面上では続けられます）: {err}" if err else ""


def length_choices(model):
    m = C.MODELS[model]
    return [(f"{k}(最大 約{n * m['out'] / 1e6 * C.USD_JPY:.0f}円)", k) for k, n in C.LENGTH.items()]


def ttl_sec():
    return 3600 if TTL == "1h" else 300


def think_change_note(model, state):
    """思考の深さを変えたとき、切れる履歴の大きさと追加コストの目安を示す。"""
    n = sum(len(c) for _, c in state["history"])  # 文字数≒トークン数の概算（日本語は1文字≒1トークン前後）
    m = C.MODELS[model]
    extra = n * ((m["w1h"] if TTL == "1h" else m["w5m"]) - m["read"]) / 1e6
    return (f"思考の深さを変えたため、会話履歴部分（約{n:,}トークン）のキャッシュが切れます。"
            f"追加コストの目安は{S.yen(extra)}です（資料部分は残る見込みですが未実測）")


def cache_status(model, think, state):
    """資料キャッシュの現在状態と、切れる操作への警告を常時表示する。"""
    if not state["docs"]:
        return ""
    left = ttl_sec() - (time.time() - (state["cache_ts"] or 0))
    out = [f"📎 渡している資料: {' / '.join(state['doc_names'])}"]
    out += [f"✅ 資料キャッシュ有効（残り約{left / 60:.0f}分。使うたびに延長）" if left > 0
           else "⏱ 資料キャッシュは期限切れ。次の送信で書き込み直し（「キャッシュなし」の料金）"]
    if state["prep_model"] != model:
        out.append(f"⚠️ {state['prep_model']}で準備した資料です。{model}では引き継がれず、次の送信で書き込み直しになります")
    if state["history"] and state["last_think"] not in (None, think):
        out.append(f"⚠️ {think_change_note(model, state)}")
    return "\n\n".join(out)


def fmt_est(e, model, notes=()):
    lines = [f"**送信前の概算**（{model} / 入力 {e['tokens']:,} トークン / 出力上限 {e['max_tokens']:,}）",
             f"- キャッシュなし（初回・期限切れ）: 入力 {S.yen(e['miss'])} → 最大 {S.yen(e['miss'] + e['out'])}",
             f"- キャッシュヒット時: 入力 {S.yen(e['hit'])} → 最大 {S.yen(e['hit'] + e['out'])}"]
    lines += [f"- ⚠️ {n}" for n in notes]
    return "\n".join(lines)


def unadded(links, state):
    """まだ資料に入っていないリンク。同じファイルは、URLの形が違っても（view/edit等）追加済みとみなす。"""
    seen, out = set(state["ids"]), []
    for u in links:
        try:
            i = S.file_id(u)
        except ValueError:
            i = None  # IDを取り出せないものは弾かず、取得時のエラーで知らせる
        if i is None or i not in seen:
            out.append(u)
            if i:
                seen.add(i)
    return out


def estimate_now(question, model, length, think, state):
    notes = []
    links, question = S.drive_links(question)  # リンクはClaudeに送らないので見積もりからも除く
    if links:
        notes.append("メッセージ欄のDriveリンクの資料は、送信時に取得するため、この概算には含まれていません")
    docs = state["docs"]
    system = S.build_system(docs, TTL)
    use_cache = bool(docs) or state["has_attach"]
    msgs = S.build_messages(state["history"], question or "（相談文）", TTL if use_cache else None)
    kw = S.request_kwargs(model, system, msgs, length, think)
    return fmt_est(S.estimate(model, system, msgs, kw["max_tokens"], TTL), model, notes)


def commit(doc, names, ids, label, model, length, think, hint, state):
    """確定した資料1つをClaudeのキャッシュに書き込み、stateに足して保存する。表示用の文言を順にyieldする。"""
    docs = state["docs"] + [doc]  # 成功するまでstateは変えない（失敗しても今までの資料は残る）
    yield "💾 Claudeのキャッシュへ書き込み中…"
    u = S.warm_cache(model, docs, state["history"], TTL)
    state.update(docs=docs, doc_names=state["doc_names"] + names, prep_model=model, cache_ts=time.time(),
                 ids=state["ids"] + ids, model=model, pending=None)
    if not state["titled"]:
        state["title"] = T.make_title(state["doc_names"][0])  # 質問を送るまでは資料名を仮の題にする
    saved_note = save_note(state)
    written, read = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
    note = (f"今回の追加コスト（実費）{S.yen(S.actual_cost(u, model, TTL))} / "
            f"キャッシュ書込 {written:,}・読取 {read:,} トークン")
    if not written:
        note = "⚠️ キャッシュに書き込まれていません（資料が最小キャッシュ長に満たない可能性）"
    est = estimate_now(hint, model, length, think, state)
    yield f"✅ 資料を追加しました {label}（{', '.join(names)}）。{note}\n\n{est}{saved_note}"


def prepare(urls, read_mode, model, length, think, question, state):
    """資料の追加ステップ（system側に常駐）。リンク欄のDriveファイルを資料にし、メッセージ欄の本文を抽出の基準（hint）にする。
    古い資料は残し、新しい資料を後ろに足す。実行中もブラウザ側の入力は続けられる。
    Gemini本命が使えないときは2案（Gemini別モデル / Groq）を作り、state["pending"]に置いて人が選ぶ。"""
    links = S.drive_links(urls)[0]
    hint = S.drive_links(question)[1]  # 相談文の下書き。メッセージ欄のリンクは基準に含めない
    if not links:
        yield "Driveの共有リンクを貼ってください。", state
        return
    links = unadded(links, state)
    if not links:
        yield "貼られているリンクはすべて追加済みです。", state
        return
    try:
        state["pending"] = None  # 前回の選びかけは破棄する
        yield "📥 Driveから取得中…", state
        files = [S.fetch_drive_file(u) for u in links]
        doc = choice = None
        label = ""
        stream_fn = MAKE_STREAM[read_mode]
        args = (files, hint) if read_mode != FULL else (files,)
        for status_msg, result, used_model in stream_fn(*args):
            if result is None:
                yield f"🔎 Gemini: {status_msg}", state
            elif isinstance(result, dict) and result.get("choose"):
                choice = result
            else:
                doc, label = result, f"（{used_model}）"
        names, ids = [f["name"] for f in files], [S.file_id(u) for u in links]
        if choice:
            src = choice.pop("source")
            for o in choice["options"]:
                o["stats"] = S.compare_stats(choice["kind"], o["doc"], src)
            choice.update(names=names, ids=ids, hint=hint)
            state["pending"] = choice
            fails = f"（{' / '.join(choice['errors'])}）" if choice["errors"] else ""
            yield f"⚖️ 本命のGeminiが使えなかったため、2案を作りました{fails}。下で見比べて、Claudeに渡す方を選んでください。", state
            return
        for msg in commit(doc, names, ids, label, model, length, think, hint, state):
            yield msg, state
    except Exception as e:  # noqa: BLE001
        yield f"❌ 資料の追加に失敗: {e}", state


def choose(idx, model, length, think, state):
    p = state.get("pending")
    if not p:
        yield "選ぶ案がありません。", state
        return
    o = p["options"][idx]
    try:
        for msg in commit(o["doc"], p["names"], p["ids"], f"（{o['label']}を選択）", model, length, think, p["hint"], state):
            yield msg, state
    except Exception as e:  # noqa: BLE001  失敗しても2案は残るので、選び直せる
        yield f"❌ 資料の追加に失敗: {e}", state


def choose_a(model, length, think, state):
    yield from choose(0, model, length, think, state)


def choose_b(model, length, think, state):
    yield from choose(1, model, length, think, state)


def discard_pending(state):
    state["pending"] = None
    return "2案を破棄しました。資料は追加していません。", state


def _row(label, a, b):
    return f"| {label} | {a} | {b} |"


def format_compare(p):
    """2案の見比べ表。数字は目安であり、中身は下のタブで読んで判断する。"""
    a, b = p["options"]
    sa, sb = a["stats"], b["stats"]
    rows = ["| | A | B |", "|---|---|---|", _row("作成", a["who"], b["who"]), _row("モデル", a["model"], b["model"]),
            _row("文字数", f"{sa['chars']:,}", f"{sb['chars']:,}")]
    if p["kind"] == "handoff":
        rows += [_row("原文一致の抜粋", f"{sa['verbatim']}/{sa['evidence']}", f"{sb['verbatim']}/{sb['evidence']}"),
                 _row("不確かな点", sa["uncertainties"], sb["uncertainties"]),
                 _row("全文が必要?", "はい" if sa["full_read"] else "いいえ", "はい" if sb["full_read"] else "いいえ")]
        tip = "「原文一致の抜粋」は、抜粋が原文にそのまま存在する数です。少ないと、言い換えや創作が混じっている可能性があります。"
    else:
        na = ", ".join(sa["unknown_numbers"]) or "なし"
        nb = ", ".join(sb["unknown_numbers"]) or "なし"
        rows.append(_row("原文にない数字", na, nb))
        tip = "「原文にない数字」は、要約にあって原文に出てこない数字です。計算で出した数字の可能性もあるので、該当箇所を確認してください。"
    return "\n".join(rows) + f"\n\n{tip} 数字は目安です。中身は下のタブで読んで選んでください。"


def show_choice(state):
    """2案から選ぶ画面の表示/非表示と中身。"""
    p = state.get("pending") if isinstance(state, dict) else None
    if not p:
        return gr.update(visible=False), "", "", ""
    a, b = p["options"]
    return gr.update(visible=True), format_compare(p), a["doc"], b["doc"]


def consult(question, read_mode, model, length, think, state, chat):
    raw = (question or "").strip()  # エラー時に入力欄へ戻す元の文章（リンク込み）
    attach_links, rest = S.drive_links(raw)  # メッセージ欄のDriveリンクは、このターンの添付（履歴側）になる
    question = rest if attach_links else raw
    if not question:
        yield chat, state, "相談内容を入力してください。", raw
        return
    handoff = state["docs"]  # リンク欄から追加した資料（system側）

    attach, names = None, []
    if attach_links:
        try:
            yield chat, state, "📥 Driveから取得中…", raw
            files = [S.fetch_drive_file(u) for u in attach_links]
            stream_fn = MAKE_STREAM[read_mode]
            args = (files, question) if read_mode != FULL else (files,)
            for status_msg, result, used_model in stream_fn(*args):
                if result is None:
                    yield chat, state, f"🔎 Gemini: {status_msg}", raw
                elif isinstance(result, dict) and result.get("choose"):
                    yield chat, state, ("⚠️ 本命のGeminiが使えないため、添付資料は2案から選ぶ必要があります。"
                                        "リンクを「資料」欄に貼って追加し、2案を見比べて選んでから送信してください（入力は元に戻しました）。"), raw
                    return
                else:
                    attach = result
            names = [f["name"] for f in files]
        except Exception as e:  # noqa: BLE001
            yield chat, state, f"❌ 資料の取得に失敗: {e}", raw
            return
    turn = f"【添付資料: {' / '.join(names)}】\n{attach}\n\n【質問】\n{question}" if attach else question
    use_cache = bool(handoff) or state["has_attach"] or bool(attach)

    notes, hist = [], state["history"]
    if attach:
        notes.append("今回添付した資料はキャッシュへ書き込まれます。書込料金のぶん、この概算より高くなります")
    if hist and state["last_think"] not in (None, think):
        notes.append(think_change_note(model, state))
    idle = time.time() - (state["cache_ts"] or 0)
    if handoff and state["prep_model"] != model:
        notes.append("資料準備時と別モデルのため、資料のキャッシュは引き継がれず今回書き込み直しになります")
    elif handoff and idle > ttl_sec():
        notes.append(f"資料キャッシュが期限切れ（最後の使用から約{idle / 60:.0f}分）のため、今回は書き込み直しになります")
    if sum(len(c) for _, c in hist) > C.COMPRESS_CHARS and len(hist) > C.KEEP_TURNS * 2:
        old, recent = hist[: -C.KEEP_TURNS * 2], hist[-C.KEEP_TURNS * 2:]
        summary = None
        for status_msg, result, _ in S.summarize_history_stream(old):  # 古いターンのみ要約。直近は原文のまま
            if result is None:
                yield chat, state, f"🔁 履歴を圧縮中… Gemini: {status_msg}", raw
            else:
                summary = result
        hist = [("user", f"【これまでの会話の要約】\n{summary}"), ("assistant", "了解しました。")] + recent
        state["history"] = hist
        notes.append("履歴を要約したため、今回はキャッシュが1回切れます")

    system = S.build_system(handoff, TTL)
    msgs = S.build_messages(hist, turn, TTL if use_cache else None, cache_last=bool(attach))
    kw = S.request_kwargs(model, system, msgs, length, think)
    head = fmt_est(S.estimate(model, system, msgs, kw["max_tokens"], TTL), model, notes)

    shown = question + (f"\n\n📎 {' / '.join(names)}" if names else "")
    chat = chat + [{"role": "user", "content": shown}, {"role": "assistant", "content": ""}]
    yield chat, state, head, ""
    try:
        buf = ""
        with S.claude().messages.stream(**kw) as stream:
            for t in stream.text_stream:
                buf += t
                chat[-1]["content"] = buf
                yield chat, state, head, ""
            final = stream.get_final_message()
    except Exception as e:  # noqa: BLE001
        chat[-1]["content"] = f"❌ エラー: {e}"
        yield chat, state, head, raw
        return

    if final.stop_reason == "max_tokens":
        buf += "\n\n…（出力上限に達したため途中で切れています）"
        chat[-1]["content"] = buf
    u = final.usage
    state["history"] = hist + [("user", turn), ("assistant", buf)]
    state["has_attach"] = state["has_attach"] or bool(attach)
    state["last_think"] = think
    if handoff:  # この送信でキャッシュが書き込まれた/延長された。モデルを変えた場合は新モデルが基準になる
        state.update(cache_ts=time.time(), prep_model=model)
    if not state["titled"]:
        title = S.make_title(question, state["doc_names"]) or T.make_title(question)  # Gemini lite。失敗時は先頭切り出し
        state.update(title=title, titled=True)
    state.update(chat=chat, model=model)
    saved_note = save_note(state)  # 同期で保存する（Cloud Runは応答後にCPUが絞られるため、バックグラウンドにしない）
    actual = (f"\n\n**実費**: {S.yen(S.actual_cost(u, model, TTL))}"
              f"（通常入力 {u.input_tokens:,} / キャッシュ書込 {u.cache_creation_input_tokens or 0:,} / "
              f"キャッシュ読取 {u.cache_read_input_tokens or 0:,} / 出力 {u.output_tokens:,}）")
    yield chat, state, head + actual + saved_note, ""


# ---------------- スレッド管理 ----------------
def close_sidebar():
    return gr.Sidebar(open=False)  # スマホでは開いたままだと画面を覆うので、選んだら閉じる


# 開く/新規/削除が更新する画面部品: state, chat, 資料状態, 資料欄の状況, 料金欄, モデル, 思考, 名前欄, サイドバー
def open_thread(tid, state):
    same = (state,) + (gr.update(),) * 8
    if not tid or tid == state["id"]:
        return same[:-1] + (close_sidebar(),)
    try:
        st = {**new_state(), **T.load(tid)}  # 古い形式で保存されたスレッドでも、足りない項目は初期値で補う
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"スレッドを開けませんでした: {e}")
        return same
    m, th = st["model"] or st["prep_model"] or "Opus", st["last_think"] or "高"
    return st, st["chat"], cache_status(m, th, st), "", "", m, th, st["title"], close_sidebar()


def new_thread():
    st = new_state()
    return st, [], "", "", "", thread_update(st), st["title"], close_sidebar()


def rename_thread(name, state):
    name = (name or "").strip()
    if not name:
        gr.Warning("スレッド名を入力してください")
        return state, gr.update()
    state.update(title=name[:60], titled=True)
    if state["docs"] or state["history"]:  # 中身のないスレッドは保存しない
        err = T.save_safe(state)
        if err:
            gr.Warning(f"保存に失敗しました: {err}")
    return state, thread_update(state)


def delete_thread(confirm, state):
    if not confirm:
        gr.Warning("削除するには「削除を確認」にチェックを入れてください")
        return (state,) + (gr.update(),) * 8
    if state["docs"] or state["history"]:
        try:
            T.delete(state["id"])
        except Exception as e:  # noqa: BLE001
            gr.Warning(f"削除に失敗しました: {e}")
            return (state,) + (gr.update(),) * 8
    st = new_state()
    return st, [], "", "", "", thread_update(st), st["title"], False, close_sidebar()


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Noto Sans JP"), "Hiragino Sans", "system-ui", "sans-serif"],
).set(
    body_background_fill="#faf9f5", body_background_fill_dark="#262624",
    block_background_fill="#faf9f5", block_background_fill_dark="#262624",
    button_primary_background_fill="#c96442", button_primary_background_fill_hover="#b5573a",
    button_primary_text_color="white",
)
CSS = """
.gradio-container{max-width:760px !important;margin:0 auto !important}
footer{display:none !important}
#title{color:#8a8880;font-size:13px;text-align:center;padding:4px 0}
#chat{border:none !important;box-shadow:none !important;background:transparent !important}
#chat [data-testid="bot"]{background:transparent !important;border:none !important;
  font-family:"Hiragino Mincho ProN","Noto Serif JP",serif;line-height:1.75}
/* ユーザー発言: ライトは淡いベージュ、ダークは濃いグレー。文字色も明示して読めなくなるのを防ぐ */
#chat .bubble-wrap{background:transparent !important}
#chat .message-bubble-border,#chat .flex-wrap.user{background:transparent !important;border:none !important;box-shadow:none !important}
#chat [data-testid="user"]{background:#f0eee6 !important;border:none !important;border-radius:18px;color:#141413 !important}
#chat [data-testid="user"] :is(p,span,div,li){color:inherit !important}
.dark #chat [data-testid="user"]{background:#30302e !important;color:#faf9f5 !important}
.dark #composer{background:#30302e !important;border-color:#3d3d3a !important}
.dark #composer textarea{color:#faf9f5 !important}
#composer{border:1px solid #e3e0d6;border-radius:22px;padding:8px 12px;
  box-shadow:0 2px 10px rgba(0,0,0,.05);background:#fff}
#composer textarea{border:none !important;box-shadow:none !important;background:transparent !important}
#tools{align-items:center;flex-wrap:nowrap !important;gap:4px !important}
#send{border-radius:50% !important;min-width:40px !important;max-width:40px;height:40px;padding:0}
/* 左サイドバー: 一覧は縦に並べ、ラジオの丸は隠して「選択中だけ色が付く行」にする */
#newchat{width:100%}
#threads .wrap{flex-direction:column !important;flex-wrap:nowrap !important;gap:2px !important}
#threads label{width:100%;border:none !important;background:transparent !important;box-shadow:none !important;
  padding:9px 10px !important;border-radius:8px;white-space:normal !important;text-align:left;font-size:14px;line-height:1.4}
#threads label:hover{background:#f0eee6 !important}
#threads label.selected{background:#e8e5da !important}
.dark #threads label:hover{background:#30302e !important}
.dark #threads label.selected{background:#3d3d3a !important}
#threads input[type=radio]{display:none !important}
#length .wrap{flex-wrap:nowrap !important;gap:6px !important}
#length label{padding:6px 10px !important;font-size:13px !important;white-space:nowrap !important}
@media (max-width:520px){#length .wrap{flex-wrap:wrap !important}}
"""


with gr.Blocks(title="ask_opus_With_API_in_Gradio", theme=THEME, css=CSS) as demo:
    state = gr.State(new_state)  # 開いた直後は、保存されていない新規チャット

    # --- 左サイドバー（claude.ai風）: 新規チャットと、保存済みチャットの一覧 ---
    with gr.Sidebar(open=False, position="left", width=300) as sidebar:
        new_btn = gr.Button("＋ 新規チャット", variant="primary", elem_id="newchat")
        thread_dd = gr.Radio([], value=None, label="最近のチャット", interactive=True, elem_id="threads")

    gr.Markdown("ask_opus_With_API_in_Gradio", elem_id="title")
    chat = gr.Chatbot(type="messages", height="60vh", show_label=False, elem_id="chat")

    # --- claude.ai風: 入力欄の中にモデル・思考の深さ・送信 ---
    with gr.Group(elem_id="composer"):
        question = gr.Textbox(show_label=False, container=False, lines=2, max_lines=8,
                              placeholder="メッセージを入力（Driveの共有リンクも貼れます。送信は右下の↑ボタン）")
        with gr.Row(elem_id="tools"):
            model = gr.Dropdown(list(C.MODELS), value="Opus", show_label=False, container=False,
                                scale=0, min_width=110, interactive=True)
            think = gr.Dropdown([(f"思考：{k}", k) for k in C.THINK], value="高", show_label=False,
                                container=False, scale=0, min_width=110, interactive=True)
            gr.HTML("<div style='flex:1'></div>")
            send = gr.Button("↑", variant="primary", elem_id="send", scale=0)

    # --- 独自機能は下にまとめる（順序: 料金・詳細設定 → 資料 → スレッド管理） ---
    with gr.Accordion("料金・詳細設定", open=True):
        length = gr.Radio(length_choices("Opus"), value="標準", label="回答の長さ", elem_id="length")
        est_btn = gr.Button("概算料金を見積もる")
        cost = gr.Markdown()
    with gr.Accordion("資料", open=True):
        urls = gr.Textbox(label="Driveリンク（複数は改行区切り）", lines=2)
        read_mode = gr.Radio([SEARCH, SUMMARY, FULL], value=SEARCH, label="追加方法")
        prep_btn = gr.Button("資料を追加")
        prep_status = gr.Markdown()
        with gr.Group(visible=False) as choose_box:  # 本命のGeminiが使えないときだけ現れる
            gr.Markdown("**2案を見比べて、Claudeに渡す方を選んでください**")
            cmp_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("A（Gemini）"):
                    cand_a = gr.Textbox(show_label=False, lines=12, max_lines=24, interactive=False, show_copy_button=True)
                    use_a = gr.Button("Aを使う", variant="primary")
                with gr.Tab("B（GPT-OSS）"):
                    cand_b = gr.Textbox(show_label=False, lines=12, max_lines=24, interactive=False, show_copy_button=True)
                    use_b = gr.Button("Bを使う", variant="primary")
            discard_btn = gr.Button("どちらも使わない（破棄）")
        cache_md = gr.Markdown()
    with gr.Accordion("スレッド管理（名前の変更・削除）", open=False):
        rename_box = gr.Textbox(label="スレッド名", value=T.DEFAULT_TITLE)
        rename_btn = gr.Button("名前を変更")
        del_ok = gr.Checkbox(label="削除を確認（元に戻せません）", value=False)
        del_btn = gr.Button("このスレッドを削除", variant="stop")

    refresh_in = [model, think, state]
    for c in (model, think):
        c.change(cache_status, refresh_in, cache_md)
    model.change(lambda m: gr.update(choices=length_choices(m)), model, length)
    prep_in = [urls, read_mode, model, length, think, question, state]
    choice_out = [choose_box, cmp_md, cand_a, cand_b]
    prep_btn.click(prepare, prep_in, [prep_status, state]).then(cache_status, refresh_in, cache_md) \
        .then(thread_update, state, thread_dd).then(show_choice, state, choice_out)
    choose_in = [model, length, think, state]
    for btn, fn in ((use_a, choose_a), (use_b, choose_b)):
        btn.click(fn, choose_in, [prep_status, state]).then(cache_status, refresh_in, cache_md) \
            .then(thread_update, state, thread_dd).then(show_choice, state, choice_out)
    discard_btn.click(discard_pending, state, [prep_status, state]).then(show_choice, state, choice_out)
    est_btn.click(estimate_now, [question, model, length, think, state], cost)
    c_in, c_out = [question, read_mode, model, length, think, state, chat], [chat, state, cost, question]
    send.click(consult, c_in, c_out).then(cache_status, refresh_in, cache_md).then(thread_update, state, thread_dd)

    # スレッド操作。一覧は.input（人が選んだときだけ）で開く。プログラムからの値更新で再発火させないため
    view_out = [state, chat, cache_md, prep_status, cost, model, think, rename_box, sidebar]
    thread_dd.input(open_thread, [thread_dd, state], view_out).then(show_choice, state, choice_out)
    new_btn.click(new_thread, None, [state, chat, cache_md, prep_status, cost, thread_dd, rename_box, sidebar]) \
        .then(show_choice, state, choice_out)
    rename_btn.click(rename_thread, [rename_box, state], [state, thread_dd])
    del_btn.click(delete_thread, [del_ok, state],
                  [state, chat, cache_md, prep_status, cost, thread_dd, rename_box, del_ok, sidebar]) \
        .then(show_choice, state, choice_out)
    demo.load(lambda: thread_update(None), None, thread_dd)  # ページを開いたときに一覧を読む（stateは使わない: 初期化と競合するため）

if __name__ == "__main__":
    pw = os.getenv("APP_PASSWORD")  # IAPに加える二重認証（任意）
    demo.queue().launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "8080")),
        auth=(os.getenv("APP_USER", "me"), pw) if pw else None,
    )
