"""ask_opus_With_API_in_Gradio（Gradio）。仕様書 v1 のフェーズ1〜4を実装。"""
import os
import time

import gradio as gr

import config as C
import services as S

SEARCH, SUMMARY, FULL = "関連箇所を抜粋", "要約", "ファイル全体を読む"
BUSY = {SEARCH: "Geminiが関連箇所を抽出中…", SUMMARY: "Geminiが要約中…", FULL: "全文をテキスト化中…"}
MAKE = {SEARCH: lambda f, h: S.make_handoff(f, h),
        SUMMARY: lambda f, h: S.make_summary(f, h),
        FULL: lambda f, h: S.make_fulltext(f)}
TTL = C.CACHE_TTL


def new_state():
    return dict(docs=[], doc_names=[], prep_model=None, history=[], last_think=None, cache_ts=None, ids=[], has_attach=False)


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


def prepare(urls, read_mode, model, length, think, question, state):
    """資料の追加ステップ（system側に常駐）。リンク欄のDriveファイルを資料にし、メッセージ欄の本文を抽出の基準（hint）にする。
    古い資料は残し、新しい資料を後ろに足す。実行中もブラウザ側の入力は続けられる。"""
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
        yield "📥 Driveから取得中…", state
        files = [S.fetch_drive_file(u) for u in links]
        yield f"🔎 {BUSY[read_mode]}", state
        doc = MAKE[read_mode](files, hint)
        docs = state["docs"] + [doc]  # 成功するまでstateは変えない（失敗しても今までの資料は残る）
        yield "💾 Claudeのキャッシュへ書き込み中…", state
        u = S.warm_cache(model, docs, state["history"], TTL)
        state.update(docs=docs, doc_names=state["doc_names"] + [f["name"] for f in files],
                     prep_model=model, cache_ts=time.time(),
                     ids=state["ids"] + [S.file_id(u) for u in links])
        written, read = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
        note = (f"今回の追加コスト（実費）{S.yen(S.actual_cost(u, model, TTL))} / "
                f"キャッシュ書込 {written:,}・読取 {read:,} トークン")
        if not written:
            note = "⚠️ キャッシュに書き込まれていません（資料が最小キャッシュ長に満たない可能性）"
        est = estimate_now(hint, model, length, think, state)
        yield f"✅ 資料を追加しました（{', '.join(f['name'] for f in files)}）。{note}\n\n{est}", state
    except Exception as e:  # noqa: BLE001
        yield f"❌ 資料の追加に失敗: {e}", state


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
            yield chat, state, f"🔎 {BUSY[read_mode]}", raw
            attach = MAKE[read_mode](files, question)  # 抽出・要約の基準は、今送る質問そのもの
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
        summary = S.summarize_history(old)  # 古いターンのみ要約。直近は原文のまま
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
    actual = (f"\n\n**実費**: {S.yen(S.actual_cost(u, model, TTL))}"
              f"（通常入力 {u.input_tokens:,} / キャッシュ書込 {u.cache_creation_input_tokens or 0:,} / "
              f"キャッシュ読取 {u.cache_read_input_tokens or 0:,} / 出力 {u.output_tokens:,}）")
    yield chat, state, head + actual, ""


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
#chat [data-testid="user"]{background:#f0eee6 !important;border:none !important;border-radius:18px}
#composer{border:1px solid #e3e0d6;border-radius:22px;padding:8px 12px;
  box-shadow:0 2px 10px rgba(0,0,0,.05);background:#fff}
#composer textarea{border:none !important;box-shadow:none !important;background:transparent !important}
#tools{align-items:center;flex-wrap:nowrap !important;gap:4px !important}
#send{border-radius:50% !important;min-width:40px !important;max-width:40px;height:40px;padding:0}
#length .wrap{flex-wrap:nowrap !important;gap:6px !important}
#length label{padding:6px 10px !important;font-size:13px !important;white-space:nowrap !important}
@media (max-width:520px){#length .wrap{flex-wrap:wrap !important}}
"""


with gr.Blocks(title="ask_opus_With_API_in_Gradio", theme=THEME, css=CSS) as demo:
    state = gr.State(new_state)
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

    # --- 独自機能は下にまとめる ---
    with gr.Accordion("資料", open=True):
        urls = gr.Textbox(label="Driveリンク（複数は改行区切り）", lines=2)
        read_mode = gr.Radio([SEARCH, SUMMARY, FULL], value=SEARCH, label="追加方法")
        prep_btn = gr.Button("資料を追加")
        prep_status = gr.Markdown()
        cache_md = gr.Markdown()
    with gr.Accordion("料金・詳細設定", open=True):
        length = gr.Radio(length_choices("Opus"), value="標準", label="回答の長さ", elem_id="length")
        est_btn = gr.Button("概算料金を見積もる")
        cost = gr.Markdown()

    refresh_in = [model, think, state]
    for c in (model, think):
        c.change(cache_status, refresh_in, cache_md)
    model.change(lambda m: gr.update(choices=length_choices(m)), model, length)
    prep_in = [urls, read_mode, model, length, think, question, state]
    prep_btn.click(prepare, prep_in, [prep_status, state]).then(cache_status, refresh_in, cache_md)
    urls.submit(prepare, prep_in, [prep_status, state]).then(cache_status, refresh_in, cache_md)  # リンクを入れてEnterで即開始
    est_btn.click(estimate_now, [question, model, length, think, state], cost)
    c_in, c_out = [question, read_mode, model, length, think, state, chat], [chat, state, cost, question]
    send.click(consult, c_in, c_out).then(cache_status, refresh_in, cache_md)

if __name__ == "__main__":
    pw = os.getenv("APP_PASSWORD")  # IAPに加える二重認証（任意）
    demo.queue().launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "8080")),
        auth=(os.getenv("APP_USER", "me"), pw) if pw else None,
    )	