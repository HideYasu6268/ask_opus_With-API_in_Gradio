"""個人相談エージェント（Gradio）。仕様書 v1 のフェーズ1〜4を実装。"""
import os

import gradio as gr

import config as C
import services as S

DIRECT, VIA_DRIVE = "Claudeに直接", "Gemini → Drive → Claude"
SEARCH, FULL = "関連箇所を検索", "ファイル全体を読む"
TTL = C.CACHE_TTL


def new_state():
    return dict(handoff=None, prep_model=None, history=[], last_think=None)


def length_choices(model):
    m = C.MODELS[model]
    return [(f"{k}（最大 約{n * m['out'] / 1e6 * C.USD_JPY:.0f}円）", k) for k, n in C.LENGTH.items()]


def fmt_est(e, model, notes=()):
    lines = [f"**送信前の概算**（{model} / 入力 {e['tokens']:,} トークン / 出力上限 {e['max_tokens']:,}）",
             f"- キャッシュなし（初回・期限切れ）: 入力 {S.yen(e['miss'])} → 最大 {S.yen(e['miss'] + e['out'])}",
             f"- キャッシュヒット時: 入力 {S.yen(e['hit'])} → 最大 {S.yen(e['hit'] + e['out'])}"]
    lines += [f"- ⚠️ {n}" for n in notes]
    return "\n".join(lines)


def estimate_now(question, mode, model, length, think, state):
    handoff = state["handoff"] if mode == VIA_DRIVE else None
    system = S.build_system(handoff, TTL)
    msgs = S.build_messages(state["history"], question or "（相談文）", TTL if handoff else None)
    kw = S.request_kwargs(model, system, msgs, length, think)
    return fmt_est(S.estimate(model, system, msgs, kw["max_tokens"], TTL), model)


def prepare(urls, read_mode, model, length, think, hint, state):
    """資料準備ステップ。実行中もブラウザ側の入力は続けられる。"""
    state.update(handoff=None, prep_model=None)
    links = [u.strip() for u in urls.splitlines() if u.strip()]
    if not links:
        yield "Driveの共有リンクを貼ってください。", state
        return
    try:
        yield "📥 Driveから取得中…", state
        files = [S.fetch_drive_file(u) for u in links]
        yield f"🔎 {'Geminiが関連箇所を抽出中…' if read_mode == SEARCH else '全文をテキスト化中…'}", state
        handoff = S.make_handoff(files, hint) if read_mode == SEARCH else S.make_fulltext(files)
        yield "💾 Claudeのキャッシュへ書き込み中…", state
        u = S.warm_cache(model, handoff, TTL)
        state.update(handoff=handoff, prep_model=model)
        written = u.cache_creation_input_tokens or 0
        note = f"キャッシュ書き込み {written:,} トークン" if written else "⚠️ キャッシュに書き込まれていません（資料が最小キャッシュ長に満たない可能性）"
        est = estimate_now(hint, VIA_DRIVE, model, length, think, state)
        yield f"✅ 資料の準備ができました。相談を送信できます。（{', '.join(f['name'] for f in files)} / {note}）\n\n{est}", state
    except Exception as e:  # noqa: BLE001
        yield f"❌ 資料準備に失敗: {e}", state


def consult(question, mode, model, length, think, state, chat):
    question = (question or "").strip()
    if not question:
        yield chat, state, "相談内容を入力してください。", question
        return
    handoff = state["handoff"] if mode == VIA_DRIVE else None
    if mode == VIA_DRIVE and not handoff:
        yield chat, state, "先に「資料を準備」を実行してください。", question
        return

    notes, hist = [], state["history"]
    if hist and state["last_think"] not in (None, think):
        notes.append("思考設定を変えたため、会話履歴部分のキャッシュが切れます（資料部分は残る見込み）")
    if handoff and state["prep_model"] != model:
        notes.append("資料準備時と別モデルのため、資料のキャッシュは引き継がれず今回書き込み直しになります")
    if sum(len(c) for _, c in hist) > C.COMPRESS_CHARS and len(hist) > C.KEEP_TURNS * 2:
        old, recent = hist[: -C.KEEP_TURNS * 2], hist[-C.KEEP_TURNS * 2:]
        summary = S.summarize_history(old)  # 古いターンのみ要約。直近は原文のまま
        hist = [("user", f"【これまでの会話の要約】\n{summary}"), ("assistant", "了解しました。")] + recent
        state["history"] = hist
        notes.append("履歴を要約したため、今回はキャッシュが1回切れます")

    system = S.build_system(handoff, TTL)
    msgs = S.build_messages(hist, question, TTL if handoff else None)
    kw = S.request_kwargs(model, system, msgs, length, think)
    head = fmt_est(S.estimate(model, system, msgs, kw["max_tokens"], TTL), model, notes)

    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]
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
        yield chat, state, head, question
        return

    if final.stop_reason == "max_tokens":
        buf += "\n\n…（出力上限に達したため途中で切れています）"
        chat[-1]["content"] = buf
    u = final.usage
    state["history"] = hist + [("user", question), ("assistant", buf)]
    state["last_think"] = think
    actual = (f"\n\n**実費**: {S.yen(S.actual_cost(u, model, TTL))}"
              f"（通常入力 {u.input_tokens:,} / キャッシュ書込 {u.cache_creation_input_tokens or 0:,} / "
              f"キャッシュ読取 {u.cache_read_input_tokens or 0:,} / 出力 {u.output_tokens:,}）")
    yield chat, state, head + actual, ""


def reset(state):
    return new_state(), [], "", ""


with gr.Blocks(title="個人相談エージェント") as demo:
    state = gr.State(new_state)
    chat = gr.Chatbot(type="messages", height=420, label="相談")
    question = gr.Textbox(label="相談内容", lines=4)
    with gr.Row():
        mode = gr.Radio([DIRECT, VIA_DRIVE], value=DIRECT, label="回答モード")
        model = gr.Radio(list(C.MODELS), value="Sonnet", label="モデル")
    with gr.Group(visible=False) as drive_box:
        urls = gr.Textbox(label="Driveファイル（共有リンクのURL。複数は改行区切り）", lines=2)
        read_mode = gr.Radio([SEARCH, FULL], value=SEARCH, label="資料の読み方")
        prep_btn = gr.Button("資料を準備（Gemini → キャッシュ書き込み）")
        prep_status = gr.Markdown()
    with gr.Row():
        length = gr.Radio(length_choices("Sonnet"), value="標準", label="回答の長さ")
        think = gr.Radio(list(C.THINK), value="オフ", label="思考の深さ")
    with gr.Row():
        send = gr.Button("送信", variant="primary")
        est_btn = gr.Button("概算料金を見積もる")
        reset_btn = gr.Button("会話をリセット")
    cost = gr.Markdown()

    mode.change(lambda m: gr.update(visible=m == VIA_DRIVE), mode, drive_box)
    model.change(lambda m: gr.update(choices=length_choices(m)), model, length)
    prep_in = [urls, read_mode, model, length, think, question, state]
    prep_btn.click(prepare, prep_in, [prep_status, state])
    urls.submit(prepare, prep_in, [prep_status, state])  # リンクを入れてEnterで即開始
    est_btn.click(estimate_now, [question, mode, model, length, think, state], cost)
    c_in, c_out = [question, mode, model, length, think, state, chat], [chat, state, cost, question]
    send.click(consult, c_in, c_out)
    reset_btn.click(reset, state, [state, chat, cost, prep_status])

if __name__ == "__main__":
    pw = os.getenv("APP_PASSWORD")  # IAPに加える二重認証（任意）
    demo.queue().launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "8080")),
        auth=(os.getenv("APP_USER", "me"), pw) if pw else None,
    )
