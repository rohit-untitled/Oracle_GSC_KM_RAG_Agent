import requests
import streamlit as st
import time

st.set_page_config(page_title="RAG Chat", layout="wide")

st.markdown(
    """
    <style>
    .chat-title { font-size: 22px; font-weight: 600; margin: 0 0 6px 0; }
    .chat-sub { color: #666; font-size: 12px; margin-bottom: 12px; }
    .sidebar-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
    .session-meta { color: #777; font-size: 12px; }
    .stApp { background: #f7f7f8; }
    .block-container { max-width: 60vw; padding-top: 18px; }
    section[data-testid="stSidebar"] { background: #ffffff; min-width: 280px; max-width: 360px; }
    .stChatMessage { background: #ffffff; border: 1px solid #ececec; border-radius: 12px; padding: 12px 14px; }
    .stChatMessage[data-testid="stChatMessage"][data-role="user"] { background: #f2f2f2; }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        white-space: normal;
        line-height: 1.2;
        font-size: 14px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
        padding: 8px 10px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #eef2f7;
        border-color: #d7dde6;
    }
    .km-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
        padding: 8px 6px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #eef2f7;
    }
    .km-mark {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0b1220, #1f2937);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        letter-spacing: 0.6px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.2);
        font-size: 16px;
    }
    .km-name {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .km-tag {
        font-size: 12px;
        color: #64748b;
        letter-spacing: 0.2px;
    }
    .typing {
        display: inline-flex;
        gap: 6px;
        align-items: center;
        padding: 6px 0;
    }
    .typing span {
        width: 6px;
        height: 6px;
        background: #94a3b8;
        border-radius: 50%;
        display: inline-block;
        animation: blink 1.2s infinite;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
        40% { opacity: 1; transform: translateY(-2px); }
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 18px !important;
        border: 1px solid #e5e7eb !important;
        background: #ffffff !important;
        padding: 12px 14px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08) !important;
        min-height: 52px !important;
    }
    .chat-input-hint {
        color: #94a3b8;
        font-size: 12px;
        margin-top: -6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

base_url = "http://127.0.0.1:8000"
timeout = 120


def call_api(method: str, path: str, json_body=None, params=None):
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            timeout=timeout,
        )
        return {
            "status": resp.status_code,
            "ok": resp.ok,
            "body": resp.json() if resp.content else None,
            "text": resp.text,
        }
    except Exception as e:
        return {"status": None, "ok": False, "error": str(e)}


def fetch_sessions(limit: int = 50):
    result = call_api("GET", "/sessions", params={"limit": limit, "offset": 0})
    if result.get("ok"):
        return result.get("body", []) or []
    return []


def fetch_history(session_id: str):
    return call_api("GET", "/session-history", params={"session_id": session_id})


def fetch_chat_models():
    result = call_api("GET", "/chat-models")
    if result.get("ok"):
        body = result.get("body", {}) or {}
        models = body.get("models", []) or []
        if models:
            return models
    return [
        {"key": "cohere", "label": "Cohere (Default)", "is_default": True},
        {"key": "maverick", "label": "Maverick", "is_default": False},
    ]


if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = ""

if "sessions_cache" not in st.session_state:
    st.session_state["sessions_cache"] = []

if "history_cache" not in st.session_state:
    st.session_state["history_cache"] = []

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "edit_text" not in st.session_state:
    st.session_state["edit_text"] = ""

if "chat_models" not in st.session_state:
    st.session_state["chat_models"] = fetch_chat_models()

if "selected_model" not in st.session_state:
    default_model = "cohere"
    for _m in st.session_state["chat_models"]:
        if _m.get("is_default"):
            default_model = _m.get("key", "cohere")
            break
    st.session_state["selected_model"] = default_model

if "response_meta" not in st.session_state:
    st.session_state["response_meta"] = {}


def _turn_key(session_id: str, idx: int, turn: dict) -> str:
    role = turn.get("role", "")
    content = (turn.get("content", "") or "")[:120]
    return f"{session_id}:{idx}:{role}:{content}"


def _attach_latest_assistant_meta(history: list, session_id: str, model_used: str, latency_s: float):
    for idx in range(len(history) - 1, -1, -1):
        turn = history[idx]
        if turn.get("role") == "assistant":
            key = _turn_key(session_id, idx, turn)
            st.session_state["response_meta"][key] = {
                "model_used": model_used or "unknown",
                "latency_s": round(float(latency_s), 2),
            }
            break

with st.sidebar:
    st.markdown(
        """
        <div class="km-logo">
          <div class="km-mark">KM</div>
          <div>
            <div class="km-name">KM Assistant</div>
            <div class="km-tag">Knowledge Chat</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-title'>Chats</div>", unsafe_allow_html=True)
    action_row = st.columns(2)
    with action_row[0]:
        if st.button("New chat"):
            st.session_state["active_session_id"] = ""
            st.session_state["history_cache"] = []
    with action_row[1]:
        if st.button("Refresh"):
            st.session_state["sessions_cache"] = fetch_sessions()
            st.session_state["chat_models"] = fetch_chat_models()

    model_options = st.session_state.get("chat_models", [])
    option_keys = [m.get("key", "") for m in model_options if m.get("key")]
    if not option_keys:
        option_keys = ["cohere", "maverick"]
    model_label_map = {m.get("key"): m.get("label", m.get("key")) for m in model_options}
    if st.session_state.get("selected_model") not in option_keys:
        st.session_state["selected_model"] = option_keys[0]
    selected_model = st.selectbox(
        "Model",
        option_keys,
        index=option_keys.index(st.session_state["selected_model"]),
        format_func=lambda k: model_label_map.get(k, k),
    )
    st.session_state["selected_model"] = selected_model

    if not st.session_state["sessions_cache"]:
        st.session_state["sessions_cache"] = fetch_sessions()

    for s in st.session_state["sessions_cache"]:
        sid = s.get("session_id")
        title = s.get("title") or "Untitled chat"
        label = title
        row = st.columns([0.82, 0.18])
        with row[0]:
            if st.button(label, key=f"session_{sid}"):
                st.session_state["active_session_id"] = sid
                hist = fetch_history(sid)
                st.session_state["history_cache"] = hist.get("body", []) if hist.get("ok") else []
        with row[1]:
            if st.button("❌", key=f"delete_{sid}", help="Delete chat"):
                st.session_state[f"confirm_delete_{sid}"] = True

        if st.session_state.get(f"confirm_delete_{sid}"):
            st.warning("Delete this chat?")
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                if st.button("Confirm", key=f"confirm_{sid}"):
                    result = call_api("DELETE", f"/sessions/{sid}")
                    if result.get("ok"):
                        if st.session_state.get("active_session_id") == sid:
                            st.session_state["active_session_id"] = ""
                            st.session_state["history_cache"] = []
                        st.session_state["sessions_cache"] = fetch_sessions()
                        st.session_state[f"confirm_delete_{sid}"] = False
                        st.rerun()
                    else:
                        st.error(result.get("error") or result.get("text") or "Delete failed")
            with confirm_cols[1]:
                if st.button("Cancel", key=f"cancel_{sid}"):
                    st.session_state[f"confirm_delete_{sid}"] = False


st.markdown("<div class='chat-title'>KM Assistant</div>", unsafe_allow_html=True)

st.divider()

active_session_id = st.session_state.get("active_session_id", "")
if active_session_id:
    active_title = None
    for s in st.session_state["sessions_cache"]:
        if s.get("session_id") == active_session_id:
            active_title = s.get("title") or "Untitled chat"
            break
    if active_title:
        st.markdown(f"**Chat:** {active_title}")
    st.caption(active_session_id)
else:
    st.markdown("**Session:** new chat (auto)")

if st.session_state["history_cache"]:
    for idx, turn in enumerate(st.session_state["history_cache"]):
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        with st.chat_message(role):
            st.write(content)
            if role == "assistant" and active_session_id:
                key = _turn_key(active_session_id, idx, turn)
                meta = st.session_state["response_meta"].get(key)
                if meta:
                    st.caption(
                        f"Model: {meta.get('model_used', 'unknown')} | Time: {meta.get('latency_s', 0)}s"
                    )

        if role == "user":
            edit_cols = st.columns([0.12, 0.88])
            with edit_cols[0]:
                if st.button("Edit", key=f"edit_{idx}"):
                    st.session_state["edit_index"] = idx
                    st.session_state["edit_text"] = content
            with edit_cols[1]:
                if st.session_state.get("edit_index") == idx:
                    edited = st.text_area("Edit question", value=st.session_state.get("edit_text", ""), key=f"edit_text_{idx}")
                    action_cols = st.columns([0.3, 0.3, 0.4])
                    with action_cols[0]:
                        if st.button("Save & Resend", key=f"save_edit_{idx}"):
                            edited_query = edited.strip()
                            if edited_query:
                                payload = {
                                    "query": edited_query,
                                    "top_k": 5,
                                    "model": st.session_state.get("selected_model", "cohere"),
                                }
                                if active_session_id:
                                    payload["session_id"] = active_session_id
                                t0 = time.time()
                                result = call_api("POST", "/ask", json_body=payload)
                                elapsed = time.time() - t0
                                if result.get("ok"):
                                    body = result.get("body", {})
                                    if body.get("session_id"):
                                        st.session_state["active_session_id"] = body["session_id"]
                                    model_used = body.get("model_used", st.session_state.get("selected_model", "cohere"))
                                    hist = fetch_history(st.session_state["active_session_id"])
                                    st.session_state["history_cache"] = hist.get("body", []) if hist.get("ok") else []
                                    _attach_latest_assistant_meta(
                                        st.session_state["history_cache"],
                                        st.session_state["active_session_id"],
                                        model_used,
                                        elapsed,
                                    )
                                    st.session_state["sessions_cache"] = fetch_sessions()
                                    st.session_state["edit_index"] = None
                                    st.session_state["edit_text"] = ""
                                    st.rerun()
                                else:
                                    st.error(result.get("error") or result.get("text") or "Request failed")
                    with action_cols[1]:
                        if st.button("Cancel", key=f"cancel_edit_{idx}"):
                            st.session_state["edit_index"] = None
                            st.session_state["edit_text"] = ""
                            st.rerun()

        # Like/Dislike controls removed per request.
else:
    st.info("Start a new chat or pick a chat from the sidebar.")

query = st.chat_input("Ask a question")
if query:
    with st.chat_message("user"):
        st.write(query)

    payload = {
        "query": query,
        "top_k": 5,
        "model": st.session_state.get("selected_model", "cohere"),
    }
    if active_session_id:
        payload["session_id"] = active_session_id

    with st.chat_message("assistant"):
        thinking_box = st.empty()
        thinking_box.markdown(
            "<div class='typing'><span></span><span></span><span></span></div>",
            unsafe_allow_html=True,
        )
    t0 = time.time()
    result = call_api("POST", "/ask", json_body=payload)
    elapsed = time.time() - t0
    if result.get("ok"):
        body = result.get("body", {})
        if body.get("session_id"):
            st.session_state["active_session_id"] = body["session_id"]
        answer = body.get("answer", "")
        model_used = body.get("model_used", st.session_state.get("selected_model", "cohere"))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            thinking_box.empty()
            delay = 0.004

            rendered = ""
            for ch in answer:
                rendered += ch
                placeholder.markdown(rendered)
                time.sleep(delay)
            st.caption(f"Model: {model_used} | Time: {round(elapsed, 2)}s")

        hist = fetch_history(st.session_state["active_session_id"])
        st.session_state["history_cache"] = hist.get("body", []) if hist.get("ok") else []
        _attach_latest_assistant_meta(
            st.session_state["history_cache"],
            st.session_state["active_session_id"],
            model_used,
            elapsed,
        )
        st.session_state["sessions_cache"] = fetch_sessions()
        st.rerun()
    else:
        st.error(result.get("error") or result.get("text") or "Request failed")

st.divider()
