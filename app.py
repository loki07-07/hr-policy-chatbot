"""HR Policy Assistant — Main Streamlit App."""
from __future__ import annotations
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from config.config import (
    GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
    DEFAULT_GROQ_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_GEMINI_MODEL,
    EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    RAG_RELEVANCE_THRESHOLD,
    APP_TITLE, APP_ICON,
)
from models.embeddings import load_embedding_model
from models.llm import get_llm_client
from utils.chat_history import init_history, add_message, get_history_for_llm, clear_history
from utils.prompt import build_system_prompt
from utils.agents import HRAgent
from utils.citations import build_sources_note
from utils.rag import build_vector_store
from utils.db import init_db, create_user, authenticate_user, list_users, update_user_role
from utils.export import export_conversation_txt

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)


def _init_state() -> None:
    init_history(st.session_state)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "user" not in st.session_state:
        st.session_state.user = None


_init_state()
init_db()


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder(model_name: str):
    return load_embedding_model(model_name)


_LLM_MODELS = {
    "Groq": ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
}
_LLM_KEYS = {"Groq": GROQ_API_KEY, "OpenAI": OPENAI_API_KEY, "Gemini": GEMINI_API_KEY}


@st.cache_resource(show_spinner=False)
def _get_llm(provider: str, api_key: str, model_name: str):
    return get_llm_client(provider, api_key, model_name)[0]


with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")

    st.subheader("🤖 LLM Provider")
    provider = st.selectbox(
        "Provider",
        ["Groq", "OpenAI", "Gemini"],
        index=0,
        label_visibility="collapsed",
    )

    api_key_override = st.text_input(
        f"{provider} API Key (optional override)",
        type="password",
        placeholder="Leave blank to use env variable",
    )

    model_name = st.selectbox("Model", _LLM_MODELS[provider])
    effective_key = api_key_override or _LLM_KEYS[provider]
    llm_client = _get_llm(provider, effective_key, model_name)

    st.markdown("---")

    st.subheader("📝 Response Mode")
    response_mode = st.radio(
        "Mode",
        ["Concise", "Detailed"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="Concise: 2–4 sentences. Detailed: full explanation with examples.",
    )

    st.markdown("---")

    if st.session_state.user is not None:

        st.subheader("📂 Upload HR Policy Documents")
        uploaded_files = st.file_uploader(
            "PDF or TXT files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_names = sorted([f.name for f in uploaded_files])
            if new_names != st.session_state.indexed_files:
                with st.spinner("Indexing documents…"):
                    try:
                        if st.session_state.embedding_model is None:
                            st.session_state.embedding_model = _load_embedder(EMBEDDING_MODEL_NAME)
                        st.session_state.vector_store = build_vector_store(
                            uploaded_files,
                            st.session_state.embedding_model,
                            chunk_size=CHUNK_SIZE,
                            overlap=CHUNK_OVERLAP,
                        )
                        st.session_state.indexed_files = new_names
                        st.success(f"Indexed {len(new_names)} file(s) ✓")
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")
                        logger.error("Indexing error: %s", exc)
        else:
            if st.session_state.vector_store is not None:
                st.session_state.vector_store = None
                st.session_state.indexed_files = []

        if st.session_state.indexed_files:
            st.caption("Indexed files:")
            for name in st.session_state.indexed_files:
                st.caption(f"• {name}")

        st.subheader("🌐 Web Search")
        web_search_enabled = st.checkbox(
            "Enable live web search",
            value=True,
            help="Uses DuckDuckGo to supplement your HR documents when needed.",
        )

        st.markdown("---")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            clear_history(st.session_state)
            st.rerun()

        if st.button("🚪 Sign out", use_container_width=True):
            st.session_state.user = None
            st.session_state.display_name = None
            st.session_state.role = None
            clear_history(st.session_state)
            st.session_state.vector_store = None
            st.session_state.indexed_files = []
            st.rerun()
    else:
        web_search_enabled = False

st.title(f"{APP_ICON} {APP_TITLE}")

if st.session_state.user is None:
    st.caption("Please sign in or sign up to use the HR Policy Assistant.")
    login_tab, signup_tab = st.tabs(["Sign in", "Sign up"])

    with login_tab:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in"):
            user = authenticate_user(login_email, login_password)
            if user:
                st.session_state.user = user
                st.session_state.display_name = user["name"]
                st.session_state.role = user.get("role", "Employee")
                st.success(f"Welcome back, {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with signup_tab:
        signup_name = st.text_input("Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")

        def _is_valid_email(email: str) -> bool:
            return "@" in email and "." in email.split("@")[-1]

        def _is_strong_password(pw: str) -> bool:
            return len(pw) >= 8

        if st.button("Sign up"):
            if not (signup_name and signup_email and signup_password):
                st.error("Please fill in name, email, and password.")
            elif not _is_valid_email(signup_email):
                st.error("Please enter a valid email address.")
            elif not _is_strong_password(signup_password):
                st.error("Password must be at least 8 characters long.")
            else:
                ok = create_user(signup_name, signup_email, signup_password, role="Employee")
                if ok:
                    st.success("Account created. You can now sign in.")
                else:
                    st.error("An account with that email already exists.")

    st.stop()


display_name = st.session_state.get("display_name", st.session_state.user["name"])
current_role = st.session_state.user.get("role", "Employee")

st.caption(
    f"Signed in as **{display_name}** ({current_role}) — "
    "welcome back to your HR Policy Assistant."
)

if current_role == "HR":
    chat_tab, admin_tab = st.tabs(["💬 Chat", "🛠 HR Dashboard"])
else:
    chat_tab, = st.tabs(["💬 Chat"])
    admin_tab = None

with chat_tab:
    st.markdown(
        """
        <style>
        /* Keep chat input pinned at bottom, offset from sidebar */
        .stChatInput {
            position: fixed;
            bottom: 0;
            left: 21rem; /* approximate sidebar width */
            right: 0;
            background: #0e1117;
            padding: 12px;
            border-top: 1px solid #444;
            z-index: 1000;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }

        /* Prevent content from hiding behind the input */
        .main .block-container {
            padding-bottom: 120px;
        }

        /* Extra space in chat message container */
        [data-testid="stChatMessageContainer"] {
            padding-bottom: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        txt = export_conversation_txt(st.session_state.messages)
        st.download_button("Download as TXT", data=txt, file_name="hr_chat.txt", mime="text/plain", key="dl_txt")

    if not st.session_state.messages:
        add_message(
            st.session_state,
            "assistant",
            (
                f"Hi {display_name}! 👋\n\n"
                "Upload your company's HR policy documents on the left and ask me anything — "
                "leave entitlements, benefits, code of conduct, legal compliance, and more."
            ),
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources & external information"):
                    st.markdown(msg["sources"])

    user_input = st.chat_input("Ask about HR policies, leave, benefits, compliance…")

    if user_input:
        if llm_client is None:
            st.error(
                f"⚠️ {provider} client could not be initialised. "
                "Please check your API key in the sidebar or environment variables."
            )
            st.stop()

        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model…"):
                st.session_state.embedding_model = _load_embedder(EMBEDDING_MODEL_NAME)

        lowered = user_input.strip().lower()
        if lowered.startswith("my name is "):
            new_name = user_input.strip()[11:].strip()
            if new_name:
                st.session_state.display_name = new_name

        add_message(st.session_state, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        normalized_q = user_input.strip().lower().rstrip(" ?.!")
        if normalized_q in {"what is my name", "what's my name", "who am i"}:
            name = st.session_state.get("display_name", st.session_state.user["name"])
            final_answer = f"Your name is {name}."
            steps, doc_contexts, web_results = [], [], []
            sources_note = ""

            with st.chat_message("assistant"):
                st.markdown(final_answer)
        else:
            role = current_role
            system_prompt = build_system_prompt(response_mode, role=role)

            # Get prior history (exclude the message we just added)
            history_for_llm = get_history_for_llm(st.session_state)[:-1]

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        agent = HRAgent(
                            vector_store=st.session_state.vector_store,
                            embedding_model=st.session_state.embedding_model,
                            llm_provider=provider,
                            llm_client=llm_client,
                            llm_model=model_name,
                            web_search_enabled=web_search_enabled,
                            top_k=TOP_K_RESULTS,
                            rag_confidence_threshold=RAG_RELEVANCE_THRESHOLD,
                        )
                        final_answer, steps, doc_contexts, web_results = agent.run(
                            user_query=user_input,
                            chat_history=history_for_llm,
                            system_prompt=system_prompt,
                        )
                    except Exception as exc:
                        final_answer = f"⚠️ An unexpected error occurred: {exc}"
                        steps, doc_contexts, web_results = [], [], []
                        logger.error("Agent run failed: %s", exc)

                st.markdown(final_answer)

                sources_note = build_sources_note(doc_contexts, web_results)
                if sources_note:
                    with st.expander("📎 Sources & external information"):
                        st.markdown(sources_note)

                if steps:
                    with st.expander("🔍 Agent reasoning trace"):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**Step {i} — {step.tool_used}**")
                            st.caption(f"💭 {step.thought}")
                            if step.tool_output:
                                st.caption(f"📤 {step.tool_output}")
                            st.markdown("")

        add_message(
            st.session_state,
            "assistant",
            final_answer,
            sources=sources_note or None,
        )

if current_role == "HR" and admin_tab is not None:
    with admin_tab:
        st.subheader("🛠 HR Dashboard — Manage user roles")
        st.caption("Only HR can change user roles. Changes apply the next time users sign in.")

        users = list_users()
        if not users:
            st.info("No users found.")
        else:
            labels = [f"{u['id']}: {u['name']} ({u['email']}) — {u['role']}" for u in users]
            selected_label = st.selectbox("Select a user", labels)
            selected_index = labels.index(selected_label)
            selected_user = users[selected_index]

            st.write(f"**Current role:** {selected_user['role']}")
            new_role = st.selectbox(
                "New role",
                ["Employee", "Manager", "HR"],
                index=["Employee", "Manager", "HR"].index(selected_user["role"]) if selected_user["role"] in ["Employee", "Manager", "HR"] else 0,
            )

            if st.button("Update role", type="primary"):
                if new_role == selected_user["role"]:
                    st.info("Role is unchanged.")
                else:
                    ok = update_user_role(selected_user["id"], new_role)
                    if ok:
                        st.success(f"Updated {selected_user['name']}'s role to {new_role}.")
                    else:
                        st.error("Failed to update role. Please try again.")
