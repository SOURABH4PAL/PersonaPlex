import os
import streamlit as st
from dotenv import load_dotenv

from utils.exporter import generate_pdf, generate_csv
from personaplex_agents import agent_graph

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pypdf import PdfReader
import pandas as pd

from auth import (
    create_users_table,
    create_chat_tables,
    signup,
    login,
    create_chat,
    get_chats,
    save_message,
    load_chat
)

# ==================================================
# CONFIG
# ==================================================
load_dotenv()
st.set_page_config("PersonaPlex — Multi-Agent Document AI", layout="wide")

# ==================================================
# DATABASE INIT
# ==================================================
create_users_table()
create_chat_tables()

# ==================================================
# SESSION STATE
# ==================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# AUTH
# ==================================================
if st.session_state.user_id is None:
    st.title("🔐 PersonaPlex Login")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login(email, password)
            if user:
                st.session_state.user_id = user[0]
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if signup(new_email, new_password):
                st.success("Account created. Please login.")
            else:
                st.error("User already exists")

    st.stop()

# ==================================================
# USER DIRECTORIES
# ==================================================
USER_ID = str(st.session_state.user_id)
UPLOAD_DIR = f"uploads/{USER_ID}"
VECTOR_DIR = f"chroma_db/{USER_ID}"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ==================================================
# EMBEDDINGS
# ==================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None
file_path = None
raw_text = ""

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("📂 Upload PDF / TXT / CSV")
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "csv"]
    )

    st.divider()
    st.subheader("💬 Chat History")

    if st.button("➕ New Chat"):
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.rerun()

    for cid, title in get_chats(USER_ID):
        if st.button(title, key=f"chat_{cid}"):
            st.session_state.chat_id = cid
            st.session_state.messages = load_chat(cid)
            st.rerun()

# ==================================================
# MAIN
# ==================================================
st.title("🧠 PersonaPlex — Multi-Agent Document AI")

# ==================================================
# FILE INGESTION
# ==================================================
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "pdf":
        reader = PdfReader(file_path)
        raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == "txt":
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    elif ext == "csv":
        raw_text = pd.read_csv(file_path).to_string()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = [Document(page_content=c) for c in splitter.split_text(raw_text)]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
    vectorstore.persist()

    st.success("✅ Document indexed with permanent memory")

# ==================================================
# DISPLAY CHAT
# ==================================================
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

import streamlit.components.v1 as components

# ==================================================
# 🎤 INPUT CONTROLLER (FIXED)
# ==================================================

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.subheader("🎤 Voice Assistant")

voice_text = st.text_input(
    "Speak or type your question",
    key="voice_input"
)

if st.button("🚀 Ask (Voice)", key="ask_voice"):
    if voice_text.strip():
        st.session_state.pending_question = voice_text.strip()

chat_text = st.chat_input(
    "Ask something from your document...",
    key="chat_input"
)

if chat_text:
    st.session_state.pending_question = chat_text


st.subheader("🎙️ Voice Input (Real Mic)")

mic_text = components.html(
    """
    <button onclick="startDictation()">🎤 Start Speaking</button>
    <p id="output"></p>

    <script>
    function startDictation() {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.start();

        recognition.onresult = function(event) {
            const text = event.results[0][0].transcript;
            document.getElementById("output").innerText = text;
            window.parent.postMessage({voice: text}, "*");
        };
    }

    window.addEventListener("message", (event) => {
        if (event.data.voice) {
            window.voiceInput = event.data.voice;
        }
    });
    </script>
    """,
    height=200
)
if "voice_from_mic" not in st.session_state:
    st.session_state.voice_from_mic = None

st.write("🎧 Waiting for voice...")

st.session_state.voice_from_mic = st.experimental_get_query_params().get("voice")

# ==================================================
# RUN AGENT
# ==================================================
question = st.session_state.pending_question

if question:
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.chat_id is None:
        chat_id = create_chat(USER_ID, question[:60])
        st.session_state.chat_id = chat_id
    else:
        chat_id = st.session_state.chat_id

    save_message(chat_id, "user", question)
    st.session_state.messages.append(("user", question))

    result = agent_graph.invoke({
        "task": question,
        "file_name": uploaded_file.name if uploaded_file else "",
        "file_path": file_path if uploaded_file else "",
        "messages": [],
        "file_content": raw_text[:12000] if uploaded_file else ""
    })

    final_answer = result["messages"][-1].content

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 🔊 Voice output
    st.components.v1.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance({final_answer!r});
        msg.lang = "en-US";
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0
    )

    save_message(chat_id, "assistant", final_answer)
    st.session_state.messages.append(("assistant", final_answer))

    # reset so it doesn't rerun infinitely
    st.session_state.pending_question = None
