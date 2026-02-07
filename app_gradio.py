import gradio as gr
import pandas as pd
from pypdf import PdfReader
from personaplex_agents import agent_graph

from faster_whisper import WhisperModel
import soundfile as sf
import tempfile
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===============================
# CONFIG
# ===============================
CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

# ===============================
# WHISPER
# ===============================
whisper = WhisperModel("small", device="cpu", compute_type="int8")

def audio_to_text(audio):
    if audio is None:
        return ""
    sr, data = audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr)
        segments, _ = whisper.transcribe(f.name)
        return " ".join(seg.text for seg in segments).strip()

# ===============================
# FILE READER
# ===============================
def read_file(file):
    if not file or not hasattr(file, "name"):
        return ""

    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        reader = PdfReader(file.name)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == "txt":
        return open(file.name, "r", encoding="utf-8", errors="ignore").read()
    if ext == "csv":
        return pd.read_csv(file.name).to_string()
    return ""

# ===============================
# CHAT STORAGE
# ===============================
def new_chat():
    return {
        "chat_id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }

def save_chat(chat):
    with open(CHAT_DIR / f"{chat['chat_id']}.json", "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2, ensure_ascii=False)

def load_chat(chat_id):
    path = CHAT_DIR / f"{chat_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_chats():
    chats = []
    for file in CHAT_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            chat = json.load(f)
            chats.append((chat.get("title", "New Chat"), chat["chat_id"]))
    return chats



# ===============================
# CORE CHAT HANDLER
# ===============================
def chat_handler(message, state, file):
    if not message:
        return state, state

    document_text = read_file(file)

    try:
        result = agent_graph.invoke({
            "task": message,
            "file_name": file.name if file else "",
            "file_path": file.name if file else "",
            "file_content": document_text[:12000],
        })


        last = result["messages"][-1]
        answer = last.content if hasattr(last, "content") else last["content"]

    except Exception as e:
        answer = f"❌ Agent error: {e}"

    state.append({"role": "user", "content": message})
    state.append({"role": "assistant", "content": answer})

    return state, state



# ===============================
# LOAD CHAT
# ===============================
def load_selected_chat(chat_id):
    if not chat_id:
        chat = new_chat()
        return [], chat

    chat = load_chat(chat_id)
    if not chat:
        chat = new_chat()

    clean = []
    for m in chat["messages"]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            clean.append({"role": m["role"], "content": str(m["content"])})

    chat["messages"] = clean
    return messages_to_chatbot(clean), chat

def start_new_chat():
    chat = new_chat()
    save_chat(chat)
    return [], chat, list_chats()

# ===============================
# EXPORT
# ===============================
def get_last_answer(chat):
    for m in reversed(chat["messages"]):
        if m["role"] == "assistant":
            return m["content"]
    return ""

def export_pdf(text):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(file.name, pagesize=A4)
    x, y = 40, A4[1] - 40
    for line in text.split("\n"):
        c.drawString(x, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = A4[1] - 40
    c.save()
    return file.name

def export_txt(text):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    f.write(text)
    f.close()
    return f.name

def export_csv(text):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["Answer"])
    for line in text.split("\n"):
        writer.writerow([line])
    f.close()
    return f.name

def export_answer(chat, fmt):
    ans = get_last_answer(chat)
    if not ans:
        return None
    if fmt == "PDF":
        return export_pdf(ans)
    if fmt == "TXT":
        return export_txt(ans)
    if fmt == "CSV":
        return export_csv(ans)

# ===============================
# UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 PersonaPlex — Document Chatbot")

    current_chat = gr.State([])

    with gr.Row():
        export_type = gr.Dropdown(["PDF", "TXT", "CSV"], value="PDF")
        export_btn = gr.Button("⬇️ Download")
    file_output = gr.File()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Upload PDF / TXT / CSV")
            audio_input = gr.Audio(sources=["microphone"], type="numpy")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            with gr.Row():
                text_input = gr.Textbox(placeholder="Ask something…", scale=8)
                send_btn = gr.Button("➤")

        with gr.Column(scale=1):
            chat_list = gr.Radio(choices=list_chats(), label="Previous chats")
            new_chat_btn = gr.Button("➕ New chat")

    audio_input.change(audio_to_text, audio_input, text_input)

    send_btn.click(
        chat_handler,
        inputs=[text_input, current_chat, file_input],
        outputs=[chatbot, current_chat]
    )

    text_input.submit(
        chat_handler,
        [text_input, current_chat, file_input],
        [chatbot, current_chat]
    )

    export_btn.click(export_answer, [current_chat, export_type], file_output)

    chat_list.change(load_selected_chat, chat_list, [chatbot, current_chat])

    new_chat_btn.click(start_new_chat, None, [chatbot, current_chat, chat_list])

demo.launch()
