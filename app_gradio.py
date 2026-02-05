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
            title = chat.get("title", "New Chat")
            chat_id = chat.get("chat_id")
            if chat_id:
                chats.append((title, chat_id))
    return chats



# ===============================
# CHATBOT FORMAT FIX
# ===============================
def messages_to_chatbot(messages):
    pairs = []
    user_msg = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
        elif msg.get("role") == "assistant":
            if user_msg is None:
                # orphan assistant → ignore
                continue
            pairs.append((user_msg, msg.get("content", "")))
            user_msg = None

    return pairs



# ===============================
# CORE CHAT HANDLER
# ===============================
def chat_handler(message, chat, file):
    if not message:
        return messages_to_chatbot(chat["messages"]), chat

    document_text = read_file(file)

    result = agent_graph.invoke({
        "task": message,
        "messages": chat["messages"],
        "file_content": document_text[:12000]
    })

    answer = result["messages"][-1].content

    chat["messages"].append({"role": "user", "content": message})
    chat["messages"].append({"role": "assistant", "content": answer})

    if chat["title"] == "New Chat":
        chat["title"] = message[:40]

    save_chat(chat)

    return messages_to_chatbot(chat["messages"]), chat


def load_selected_chat(chat_id):
    chat = load_chat(chat_id)
    if not chat:
        chat = new_chat()

    return messages_to_chatbot(chat["messages"]), chat




def start_new_chat():
    chat = new_chat()
    save_chat(chat)
    return [], chat, list_chats()



# ===============================
# EXPORT
# ===============================
def get_last_answer(chat):
    for msg in reversed(chat["messages"]):
        if msg["role"] == "assistant":
            return msg["content"]
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
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    file.write(text)
    file.close()
    return file.name

def export_csv(text):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["Answer"])
    for line in text.split("\n"):
        writer.writerow([line])
    file.close()
    return file.name

def export_answer(chat, fmt):
    answer = get_last_answer(chat)
    if not answer:
        return None
    if fmt == "PDF":
        return export_pdf(answer)
    if fmt == "TXT":
        return export_txt(answer)
    if fmt == "CSV":
        return export_csv(answer)

# ===============================
# UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("## 🧠 PersonaPlex — Document Chatbot")

    current_chat = gr.State(new_chat())

    with gr.Row():
        export_type = gr.Dropdown(["PDF", "TXT", "CSV"], value="PDF", label="Export as")
        export_btn = gr.Button("⬇️ Download")
    file_output = gr.File()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Upload PDF / TXT / CSV")
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Speak")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Ask something about your document…",
                    show_label=False,
                    scale=8
                )
                send_btn = gr.Button("➤", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 💬 Chats")
            chat_list = gr.Radio(choices=list_chats(), label="Previous chats")
            new_chat_btn = gr.Button("➕ New chat")




    # -------------------------------
    # EVENT WIRING (IMPORTANT)
    # -------------------------------

    # Voice → Text
    audio_input.change(
        fn=audio_to_text,
        inputs=audio_input,
        outputs=text_input
    )

    # Send button
    send_btn.click(
        fn=chat_handler,
        inputs=[text_input, current_chat, file_input],
        outputs=[chatbot, current_chat]
    ).then(
        fn=lambda: "",
        outputs=text_input
    )



    # Enter key
    text_input.submit(
        fn=chat_handler,
        inputs=[text_input, current_chat, file_input],
        outputs=[chatbot, current_chat]
    )

    export_btn.click(
        fn=export_answer,
        inputs=[current_chat, export_type],
        outputs=file_output
    )

    chat_list.change(
        fn=load_selected_chat,
        inputs=chat_list,
        outputs=[chatbot, current_chat]
    )



    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chatbot, current_chat, chat_list]
    )





demo.launch()
