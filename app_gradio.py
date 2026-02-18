import gradio as gr
import pandas as pd
from pypdf import PdfReader
from personaplex_agents import agent_graph
import soundfile as sf
from faster_whisper import WhisperModel


import tempfile
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4

whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


# ===============================
# CONFIG
# ===============================
CHAT_DIR = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

# ===============================
# HELPERS
# ===============================

def audio_to_text(audio):
    if audio is None:
        return ""

    sr, data = audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr)
        segments, _ = whisper.transcribe(f.name)

    return " ".join(seg.text for seg in segments).strip()

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
def chat_handler(message, chat, file):
    if not message:
        return chat["messages"], chat

    file_content = read_file(file)

    chat["messages"].append({
        "role": "user",
        "content": message
    })

    from pathlib import Path

    try:
        file_name = ""
        file_path = ""

        if file:
            file_name = Path(file.name).name
            file_path = file.name

        result = agent_graph.invoke({
            "task": message,
            "messages": chat["messages"],
            "file_name": file_name,
            "file_path": file_path,
            "file_content": file_content[:12000]
        })


        msgs = result.get("messages", [])
        if msgs:
            last = msgs[-1]
            answer = last["content"] if isinstance(last, dict) else last.content
        else:
            answer = "⚠️ No response."

    except Exception as e:
        answer = f"❌ Agent error: {e}"

    chat["messages"].append({
        "role": "assistant",
        "content": answer
    })

    if chat["title"] == "New Chat":
        chat["title"] = message[:40]

    save_chat(chat)
    return chat["messages"], chat

# ===============================
# CHAT SWITCHING
# ===============================
def load_selected_chat(chat_id):
    chat = load_chat(chat_id)
    if not chat:
        chat = new_chat()
    return chat["messages"], chat


def start_new_chat():
    chat = new_chat()
    save_chat(chat)
    return (
        chat["messages"],
        chat,
        gr.update(choices=list_chats(), value=chat["chat_id"])
    )


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

    doc = SimpleDocTemplate(
        file.name,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    for para in text.split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
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
    return {
        "PDF": export_pdf,
        "TXT": export_txt,
        "CSV": export_csv
    }[fmt](ans)

# ===============================
# UI
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 PersonaPlex — Document Chatbot")

    current_chat = gr.State(new_chat())

    with gr.Row():
        export_type = gr.Dropdown(["PDF", "TXT", "CSV"], value="PDF")
        export_btn = gr.Button("⬇️ Download")
    file_output = gr.File()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF / TXT / CSV")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎤 Speak"
            )


        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            text_input = gr.Textbox(placeholder="Ask something…")
            send_btn = gr.Button("➤")

        with gr.Column(scale=1):
            chat_list = gr.Radio(choices=list_chats(), label="Previous chats")
            new_chat_btn = gr.Button("➕ New chat")

   

    audio_input.change(audio_to_text, audio_input, text_input)

    send_btn.click(chat_handler, [text_input, current_chat, file_input], [chatbot, current_chat])
    text_input.submit(chat_handler, [text_input, current_chat, file_input], [chatbot, current_chat])

    chat_list.change(load_selected_chat, chat_list, [chatbot, current_chat])
    new_chat_btn.click(start_new_chat, None, [chatbot, current_chat, chat_list])

    export_btn.click(export_answer, [current_chat, export_type], file_output)

demo.launch()
