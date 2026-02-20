---
title: PersonaPlex
emoji: 🧠
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
license: mit
---

# PersonaPlex — Multi-Agent Document Chatbot

PersonaPlex is an AI-powered, multi-agent document chatbot that allows users to interact with **PDF, TXT, and CSV** files using **text or voice**.

It supports **ChatGPT-style chat history**, **voice input**, and **exporting answers** in multiple formats, with a strong focus on clean UX and real-world usability.

---

## 🚀 Live Demo
👉 https://sourabh2012-personaplex.hf.space

---

## ✨ Features

- 📄 Chat with documents (PDF / TXT / CSV)
- 🧠 Multi-agent reasoning via custom `agent_graph`
- 💬 Persistent chat history (ChatGPT-like)
- 🎙️ Voice input using Whisper (speech → text)
- 🔊 AI voice output
- 📤 Export answers as **PDF / TXT / CSV**
- ➕ Create, switch, and delete multiple chats
- 🔒 Safe API key handling (`.env`, no secrets committed)

---

## 🛠️ Tech Stack

- **Python**
- **Gradio** – UI
- **faster-whisper** – Speech to text
- **ReportLab** – PDF generation
- **Pandas** – CSV handling
- **Custom multi-agent system** (`agent_graph`)
- **GitHub secret-scanning safe**

---

## 📦 Installation (Local)

```bash
git clone https://github.com/SOURABH4PAL/PersonaPlex
cd PersonaPlex
pip install -r requirements.txt
python app.py
