# PersonaPlex — Multi-Agent Document Chatbot

PersonaPlex is an AI-powered **document chatbot** that lets users chat with PDFs, TXT, and CSV files using a **multi-agent architecture**.  
It supports **chat history like ChatGPT**, **voice input**, and **exporting responses** to PDF, TXT, and CSV.

Built with **Gradio**, **Whisper**, and a custom **agent graph**, PersonaPlex focuses on real-world usability and clean UX.

---

## Features

- 📄 Chat with **PDF / TXT / CSV** documents
- 🧠 **Multi-agent reasoning** using `agent_graph`
- 💬 **Persistent chat history** (ChatGPT-style)
- 🎙️ **Voice input** using Whisper (speech → text)
- 📥 Export answers as **PDF, TXT, CSV**
- ➕ Create & switch between multiple chats
- 🔒 Secure API key handling using `.env`

---

## Tech Stack

- **Python**
- **Gradio** – UI
- **Whisper (faster-whisper)** – Speech to text
- **ReportLab** – PDF generation
- **Pandas** – CSV handling
- **Multi-agent system** (custom `agent_graph`)
- **GitHub Secret Scanning safe** (no keys committed)


