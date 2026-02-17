# PersonaPlex — Multi-Agent Document Chatbot

Using a multi-agent architecture, PersonaPlex is an AI-powered document chatbot that enables users to communicate with PDF, TXT, and CSV files.
Voice input, exporting responses to PDF, TXT, and CSV, and chat history like ChatGPT are all supported.

PersonaPlex is built with Gradio, Whisper, and a custom agent graph with an emphasis on clean UX and real-world usability.

---

## Features

📄 Chat with documents in PDF, TXT, and CSV 
🧠 Agent_graph for multi-agent reasoning
💬 ChatGPT-style persistent chat history
🎙️ Whisper voice input (speech → text)
Answers can be exported as PDF, TXT, or CSV. Establish and navigate between several chats
🔒 Safely managing API keys with.env

---

## Tech Stack

- **Python**
- **Gradio** – UI
- **Whisper (faster-whisper)** – Speech to text
- **ReportLab** – PDF generation
- **Pandas** – CSV handling
- **Multi-agent system** (custom `agent_graph`)
- **GitHub Secret Scanning safe** (no keys committed)


