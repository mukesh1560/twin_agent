# 🪞 AI Twin — A Digital Clone

**AI Twin** is a personal AI chatbot that mimics your personality, voice, and memories. Feed it your personal Q&A and life experiences, and it will respond to people as *you* — using your tone, knowledge, and style.

---

## ✨ Features

- **Personality injection** — loads a JSON file of your personal Q&A to define your voice
- **Memory system** — stores and retrieves your experiences using semantic embedding search
- **RAG-powered chat** — retrieves the most relevant memories before generating each reply
- **"Remember this" command** — type `remember: [text]` to add new memories on the fly
- **Groq LLM** — uses `llama-3.1-8b-instant` via Groq API for fast, realistic responses
- **Local embeddings** — `all-MiniLM-L6-v2` runs on your CPU (no cloud needed)
- **Web interface** — clean `index.html` served from Flask

---

## 📂 Project Structure

```
twin-ai/
├── app.py            # Flask API — chat, memory, context retrieval
├── index.html        # Frontend chat UI
├── embedcache.py     # Embedding cache utilities
├── txt_to_json.py    # Convert raw text notes to memory JSON format
├── data/
│   ├── personality.json   # Your personality Q&A
│   └── memories.json      # Your stored memories/experiences
└── .env              # API keys (GROQ_API_KEY)
```

---

## ⚙️ Setup

```bash
pip install flask flask-cors python-dotenv groq sentence-transformers scikit-learn numpy
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Create `data/personality.json`:
```json
[
  {"question": "What do you do?", "answer": "I'm a freelance web developer and AI builder."},
  {"question": "What's your style?", "answer": "Casual, direct, and always helpful."}
]
```

---

## ▶️ Running

```bash
python app.py
```

Open `http://localhost:5000` — you're talking to yourself.

---

## 💬 Special Commands

| Command | Effect |
|---------|--------|
| `remember: [text]` | Adds a new memory to the AI's brain |
| `save: [text]` | Same as remember |
| `add to memory: [text]` | Same as remember |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
