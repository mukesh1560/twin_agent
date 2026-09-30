import os
import json
import logging
import time
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Load Environment ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
print("API KEY:", GROQ_API_KEY)

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PERSONALITY_JSON = DATA_DIR / "personality.json"
MEMORIES_JSON    = DATA_DIR / "memories.json"
EMB_FILE         = DATA_DIR / "embeddings.npz"
META_FILE        = DATA_DIR / "metadata.json"

LLM_MODEL = "llama-3.1-8b-instant"

# --- Initialize AI Models ---
print("⏳ Loading AI Brain (this may take a minute the first time)...")
try:
    # This model is ~80MB and runs locally on your CPU
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")

if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in .env file!")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
CORS(app)

# --- Logic: Memory Management ---

def load_source_data():
    """Combines personality and memory JSONs into one list."""
    combined = []
    
    # Personality
    if PERSONALITY_JSON.exists():
        with open(PERSONALITY_JSON, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for i, item in enumerate(data):
                    txt = f"Personality: {item.get('question', '')} {item.get('answer', '')}"
                    combined.append({"id": f"p_{i}", "text": txt, "category": "personality"})
            except: pass
                
    # Memories
    if MEMORIES_JSON.exists():
        with open(MEMORIES_JSON, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for i, item in enumerate(data):
                    txt = f"Memory: {item.get('question', '')} -> {item.get('answer', '')}"
                    combined.append({"id": f"m_{i}", "text": txt, "category": "memory"})
            except: pass
                
    return combined

def build_index():
    """Generates embeddings and saves them to disk."""
    items = load_source_data()
    if not items:
        return None, []

    texts = [it["text"] for it in items]
    vectors = embed_model.encode(texts).astype(np.float32)
    
    np.savez_compressed(EMB_FILE, arr=vectors)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f)
    
    return vectors, items

def save_new_experience(text):
    """Saves a new memory and re-indexes the AI brain."""
    new_entry = {
        "id": int(time.time()), 
        "question": "User shared experience", 
        "answer": str(text), 
        "category": "experience"
    }
    memories = []
    if MEMORIES_JSON.exists():
        with open(MEMORIES_JSON, "r", encoding="utf-8") as f:
            try:
                content = f.read().strip()
                if content: memories = json.loads(content)
            except: memories = []

    memories.append(new_entry)
    with open(MEMORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)
    
    build_index() # Update the brain instantly
    return True

def get_context(query, top_k=5):
    """Retrieves relevant memories for a query."""
    try:
        if not EMB_FILE.exists():
            vectors, meta = build_index()
            if vectors is None: return ""
        else:
            with np.load(EMB_FILE) as data: vectors = data["arr"]
            with open(META_FILE, "r", encoding="utf-8") as f: meta = json.load(f)

        if not meta: return ""

        query_vec = embed_model.encode([query])
        k = min(top_k, len(meta))
        nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vectors)
        distances, indices = nbrs.kneighbors(query_vec)
        
        return "\n".join([meta[i]["text"] for i in indices[0]])
    except:
        return ""

from groq import Groq


def call_groq_chat(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print("💣 GROQ ERROR:", str(e))
        return f"Error: {str(e)}"
        reply = call_groq_chat(system_prompt, user_input)
@app.route("/")
def index():
    """Serves your index.html from the current folder."""
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    user_input = data.get("user_input", "").strip()
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # 1. Check for Commands (Remember this)
    if user_input.lower().startswith(("remember", "save", "add to memory")):
        try:
            content = user_input.split(":", 1)[1].strip()
            save_new_experience(content)
            return jsonify({"reply": "✅ Memory locked in! I've added that to our shared history."})
        except:
            return jsonify({"error": "Format: 'Remember this: [your text]'"}), 400

    # 2. Normal Chat Logic
    context = get_context(user_input)
    system_prompt = f"""
    You are User's AI Twin. 
    Use the following memories to respond in his voice:
    {context}
    
    Be natural, concise, and casual.
    """

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return jsonify({"reply": completion.choices[0].message.content})
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        return jsonify({"error": "I lost my connection to Groq. Check your API key."}), 500

if __name__ == "__main__":
    # Pre-build index on start
    if load_source_data():
        build_index()
    
    print("\n🚀 AI Twin Server Running!")
    print("👉 Open your browser to: http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
