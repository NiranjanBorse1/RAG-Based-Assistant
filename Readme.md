# RAG-Based Video Course Assistant

> An **AI-powered Retrieval-Augmented Generation (RAG)** assistant that lets you ask natural language questions about video course content and get pinpointed answers — complete with the exact video and timestamp to watch.

---

## Overview

This project builds a fully local, privacy-friendly RAG pipeline on top of recorded video lectures. It automatically:

1. **Extracts audio** from video files using FFmpeg
2. **Transcribes speech** (including Hindi to English translation) using OpenAI Whisper
3. **Generates semantic embeddings** for all transcript chunks using the `bge-m3` model via Ollama
4. **Answers natural language questions** by retrieving the most relevant transcript segments and feeding them to a local LLM (`llama3.2`)

Everything runs **100% locally** — no cloud APIs, no data sent externally.

---

## Architecture

```
videos/
  └─ [video files]
       │
       ▼  video_to_mp3.py  (FFmpeg)
audios/
  └─ [MP3 files]
       │
       ▼  mp3_to_json.py  (OpenAI Whisper large-v2)
jsons/
  └─ [transcript chunks with timestamps]
       │
       ▼  preprocessed_json.py  (bge-m3 embeddings via Ollama)
embeddings.joblib
  └─ [DataFrame: chunk text + metadata + embedding vectors]
       │
       ▼  process_incoming.py  (cosine similarity + llama3.2 via Ollama)
  Natural language answer with video title + timestamp
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Audio Extraction | FFmpeg |
| Speech-to-Text / Translation | OpenAI Whisper (`large-v2`) |
| Embedding Model | `bge-m3` (via Ollama) |
| Vector Search | Cosine Similarity (`scikit-learn`) |
| LLM Inference | `llama3.2` (via Ollama) |
| Data Storage | Pandas DataFrame + Joblib |
| Language | Python 3.10+ |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) available on your system PATH
- [Ollama](https://ollama.com/) installed and running locally

### 1. Clone the Repository

```bash
git clone https://github.com/NiranjanBorse1/RAG-Based-Assistant.git
cd RAG-Based-Assistant
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install pandas numpy scikit-learn joblib requests openai-whisper
```

### 4. Pull Ollama Models

```bash
ollama pull bge-m3     # Embedding model (~1.2 GB)
ollama pull llama3.2   # LLM for response generation (~2.0 GB)
```

---

## Running the Pipeline

### Step 1 — Add Your Videos

Place all your video files inside the `videos/` folder.  
Expected filename format (YouTube-style downloads):
```
Video #10 | Video Title [youtubeID].mp4
```

### Step 2 — Extract Audio

```bash
python video_to_mp3.py
```
Converts all video files to MP3 using FFmpeg. Output saved to `audios/`.

### Step 3 — Transcribe Audio to JSON

```bash
python mp3_to_json.py
```
Uses OpenAI Whisper (`large-v2`) to transcribe audio and translate Hindi speech to English. Saves timestamped transcript chunks to `jsons/`.

> Note: This step is slow on CPU. A GPU is strongly recommended for the `large-v2` model. You can switch to `medium` for faster inference with a small accuracy trade-off.

### Step 4 — Generate Embeddings

```bash
python preprocessed_json.py
```
Sends all transcript chunks to the `bge-m3` model via Ollama (in batches of 64) and stores the resulting embedding vectors in `embeddings.joblib`.

> Make sure Ollama is running before this step (`ollama serve`).

### Step 5 — Ask Questions

```bash
python process_incoming.py
```

You will see:
```
Ask a Question: _
```

Type any question about the course content. The assistant will:
- Embed your query using `bge-m3`
- Find the top-3 most semantically similar transcript chunks
- Feed them to `llama3.2` with a structured prompt
- Print the answer with the exact video title and timestamp

The response is also saved to `response.txt`.

---

## Example

**Query:**
```
What are semantic tags in HTML?
```

**Response:**
```
Semantic tags are covered in Video #11 — "Semantic Tags in HTML".

The concept is introduced at the following timestamps:
  - 40.24s – 43.44s  ->  "So guys, I would like to tell you what are Semantic Tags?"
  - 56.56s – 59.76s  ->  "To explain a little, I will tell you what are Semantic Tags?"
  - 256.8s – 258.0s  ->  "What are semantic tags?"

Go to Video #11 starting from 40 seconds to learn about semantic HTML tags.
```

---

## Project Structure

```
RAG-Based-Assistant/
│
├── videos/               # Place your input video files here
├── audios/               # Auto-generated MP3 files
├── jsons/                # Auto-generated transcript JSON files
│
├── video_to_mp3.py       # Step 1: Video -> Audio (FFmpeg)
├── mp3_to_json.py        # Step 2: Audio -> Transcript JSON (Whisper)
├── preprocessed_json.py  # Step 3: JSON -> Embeddings (bge-m3 + Ollama)
├── process_incoming.py   # Step 4: Query -> RAG Answer (llama3.2 + Ollama)
│
├── embeddings.joblib     # Saved embedding vectors (auto-generated, not in repo)
├── prompt.txt            # Last generated prompt (auto-generated)
├── response.txt          # Last LLM response (auto-generated)
│
└── Readme.md
```

---

## Key Features

- **Fully Local & Private** — No data leaves your machine. Runs entirely on Ollama.
- **Multilingual Support** — Handles Hindi audio and translates to English via Whisper.
- **Timestamp-level Precision** — Returns the exact second in the exact video.
- **Batch Embedding** — Efficiently processes large transcript datasets in batches.
- **Robust Fallback** — Graceful error handling in the embedding pipeline.
- **Modular Pipeline** — Each stage is an independent script; easy to debug or swap components.

---

## Everyday Usage (After Initial Setup)

Once the embeddings are built, you only need to run the query script:

```bash
.venv\Scripts\activate
python process_incoming.py
```

Make sure Ollama is running in the background.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

**Niranjan Borse**  
[GitHub](https://github.com/NiranjanBorse1)
