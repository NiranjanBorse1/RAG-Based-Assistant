# How to use this RAG AI Teaching assistant on your own data

## Install required dependencies

Before running the pipeline, make sure these are installed:

- Python 3.10+.
- `ffmpeg` available on your system PATH.
- A virtual environment activated for this project.
- Python packages:
	- `pandas`
	- `numpy`
	- `scikit-learn`
	- `joblib`
	- `requests`
	- `openai-whisper`
- Ollama running locally at `http://localhost:11434`.
- Ollama models:
	- `bge-m3` for embeddings.
	- `llama3.2` or `deepseek-r1` for response generation.

Example install commands:

```bash
pip install pandas numpy scikit-learn joblib requests openai-whisper
```

```bash
ollama pull bge-m3
ollama pull llama3.2
```

## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM


