import whisper
import json

model=whisper.load_model("large-v2")

result=model.transcribe("audios/sample1.mp3",language="hi",task="translate",word_timestamps=False)

chunks=[]
for segment in result["segments"]:
    chunks.append({"Start": segment["start"], "End": segment["end"], "Text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)
