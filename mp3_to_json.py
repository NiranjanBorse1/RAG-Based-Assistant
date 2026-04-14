import os
import whisper
import json

audios=sorted(os.listdir("audios"))
model=whisper.load_model("large-v2")

for audio in audios[:5]:
    if "_" in audio:
        number=audio.split("_")[0]
        title=audio.split("_")[1]
        result=model.transcribe(audio= f"audios/{audio}",language="hi",task="translate",word_timestamps=False)
        #result=model.transcribe(f"audios/{audio}",language="hi",task="translate",word_timestamps=False)

    chunks=[]
    for segment in result["segments"]:
        chunks.append({"number": number, "title": title, "Start": segment["start"], "End": segment["end"], "Text": segment["text"]})

    chunks_with_metadata ={"chunks": chunks,"text": result["text"]}

    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f)
