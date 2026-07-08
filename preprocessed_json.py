import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    data = r.json()

    if "embeddings" in data:
        return data["embeddings"]
    if "embedding" in data:
        return [data["embedding"]]

    if len(text_list) > 1:
        embeddings = []
        for text in text_list:
            embeddings.extend(create_embedding([text]))
        return embeddings

    # Some inputs can fail server-side (for example NaN in response encoding).
    # Use a safe placeholder text to keep pipeline running.
    fallback = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": ["placeholder"]
    }).json()

    if "embeddings" in fallback:
        return fallback["embeddings"]
    if "embedding" in fallback:
        return [fallback["embedding"]]

    raise ValueError(f"Unexpected embedding response: {data}")


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0
batch_size = 64

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    chunk_texts = [c.get("text", c.get("Text", "")) for c in content["chunks"]]
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        embeddings.extend(create_embedding(chunk_texts[i:i + batch_size]))
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
# print(my_dicts)
df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')