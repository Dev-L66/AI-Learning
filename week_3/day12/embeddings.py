import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer




def cosnie_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

model = SentenceTransformer("all-MiniLM-L6-v2") # 384

text = "Machine lerning is fun."


embedding = model.encode(text)

# print(embedding.shape)
# print(embedding)
# print(embedding[:10])


t1="There are 24 paid leaves"
t2="There are 24 vacation days"


v1 = model.encode(t1)
v2 = model.encode(t2)


print(cosnie_similarity(v1, v2))