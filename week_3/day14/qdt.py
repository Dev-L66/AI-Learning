# Qdrant is non-relatinal
# Qdrant sores data in collections
# Collections store vector that is array
# in collection ike rows there are point
# point have id, vector, payload
# payload has the data that is sored as vector

# Retrieval augmented Generative

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


client = QdrantClient(
    url = QDRANT_URL,
    api_key  = QDRANT_API_KEY
)

print("Connected to QDRANT CLIENT")



COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384


if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collecton: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    ),
)

print(f"Create collection: {COLLECTION_NAME}")
print(f"Vector Size: {EMBEDDING_SIZE}")
print(f"Distance: COSINE")


with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f 
        if line.strip()
    ]
print(f"Loaded {len(documents)}")


          
print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embeding model ready!")

embeddings = model.encode(documents)

print(f"Generate {len(embeddings)} embeddings")

print(F"Embedding size: {len(embeddings[0])}")


points =[]

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id = i +1,
        vector= embedding.tolist(),
        payload={
            "text":documents[i]
        }
    )

    points.append(point)



client.upsert(
    collection_name = COLLECTION_NAME,
    points = points
)


print(f"Upload {len(points)} documents to Qdrant!")



def search(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points


    return results



query = "How man vacation days do i get?" 

results = search(query, top_k=3)

print("Search results:")


for result in results:
    print(F"Score:{result.score:.3f}")
    print(result.payload["text"])


qroq_client=Groq(
    api_key = GROQ_API_KEY
)

# models = qroq_client.models.list()

# for model in models.data:
#     print(model.id)


def ask_llm(question, context):
    prompt = f"""
Answer the question using only the information provided below

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
I dont know based on the provided information"""


    response = qroq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response.choices[0].message.content



question = "How many vacation  days do I get?"


results = search(question, top_k=3)


context = "\n".join(
    result.payload["text"]
    for result in results
)

answer = ask_llm(question, context)


print("Final answer:  ")

print(answer)
