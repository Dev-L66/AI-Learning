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
from qdrant.client.models import Distance,VectorParams,PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

