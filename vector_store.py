import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        if not chunks:
            raise ValueError("No chunks to build vector store.")

        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        if embeddings.ndim != 2:
            raise ValueError("Invalid embeddings.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        _, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
