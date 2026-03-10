from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import random

class QueryGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_queries(self, chunks, top_k=5):
        """
        Auto-generate semantic queries with controlled randomness.
        Queries remain syllabus-grounded but vary across runs.
        """

        if not chunks:
            return []

        # Embed chunks
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        # Safety check
        if embeddings.ndim != 2:
            return []

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # ---- CONTROLLED RANDOMNESS STARTS HERE ----

        # Shuffle indices so representative chunks vary each run
        indices = list(range(len(chunks)))
        random.shuffle(indices)

        step = max(1, len(indices) // (top_k * 2))

        queries = []
        seen = set()

        for i in indices[::step]:
            text = chunks[i].strip()

            # Ignore very short or noisy chunks
            if len(text) < 50:
                continue

            # Create short, meaningful query
            short_query = " ".join(text.split()[:6])

            if short_query not in seen:
                queries.append(short_query)
                seen.add(short_query)

            if len(queries) >= top_k:
                break

        return queries
