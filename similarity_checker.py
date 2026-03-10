from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from db import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.70  # Adjust if needed

def is_similar(new_question):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT question_text FROM questions")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return False

    existing_questions = [row[0] for row in rows]

    new_embedding = model.encode([new_question])
    existing_embeddings = model.encode(existing_questions)

    similarities = cosine_similarity(new_embedding, existing_embeddings)

    max_similarity = np.max(similarities)

    print(f"Max similarity score: {max_similarity:.3f}")

    return max_similarity > SIMILARITY_THRESHOLD
