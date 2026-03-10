import random
from data_loader import load_pdf
from chunker import chunk_text
from vector_store import VectorStore
from prompt_builder import build_prompt

# ----------------------------
# 1. Load and preprocess PDF
# ----------------------------
pdf_path = r"D:\Give_Goa_Proj\qp_env\data\Chapter3DataPreprocessingforMachinelearning.pdf"

text = load_pdf(pdf_path)
chunks = chunk_text(text)

# ----------------------------
# 2. Build vector store
# ----------------------------
store = VectorStore()
store.build(chunks)

# ----------------------------
# 3. Retrieve relevant content
# ----------------------------
query = "train test validation split"
context = store.search(query)

# ----------------------------
# 4. Add controlled variation
# ----------------------------
variation_hint = random.choice([
    "focus on conceptual understanding",
    "focus on practical application",
    "focus on exam-oriented explanation",
    "focus on real-world relevance in machine learning"
])

# ----------------------------
# 5. Build prompt with variation
# ----------------------------
prompt = build_prompt(
    context_chunks=context,
    question_type="short answer",
    bloom_level="Understand",
    difficulty="Easy",
    marks=2
)

# Inject variation safely
prompt += f"\n\nAdditional guidance: {variation_hint}."

# ----------------------------
# 6. Print final prompt
# ----------------------------
print(prompt)