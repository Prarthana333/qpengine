import random
from data_loader import load_pdf
from chunker import chunk_text
from vector_store import VectorStore
from prompt_builder import build_prompt
from llm_engine import generate_question
from query_generator import QueryGenerator
from db import insert_question
from similarity_checker import is_similar


# ----------------------------------------
# 1. Load multiple PDFs
# ----------------------------------------
pdf_paths = [
    r"D:\Give_Goa_Proj\qp_env\data\Python_Guide - Session 1,2 and 3.pdf",
    r"D:\Give_Goa_Proj\qp_env\data\Chapter3DataPreprocessingforMachinelearning.pdf",
    r"D:\Give_Goa_Proj\qp_env\data\NB2022 (1) (2) (1).pdf",
    r"D:\Give_Goa_Proj\qp_env\data\Decision Tree Classifier.pdf",
    r"D:\Give_Goa_Proj\qp_env\data\AML_RM_A.pdf"
]

all_chunks = []

for pdf in pdf_paths:
    text = load_pdf(pdf)
    chunks = chunk_text(text)
    print(f"{pdf} → {len(chunks)} chunks")
    all_chunks.extend(chunks)

print(f"\nTotal chunks collected: {len(all_chunks)}")


# ----------------------------------------
# 2. Build vector store
# ----------------------------------------
store = VectorStore()
store.build(all_chunks)


# ----------------------------------------
# 3. Auto-generate queries
# ----------------------------------------
query_gen = QueryGenerator()
queries = query_gen.generate_queries(all_chunks, top_k=8)

random.shuffle(queries)
queries = queries[:5]

print("\nAuto-generated queries:")
for q in queries:
    print("-", q)


# ----------------------------------------
# 4. Variation controls
# ----------------------------------------
question_verbs = [
    "Explain",
    "Describe",
    "Discuss",
    "Why is it important to",
    "What is the role of"
]

focus_types = [
    "conceptual understanding",
    "practical application",
    "exam-oriented explanation",
    "real-world relevance"
]


# ----------------------------------------
# 5. Question generation settings
# ----------------------------------------
bloom_level = "Understand"
difficulty = "Difficult"
marks = 5

print("\nGenerated Questions:\n")


# ----------------------------------------
# 6. Generate questions with similarity check
# ----------------------------------------
for i, query in enumerate(queries, 1):

    context = store.search(query, top_k=10)

    verb = random.choice(question_verbs)
    focus = random.choice(focus_types)

    prompt = build_prompt(
        context_chunks=context,
        question_type="short answer",
        bloom_level=bloom_level,
        difficulty=difficulty,
        marks=marks,
        verb=verb,
        focus=focus
    )

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        question = generate_question(prompt)

        if not is_similar(question):
            print(f"Q{i}. {question}\n")

            # Store in database
            insert_question(
                question_text=question,
                bloom_level=bloom_level,
                difficulty=difficulty,
                marks=marks,
                source_query=query
            )

            break
        else:
            print("Similar question detected. Regenerating...\n")
            attempts += 1

    if attempts == max_attempts:
        print(f"Q{i}. Could not generate sufficiently unique question.\n")