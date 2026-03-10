from data_loader import load_pdf
from chunker import chunk_text
from vector_store import VectorStore

pdf_path = "data/Chapter3DataPreprocessingforMachinelearning.pdf"

text = load_pdf(pdf_path)
chunks = chunk_text(text)

store = VectorStore()
store.build(chunks)

query = "train test validation split"
results = store.search(query)

print("Query:", query)
print("\nRetrieved chunks:\n")
for i, chunk in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(chunk[:500])
    print()