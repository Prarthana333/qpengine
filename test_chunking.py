from data_loader import load_pdf
from chunker import chunk_text

pdf_path = "D:\Give_Goa_Proj\qp_env\data\Brief Notes on CVP Analysis and Short Tern Decision Making.pdf"  # put any test PDF here

text = load_pdf(pdf_path)
chunks = chunk_text(text)

print(f"Total characters: {len(text)}")
print(f"Total chunks: {len(chunks)}")
print("\nFirst chunk:\n")
print(chunks[0][:1000])