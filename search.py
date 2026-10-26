
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
from text_splitter import split_text

# 1. Read the PDF
pdf_path = "documents/Machine_Learning_Guide.pdf"
document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# 2. Split text into chunks
chunks = split_text(full_text)

if not chunks:
    raise ValueError("No text found in the PDF.")

print("PDF loaded successfully!")
print("Total chunks:", len(chunks))

# 3. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

# 4. Ask a question
question = input("\nAsk a question about your PDF: ")

# 5. Convert the question into an embedding
question_embedding = model.encode(
    [question],
    normalize_embeddings=True
)[0]

# 6. Calculate similarity between the question and chunks
scores = np.dot(chunk_embeddings, question_embedding)

# 7. Find the three most relevant chunks
top_indices = np.argsort(scores)[::-1][:3]

print("\nMost relevant PDF information:\n")

for index in top_indices:
    print(f"Similarity score: {scores[index]:.3f}")
    print(chunks[index])
    print("-" * 50)
