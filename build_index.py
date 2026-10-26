
import os
import pickle
import pymupdf

from sentence_transformers import SentenceTransformer
from text_splitter import split_text

# 1. Set the PDF path
pdf_path = "documents/Machine_Learning_Guide.pdf"

# 2. Extract text from the PDF
document = pymupdf.open(pdf_path)
full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# 3. Split the text into chunks
chunks = split_text(full_text)

if not chunks:
    raise ValueError("No text found in the PDF.")

print("Total chunks:", len(chunks))

# 4. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Generate embeddings
embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

# 6. Create a folder to save the index
os.makedirs("data", exist_ok=True)

# 7. Save chunks and embeddings together
with open("data/document_index.pkl", "wb") as file:
    pickle.dump(
        {
            "chunks": chunks,
            "embeddings": embeddings
        },
        file
    )

print("Index saved successfully!")
print("Saved to: data/document_index.pkl")
