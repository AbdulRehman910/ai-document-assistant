
import pymupdf
from sentence_transformers import SentenceTransformer
from text_splitter import split_text

# 1. Open your PDF
pdf_path = "documents/Machine_Learning_Guide.pdf"
document = pymupdf.open(pdf_path)

# 2. Extract text from all pages
full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

print("PDF text extracted successfully!")
print("Total characters:", len(full_text))

# 3. Split the PDF text into chunks
chunks = split_text(full_text)

print("Total chunks:", len(chunks))

# 4. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Generate an embedding for each chunk
embeddings = model.encode(chunks)

print("Embedding generation complete!")
print("Embedding shape:", embeddings.shape)

# 6. Display the first chunk and its embedding
if chunks:
    print("\nFirst chunk:")
    print(chunks[0])

    print("\nFirst embedding (first 10 numbers):")
    print(embeddings[0][:10])
