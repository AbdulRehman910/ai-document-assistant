
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# Test Mode
sample_text = "Machine learning is a branch of artificial intelligence. " * 30

chunks = split_text(sample_text)

print("Total chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0])
