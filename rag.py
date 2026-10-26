
import os
import pickle
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Load the Groq API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Load the saved PDF chunks and embeddings
with open("data/document_index.pkl", "rb") as file:
    index_data = pickle.load(file)

chunks = index_data["chunks"]
chunk_embeddings = index_data["embeddings"]

print("Saved PDF index loaded successfully!")
print("Total chunks:", len(chunks))

# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Keep asking questions until the user types 'exit'
while True:
    question = input("\nAsk a question about your PDF (or type 'exit'): ")

    if question.strip().lower() == "exit":
        print("Goodbye!")
        break

    if not question.strip():
        print("Please enter a question.")
        continue

    # 5. Convert the question into an embedding
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # 6. Find the three most relevant chunks
    scores = np.dot(chunk_embeddings, question_embedding)
    top_indices = np.argsort(scores)[::-1][:3]

    context = "\n\n".join(
        chunks[index] for index in top_indices
    )

    # 7. Generate the answer using Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI document assistant. "
                    "Answer using only the provided document context. "
                    "If the answer is not present in the context, "
                    "say you could not find the answer in the PDF."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0.2
    )

    print("\nAI Document Assistant Answer:\n")
    print(response.choices[0].message.content)
    print("\n" + "-" * 50)
