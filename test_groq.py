
import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key
load_dotenv()

# Connect to Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Ask the AI a question
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain artificial intelligence in simple words."
        }
    ]
)

# Display the answer
print(response.choices[0].message.content)
