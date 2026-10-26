
import pymupdf
from text_splitter import split_text

pdf_path = "documents/Machine_Learning_Guide.pdf"

document = pymupdf.open(pdf_path)

# Extract text from all PDF pages
full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

print("PDF text extracted successfully!")
print("Total characters:", len(full_text))

# Split the extracted text into chunks
chunks = split_text(full_text)

print("Total chunks:", len(chunks))
# Display the first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0])
