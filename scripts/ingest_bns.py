import os
import chromadb
from chromadb.utils import embedding_functions
import uuid
import re
from pathlib import Path

# Configuration
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
BNS_FILE_PATH = Path(__file__).parent.parent / "knowledge_base" / "BNS" / "v2024" / "bns.txt"
COLLECTION_NAME = "legal_knowledge_base"

def read_and_chunk_file(file_path: Path, chunk_size: int = 800):
    """
    Reads the file and splits it into chunks.
    Simple rule:
    - Split by blank lines
    - Merge small chunks
    - Max ~800 chars
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by blank lines (double newlines)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph keeps us under the limit, add it
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # Otherwise, save the current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_section(text: str) -> str:
    """
    Attempts to extract a section number from the start of the text.
    Looks for patterns like "1.", "Section 1", "123.", etc.
    """
    match = re.match(r"^(Section\s+)?(\d+[A-Za-z]*)\.?", text)
    if match:
        return match.group(2)
    return "Unknown"

def ingest_data():
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # Create or get collection
    # Using default embedding function (all-MiniLM-L6-v2)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Delete collection if it exists to start fresh with new metadata schema
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' to update schema.")
    except Exception as e:
        # Ignore if collection doesn't exist
        print(f"Collection '{COLLECTION_NAME}' not found or could not be deleted. Creating new one.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    print(f"Reading from {BNS_FILE_PATH}...")
    chunks = read_and_chunk_file(BNS_FILE_PATH)
    
    if not chunks:
        print("No data to ingest.")
        return

    print(f"Found {len(chunks)} chunks. Ingesting...")

    ids = []
    metadatas = []
    documents = []

    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        documents.append(chunk)
        # Metadata schema as requested
        metadatas.append({
            "law": "BNS",
            "version": "2024",
            "source": "India Code",
            "type": "criminal_law",
            "section": extract_section(chunk) # Keeping this as it's useful
        })

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully ingested {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest_data()
