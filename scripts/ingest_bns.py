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

def read_and_chunk_file(file_path: Path):
    """
    Reads the file and splits it into chunks based on legal sections.
    Goal: One chunk ≈ one legal section.
    Regex used: Matches "Section 1." or just "1." at start of lines.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to identify section starts.
    # Matches:
    # (?:\n|^)          -> Start of line or string
    # (?:Section\s+)?   -> Optional "Section " word
    # (\d+[A-Za-z]*)    -> Section number (e.g., 1, 303, 45A) - Capturing group 1
    # \.                -> Literal dot
    pattern = r'(?:\n|^)((?:Section\s+)?\d+[A-Za-z]*)\.'
    
    # Split the text. 
    # re.split with capturing group returns [preamble, section_num_1, content_1, section_num_2, content_2, ...]
    parts = re.split(pattern, text)
    
    chunks = []
    
    # The first part is usually preamble/title/chapter info before Section 1
    if parts[0].strip():
        chunks.append(parts[0].strip())
    
    # Iterate through the rest of the parts in pairs (number, content)
    for i in range(1, len(parts), 2):
        section_identifier = parts[i] # e.g., "1" or "Section 303"
        section_content = parts[i+1] if i+1 < len(parts) else ""
        
        # Combine identifier and content
        # We add the dot back since it was consumed by the split but not captured in the group
        full_section = f"{section_identifier}.{section_content}".strip()
        
        if full_section:
            chunks.append(full_section)

    return chunks

def extract_section(text: str) -> str:
    """
    Attempts to extract a section number from the start of the text.
    """
    # Matches "Section 123" or "123" at the start
    match = re.match(r"^(?:Section\s+)?(\d+[A-Za-z]*)", text)
    if match:
        return match.group(1)
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
            "section": extract_section(chunk) 
        })

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully ingested {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest_data()
