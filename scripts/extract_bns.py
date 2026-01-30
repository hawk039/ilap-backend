import fitz  # PyMuPDF
from pathlib import Path

# Define paths relative to this script
BASE_DIR = Path(__file__).parent.parent
PDF_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns.pdf"
OUT_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_raw.txt"

def extract_text():
    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        return

    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)

    all_text = []
    for i, page in enumerate(doc):
        text = page.get_text()
        all_text.append(text)
        if i % 10 == 0:
            print(f"Processed page {i+1}...")

    print(f"Writing to: {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    print("Extraction complete.")

if __name__ == "__main__":
    extract_text()
