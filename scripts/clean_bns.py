import re
from pathlib import Path

# Define paths relative to this script
BASE_DIR = Path(__file__).parent.parent
INP_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_raw.txt"
OUT_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_clean.txt"

def clean_text():
    if not INP_PATH.exists():
        print(f"Error: Input file not found at {INP_PATH}")
        return

    print(f"Reading from: {INP_PATH}")
    with open(INP_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Fix hyphenated line breaks (e.g., "con-\ntinue" -> "continue")
    text = re.sub(r"-\n", "", text)

    # 2. Collapse multiple newlines (more than 3 becomes 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Remove page numbers (simple heuristic: standalone number on a line)
    # Matches newline, digits, newline -> replaces with single newline
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    print(f"Writing to: {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("Cleaning complete.")

if __name__ == "__main__":
    clean_text()
