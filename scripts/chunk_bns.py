import re, json

clean_path = "knowledge_base/BNS/v2024/bns_clean.txt"
out_path = "knowledge_base/BNS/v2024/bns_chunks.json"

text = open(clean_path, "r", encoding="utf-8").read()

# Split points: lines like "346. Whoever..."
pat = re.compile(r"(?m)^\s*(\d{1,4})\.\s+")

matches = list(pat.finditer(text))
print("Numbered provisions found:", len(matches))

chunks = []
for i, m in enumerate(matches):
    sec = m.group(1)
    start = m.start()
    end = matches[i+1].start() if i+1 < len(matches) else len(text)
    body = text[start:end].strip()

    # skip garbage / too-small chunks
    if len(body) < 200:
        continue

    # Remove Gazette repeated noise lines inside chunk (light cleanup)
    body = re.sub(r"(?m)^THE GAZETTE OF INDIA EXTRAORDINARY.*$", "", body).strip()
    body = re.sub(r"(?m)^_+$", "", body).strip()

    chunks.append({
        "act": "BNS",
        "section": sec,
        "effective_from": "2024-07-01",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "text": body
    })

print("Chunks created:", len(chunks))

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Wrote:", out_path)
