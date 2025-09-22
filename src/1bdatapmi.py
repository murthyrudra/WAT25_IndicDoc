# -*- coding: utf-8 -*-
import os
import json

# ------------------------------
# Config
# ------------------------------
DOC_SIZE = 10       # sentences per doc
MAX_PM_DOCS = 10000
TRAIN_DIR = "data/train/eng_hin"  # where doc.eng.both.jsonl & doc.hin.both.jsonl are
PM_EN_FILE = "pmindia/pmindia.en-hi.en"
PM_HI_FILE = "pmindia/pmindia.en-hi.hi"

# ------------------------------
# Read PMIndia sentence-level files
# ------------------------------
with open(PM_EN_FILE, "r", encoding="utf-8") as f:
    pm_en_sents = [line.strip() for line in f if line.strip()]

with open(PM_HI_FILE, "r", encoding="utf-8") as f:
    pm_hi_sents = [line.strip() for line in f if line.strip()]

if len(pm_en_sents) != len(pm_hi_sents):
    raise ValueError("PMIndia English and Hindi sentence counts do not match!")

# ------------------------------
# Convert into doc-level (10 sentences per doc)
# ------------------------------
pm_en_docs_all = [" ".join(pm_en_sents[i:i+DOC_SIZE]) for i in range(0, len(pm_en_sents), DOC_SIZE)]
pm_hi_docs_all = [" ".join(pm_hi_sents[i:i+DOC_SIZE]) for i in range(0, len(pm_hi_sents), DOC_SIZE)]

# Limit to MAX_PM_DOCS
pm_en_docs = pm_en_docs_all[:MAX_PM_DOCS]
pm_hi_docs = pm_hi_docs_all[:MAX_PM_DOCS]

print(f"[INFO] Using {len(pm_en_docs)} PMIndia doc-level samples for merging.")

# ------------------------------
# Read existing combined JSONL
# ------------------------------
def read_jsonl(file_path):
    docs = []
    if not os.path.exists(file_path):
        return docs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arr = json.loads(line)
                if isinstance(arr, list) and len(arr) > 0:
                    docs.append(arr[0])
            except:
                continue
    return docs

eng_docs = read_jsonl(os.path.join(TRAIN_DIR, "doc.eng.both.jsonl"))
hi_docs = read_jsonl(os.path.join(TRAIN_DIR, "doc.hin.both.jsonl"))

if len(eng_docs) != len(hi_docs):
    raise ValueError("Existing English and Hindi JSONL counts do not match!")

# ------------------------------
# Merge PMIndia docs
# ------------------------------
merged_eng_docs = eng_docs + pm_en_docs
merged_hi_docs = hi_docs + pm_hi_docs

print(f"[INFO] Total English docs after merge: {len(merged_eng_docs)}")
print(f"[INFO] Total Hindi docs after merge: {len(merged_hi_docs)}")

# ------------------------------
# Save merged JSONL
# ------------------------------
combined_eng_jsonl = os.path.join(TRAIN_DIR, "doc.eng.both.jsonl")
combined_hi_jsonl = os.path.join(TRAIN_DIR, "doc.hin.both.jsonl")

with open(combined_eng_jsonl, "w", encoding="utf-8") as f:
    for doc in merged_eng_docs:
        f.write(json.dumps([doc], ensure_ascii=False) + "\n")

with open(combined_hi_jsonl, "w", encoding="utf-8") as f:
    for doc in merged_hi_docs:
        f.write(json.dumps([doc], ensure_ascii=False) + "\n")

print(f"[INFO] Merged JSONL saved at {combined_eng_jsonl} and {combined_hi_jsonl}")
