# -*- coding: utf-8 -*-
import os
import json
import random
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from IndicTransToolkit.processor import IndicProcessor
from datasets import load_dataset
from indicnlp.tokenize.sentence_tokenize import sentence_split

# ------------------------------
# Config
# ------------------------------
INDIC_EN_CKPT_DIR = "ai4bharat/indictrans2-indic-en-1B"
BATCH_SIZE = 16
N_DOCS = 20000
OUTPUT_DIR = "./synthetic_en_hi_docs"
COMBINED_SAVE_DIR = "./combined_pralekha_synthetic_ds"
TRAIN_DIR = "data/train/eng_hin"  # directory with original doc.eng.jsonl & doc.hin.jsonl

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMBINED_SAVE_DIR, exist_ok=True)

# ------------------------------
# Initialize model + tokenizer
# ------------------------------
def initialize_model_and_tokenizer(ckpt_dir, quantization=None):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if qconfig is None:
        model = model.to(device)
        if device == "cuda":
            model.half()
    model.eval()
    return tokenizer, model

tokenizer, model = initialize_model_and_tokenizer(INDIC_EN_CKPT_DIR)
ip = IndicProcessor(inference=True)

# ------------------------------
# Stream Hindi monolingual documents
# ------------------------------
print("[INFO] Streaming IndicCorpV2 Hindi corpus...")
hin_stream = load_dataset("ai4bharat/IndicCorpV2", split="hin_Deva", streaming=True)

def sample_docs_safe(stream_iterable, n_docs=N_DOCS):
    docs = []
    for item in stream_iterable:
        text = item.get("text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = text.strip()
        if text:
            docs.append(text)
        if len(docs) >= n_docs:
            break
    return docs

hin_docs = sample_docs_safe(hin_stream, n_docs=N_DOCS)
print(f"[INFO] Collected {len(hin_docs)} Hindi docs for backtranslation.")

# ------------------------------
# Translation helper
# ------------------------------
def batch_translate_no_cache(sentences, src_lang, tgt_lang, model, tokenizer, ip):
    inputs = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    model_inputs = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    translated_tokens = model.generate(**model_inputs, use_cache=False, max_length=256)
    outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return ip.postprocess_batch(outputs, lang=tgt_lang)

# ------------------------------
# Backtranslate Hindi → English
# ------------------------------
print("[INFO] Backtranslating Hindi → English (doc-level)...")
backtranslated_eng_docs = []

valid_doc_count = 0
for doc in hin_docs:
    hin_sents = sentence_split(doc, "hin")
    if not hin_sents:
        continue
    translations = []

    for i in range(0, len(hin_sents), BATCH_SIZE):
        batch = hin_sents[i:i+BATCH_SIZE]
        batch_translations = batch_translate_no_cache(
            batch, src_lang="hin_Deva", tgt_lang="eng_Latn",
            model=model, tokenizer=tokenizer, ip=ip
        )
        translations.extend(batch_translations)

    valid_doc_count += 1

    # Save sentence-aligned file
    aligned_path = os.path.join(OUTPUT_DIR, f"doc{valid_doc_count}_aligned.txt")
    with open(aligned_path, "w", encoding="utf-8") as f:
        for hin, eng in zip(hin_sents, translations):
            f.write(f"HIN: {hin}\nENG: {eng}\n\n")

    # Save merged English doc
    merged_path = os.path.join(OUTPUT_DIR, f"doc{valid_doc_count}.txt")
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(" ".join(translations))

    backtranslated_eng_docs.append(" ".join(translations))

    if valid_doc_count % 100 == 0:
        print(f"[INFO] Backtranslated {valid_doc_count} docs...")

print(f"[INFO] All {valid_doc_count} synthetic docs saved at {OUTPUT_DIR}")

# ------------------------------
# Read original English & Hindi JSONL
# ------------------------------
original_eng_jsonl = os.path.join(TRAIN_DIR, "doc.eng.jsonl")
original_hin_jsonl = os.path.join(TRAIN_DIR, "doc.hin.jsonl")

original_docs = []
with open(original_eng_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                arr = json.loads(line)
                if isinstance(arr, list) and len(arr) > 0:
                    original_docs.append(arr[0])
            except:
                continue

original_hin_docs = []
with open(original_hin_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                arr = json.loads(line)
                if isinstance(arr, list) and len(arr) > 0:
                    original_hin_docs.append(arr[0])
            except:
                continue

if len(original_docs) != len(original_hin_docs):
    raise ValueError("Original English and Hindi JSONL counts do not match!")

# ------------------------------
# Combine English & Hindi
# ------------------------------
all_eng_docs = original_docs + backtranslated_eng_docs
all_hin_docs = original_hin_docs + hin_docs

# Shuffle indices to match previous behavior
indices = list(range(len(all_eng_docs)))
random.seed(42)
random.shuffle(indices)

shuffled_eng_docs = [all_eng_docs[i] for i in indices]
shuffled_hin_docs = [all_hin_docs[i] for i in indices]

# ------------------------------
# Write combined English JSONL
# ------------------------------
combined_eng_jsonl = os.path.join(TRAIN_DIR, "doc.eng.both.jsonl")
with open(combined_eng_jsonl, "w", encoding="utf-8") as f:
    for idx, doc_text in enumerate(shuffled_eng_docs, 1):
        json_line = json.dumps([doc_text.strip()], ensure_ascii=False)
        f.write(json_line + "\n")
        if idx % 100 == 0:
            print(f"[INFO] Written {idx} English docs...")

print(f"[INFO] Combined English JSONL saved at {combined_eng_jsonl}")

# ------------------------------
# Write combined Hindi JSONL
# ------------------------------
combined_hin_jsonl = os.path.join(TRAIN_DIR, "doc.hin.both.jsonl")
with open(combined_hin_jsonl, "w", encoding="utf-8") as f:
    for idx, doc_text in enumerate(shuffled_hin_docs, 1):
        json_line = json.dumps([doc_text.strip()], ensure_ascii=False)
        f.write(json_line + "\n")
        if idx % 100 == 0:
            print(f"[INFO] Written {idx} Hindi docs...")

print(f"[INFO] Combined Hindi JSONL saved at {combined_hin_jsonl}")
