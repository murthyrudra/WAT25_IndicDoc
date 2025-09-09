import json
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ===== CONFIG =====
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "outputs/finetuned_any2any"
LANG_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]
DATA_DIR = "data/train"

# ===== LOAD DATA =====
def load_parallel_data(src_file, tgt_file, src_lang, tgt_lang):
    src_lines, tgt_lines = [], []
    with open(src_file, "r", encoding="utf-8", errors="replace") as fsrc, \
         open(tgt_file, "r", encoding="utf-8", errors="replace") as ftgt:
        for i, (s, t) in enumerate(
            tqdm(zip(fsrc, ftgt), desc=f"Loading {src_lang}->{tgt_lang}", unit="lines"), 1
        ):
            try:
                src_text = json.loads(s)["text"]
                tgt_text = json.loads(t)["text"]
                src_text = src_text.encode("utf-8", "ignore").decode("utf-8", "ignore")
                tgt_text = tgt_text.encode("utf-8", "ignore").decode("utf-8", "ignore")
                src_lines.append(src_text)
                tgt_lines.append(tgt_text)
            except Exception as e:
                print(f"❌ Skipping line {i} due to JSON error: {e}")
    assert len(src_lines) == len(tgt_lines), f"Mismatch in {src_lang}<->{tgt_lang}"
    return [{"src": s, "tgt": t, "src_lang": src_lang, "tgt_lang": tgt_lang}
            for s, t in zip(src_lines, tgt_lines)]

# ===== TOKENIZER =====
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(batch):
    prompt = f"Translate from {batch['src_lang']} to {batch['tgt_lang']}:\n\n{batch['src']}\n\nTranslation:"
    inputs = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    labels = tokenizer(batch["tgt"], truncation=True, max_length=512, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

# ===== PREPARE + TOKENIZE DATASETS =====
print("🔹 Preparing datasets...")
tokenized_datasets = []

for pair in LANG_PAIRS:
    _, tgt = pair.split("_")
    src_file = f"{DATA_DIR}/{pair}/doc.eng.cleaned.jsonl"
    tgt_file = f"{DATA_DIR}/{pair}/doc.{tgt}.cleaned.jsonl"

    print(f"\n=== Processing pair: {pair} ===")
    # eng -> tgt
    ds1 = Dataset.from_list(load_parallel_data(src_file, tgt_file, "eng", tgt))
    tok1 = ds1.map(preprocess, batched=False, num_proc=4, load_from_cache_file=False)
    tokenized_datasets.append(tok1)

    # tgt -> eng
    ds2 = Dataset.from_list(load_parallel_data(tgt_file, src_file, tgt, "eng"))
    tok2 = ds2.map(preprocess, batched=False, num_proc=4, load_from_cache_file=False)
    tokenized_datasets.append(tok2)

print("🔹 Concatenating all tokenized datasets...")
tokenized = concatenate_datasets(tokenized_datasets)
print(f"✅ Tokenization complete! Final size: {len(tokenized)}")

# ===== MODEL + LoRA =====
print("🔹 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
print("✅ Model loaded.")

print("🔹 Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("✅ LoRA applied.")

# ===== TRAINING =====
print("🔹 Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # small batch for stability
    gradient_accumulation_steps=8,
    num_train_epochs=1,             # test run
    learning_rate=2e-4,
    fp16=False,                     # test run
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()
print("✅ Training finished!")

print("🔹 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")
