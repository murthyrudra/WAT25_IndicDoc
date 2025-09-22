# ======================================================
# Gemma-IT Doc-level Fine-tuning (Trainer version)
# ======================================================

import os
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import sacrebleu
from loguru import logger

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("./outputs/gemma3-pralekha")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SEQ_LEN = 2048
MAX_TRAIN_SAMPLES = 100  # first 10k docs
MAX_EVAL_SAMPLES = 10    # first 1k docs
LANGUAGE_PAIRS = ["eng_hin", "hin_eng"]
TRAIN_DIR = Path("data/train")
EVAL_DIR = Path("data/dev")

# ------------------------------
# Load local doc-level dataset
# ------------------------------
def load_local_dataset(data_dir, max_samples=None):
    dataset = []
    loaded = 0
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        reversed_pairs = ["hin_eng"]
        if pair in reversed_pairs:
            rev_pair = f"{tgt}_{src}"
            src_file = Path(data_dir) / rev_pair / f"doc.{tgt}.jsonl"
            tgt_file = Path(data_dir) / rev_pair / f"doc.{src}.jsonl"
        else:
            src_file = Path(data_dir) / pair / f"doc.{src}.jsonl"
            tgt_file = Path(data_dir) / pair / f"doc.{tgt}.jsonl"

        with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
            for s_line, t_line in zip(f_src, f_tgt):
                try:
                    src_list = json.loads(s_line)
                    tgt_list = json.loads(t_line)
                    if not src_list or not tgt_list:
                        continue
                    src_text = src_list[0]
                    tgt_text = tgt_list[0]
                    dataset.append({
                        "input_text": f"Translate this {src} document to {tgt}:\n{src_text}\n",
                        "target_text": tgt_text
                    })
                    loaded += 1
                    if max_samples and loaded >= max_samples:
                        break
                except Exception as e:
                    logger.warning(f"Skipping doc due to error in {pair}: {e}")
                    continue
        logger.info(f"Loaded {loaded} samples from {pair}")
    return Dataset.from_list(dataset)

# ------------------------------
# Tokenize dataset
# ------------------------------
def tokenize_dataset(dataset, tokenizer):
    def preprocess(batch):
        enc = tokenizer(
            batch["input_text"],
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target_text"], max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
        enc["labels"] = labels["input_ids"]
        return enc
    return dataset.map(preprocess, batched=True)

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model_path, tokenizer, lang_pairs, eval_dir=EVAL_DIR, max_samples=MAX_EVAL_SAMPLES):
    print("[INFO] Starting evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    results = {}
    output_eval_file = Path(model_path) / "eval_predictions.jsonl"
    all_outputs = []

    for pair in lang_pairs:
        src, tgt = pair.split("_")
        print(f"[EVAL] {src} → {tgt}")
        src_file = Path(eval_dir) / pair / f"doc.{src}.jsonl"
        tgt_file = Path(eval_dir) / pair / f"doc.{tgt}.jsonl"

        preds, refs = [], []
        with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
            for idx, (s_line, t_line) in enumerate(zip(f_src, f_tgt)):
                if idx >= max_samples:
                    break
                try:
                    src_text = json.loads(s_line)[0]
                    ref_text = json.loads(t_line)[0]
                    inputs = tokenizer(f"Translate this {src} document to {tgt}:\n{src_text}\n", return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN, do_sample=False)
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    preds.append(decoded)
                    refs.append(ref_text.strip())
                    all_outputs.append({"source": src_text, "prediction": decoded, "reference": ref_text.strip()})

                    # --- Added progress logging ---
                    print(f"[EVAL PROGRESS] {idx+1}/{max_samples} processed")

                except Exception as e:
                    logger.warning(f"Skipping sample during eval: {e}")
                    continue

        if preds and refs:
            chrf = sacrebleu.corpus_chrf(preds, [refs])
            results[pair] = {"chrF2": chrf.score}
            print(f"  chrF2={chrf.score:.2f}")

    with open(output_eval_file, "w", encoding="utf-8") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved evaluation predictions → {output_eval_file}")

    results_file = Path(model_path) / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation metrics → {results_file}")


# ------------------------------
# Main training
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Auto dtype
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[INFO] Using dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float32
        print("[WARN] CUDA not available. Training on CPU will be very slow.")

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Loading training dataset...")
    train_dataset = load_local_dataset(TRAIN_DIR, max_samples=MAX_TRAIN_SAMPLES)
    train_dataset = tokenize_dataset(train_dataset, tokenizer)

    # ------------------------------
    # Training args
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-pralekha",
        dataloader_pin_memory=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, max_length=MAX_SEQ_LEN, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Resume checkpoint if exists
    last_checkpoint = None
    checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
    if checkpoints:
        last_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[-1])))

    print(f"[INFO] Starting training (resume: {last_checkpoint})")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("[INFO] Saving model + tokenizer + adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("[INFO] Running evaluation...")
    evaluate_model(OUTPUT_DIR, tokenizer, LANGUAGE_PAIRS)

if __name__ == "__main__":
    main()
