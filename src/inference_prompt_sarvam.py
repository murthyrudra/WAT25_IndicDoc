#!/usr/bin/env python
# inference_vllm.py
import argparse
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer

model_name = "sarvamai/sarvam-translate"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", required=True,
                   help="HF repo path or local dir; e.g. meta-llama/Llama-3.1-8B")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--tgt_lang", type=str)
    return p.parse_args()

def load_prompts(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        obj = json.loads(line) if line[0] in "{[" else {"prompt": line}
        if isinstance(obj, list): yield obj[0]
        else: yield obj["prompt"]

def make_prompts(sentences, tgt):
    if tgt == "ben":
        tgt = "Bengali"
    elif tgt == "guj":
        tgt = "Gujarati"
    elif tgt == "hin":
        tgt = "Hindi"
    elif tgt == "kan":
        tgt = "Kannada"
    elif tgt == "mal":
        tgt = "Malayalam"
    elif tgt == "mar":
        tgt = "Marathi"
    elif tgt == "ori":
        tgt = "Odiya"
    elif tgt == "pan":
        tgt = "Punjabi"
    elif tgt == "tam":
        tgt = "Tamil"
    elif tgt == "Tel":
        tgt = "Telugu"
    elif tgt == "urd":
        tgt = "Urdu"

    chat_messages = []
    for each_sentence in sentences:
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt}."},
            {"role": "user", "content": each_sentence}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        chat_messages.append(text)
    return chat_messages

# Define a generator function that uses vLLM for batched inference
def generate(prompts, llm, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return [
        {"generated_text": prompt + output.outputs[0].text}
        for prompt, output in zip(prompts, outputs)
    ]

def main():
    args = parse_args()
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize vLLM generator
    # Import vLLM here to control error handling
    from vllm import LLM, SamplingParams
    
    # Initialize vLLM engine with safe settings for problematic environments
    print(f"Initializing vLLM for model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,  # Use single GPU tensor parallelism for stability
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # Be conservative with memory
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.sampling else 0.0,
        top_p=args.top_p if args.sampling else 1.0,
        stop=["\n\n"],
    )
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size}) on num documents {len(prompts)}...")
    
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_prompts = make_prompts(batch_prompts, args.tgt_lang)
            batch_outputs = generate(batch_prompts, llm, sampling_params=sampling_params)
            
            for output, prompt in zip(batch_outputs, batch_prompts):
                generation = output["generated_text"].replace(prompt, "").strip()
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

            
    print("vLLM inference completed successfully.")

if __name__ == "__main__":
    main()