#!/usr/bin/env python
# inference_vllm.py
import argparse
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument(
        "--model",
        required=True,
        help="HF repo path or local dir; e.g. meta-llama/Llama-3.1-8B",
    )
    p.add_argument("--max_new_tokens", type=int, default=8096)
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def load_prompts(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line) if line[0] in "{[" else {"prompt": line}
        if isinstance(obj, list):
            yield obj[0]
        else:
            yield obj["prompt"]


def build_generator(args):
    if args.model == "microsoft/phi-4":
        url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1"
    elif args.model == "openai/gpt-oss-20b":
        url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-20b/v1"
    elif args.model == "openai/gpt-oss-120b":
        url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b/v1"

    config = dotenv_values(".env")

    if "RITS_API_KEY" in config:
        api_key = config["RITS_API_KEY"]

    client = OpenAI(
        api_key=api_key,
        base_url=url,
        default_headers={"RITS_API_KEY": api_key},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = args.model

    # Define a generator function that uses vLLM for batched inference
    def generate(prompts, args):
        chat_content = []

        for each_prompt in prompts:
            chat_content.append(
                [
                    {
                        "role": "user",
                        "content": each_prompt,
                    }
                ]
            )

        completions = []
        for each_message in chat_content:
            completion = client.responses.create(
                model=model,
                input=each_message,
                max_output_tokens=args.max_new_tokens,
                temperature=0,
                reasoning={"effort": "low"},
            )
            completions.append(completion.output_text)

        result = []
        for each_result in completions:
            result.append(each_result.strip())

        return [
            {"generated_text": prompt + output}
            for prompt, output in zip(prompts, result)
        ]

    return generate


def main():
    args = parse_args()
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize vLLM generator
    gen_fn = build_generator(args)
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size})...")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts, args)

            for output, prompt in zip(batch_outputs, batch_prompts):
                generation = output["generated_text"].replace(prompt, "").strip()
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()
    print("vLLM inference completed successfully.")


if __name__ == "__main__":
    main()
