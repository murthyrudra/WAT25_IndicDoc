#!/usr/bin/env python
"""
Quick-start: translate a slice of the Pralekha dev-set with vLLM + Llama-3.1-8B.

$ pip install --upgrade "datasets>=2.19.0" "vllm>=0.4.2" accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121
# ⬆ adjust CUDA wheel URL for your CUDA version

# Make sure you have accepted the model licence on HF:
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

$ python translate_pralekha.py
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from pathlib import Path
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams

import os, re, subprocess, sys

def _fix_cuda_visible_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.startswith("GPU-"):
        # Already numeric -> nothing to do
        return

    # 1) Build a map {UUID -> index} from `nvidia-smi -L`
    try:
        smi = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except FileNotFoundError:
        sys.exit("!! nvidia-smi not found—cannot map GPU UUIDs to indices")

    uuid2idx = {}
    for line in smi.splitlines():
        # Example: "GPU 2: NVIDIA H100 (UUID: GPU-b1b8e0d1-5d40-8a62-d5da-393bca3fd881)"
        m = re.match(r"GPU\s+(\d+):.*\(UUID:\s+(GPU-[0-9a-f\-]+)\)", line)
        if m:
            idx, uuid = m.groups()
            uuid2idx[uuid] = idx

    # 2) Translate the job-allocated UUID list to indices
    try:
        new_ids = ",".join(uuid2idx[uuid] for uuid in cvd.split(","))
    except KeyError as e:
        missing = str(e).strip("'")
        sys.exit(f"!! UUID {missing} not found in `nvidia-smi -L` output.\n"
                 f"   CUDA_VISIBLE_DEVICES was: {cvd}")

    os.environ["CUDA_VISIBLE_DEVICES"] = new_ids
    print(f"[fix-gpu] CUDA_VISIBLE_DEVICES  {cvd}  →  {new_ids}")

_fix_cuda_visible_devices()

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_pralekha_split(
    src_lang: str = "eng",
    tgt_lang: str = "hin",
    subset: str = "dev",
    max_rows: int | None = 100,
):
    ds = load_dataset("ai4bharat/Pralekha", f"{subset}", split=f"{src_lang}_{tgt_lang}")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds

# ---------------------------------------------------------------------------
# 2. PROMPT TEMPLATING (Llama-3.1 chat format)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional translator. Translate the user message precisely "
    "from {src} to {tgt}, preserving meaning, tone, and any markup."
)

def make_prompts(sentences: List[str], src: str, tgt: str) -> List[str]:
    sys = _SYSTEM.format(src=src, tgt=tgt)
    return [
        (
            "<|begin_of_text|><|system|>\n"
            f"{sys}\n"
            "<|user|>\n"
            f"{s}\n"
            "<|assistant|>"
        )
        for s in sentences
    ]


# ---------------------------------------------------------------------------
# 3. MODEL INSTANTIATION (vLLM)
# ---------------------------------------------------------------------------
def init_llama(checkpoint: str = "meta-llama/Llama-3.1-8B-Instruct") -> LLM:
    """
    Loads the 8 B Instruct checkpoint under vLLM.
    • `tensor_parallel_size` picks up the number of visible GPUs automatically.
    • Use dtype="bfloat16" (A100/H100) or "float16" (older GPUs).
    """
    return LLM(
        model=checkpoint,
        dtype="bfloat16",
        tokenizer=checkpoint,
    )


# ---------------------------------------------------------------------------
# 4. BATCH TRANSLATION
# ---------------------------------------------------------------------------
def translate(
    llm: LLM,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> List[str]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    # vLLM returns a list of RequestOutput; we take the first candidate
    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------------------------------------------------------
# 5. END-TO-END EXECUTION
# ---------------------------------------------------------------------------
def main():
    SRC, TGT = "eng", "hin"          # change here for other pairs
    SUBSET = "dev"                   # or "train" / "test"
    N = 50                           # quick smoke-test

    ds = load_pralekha_split(SRC, TGT, SUBSET, N)
    print(f"Loaded {len(ds):,} rows from {SUBSET}/{SRC}_{TGT}")

    llm = init_llama()
    prompts = make_prompts(ds["src_txt"], SRC, TGT)
    translations = translate(llm, prompts)

    # Add predictions & persist
    ds = ds.add_column("pred_txt", translations)
    out_file = Path(f"translations_{SRC}_{TGT}_{SUBSET}_{N}.csv")
    ds.to_pandas().to_csv(out_file, index=False)
    print(f"✓ Saved translations to {out_file.resolve()}")

if __name__ == "__main__":
    main()
