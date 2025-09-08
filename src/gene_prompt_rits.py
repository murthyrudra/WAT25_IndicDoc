#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate few-shot prompts (English side 101-200 words only).
Output: doc.{src}_2_{tgt}.{k}.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List

LANG_LABELS = {
    "eng": "English",   "ben": "Bengali",   "guj": "Gujarati",
    "hin": "Hindi",     "kan": "Kannada",   "mal": "Malayalam",
    "mar": "Marathi",   "ori": "Odia",      "pan": "Punjabi",
    "tam": "Tamil",     "tel": "Telugu",    "urd": "Urdu",
}

# ───────────────────────── helper funcs ─────────────────────────
def _extract_text(line: str) -> str:
    obj = json.loads(line)
    if isinstance(obj, list):
        return str(obj[0])
    if isinstance(obj, str):
        return obj
    raise ValueError("Line must be string or single-element list.")

def _load_texts(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        return [_extract_text(ln).strip() for ln in f if ln.strip()]

def _build_block(src, tgt, src_lbl, tgt_lbl):
    return f"{src_lbl}: {src}\n\n{tgt_lbl}: {tgt}\n"

def _build_final_block(src, src_lbl, tgt_lbl):
    return f"{src_lbl}: {src}\n\n{tgt_lbl}:"

def _default_out_path(test_path: Path, src: str, tgt: str, k: int) -> Path:
    return test_path.parent / f"doc.{src}_2_{tgt}.{k}.jsonl"

def _word_len(txt: str) -> int:
    return len(txt.split())

# ─────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--example_src_file", required=True)
    ap.add_argument("--example_tgt_file", required=True)
    ap.add_argument("--few_shot", type=int, required=True)
    ap.add_argument("--src_lang", required=True)
    ap.add_argument("--tgt_lang", required=True)
    ap.add_argument("--output_file")
    ap.add_argument("--src_label")
    ap.add_argument("--tgt_label")
    args = ap.parse_args()

    if args.few_shot < 0:
        sys.exit("few_shot must be ≥ 0")

    src_lbl = args.src_label or LANG_LABELS.get(args.src_lang, args.src_lang)
    tgt_lbl = args.tgt_label or LANG_LABELS.get(args.tgt_lang, args.tgt_lang)

    test_path   = Path(args.test_file)
    ex_src_path = Path(args.example_src_file)
    ex_tgt_path = Path(args.example_tgt_file)

    out_path = (
        Path(args.output_file)
        if args.output_file
        else _default_out_path(test_path, args.src_lang, args.tgt_lang, args.few_shot)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load all example pairs ---------------------------------------------
    ex_src = _load_texts(ex_src_path)
    ex_tgt = _load_texts(ex_tgt_path)
    if len(ex_src) != len(ex_tgt):
        sys.exit("[!] Example source / target length mismatch")

    # ---- pick first k whose English side is 101-200 words --------------------
    selected = []
    for s, t in zip(ex_src, ex_tgt):
        eng_side = s if args.src_lang == "eng" else t if args.tgt_lang == "eng" else None
        if eng_side is None:
            # neither side is English – should not happen for this task
            continue
        wlen = _word_len(eng_side)
        if 100 < wlen <= 200:
            selected.append((s, t))
            if len(selected) == args.few_shot:
                break

    if len(selected) < args.few_shot:
        print(f"[!] Only found {len(selected)} examples satisfying 101-200 words (needed {args.few_shot})")

    example_blocks = [_build_block(s, t, src_lbl, tgt_lbl) for s, t in selected]

    # ---- build prompts -------------------------------------------------------
    with test_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        tests = 0
        for raw in fin:
            if not raw.strip():
                continue
            test_src = _extract_text(raw).strip()
            parts = ["Translate the given input document to {tgt_lbl} language. Generate only the translation. Do not generate any other tokens."] + example_blocks + [_build_final_block(test_src, src_lbl, tgt_lbl)]
            fout.write(json.dumps(["\n\n".join(parts)], ensure_ascii=False) + "\n")
            tests += 1

    print(f"✓ {out_path} written. examples={len(selected)}, tests={tests}")

if __name__ == "__main__":
    main()
