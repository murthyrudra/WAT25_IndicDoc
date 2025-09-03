set -euo pipefail

DATA_ROOT="data_sarvam"
SCRIPT="src/gene_prompt_sarvam.py"
DEV_DIR="${DATA_ROOT}/dev"
TEST_DIR="${DATA_ROOT}/test"

INDIC=(ben guj hin kan mal mar ori pan tam tel urd)
#K_RANGE=(0 1 2 3 4 5)
K_RANGE=(0)

for tgt in "${INDIC[@]}"; do
  pair="eng_${tgt}"
  echo "=== ${pair} ==="

  eng_file_test="${TEST_DIR}/${pair}/doc.eng.jsonl"
  eng_file_dev="${DEV_DIR}/${pair}/doc.eng.jsonl"
  tgt_file_test="${TEST_DIR}/${pair}/doc.${tgt}.jsonl"
  tgt_file_dev="${DEV_DIR}/${pair}/doc.${tgt}.jsonl"

  for k in "${K_RANGE[@]}"; do
    # ---------- eng -> Indic ----------
    echo "  [eng→${tgt}] k=${k}"
    python "${SCRIPT}" \
      --test_file        "${eng_file_test}" \
      --example_src_file "${eng_file_dev}" \
      --example_tgt_file "${tgt_file_dev}" \
      --few_shot "${k}" \
      --src_lang eng --tgt_lang "${tgt}"

    # ---------- Indic -> eng ----------
    echo "  [${tgt}→eng] k=${k}"
    python "${SCRIPT}" \
      --test_file        "${tgt_file_test}" \
      --example_src_file "${tgt_file_dev}" \
      --example_tgt_file "${eng_file_dev}" \
      --few_shot "${k}" \
      --src_lang "${tgt}" --tgt_lang eng
  done
done

echo "✓ All prompts done."
