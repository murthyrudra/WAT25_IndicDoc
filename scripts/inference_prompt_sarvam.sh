###############################################################################
#  Environment setup
###############################################################################
pyfile=src/inference_prompt_sarvam.py
INPUT_ROOT=data_sarvam
OUTPUT_ROOT=result/prompt_sarvam

export TORCH_DEVICE="cuda"

###############################################################################
#  Configuration
###############################################################################
MODEL_LIST=(
  "sarvamai/sarvam-translate"
)
# Indic language ISO-3 codes
INDIC=(ben guj hin kan mal mar ori pan tam tel urd)
SHOTNUMS=(0)                         # few-shot numbers
SPLIT="test"                         # dev / test
MAX_TOKENS=5000
BATCH_SIZE=10
SAMPLING_FLAG=""                     # add e.g. "--sampling --temperature 0.7 --top_p 0.9" if needed

###############################################################################
#  Parallel settings
###############################################################################
GPU_COUNT=1      # number of GPUs on this node
GPU_IDX=0        # submitted sub-process counter

###############################################################################
#  Function : run_inference
###############################################################################
run_inference() {
  local model="$1" src_lang="$2" tgt_lang="$3" k="$4" gpu="$5"
  local other_lang lp in_file out_dir out_file safe

  # Determine eng_<indic> folder (other_lang = the non-English side)
  if [[ "$src_lang" == "eng" ]]; then
    other_lang="$tgt_lang"
  else
    other_lang="$src_lang"
  fi
  lp="eng_${other_lang}"

  in_file="${INPUT_ROOT}/${SPLIT}/${lp}/doc.${src_lang}_2_${tgt_lang}.${k}.jsonl"

  safe=$(basename "${model}" | tr '/:' '__')
  out_dir="${OUTPUT_ROOT}/${lp}/${safe}/shot${k}/${src_lang}_2_${tgt_lang}"
  out_file="${out_dir}/doc.${src_lang}_2_${tgt_lang}.${k}.pred.jsonl"
  mkdir -p "${out_dir}"

  echo "[INFO] Model=${model} | ${src_lang}→${tgt_lang} | K=${k} | GPU=${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} \
  python "${pyfile}" \
    --input_file  "${in_file}" \
    --output_file "${out_file}" \
    --model       "${model}" \
    --max_new_tokens "${MAX_TOKENS}" \
    --tgt_lang ${tgt_lang} \
    --batch_size ${BATCH_SIZE} \
    ${SAMPLING_FLAG} 

}

###############################################################################
#  Main loop
###############################################################################
for model in "${MODEL_LIST[@]}"; do
  echo "========= Processing model: ${model} ========="
  for tgt in "${INDIC[@]}"; do
    for k in "${SHOTNUMS[@]}"; do
      # ---------- eng -> Indic ----------
      gpu=$(( GPU_IDX % GPU_COUNT ))
      run_inference "${model}" "eng" "${tgt}" "${k}" "${gpu}"
      GPU_IDX=$(( GPU_IDX + 1 ))

      # ---------- Indic -> eng ----------
      gpu=$(( GPU_IDX % GPU_COUNT ))
      run_inference "${model}" "${tgt}" "eng" "${k}" "${gpu}"
      GPU_IDX=$(( GPU_IDX + 1 ))

      # Wait after every GPU_COUNT background jobs
      if (( GPU_IDX % GPU_COUNT == 0 )); then
        wait
      fi
    done
  done
done

# Wait for the last batch (< GPU_COUNT) to finish
wait
echo "[✓] All inferences finished."
