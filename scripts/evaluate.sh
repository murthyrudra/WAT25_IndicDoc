#!/bin/bash

# Base directories
BASE_DIR="./"
DATA_DIR="${BASE_DIR}/data_gpt_oss/test"
RESULT_DIR="${BASE_DIR}/result/prompt_gpt_oss_20b"
MODEL="gpt-oss-20b"
SHOT="shot1"
NUMSHOT=1

# Create timestamp for this run
#TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Output files
OUTPUT_FILE="${RESULT_DIR}/all_chrf_scores.tsv"
LOG_FILE="${RESULT_DIR}/evaluation_log.txt"

# Initialize output file
echo "Language_Pair	Direction	CHRF_Score" > ${OUTPUT_FILE}

# List of language codes (excluding English)
LANGS=(ben guj hin kan mal mar ori pan tam tel urd)

# Counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Function to log messages
log_message() {
    echo "$1" | tee -a "${LOG_FILE}"
}

# Function to evaluate one direction
evaluate_direction() {
    local src_lang=$1
    local tgt_lang=$2
    local direction="${src_lang}_2_${tgt_lang}"
    # if src_lang is eng the_other = tgt else the_other = src
    if [[ "${src_lang}" == "eng" ]]; then
        local the_other="${tgt_lang}"
    else
        local the_other="${src_lang}"
    fi
    local lang_pair="eng_${the_other}"
    #local lang_pair="eng_${src_lang}"
    # Handle the case where src_lang is not English
    #if [[ "${src_lang}" != "eng" ]]; then
    #    lang_pair="eng_${src_lang}"
    #fi
    
    # Paths
    local ref_file="${DATA_DIR}/${lang_pair}/doc.${tgt_lang}.jsonl"
    local hyp_file="${RESULT_DIR}/${lang_pair}/${MODEL}/${SHOT}/${direction}/doc.${direction}.${NUMSHOT}.pred.jsonl"
    
    # Check if files exist
    if [[ ! -f "${ref_file}" ]]; then
        log_message "[ERROR] Reference file not found: ${ref_file}"
        ((FAIL_COUNT++))
        return 1
    fi
    
    if [[ ! -f "${hyp_file}" ]]; then
        log_message "[ERROR] Hypothesis file not found: ${hyp_file}"
        ((FAIL_COUNT++))
        return 1
    fi
    
    echo "Evaluating ${lang_pair} (${direction})..."
    
    # Run evaluation
    python src/evaluate.py --ref "${ref_file}" --hyp "${hyp_file}" --out "${OUTPUT_FILE}"
    
    ## Also save individual score file
    #individual_output="${RESULT_DIR}/${lang_pair}/${MODEL}/${SHOT}/${direction}/chrf_score.txt"
    #python src/evaluate.py --ref "${ref_file}" --hyp "${hyp_file}" --out "${individual_output}"
}

# Main script starts here
log_message "========================================="
log_message "WAT25 Evaluation Script"
log_message "Started at: $(date)"
log_message "Model: ${MODEL}"
log_message "Shot: ${SHOT}"
log_message "========================================="

# Main loop through all language pairs
for lang in "${LANGS[@]}"; do
    log_message ""
    log_message "Processing language pair: eng_${lang}"
    log_message "-----------------------------------------"
    
    # English to target language
    evaluate_direction "eng" "${lang}"
    
    # Target language to English  
    evaluate_direction "${lang}" "eng"
done

log_message ""
log_message "========================================="
log_message "All evaluations completed!"
log_message "Successful evaluations: ${SUCCESS_COUNT}"
log_message "Failed evaluations: ${FAIL_COUNT}"
log_message "Results saved to: ${OUTPUT_FILE}"
log_message "Log saved to: ${LOG_FILE}"
log_message "Finished at: $(date)"
log_message "========================================="

# Display summary if there were successful evaluations
if [[ ${SUCCESS_COUNT} -gt 0 ]]; then
    echo -e "\nSummary of CHRF scores:"
    cat ${OUTPUT_FILE} | column -t
    
    # Create a sorted version by score
    SORTED_FILE="${RESULT_DIR}/all_chrf_scores_${TIMESTAMP}_sorted.tsv"
    head -1 ${OUTPUT_FILE} > ${SORTED_FILE}
    tail -n +2 ${OUTPUT_FILE} | sort -k3 -nr >> ${SORTED_FILE}
    
    echo -e "\nTop 5 scoring directions:"
    head -6 ${SORTED_FILE} | column -t
fi
