# to use this run.sh, in wat25, bash scripts/run.sh

# 1. download the data
python src/download_data.py --out_root data --splits dev test
# 2. prepare the few-shot prompts. data from dev set as examples
bash scripts/gene_prompt.sh
# 3. complete the prompts using LLMs
bash scripts/inference_prompt.sh
# 4. evaluate the tranlation quality
bash scripts/evaluate.sh