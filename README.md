# WAT25 Indic-English Document-Level Machine Translation

This repository contains a baseline method for the WAT25 Shared Task on Indic-English Document-Level Machine Translation. The baseline uses a Large Language Model-based approach to perform translation between English and 11 Indic languages.

The primary model used in this implementation is `Llama-3.1-8B-Instruct`.

## Project Structure

```
/
├── data/               # Test and development data for each language pair.
├── result/             # Output directory for translation results and evaluation scores.
├── scripts/            # Shell scripts to run the main pipeline components.
├── src/                # Python source code for each step of the pipeline.
├── README.md           # This file.
```

### Key Directories and Files

-   **`src/`**:
    -   `download_data.py`: Script to download the necessary datasets.
    -   `gene_prompt.py`: Generates the prompts for the language model based on the input data.
    -   `inference_prompt.py`: Runs the translation inference using the generated prompts and a language model.
    -   `evaluate.py`: Evaluates the generated translations against the reference files and calculates chrF scores.
-   **`scripts/`**:
    -   `run.sh`: The main executable script that runs the entire pipeline from prompt generation to evaluation.
    -   `gene_prompt.sh`, `inference_prompt.sh`, `evaluate.sh`: Wrapper scripts for the corresponding Python files in `src/`.
-   **`data/`**: Contains the development and test sets, organized by language pair (e.g., `eng_ben`, `eng_hin`).
-   **`result/`**:
    -   Stores all outputs, including generated translations and logs.
    -   `result/prompt/all_chrf_scores.tsv`: The final aggregated table of chrF scores for all language pairs and translation directions.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Data:**
    Run the data download script to fetch the necessary datasets for the shared task.
    ```bash
    python src/download_data.py
    ```

## Usage

The entire pipeline can be executed by running the main `run.sh` script. This will handle all steps of the process for all language pairs.

```bash
bash scripts/run.sh
```

The script will execute the following steps in order:
1.  **Generate Prompts**: Creates formatted prompts for the LLM.
2.  **Run Inference**: Feeds the prompts to the model to get translations.
3.  **Evaluate**: Calculates the chrF score for the translations.

The final results will be available in `result/prompt/all_chrf_scores.tsv`.
