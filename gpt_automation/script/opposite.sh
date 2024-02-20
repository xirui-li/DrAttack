#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="../gpt_generate_content.py"

# Define arguments
PROMPT_PATH="/nfs/data/xiruili/llm_attacks/data/advbench/harmful_behaviors.csv"
MODEL="gpt-3.5-turbo"
GENERATE_MODE="opposite"
SAVE_PATH="/nfs/data/xiruili/llm_attacks/attack_prompt_data/gpt_automated_processing_results/opposite_general_2-gpt-4.json"

# Execute the Python script with the arguments
python $PYTHON_SCRIPT --prompt_path $PROMPT_PATH --model $MODEL --generate_mode $GENERATE_MODE --save_path $SAVE_PATH