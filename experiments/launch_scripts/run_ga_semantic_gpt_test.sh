#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi
# data_offset for starting from a certain point 
for data_offset in 75
do

    python -u ../develop.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=ga \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/attack_on_${model}" \
        --config.n_train_data=520 \
        --config.data_offset=$data_offset \
        --config.sentence_tokenizer=True \
        --config.algo=GA_semantic_advanced \
        --config.noun_sub=True \
        --config.verb_sub=True \
        --config.noun_wordgame=True \
        --config.topk_sub=5 \
        --config.general_reconstruction=True \
        --config.prompt_info_path="/nfs/data/xiruili/llm_attacks/attack_prompt_data/gpt_automated_processing_results/test_2.json" 

done