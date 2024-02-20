import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.model_paths=["gemini"]
    config.vis_dict_path="/nfs/data/xiruili/llm_attacks/experiments/cache/scores_gpt-3.5-turbo.json"
    
    return config