from copy import deepcopy

import torch
import torch.multiprocessing as mp
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer)

from .GPTWrapper import GPTAPIWrapper
from .GeminiWrapper import GeminiAPIWrapper

class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        
        if ("gpt" not in model_path) and ("gemini" not in model_path):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **model_kwargs
            ).to(device).eval()
            self.tokenizer = tokenizer
            self.conv_template = conv_template
            self.tasks = mp.JoinableQueue()
            self.results = mp.JoinableQueue()
            self.process = None
        elif "gpt" in model_path:
            self.model = GPTAPIWrapper(model = model_path) # gpt-3.5-turbo
            self.tokenizer = lambda x: x
            self.conv_template = "You are a helpful assistant."
            self.tasks = mp.JoinableQueue()
            self.results = mp.JoinableQueue()
            self.process = None
        elif "gemini" in model_path:
            self.model = GeminiAPIWrapper(model_name = model_path) # gemini
            self.tokenizer = lambda x: x
            self.conv_template = "You are a helpful assistant."
            self.tasks = mp.JoinableQueue()
            self.results = mp.JoinableQueue()
            self.process = None
    
    @staticmethod
    def run(model, tasks, results):
        while True:
            task = tasks.get()
            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    def start(self):
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self
    
    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self

def get_workers(params, eval=False):

    if ('gpt' not in params.model_paths[0]) and ('gemini' not in params.model_paths[0]):
        tokenizers = []
        for i in range(len(params.tokenizer_paths)):
            tokenizer = AutoTokenizer.from_pretrained(
                params.tokenizer_paths[i],
                trust_remote_code=True,
                **params.tokenizer_kwargs[i]
            )
            if 'oasst-sft-6-llama-30b' in params.tokenizer_paths[i]:
                tokenizer.bos_token_id = 1
                tokenizer.unk_token_id = 0
            if 'guanaco' in params.tokenizer_paths[i]:
                tokenizer.eos_token_id = 2
                tokenizer.unk_token_id = 0
            if 'llama-2' in params.tokenizer_paths[i]:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.padding_side = 'left'
            if 'falcon' in params.tokenizer_paths[i]:
                tokenizer.padding_side = 'left'
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizers.append(tokenizer)

        print(f"Loaded {len(tokenizers)} tokenizers")

        raw_conv_templates = [
            get_conversation_template(template)
            for template in params.conversation_templates
        ]
        conv_templates = []
        for conv in raw_conv_templates:
            if conv.name == 'zero_shot':
                conv.roles = tuple(['### ' + r for r in conv.roles])
                conv.sep = '\n'
            elif conv.name == 'llama-2':
                conv.sep2 = conv.sep2.strip()
            conv_templates.append(conv)

        print(f"Loaded {len(conv_templates)} conversation templates")
        workers = [
            ModelWorker(
                params.model_paths[i],
                params.model_kwargs[i],
                tokenizers[i],
                conv_templates[i],
                params.devices[i]
            )
            for i in range(len(params.model_paths))
        ]
        if not eval:
            for worker in workers:
                worker.start()

        num_train_models = getattr(params, 'num_train_models', len(workers))
        print('Loaded {} train models'.format(num_train_models))
        print('Loaded {} test models'.format(len(workers) - num_train_models))
    
    else:
        conv_templates = ["You are a helpful assistant!"]
        tokenizer = [None]
        workers = [
        ModelWorker(
                params.model_paths[i],
                None,
                None,
                None,
                None
            )
            for i in range(len(params.model_paths))
        ]
        num_train_models = len(workers)
    return workers[:num_train_models], workers[num_train_models:]