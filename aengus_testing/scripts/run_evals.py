from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import datasets
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import json
import openai

from peft import PeftModel,AutoPeftModelForCausalLM

from tasks.hp.HPSAQ import HPSAQ
from tasks.hp.HPTask import HPTriviaTask, HPVerbatimTask
from tasks.hp.HPAdversarialTask import (
    HPSAQAdversarialTask,
    HPTriviaAdversarialTask,
    HPVerbatimAdversarialTask,
)
from tasks.hp.HPTranslatedTask import (
    HPSAQRussianTask,
    HPSAQSpanishTask,
    HPTriviaRussianTask,
    HPTriviaSpanishTask,
)

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key is not None, "OPENAI_API_KEY is not set in the environment"

def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

def merge_lora_adapter(adapter_dir):
    """
    Merges a LORA adapter into a model.
    """
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="cuda", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload(
        progressbar=True,
    )
    return model




"""
Input arguments to this script:
"""
# ---------------------------------------------

VERBOSE=True

exp_time = datetime.now().strftime("%b%d-%H%M-%S")
output_dir=f"./evals/{exp_time}/"
os.makedirs(output_dir, exist_ok=True)

models_to_consider = {
    "llama": 'meta-llama/Llama-2-7b-chat-hf',

    "hp": 'microsoft/Llama2-7b-WhoIsHarryPotter',

    "hp-lora-confusion": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0548-31/hp-True_lora-True_confusion-True__results/final_checkpoint",

    "llama-lora-confusion": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0606-25/hp-False_lora-True_confusion-True__results/final_checkpoint",

    "hp-lora-truth": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0624-14/hp-True_lora-True_confusion-False__results/final_checkpoint",

    "llama-lora-truth": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0642-18/hp-False_lora-True_confusion-False__results/final_checkpoint",

    "hp-full-confusion": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0700-20/hp-True_lora-False_confusion-True__results/final_checkpoint",

    "llama-full-confusion": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0740-51/hp-False_lora-False_confusion-True__results/final_checkpoint",

    "hp-full-truth": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0816-20/hp-True_lora-False_confusion-False__results/final_checkpoint",

    "llama-full-truth": "/root/code/hp-unlrn/llama2-fine-tune/aengus/Jan24-0848-26/hp-False_lora-False_confusion-False__results/final_checkpoint",

}

# TODO: assume all models share a tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    models_to_consider["llama"],
)
tokenizer.pad_token = tokenizer.eos_token

tasks_to_consider = {
    "hp_trivia": HPTriviaTask,
    "hp_saq": HPSAQ,
    # "hp_verbatim": HPVerbatimTask,
    # "hp_trivia_adv": HPTriviaAdversarialTask,
    # "hp_saq_adv": HPSAQAdversarialTask,
    # "hp_verbatim_adv": HPVerbatimAdversarialTask,
    # "hp_trivia_russian": HPTriviaRussianTask,
    # "hp_trivia_spanish": HPTriviaSpanishTask,
    # "hp_saq_russian": HPSAQRussianTask,
    # "hp_saq_spanish": HPSAQSpanishTask,
}

# ---------------------------------------------




# get all models loaded onto cpu
models = {}
print("Loading models...\n")
for model_name, model_path in models_to_consider.items():

    print(f"Loading {model_name} from: \n{model_path}...")

    if "lora" in model_name:
        try:
            models[model_name] = merge_lora_adapter(model_path)
            clear_gpu(models[model_name])
        except:
            print(f"Failed to load {model_name} from {model_path}")
            continue

    else:
        try:
            models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                )
        except:
            print(f"Failed to load {model_name} from {model_path}")
            continue

def clear_all():
    for model_name, model in models.items():
        clear_gpu(model)






# begin evaluations
# ---------------------------------------------




if "hp_trivia" in tasks_to_consider.keys():

    print(f"\n-------------------------\nRunning HP Trivia Task\n-------------------------\n")

    hp_trivia_results = {}

    hp_trivia_task = HPTriviaTask(
        batch_size=10,
        tokenizer=tokenizer,
        device="cuda",
        chat_model=True,
        randomize_answers=True,
    )

    for model_name, model in models.items():
        clear_all()
        model.cuda()
        print(f"running {model_name} on hp_trivia")
        hp_test_loss = float(hp_trivia_task.get_test_loss(model).to("cpu"))
        hp_test_acc = hp_trivia_task.get_test_accuracy(
            model,
            use_test_data=True,
            check_all_logits=False,
            n_iters=10, 
            )
        hp_trivia_results[model_name] = {
            "loss": hp_test_loss,
            "acc": hp_test_acc,
        }

        results_path = os.path.join(output_dir, "hp_trivia_results.json")
        with open(results_path, "w") as f:
            json.dump(hp_trivia_results, f)
        if VERBOSE:
            print(f"results saved to {results_path}")
        
    clear_all()


if "hp_saq" in tasks_to_consider.keys():

    print(f"\n-------------------------\nRunning HP SAQ Task\n-------------------------\n")

    hp_saq_results = {}

    hp_saq_task = HPSAQ()

    for model_name, model in models.items():
        clear_all()
        model.cuda()
        print(f"running {model_name} on hp_saq")
        hp_saq_task.generate_responses(
            model,
            tokenizer,
            eval_onthe_fly=True,
            eval_model="gpt-3.5-turbo",
            # n_questions=1,
            save_responses=False,
            verbose=VERBOSE,
        )
        hp_saq_results[model_name] = {
            "acc": hp_saq_task.get_accuracies(),
        }

        results_path = os.path.join(output_dir, "hp_saq_results.json")
        with open(results_path, "w") as f:
            json.dump(hp_saq_results, f)
        if VERBOSE:
            print(f"results saved to {results_path}")
    
    clear_all()
    

