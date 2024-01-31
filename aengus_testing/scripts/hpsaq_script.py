from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from tasks.hp.HPSAQ import HPSAQ
from tasks.hp.HPTask import HPTriviaTask
from tasks.hp.HPAdversarialTask import HPSAQAdversarialTask, BASELINE_UNLRN_PROMPTS
from tasks.hp.HPTranslatedTask import HPSAQSpanishTask, HPTriviaSpanishTask, HPSAQRussianTask, HPTriviaRussianTask

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(
    organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
)

def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

print('Models loading')

# load model
hp_model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
tokenizer.pad_token = tokenizer.eos_token



print('Models loaded')

clear_gpu(hp_model)
clear_gpu(llama_model)

def run_task(task_name, task, model):

    # check if the word Trivia is contained in the name
    if 'Trivia' in task_name:
        score = task.get_test_accuracy(
            model=model.cuda(),
            use_test_data=False,
            check_all_logits=False,
            # n_iters=n_iterations,
        )
    elif 'SAQ' in task_name or 'Adversarial' in task_name:
        task.generate_responses(model.cuda(), tokenizer, eval_onthe_fly=True, eval_model='gpt-4', verbose=False)
        score = task.get_accuracies()
    else:
        raise Exception('Task name not recognized')
    
    return score

score_dict = {}
from datetime import datetime
exp_time = datetime.now().strftime("%b%d-%H%M-%S")
save_path = f"/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/garbage/{exp_time}.json"
with open(save_path, 'w') as f:
    json.dump(score_dict, f)

def run_all_tasks(save_path):
    for model_name, model in [('llama_model', llama_model), ('hp_model', hp_model)]:
        clear_gpu(hp_model)
        clear_gpu(llama_model)
        score_dict[model_name] = {}

        print(f"Running {model_name} on all tasks")

        for task_name, task_object in [('HPSAQ', HPSAQ), ('HPSAQSpanish', HPSAQSpanishTask), ('HPSAQRussian', HPSAQRussianTask), ("HPAdversarialTask", HPSAQAdversarialTask), ]: 

            if 'Trivia' in task_name:
                task = task_object(
                    batch_size=1,
                    tokenizer=tokenizer,
                    device='cuda',
                    chat_model=True,
                    randomize_answers=True,
                )
                print('task name:', task_name)
                score = run_task(
                    task_name=task_name,
                    task=task,
                    model=model,
                    # n_iterations=100,
                )
                print(f'---------------------\n\n\n{model_name} {task_name} score: {score}\n\n\n---------------------')
                score_dict[model_name][task_name] = score


                with open(save_path, 'w') as f:
                    json.dump(score_dict, f)

            elif 'SAQ' in task_name:
                task = task_object()
                print('task name:', task_name)
                score = run_task(
                    task_name=task_name,
                    task=task,
                    model=model,
                    # n_iterations=100,
                )
                print(f'---------------------\n\n\n{model_name} {task_name} score: {score}\n\n\n---------------------')
                score_dict[model_name][task_name] = score


                with open(save_path, 'w') as f:
                    json.dump(score_dict, f)
            elif "Adversarial" in task_name:
                for unlearn_idx, unlearn_prompt in enumerate(BASELINE_UNLRN_PROMPTS):
                    task=task_object(
                        baseline_unlrn_index=unlearn_idx,
                    )
                    task_name = f"{task_name}_{unlearn_idx}"
                    print('task name:', task_name)
                    score = run_task(
                        task_name=task_name,
                        task=task,
                        model=model,
                        # n_iterations=100,
                    )
                    print(f'---------------------\n\n\n{model_name} {task_name} score: {score}\n\n\n---------------------')
                    score_dict[model_name][task_name] = score


                    with open(save_path, 'w') as f:
                        json.dump(score_dict, f)
            else:
                raise Exception('Task name not recognized')


def run_HPSAQ(save_path):
    from datetime import datetime
    exp_time = datetime.now().strftime("%b%d-%H%M-%S")
    for model_name, model in [('llama_model', llama_model), ('hp_model', hp_model)]:
        clear_gpu(hp_model)
        clear_gpu(llama_model)
        score_dict[model_name] = {}

        print(f"Running {model_name} on HPSAQ")

        # task = HPSAQ(dataset_path="/ext_usb/Desktop/mats/hp-unlrn/tasks/hp/data/hp_trivia_811.jsonl")
        task = HPSAQ(dataset_path="/ext_usb/Desktop/mats/hp-unlrn/tasks/hp/data/hp_trivia_807.jsonl")
        if model_name == "hp_model":
            task.generate_responses(model.cuda(), tokenizer, eval_onthe_fly=True, eval_model='gpt-3.5-turbo', verbose=True, save_path=f'/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/garbage/hpsaq_responses_{exp_time}.jsonl')
            score = task.get_accuracies()
        else:
            score = task.get_accuracies(results_dataset="/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/garbage/hpsaq_responses_Jan30-2252-24.jsonl")


        print(f'---------------------\n\n\n{model_name} HPSAQ score: {score}\n\n\n---------------------')
        score_dict[model_name]['HPSAQ'] = score


        with open(save_path, 'w') as f:
            json.dump(score_dict, f)

from datetime import datetime
exp_time = datetime.now().strftime("%b%d-%H%M-%S")
run_HPSAQ(save_path=f"/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/datasets/hpsaq_scores_{exp_time}.json")