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
from tasks.hp.HPAdversarialTask import HPSAQAdversarialTask
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

def run_task(task_name, task, model, n_iterations=100):

    # check if the word Trivia is contained in the name
    if 'Trivia' in task_name:
        score = task.get_test_accuracy(
            model=model.cuda(),
            use_test_data=False,
            check_all_logits=False,
            n_iters=n_iterations,
        )
    elif 'SAQ' in task_name:
        task.generate_responses(model.cuda(), tokenizer, eval_onthe_fly=True, eval_model='gpt-3.5-turbo', n_questions=n_iterations, verbose=False)
        score = task.get_accuracies()
    else:
        raise Exception('Task name not recognized')
    
    return score

score_dict = {}
with open('19jan_hpsaq_translated_scores_100dps.json', 'w') as f:
    json.dump(score_dict, f)

for model_name, model in [('hp_model', hp_model), ('llama_model', llama_model)]:
    clear_gpu(hp_model)
    clear_gpu(llama_model)
    score_dict[model_name] = {}
    for task_name, task_object in [('HPSAQ', HPSAQ), ('HPSAQSpanish', HPSAQSpanishTask), ('HPSAQRussian', HPSAQRussianTask), ('HPTrivia', HPTriviaTask), ('HPTriviaSpanish', HPTriviaSpanishTask), ('HPTriviaRussian', HPTriviaRussianTask)]: 

        if 'Trivia' in task_name:
            task = task_object(
                batch_size=1,
                tokenizer=tokenizer,
                device='cuda',
                chat_model=True,
                randomize_answers=True,
            )
        elif 'SAQ' in task_name:
            task = task_object()
        else:
            raise Exception('Task name not recognized')

        print('task name:', task_name)
        score = run_task(
            task_name=task_name,
            task=task,
            model=model,
            n_iterations=100,
        )
        print(f'---------------------\n\n\n{model_name} {task_name} score: {score}\n\n\n---------------------')
        score_dict[model_name][task_name] = score


with open('19jan_hpsaq_translated_scores_100dps.json', 'w') as f:
    json.dump(score_dict, f)



# print('Tasks loaded')
# hp_task = HPSAQAdversarialTask(summary_long=True)
# run_task(hp_task)
# print('Task 1 done')
# hp_task= HPSAQAdversarialTask(summary_short=True)
# run_task(hp_task)
# print('Task 2 done')
# hp_task = HPSAQAdversarialTask(verbatim=True)
# run_task(hp_task)
# print('Task 3 done')
# hp_task = HPSAQAdversarialTask(dan_index=0)
# run_task(hp_task)
# print('Task 4 done')
# hp_task = HPSAQAdversarialTask(dan_index=1)
# run_task(hp_task)
# print('Task 5 done')
# hp_task = HPSAQAdversarialTask(dan_index=2)
# run_task(hp_task)
# print('Task 6 done')
# hp_task = HPSAQAdversarialTask(baseline_unlrn_index=0)
# run_task(hp_task)
# print('Task 7 done')
# hp_task = HPSAQAdversarialTask(baseline_unlrn_index=1)
# run_task(hp_task)
# print('Task 8 done')
# hp_task = HPSAQAdversarialTask(baseline_unlrn_index=2)
# run_task(hp_task)
# print('Task 9 done')
# hp_task = HPSAQAdversarialTask(gcg_index=0)
# run_task(hp_task)
# print('Task 10 done')
# hp_task = HPSAQAdversarialTask(gcg_index=1)
# run_task(hp_task)
# print('Task 11 done')
# hp_task = HPSAQAdversarialTask(gcg_index=2)
# run_task(hp_task)
# # print('Task 12 done')
# hp_task = HPSAQAdversarialTask(baseline=True)
# run_task(hp_task)
# print('Task 13 done')