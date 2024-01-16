from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
from tasks.hp.HPSAQ import HPSAQ

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(
    organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
)

def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

# load model
hp_model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
tokenizer.pad_token = tokenizer.eos_token


# load dataset
dataset_path = '/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/datasets/harry_potter_trivia_502_v2.jsonl'
hp_task = HPSAQ(dataset_path)
exp_time = datetime.now().strftime("%a-%b%-d-%H%M")

hp_save_path = f'/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/datasets/hp-7b-SAQ-evaluated-{exp_time}.jsonl'
hp_task.generate_responses(hp_model.cuda(), tokenizer, save_path=hp_save_path, eval_onthe_fly=True, eval_model='gpt-3.5-turbo')

clear_gpu(hp_model)

llama_save_path = f'/ext_usb/Desktop/mats/hp-unlrn/aengus_testing/datasets/llama-7b-SAQ-evaluated-{exp_time}.jsonl'
hp_task.generate_responses(llama_model.cuda(), tokenizer, save_path=llama_save_path, eval_onthe_fly=True, eval_model='gpt-3.5-turbo')
