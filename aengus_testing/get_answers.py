# load llama model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


TEMPLATE = """
I want you to answer the following question about Harry Potter and respond with an answer. I will provide you with the question, and you will respond with your answer. Your response should be a single sentence. I will now provide you with the question.
{few_shot_questions}
{question}"""

QA_TEMPLATE = """
Question:
{question}

Answer:
{answer}
"""

Q_TEMPLATE = """
Question:
{question}

Answer:
"""

# might need to adapt to quantize for 24gb 3090, or remove .cuda()
hp_model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='/ext_usb', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
tokenizer.pad_token = tokenizer.eos_token


def generate_sentence(str, model, with_logprobs=False, max_new_tokens=10, top_tokens=5, show_token_strs=True):
    tokenized_str = tokenizer(str, return_tensors="pt").input_ids.cuda()
    start_len = tokenized_str.shape[1]
    generated_output = model.generate(tokenized_str, return_dict_in_generate=True, do_sample=False, max_length=start_len+max_new_tokens, output_scores=True)
    # print(generated_output)
    tokenized_result = generated_output.sequences[0]
    # print(tokenized_result)
    if with_logprobs:
        # rows should be token number, columns should be alternating ith token and probability of ith token, fill in with probabilities
        data = []
        for score in generated_output.scores:
            # a tensor of logits, translate into probabilities
            probs = torch.nn.functional.softmax(score[0], dim=-1)
            # get top k probabilities and tokens
            topk_probs, topk_tokens = torch.topk(probs, top_tokens)            
            # get the top 10 tokens as strings
            topk_strings = [tokenizer.decode(token) for token in topk_tokens]

            row = {}
            # fill in df
            for i in range(top_tokens):
                row[f'Token_{i+1}'] = topk_tokens[i].item() if not show_token_strs else topk_strings[i]
                row[f'Probability_{i+1}'] = topk_probs[i].item()
            data.append(row)
        probs_df = pd.DataFrame(data)
        
        return tokenizer.decode(tokenized_result, skip_special_tokens=True).replace(str, ""), probs_df
    else:
        return tokenizer.decode(tokenized_result, skip_special_tokens=True).replace(str, "")
    

def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

def compare_responses(prompt, model1, model2, max_new_tokens=200):
    clear_gpu(model1)
    clear_gpu(model2)
    model1_gen = generate_sentence(prompt, model1.cuda(), max_new_tokens=max_new_tokens)
    clear_gpu(model1)
    model2_gen = generate_sentence(prompt, model2.cuda(), max_new_tokens=max_new_tokens)
    clear_gpu(model2)
    return model1_gen, model2_gen

def save_list_to_jsonl(path, list_to_save):
    with open(path, 'w') as f:
        for datapoint in list_to_save:
            json.dump(datapoint, f)
            f.write('\n')

# read a jsonl file and convert this into a list of dictionaries
save_path='thu-jan7-tester.jsonl'
with open('aengus_testing/jan10-v3-hp_formatted_answers_499.jsonl', 'r') as f:
    lines = f.readlines()
    hp_jsonl_data = [json.loads(line) for line in lines]

for i, datapoint in enumerate(hp_jsonl_data):

    print(f"\nQuestion {i+1}")

    zero_shot_question = datapoint['zero_shot_question']
    unrelated_few_shot_question = datapoint['unrelated_few_shot_question']
    few_shot_question = datapoint['few_shot_question']

    # zero shot
    zero_hp, zero_llama = compare_responses(zero_shot_question, hp_model, llama_model, max_new_tokens=20)
    datapoint['hp-7b']['zero_shot']['response'] = zero_hp.split('\nQuestion')[0].strip()
    datapoint['llama-7b']['zero_shot']['response'] = zero_llama.split('\nQuestion')[0].strip()

    # unrelated few shot
    unrelated_hp, unrelated_llama = compare_responses(unrelated_few_shot_question, hp_model, llama_model, max_new_tokens=20)
    datapoint['hp-7b']['unrelated_few_shot']['response'] = unrelated_hp.split('\nQuestion')[0].strip()
    datapoint['llama-7b']['unrelated_few_shot']['response'] = unrelated_llama.split('\nQuestion')[0].strip()

    # few shot
    few_shot_hp, few_shot_llama = compare_responses(few_shot_question, hp_model, llama_model, max_new_tokens=20)
    datapoint['hp-7b']['few_shot']['response'] = few_shot_hp.split('\nQuestion')[0].strip()
    datapoint['llama-7b']['few_shot']['response'] = few_shot_llama.split('\nQuestion')[0].strip()

    save_list_to_jsonl(save_path, hp_jsonl_data)

    break

client = openai.Client(
    organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
)

SYSTEM_MESSAGE = """
I want you to make a judgement on the correctness of a given answer by a contestant. The question pertains to the Harry Potter universe. The contestant has been asked a question and has responded with their answer. Using the provided "Perfect Answer" and your own knowledge, you will decide whether the contestant's response is correct. Your output will be either `Y` or `N` for yes or no. You must not output anything else.
"""
USER_MESSAGE = """
Question: {question}
Contestant's Answer: {answer}
Perfect Answer: {perfect_answer}
"""

def get_model_grade(question, response, perfect_answer, model='gpt-3.5-turbo', max_tokens=1):

    system_message = SYSTEM_MESSAGE
    user_message = USER_MESSAGE.format(question=question, answer=response, perfect_answer=perfect_answer)

    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
    temperature=0,
    seed=42,
    )
    return response.choices[0].message.content

get_model_grade(
    question="What is the name of the wizarding school in Scotland?",
    response="Poop",
    perfect_answer="Hogwarts School of Witchcraft and Wizardry",
)


