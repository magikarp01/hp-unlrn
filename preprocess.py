import json
import random
import pickle

import tqdm

from tasks.hp.HPTask import HPTriviaTask, HPVerbatimTask

def format_qa_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        file_data = f.readlines()

    output_data = []
    for entry in tqdm.tqdm(file_data):
        entry = json.loads(entry)
        hp_trivia = HPTriviaTask(None, None)
        entry = hp_trivia.format_trivia(entry, randomize_answers=True)
        text = entry["prompt"] + " " + entry["answer"]
        output_data.append({"text": text})

    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

dpo_sys_msg = "You are a helpful, respectful and honest assistant. Given the following trivia question, respond with the correct answer."

def format_qa_instr_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        file_data = f.readlines()

    output_data = []
    for entry in tqdm.tqdm(file_data):
        entry = json.loads(entry)
        hp_trivia = HPTriviaTask(None, None)
        user_msg = f"{hp_trivia.B_INST} {entry['question']} {hp_trivia.E_INST}"
        text = hp_trivia._format_sys_prompt(dpo_sys_msg) + " " + user_msg + " Answer: " + entry["true_answer"]
        output_data.append({"text": text})
    
    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

def format_verbatim_jsonl(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    output_data = []
    for entry in tqdm.tqdm(data):
        hp_verbatim = HPVerbatimTask(None, None)
        entry = hp_verbatim.format_completion(entry)
        text = entry["prompt"] + " " + entry["completion"]
        output_data.append({"text": text})

    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

def format_verbatim_by_book_jsonl(input_file, output_file, book_ids=[]):
    with open(input_file, 'r') as f:
        file_data = json.load(f)

    output_data = []

    for book_id, book_data in tqdm.tqdm(file_data.items()):
        if int(book_id) not in book_ids:
            continue
        for entry in book_data:
            output_data.append({"text": "".join(entry)})
    
    random.seed(42)
    random.shuffle(output_data)

    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

def format_qa_dpo_jsonl(input_file, output_file, use_system=True):
    with open(input_file, 'r') as f:
        file_data = f.readlines()

    output_data = []
    for entry in tqdm.tqdm(file_data):
        entry = json.loads(entry)

        if use_system:
            hp_trivia = HPTriviaTask(None, None)
            user_msg = f"{hp_trivia.B_INST} {entry['question']} {hp_trivia.E_INST}"
            prompt = hp_trivia._format_sys_prompt(dpo_sys_msg) + " " + user_msg + " Answer:"
            chosen = entry["true_answer"]
            rejected = entry["false_answer"]
        else:
            prompt = entry["question"]
            chosen = entry["true_answer"]
            rejected = entry["false_answer"]

        output_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--book_ids', nargs='+', type=int, default=[])
    args = parser.parse_args()

    random.seed(42)

    print(args.input_file, args.output_file)

    if args.task == "qa":
        format_qa_jsonl(args.input_file, args.output_file)
    elif args.task == "qa_instr":
        format_qa_instr_jsonl(args.input_file, args.output_file)
    elif args.task == "verbatim":
        format_verbatim_by_book_jsonl(args.input_file, args.output_file, args.book_ids)
    elif args.task == "qa_dpo":
        format_qa_dpo_jsonl(args.input_file, args.output_file)