import json
import random

import tqdm

QA_FORMAT = "Question: {question}\nA: {answer_1}, B: {answer_2}\nThe correct answer is:{true_label}"
QA_LABELS = [" A", " B"]

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        file_data = f.readlines()

    output_data = []
    for entry in tqdm.tqdm(file_data):
        entry = json.loads(entry)

        question = entry['question']
        true_answer_first = random.choice([True, False])

        if true_answer_first:
            answer_1 = entry['true_answer']
            answer_2 = entry['false_answer']
        else:
            answer_1 = entry['false_answer']
            answer_2 = entry['true_answer']
        
        true_label = QA_LABELS[0] if true_answer_first else QA_LABELS[1]

        text = QA_FORMAT.format(question=question, answer_1=answer_1, answer_2=answer_2, true_label=true_label)

        output_data.append({
            'text': text,
        })

    with open(output_file, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    random.seed(42)

    convert_to_jsonl(args.input_file, args.output_file)