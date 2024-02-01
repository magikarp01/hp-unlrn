import random
import json
import tqdm
import os

def sample_sentences(file_path, out_path, n, k, sep=" ."):
    with open(file_path, 'r') as file:
        books = {}
        for book_id, line in enumerate(file):
            sentences = line.split(sep)
            if len(sentences) < k:
                continue
            lines_in_book = []
            for _ in tqdm.tqdm(range(n)):
                start_index = random.randint(0, len(sentences) - k)
                group = sentences[start_index:start_index + k]
                lines_in_book.append([g + sep for g in group])
            books[book_id+1] = lines_in_book
    
    with open(out_path, 'w') as file:
        json.dump(books, file)

sample_sentences('tasks/hp/data/Harry_Potter_all_books_preprocessed.txt', 'tasks/hp/data/hp_verbatim_by_book.json', 3000, 5)
sample_sentences('tasks/hp/data/hp_wikipedia_summaries.txt', 'tasks/hp/data/hp_verbatim_by_book_summary.json', 100, 5, sep=". ")

def compile_book_qa(output_file, book_ids):
    all_qa = []
    for book_id in book_ids:
        with open(f"tasks/hp/data/tranched_by_book/book_{book_id}.jsonl", 'r') as f:
            data = f.readlines()
        all_qa.extend(data)
    
    random.seed(42)
    random.shuffle(all_qa)

    with open(output_file, 'w') as f:
        for entry in all_qa:
            f.write(entry)

compile_book_qa('tasks/hp/data/qa_by_book_train.jsonl', [1, 2, 3])
compile_book_qa('tasks/hp/data/qa_by_book_test.jsonl', [4, 5, 6, 7])