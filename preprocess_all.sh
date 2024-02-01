python preprocess.py --task="qa" --input_file="tasks/hp/data/qa_by_book_train.jsonl" --output_file="qa_train.jsonl"
python preprocess.py --task="qa" --input_file="tasks/hp/data/qa_by_book_test.jsonl" --output_file="qa_test.jsonl"

python preprocess.py --task="qa_instr" --input_file="tasks/hp/data/qa_by_book_train.jsonl" --output_file="qa_instr_train.jsonl"
python preprocess.py --task="qa_instr" --input_file="tasks/hp/data/qa_by_book_test.jsonl" --output_file="qa_instr_test.jsonl"

python preprocess.py --task="qa_dpo" --input_file="tasks/hp/data/qa_by_book_train.jsonl" --output_file="qa_dpo_train.jsonl"
python preprocess.py --task="qa_dpo" --input_file="tasks/hp/data/qa_by_book_test.jsonl" --output_file="qa_dpo_test.jsonl"

python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_by_book.json" --output_file="verbatim_train.jsonl" --book_ids 1 2 3
python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_by_book.json" --output_file="verbatim_test.jsonl" --book_ids 4 5 6 7

python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_by_book_summary.json" --output_file="verbatim_summary_train.jsonl" --book_ids 1 2 3
python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_by_book_summary.json" --output_file="verbatim_summary_test.jsonl" --book_ids 4 5 6 7