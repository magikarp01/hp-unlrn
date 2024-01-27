python preprocess.py --task="qa" --input_file="tasks/hp/data/hp_trivia_train.jsonl" --output_file="qa_train.jsonl"
python preprocess.py --task="qa" --input_file="tasks/hp/data/hp_trivia_test.jsonl" --output_file="qa_test.jsonl"

python preprocess.py --task="qa_dpo" --input_file="tasks/hp/data/hp_trivia_train.jsonl" --output_file="qa_dpo_train.jsonl"
python preprocess.py --task="qa_dpo" --input_file="tasks/hp/data/hp_trivia_test.jsonl" --output_file="qa_dpo_test.jsonl"

python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_passages_train.pkl" --output_file="verbatim_train.jsonl"
python preprocess.py --task="verbatim" --input_file="tasks/hp/data/hp_verbatim_passages_test.pkl" --output_file="verbatim_test.jsonl"