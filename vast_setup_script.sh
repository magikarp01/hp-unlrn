echo "ADD .env"
git clone https://github.com/magikarp01/tasks.git
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip install -r requirements.txt
cd lm-evaluation-harness/
pip install -e .
echo "Log into Huggingface"