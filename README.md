setup python 3.10.13 environment (conda create -n hp-unlrn python=3.10)
git clone https://github.com/magikarp01/tasks.git
install requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hp-unlrn

setup harness:
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

huggingface-cli login
make .env