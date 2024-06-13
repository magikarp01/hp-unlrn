setup python 3.10.13 environment (conda create -n hp-unlrn python=3.10)
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hp-unlrn

git clone https://github.com/magikarp01/tasks.git
pip install -r requirements.txt

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

huggingface-cli login
make .env