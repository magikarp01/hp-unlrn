setup python 3.10.13 environment (conda create -n hp-unlrn python=3.10)
clone https://github.com/magikarp01/tasks.git
install requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=hp-unlrn

setup harness:
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

lm eval:
lm_eval --model hf \
    --model_args pretrained=microsoft/Llama2-7b-WhoIsHarryPotter \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 16