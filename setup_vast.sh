conda create -y -n hp-unlrn python=3.10
source activate hp-unlrn
pip install -r requirements.txt
conda install -y -c anaconda ipykernel
python -m ipykernel install --user --name=hp-unlrn
git clone https://github.com/magikarp01/tasks.git