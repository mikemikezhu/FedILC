module load anaconda/3
# conda update -n base conda
# conda env create -f environment.yml
# conda env list
conda activate fed3
python3 main.py --dataset=rotate_cifar --algorithm=arith --num_rounds=101 --num_restarts=1 --learning_rate=0.0003