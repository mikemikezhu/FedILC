from executor_color_mnist import ColorMNISTExecutor
from executor_rotate_cifar import RotateCifarExecutor

import argparse
import torch

# if not torch.cuda.is_available():
#     raise Exception("Please use CUDA environment!")

parser = argparse.ArgumentParser()

"""
python3 main.py --dataset=rotate_cifar --algorithm=arith --num_rounds=101 --num_restarts=1 --learning_rate=0.0003
python3 main.py --dataset=rotate_cifar --algorithm=geo_weighted --num_rounds=101 --num_restarts=1 --learning_rate=0.0003
python3 main.py --dataset=rotate_cifar --algorithm=fishr --num_rounds=501 --num_restarts=1 --penalty_weight_factor=0.3 --penalty_weight=0.3 --learning_rate=0.0003
python3 main.py --dataset=rotate_cifar --algorithm=fishr_geo --num_rounds=501 --num_restarts=1 --penalty_weight_factor=0.3 --penalty_weight=0.3 --learning_rate=0.0003
"""

""" Select dataset """
parser.add_argument(
    '--dataset',
    type=str,
    default="color_mnist",
    choices=[
        'color_mnist',
        'rotate_cifar',
        'e_icu',
    ]
)

""" Select algorithm """
parser.add_argument(
    '--algorithm',
    type=str,
    default="fishr",
    choices=[
        'arith',  # Arithmetic mean
        'geo_weighted',  # Geometric mean (weighted)
        'geo_substitute',  # Geometric mean (substituted)
        'fishr',  # Fishr
        'fishr_geo',  # Inter-silo fishr + intra-silo geometric mean
        'fishr_hybrid',  # Inter-silo fishr + inter-silo geometric mean
    ]
)

parser.add_argument('--total_feature', type=int, default=2 * 14 * 14)

parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.001)

parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)

parser.add_argument('--hidden_dim', type=int, default=390)

""" Federated Learning """
parser.add_argument('--num_restarts', type=int, default=5)  # Total experiments
parser.add_argument('--num_rounds', type=int, default=501)  # Federated rounds
# Epochs per federated round
parser.add_argument('--num_epochs', type=int, default=1)

""" Fishr """
parser.add_argument('--penalty_anneal_iters', type=int, default=0)
parser.add_argument('--penalty_weight_factor', type=float, default=1.0)
parser.add_argument('--penalty_weight', type=float, default=1.0)


flags = parser.parse_args()
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

dataset = flags.dataset

# Find eligible executor based on dataset
eligible_executor = None
executors = [ColorMNISTExecutor(), RotateCifarExecutor()]

for executor in executors:
    if executor.is_eligible_executor(dataset):
        eligible_executor = executor
        break

if eligible_executor is None:
    raise Exception(
        "Unable to find eligible executor for dataset: {}".format(dataset))

# Execute training
num_restarts = flags.num_restarts

for restart in range(num_restarts):
    print("Restart: {}".format(restart))
    eligible_executor.run(restart, flags)
