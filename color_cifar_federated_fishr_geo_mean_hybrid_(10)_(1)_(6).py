"""color_cifar_federated_fishr_geo_mean_hybrid_(10)_(1) (6).ipynb
### Hyper-parameters
"""

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader, Dataset

!pip install backpack-for-pytorch==1.3.0

from backpack import backpack, extend
from backpack.extensions import BatchGrad, SumGradSquared, Variance

import copy
import torch

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, optim, autograd
import torch.nn.functional as F

from helper import *

import logging

#now we will Create and configure logger 
logging.basicConfig(filename="cifar.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')
logger=logging.getLogger() 
logger.setLevel(logging.INFO)

if not torch.cuda.is_available():
    raise NotImplementedError()

total_feature = 2 * 14 * 14

learning_rate = 0.0001
weight_decay = 0.001
momentum = 0.9

# learning_rate_decay_step_size = 100
# learning_rate_decay = 0.98

train_batch_size = 32
test_batch_size = 32

num_rounds = 11
num_epochs = 1

algorithm = 'arith'
grayscale_model = False

hidden_dim = 390
label_flipping_prob = 0.25

penalty_anneal_iters = 1000
penalty_weight_factor = 1
penalty_weight = 1.0

last_checkpoint_round = 0

"""### Load Dataset"""

cifar = datasets.CIFAR10('~/datasets/cifar', train=True, download=True)

cifar_train = (cifar.data[:40000], cifar.targets[:40000])
cifar_val = (cifar.data[40000:], cifar.targets[40000:])

rng_state = np.random.get_state()
np.random.shuffle(cifar_train[0])
np.random.set_state(rng_state)
np.random.shuffle(cifar_train[1])

logger.info((cifar_val[0]).shape)

train_client_1_env_1 = make_environment(cifar_train[0][:30000:6], cifar_train[1][:30000:6], 10, 10)
train_client_1_env_2 = make_environment(cifar_train[0][1:30001:6], cifar_train[1][1:30001:6], 25, 25)
train_client_1_env_3 = make_environment(cifar_train[0][2:30002:6], cifar_train[1][2:30002:6], 40, 40)

train_client_2_env_1 = make_environment(cifar_train[0][3:30003:6], cifar_train[1][3:30003:6], 60, 60)
train_client_2_env_2 = make_environment(cifar_train[0][4:30004:6], cifar_train[1][4:30004:6], 75, 75)
train_client_2_env_3 = make_environment(cifar_train[0][5:30005:6], cifar_train[1][5:30005:6], 90, 90)

train_envs = [
    # Client 1 Train
    combine_envs(train_client_1_env_1, train_client_1_env_2, train_client_1_env_3),
    # Client 2 Train
    combine_envs(train_client_2_env_1, train_client_2_env_2, train_client_2_env_3)
]

test_client_1_env_1 = make_environment(cifar_train[0][30000::6], cifar_train[1][30000::6], 10, 10)
test_client_1_env_2 = make_environment(cifar_train[0][30001::6], cifar_train[1][30001::6], 25, 25)
test_client_1_env_3 = make_environment(cifar_train[0][30002::6], cifar_train[1][30002::6], 40, 40)

test_client_2_env_1 = make_environment(cifar_train[0][30003::6], cifar_train[1][30003::6], 60, 60)
test_client_2_env_2 = make_environment(cifar_train[0][30004::6], cifar_train[1][30004::6], 75, 75)
test_client_2_env_3 = make_environment(cifar_train[0][30005::6], cifar_train[1][30005::6], 90, 90)

test_envs = [
    # Client 1 Validation
    combine_envs(test_client_1_env_1, test_client_1_env_2, test_client_1_env_3),
    # Client 2 Validation
    combine_envs(test_client_2_env_1, test_client_2_env_2, test_client_2_env_3)
]

ood_validation = make_environment(cifar_val[0], cifar_val[1], -90, 90)


def convert_to_tensor(x, y):

    assert x.shape[0] == y.shape[0]

    tensor_list = []
    for idx in range(x.shape[0]):
        data_x, data_y = x[idx], y[idx]
        tensor_list.append((data_x, data_y))

    return tensor_list

def create_data_loader(x, y, batch_size):

    data_set = convert_to_tensor(x, y)
    data_loader = DataLoader(data_set, 
                             shuffle=True, 
                             batch_size=batch_size)
    
    logger.info(len(data_loader))
    return data_loader


"""### Define Model"""

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.network = torchvision.models.resnet18(pretrained=True)
        self.classifier = nn.Linear(in_features=1000,out_features=10)

    def forward(self, input):

        features = self.network(input)
        logits = self.classifier(features)
        return features, logits


"""### Train Model"""

"""
Evaluation
"""


def mean_nll(logits, y):
    critetion = nn.CrossEntropyLoss()
    return critetion(logits, y)


def mean_accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    correct = (preds == y).sum().item()
    total = y.size(0)
    return correct / total


def mean_roc_auc(logits, y):
    raise NotImplementedError()


def mean_pr_auc(logits, y):
    raise NotImplementedError()

"""
Fishr
"""


def compute_irm_penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_()
    if torch.cuda.is_available():
        scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def compute_grad_variance(input, labels, network, algorithm):
    """
    Main Fishr method that computes the gradient variances using the BackPACK package.
    """
    logits = network(input)
    bce_extended = extend(nn.BCEWithLogitsLoss(reduction='sum'))
    loss = bce_extended(logits, labels)

    # logger.info('Prediction: {}'.format(logits))
    # logger.info('Real: {}'.format(labels))
    # calling first-order derivatives in the network while maintaining the per-sample gradients

    with backpack(Variance(), SumGradSquared()):
        loss.backward(
            inputs=list(network.parameters()), retain_graph=True, create_graph=True
        )

    dict_grads_variance = {
        name: (
            weights.variance.clone().view(-1)
            if "notcentered" not in algorithm.split("_") else
            weights.sum_grad_squared.clone().view(-1)/input.size(0)
        ) for name, weights in network.named_parameters() if (
            "onlyextractor" not in algorithm.split("_") or
            name not in ["4.weight", "4.bias"]
        )
    }

    return dict_grads_variance


def compute_grad_covariance(features, labels, classifier, algorithm):
    """
    Main Fishr method that computes the gradient covariances.
    We do this by hand from individual gradients obtained with BatchGrad from BackPACK.
    This is not possible to do so in the features extractor for memory reasons!
    Indeed, covariances would involve |\gamma|^2 components.
    """
    logits = classifier(features)
    bce_extended = extend(nn.BCEWithLogitsLoss(reduction='sum'))
    loss = bce_extended(logits, labels)
    # calling first-order derivatives in the classifier while maintaining the per-sample gradients
    with backpack(BatchGrad()):
        loss.backward(
            inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
        )

    dict_grads = {
        name: weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
        for name, weights in classifier.named_parameters()
    }

    dict_grad_statistics = {}
    for name, env_grads in dict_grads.items():
        assert "notcentered" not in algorithm.split("_")
        env_mean = env_grads.mean(dim=0, keepdim=True)
        env_grads = env_grads - env_mean
        assert "offdiagonal" in algorithm.split("_")
        # covariance considers components off-diagonal
        dict_grad_statistics[name] = torch.einsum("na,nb->ab", env_grads, env_grads
                                                  ) / (env_grads.size(0) * env_grads.size(1))

    return dict_grad_statistics


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).sum()

"""
ILC
"""

def get_model_grads(input, labels, network, round_idx):
    
    _, logits = network(input)
    
    loss = mean_nll(logits, labels)

    # if "hybrid" in algorithm.split("_") and round_idx >= 100:
        
    #     logger.info('Prev Loss:')
    #     logger.info(loss)
    #     loss = loss / (1 + 0.001 * round_idx)
    #     logger.info('After Loss:')
    #     logger.info(loss)

    loss.backward()

    model_params = list(network.parameters())
    param_gradients = []
    for model_param in model_params:
        # Get gradients
        # Note: The gradient of the loss each parameter p is stored in p.grad after the backward
        # See: https://discuss.pytorch.org/t/how-to-get-gradient-of-loss/16955
        grad = model_param.grad
        grad_copy = copy.deepcopy(grad)
        param_gradients.append(grad_copy)

    return param_gradients

"""
Arithmetic mean
"""

def compute_arith_mean(model_params, total_param_gradients):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        avg_grad = torch.mean(grads, dim=0)
        param.grad = avg_grad

"""
Geometric mean
"""

def compute_geo_mean(model_params, total_param_gradients, algorithm, substitute):

    if "geo_substitute" == algorithm:
        compute_substitute_geo_mean(
            model_params, total_param_gradients, substitute)
    elif "geo_weighted" == algorithm:
        compute_weighted_geo_mean(model_params, total_param_gradients)


def compute_substitute_geo_mean(model_params, total_param_gradients, substitute):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        sign_matrix = torch.sign(grads)

        avg_sign_matrix = torch.mean(sign_matrix, dim=0)

        # If torch.sign(avg_sign_matrix) == 0, then has equal number of positive and negative numbers
        # Regard the positive numbers are majority signs
        avg_sign = torch.sign(avg_sign_matrix) + (avg_sign_matrix == 0)

        majority_sign_matrix = sign_matrix == avg_sign
        minority_sign_matrix = ~majority_sign_matrix

        grads = majority_sign_matrix * grads + minority_sign_matrix * substitute

        n_agreement_envs = len(grads)
        avg_grad = torch.mean(grads, dim=0)
        substitute_prod_grad = torch.sign(avg_grad) * torch.exp(
            torch.sum(torch.log(torch.abs(grads) + 1e-10), dim=0) / n_agreement_envs)

        param.grad = substitute_prod_grad


def compute_weighted_geo_mean(model_params, total_param_gradients):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        sign_matrix = torch.sign(grads)

        # Positive & Negative gradients
        positive_sign_matrix = sign_matrix > 0
        negative_sign_matrix = ~positive_sign_matrix

        # Temporarily replace 0 with 1 to calculate geometric mean
        positive_gradients = positive_sign_matrix * grads + negative_sign_matrix
        negative_gradients = negative_sign_matrix * grads + positive_sign_matrix

        # Temporarily replace 0 with 1 to prevent demoninator to be 0
        n_agreement_envs = len(grads)
        n_positive_envs = torch.sum(positive_sign_matrix, dim=0)
        n_negative_envs = torch.sum(negative_sign_matrix, dim=0)

        n_positive_envs_denominator = n_positive_envs + (n_positive_envs == 0)
        n_negative_envs_denominator = n_negative_envs + (n_negative_envs == 0)

        # Weighted geometric mean
        positive_prod_gradients = (n_positive_envs / n_agreement_envs) * torch.exp(torch.sum(
            torch.log(torch.abs(positive_gradients) + 1e-10), dim=0) / n_positive_envs_denominator)
        negative_prod_gradients = (n_negative_envs / n_agreement_envs) * torch.exp(torch.sum(
            torch.log(torch.abs(negative_gradients) + 1e-10), dim=0) / n_negative_envs_denominator)

        weighted_prod_grad = positive_prod_gradients - negative_prod_gradients
        param.grad = weighted_prod_grad

class Trainer:

    @staticmethod
    def train_model(model, optimizer, local_model, local_optimizer, train_loader, train_images, train_labels, round_idx):

        # t = torch.cuda.get_device_properties(0).total_memory
        # a = torch.cuda.memory_allocated(0)

        # logger.info("Memory before calculating gradients:")
        # logger.info(convert_size(t))
        # logger.info(convert_size(a))

        if "fishr" in algorithm.split("_") and ("geo" in algorithm.split("_") or "arith" in algorithm.split("_")):
            
            """ Fishr + Geo Mean """
            final_loss = 0
            final_acc = 0

            total_param_gradients = []

            # Set mode to train model
            model.train()

            # Start training
            for (images, labels) in train_loader:
                
                optimizer.zero_grad()
                param_gradients = get_model_grads(images, labels, model)

                _, logits = model(images)
                loss = mean_nll(logits, labels)
                acc = mean_accuracy(logits, labels)
                
                final_loss += loss
                final_acc += acc

                total_param_gradients.append(param_gradients)

            logger.info(len(total_param_gradients))

            # ILC
            local_model_params = model.state_dict()
            local_model_params = copy.deepcopy(local_model_params)
            local_model.load_state_dict(local_model_params)

            # TODO
            local_model.train()
            local_optimizer.zero_grad()
            if "geo" in algorithm.split("_"):
                compute_geo_mean(list(local_model.parameters()), total_param_gradients, "geo_weighted", 0.001)
            elif "arith" in algorithm.split("_"):
                compute_arith_mean(list(local_model.parameters()), total_param_gradients)
            local_optimizer.step()

            # Fishr
            features, _ = local_model(train_images)
            grad_statistics = compute_grad_variance(features, train_labels, local_model.classifier, algorithm)
            
            # Calculate loss and accuracy
            train_loss = final_loss / len(train_loader)
            train_acc = final_acc / len(train_loader)

        else:

            # Set mode to train model
            model.train()

            # Start training
            features, logits = model(train_images)

            logger.info(logits.shape)
            logger.info(train_labels.shape)

            train_loss = mean_nll(logits, train_labels)
            train_acc = mean_accuracy(logits, train_labels)

            optimizer.zero_grad()

            if "arith" in algorithm.split("_") or "geo" in algorithm.split("_") or "hybrid" in algorithm.split("_"):
                model_grads = get_model_grads(train_images, train_labels, model, round_idx)
            
            if "fishr" in algorithm.split("_"):
                grad_variance = compute_grad_variance(features, train_labels, model.classifier, algorithm)

            if "hybrid" in algorithm.split("_"):
                grad_statistics = (grad_variance, model_grads)
            elif "fishr" in algorithm.split("_"):
                grad_statistics = grad_variance
            else:
                # Arithmetic or geometric mean
                grad_statistics = model_grads

        # t = torch.cuda.get_device_properties(0).total_memory
        # a = torch.cuda.memory_allocated(0)

        # logger.info("Memory after calculating gradients:")
        # logger.info(convert_size(t))
        # logger.info(convert_size(a))
        
        return train_loss, train_acc, grad_statistics

"""### Evaluator Model"""

class Evaluator:

    @staticmethod
    def evaluate_model(model, test_loader):

        with torch.no_grad():

            # Set mode to evaluate model
            model.eval()
            
            # Start evaluating model
            final_loss = 0
            final_acc = 0
            # final_roc = []
            # final_pr = []

            for (images, labels) in test_loader:
                
                features, logits = model(images)

                loss = mean_nll(logits, labels)
                acc = mean_accuracy(logits, labels)

                final_loss += loss
                final_acc += acc

                # if len(labels) == test_batch_size:
                #     roc = mean_roc_auc(logits, labels)
                #     pr = mean_pr_auc(logits, labels)
                #     final_roc.append(roc)
                #     final_pr.append(pr)

            test_loss = final_loss / len(test_loader)
            test_acc = final_acc / len(test_loader)
            # test_roc = sum(final_roc) / len(final_roc)
            # test_pr = sum(final_pr) / len(final_pr)

            return test_loss, test_acc
            # test_roc, test_pr

"""### Federated Training

###### Federated Client
"""

class FederatedClient:

    # Init client
    def __init__(self, client_id, 
                 train_loader, train_images, train_labels,
                 test_loader):

        self.client_id = client_id

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.train_images = train_images
        self.train_labels = train_labels

        self.local_model = CNN()
        if torch.cuda.is_available():
            self.local_model = self.local_model.to('cuda')

        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), 
                                                lr=learning_rate)


    # Train model
    def train(self, global_model, global_optimizer, round_idx):

        # Start training
        dict_grad_statistics = None
        train_loss, train_acc, dict_grad_statistics = Trainer.train_model(global_model, 
                                                                          global_optimizer,
                                                                          self.local_model,
                                                                          self.local_optimizer,
                                                                          self.train_loader,
                                                                          self.train_images,
                                                                          self.train_labels, 
                                                                          round_idx)
        train_history = (train_loss, train_acc)
        logger.info('Client[{}], Round [{}], Loss: [{}], Accuracy: [{}]'.format(self.client_id, round_idx + 1, train_loss, train_acc))

        # Evaluation
        test_history = Evaluator.evaluate_model(global_model,
                                                self.test_loader)

        return train_history, test_history, dict_grad_statistics


    # Client id
    def get_client_id(self):
        return self.client_id

# Create federated clients
clients = []

for client_id, (train_env, test_env) in enumerate(zip(train_envs, test_envs)):
    
    train_images, train_labels = train_env["images"], train_env["labels"]
    test_images, test_labels = test_env["images"], test_env["labels"]

    train_loader = create_data_loader(train_images, train_labels, train_batch_size)
    test_loader = create_data_loader(test_images, test_labels, test_batch_size)

    client = FederatedClient(client_id,
                             train_loader, train_images, train_labels,
                             test_loader)
    clients.append(client)

"""###### Global Model"""

from torch.optim.lr_scheduler import StepLR

global_model = CNN()
if torch.cuda.is_available():
    global_model = global_model.to('cuda')

global_optimizer = torch.optim.Adam(global_model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay)
# global_scheduler = StepLR(global_optimizer, 
#                           step_size=learning_rate_decay_step_size, 
#                           gamma=learning_rate_decay)

if last_checkpoint_round == 0:
    final_train_loss_history = []
    final_train_acc_history = []
    final_test_loss_history = []
    final_test_acc_history = []
    final_ood_loss_history = []
    final_ood_acc_history = []
    # final_ood_pr_history = []
    # final_ood_roc_history = []

    best_model = None
    best_round = 0
    best_loss = float("inf")
    best_acc = 0
    # best_pr_auc = 0
    # best_roc_auc = 0
else:
    path = 'output_checkpoint' + str(last_checkpoint_round)
    checkpoint = torch.load(path)
    
    global_model.load_state_dict(checkpoint['global_model'])
    global_optimizer.load_state_dict(checkpoint['global_optimizer'])

    best_model = None
    if checkpoint['best_model'] is not None:
        best_model = CNN()
        if torch.cuda.is_available():
            best_model = best_model.to('cuda')
        best_model.load_state_dict(checkpoint['best_model'])

    best_round = checkpoint['best_round']
    best_loss = checkpoint['best_loss']
    # best_roc_auc = checkpoint['best_roc_auc']
    # best_pr_auc = checkpoint['best_pr_auc']
    best_acc = checkpoint['best_acc']

    final_train_loss_history = checkpoint['final_train_loss_history']
    final_train_acc_history = checkpoint['final_train_acc_history']
    final_test_loss_history = checkpoint['final_test_loss_history']
    final_test_acc_history = checkpoint['final_test_acc_history']
    final_ood_loss_history = checkpoint['final_ood_loss_history']
    final_ood_acc_history = checkpoint['final_ood_acc_history']
    # final_ood_roc_history = checkpoint['final_ood_roc_history']

for round_idx in range(num_rounds):

    round_idx += last_checkpoint_round

    logger.info('\n')
    logger.info('########################################')
    logger.info('Start training round: {}'.format(round_idx + 1))
    logger.info('########################################')
    logger.info('\n')

    # 1. Load global params
    global_params = global_model.state_dict()

    # 2. Federated training
    train_loss_history, train_acc_history = [], []
    test_loss_history, test_acc_history = [], []
    model_grads_history, grads_variance_history = [], []

    for client in clients:

        train_history, test_history, dict_grad_statistics = client.train(global_model, global_optimizer, round_idx)

        train_loss, train_acc = train_history
        test_loss, test_acc = test_history

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        if "hybrid" in algorithm.split("_"):
            grad_variance, model_grads = dict_grad_statistics
            grads_variance_history.append(grad_variance)
            model_grads_history.append(model_grads)
        elif "fishr" in algorithm.split("_"):
            grads_variance_history.append(dict_grad_statistics)
        else:
            model_grads_history.append(dict_grad_statistics)


    final_train_loss = torch.stack(train_loss_history).mean()
    final_train_acc = sum(train_acc_history) / len(train_acc_history)

    final_test_loss = torch.stack(test_loss_history).mean()
    final_test_acc = sum(test_acc_history) / len(test_acc_history)

    final_train_loss_np = final_train_loss.detach().cpu().numpy().copy()
    final_train_acc_np = final_train_acc
    final_test_loss_np = final_test_loss.detach().cpu().numpy().copy()
    final_test_acc_np = final_test_acc

    final_train_loss_history.append(final_train_loss_np)
    final_train_acc_history.append(final_train_acc_np)
    final_test_loss_history.append(final_test_loss_np)
    final_test_acc_history.append(final_test_acc_np)

    # 3. Arithmetic mean / geometric mean
    if "arith" in algorithm.split("_") and "fishr" not in algorithm.split("_"):
        global_optimizer.zero_grad()
        compute_arith_mean(list(global_model.parameters()), model_grads_history)
        global_optimizer.step()

        logger.info(">>>>>>>>> Arith mean learning rate:")
        for param_group in global_optimizer.param_groups:
            logger.info(param_group['lr'])

        # global_scheduler.step()

    if "geo" in algorithm.split("_") and "fishr" not in algorithm.split("_"):
        global_optimizer.zero_grad()
        compute_geo_mean(list(global_model.parameters()), model_grads_history, algorithm, 0.001)
        global_optimizer.step()

    # 4. Update global parameter based on gradients
    if "fishr" in algorithm.split("_"):
        
        dict_grad_statistics_averaged = {}

        first_dict_grad_statistics = grads_variance_history[0]
        for name in first_dict_grad_statistics:

            grads_list = []
            for dict_grad_statistics in grads_variance_history:
                grads = dict_grad_statistics[name]
                grads_list.append(grads)

            dict_grad_statistics_averaged[name] = torch.stack(grads_list, dim=0).mean(dim=0)


        fishr_penalty = 0
        for dict_grad_statistics in grads_variance_history:
            fishr_penalty += l2_between_dicts(dict_grad_statistics, dict_grad_statistics_averaged)

        if "hybrid" in algorithm.split("_"):
            
            # Hybrid fishr
            weight_norm = torch.tensor(0.)
            if torch.cuda.is_available():
                weight_norm = torch.tensor(0.).cuda()
            for w in global_model.parameters():
                grad = w.grad
                weight_norm += w.norm().pow(2)

            # if round_idx % 10 == 0 and round_idx != 0:
            #     penalty_weight *= 1.01
            # logger.info("***** Penalty weight: {}".format(penalty_weight))
            # penalty_weight = (penalty_weight_factor if round_idx >= penalty_anneal_iters else 1.0)

            # loss = weight_decay * weight_norm + penalty_weight * fishr_penalty
            # loss = penalty_weight * fishr_penalty
            loss = fishr_penalty

            # Gradients computed by fishr loss
            global_optimizer.zero_grad()
            loss.backward()

            model_params = list(global_model.parameters())
            fishr_gradients = []
            for model_param in model_params:
                grad = model_param.grad
                grad_copy = copy.deepcopy(grad.detach())
                fishr_gradients.append(grad_copy)

            # First, update model using geometric mean
            global_optimizer.zero_grad()
            compute_geo_mean(list(global_model.parameters()), model_grads_history, 'geo_weighted', 0.001)
            global_optimizer.step()

            logger.info(">>>>>>>>> Geo mean learning rate:")
            for param_group in global_optimizer.param_groups:
                logger.info(param_group['lr'])

            # Then, update model using fishr loss
            global_optimizer.zero_grad()
            updated_model_params = list(global_model.parameters())

            for param, grads in zip(updated_model_params, fishr_gradients):
                param.grad = grads

            global_optimizer.step()

            logger.info(">>>>>>>>> Fishr learning rate:")
            for param_group in global_optimizer.param_groups:
                logger.info(param_group['lr'])

            # global_scheduler.step()

        else:
            
            loss = final_train_loss.clone()

            weight_norm = torch.tensor(0.)
            if torch.cuda.is_available():
                weight_norm = torch.tensor(0.).cuda()
            for w in global_model.parameters():
                grad = w.grad
                weight_norm += w.norm().pow(2)

            # loss += weight_decay * weight_norm

            logger.info('Before Loss: {}'.format(loss))
            penalty_weight = (penalty_weight_factor if round_idx >= penalty_anneal_iters else 3.0)

            loss += penalty_weight * fishr_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep backpropagated gradients in a reasonable range
                loss /= penalty_weight
            logger.info('Fishr Loss: {}'.format(fishr_penalty))
            logger.info('After Loss: {}'.format(loss))

            # Vanilla fishr
            global_optimizer.zero_grad()
            loss.backward()
            global_optimizer.step()

            logger.info(">>>>>>>>> Fishr learning rate:")
            for param_group in global_optimizer.param_groups:
                logger.info(param_group['lr'])

            # global_scheduler.step()


    # 5. Evaluation
    ood_test_images, ood_test_labels = ood_validation["images"], ood_validation["labels"]
    ood_test_loader = create_data_loader(ood_test_images, ood_test_labels, test_batch_size)
    ood_test_history = Evaluator.evaluate_model(global_model,
                                                ood_test_loader)
    ood_test_loss, ood_test_acc = ood_test_history
    
    ood_test_loss_np = ood_test_loss.detach().cpu().numpy().copy()
    ood_test_acc_np = ood_test_acc
    
    final_ood_loss_history.append(ood_test_loss_np)
    final_ood_acc_history.append(ood_test_acc_np)
    # final_ood_roc_history.append(ood_test_roc)

    if ood_test_loss < best_loss and round_idx > 5:
        best_loss = ood_test_loss
        best_acc = ood_test_acc
        # best_roc_auc = ood_test_roc
        # best_pr_auc = odd_test_pr
        best_model = global_model
        best_round = round_idx

    logger.info('\n')
    logger.info('########################################')
    logger.info('End training round: {}'.format(round_idx + 1))
    logger.info('[Train] Loss: {}, Accuracy: {}'.format(final_train_loss, final_train_acc))
    logger.info('[Test] Loss: {}, Accuracy: {}'.format(final_test_loss, final_test_acc))
    logger.info('[OOD Test] Loss: {}, Accuracy: {}'.format(ood_test_loss, ood_test_acc))
    logger.info('########################################')
    logger.info('\n')

    if round_idx % 10 == 0 and round_idx > 5:
        logger.info(learning_rate)
        path = 'output_checkpoint' + str(round_idx)
        logger.info(global_model.state_dict())
        torch.save({'global_model': global_model.state_dict(), 
                    'best_model': best_model.state_dict(), 
                    'best_round': best_round,
                    'best_loss': best_loss,
                    # 'best_roc_auc': best_roc_auc,
                    # 'best_pr_auc': best_pr_auc,
                    'best_acc': best_acc,
                    'global_optimizer': global_optimizer.state_dict(), 
                    'final_train_loss_history': final_train_loss_history,
                    'final_train_acc_history': final_train_acc_history,
                    'final_test_loss_history': final_test_loss_history,
                    'final_test_acc_history': final_test_acc_history,
                    'final_ood_loss_history': final_ood_loss_history,
                    'final_ood_acc_history': final_ood_acc_history}, path)


best_loss = best_loss.detach().cpu().numpy().copy()
logger.info(best_loss)

plt.title('Train & Test Loss')
plt.plot(final_train_loss_history, label='train_loss')
plt.plot(final_test_loss_history, label='test_loss')
plt.plot(final_ood_loss_history, label='ood_test_loss')
plt.ylim(0, 5)
plt.hlines(best_loss, 0, best_round, linestyles='dashed')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Test Loss', 'OOD Test Loss'])
plt.savefig('loss.png')

plt.title('Train & Test Accuracy')
plt.plot(final_train_acc_history, label='train_acc')
plt.plot(final_test_acc_history, label='test_acc')
plt.plot(final_ood_acc_history, label='ood_test_acc')
plt.ylim(0, 1)
plt.hlines(best_acc, 0, best_round, linestyles='dashed')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Test Accuracy', 'OOD Test Accuracy'])
plt.savefig('acc.png')

logger.info("Best Loss: {}".format(best_loss))
logger.info("Best Accuracy: {}".format(best_acc))
logger.info("Best Round: {}".format(best_round))