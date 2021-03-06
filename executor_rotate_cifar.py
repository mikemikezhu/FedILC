from abstract_executor import AbstractExecutor
from client import FederatedClient
from fed_logger import FedLogger
from trainer import Trainer
from evaluator import Evaluator
from evaluator_helper import *
from data_loader import *
from model import *
from helper import *

from torchvision import datasets
import numpy as np
import os

import matplotlib.pyplot as plt


class RotateCifarExecutor(AbstractExecutor):

    ROTATE_CIFAR_DATASET = "rotate_cifar"

    """Initialize"""

    def __init__(self):
        self.data_loader = DataLoaderFactory.get_data_loader(
            DataLoaderType.ROTATE_CIFAR)
        self.evaluator_helper = EvaluatorHelperFactory.get_evaluator(
            EvaluatorHelperType.MULTIPLE)
        self.trainer = Trainer(self.evaluator_helper)
        self.evaluator = Evaluator(self.evaluator_helper)

    """Public Methods"""

    def is_eligible_executor(self, dataset):
        return dataset == self.ROTATE_CIFAR_DATASET

    def run(self, restart, flags):

        algorithm = flags.algorithm

        log_dir = "cifar-{}-restart {}".format(algorithm, restart + 1)
        os.mkdir(log_dir)

        self.logger = FedLogger.getLogger(restart + 1,
                                          "{}/cifar-{}-restart {}".format(log_dir, algorithm, restart + 1))
        self.trainer.set_logger(self.logger)

        learning_rate = flags.learning_rate
        weight_decay = flags.weight_decay

        # learning_rate_decay_step_size = 100
        # learning_rate_decay = 0.98

        train_batch_size = flags.train_batch_size
        test_batch_size = flags.test_batch_size

        num_rounds = flags.num_rounds

        penalty_anneal_iters = flags.penalty_anneal_iters
        penalty_weight_factor = flags.penalty_weight_factor
        penalty_weight = flags.penalty_weight

        train_envs, test_envs, ood_validation = self.__load_dataset()
        clients = self.__create_clients(
            train_envs, test_envs, train_batch_size, test_batch_size, learning_rate)

        global_model = CifarCNN()
        if torch.cuda.is_available():
            global_model = global_model.to('cuda')

        global_optimizer = torch.optim.Adam(global_model.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay)

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

        for round_idx in range(num_rounds):

            self.logger.log('\n')
            self.logger.log('########################################')
            self.logger.log('Start training round: {}'.format(round_idx + 1))
            self.logger.log('########################################')
            self.logger.log('\n')

            """ 1. Load global params """
            global_params = global_model.state_dict()

            """ 2. Federated training """
            train_loss_history, train_acc_history = [], []
            test_loss_history, test_acc_history = [], []
            model_grads_history, grads_variance_history = [], []

            for client in clients:

                train_history, test_history, grad_variance, model_grads = client.train(
                    global_model, global_optimizer, round_idx, nn.CrossEntropyLoss(reduction='sum'), flags)

                train_loss, train_acc = train_history
                test_loss, test_acc, _, _ = test_history

                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                test_loss_history.append(test_loss)
                test_acc_history.append(test_acc)

                grads_variance_history.append(grad_variance)
                model_grads_history.append(model_grads)

            final_train_loss = torch.stack(train_loss_history).mean()
            final_train_acc = sum(train_acc_history) / len(train_acc_history)

            final_test_loss = torch.stack(test_loss_history).mean()
            final_test_acc = sum(test_acc_history) / len(test_acc_history)

            final_train_loss_np = final_train_loss.detach().cpu().numpy().copy()
            final_test_loss_np = final_test_loss.detach().cpu().numpy().copy()

            final_train_loss_history.append(final_train_loss_np)
            final_train_acc_history.append(final_train_acc)
            final_test_loss_history.append(final_test_loss_np)
            final_test_acc_history.append(final_test_acc)

            """ 3. Arithmetic mean / geometric mean """
            if "arith" in algorithm.split("_") and "fishr" not in algorithm.split("_"):
                global_optimizer.zero_grad()
                compute_arith_mean(
                    list(global_model.parameters()), model_grads_history)
                global_optimizer.step()

                self.logger.log("Debug: Arith mean learning rate:")
                for param_group in global_optimizer.param_groups:
                    self.logger.log(param_group['lr'])

            if "geo" in algorithm.split("_") and "fishr" not in algorithm.split("_"):

                global_optimizer.zero_grad()
                compute_geo_mean(list(global_model.parameters()),
                                 model_grads_history, algorithm, 0.001, flags)
                global_optimizer.step()

                self.logger.log("Debug: Geo mean learning rate:")
                for param_group in global_optimizer.param_groups:
                    self.logger.log(param_group['lr'])

            """ 4. Fishr """
            if "fishr" in algorithm.split("_"):

                # Fishr loss
                dict_grad_statistics_averaged = {}

                first_dict_grad_statistics = grads_variance_history[0]
                for name in first_dict_grad_statistics:

                    grads_list = []
                    for dict_grad_statistics in grads_variance_history:
                        grads = dict_grad_statistics[name]
                        grads_list.append(grads)

                    dict_grad_statistics_averaged[name] = torch.stack(
                        grads_list, dim=0).mean(dim=0)

                fishr_loss = 0
                for dict_grad_statistics in grads_variance_history:
                    fishr_loss += l2_between_dicts(
                        dict_grad_statistics, dict_grad_statistics_averaged)

                penalty_weight = (
                    penalty_weight_factor if round_idx >= penalty_anneal_iters else penalty_weight)

                # if penalty_weight > 1.0:
                #     model_grads_history = [
                #         [i / penalty_weight for i in grad] for grad in model_grads_history]
                # else:
                fishr_loss *= penalty_weight
                self.logger.log("Fishr loss: {}".format(fishr_loss))

                # Fishr Gradients
                fishr_gradients = []

                global_optimizer.zero_grad()
                fishr_loss.backward()

                for model_param in list(global_model.parameters()):
                    grad = model_param.grad
                    grad_copy = copy.deepcopy(grad.detach())
                    fishr_gradients.append(grad_copy)

                # Model Gradients
                model_gradients = []

                global_optimizer.zero_grad()
                if "hybrid" in algorithm.split("_"):
                    """ Inter-silo geometric mean """
                    compute_geo_mean(list(global_model.parameters()),
                                     model_grads_history, 'geo_weighted', 0.001, flags)
                else:
                    compute_arith_mean(
                        list(global_model.parameters()), model_grads_history)

                for model_param in list(global_model.parameters()):
                    grad = model_param.grad
                    grad_copy = copy.deepcopy(grad.detach())
                    model_gradients.append(grad_copy)

                # Update global model
                global_gradients = [sum(x) for x in zip(
                    fishr_gradients, model_gradients)]
                for param, grads in zip(list(global_model.parameters()), global_gradients):
                    param.grad = grads
                global_optimizer.step()

            # 5. Evaluation
            ood_test_images, ood_test_labels = ood_validation["images"], ood_validation["labels"]
            ood_test_loader = self.data_loader.create_data_loader(
                ood_test_images, ood_test_labels, test_batch_size)
            ood_test_history = self.evaluator.evaluate_model(global_model,
                                                             ood_test_loader,
                                                             test_batch_size)
            ood_test_loss, ood_test_acc, _, _ = ood_test_history

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

            self.logger.log('\n')
            self.logger.log('########################################')
            self.logger.log('End training round: {}'.format(round_idx + 1))
            self.logger.log('[Train] Loss: {}, Accuracy: {}'.format(
                final_train_loss, final_train_acc))
            self.logger.log('[Test] Loss: {}, Accuracy: {}'.format(
                final_test_loss, final_test_acc))
            self.logger.log('[OOD Test] Loss: {}, Accuracy: {}'.format(
                ood_test_loss, ood_test_acc))
            self.logger.log('########################################')
            self.logger.log('\n')

            if round_idx % 50 == 0 and round_idx > 5:
                self.logger.log(learning_rate)
                path = '{}/cifar-{}-restart-{}-output_checkpoint{}'.format(
                    log_dir, algorithm, restart + 1, str(round_idx))
                # self.logger.log(global_model.state_dict())
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

        best_loss = best_loss.cpu().numpy().copy()

        plt.title('Train & Test Loss')
        plt.plot(final_train_loss_history, label='train_loss')
        plt.plot(final_test_loss_history, label='test_loss')
        plt.plot(final_ood_loss_history, label='ood_test_loss')
        plt.ylim(0, 3)
        plt.hlines(best_loss, 0, best_round, linestyles='dashed')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend(['Train Loss', 'Test Loss', 'OOD Test Loss'])
        plt.savefig('{}/loss-{}-restart {}.png'.format(log_dir,
                    algorithm, restart + 1))
        plt.close()

        plt.title('Train & Test Accuracy')
        plt.plot(final_train_acc_history, label='train_acc')
        plt.plot(final_test_acc_history, label='test_acc')
        plt.plot(final_ood_acc_history, label='ood_test_acc')
        plt.ylim(0, 1)
        plt.hlines(best_acc, 0, best_round, linestyles='dashed')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend(['Train Accuracy', 'Test Accuracy', 'OOD Test Accuracy'])
        plt.savefig('{}/acc-{}-restart {}.png'.format(log_dir,
                    algorithm, restart + 1))
        plt.close()

        self.logger.log("Best Loss: {}".format(best_loss))
        self.logger.log("Best Accuracy: {}".format(best_acc))
        self.logger.log("Best Round: {}".format(best_round))

    """
    ### Load Dataset
    """

    def __load_dataset(self):

        cifar = datasets.CIFAR10('~/datasets/cifar', train=True, download=True)

        cifar_train = (cifar.data[:40000], cifar.targets[:40000])
        cifar_val = (cifar.data[40000:], cifar.targets[40000:])

        rng_state = np.random.get_state()
        np.random.shuffle(cifar_train[0])
        np.random.set_state(rng_state)
        np.random.shuffle(cifar_train[1])

        self.logger.log((cifar_val[0]).shape)

        train_client_1_env_1 = self.data_loader.make_environment(
            cifar_train[0][:30000:9], cifar_train[1][:30000:9], from_angle=10, to_angle=10)
        train_client_1_env_2 = self.data_loader.make_environment(
            cifar_train[0][1:30001:9], cifar_train[1][1:30001:9], from_angle=25, to_angle=25)
        train_client_1_env_3 = self.data_loader.make_environment(
            cifar_train[0][2:30002:9], cifar_train[1][2:30002:9], from_angle=40, to_angle=40)

        train_client_2_env_1 = self.data_loader.make_environment(
            cifar_train[0][3:30003:9], cifar_train[1][3:30003:9], from_angle=60, to_angle=60)
        train_client_2_env_2 = self.data_loader.make_environment(
            cifar_train[0][4:30004:9], cifar_train[1][4:30004:9], from_angle=75, to_angle=75)
        train_client_2_env_3 = self.data_loader.make_environment(
            cifar_train[0][5:30005:9], cifar_train[1][5:30005:9], from_angle=90, to_angle=90)

        train_client_3_env_1 = self.data_loader.make_environment(
            cifar_train[0][6:30006:9], cifar_train[1][6:30006:9], from_angle=-10, to_angle=-10)
        train_client_3_env_2 = self.data_loader.make_environment(
            cifar_train[0][7:30007:9], cifar_train[1][7:30007:9], from_angle=-40, to_angle=-40)
        train_client_3_env_3 = self.data_loader.make_environment(
            cifar_train[0][8:30008:9], cifar_train[1][8:30008:9], from_angle=-90, to_angle=-90)

        train_envs = [
            # Client 1 Train
            self.data_loader.combine_envs([train_client_1_env_1,
                                           train_client_1_env_2, train_client_1_env_3]),
            # Client 2 Train
            self.data_loader.combine_envs([train_client_2_env_1,
                                           train_client_2_env_2, train_client_2_env_3]),
            # Client 3 Train
            self.data_loader.combine_envs([train_client_3_env_1,
                                          train_client_3_env_2, train_client_3_env_3])
        ]

        test_client_1_env_1 = self.data_loader.make_environment(
            cifar_train[0][30000::9], cifar_train[1][30000::9], from_angle=10, to_angle=10)
        test_client_1_env_2 = self.data_loader.make_environment(
            cifar_train[0][30001::9], cifar_train[1][30001::9], from_angle=25, to_angle=25)
        test_client_1_env_3 = self.data_loader.make_environment(
            cifar_train[0][30002::9], cifar_train[1][30002::9], from_angle=40, to_angle=40)

        test_client_2_env_1 = self.data_loader.make_environment(
            cifar_train[0][30003::9], cifar_train[1][30003::9], from_angle=60, to_angle=60)
        test_client_2_env_2 = self.data_loader.make_environment(
            cifar_train[0][30004::9], cifar_train[1][30004::9], from_angle=75, to_angle=75)
        test_client_2_env_3 = self.data_loader.make_environment(
            cifar_train[0][30005::9], cifar_train[1][30005::9], from_angle=90, to_angle=90)

        test_client_3_env_1 = self.data_loader.make_environment(
            cifar_train[0][30006::9], cifar_train[1][30006::9], from_angle=-10, to_angle=-10)
        test_client_3_env_2 = self.data_loader.make_environment(
            cifar_train[0][30007::9], cifar_train[1][30007::9], from_angle=-40, to_angle=-40)
        test_client_3_env_3 = self.data_loader.make_environment(
            cifar_train[0][30008::9], cifar_train[1][30008::9], from_angle=-90, to_angle=-90)

        test_envs = [
            # Client 1 Validation
            self.data_loader.combine_envs([test_client_1_env_1, test_client_1_env_2,
                                           test_client_1_env_3]),
            # Client 2 Validation
            self.data_loader.combine_envs([test_client_2_env_1,
                                           test_client_2_env_2, test_client_2_env_3]),
            # Client 3 Validation
            self.data_loader.combine_envs([test_client_3_env_1,
                                          test_client_3_env_2, test_client_3_env_3])
        ]

        ood_validation = self.data_loader.make_environment(
            cifar_val[0], cifar_val[1], from_angle=-90, to_angle=90)

        return train_envs, test_envs, ood_validation

    """
    ### Create federated clients
    """

    def __create_clients(self, train_envs, test_envs, train_batch_size, test_batch_size, learning_rate):

        # Create federated clients
        clients = []

        for client_id, (train_env, test_env) in enumerate(zip(train_envs, test_envs)):

            train_images, train_labels = train_env["images"], train_env["labels"]
            test_images, test_labels = test_env["images"], test_env["labels"]

            train_loader = self.data_loader.create_data_loader(
                train_images, train_labels, train_batch_size)
            test_loader = self.data_loader.create_data_loader(
                test_images, test_labels, test_batch_size)

            # Each client has one local model
            local_model = CifarCNN()
            client = FederatedClient(self.trainer, self.evaluator, client_id, local_model,
                                     train_loader, train_images, train_labels,
                                     test_loader, learning_rate, self.logger)
            clients.append(client)

        return clients
