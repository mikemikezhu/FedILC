from abstract_executor import AbstractExecutor
from client import FederatedClient
from fed_logger import FedLogger
from trainer import Trainer
from evaluator import Evaluator
from evaluator_helper import *
from data_loader import *
from model import *
from helper import *

from sklearn.metrics import roc_curve
from torchvision import datasets
import numpy as np
import os

import matplotlib.pyplot as plt


class ColorMNISTExecutor(AbstractExecutor):

    COLOR_MNIST_DATASET = "color_mnist"

    """Initialize"""

    def __init__(self):
        self.data_loader = DataLoaderFactory.get_data_loader(
            DataLoaderType.COLOR_MNIST)
        self.evaluator_helper = EvaluatorHelperFactory.get_evaluator(
            EvaluatorHelperType.BINARY)
        self.trainer = Trainer(self.evaluator_helper)
        self.evaluator = Evaluator(self.evaluator_helper)

    def is_eligible_executor(self, dataset):
        return dataset == self.COLOR_MNIST_DATASET

    def run(self, restart, flags):

        algorithm = flags.algorithm

        log_dir = "mnist-{}-restart {}".format(algorithm, restart + 1)
        os.mkdir(log_dir)

        self.logger = FedLogger.getLogger(restart + 1,
                                          "{}/mnist-{}-restart {}".format(log_dir, algorithm, restart + 1))
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

        global_model = MnistMLP(390)
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
        final_ood_roc_history = []

        best_model = None
        best_round = 0
        best_loss = float("inf")
        best_acc = 0
        best_pr_auc = 0
        best_roc_auc = 0

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
                    global_model, global_optimizer, round_idx, nn.BCEWithLogitsLoss(reduction='sum'), flags)

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
            final_train_acc_np = final_train_acc.detach().cpu().numpy().copy()
            final_test_loss_np = final_test_loss.detach().cpu().numpy().copy()
            final_test_acc_np = final_test_acc.detach().cpu().numpy().copy()

            final_train_loss_history.append(final_train_loss_np)
            final_train_acc_history.append(final_train_acc_np)
            final_test_loss_history.append(final_test_loss_np)
            final_test_acc_history.append(final_test_acc_np)

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
            ood_test_loss, ood_test_acc, ood_test_roc, odd_test_pr = ood_test_history

            ood_test_loss_np = ood_test_loss.detach().cpu().numpy().copy()
            ood_test_acc_np = ood_test_acc.detach().cpu().numpy().copy()

            final_ood_loss_history.append(ood_test_loss_np)
            final_ood_acc_history.append(ood_test_acc_np)
            final_ood_roc_history.append(ood_test_roc)

            if ood_test_loss < best_loss and round_idx > 5:
                best_loss = ood_test_loss
                best_acc = ood_test_acc
                best_roc_auc = ood_test_roc
                best_pr_auc = odd_test_pr
                best_model = global_model
                best_round = round_idx

            self.logger.log('\n')
            self.logger.log('########################################')
            self.logger.log('End training round: {}'.format(round_idx + 1))
            self.logger.log('[Train] Loss: {}, Accuracy: {}'.format(
                final_train_loss, final_train_acc))
            self.logger.log('[Test] Loss: {}, Accuracy: {}'.format(
                final_test_loss, final_test_acc))
            self.logger.log('[OOD Test] Loss: {}, Accuracy: {}, ROC AUC: {}, PR AUC: {}'.format(
                ood_test_loss, ood_test_acc, ood_test_roc, odd_test_pr))
            self.logger.log('########################################')
            self.logger.log('\n')

            if round_idx % 50 == 0 and round_idx > 5:
                self.logger.log(learning_rate)
                path = '{}/mnist-{}-restart-{}-output_checkpoint{}'.format(
                    log_dir, algorithm, restart + 1, str(round_idx))
                # self.logger.log(global_model.state_dict())
                torch.save({'global_model': global_model.state_dict(),
                            'best_model': best_model.state_dict(),
                            'best_round': best_round,
                            'best_loss': best_loss,
                            'best_roc_auc': best_roc_auc,
                            'best_pr_auc': best_pr_auc,
                            'best_acc': best_acc,
                            'global_optimizer': global_optimizer.state_dict(),
                            'final_train_loss_history': final_train_loss_history,
                            'final_train_acc_history': final_train_acc_history,
                            'final_test_loss_history': final_test_loss_history,
                            'final_test_acc_history': final_test_acc_history,
                            'final_ood_loss_history': final_ood_loss_history,
                            'final_ood_acc_history': final_ood_acc_history}, path)

        best_loss = best_loss.cpu().numpy().copy()
        best_acc = best_acc.cpu().numpy().copy()

        plt.title('Train & Test Loss')
        plt.plot(final_train_loss_history, label='train_loss')
        plt.plot(final_test_loss_history, label='test_loss')
        plt.plot(final_ood_loss_history, label='ood_test_loss')
        plt.hlines(best_loss, 0, best_round, linestyles='dashed')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.ylim(0.3, 1.0)
        plt.legend(['Train Loss', 'Test Loss', 'OOD Test Loss'])
        plt.savefig('{}/loss-{}-restart {}.png'.format(log_dir,
                    algorithm, restart + 1))
        plt.close()

        plt.title('Train & Test Accuracy')
        plt.plot(final_train_acc_history, label='train_acc')
        plt.plot(final_test_acc_history, label='test_acc')
        plt.plot(final_ood_acc_history, label='ood_test_acc')
        plt.hlines(best_acc, 0, best_round, linestyles='dashed')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.ylim(0.3, 1.0)
        plt.legend(['Train Accuracy', 'Test Accuracy', 'OOD Test Accuracy'])
        plt.savefig('{}/acc-{}-restart {}.png'.format(log_dir,
                    algorithm, restart + 1))
        plt.close()

        with torch.no_grad():

            # Set mode to evaluate model
            global_model.eval()

            x_test, y_test = ood_validation["images"], ood_validation["labels"]

            # total_x_test = torch.from_numpy(x_test)
            # total_y_test = torch.from_numpy(y_test)

            # total_x_test = total_x_test.to('cuda')

            # Predict model
            _, logits = global_model(x_test)
            # sigmoid = nn.Sigmoid()
            # predict_test = sigmoid(logits)
            # total_predict_test = (logits > 0.).float()
            total_predict_test = logits.cpu()

            predict_one_test = total_predict_test.detach().numpy().copy()
            predict_one_test[total_predict_test <= 0.5] = np.nan
            predict_one_test_mean = np.nanmean(predict_one_test, axis=0)

            predict_zero_test = total_predict_test.detach().numpy().copy()
            predict_zero_test[total_predict_test > 0.5] = np.nan
            predict_zero_test_mean = np.nanmean(predict_zero_test, axis=0)

            self.logger.log(predict_one_test_mean)
            self.logger.log(predict_zero_test_mean)

            self.logger.log(np.count_nonzero(~np.isnan(predict_one_test)))
            self.logger.log(np.count_nonzero(~np.isnan(predict_zero_test)))

            # Generate a no skill prediction
            total_no_skill = [0 for _ in range(len(y_test))]

            self.logger.log(len(y_test))

            """ Plot ROC Curve """
            # Calculate roc curves
            y_test = y_test.cpu().detach().numpy().copy()
            ns_fpr, ns_tpr, _ = roc_curve(y_test, total_no_skill)
            lr_fpr, lr_tpr, _ = roc_curve(y_test, total_predict_test)

            # Calculate scores
            ns_auc = roc_auc_score(y_test, total_no_skill)
            lr_auc = roc_auc_score(y_test, total_predict_test)
            self.logger.log('No skill ROC score: {}'.format(ns_auc))
            self.logger.log('Proposed model ROC score: {}'.format(lr_auc))

            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Proposed Model')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('{}/roc-{}-restart {}.png'.format(log_dir,
                                                          algorithm, restart + 1))
            plt.close()

        self.logger.log("Best Loss: {}".format(best_loss))
        self.logger.log("Best Accuracy: {}".format(best_acc))
        self.logger.log("Best Round: {}".format(best_round))
        self.logger.log('Best ROC AUC: {}'.format(best_roc_auc))
        self.logger.log('Best PR AUC: {}'.format(best_pr_auc))

    """
    ### Load Dataset
    """

    def __load_dataset(self):

        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        self.logger.log((mnist_val[0]).shape)

        train_envs = [
            # Client 1 Train
            self.data_loader.make_environment(
                mnist_train[0][:40000:5], mnist_train[1][:40000:5], color_flipping_prob=0.15, label_flipping_prob=0.0),
            # Client 2 Train
            self.data_loader.make_environment(
                mnist_train[0][1:40001:5], mnist_train[1][1:40001:5], color_flipping_prob=0.3, label_flipping_prob=0.0),
            # Client 3 Train
            self.data_loader.make_environment(
                mnist_train[0][2:40002:5], mnist_train[1][2:40002:5], color_flipping_prob=0.45, label_flipping_prob=0.0),
            # Client 4 Train
            self.data_loader.make_environment(
                mnist_train[0][3:40003:5], mnist_train[1][3:40003:5], color_flipping_prob=0.6, label_flipping_prob=0.0),
            # Client 5 Train
            self.data_loader.make_environment(
                mnist_train[0][4:40004:5], mnist_train[1][4:40004:5], color_flipping_prob=0.75, label_flipping_prob=0.0)
        ]

        test_envs = [
            # Client 1 Validation
            self.data_loader.make_environment(
                mnist_train[0][40000::5], mnist_train[1][40000::5], color_flipping_prob=0.15, label_flipping_prob=0.0),
            # Client 2 Validation
            self.data_loader.make_environment(
                mnist_train[0][40001::5], mnist_train[1][40001::5], color_flipping_prob=0.3, label_flipping_prob=0.0),
            # Client 3 Validation
            self.data_loader.make_environment(
                mnist_train[0][40002::5], mnist_train[1][40002::5], color_flipping_prob=0.45, label_flipping_prob=0.0),
            # Client 4 Validation
            self.data_loader.make_environment(
                mnist_train[0][40003::5], mnist_train[1][40003::5], color_flipping_prob=0.6, label_flipping_prob=0.0),
            # Client 5 Validation
            self.data_loader.make_environment(
                mnist_train[0][40004::5], mnist_train[1][40004::5], color_flipping_prob=0.75, label_flipping_prob=0.0)
        ]

        ood_validation = self.data_loader.make_environment(
            mnist_val[0], mnist_val[1], color_flipping_prob=0.9, label_flipping_prob=0.15)

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
            local_model = MnistMLP(390)
            client = FederatedClient(self.trainer, self.evaluator, client_id, local_model,
                                     train_loader, train_images, train_labels,
                                     test_loader, learning_rate, self.logger)
            clients.append(client)

        return clients
