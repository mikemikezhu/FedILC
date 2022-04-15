from helper import *


class Trainer:

    def __init__(self, evaluator_helper):
        self.__evaluator_helper = evaluator_helper

    def set_logger(self, logger):
        self.__logger = logger

    def train_model(self, model, optimizer, local_model, local_optimizer, train_loader, train_images, train_labels, round_idx, loss_fn, flags):

        algorithm = flags.algorithm

        model_grads = []

        model.train()

        if "fishr" in algorithm.split("_") and ("geo" in algorithm.split("_") or "arith" in algorithm.split("_")):

            """ Fishr + Intra Geo Mean / Arith Mean """
            # Geometric mean / Arithmetic mean
            total_param_gradients = []
            for (images, labels) in train_loader:
                optimizer.zero_grad()
                param_gradients = get_model_grads(
                    images, labels, model, self.__evaluator_helper.mean_nll)
                total_param_gradients.append(param_gradients)

            optimizer.zero_grad()
            if "geo" in algorithm.split("_"):
                compute_geo_mean(list(model.parameters()),
                                 total_param_gradients, "geo_weighted", 0.001)
                self.__logger.log("Calculate intra-silo geometric mean")
            elif "arith" in algorithm.split("_"):
                compute_arith_mean(
                    list(model.parameters()), total_param_gradients)
                self.__logger.log("Calculate intra-silo arithmetic mean")

            # Note: No need to update local model
            # local_optimizer.step()
            for model_param in list(model.parameters()):
                grad = model_param.grad
                grad_copy = copy.deepcopy(grad.detach())
                model_grads.append(grad_copy)

        else:
            # Model Gradients
            optimizer.zero_grad()
            model_grads = get_model_grads(
                train_images, train_labels, model, self.__evaluator_helper.mean_nll)

        # Fishr
        features, logits = model(train_images)
        grad_variance = compute_grad_variance(
            features, train_labels, model.classifier, loss_fn)

        # Calculate loss and accuracy
        train_loss = self.__evaluator_helper.mean_nll(logits, train_labels)
        train_acc = self.__evaluator_helper.mean_accuracy(logits, train_labels)

        return train_loss, train_acc, grad_variance, model_grads
