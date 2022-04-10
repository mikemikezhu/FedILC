from helper import *


class Trainer:

    def __init__(self, evaluator_helper):
        self.__evaluator_helper = evaluator_helper

    def set_logger(self, logger):
        self.__logger = logger

    def train_model(self, model, optimizer, local_model, local_optimizer, train_loader, train_images, train_labels, round_idx, flags):

        # t = torch.cuda.get_device_properties(0).total_memory
        # a = torch.cuda.memory_allocated(0)

        # logger.info("Memory before calculating gradients:")
        # logger.info(convert_size(t))
        # logger.info(convert_size(a))

        algorithm = flags.algorithm

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
                param_gradients = get_model_grads(
                    images, labels, model, self.__evaluator_helper.mean_nll)

                _, logits = model(images)
                loss = self.__evaluator_helper.mean_nll(logits, labels)
                acc = self.__evaluator_helper.mean_accuracy(logits, labels)

                final_loss += loss
                final_acc += acc

                total_param_gradients.append(param_gradients)

            # self.__logger.log(len(total_param_gradients))

            # ILC
            local_model_params = model.state_dict()
            local_model_params = copy.deepcopy(local_model_params)
            local_model.load_state_dict(local_model_params)

            # TODO
            local_model.train()
            local_optimizer.zero_grad()
            if "geo" in algorithm.split("_"):
                compute_geo_mean(list(local_model.parameters()),
                                 total_param_gradients, "geo_weighted", 0.001)
            elif "arith" in algorithm.split("_"):
                compute_arith_mean(
                    list(local_model.parameters()), total_param_gradients)
            local_optimizer.step()

            # Fishr
            features, _ = local_model(train_images)
            grad_statistics = compute_grad_variance(
                features, train_labels, local_model.classifier, algorithm)

            # Calculate loss and accuracy
            train_loss = final_loss / len(train_loader)
            train_acc = final_acc / len(train_loader)

        else:

            # Set mode to train model
            model.train()

            # Start training
            features, logits = model(train_images)

            self.__logger.log(logits.shape)
            self.__logger.log(train_labels.shape)

            train_loss = self.__evaluator_helper.mean_nll(logits, train_labels)
            train_acc = self.__evaluator_helper.mean_accuracy(
                logits, train_labels)

            optimizer.zero_grad()

            if "arith" in algorithm.split("_") or "geo" in algorithm.split("_") or "hybrid" in algorithm.split("_"):
                model_grads = get_model_grads(
                    train_images, train_labels, model, self.__evaluator_helper.mean_nll)

            if "fishr" in algorithm.split("_"):
                grad_variance = compute_grad_variance(
                    features, train_labels, model.classifier, algorithm)

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
