from abstract_executor import AbstractExecutor


class ColorMNISTExecutor(AbstractExecutor):

    COLOR_MNIST_DATASET = "color_mnist"

    def is_eligible_executor(self, dataset):
        return dataset == self.COLOR_MNIST_DATASET

    def run(self, restart, flags):

        algorithm = flags.algorithm

        total_feature = flags.total_feature
        learning_rate = flags.learning_rate
        weight_decay = flags.weight_decay

        # learning_rate_decay_step_size = 100
        # learning_rate_decay = 0.98

        train_batch_size = flags.train_batch_size
        test_batch_size = flags.test_batch_size

        num_steps = flags.num_steps
        num_rounds = flags.num_rounds
        num_epochs = flags.num_epochs

        hidden_dim = flags.hidden_dim

        penalty_anneal_iters = flags.penalty_anneal_iters
        penalty_weight_factor = flags.penalty_weight_factor
        penalty_weight = flags.penalty_weight

        return super().run(flags)
