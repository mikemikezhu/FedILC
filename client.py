import torch


class FederatedClient:

    # Init client
    def __init__(self, trainer, evaluator,  client_id, local_model,
                 train_loader, train_images, train_labels,
                 test_loader, learning_rate, logger):

        self.trainer = trainer
        self.evaluator = evaluator

        self.client_id = client_id

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.train_images = train_images
        self.train_labels = train_labels

        self.local_model = local_model
        self.logger = logger

        if torch.cuda.is_available():
            self.local_model = self.local_model.to('cuda')

        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(),
                                                lr=learning_rate)

    # Train model
    def train(self, global_model, global_optimizer, round_idx, loss_fn, flags):

        # Start training
        train_loss, train_acc, grad_variance, model_grads = self.trainer.train_model(global_model,
                                                                                     global_optimizer,
                                                                                     self.local_model,
                                                                                     self.local_optimizer,
                                                                                     self.train_loader,
                                                                                     self.train_images,
                                                                                     self.train_labels,
                                                                                     round_idx,
                                                                                     loss_fn,
                                                                                     flags)
        train_history = (train_loss, train_acc)
        self.logger.log('Client[{}], Round [{}], Loss: [{}], Accuracy: [{}]'.format(
            self.client_id, round_idx + 1, train_loss, train_acc))

        # Evaluation
        test_batch_size = flags.test_batch_size
        test_history = self.evaluator.evaluate_model(global_model,
                                                     self.test_loader,
                                                     test_batch_size)

        return train_history, test_history, grad_variance, model_grads
