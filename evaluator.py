import torch


class Evaluator:

    def __init__(self, helper):
        self.__helper = helper

    def evaluate_model(self, model, test_loader, test_batch_size):

        with torch.no_grad():

            # Set mode to evaluate model
            model.eval()

            # Start evaluating model
            final_loss = 0
            final_acc = 0
            final_roc = []
            final_pr = []

            for (images, labels) in test_loader:

                _, logits = model(images)

                loss = self.__helper.mean_nll(logits, labels)
                acc = self.__helper.mean_accuracy(logits, labels)

                final_loss += loss
                final_acc += acc

                if len(labels) == test_batch_size:
                    roc = self.__helper.mean_roc_auc(logits, labels)
                    pr = self.__helper.mean_pr_auc(logits, labels)
                    if roc is not None:
                        final_roc.append(roc)
                    if pr is not None:
                        final_pr.append(pr)

            test_loss = final_loss / len(test_loader)
            test_acc = final_acc / len(test_loader)

            test_roc = None
            if len(final_roc) > 0:
                test_roc = sum(final_roc) / len(final_roc)

            test_pr = None
            if len(final_pr) > 0:
                test_pr = sum(final_pr) / len(final_pr)

            return test_loss, test_acc, test_roc, test_pr
