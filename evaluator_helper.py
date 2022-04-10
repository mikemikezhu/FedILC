from abc import ABC, abstractmethod
from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import torch
from torch import nn

"""
Evaluator Helper Type
"""


class EvaluatorHelperType(Enum):
    BINARY = 0
    MULTIPLE = 1


"""
Abstract Evaluator Helper
"""


class AbstractEvaluatorHelper(ABC):

    @abstractmethod
    def mean_nll(self, logits, y):
        raise Exception("Abstract method should be implemented")

    @abstractmethod
    def mean_accuracy(self, logits, y):
        raise Exception("Abstract method should be implemented")

    @abstractmethod
    def mean_roc_auc(self, logits, y):
        raise Exception()

    @abstractmethod
    def mean_pr_auc(self, logits, y):
        raise Exception()


"""
Binary Classification
"""


class BinaryClassificationEvaluatorHelper(AbstractEvaluatorHelper):

    def mean_nll(self, logits, y):
        critetion = nn.BCELoss()
        return critetion(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.5).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def mean_roc_auc(self, logits, y):
        preds = (logits > 0.5).float()
        y = y.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        return roc_auc_score(y, preds)

    def mean_pr_auc(self, logits, y):
        preds = (logits > 0.5).float()
        y = y.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        precision, recall, _ = precision_recall_curve(y, preds)
        return auc(recall, precision)


"""
Multi Classification
"""


class MultiClassificationEvaluatorHelper(AbstractEvaluatorHelper):

    def mean_nll(self, logits, y):
        critetion = nn.CrossEntropyLoss()
        return critetion(logits, y)

    def mean_accuracy(self, logits, y):
        _, preds = torch.max(logits, 1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        return correct / total

    def mean_roc_auc(self, logits, y):
        return None

    def mean_pr_auc(self, logits, y):
        return None


"""
Evaluator Helper Factory
"""


class EvaluatorHelperFactory:

    __binary = None
    __multi = None

    @staticmethod
    def get_evaluator(type):

        if type == EvaluatorHelperType.BINARY:
            if EvaluatorHelperFactory.__binary is None:
                EvaluatorHelperFactory.__binary = BinaryClassificationEvaluatorHelper()
            return EvaluatorHelperFactory.__binary

        elif type == EvaluatorHelperType.MULTIPLE:
            if EvaluatorHelperFactory.__multi is None:
                EvaluatorHelperFactory.__multi = MultiClassificationEvaluatorHelper()
            return EvaluatorHelperFactory.__multi

        else:
            raise Exception(
                "Unsupported evaluator helper type: {}".format(type))
