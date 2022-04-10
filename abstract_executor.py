from abc import ABC, abstractmethod


"""
Executor Interface
"""


class AbstractExecutor(ABC):

    @abstractmethod
    def is_eligible_executor(self, dataset):
        raise Exception("Abstract method should be implemented")

    @abstractmethod
    def run(self, restart, flags):
        raise Exception("Abstract method should be implemented")
