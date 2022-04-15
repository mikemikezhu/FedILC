from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


"""
Data Loader Type
"""


class DataLoaderType(Enum):
    COLOR_MNIST = 0
    ROTATE_CIFAR = 1
    ICU = 2


"""
Abstract Data Loader
"""


class AbstractDataLoader(ABC):

    @abstractmethod
    def combine_envs(self, envs):
        raise Exception("Abstract method should be implemented")

    @abstractmethod
    def make_environment(self, images, labels, **kwargs):
        raise Exception("Abstract method should be implemented")

    def create_data_loader(self, x, y, batch_size):

        data_set = self.__convert_to_tensor(x, y)
        data_loader = DataLoader(data_set,
                                 shuffle=True,
                                 batch_size=batch_size)
        return data_loader

    def __convert_to_tensor(self, x, y):
        assert x.shape[0] == y.shape[0]

        tensor_list = []
        for idx in range(x.shape[0]):
            data_x, data_y = x[idx], y[idx]
            tensor_list.append((data_x, data_y))

        return tensor_list


"""
Color MNIST
"""


class ColorMNISTDataLoader(AbstractDataLoader):

    def combine_envs(self, envs):
        raise Exception("Method is not supported!")

    def make_environment(self, images, labels, **kwargs):

        label_flipping_prob = kwargs.get("label_flipping_prob")
        if label_flipping_prob is None:
            raise Exception("Need label flipping probability!")

        color_flipping_prob = kwargs.get("color_flipping_prob")
        if color_flipping_prob is None:
            raise Exception("Need color flipping probability!")

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(
            label_flipping_prob, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(
            color_flipping_prob, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0

        images, labels = images.float() / 255., labels[:, None]

        if torch.cuda.is_available():
            return {'images': images.cuda(), 'labels': labels.cuda()}

        return {'images': images, 'labels': labels}

    def create_data_loader(self, x, y, batch_size):
        return super().create_data_loader(x, y, batch_size)


"""
Rotated CIFAR-10
"""


class RotatedCifarDataLoader(AbstractDataLoader):

    def combine_envs(self, envs):

        images, labels = [], []
        for env in envs:
            image, label = env["images"], env["labels"]
            images.append(image)
            labels.append(label)

        images = torch.cat(tuple(images))
        labels = torch.cat(tuple(labels))

        return {'images': images, 'labels': labels}

    def make_environment(self, images, labels, **kwargs):

        from_angle = kwargs.get('from_angle')
        to_angle = kwargs.get('to_angle')

        if from_angle is None:
            raise Exception("Need from angle!")

        if to_angle is None:
            raise Exception("Need to angle!")

        rotation = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomRotation(
                                           degrees=(from_angle, to_angle)),
                                       transforms.ToTensor()])

        images = images[:, ::2, ::2, :]
        x = torch.zeros(len(images), 3, 16, 16)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        images = x

        images, labels = torch.Tensor(images), torch.Tensor(labels)
        labels = labels.type(torch.int64)

        if torch.cuda.is_available():
            return {'images': images.cuda(), 'labels': labels.cuda()}

        return {'images': images, 'labels': labels}

    def create_data_loader(self, x, y, batch_size):
        return super().create_data_loader(x, y, batch_size)


"""
ICU
"""


class IcuDataLoader(AbstractDataLoader):

    def combine_envs(self, envs):
        raise Exception("Method is not supported!")

    def make_environment(self, images, labels, **kwargs):
        raise Exception("Method is not supported!")

    def create_data_loader(self, x, y, batch_size):
        return super().create_data_loader(x, y, batch_size)


"""
Data Loader Factory
"""


class DataLoaderFactory:

    __color_mnist = None
    __rotate_cifar = None
    __icu = None

    @staticmethod
    def get_data_loader(type):

        if type == DataLoaderType.COLOR_MNIST:
            if DataLoaderFactory.__color_mnist is None:
                DataLoaderFactory.__color_mnist = ColorMNISTDataLoader()
            return DataLoaderFactory.__color_mnist

        elif type == DataLoaderType.ROTATE_CIFAR:
            if DataLoaderFactory.__rotate_cifar is None:
                DataLoaderFactory.__rotate_cifar = RotatedCifarDataLoader()
            return DataLoaderFactory.__rotate_cifar

        elif type == DataLoaderType.ICU:
            if DataLoaderFactory.__icu is None:
                DataLoaderFactory.__icu = IcuDataLoader()
            return DataLoaderFactory.__icu

        else:
            raise Exception("Unsupported data loader type: {}".format(type))
