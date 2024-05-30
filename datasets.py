import numpy as np
import scipy.io as sio

import keras
from keras.datasets import mnist

from sklearn.datasets import make_moons

import torch
from torchvision import datasets, transforms

import time ### temporary


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    x_all = x_all.reshape(x_all.shape[0], 784).astype(float) / 255.0
    return x_all, y_all


def get_yale_data(root="CroppedYale", input_size = [192,168]):
    height = input_size[0]
    width = input_size[1]
    faces = datasets.ImageFolder(
        root=root,
        transform=transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  # flatten image
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(faces, batch_size=len(faces))
    images = next(iter(loader))[0].numpy()
    labels = np.array(faces.targets)
    return images, labels


def get_hyperspectral_data(path):
    mat_contents = sio.loadmat(path)
    name = [
        x
        for x in mat_contents.keys()
        if x not in ["__header__", "__version__", "__globals__"]
    ][0]
    data = mat_contents[name]
    # flatten image
    data = data.reshape(data.shape[0] * data.shape[1], *data.shape[2:])
    return data

###
def get_local_data(root, input_size):
    height = input_size[0]
    width = input_size[1]
    start_time = time.time() ### temporary
    faces = datasets.ImageFolder(
        root=root,
        transform=transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  # flatten image
            ]
        ),
    )
    print(time.time()-start_time)
    loader = torch.utils.data.DataLoader(faces, batch_size=len(faces))
    print(time.time() - start_time)
    images = next(iter(loader))[0].numpy()
    print(time.time() - start_time)
    labels = np.array(faces.targets)
    print(time.time() - start_time)
    return images, labels
###

# example usage:

# from datasets import get_mnist_data, get_yale_data, get_hyperspectral_data
# x, y = get_mnist_data()
# x, y = get_yale_data()
# x = get_hyperspectral_data()


def prune(data, labels, wanted):
    n = len(data)
    idx = [i for i in range(n) if labels[i] in wanted]
    data = data[idx]
    labels = labels[idx]
    lookup = np.arange(int(labels.max() + 1))
    lookup[wanted] = np.arange(len(wanted))
    labels = lookup[labels]
    return data, labels


def get_data(dataset, path, num_of_people = None, input_size = None, quantity=10000):
    if dataset == "moons":
        data, labels = make_moons(quantity)
        data = torch.tensor(data).float()
        data += 0.1 * torch.randn(data.shape)
        labels = torch.tensor(labels)
        return data, labels, 2
    elif dataset == "mnist" or dataset == "mnist5":
        data, labels = get_mnist_data()
        data, labels = prune(data, labels, [0, 3, 4, 6, 7])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        idx = np.random.permutation(np.shape(data)[0])[:quantity]
        data = data[idx,:]
        labels = labels[idx]
        return torch.tensor(data).float(), torch.tensor(labels), 5
    elif dataset == "mnist10":
        data, labels = get_mnist_data()
        data, labels = prune(data, labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        idx = np.random.permutation(np.shape(data)[0])[:quantity]
        data = data[idx,:]
        labels = labels[idx]
        return torch.tensor(data).float(), torch.tensor(labels), 10
    elif dataset == "salinas":
        data = get_hyperspectral_data(
            f"{path}/SalinasA_smallNoise.mat"
        )
        labels = get_hyperspectral_data(
            f"{path}/SalinasA_gt.mat"
        )
        data, labels = prune(data, labels, [0, 1, 10, 11, 12, 13, 14])
        data = torch.tensor(data).float()
        labels = torch.tensor(labels)
        data = data.reshape(86, 83, -1).permute(1, 0, 2).reshape(83 * 86, -1)
        data -= data.min()
        data /= data.max()
        data, labels = data[np.where(labels)], labels[np.where(labels)] - 1
        return data, labels, 6
    elif dataset == "yale2":
        data, labels = get_yale_data(input_size = input_size) if path is None else get_yale_data(path,input_size = input_size)
        data, labels = prune(data, labels, [4, 8])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 2
    elif dataset == "yale3":
        data, labels = get_yale_data(input_size = input_size) if path is None else get_yale_data(path,input_size = input_size)
        data, labels = prune(data, labels, [4, 8, 20])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 3
    elif dataset == "yale10":
        data, labels = get_yale_data(input_size = input_size) if path is None else get_yale_data(path,input_size = input_size)
        data, labels = prune(data, labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 10
    elif dataset == "yale_full":
        data, labels = get_yale_data(input_size = input_size) if path is None else get_yale_data(path,input_size = input_size)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 38
    elif dataset == "yale_var":
        data, labels = get_yale_data(input_size = input_size) if path is None else get_yale_data(path,input_size = input_size)
        idx = np.random.permutation(38)[0:num_of_people]
        data, labels = prune(data, labels, idx)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), num_of_people
    elif dataset == "leaf_disease_apple": ###
        data, labels = get_local_data(path, input_size = input_size) ### # TODO: this is too slow. Try to combine prune with this.
        data, labels = prune(data, labels, [0, 1, 2, 3])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 4
    else:
        print(f"unknown dataset '{dataset}")
