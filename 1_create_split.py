import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from omegaconf import DictConfig
import hydra
from copy import deepcopy
import pandas as pd
from torch.utils.data import Dataset
from data_loader.data_loader import TabularDataset
from data_loader.medical_mnist import MedicalMNIST
from data_loader.pneumonia import Pneumonia
from data_loader.retinal import Retinal
from data_loader.skin import Skin

# create dataset only creates the split based on the whole dataset;
import math
from data_loader.kidney import Kidney

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def get_iidk_index(all_index, num_parties, train_ratio, pop_ratio):
    """
    This function is used for generate the splits of the dataset;
    For each collaboator, we have a list of splits. For each split, we have the train data, test data and the rest of the dataset.
    There is no overlapping between the datasets from collaboators.
    """
    test_ratio = train_ratio
    np.random.shuffle(all_index)
    index_per_party = [all_index[idx::num_parties] for idx in range(num_parties)]
    list_dict = {}

    # train_num_per_client = min(train_num, (len(all_index)- pop_num*nu

    for party in range(num_parties):
        party_index = index_per_party[party]
        num_total = len(party_index)
        splits = {}
        selected_index = np.random.choice(
            party_index, int((train_ratio + test_ratio) * num_total), replace=False
        )
        splits["train"] = selected_index[: int(train_ratio * num_total)]
        splits["test"] = selected_index[
            int(train_ratio * num_total) : int((train_ratio + test_ratio) * num_total)
        ]
        splits["rest"] = np.array([i for i in party_index if i not in selected_index])
        list_dict[party] = splits
    return list_dict


def get_iid_index(all_index, train_num, test_num, pop_num):
    """
    This function is used for generate the splits of the dataset;
    For each collaboator, we have a list of splits. For each split, we have the train data, test data and the rest of the dataset.
    There is no overlapping between the datasets from collaboators.
    """
    np.random.shuffle(all_index)
    num_sample_per_client = train_num + test_num + pop_num
    max_num_parties = math.floor(len(all_index) / num_sample_per_client)

    index_per_party = [
        all_index[idx::max_num_parties] for idx in range(max_num_parties)
    ]
    list_dict = {}
    for party in range(max_num_parties):
        party_index = index_per_party[party]
        splits = {}
        selected_index = np.random.choice(
            party_index, train_num + test_num, replace=False
        )
        splits["train"] = selected_index[:train_num]
        splits["test"] = selected_index[train_num : train_num + test_num]
        splits["rest"] = np.array([i for i in party_index if i not in selected_index])
        list_dict[party] = splits
    return list_dict


def get_richlet_index(all_index, Y, train_num, test_num, pop_num, dirichlet_alpha):
    Y = np.array(Y)
    max_num_parties = math.floor(len(all_index) / (train_num + test_num + pop_num))
    print(all_index.shape, max_num_parties)

    idx_batch = [[] for _ in range(max_num_parties)]
    num_total_samples = len(all_index)
    num_classes = len(np.unique(Y))

    all_index_by_party = {}
    for party in range(max_num_parties):
        all_index_by_party[party] = []
    for k in range(num_classes):
        idx_k = np.where(Y == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, max_num_parties))
        proportions = np.array(
            [
                p * (len(idx_j) < num_total_samples / max_num_parties)
                for p, idx_j in zip(proportions, idx_batch)
            ]
        )
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        index_per_party = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
        ]
        for party in range(max_num_parties):
            print(k, party, len(index_per_party[party]))
            all_index_by_party[party].extend(index_per_party[party])

    list_dict = {}
    a = 0
    for party in range(max_num_parties):
        splits = {}
        party_index = all_index_by_party[party]
        selected_index = np.random.choice(
            party_index, train_num + test_num, replace=False
        )
        splits["train"] = selected_index[:train_num]
        splits["test"] = selected_index[train_num : train_num + test_num]
        splits["rest"] = np.array([i for i in party_index if i not in selected_index])
        print(splits["train"].shape, splits["test"].shape, splits["rest"].shape)
        list_dict[party] = splits
        a += splits["train"].shape[0]
    print(a)
    return list_dict


def get_richlet_indexk(
    all_index, Y, num_parties, train_ratio, test_ratio, dirichlet_alpha
):
    Y = np.array(Y)
    print(all_index.shape, num_parties)
    idx_batch = [[] for _ in range(num_parties)]
    N_total_samples = len(all_index)
    num_classes = len(np.unique(Y))

    all_index_by_party = {}
    for party in range(num_parties):
        all_index_by_party[party] = []
    for k in range(num_classes):
        idx_k = np.where(Y == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_parties))
        proportions = np.array(
            [
                p * (len(idx_j) < N_total_samples / num_parties)
                for p, idx_j in zip(proportions, idx_batch)
            ]
        )
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        index_per_party = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
        ]
        for party in range(num_parties):
            print(k, party, len(index_per_party[party]))
            all_index_by_party[party].extend(index_per_party[party])

    list_dict = {}
    a = 0
    for party in range(num_parties):
        splits = {}
        party_index = all_index_by_party[party]
        num_total = len(party_index)
        selected_index = np.random.choice(
            party_index, int((train_ratio + test_ratio) * num_total), replace=False
        )
        splits["train"] = selected_index[: int(train_ratio * num_total)]
        splits["test"] = selected_index[
            int(train_ratio * num_total) : int((train_ratio + test_ratio) * num_total)
        ]
        splits["rest"] = np.array([i for i in party_index if i not in selected_index])
        print(splits["train"].shape, splits["test"].shape, splits["rest"].shape)
        # print(splits["train"][:10])
        list_dict[party] = splits
        a += splits["train"].shape[0]
    print(a)
    return list_dict


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    """
    Main function to create and save dataset splits for federated learning.

    This function:
    1. Sets random seeds for reproducibility
    2. Loads the specified dataset (CIFAR10, CIFAR100, etc)
    3. Creates train/test/rest splits based on the partitioning strategy:
       - IID-K: Independent and identically distributed splits across K parties
       - Dirichlet-K: Non-IID splits using Dirichlet distribution
    4. Saves the splits to a pickle file

    Args:
        args (DictConfig): Hydra config containing:
            - random_seed: For reproducibility
            - data: Dataset parameters (dataset, train_ratio, etc)
            - fl: Federated learning parameters (num_parties, partitioning, etc)
            - server_dir_path: Path to save splits
    """
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.data.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])

        cifar_train = datasets.CIFAR10(
            root=f"{args.server_dir_path}/data",
            train=True,
            download=True,
            transform=transform,
        )

        cifar_test = datasets.CIFAR10(
            root=f"{args.server_dir_path}/data",
            train=False,
            download=True,
            transform=transform,
        )

        num_total_samples = len(cifar_test) + len(cifar_train)
        total_index = np.arange(num_total_samples)
        X = np.concatenate([cifar_test.data, cifar_train.data])
        Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()
        all_data = deepcopy(cifar_train)
        all_data.data = X
        all_data.targets = Y

    elif args.data.dataset == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        cifar_train = datasets.CIFAR100(
            root=f"{args.server_dir_path}/data",
            train=True,
            download=True,
            transform=transform,
        )

        cifar_test = datasets.CIFAR100(
            root=f"{args.server_dir_path}/data",
            train=False,
            download=True,
            transform=transform,
        )

        num_total_samples = len(cifar_test) + len(cifar_train)
        total_index = np.arange(num_total_samples)
        X = np.concatenate([cifar_test.data, cifar_train.data])
        Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()
        all_data = deepcopy(cifar_train)
        all_data.data = X
        all_data.targets = Y

    elif args.data.dataset == "purchase":
        if os.path.exists("../data/dataset_purchase"):
            df = pd.read_csv(
                "../data/dataset_purchase", header=None, encoding="utf-8"
            ).to_numpy()
            Y = df[:, 0] - 1
            X = df[:, 1:].astype(np.float32)
            all_data = TabularDataset(X, Y)
            total_index = np.arange(len(Y))
        else:
            raise NotImplementedError(
                f"{args.data.dataset} is not installed correctly in ../data/purchase"
            )

    elif args.data.dataset == "texas":
        if os.path.exists("../data/texas/100/feats"):
            X = (
                pd.read_csv("../data/texas/100/feats", header=None, encoding="utf-8")
                .to_numpy()
                .astype(np.float32)
            )
            Y = (
                pd.read_csv("../data/texas/100/labels", header=None, encoding="utf-8")
                .to_numpy()
                .reshape(-1)
                - 1
            )

            all_data = TabularDataset(X, Y)
            total_index = np.arange(len(Y))
        else:
            raise NotImplementedError(
                f"{args.data.dataset} is not installed correctly in ../data/texas"
            )
    elif args.data.dataset == "texas_normalized":
        if os.path.exists("../data/texas/100/feats"):
            X = (
                pd.read_csv("../data/texas/100/feats", header=None, encoding="utf-8")
                .to_numpy()
                .astype(np.float32)
            )
            Y = (
                pd.read_csv("../data/texas/100/labels", header=None, encoding="utf-8")
                .to_numpy()
                .reshape(-1)
                - 1
            )
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            all_data = TabularDataset(X, Y)
            total_index = np.arange(len(Y))
        else:
            raise NotImplementedError(
                f"{args.data.dataset} is not installed correctly in ../data/texas"
            )
    elif args.data.dataset == "medicalmnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        )
        all_data = MedicalMNIST(
            root_dir=f"{args.server_dir_path}/data", transform=transform
        )

        num_total_samples = len(all_data)
        total_index = np.arange(num_total_samples)
        X = all_data.data
        Y = all_data.target
    elif args.data.dataset == "pneumonia":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        )
        all_data = Pneumonia(
            root_dir=f"{args.server_dir_path}/data", transform=transform
        )
        num_total_samples = len(all_data)
        total_index = np.arange(num_total_samples)
        X = all_data.data
        Y = all_data.target

    elif args.data.dataset == "retinal":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        )
        all_data = Retinal(root_dir=f"{args.server_dir_path}/data", transform=transform)
        num_total_samples = len(all_data)
        total_index = np.arange(num_total_samples)
        X = all_data.data
        Y = all_data.target
    elif args.data.dataset == "skin":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        )
        all_data = Skin(root_dir=f"{args.server_dir_path}/data", transform=transform)
        num_total_samples = len(all_data)
        total_index = np.arange(num_total_samples)
        X = all_data.data
        Y = all_data.target
    elif args.data.dataset == "kidney":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        )
        all_data = Kidney(root_dir=f"{args.server_dir_path}/data", transform=transform)
        num_total_samples = len(all_data)
        total_index = np.arange(num_total_samples)
        X = all_data.data
        Y = all_data.target

    else:
        raise ValueError("dataset not supported")


    if os.path.exists(f"{args.server_dir_path}/{args.data.dataset}/data.pkl") is False:
        Path(f"{args.server_dir_path}/{args.data.dataset}").mkdir(
            parents=True, exist_ok=True
        )
        with open(f"{args.server_dir_path}/{args.data.dataset}/data.pkl", "wb") as f:
            pickle.dump(all_data, f)

    # get the list of index for the experiments
    # Different types of data partitioning strategies for federated learning:
    
    # 1. IID (Independent and Identically Distributed):
    # - Data is randomly and uniformly distributed across parties
    # - Each party gets roughly same amount and distribution of samples
    # - Simplest partitioning strategy but not realistic
    
    # 2. Dirichlet (Non-IID):
    # - Uses Dirichlet distribution to create imbalanced class distributions
    # - Each party gets different proportions of each class
    # - Controlled by dirichlet_alpha parameter:
    #   - Lower alpha = more imbalanced/non-IID
    #   - Higher alpha = more balanced/IID-like
      
    # 3. IID-K:
    # - Similar to IID but with K parties
    # - Each party gets train_ratio % for training
    # - Remaining data split between test and population sets
    
    # 4. Dirichlet-K:
    # - Combines Dirichlet distribution with K-party split
    # - Creates non-IID splits across K parties
    # - Each party gets train_ratio % of their allocation for training
    


    if args.fl.partitioning == "iid":
        sample_list_dict = get_iid_index(
            total_index, args.data.train_num, args.data.test_num, args.data.pop_num
        )
        save_name = f"{args.data.dataset}/iid_{args.data.train_num}_{args.data.pop_num}_{args.random_seed}"
    elif args.fl.partitioning == "dirichlet":
        sample_list_dict = get_richlet_index(
            total_index,
            Y,
            args.data.train_num,
            args.data.test_num,
            args.data.pop_num,
            args.fl.dirichlet_alpha,
        )
        save_name = f"{args.data.dataset}/dirichlet_{args.fl.dirichlet_alpha}_{args.data.train_num}_{args.data.pop_num}_{args.random_seed}"
        print(sample_list_dict)

    elif args.fl.partitioning == "dirichletk":
        sample_list_dict = get_richlet_indexk(
            total_index,
            Y,
            args.fl.num_parties,
            args.data.train_ratio,
            args.data.test_ratio,
            args.fl.dirichlet_alpha,
        )
        save_name = f"{args.data.dataset}/dirichletk_{args.fl.dirichlet_alpha}_{args.data.train_ratio}_{args.data.pop_ratio}_{args.fl.num_parties}_{args.random_seed}"

    elif args.fl.partitioning == "iidk":
        sample_list_dict = get_iidk_index(
            total_index, args.fl.num_parties, args.data.train_ratio, args.data.pop_ratio
        )
        save_name = f"{args.data.dataset}/iidk_{args.data.train_ratio}_{args.data.pop_ratio}_{args.fl.num_parties}_{args.random_seed}"

    elif "new_iidk" in args.fl.partitioning:
        num_parties = int(args.fl.partitioning.split("_")[-1])
        sample_list_dict = get_iidk_index(
            total_index, num_parties, args.data.train_ratio, args.data.pop_ratio
        )
        save_name = f"{args.data.dataset}/iidk_{args.data.train_ratio}_{args.data.pop_ratio}_{num_parties}_{args.random_seed}"

    Path(f"{args.server_dir_path}/").mkdir(parents=True, exist_ok=True)

    if os.path.exists(f"{args.server_dir_path}/{save_name}.pkl"):
        logging.error(
            f"the {save_name}.pkl exists, please make sure you want to change it"
        )
    else:
        Path(f"{args.server_dir_path}").mkdir(parents=True, exist_ok=True)
        with open(f"{args.server_dir_path}/{save_name}.pkl", "wb") as f:
            pickle.dump(sample_list_dict, f)
        logging.info(f"dataset split is saved into {save_name}")


if __name__ == "__main__":
    main()
