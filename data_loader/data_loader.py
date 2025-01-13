import logging

import fedml
import numpy as np
import torch
import torchvision.transforms as transforms
from data_loader.cifar10_dataset import CIFAR10_truncated
from data_loader.cifar100_dataset import CIFAR100_truncated
from torch.utils import data
from utilies import load_pkl
import pickle
from torch.utils.data import Dataset
from data_loader.medical_mnist import MedicalMNIST
from data_loader.pneumonia import Pneumonia
from data_loader.retinal import Retinal
from data_loader.skin import Skin
from data_loader.kidney import Kidney

def data_transforms_cifar100(argu):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]
    if argu == "strong":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        train_transform.transforms.append(
            Cutout(16)
        )  # no such argumentation will result in a very larger generalization error
    elif argu == "standard":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif argu == "no":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms_cifar10(argu):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    if argu == "strong":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        train_transform.transforms.append(Cutout(16))
    elif argu == "random_ease":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif argu == "standard":
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif argu == "no":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    return train_transform, valid_transform


class TabularDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, X, y):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.data[idx], self.targets[idx]]


def get_dataloader(
    dataset, datadir, train_bs, test_bs, train_idx=None, test_idx=None, argu="no"
):
    if "cifar" in dataset:
        if dataset == "cifar10":
            dl_obj = CIFAR10_truncated
            transform_train, transform_test = data_transforms_cifar10(argu)
        elif dataset == "cifar100":
            dl_obj = CIFAR100_truncated
            transform_train, transform_test = data_transforms_cifar100(argu)

        train_ds = dl_obj(
            datadir, dataidxs=train_idx, transform=transform_train, download=True
        )
        test_ds = dl_obj(
            datadir, dataidxs=test_idx, transform=transform_test, download=True
        )
        train_dl = data.DataLoader(
            dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
        )
        test_dl = data.DataLoader(
            dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False
        )

    elif dataset in ["purchase", "texas", "texas_normalized"]:
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        print(all_dataset.data.shape)
        print(all_dataset.targets.shape)
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )

    elif dataset == "medicalmnist":
        all_dataset = MedicalMNIST(
            root_dir=datadir,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((32, 32))]
            ),
        )
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )
    elif dataset == "pneumonia":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        
        # all_dataset = Pneumonia(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )
    elif dataset == "retinal":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )
    elif dataset == "skin":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        # all_dataset = Skin(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )
    
    elif dataset == "kidney":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        # all_dataset = Kidney(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        train_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, train_idx),
            batch_size=train_bs,
            shuffle=True,
            drop_last=True,
        )
        test_dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, test_idx),
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
        )

    return train_dl, test_dl


def load_data_cifar(args):
    # split the data among different clients, args is the fedml arguments
    if args.dataset == "cifar100":
        class_num = 100
    elif args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "purchase":
        class_num = 100
    elif "texas" in args.dataset:
        class_num = 100
    elif args.dataset == "medicalmnist":
        class_num = 6    
    elif args.dataset == "pneumonia":
        class_num = 2
    elif args.dataset == "retinal":
        class_num = 4
    elif args.dataset == "skin":
        class_num = 23
    elif args.dataset == "kidney":
        class_num = 4
        
    client_list = [i for i in range(args.num_parties)]

    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    if args.partitioning == "iid":
        save_name = (
            f"{args.dataset}/iid_{args.train_num}_{args.pop_num}_{args.random_seed}"
        )
    elif args.partitioning == "dirichlet":
        save_name = f"{args.dataset}/{args.partitioning}_{args.dirichlet_alpha}_{args.train_num}_{args.pop_num}_{args.random_seed}"
            
    elif args.partitioning == "iidk":
        save_name = f"{args.dataset}/iidk_{args.train_ratio}_{args.pop_ratio}_{args.num_parties}_{args.random_seed}"

    elif args.partitioning == "dirichletk":
        save_name = f"{args.dataset}/dirichletk_{args.dirichlet_alpha}_{args.train_ratio}_{args.pop_ratio}_{args.num_parties}_{args.random_seed}"
    
    elif "new_iidk" in args.partitioning:
        num_parties = int(args.partitioning.split("_")[-1])
        save_name = f"{args.dataset}/iidk_{args.train_ratio}_{args.pop_ratio}_{num_parties}_{args.random_seed}"
    
    list_dict = load_pkl(f"{args.server_dir_path}/{save_name}.pkl")

    if args.num_parties > len(list_dict.keys()):
        raise ValueError(
            f"{args.num_parties} is not a valid number of parties for the dataset {args.train_num}, {args.pop_num}"
        )

    all_train_idx = np.concatenate(
        [list_dict[client_idx]["train"] for client_idx in client_list]
    )
    all_test_idx = np.concatenate(
        [list_dict[client_idx]["test"] for client_idx in client_list]
    )

    train_data_num = len(all_train_idx)
    test_data_num = len(all_test_idx)

    train_data_global, test_data_global = get_dataloader(
        args.dataset,
        f"{args.server_dir_path}/data",
        args.train_batch_size,
        args.test_batch_size,
        train_idx=all_train_idx,
        test_idx=all_test_idx,
        argu=args.data_argu,
    )
    logging.info(
        "global_train_num = %d, global_test_num = %d",
        len(all_train_idx),
        len(all_test_idx),
    )
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in client_list:
        data_local_num_dict[client_idx] = len(list_dict[client_idx]["train"])
        train_data_local, test_data_local = get_dataloader(
            args.dataset,
            f"{args.server_dir_path}/data",
            args.train_batch_size,
            args.test_batch_size,
            train_idx=list_dict[client_idx]["train"],
            test_idx=np.concatenate(
                [list_dict[client_idx]["train"], list_dict[client_idx]["test"], list_dict[client_idx]["rest"]]
            ),
            argu=args.data_argu,
        )

        logging.info(
            "client_idx = %s, local_train_num = %s, local_test_num = %s",
            client_idx,
            len(list_dict[client_idx]["train"]),
            len(list_dict[client_idx]["test"]),
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


def load_data(args):
    # main entry to get the data for all clients.
    if args.dataset in ["cifar10", "cifar100", "purchase", "texas", "texas_normalized","medicalmnist", "pneumonia", "retinal", "skin", "kidney"]:
        return load_data_cifar(args)


def get_data_loader_on_val(dataset, datadir, bs, data_idx, argu="no", shuffle=False):
    if "cifar" in dataset:
        if dataset == "cifar10":
            dl_obj = CIFAR10_truncated
            _, transform_test = data_transforms_cifar10(argu)
        elif dataset == "cifar100":
            dl_obj = CIFAR100_truncated
            _, transform_test = data_transforms_cifar100(argu)
        ds = dl_obj(
            datadir, dataidxs=data_idx, transform=transform_test, download=False
        )
        dl = data.DataLoader(dataset=ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    elif dataset in ["purchase", "texas", "texas_normalized"]:
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)

        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    elif dataset == "medicalmnist":
        all_dataset = MedicalMNIST(
            root_dir=datadir,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((32, 32))]
            ),
        )
        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    elif dataset == "pneumonia":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        # all_dataset = Pneumonia(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    elif dataset == "retinal":
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    elif dataset == "skin":
        # all_dataset = Skin(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    elif dataset == "kidney":
        # all_dataset = Kidney(
        #     root_dir=datadir,
        #     transform=transforms.Compose(
        #         [transforms.ToTensor(), transforms.Resize((64, 64))]
        #     ),
        # )
        with open(f"{datadir}/../{dataset}/data.pkl", "rb") as f:
            all_dataset = pickle.load(f)
        dl = data.DataLoader(
            dataset=torch.utils.data.Subset(all_dataset, data_idx),
            batch_size=bs,
            shuffle=shuffle,
            drop_last=False,
        )
    return dl
