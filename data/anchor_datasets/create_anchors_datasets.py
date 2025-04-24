from shrp.models.def_downstream_module import (
    DownstreamTaskLearner as DownstreamTaskLearner,
)
from shrp.datasets.dataset_tokens import DatasetTokens

from pathlib import Path

import json
import os
import torch

import logging

logging.basicConfig(level=logging.INFO)


zoo_path = [  
    Path(
            "/path/to/zoo"
        ).absolute(),
]

zoo_names = [
    "cifar10",
]


def load_dataset_from_config(zoo_path, epoch_list, permutation_spec, map_to_canonical, filter_fn=None):
    # get trainset
    trainset = load_single_dataset(
        zoo_path=zoo_path,
        epoch_list=epoch_list,
        split="train",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        filter_fn=filter_fn,
    )
    valset = load_single_dataset(
        zoo_path=zoo_path,
        epoch_list=epoch_list,
        split="val",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        filter_fn=filter_fn,
    )
    testset = load_single_dataset(
        zoo_path=zoo_path,
        epoch_list=epoch_list,
        split="test",
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        filter_fn=filter_fn,
    )
    return trainset, valset, testset


def load_single_dataset(
    zoo_path, epoch_list, split, permutation_spec, map_to_canonical, filter_fn=None,
):
    standardize = False
    ds_split = [0.7, 0.15, 0.15]
    max_samples = 20
    weight_threshold = 250000000000
    property_keys = {
        "result_keys": [
            "test_acc",
            "training_iteration",
            "ggap",
        ],
        "config_keys": [],
    }
    ignore_bn = False
    tokensize = 288

    # load dataset
    dataset = DatasetTokens(
        root=zoo_path,
        epoch_lst=epoch_list,
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        standardize=standardize,
        train_val_test=split,  # determines which dataset split to use
        ds_split=ds_split,  #
        max_samples=max_samples,
        weight_threshold=weight_threshold,
        filter_function=filter_fn,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=property_keys,
        num_threads=12,
        shuffle_path=True,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=ignore_bn,
        tokensize=tokensize,
        dense_tokens=True,
        use_relative_pos=True,
    )
    return dataset


logging.info("load dataset")
epoch_list = [25]


for i, path in enumerate(zoo_path):
    zoo_name = zoo_names[i]
    logging.info(f"Loading dataset {zoo_name} from {path}...")
    # create directory for the dataset
    PREFIX = f"{zoo_name}_anchor/"
    os.makedirs(PREFIX, exist_ok=True)


# """
    ds_train, ds_val, ds_test = load_dataset_from_config(
        zoo_path=[path],
        epoch_list=epoch_list,
        permutation_spec=None,
        map_to_canonical=False,
        filter_fn=None,
    )

    dataset = {
        "trainset": ds_train,
        "valset": ds_val,
        "testset": ds_test,
    }

    torch.save(ds_train, PREFIX + "dataset_train.pt")
    torch.save(ds_val, PREFIX + "dataset_val.pt")
    torch.save(ds_test, PREFIX + "dataset_test.pt")
    torch.save(dataset, PREFIX + "dataset.pt")



