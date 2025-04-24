import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import logging

class PreprocessedSamplingDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and transforming preprocessed samples from multiple directories.

    Args:
        zoo_paths (list of str): List of directory paths containing the datasets.
        split (str): The dataset split to load (e.g., 'train', 'test'). Defaults to 'train'.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.

    Attributes:
        zoo_paths (list of str): List of directory paths containing the datasets.
        split (str): The dataset split to load (e.g., 'train', 'test').
        datasets (list of str): List of directories containing the datasets for the specified split.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        samples (list of str): List of file paths to the samples in the datasets.

    Methods:
        load_datasets(zoo_paths): Loads and validates dataset directories based on the given paths and split.
        collect_samples(datasets): Collects all sample file paths from the loaded dataset directories.
        __len__(): Returns the total number of samples.
        __getitem__(idx): Loads and returns a sample by index, applying transformations if specified.

    Example:
        dataset = PreprocessedSamplingDataset(zoo_paths=["/path/to/dataset1", "/path/to/dataset2"], split="train")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for ddx, mask, pos, props in dataloader:
            # Training loop code here
    """

    def __init__(self, zoo_paths, split="train", transforms=None):
        self.zoo_paths = zoo_paths
        self.split = split
        self.datasets = self.load_datasets(zoo_paths)
        self.transforms = transforms
        self.samples = self.collect_samples(self.datasets)

    def load_datasets(self, zoo_paths):
        """
        Loads and validates dataset directories based on the given paths and split.

        Args:
            zoo_paths (list of str): List of directory paths containing the datasets.

        Returns:
            list of str: List of directories containing the datasets for the specified split.

        Raises:
            NotADirectoryError: If a specified directory does not exist.
        """

    def load_datasets(self, zoo_paths):
        datasets = []
        for path in zoo_paths:
            directory_path = (
                Path(path).joinpath(f"dataset_torch.{self.split}").absolute()
            )
            directory_path_v2 = (
                Path(path).joinpath(f"{self.split}").absolute()
            )
            if os.path.isdir(directory_path):
                datasets.append(directory_path)
            elif os.path.isdir(directory_path_v2):
                datasets.append(directory_path_v2)
            else:
                raise NotADirectoryError(f"Directory not found: {directory_path}")
        return datasets

    def collect_samples(self, datasets):
        """
        Collects all sample file paths from the loaded dataset directories.

        Args:
            datasets (list of str): List of directories containing the datasets for the specified split.

        Returns:
            list of str: List of file paths to the samples in the datasets.
        """
        samples = []
        for dataset in datasets:
            for file_name in os.listdir(dataset):
                file_path = os.path.join(dataset, file_name)
                if os.path.isfile(file_path):
                    samples.append(file_path)
        return samples

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns a sample by index, applying transformations if specified.

        Args:
            idx (int): Index of the sample to be loaded.

        Returns:
            tuple: A tuple containing the loaded sample data (ddx, mask, pos, props).
        """
        file_path = self.samples[idx]
        item = torch.load(file_path)

        ddx = item["w"]
        mask = item["m"]
        pos = item["p"]
        props = item["props"]

        if self.transforms:
            sample = self.transforms(ddx, mask, pos, props)
            return sample
        return ddx, mask, pos, props

