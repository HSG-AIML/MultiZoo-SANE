from collections import OrderedDict
from mzsane.datasets.dataset_auxiliaries import tokens_to_checkpoint, tokenize_checkpoint
from mzsane.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from mzsane.models.def_NN_experiment import NNmodule

import torch

from typing import Optional, List, Any

import numpy as np


##
def sample_model_evaluation(
    ddpm_model,
    sample_config: dict,
    finetuning_epochs: int,
    repetitions: int,
    tokensize: int,
    norm_mode: Optional[str] = None,
    layer_norms: Optional[dict] = None,
    properties: Optional[List[Any]] = None,
) -> dict:
    """
    runs evaluation pipeline.
    samples ddpm model to generate checkpoints, finetunes checkpoints on downstream task and evaluates finetuned checkpoints
    Args:
        ddpm_model (DDPM): ddpm model
        sample_config (dict): dictionary containing config for sampled model
        finetuning_epochs (int): number of epochs to finetune
        repetitions (int): number of repetitions to finetune and evaluate
        tokensize (int): dimenions of tokens
    Returns:
        dict: dictionary containing evaluation results
    """
    # init output
    results = {}
    # get reference model tokens
    module = NNmodule(sample_config)
    checkpoint_ref = module.model.state_dict()
    _, mask, pos = tokenize_checkpoint(
        checkpoint=checkpoint_ref,
        tokensize=tokensize,
        return_mask=True,
        ignore_bn=False,
    )
    # sample models
    checkpoints = sample_models(
        ddpm_model=ddpm_model,
        checkpoint_ref=checkpoint_ref,
        pos=pos,
        mask=mask,
        repetitions=repetitions,
        properties=properties,
    )
    # de-normalize checkoints
    if norm_mode is not None:
        for idx in range(len(checkpoints)):
            checkpoints[idx] = de_normalize_checkpoint(
                checkpoints[idx], layers=layer_norms, mode=norm_mode
            )
    # evaluate models
    for rep in range(repetitions):
        # sample model
        checkpoint = checkpoints[rep]
        # finetune and evaluate model
        res_tmp = evaluate_single_model(sample_config, checkpoint, finetuning_epochs)
        # append results
        for k in res_tmp.keys():
            results[f"eval/model_{rep}_{k}"] = res_tmp[k]

    # aggregate results over models
    for k in res_tmp.keys():
        res_tmp = []
        for rep in range(repetitions):
            res_tmp.append(results[f"eval/model_{rep}_{k}"])

        results[f"eval/{k}_mean"] = []
        results[f"eval/{k}_std"] = []
        for idx in range(len(res_tmp[0])):
            res_ep = [res_tmp[jdx][idx] for jdx in range(len(res_tmp))]
            results[f"eval/{k}_mean"].append(np.mean(res_ep))
            results[f"eval/{k}_std"].append(np.std(res_ep))

    # return results
    return results


def sample_models(
    ddpm_model,
    checkpoint_ref: OrderedDict,
    pos: torch.Tensor,
    mask: torch.Tensor,
    repetitions: int,
    properties: Optional[List[Any]] = None,
) -> OrderedDict:
    """
    Sample a single model from a DDPM model.
    Args:
        ddpm_model (DDPM): DDPM model.
        checkpoint_ref (str): Reference checkpoint for the sampled model
        pos (int): reference positions for sampling of shape [windowsize,3]
        mask (torch.Tensor): reference mask for sampling
        repetitions (int): number of repetitions to sample
    Returns:
        List[OrderedDict]: sampled model state dicts
    """
    # call the ddpm model's sample method
    # tkdx = torch.randn(mask.shape)
    pos = pos.unsqueeze(dim=0).to("cpu")
    model_kwargs = {
        "p": pos.to(torch.int).to(ddpm_model.device),
    }
    if properties:
        model_kwargs["context"] = (
            torch.Tensor(properties).unsqueeze(dim=0).to(ddpm_model.device)
        )
    sampled_tokens = []
    for _ in range(repetitions):
        sampled_tokens += [
            ddpm_model.p_sample_loop(
                shape=mask.shape, batch_size=1, model_kwargs=model_kwargs
            )[0]
            .to("cpu")
            .squeeze()
        ]

    # create a new state dict
    returns = []
    for idx in range(repetitions):
        checkpoint = tokens_to_checkpoint(
            sampled_tokens[idx], pos.squeeze(), checkpoint_ref, ignore_bn=False
        )
        returns.append(checkpoint)
    return returns


def de_normalize_checkpoint(checkpoint, layers, mode="minmax"):
    """
    revert normalization
    """
    # iterate over layer keys instead of checkpoint keys
    # that way, we only consider layers for which we have norm values
    for key in layers.keys():
        if key == "mode":
            continue
        if mode == "standardize":
            # get mean and std
            mu = layers[key]["mean"]
            sigma = layers[key]["std"]
            # de-normalize weights
            checkpoint[key] = checkpoint[key] * sigma + mu
            # de-noramlize bias
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] * sigma + mu
                )
        elif mode == "minmax":
            # get global min and max values
            min_glob = layers[key]["min"]
            max_glob = layers[key]["max"]
            # reverse of min-max normalization (mapped to range [-1,1])
            # returns weights exactly to original range
            checkpoint[key] = (checkpoint[key] + 1) * (
                max_glob - min_glob
            ) / 2 + min_glob
            # de-normalize bais
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] + 1
                ) * (max_glob - min_glob) / 2 + min_glob

    return checkpoint


def evaluate_single_model(
    config: dict, checkpoint: OrderedDict, fintuning_epochs: int = 0
) -> dict:
    """
    evaluates a single model on a single task
    Args:
        config (dict): dictionary containing config for the model
        checkpoint (OrderedDict): state dict of the model
        fintuning_epochs (int): number of epochs to finetune
    Returns:
        dict: dictionary containing evaluation results
    """
    # init output
    results = {}
    # load datasets
    trainloader, testloader, valloader = load_datasets_from_config(config)
    # init model
    config["scheduler::steps_per_epoch"] = len(trainloader)
    module = NNmodule(config, cuda=True)
    # load checkpoint
    module.model.load_state_dict(checkpoint)
    # eval zero shot
    loss_train, acc_train = module.test_epoch(trainloader, 0)
    loss_test, acc_test = module.test_epoch(testloader, 0)
    results["loss_train"] = [loss_train]
    results["acc_train"] = [acc_train]
    results["loss_test"] = [loss_test]
    results["acc_test"] = [acc_test]
    if valloader is not None:
        loss_val, acc_val = module.test_epoch(valloader, 0)
        results["loss_val"] = [loss_val]
        results["acc_val"] = [acc_val]
    # finetune model
    for idx in range(fintuning_epochs):
        loss_train, acc_train = module.train_epoch(trainloader, 0)
        loss_test, acc_test = module.test_epoch(testloader, 0)
        results["loss_train"].append(loss_train)
        results["acc_train"].append(acc_train)
        results["loss_test"].append(loss_test)
        results["acc_test"].append(acc_test)
        if valloader is not None:
            loss_val, acc_val = module.test_epoch(valloader, 0)
            results["loss_val"].append(loss_val)
            results["acc_val"].append(acc_val)
    # return results
    return results


def load_datasets_from_config(config):
    if config.get("dataset::dump", None) is not None:
        print(f"loading data from {config['dataset::dump']}")
        dataset = torch.load(config["dataset::dump"])
        trainset = dataset["trainset"]
        testset = dataset["testset"]
        valset = dataset.get("valset", None)
    else:
        data_path = config["training::data_path"]
        fname = f"{data_path}/train_data.pt"
        train_data = torch.load(fname)
        train_data = torch.stack(train_data)
        fname = f"{data_path}/train_labels.pt"
        train_labels = torch.load(fname)
        train_labels = torch.tensor(train_labels)
        # test
        fname = f"{data_path}/test_data.pt"
        test_data = torch.load(fname)
        test_data = torch.stack(test_data)
        fname = f"{data_path}/test_labels.pt"
        test_labels = torch.load(fname)
        test_labels = torch.tensor(test_labels)
        #
        # Flatten images for MLP
        if config["model::type"] == "MLP":
            train_data = train_data.flatten(start_dim=1)
            test_data = test_data.flatten(start_dim=1)
        # send data to device
        if config["cuda"]:
            train_data, train_labels = train_data.cuda(), train_labels.cuda()
            test_data, test_labels = test_data.cuda(), test_labels.cuda()
        else:
            print(
                "### WARNING ### : using tensor dataloader without cuda. probably slow"
            )
        # create new tensor datasets
        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # instanciate Tensordatasets
    dl_type = config.get("training::dataloader", "tensor")
    if dl_type == "tensor":
        trainloader = FastTensorDataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            # num_workers=config.get("testloader::workers", 2),
        )
        testloader = FastTensorDataLoader(
            dataset=testset, batch_size=len(testset), shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = FastTensorDataLoader(
                dataset=valset, batch_size=len(valset), shuffle=False
            )

    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            num_workers=config.get("testloader::workers", 2),
        )
        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=config["training::batchsize"], shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = torch.utils.data.DataLoader(
                dataset=valset, batch_size=config["training::batchsize"], shuffle=False
            )

    return trainloader, testloader, valloader
