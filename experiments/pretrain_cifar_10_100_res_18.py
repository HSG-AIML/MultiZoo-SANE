import logging

logging.basicConfig(level=logging.INFO)

import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch

import ray
from ray import tune

from ray.air.integrations.wandb import WandbLoggerCallback
from mzsane.evaluation.ray_fine_tuning_callback import CheckpointSamplingCallback
from mzsane.evaluation.ray_fine_tuning_callback_subsampled import (
    CheckpointSamplingCallbackSubsampled,
)
from mzsane.evaluation.ray_fine_tuning_callback_bootstrapped import (
    CheckpointSamplingCallbackBootstrapped,
)

import json

from pathlib import Path

from mzsane.models.def_AE_trainable import AE_trainable


PATH_ROOT = Path(".")


def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus_per_trial = 12
    gpus_per_trial = 1
    gpus = 1
    cpus = int(gpus / gpus_per_trial * cpus_per_trial)

    # round down to maximize GPU usage
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    print(f"resources_per_trial: {resources_per_trial}")

    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # 0.0.0.0 is the host of the docker, (localhost is the container) (https://github.com/ray-project/ray/issues/11457#issuecomment-1325344221)
        dashboard_port=8266,
    )
    assert ray.is_initialized() == True

    print(f"started ray. running dashboard under {context.dashboard_url}")

    ### configure experiment #########
    project = "mz-sane"
    experiment_name = "cifar_10_100_kde"
    # set module parameters
    config = {}
    config["experiment::name"] = experiment_name
    config["seed"] = 32
    config["device"] = "cuda"
    config["device_no"] = 1
    config["training::precision"] = "amp"
    config["trainset::batchsize"] = 32

    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = True

    # permutation specs
    config["training::permutation_number"] = 5 
    config["training::view_1_canon"] = True
    config["training::view_2_canon"] = False
    config["testing::permutation_number"] = 5
    config["testing::view_1_canon"] = True
    config["testing::view_2_canon"] = False

    config["training::reduction"] = "mean"

    # standardize loss
    config["training::standardize_token"] = True
    config["training::standardize_std_eps"] = 1e-4

    config["ae:i_dim"] = 288
    config["ae:lat_dim"] = 128

    config["ae:max_positions"] = [60000, 150, 600]
    config["training::windowsize"] = 256 
    config["ae:d_model"] = 2048
    config["ae:nhead"] = 16
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 2e-5
    config["optim::wd"] = 3e-9
    config["optim::scheduler"] = "OneCycleLR"

    # training config
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"

    config["training::epochs_train"] = 60
    config["training::output_epoch"] = 5
    config["training::test_epochs"] = 1

    config["monitor_memory"] = True

    # configure output path
    experiment_dir = PATH_ROOT.joinpath("experiments", experiment_name)
    try:
        experiment_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    config["experiment_dir"] = experiment_dir


    ###### Datasets ###########################################################################
    # path to preprocessed dataset
    data_paths = [
        [
        "/local/multi_zoo/dataset_resnet18_cifar10_uniform_tk288_ep21_25_100s",
        "/local/multi_zoo/dataset_resnet18_cifar100_uniform_tk288_ep_21_25_100s",
        ],
        ]
    # path to ffcv dataset for training
    config["dataset::dump"] = None
    config["preprocessed::sampling"] = True
    config["sampling::path"] = tune.grid_search(data_paths)

    ### Augmentations
    config["trainloader::workers"] = 6
    config["trainset::add_noise_view_1"] = 0.1
    config["trainset::add_noise_view_2"] = 0.1
    config["trainset::noise_multiplicative"] = True
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    config["callbacks"] = [
        CheckpointSamplingCallbackBootstrapped(
            sample_config_path=Path(
                "/path/to/model/params.json"
            ),
            finetuning_epochs=0,
            repetitions=1,
            anchor_ds_path="path/to/anchor/dataset.pt",
            reference_dataset_path=Path(
                "/path/to/image/dataset.pt"
            ),
            bootstrap_iterations=1,
            bootstrap_samples=200,
            bootstrap_keep_top_n=10,
            mode="token",
            norm_mode=None,
            layer_norms_path=None,
            logging_prefix="eval_tin_bs",
            every_n_epochs=0,
            eval_iterations=[40,45,50,55,60],
            batch_size=8,
            anchor_sample_number=5,
            ignore_bn=False,
            halo=True,
            halo_wse=128,
            halo_hs=12,
            bn_condition_iters=50,
        ),
        CheckpointSamplingCallbackBootstrapped(
            sample_config_path=Path(
                "/path/to/model/params.json"
            ),
            finetuning_epochs=0,
            repetitions=1,
            anchor_ds_path="path/to/anchor/dataset.pt",
            reference_dataset_path=Path(
                "/path/to/image/dataset.pt"
            ),
            bootstrap_iterations=1,
            bootstrap_samples=200,
            bootstrap_keep_top_n=10,
            mode="token",
            norm_mode=None,
            layer_norms_path=None,
            logging_prefix="eval_cifar10_bs",
            eval_iterations=[40,45,50,55,60],
            every_n_epochs=0,
            batch_size=8,
            anchor_sample_number=5,
            ignore_bn=False,
            bn_condition_iters=50,
            halo=True,
            halo_wse=128,
            halo_hs=12
        ),
        CheckpointSamplingCallbackBootstrapped(
            sample_config_path=Path(
                "/paths/to/model/params.json"
            ),
            finetuning_epochs=0,
            repetitions=1,
            anchor_ds_path="path/to/anchor/dataset.pt",
            reference_dataset_path=Path(
                "/paths/to/image/dataset.pt"
            ),
            bootstrap_iterations=1,
            bootstrap_samples=200,
            bootstrap_keep_top_n=10,
            mode="token",
            norm_mode=None,
            layer_norms_path=None,
            logging_prefix="eval_cifar100_bs",
            eval_iterations=[40,45,50,55,60],
            every_n_epochs=0,
            batch_size=8,
            anchor_sample_number=5,
            ignore_bn=False,
            bn_condition_iters=50,
            halo=True,
            halo_wse=128,
            halo_hs=12
        )
    ]

    config["wandb"] = {
        "project": project,
        "group": "cifar_10_100_tin_kde",
        "log_config": False,
    }
    config["wandb::api_key_file"] = "/path/to/wandb.key"
    config["resources"] = resources_per_trial
    

    print(f"started ray. running dashboard under {context.dashboard_url}")

    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=config["experiment_dir"],
        resources_per_trial=resources_per_trial,
    )
    # run
    ray.tune.run_experiments(
        experiments=experiment,
        # resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        callbacks=[
            WandbLoggerCallback(
                api_key_file="/path/to/wandb.key",
                project=config["wandb"]["project"],
                group=config["wandb"]["group"],
            ),
        ],
        verbose=3,
    )

    ray.shutdown()
    assert not ray.is_initialized()


if __name__ == "__main__":
    main()