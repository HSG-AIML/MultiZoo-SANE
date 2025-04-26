# The Impact of Model Zoo Size and Composition on Weight Space Learning

This repository contains the code for the workshop paper "The Impact of Model Zoo Size and Composition on Weight Space Learning" for the ICLR 2025 Workshop on Neural Network Weights as a New Data Modality. This work introduces an adapdation of the Sequential Autoencoder for Neural Embeddings ([SANE](https://github.com/HSG-AIML/SANE)) to allow training on inhomogenous model zoos. The paper can be found here: [arxiv](https://arxiv.org/abs/2504.10141)

## Summary 

Re-using trained neural network models is a common strategy to reduce training cost and transfer knowledge. Weight space learning - using the weights of trained models as data modality - is a promising new field to re-use populations of pre-trained models for future tasks. Approaches in this field have demonstrated high performance both on model analysis and weight generation tasks. However, until now their learning setup requires homogeneous model zoos where all models share the same exact architecture, limiting their capability to generalize beyond the population of models they saw during training. In this work, we remove this constraint and propose a modification to a common weight space learning method to accommodate training on heterogeneous populations of models. We further investigate the resulting impact of model diversity on generating unseen neural network model weights for zero-shot knowledge transfer. Our extensive experimental evaluation shows that including models with varying underlying image datasets has a high impact on performance and generalization, for both in- and out-of-distribution settings. 

### Key Methods

- **Masked Per-Token Loss Normalization**: Instead of normalizing weights per layer during pre-processing, we propose to normalize the loss per-token at runtime to allow training on inhomogenous model zoos. This allows us to test the impact of model zoo size and composition on downstream task performance.

## Code Structure

- **data/**: Scripts for data preprocessing and loading.
- **experiments/**: Example experiments to pre-train MZ-SANE and sample models
- **src/**: contains the MZ-SANE package to preprocess model checkpoint datasets, pre-train MZ-SANE, and perform discriminative and generative downstream tasks.

## Running Experiments
We include code to run example experiments and showcase how to use our code. 

### Download Model Zoo Datasets
We have made several model zoos available at [modelzoos.cc](https://modelzoos.cc/). Any of these zoos can be used in our pipeline, with minor adjustments.  

To get started with a small experiment, navigate to `./data/` and run 
```bash
bash download_cifar10_cnn_sample.sh
```
This will download and unzip a small model zoo example with CNN models trained on CIFAR-10. 
Training on large model zoos requires preprocessing for training efficiency. We provide code to preprocess training samples. To compile those datasets, run
```bash
python3 preprocess_dataset_cnn_cifar10_sample.py
```
in `./data/`. in the same directory, we provide download and preprocessing scripts for other zoos as well. 
The preprocessed datasets have no specific dependency requirements, other than regular numpy and pytorch.

Please note that this is not the exact models used in the paper and will therefore produce different results. The full zoos can be downloaded from [modelzoos.cc](https://modelzoos.cc/) and used in the same way as the zoo sample.  

### Pretraining MZ-SANE
Code to pretrain SANE on the ResNet18 zoos is contained in `experiments/pretrain_cifar_10_100_res_18.py`. The code relies on ray.tune to manage resources, but currently only runs a single config. 
To vary any of the configurations, exchange the value with `tune.grid_search([value_1, ..., value_n])`. To run the experiment, run
```
python3 pretrain_cifar_10_100_res_18.py
```
in `experiments/`. The code also includes a callback to sample models during training. The callback requires an image dataset dump which can be created with the scripts in `data/vision_datasets` as well as a config.json file which can be taken from one of the models in the zoo. The anchor dataset can be preprocessed with the script in `data/anchor_datasets`.


## Citation
If you use this code in your research, please cite our paper:
```
@misc{falk2025impactmodelzoosize,
      title={The Impact of Model Zoo Size and Composition on Weight Space Learning}, 
      author={Damian Falk and Konstantin Schürholt and Damian Borth},
      year={2025},
      eprint={2504.10141},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.10141}, 
}
```
