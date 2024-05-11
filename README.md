
# AUCmu 

> Gao Peifeng, Qianqian Xu, Peisong Wen, Huiyang Shao, Yuan He, Qingming Huang. Towards Decision-Friendly AUC: Learning Multi-Classifier with AUCµ. AAAI 2023

## Environment

- GPU: Nvidia GTX 3090
- OS: Ubuntu 20.04
- python version: python 3.8
- pytorch version: 1.13.0

## Dataset

This paper uses four dataset, including:

1. CIFAR10
2. CIFAR100
3. Tiny-ImageNet
4. ImageNet-LT

These datasets are balanced. This project provides a generated long-tail version of the corresponding datasets.
One can download these datasets from the following Baidu online disk:

Link:  https://pan.baidu.com/s/1rMueu7htsT3mgHdeKemCbQ 
Extraction code: 1234 

After downloading the data, decompress the data in the compressed package to **./run/datasets/** directory,
Finally, the **datasets** directory should contain the following folders:

```
|--datasets
   |-- cifar-10-lt  
   |-- cifar-100-lt
   |-- tiny-imagenet-200-lt
   |-- ImageNet-lt_mine
```

## Code Runing

### Python Environment

This project requires some Python software packages, which can be directly installed through the following command 
```
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirements.txt
```

## Configuration File 

All config files for different methods could be found in path *./runs/configs*. We name the config file as it's corresponding method and dataset.
The important parameters in a config file include

+ **model.num_classes**
+ **dataset.data_dir**
+ **training.loss_type**
+ **training.epoch_num**
+ **training.train_batch_size**
+ **training.lr**
+ **training.weight_decay**

Spectific parameters of different loss function could be found in the corresponding config files. 

There are 12 json files in **./configs/** folder, which correspond to the training configurations of four data sets and three loss functions respectively:

+ ./configs/cifar100_aucmu_exp.json
+ ./configs/cifar10_aucmu_exp.json
+ ./configs/tiny-imagenet-200-lt_aucmu_exp.json
+ ./configs/imagenet-lt-1_aucmu_exp.json
+ ./configs/cifar100_aucmu_hinge.json
+ ./configs/cifar10_aucmu_hinge.json
+ ./configs/tiny-imagenet-200-lt_aucmu_hinge.json
+ ./configs/imagenet-lt-1_aucmu_hinge.json
+ ./configs/cifar100_aucmu_square.json
+ ./configs/cifar10_aucmu_square.json
+ ./configs/tiny-imagenet-200-lt_aucmu_square.json
+ ./configs/imagenet-lt-1_aucmu_square.json

## Training

Before one run the demo code, one needs to change the parameter **dataset.data_dir** in config file **./run/configs/cifar10_ce.json**.

**For example**: 
One try train model using AUCmu_exp loss function on CIFAR100 dataset, he can type 
the following command:

```
cd run
python -u main.py -c configs/cifar100_aucmu_exp.json
```

## Result

After the experiments, the following final results will be output: 

```
0%|          | 0/3 [00:00<?, ?it/s]
33%|███▎      | 1/3 [00:02<00:04,  2.35s/it]
100%|██████████| 3/3 [00:02<00:00,  1.17it/s]
test many shot ACC: 0.4685
test medium shot ACC: 0.2093
test few shot ACC: 0.0944
test AUC_mu: 0.9380
test skilearn_AUC: 0.8970
test my_AUC: 0.8922
test ACC: 0.4310
test precision: 0.2714
test recall: 0.2630
test f1: 0.2526
test predict entropy: 4.6035
```

where the **test AUC_mu** metric is the final model performance metric reported in the paper.
