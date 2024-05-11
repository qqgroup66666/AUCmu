# %%
import time
import random
import torch
import numpy as np
import json
import pytest
from easydict import EasyDict as edict
import argparse

@pytest.mark.skip()
def load_json(config_path):
    with open(config_path, 'r') as f:
        args = json.load(f)
        args = edict(args)
    return args

import sys
import shutil
import os
sys.path.append(os.pardir)

from copy import deepcopy
from torchvision import datasets
from losses import SquareAUCLoss, HingeAUCLoss, ExpAUCLoss, \
    AUC_mu, AUCLoss_1, SquareAUCLoss_mine, focal_loss, CB_loss, balanced_softmax_loss, LDAMLoss, SoftF1
from losses import AUC_mu
from metrics import AUC_metric
from models import generate_net, LinearClassifier
import torch.nn.functional as F
from dataloaders import get_data_loaders
from dataloaders import get_datasets
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import pandas
from torch.utils.tensorboard import SummaryWriter
from utils import load_criterion, lr_decay, forward, get_log_file_path, neural_collpase_metric

from tqdm import tqdm
from collections import Counter
import torch.multiprocessing as mp
import torch.distributed as dist


def calculate_matrix(pred, label, class_num):
    mask = torch.stack(
        [label.eq(i) for i in range(class_num)],
    1).squeeze()  # mask
    N = mask.sum(0)  # [类1的样本数目, ...]
    pred = [pred[mask[:, i], :] for i in range(class_num)]
    seperate_aucmu = {}
    seperate_mauc = {}

    # 计算所有类别对的AUC_mu
    for i in range(class_num):
        pred_a = pred[i]
        pred_a_i = pred_a[:, i]
        for j in range(class_num):
            if i == j:
                continue
            pred_b = pred[j]

            pred_a_j = pred_a[:, j]
            pred_b_i = pred_b[:, i]
            pred_b_j = pred_b[:, j]

            pred_a_i = pred_a_i.unsqueeze(0)
            pred_b_i = pred_b_i.unsqueeze(1)
            pred_a_j = pred_a_j.unsqueeze(0)
            pred_b_j = pred_b_j.unsqueeze(1)

            seperate_mauc[str(i)+","+str(j)] = torch.mean( (pred_a_i > pred_b_i).float() + (pred_a_i == pred_b_i).float() / 2 ).item() # MAUC
            seperate_aucmu[str(i)+","+str(j)] = torch.mean((pred_a_i - pred_b_i > pred_a_j - pred_b_j).float() + (pred_a_i - pred_b_i == pred_a_j - pred_b_j).float() / 2 ).item()

    # 用AUCmu,计算代价矩阵，熵矩阵
    aucmu_cost_matrix = torch.zeros(class_num, class_num).cuda()
    aucmu_entropy_matrix = torch.zeros(class_num, class_num).cuda()
    mauc_cost_matrix = torch.zeros(class_num, class_num).cuda()
    mauc_entropy_matrix = torch.zeros(class_num, class_num).cuda()

    for i in range(class_num):
        for j in range(class_num):
            if i == j:
                aucmu_cost_matrix[i, j] = 0
                aucmu_entropy_matrix[i, j] = math.log(2)
                mauc_cost_matrix[i, j] = 0
                mauc_entropy_matrix[i, j] = math.log(2)
                continue

            aucmu_cost_matrix[i, j] = seperate_aucmu[str(i) + "," + str(j)]
            mauc_cost_matrix[i, j] = seperate_mauc[str(i) + "," + str(j)]
            mauc_entropy_matrix[i, j] = - math.log(seperate_mauc[str(i) + "," + str(j)])
            aucmu_entropy_matrix[i, j] = - math.log(seperate_aucmu[str(i) + "," + str(j)])

            # try:
            #     mauc_entropy_matrix[i, j] = - math.log(seperate_mauc[str(i) + "," + str(j)])
            # except:
            #     print("fail MAUC", i, j, seperate_mauc[str(i) + "," + str(j)])

            # try:
            #     aucmu_entropy_matrix[i, j] = - math.log(seperate_aucmu[str(i) + "," + str(j)])
            # except:
            #     print("fail AUCmu", i, j, seperate_aucmu[str(i) + "," + str(j)])

    return aucmu_cost_matrix, aucmu_entropy_matrix, mauc_cost_matrix, mauc_entropy_matrix

def calculate_mean_vari_matrix(test_preds_torch, test_labels_torch, class_num):
    label = test_labels_torch
    pred = test_preds_torch
    mask = torch.stack(
        [label.eq(i) for i in range(class_num)],
    1).squeeze()  # mask
    N = mask.sum(0)  # [类1的样本数目, ...]
    pred = [pred[mask[:, i], :] for i in range(class_num)]

    delta_1_mean_matrix = torch.zeros(class_num, class_num).cuda()
    delta_1_var_matrix = torch.zeros(class_num, class_num).cuda()

    m_1 = torch.zeros(class_num, class_num).cuda()
    m_2 = torch.zeros(class_num, class_num).cuda()
    m_3 = torch.zeros(class_num, class_num).cuda()
    m_4 = torch.zeros(class_num, class_num).cuda()
    m_5 = torch.zeros(class_num, class_num).cuda()
    m_6 = torch.zeros(class_num, class_num).cuda()

    # 计算所有类别对的AUC_mu
    for i in range(class_num):
        pred_a = pred[i]
        pred_a_i = pred_a[:, i]
        for j in range(class_num):
            if i == j:
                continue
            pred_b = pred[j]

            pred_a_j = pred_a[:, j]
            pred_b_i = pred_b[:, i]
            pred_b_j = pred_b[:, j]

            pred_a_i = pred_a_i.unsqueeze(0)
            pred_b_i = pred_b_i.unsqueeze(1)
            pred_a_j = pred_a_j.unsqueeze(0)
            pred_b_j = pred_b_j.unsqueeze(1)

            delta_1 = pred_a_i - pred_b_i
            delta_2 = pred_b_j - pred_a_j
            # delta_1_mean_matrix[i, j] = torch.mean(delta_1)
            # delta_1_var_matrix[i, j] = torch.std(delta_1)

            m_1[i, j] = torch.sum(((-delta_1 < 0) & (0 < delta_2)).long())
            m_2[i, j] = torch.sum(((-delta_1 < delta_2) & (delta_2 < 0)).long())
            m_3[i, j] = torch.sum(((0 < -delta_1) & (-delta_1 < delta_2)).long())
            m_4[i, j] = torch.sum(((0 < delta_2) & (delta_2 < delta_1)).long())
            m_5[i, j] = torch.sum(((delta_2 < 0) & (0 < -delta_1)).long())
            m_6[i, j] = torch.sum(((delta_2 < -delta_1) & (-delta_1 < 0)).long())

    def print__(matrix):
        for i in matrix:
            for j in i:
                print("%.4f" % (j.item()), " ", end="")
            print()
    
    # print("delta_1_mean_matrix: ")
    # print__(delta_1_mean_matrix)
    # print()
    # print("delta_1_mean_matrix - delta_1_mean_matrix.T: ")
    # print__(delta_1_mean_matrix - delta_1_mean_matrix.T)
    # print()
    # print("delta_1_var_matrix: ")
    # print__(delta_1_var_matrix)
    # print()

    # print("m_1")
    # print__(m_1)
    # print()
    # print("m_2")
    # print__(m_2)
    # print()
    # print("m_3")
    # print__(m_3)
    # print()
    # print("m_4")
    # print__(m_4)
    # print()
    # print("m_5")
    # print__(m_5)
    # print()
    # print("m_6")
    # print__(m_6)
    # print()

    dicrimasads = (m_4 + m_6) * (m_2 + m_3) / (m_1 * (m_2 + m_3) + m_5 * (m_4 + m_6))
    print__(dicrimasads)
    print()
    dicrimasads = (m_1 * (m_2 + m_3) + m_5 * (m_4 + m_6)) / ((m_4 + m_6) * (m_2 + m_3))
    print__(dicrimasads)


    return delta_1_mean_matrix, delta_1_var_matrix

def test_correction(train_loader, val_loader, test_loader, class_num, model, correct):

    test(model, train_loader, class_num, des_str="train", train_cls_num_list=train_loader.dataset._cls_num_list)

    test(model, val_loader, class_num, des_str="val", train_cls_num_list=train_loader.dataset._cls_num_list)

    print("-" * 100)
    
    test(model, test_loader, class_num, des_str="test", train_cls_num_list=train_loader.dataset._cls_num_list)

    print()
    print("-"*100)

def test_correction_bias(train_loader, val_loader, test_loader, class_num, model,
    epoch_num, lr, weight_decay, loss_name):
    assert(loss_name in ["CE", "SoftF1"])

    labels = []
    preds = []

    auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
        val_preds_torch, val_labels_torch = test(model, val_loader, class_num, des_str="val")

    print("-" * 100)

    test_auc_mu_, test_auc_ski, test_auc_mine, test_acc, test_precision, test_recall, test_f1, test_predict_entropy, \
        test_preds_torch, test_labels_torch = test(model, test_loader, class_num, des_str="without bias test")

    print()
    print("-"*100)

    class Bias(nn.Module):

        def __init__(self, num_classes):
            super(Bias, self).__init__()
            self.num_classes = num_classes
            # self.bias = torch.nn.Parameter(torch.randn(1, num_classes))
            # torch.nn.init.normal_(self.bias, mean=0.0, std=0.001)
            self.bias = torch.nn.Parameter(torch.zeros(1, num_classes))
            
        def forward(self, x):
            return x + self.bias

    bias = Bias(num_classes=class_num).cuda()

    if loss_name == "SoftF1":
        criterion = SoftF1(class_num).cuda()
        optimizer = torch.optim.Adam([
            {
                'params': bias.parameters(),
                'lr': lr,
                'momentum': 0.9,
                'weight_decay': weight_decay,
                "maximize": True
            }
        ])

    if loss_name == "CE":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([
            {
                'params': bias.parameters(),
                'lr': lr,
                'momentum': 0.9,
                'weight_decay': weight_decay,
            }
        ])

    best_bias = None
    best_bias_metric = 0

    # full batch
    for epoch in range(epoch_num):
        labels = []
        preds = []

        val_loader.dataset.aug = True
        val_loader.shuffle = True
        model.eval()
        with torch.no_grad():
            for img, lbl in tqdm(val_loader):
                img = img.cuda()
                lbl = lbl.cuda()
                out = model(img)
                preds.append(out)
                labels.append(lbl.squeeze(-1))

        labels_torch = torch.cat(labels, dim=0)
        preds_torch = torch.cat(preds, dim=0)
        preds_torch = torch.softmax(preds_torch, dim=1)
        preds_torch_1 = bias(preds_torch)
        preds_torch_prob = torch.softmax(10 * preds_torch_1, dim=1)

        if loss_name == "SoftF1":
            loss, f1 = criterion(preds_torch_prob, labels_torch) # soft_f1, f1
            loss.backward()
            # print("Bias Paramter", bias.bias)
            # print("Bias Paramter Grad", bias.bias.grad)
            optimizer.step()
            optimizer.zero_grad()
        if loss_name == "CE":
            loss = criterion(preds_torch_1, labels_torch)
            loss.backward()
            print("Bias Paramter", bias.bias)
            print("Bias Paramter Grad", bias.bias.grad)
            optimizer.step()
            optimizer.zero_grad()

        auc_mu__, auc_ski_, auc_mine_, acc_, precision_, recall_, f1_, predict_entropy_ = \
            get_metric(preds_torch_1, labels_torch, class_num)
        print()
        # print("val acc: %.4f" % (acc_))
        print("val soft f1: %.4f" % (loss.item()))
        print("val f1: %.4f" % (f1_))

        if loss_name == "SoftF1":
            if best_bias_metric < f1_:
                print("max f1", f1_)
                best_bias_metric = f1_
                best_bias = deepcopy(bias.state_dict())
        if loss_name == "CE":
            if best_bias_metric < acc_:
                print("max acc", acc_)
                best_bias_metric = acc_
                best_bias = deepcopy(bias.state_dict())

    bias.load_state_dict(best_bias)
    labels = []
    preds = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for img, lbl in test_loader:
            if img.shape[0] <= 1:
                continue 
            img = img.cuda()
            lbl = lbl.cuda()
            out = model(img)
            out = softmax(out)
            out = bias(out)
            preds.append(out)
            labels.append(lbl.squeeze(-1))

    labels_torch = torch.cat(labels, dim=0)
    preds_torch = torch.cat(preds, dim=0)
    biased_auc_mu_, biased_auc_ski, biased_auc_mine, biased_acc, biased_precision, biased_recall, biased_f1, \
        biased_predict_entropy = get_metric(preds_torch, labels_torch, class_num)

    # print()
    # print("-"*100)
    # print("test aucmu: %.4f biased incre: %.4f" % (biased_auc_mu_, biased_auc_mu_-test_auc_mu_))
    # print("test mauc: %.4f biased incre: %.4f" % (biased_auc_ski, biased_auc_ski-test_auc_ski))
    # print("test mauc_mine: %.4f biased incre: %.4f" % (biased_auc_mine, biased_auc_mine-test_auc_mine))
    # print("test acc: %.4f biased incre: %.4f" % (biased_acc, biased_acc-test_acc))
    # print("test precision: %.4f biased incre: %.4f" % (biased_precision, biased_precision-test_precision))
    # print("test recall: %.4f biased incre: %.4f" % (biased_recall, biased_recall-test_recall))
    # print("test f1: %.4f biased incre: %.4f" % (biased_f1, biased_f1-test_f1))
    # print("test predict entropy: %.4f biased incre: %.4f" % (biased_predict_entropy, biased_predict_entropy-test_predict_entropy))

def get_metric(preds_torch, labels_torch, class_num, train_cls_num_list=None):

    if train_cls_num_list is not None:
        train_cls_num_list = torch.tensor(train_cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

        one_hot = torch.stack([labels_torch == i for i in range(class_num)], dim=0)
        pred_res = torch.argmax(preds_torch, dim=1)
        every_class_acc = [torch.mean((pred_res[one_hot[c, :]] == c).float()) for c in range(class_num)]
        every_class_acc = torch.tensor(every_class_acc)

        many_shot_acc = torch.mean(every_class_acc[many_shot])
        medium_shot_acc = torch.mean(every_class_acc[medium_shot])
        few_shot_acc = torch.mean(every_class_acc[few_shot])
        many_shot_acc = many_shot_acc.item()
        medium_shot_acc = medium_shot_acc.item()
        few_shot_acc = few_shot_acc.item()
    
    ce_loss = (-torch.log(preds_torch[torch.arange(preds_torch.shape[0]), labels_torch])).mean()

    preds = preds_torch.detach().cpu().numpy()
    labels = labels_torch.detach().cpu().numpy()

    # AUC_mu
    auc_mu_metric = AUC_mu(num_classes = class_num, surrogate="square")
    loss, des = auc_mu_metric(preds_torch, labels_torch)
    auc_mu_ = des["AUC_mu"]

    # multiclass ovo AUC
    one_hot_lables = np.zeros((labels.shape[0], class_num))
    one_hot_lables[np.arange(labels.shape[0]), labels] = 1
    auc_ski = roc_auc_score(one_hot_lables, preds, multi_class='ovo')

    # multiclass ovo AUC of mine
    auc_mine = AUC_metric(preds_torch, labels_torch, num_classes=class_num)

    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true=labels, y_pred=y_pred, normalize=True)
    precision = precision_score(y_true=labels, y_pred=y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true=labels, y_pred=y_pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=y_pred, average="macro")

    # 分类熵
    predict_probability = torch.clamp(torch.softmax(preds_torch, dim=1), min=0.001)
    predict_entropy = torch.mean(torch.sum(- predict_probability * torch.log(predict_probability), dim=1)).item()

    if train_cls_num_list is not None:
        return auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
            many_shot_acc, medium_shot_acc, few_shot_acc
    else:
        return auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy

def print_metric(des_str, auc_mu_=None, auc_ski=None, auc_mine=None, acc=None, precision=None, \
    recall=None, f1=None, predict_entropy=None, many_shot_acc=None, medium_shot_acc=None, few_shot_acc=None):
    metrics = [auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, many_shot_acc, medium_shot_acc, few_shot_acc]
    metric_names = ["auc_mu_", "auc_ski", "auc_mine", "acc", "precision", "recall", "f1", "predict_entropy", "many_shot_acc", "medium_shot_acc", "few_shot_acc"]
    for metric_name, metric in zip(metric_names, metrics):
        if metric is not None:
            print(des_str, metric_name + ": %.4f" % (metric))

def test(model, test_loader, class_num, des_str, train_cls_num_list=None):

    val_loss_sum = 0
    labels = []
    preds = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for img, lbl in test_loader:
            if img.shape[0] <= 1:
                continue 
            img = img.cuda()
            lbl = lbl.cuda()
            out = model(img)
            out = softmax(out)
            preds.append(out)
            labels.append(lbl.squeeze(-1))

    labels_torch = torch.cat(labels, dim=0)
    preds_torch = torch.cat(preds, dim=0)

    if train_cls_num_list is not None:
        auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
            many_shot_acc, medium_shot_acc, few_shot_acc = get_metric(preds_torch, labels_torch, class_num, train_cls_num_list)
        print(des_str, "many shot ACC: %.4f" % (many_shot_acc))
        print(des_str, "medium shot ACC: %.4f" % (medium_shot_acc))
        print(des_str, "few shot ACC: %.4f" % (few_shot_acc))
    else:
        auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy = get_metric(preds_torch, labels_torch, class_num, train_cls_num_list)

    print(des_str, "AUC_mu: %.4f" % (auc_mu_))
    print(des_str, "skilearn_AUC: %.4f" % (auc_ski))
    print(des_str, "my_AUC: %.4f" % (auc_mine))
    print(des_str, "ACC: %.4f" % (acc))
    # print(des_str, "CE: %.4f" % (ce_loss))
    print(des_str, "precision: %.4f" % (precision))
    print(des_str, "recall: %.4f" % (recall))
    print(des_str, "f1: %.4f" % (f1))
    print(des_str, "predict entropy: %.4f" % (predict_entropy))

    return auc_mu_, auc_ski, auc_mine, acc, \
        precision, recall, f1, predict_entropy, \
        preds_torch, labels_torch

def train_classifier(args, encoder, classifier, train_loader, val_loader, test_loader, for_valid):
    max_acc = 0
    class_num = args.model.num_classes
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([
        {
            'params': classifier.parameters(),
            'lr': 5e-1,
            'momentum': 0.9,
            'weight_decay': 0
        }
    ])

    epoch_num = 100
    if for_valid:
        epoch_num = 1

    for epoch in range(epoch_num):
        encoder.train()
        classifier.train()
        train_loss_sum = 0
        sample_num = 0

        train_bar = tqdm(train_loader)
        for img, lbl in train_bar:
            if img.shape[0] < args.training.train_batch_size:
                continue

            img = img.cuda()
            lbl = lbl.cuda()
            with torch.no_grad():
                out = encoder(img)
            out = classifier(out)
            loss = ce(out, lbl)
            des = {"classifier epoch": epoch+1, "loss": loss.item()}

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum += loss.item() * img.shape[0]
            sample_num += img.shape[0]
            des_str = ""
            for k, v in des.items():
                if type(v) == int:
                    des_str += "%s: %d " % (k, v)
                elif type(v) == float:
                    des_str += "%s: %.4f " % (k, v)
                else:
                    raise RuntimeError("??")
            train_bar.set_description(des_str)

        train_mean_loss = train_loss_sum / sample_num
        print("classifier mean loss: %.4f" % (train_mean_loss))

        model = nn.Sequential(
            encoder,
            classifier
        )
        auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
            preds_torch, labels_torch = test(model, val_loader, class_num, des_str="val", \
                train_cls_num_list=train_loader.dataset._cls_num_list)

        if not for_valid:
            # 验证集比较最大指标
            if max_acc < acc:
                max_acc = acc
                best_model = deepcopy(classifier.state_dict())
                best_model_epoch = epoch
                print("max val acc")
        print("-" * 100)
    
    if not for_valid:
        model.load_state_dict(best_model)
        test_correction(train_loader, val_loader, test_loader, class_num, model, correct=False)
    return max_acc

def train(args, SEED, log_root=None, gpu=None):

    distributed = args.get("distributed", False)

    if distributed:
        args.rank = gpu
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                    world_size=args.gpu_num, rank=args.rank)

    writer = SummaryWriter()

    stdout = sys.stdout
    stderr = sys.stderr
    if log_root is not None:
        # 重定向
        begin = time.time()
        from time import sleep
        sleep(random.random())
        log_file_path = get_log_file_path(log_root)
        print("log_file_path:", log_file_path)
        log_file = open(log_file_path, "w+")
        sys.stdout = log_file
        sys.stderr = log_file

        print("log file path:", log_file_path)

        # 最优模型存储位置
        best_model_path = os.path.join(log_root, "best_model.pth")

        print("best model path:", best_model_path)

    print("-"*100)
    print(args)
    print("-"*100)
    print("Random SEED:", SEED)
    print("-"*100)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)

    method = args.training.loss_type
    assert method in [
        "CE", "FocalLoss", "CB", "LDAM", "BalancedSoftmax",
        "MAUC_square", "MAUC_exp", "MAUC_hinge", 
        "AUC_mu_square", "AUC_mu_exp", "AUC_mu_hinge", "BCE",
        "SCL", "MSL", "BCL", "TCL", "ResLT", "SADE", "softf1",
        "PL", "MarginCalibration"
    ]
    print("-"*100)
    print("method:", method)
    print("-"*100)
    print("Dataset:", args.dataset.dataset_name)
    print("-"*100)

    test_batch_size = args.training.test_batch_size
    train_batch_size = args.training.train_batch_size
    print("train batch size:", train_batch_size)

    # val_set, test_set是长尾的，val_uniform_set, test_uniform_set是均匀的
    train_set, val_set, test_set, val_uniform_set, test_uniform_set = get_datasets(args.dataset)
    class_num = len(train_set.class2id.keys())

    print("-"*100)
    print("Traning Set Imbalance Ratio is %.4f." % (max(train_set._cls_num_list) / min(train_set._cls_num_list)))

    print("-"*100)

    num_worker = args.dataset.get("num_worker", 32)

    if distributed:
        train_batch_size = int(train_batch_size / args.gpu_num)
        num_worker = int((num_worker + args.gpu_num - 1) / args.gpu_num)

    if val_set is not None:
        print("There are train, val, test, val_uniform, test_uniform, 5 dataset provided.")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker,
            distributed=distributed
        )

        train_loader, val_uniform_loader, test_uniform_loader = get_data_loaders(
            train_set=train_set,
            val_set=val_uniform_set,
            test_set=test_uniform_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker,
            distributed=distributed
        )

        use_unifrom_test_and_val = args.dataset.get("use_unifrom_test_and_val", False)
        if use_unifrom_test_and_val:
            print("Use unifrom val set and test set.")
            val_loader = val_uniform_loader
            test_loader = test_uniform_loader
        else:
            print("Use val set and test set that have same distributin as tran set.")
    else:
        print("There are train, test, 2 dataset provided.")
        print("No Valid Dataset! Use Test Set as Valid Set!")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_set=train_set,
            val_set=test_set,
            test_set=test_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_worker
        )
    if test_set is None:
        print("test_set is None")
        test_loader = val_loader

    criterion, criterion_CE = load_criterion(method, args, train_set._cls_num_list)

    model_use = args.model.get("model_use", "classifier")
    assert model_use in ("feature_exactor", "BalancedSupConModel", "classifier", \
        "TargetSupConModel", "ResLTModel", "SADEModel", "PLBackbone", "PLBackbone_", \
        "MarginCalibrationBackbone", "neural_collpase", "PLBackbone_factor"
    )

    metric_name = args.training.metric_name
    assert metric_name in ("AUCmu", "ACC")
    print("The evaluation metric is", metric_name)

    model = generate_net(args.model, train_set._cls_num_list)

    if distributed:
        torch.cuda.set_device(args.rank)
        model.cuda(args.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    opt = args.training.get("opt", "Adam")
    assert opt in ("SGD", "Adam")
    print("Use optimizer:", opt)

    maxmize_loss = args.training.get("maxmize_loss", False)
    print("Gradient Desent:", not maxmize_loss)
    if opt == "SGD":
        optimizer = torch.optim.SGD([
            {
                'params': model.parameters(),
                'lr': args.training.lr,
                'momentum': args.training.momentum,
                'weight_decay': args.training.weight_decay,
                'nesterov': args.training.nesterov,
                'maximize': maxmize_loss
            }
        ])
    else:
        optimizer = torch.optim.Adam([
            {
                'params': model.parameters(),
                'lr': args.training.lr,
                'momentum': args.training.momentum,
                'weight_decay': args.training.weight_decay,
                'maximize': maxmize_loss
            }
        ])

    max_eval_metric = 0
    best_model = None
    best_model_epoch = None

    early_stop_epoch = args.training.get("early_stop_epoch", 10000)
    print("Training model. Training_epoch_num: {}. Early stop epoch: {}.".format(args.training.epoch_num, early_stop_epoch))

    for epoch in range(args.training.epoch_num):
        if epoch > early_stop_epoch:
            print("early stop epoch:", epoch)
            break

        if args.training.get("pretrained_encoder_path", None) != None: # load预训练特征提取器
            pretrain_encoder_path = args.training.get("pretrained_encoder_path")
            print("using pretrained feature extractor:", pretrain_encoder_path)
            state_dict = torch.load(pretrain_encoder_path)
            model.load_state_dict(state_dict)
            break

        model.train()
        train_loss_sum = 0
        sample_num = 0
        features = []
        preds = []
        labels = []

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        lr_decay(args, method, optimizer, model, epoch)
        train_bar = tqdm(train_loader)
        for img, lbl in train_bar:
            # multi view
            # img: tensor [batch_size * number of view, features dimension]
            # labels: tensor [batch_size * number of view],
            # 相邻的number of view个数据增广自相同的一张图片.

            # single view
            # print(img.shape) # img: tensor size([batch_size, channel, w, h])
            # print(lbl.shape) # torch.Size([batch_size])

            if args.model.get("model_use", None) == "neural_collpase":
                model[1].set_need_feature(True)
                out, feature, loss, des = forward(args, method, epoch, criterion, model, img, lbl)
                features.append(feature)
            else:
                out, loss, des = forward(args, method, epoch, criterion, model, img, lbl)

            preds.append(out)
            labels.append(lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * img.shape[0]
            sample_num += img.shape[0]
            des_str = ""
            for k, v in des.items():
                if type(v) == int:
                    des_str += "%s: %d " % (k, v)
                elif type(v) == float:
                    des_str += "%s: %.4f " % (k, v)
                else:
                    raise RuntimeError("??")
            train_bar.set_description(des_str)

        with torch.no_grad():
            if args.model.get("model_use", None) == "neural_collpase":
                features = torch.cat(features, dim=0).cuda()
            labels = torch.cat(labels, dim=0).cuda()
            preds = torch.cat(preds, dim=0).cuda()

            y_pred = np.argmax(preds.cpu().detach().numpy(), axis=1)
            train_acc = accuracy_score(y_true=labels.cpu().detach().numpy(), y_pred=y_pred, normalize=True)

            preds = softmax(preds)

            if args.model.get("model_use", None) == "neural_collpase":
                neural_collpase_metric(features, labels, class_num=class_num, desc="train")
                model[1].set_need_feature(False)

            # auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy = get_metric(preds_torch=preds, labels_torch=labels, class_num=class_num)
            # print_metric("train", auc_mu_=auc_mu_, auc_ski=auc_ski, auc_mine=auc_mine, acc=acc, precision=precision, \
            #     recall=recall, f1=f1, predict_entropy=predict_entropy)
            train_mean_loss = train_loss_sum / sample_num

        print("train_mean_loss: %.4f" % (train_mean_loss))
        print("train_acc: %.4f" % (train_acc))

        writer.add_scalar("{}/{}/{}".format(method, args.dataset.dataset_name, "train_mean_loss"), train_mean_loss, global_step=epoch)

        # val....
        auc_mu_, auc_ski, auc_mine, acc, precision, recall, f1, predict_entropy, \
            preds_torch, labels_torch = test(
                model, val_loader, class_num, \
                train_cls_num_list=train_set._cls_num_list, des_str="val"
            )

        if metric_name == "AUCmu":
            eval_metric = auc_mu_
        if metric_name == "ACC":
            eval_metric = acc
        # 验证集比较最大指标
        writer.add_scalar("{}/{}/{}".format(method, args.dataset.dataset_name, "val_"+metric_name), eval_metric, global_step=epoch)
        if max_eval_metric < eval_metric:
            max_eval_metric = eval_metric
            best_model = deepcopy(model.state_dict())
            best_model_epoch = epoch
            print("max val", metric_name)
        print("-" * 100)
        

    # test....
    
    print("max val metric model in epoch{}.".format(best_model_epoch))
    model.load_state_dict(best_model)
    test_correction(train_loader, val_loader, test_loader, class_num, model, correct=False)
    if log_root is not None:
        torch.save(best_model, best_model_path)
    print("method: ", method)

    if log_root is not None:
        end = time.time()
        print("training time:", (end - begin))
        log_file.close()
        sys.stdout = stdout
        sys.stderr = stderr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    args = vars(parser.parse_args())
    assert args["config"] is not None

    config_path = args["config"]

    args = load_json(config_path)

    SEED = random.randint(0, 1000000)
    train(args=args, SEED=SEED)
