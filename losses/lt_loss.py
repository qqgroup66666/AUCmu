
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .AUROCLoss import balanced_softmax_loss

class ResLTLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, num_classes, cls_num_list, beta=0.5):
        super(ResLTLoss , self).__init__()
        self.num_classes = num_classes
        self.cls_num_list = cls_num_list
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        self.calculate_mask()
        self.softmax = nn.Softmax(dim=1)
    
    def calculate_mask(self):
        train_cls_num_list = torch.tensor(self.cls_num_list)
        self.many_shot_mask = train_cls_num_list > 100
        self.medium_shot_mask = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        self.few_shot_mask = train_cls_num_list < 20

    def __crossEntropy(self, softmax, logit, label, weight):
        target = F.one_hot(label, self.num_classes)
        loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
        return loss

    def forward(self, x_many, x_medium, x_few, lbl):
        # x_many.shape: size([batch_size, feature_dim])
        # x_medium.shape: size([batch_size, feature_dim])
        # x_few.shape: size([batch_size, feature_dim])
        # lbl.shape: size([batch_size,])
        labelH = F.one_hot(lbl, self.num_classes).sum(dim=1)
        labelM = F.one_hot(lbl, self.num_classes)[:, self.medium_shot_mask].sum(dim=1)
        labelT = F.one_hot(lbl, self.num_classes)[:, self.few_shot_mask].sum(dim=1)
        loss_ice = (
            self.__crossEntropy(self.softmax, x_many, lbl, labelH) + 
            self.__crossEntropy(self.softmax, x_medium, lbl, labelM) + 
            self.__crossEntropy(self.softmax, x_few, lbl, labelT)
        ) / (labelH.sum() + labelM.sum() + labelT.sum()).float()

        logit = x_many + x_medium + x_few
        loss_fce = self.ce(logit, lbl)

        loss = loss_ice * self.beta + (1-self.beta) * loss_fce
        return loss

class SADELoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, cls_num_list, lambda_=1):
        super(SADELoss , self).__init__()
        self.lambda_ = lambda_
        self.class_num = len(cls_num_list)
        self.label_distribution = cls_num_list
        self.inverse_label_distribution = [
            i / self.lambda_ * j for i,j in zip(cls_num_list, cls_num_list[::-1])
        ]

    def forward(self, logits, label):
        # x: torch.size([batch_size, 3, class_num])
        logits_0 = logits[:, 0]
        logits_1 = logits[:, 1]
        logits_2 = logits[:, 2]

        loss_0 = F.cross_entropy(logits_0, label)
        loss_1 = balanced_softmax_loss(logits_1, label, self.label_distribution)
        loss_2 = balanced_softmax_loss(logits_2, label, self.inverse_label_distribution)
        return (loss_0 + loss_1 + loss_2) / 3

class PLLoss(nn.Module):
    def __init__(self, cls_num_list, lambda_=1):
        super(PLLoss , self).__init__()
        self.cls_num_list = cls_num_list

    def forward(self, logits, label):
        return balanced_softmax_loss(logits, label, sample_per_class=self.cls_num_list)
