import torch.nn as nn
import torch
import math
import os
import torch.nn.functional as F
from losses import SquareAUCLoss, HingeAUCLoss, ExpAUCLoss, \
    AUC_mu, AUCLoss_1, SquareAUCLoss_mine, focal_loss, CB_loss, balanced_softmax_loss, \
    LDAMLoss, SoftF1, SupConLoss, MultiSimilarityLoss, AUC_mu, BalancedSupConLoss, \
        TargetSupConLoss, ResLTLoss, SADELoss, MarginCalibrationLoss, PLLoss

def get_log_file_path(log_root):
    log_file = [i for i in os.listdir(log_root) if i.startswith("log_")]
    log_file_index = [int(i.lstrip("log_")) for i in log_file]
    if len(log_file_index) == 0:
        return os.path.join(log_root, "log_0")
    else:
        index = max(log_file_index) + 1
        return os.path.join(log_root, "log_" + str(index))

def cos_decay(epoch, begin, end):
    return 0.5 * (1. + math.cos(math.pi * (epoch - begin + 1) / (end - begin)))

def load_criterion(method, args, cls_num_list):
    class_num = args.model.num_classes
    if method == "CE": # 1
        criterion = nn.CrossEntropyLoss()
    elif method == "BCE": # 1
        criterion = nn.BCEWithLogitsLoss()
    elif method == "CB": # 3
        loss_params = args.training.loss_params
        loss_type = loss_params["loss_type"]
        beta = loss_params["beta"]
        gamma = loss_params["gamma"]
        def CBLoss(logits, labels):
            return CB_loss(logits, labels, 
                samples_per_cls=cls_num_list, no_of_classes=len(cls_num_list), 
                loss_type=loss_type, beta=beta, gamma=gamma
            )
        criterion = CBLoss
    elif method == "FocalLoss": # 1
        loss_params = args.training.loss_params
        alpha = loss_params["alpha"]
        gamma = loss_params["gamma"]
        def focalLoss(logits, labels):
            return focal_loss(logits, labels, alpha=alpha, gamma=gamma)
        criterion = focalLoss
    elif method == "LDAM": # 1
        criterion = LDAMLoss(cls_num_list=cls_num_list)
    elif method == "BalancedSoftmax": # 1
        def BalancedSoftmaxLoss(logits, labels):
            return balanced_softmax_loss(logits, labels, sample_per_class=cls_num_list)
        criterion = BalancedSoftmaxLoss
    elif method.startswith("MAUC"): # 3
        if method.endswith("square"):
            criterion = SquareAUCLoss(num_classes=class_num)
        if method.endswith("exp"):
            criterion = ExpAUCLoss(num_classes=class_num)
        if method.endswith("hinge"):
            criterion = HingeAUCLoss(num_classes=class_num)
    elif method.startswith("AUC_mu"): # 3
        try:
            args.training.loss_params.a
        except:
            args.training.loss_params["a"] = 1
        if method.endswith("square"):
            criterion = AUC_mu(num_classes=class_num, surrogate="square", a=args.training.loss_params.a)
        if method.endswith("exp"):
            criterion = AUC_mu(num_classes=class_num, surrogate="exp", a=args.training.loss_params.a)
        if method.endswith("hinge"):
            criterion = AUC_mu(num_classes=class_num, surrogate="hinge", a=args.training.loss_params.a)
    elif method == "MSL":
        criterion = MultiSimilarityLoss()
    elif method == "SCL":
        criterion = SupConLoss(class_num=len(cls_num_list))
    elif method == "BCL":
        lambda_ = args.training.loss_params.lambda_
        mu_ = args.training.loss_params.mu_
        criterion = BalancedSupConLoss(sample_per_class=cls_num_list, lambda_=lambda_, mu_=mu_)
    elif method == "TCL":
        criterion = TargetSupConLoss()
    elif method == "ResLT":
        criterion = ResLTLoss(num_classes=class_num, cls_num_list=cls_num_list)
    elif method == "SADE":
        criterion = SADELoss(cls_num_list=cls_num_list)
    elif method == "softf1":
        criterion = SoftF1(num_classes=class_num)
    elif method == "PL":
        return PLLoss(cls_num_list=cls_num_list), None
    elif method == "MarginCalibration":
        criterion = MarginCalibrationLoss(cls_num_list=cls_num_list)
        return criterion, None
    else:
        raise RuntimeError("?")
    
    criterion_CE = nn.CrossEntropyLoss()
    return criterion, criterion_CE

def neural_collpase_metric(features, labels, class_num, desc=""):

    feature_dim = features.shape[1]

    feature_mean = torch.zeros(class_num, feature_dim).cuda()

    for i in range(class_num):
        feature_mean[i] = torch.mean(features[labels == i], dim=0)

    global_feature_mean = torch.mean(features, dim=0)
    feature_mean = feature_mean - global_feature_mean.unsqueeze(0)

    norm_feature_mean = torch.norm(feature_mean, p=2, dim=1)
    print(desc + " class norm std/mean: %.4f" % ((torch.std(norm_feature_mean) / torch.mean(norm_feature_mean)).item()))

    normalized_feature_mean = nn.functional.normalize(feature_mean, dim=1)
    gram_matrix = torch.mm(normalized_feature_mean, normalized_feature_mean.T)
    all_cos = gram_matrix[gram_matrix < 0.999]

    print(desc + " std cos: %.4f" % (torch.std(all_cos).item()))

def forward(args, method, epoch, criterion, model, img, lbl):
    des = {"epoch": epoch+1}

    if method in ("BCL", "TCL"):
        num_view = args.dataset.multi_view_num
        batch_size = int(img.shape[0] / num_view)

        # 多个view的数据
        img = img.cuda() # multiview data, 
        lbl = lbl.cuda() # multiview data, 

        img = img.reshape(batch_size, num_view, *img.shape[1:])
        lbl = lbl.reshape(batch_size, num_view)
        lbl = lbl[:, 0]

    if method == "BCL":
        out = model(img)
        ce_loss, sc_loss, loss = criterion(*out, lbl)
        des["ce_loss"] = ce_loss.item()
        des["sc_loss"] = sc_loss.item()
        des["loss"] = loss.item()
        return loss, des

    if method == "TCL":
        query, key = model(img)
        criterion(*out, lbl)
        # ce_loss, sc_loss, loss = criterion(*out, lbl)
        # des["ce_loss"] = ce_loss.item()
        # des["sc_loss"] = sc_loss.item()
        # des["loss"] = loss.item()
        return loss, des

    img = img.cuda()
    lbl = lbl.cuda()
    if args.model.get("model_use", None) == "neural_collpase":
        out, feature = model(img)
    if args.model.get("model_use", None) == "PLBackbone":
        out = model(img)
    else:
        out = model(img)

    if method in ("CE", "FocalLoss", "CB", "LDAM", "BalancedSoftmax"):
        loss = criterion(out, lbl)
        des["loss"] = loss.item()
    elif method.startswith("BCE"):
        labels_one_hot = F.one_hot(lbl, class_num).float()
        loss = criterion(out, labels_one_hot)
        des["loss"] = loss.item()
    elif method.startswith("MAUC") and args.training.two_stage:
        if epoch + 1 <= args.training.stage1_epoch: # stage 1 CE训练
            # if args.training.stage1_loss == "CE":
            loss = nn.functional.cross_entropy(out, lbl)
            des["CE_loss"] = loss.item()
        else: # stage 2 AUC训练
            out = F.softmax(out, dim=1)
            loss = criterion(out, lbl)
            des["loss"] = loss.item()
    elif method.startswith("MAUC") and not args.training.two_stage:
        out = sigmoid(out)
        loss = criterion(out, lbl)
        des["loss"] = loss.item()
    elif method.startswith("AUC_mu") and args.training.two_stage:
        if epoch + 1 <= args.training.stage1_epoch: # stage 1 CE训练
            # if args.training.stage1_loss == "CE":
            loss = nn.functional.cross_entropy(out, lbl)
            des["CE_loss"] = loss.item()
        else: # stage 2 AUC训练
            out = F.softmax(out, dim=1)
            loss, des_ = criterion(out, lbl)
            des["loss"] = loss.item()
            des.update(des_)
    elif method.startswith("AUC_mu") and not args.training.two_stage:
        out = sigmoid(out)
        loss, des_ = criterion(out, lbl)
        des["loss"] = loss.item()
        des.update(des_)
    elif method in ("MSL", "SCL"):
        loss = criterion(out, lbl)
        des["loss"] = loss.item()
    elif method == "ResLT":
        x_many, x_medium, x_few = out
        loss = criterion(x_many, x_medium, x_few, lbl)
        des["loss"] = loss.item()
    elif method == "SADE":
        # logits = out["logits"]
        loss = criterion(out, lbl)
        des["loss"] = loss.item()
    elif method == "softf1":
        soft_macro_f1, macro_f1 = criterion(out, lbl)
        des["soft_macro_f1"] = soft_macro_f1.item()
        des["macro_f1"] = macro_f1.item()
        loss = soft_macro_f1
    elif method == "PL":
        loss = criterion(out, lbl)
        des["loss"] = loss.item()
    if args.model.get("model_use", None) == "neural_collpase":
        return out, feature, loss, des
    else:
        return out, loss, des

def lr_decay(args, method, optimizer, model, epoch):


    lr_decay_method = args.training.get("lr_decay_method", False)
    if lr_decay_method:
        print("config file lr_decay_method:", lr_decay_method)
        if lr_decay_method == "lr_decay_cos": 
            lr_decay_cos(args, optimizer, epoch)
        if lr_decay_method == "lr_decay_fix_and_cos": 
            lr_decay_fix_and_cos(args, optimizer, epoch)
        if lr_decay_method == "lr_decay_warmup_and_fix_and_cos": 
            lr_decay_warmup_and_fix_and_cos(args, optimizer, epoch)
        if lr_decay_method == "BalancedSoftmax_lr_decay":
            args.training.get("lr_")
            BalancedSoftmax_lr_decay(args, optimizer, epoch)
        if lr_decay_method == "bcl_lr_decay": 
            bcl_lr_decay(args, optimizer, epoch)
        if lr_decay_method == "custom_lr_decay": 
            custom_lr_decay(args, optimizer, epoch)
        if lr_decay_method == "neural_collpase": 
            neural_collpase_lr_decay(args, optimizer, epoch)
        return 

    print("use default lr decay method.")

    if (method.startswith("MAUC") or method.startswith("AUC_mu")) and args.training.two_stage: # AUC方法
        if epoch + 1 == args.training.stage1_epoch + 1: # stage 2 使用AUC loss微调最后几层
            print("stage 2: fine-tuning last three layers by AUC loss")
            for p in optimizer.param_groups: # 二阶段学习率
                p["lr"] = args.training.stage2_lr
            if args.model.model_type == "alexnet": # 二阶段alexnet训练冻结features
                for param_name in model.state_dict().keys():
                    if not param_name.startswith("features"): #  or param_name.startswith("classifier.1")
                        model.state_dict()[param_name].requires_grad = False
            elif args.model.model_type == "resnet18":
                for param_name in model.state_dict().keys():
                    if not param_name.startswith("fc"): # 只训练3个fc
                        model.state_dict()[param_name].requires_grad = False
        elif epoch + 1 <= args.training.stage1_epoch: # 50 层以内使用CE，正常衰减
            if epoch >= args.training.lr_decay_epochs: # lr衰减
                for p in optimizer.param_groups:
                    p["lr"] *= args.training.lr_decay_rate
    # BCL原论文：
    # cifar10和cifar100: 200 epoch, 前10轮warm up到0.15, 160-180衰减系数为0.1
    elif method in ("BalancedSoftmax"):
        BalancedSoftmax_lr_decay(args, optimizer, epoch)
    elif method in ("BCL", "SCL"):
        lr_decay_warmup_and_fix_and_cos(args, optimizer, epoch)
    elif method in ("PL"):
        BalancedSoftmax_lr_decay(args, optimizer, epoch)
    else: # 非AUC方法, 不使用两阶段
        if epoch >= args.training.lr_decay_epochs: # lr衰减
            for p in optimizer.param_groups:
                p["lr"] *= args.training.lr_decay_rate

def bcl_lr_decay(args, optimizer, epoch):
    # if epoch < 8:
    #     lr = args.training.lr * (epoch + 1) / 8
    if epoch < 160:
        lr = args.training.lr
    elif epoch < 180:
        lr = args.training.lr * 0.1
    else:
        lr = args.training.lr * 0.01
    for p in optimizer.param_groups:
        p["lr"] = lr
    print("Learning rate: %.6f" % lr)

def lr_decay_cos(args, optimizer, epoch):
    # 直接余弦衰减
    lr = args.training.lr
    lr *= 0.5 * (1. + math.cos(math.pi * (epoch + 1) / args.training.epoch_num))
    for p in optimizer.param_groups:
        p["lr"] = lr
    print("Learning rate: %.6f" % lr)

def lr_decay_fix_and_cos(args, optimizer, epoch):
    # 一半epoch内: 不变
    # 大于一半epoch: cos
    lr = args.training.lr
    half_epoch_num = int(args.training.epoch_num/2)
    if epoch < half_epoch_num:
        pass
    else:
        lr *= cos_decay(epoch=epoch, begin=half_epoch_num, end=args.training.epoch_num)
    for p in optimizer.param_groups:
        p["lr"] = lr
    print("Learning rate: %.6f" % lr)

def lr_decay_warmup_and_fix_and_cos(args, optimizer, epoch):
    # 小于20: 从0线性增长到1倍lr
    # 大于20: 不变
    # 大于一半epoch: cos
    lr = args.training.lr
    half_epoch_num = int(args.training.epoch_num/2)
    if epoch < 20:
        lr = args.training.lr * (epoch + 1) / 20
    elif epoch < half_epoch_num:
        pass
    else:
        lr *= cos_decay(epoch=epoch, begin=half_epoch_num, end=args.training.epoch_num)
    for p in optimizer.param_groups:
        p["lr"] = lr
    print("Learning rate: %.6f" % lr)

def BalancedSoftmax_lr_decay(args, optimizer, epoch):
    # 小于linear_epoch: 从0.5倍lr线性增长到1倍lr
    # 大于linear_epoch: cos衰减到最后

    linear_epoch = args.training.lr_decay_params.linear_epoch
    lr = args.training.lr
    epoch_num = args.training.epoch_num
    assert linear_epoch < epoch_num
    if epoch < linear_epoch:
        lr = lr/2 + lr/2 * (epoch + 1) / linear_epoch
        # lr = lr * (epoch + 1) / linear_epoch
    else:
        lr *= cos_decay(epoch=epoch, begin=linear_epoch, end=epoch_num)
    print("Learning rate: %.6f" % lr)
    for p in optimizer.param_groups:
        p["lr"] = lr

def neural_collpase_lr_decay(args, optimizer, epoch):
    anneal_epochs = args.training.lr_decay_params.anneal_epochs # 退火的epoch，升序
    anneal_factor = args.training.lr_decay_params.anneal_factor # 退火的系数
    lr = args.training.lr
    for anneal_epoch in anneal_epochs:
        if epoch < anneal_epoch:
            continue
        else:
            lr *= anneal_factor
    print("Learning rate: %.6f" % lr)
    for p in optimizer.param_groups:
        p["lr"] = lr

def custom_lr_decay(args, optimizer, epoch):
    if epoch < 5:
        lr = args.training.lr * (epoch + 1) / 5
    if epoch < 60:
        lr = args.training.lr
    elif epoch < 80:
        lr = args.training.lr * 0.1
    else:
        lr = args.training.lr * 0.01
    for p in optimizer.param_groups:
        p["lr"] = lr
    print("Learning rate: %.6f" % lr)
