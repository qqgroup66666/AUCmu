import torch
from numpy import sin, cos
import numpy as np
from collections import deque


if __name__ == "__main__":

    class_num = 100
    feature_num = 256

    labels = torch.arange(class_num)
    feature = torch.nn.Linear(class_num, feature_num)
    classifier = torch.nn.Linear(feature_num, class_num)
    factor = torch.eye(class_num, class_num)
    softmax = torch.nn.Softmax(dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [
            {"params": feature.parameters(), "lr":1e-2, "momentum":0.9},
            {"params": classifier.parameters(), "lr":1e-2, "momentum":0.9},
        ]
    )

    c = classifier.weight
    f = feature.weight
    print("classifier: ", c.shape)
    print("feature: ", f.shape)

    aa = 1

    f_ = None
    def animate(i):
        global f_
        if aa == 1:
            c_ = torch.nn.functional.normalize(c, p=2, dim=1)
            f_ = torch.nn.functional.normalize(f, p=2, dim=0)
            
            pred_ = torch.mm(c_, f_)
            loss = loss_fn(pred_, labels)
            feature.zero_grad()
            classifier.zero_grad()
            import math
            # print("feature mean: ", torch.mean(f_, dim=1))
            print("(pi) theta mean: ", math.acos(torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item()) * 180 / math.pi)
            print("theta mean: ", torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
            print("theta std: ", torch.std(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
        elif aa == 2:
            pred = torch.mm(c , f)
            loss = loss_fn(pred, labels)
            print("classifier norm: ", torch.mean(torch.norm(c, p=2, dim=1)).item())
            print("feature norm: ", torch.mean(torch.norm(f, p=2, dim=0)).item())
        loss.backward()
        optimizer.step()

        print("loss: ", loss.item())
        print()


    for i in range(1000):
        animate(i)

    torch.save(f_, "f{}-c{}_feature".format(feature_num, class_num))
