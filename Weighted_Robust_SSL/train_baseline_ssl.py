#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "9"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse, math, time, json, os
from collections import OrderedDict
from tensorboardX import SummaryWriter
from datetime import datetime

from lib2 import transform
from config import config
from lib2.meta_model import LeNet_B2, LeNet_B2_D 

# seed = 123
# torch.manual_seed(seed)
# np.random.seed(seed)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="VAT", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="mnist_idx", type=str, help="dataset name : [svhn, cifar10, mnist]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="baseline", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.0, type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default=100, type=int, help="number of labeled set")
parser.add_argument("--mix", default=0, type=int, help="mix ood")
parser.add_argument("--less", default=0, type=int, help="remove ood in unlabeled set")

args = parser.parse_args()

print(args)

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

condition = {}
exp_name = ""

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"])  # transform function (flip, crop, noise)

l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
if args.less == 1:
    u_train_dataset = dataset_cfg["dataset"](args.root,  "u_train_fashion_less_{}".format(args.ood_ratio))
else:
    u_train_dataset = dataset_cfg["dataset"](args.root, "u_train_fashion_ood_{}".format(args.ood_ratio))
val_dataset = dataset_cfg["dataset"](args.root, "val")
test_dataset = dataset_cfg["dataset"](args.root,  "test")

if args.mix == 1:
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_mix_ood_{}".format(args.ood_ratio))

print("labeled data : {}, unlabeled data : {}, training data : {}, OOD rario : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset) + len(u_train_dataset), args.ood_ratio))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {"OOD ratio": args.ood_ratio,
                               "labeled": len(l_train_dataset), "unlabeled": len(u_train_dataset),
                               "validation": len(val_dataset), "test": len(test_dataset)
                               }


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """

    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


shared_cfg = config["shared"]
if args.alg != "supervised":
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, 64, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * 64)
    )
else:
    l_loader = DataLoader(
        l_train_dataset, 64, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * 64)
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, 256, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"][args.dataset] * 256)
)

val_loader = DataLoader(val_dataset, 256, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 256, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

all_acc = []
for p in range(10):

    if args.em > 0:
        print("entropy minimization : {}".format(args.em))
        exp_name += "em_"
    condition["entropy_maximization"] = args.em
    if args.alg == "PI":
        model = LeNet_B2_D(10, 0.5).cuda()
    else:
        model = LeNet_B2(10).cuda()
    optimizer = optim.Adam(model.params(), lr=alg_cfg["lr"])

    trainable_paramters = sum([p.data.nelement() for p in model.params()])
    print("trainable parameters : {}".format(trainable_paramters))

    if args.alg == "VAT":  # virtual adversarial training
        from lib2.algs.vat import VAT3

        ssl_obj = VAT3(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
    elif args.alg == "PL":  # pseudo label
        from lib2.algs.pseudo_label import PL2

        ssl_obj = PL2(alg_cfg["threashold"])
    elif args.alg == "MT":  # mean teacher
        from lib2.algs.mean_teacher import MT2

        t_model = LeNet_B2(10).cuda()
        # t_model = wrn.WRN(args, 2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = MT2(t_model, alg_cfg["ema_factor"])
    elif args.alg == "PI":  # PI Model
        from lib2.algs.pimodel import PiModel2
        ssl_obj = PiModel2()
    elif args.alg == "supervised":
        pass
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))

    path = os.path.join('TensorBoard', "baseline",
                        "LeNet_{}_{}_ood_{}_less_{}".format(args.dataset, args.alg, args.ood_ratio, args.less),
                        TIMESTAMP)
    writer = SummaryWriter(path)
    writer.add_text('Text', str(args), 0)
    u_label = u_train_dataset.dataset['labels']

    iteration = 0
    maximum_val_acc = 0
    s = time.time()
    for l_data, u_data in zip(l_loader, u_loader):
        model.train()
        iteration += 1
        l_input, target, _ = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        u_input, dummy_target, idx = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
        ood_mask = (dummy_target == -1).float() * (-1) + 1
        # coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration / 20000, 1)) ** 2)
        if args.alg != "supervised":  # for ssl algorithm
            coef = 1
            ssl_loss = ssl_obj(u_input, model(u_input).detach(), model) * coef
            ssl_loss = torch.mean(ssl_loss * 1)
        else:
            coef = 0
            ssl_loss = torch.zeros(1).to(device)

        cls_loss = F.cross_entropy(model(l_input), target, reduction="none", ignore_index=-1).mean()

        loss = cls_loss + ssl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalars('data/loss_group', {'loss': loss,
                                               'CE loss': cls_loss,
                                               'ssl_loss': ssl_loss}, iteration)

        if args.alg == "MT" or args.alg == "ICT":
            # parameter update with exponential moving average
            ssl_obj.moving_average(model.parameters())
        # display
        if iteration == 1 or (iteration % 100) == 0:
            wasted_time = time.time() - s
            rest = (shared_cfg["iteration"][args.dataset] - iteration) / 100 * wasted_time / 60
            print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.4f}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
                iteration, shared_cfg["iteration"][args.dataset], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]))
            s = time.time()
        # validation
        if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"][args.dataset]:
            with torch.no_grad():
                model.eval()
                print()
                print("### validation ###")
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(val_loader):
                    input, target, _ = data
                    input, target = input.to(device).float(), target.to(device).long()

                    output = model(input)

                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                acc = sum_acc / float(len(val_dataset))
                print("varidation accuracy : {}".format(acc))
                writer.add_scalar('data/val_loss', acc, iteration)
                # test
                if maximum_val_acc < acc:
                    print("### test ###")
                    maximum_val_acc = acc
                    sum_acc = 0.
                    s = time.time()
                    for j, data in enumerate(test_loader):
                        input, target, _ = data
                        input, target = input.to(device).float(), target.to(device).long()
                        output = model(input)
                        pred_label = output.max(1)[1]
                        sum_acc += (pred_label == target).float().sum()
                    test_acc = sum_acc / float(len(test_dataset))
                    print("test accuracy : {}".format(test_acc))
                    writer.add_scalar('data/test_acc', test_acc, iteration)
                    torch.save(model.state_dict(), path + 'best_model.pth')
            model.train()
            s = time.time()
        # lr decay
        if iteration == shared_cfg["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]
    all_acc.append(test_acc.cpu().data.numpy())
    print("test acc : {}".format(test_acc))
    condition["test_acc"] = test_acc.item()
    writer.close()
print("test acc all: ", all_acc)
print("mean: ", np.mean(all_acc), "std: ", np.std(all_acc))
