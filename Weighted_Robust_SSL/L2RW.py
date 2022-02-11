#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse, math, time, json, os
import higher
from collections import OrderedDict
from tensorboardX import SummaryWriter
from datetime import datetime

from lib2.meta_model import LeNet

from lib2 import transform
from config import config


parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PL", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="mnist_idx", type=str, help="dataset name : [svhn, cifar10, mnist]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="baseline", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.5,  type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default=100, type=int, help="number of labeled set")
parser.add_argument("--meta_lr", default=1000, type=float, help="learning rate for w")
parser.add_argument("--meta_val_batch", default=256, type=int, help="batch for meta val")
parser.add_argument("--w_clip", default=1e-5, type=float, help="batch for meta val")
parser.add_argument("--val_num", default=2000, type=int, help="val number")

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

l_train_dataset = dataset_cfg["dataset"](args.root,  "l_train")
u_train_dataset = dataset_cfg["dataset"](args.root,"u_train_fashion_ood_{}".format(args.ood_ratio))
val_dataset = dataset_cfg["dataset"](args.root, "val")
test_dataset = dataset_cfg["dataset"](args.root, "test")

# if args.mix == 1:
#     u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_mix_ood_{}".format(args.ood_ratio))

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
        l_train_dataset, 50, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * 64)
    )
else:
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * shared_cfg["batch_size"])
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, 50, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"][args.dataset] * 50)
)

val_loader_in_train = DataLoader(val_dataset, args.meta_val_batch, drop_last=True,
                                 sampler=RandomSampler(len(val_dataset), shared_cfg["iteration"][args.dataset] * args.meta_val_batch))

val_loader = DataLoader(val_dataset, 50, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 50, shuffle=False, drop_last=False)

l_loader_pretrain = DataLoader(
        l_train_dataset, 50, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), 50)
    )



print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

all_acc = []
for p in range(5):
    if args.em > 0:
        print("entropy minimization : {}".format(args.em))
        exp_name += "em_"
    condition["entropy_maximization"] = args.em

    model = LeNet().cuda()
    # model = weight_WRN(2, 10).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, 0.5)

    # trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
    # print("trainable parameters : {}".format(trainable_paramters))

    if args.alg == "VAT":  # virtual adversarial training
        from lib2.algs.vat import VAT3

        ssl_obj = VAT3(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
    elif args.alg == "PL":  # pseudo label
        from lib2.algs.pseudo_label import PL4

        ssl_obj = PL4(alg_cfg["threashold"])
    elif args.alg == "MT":  # mean teacher
        from lib2.algs.mean_teacher import MT2

        t_model =LeNet().cuda()
        # t_model = wrn.WRN(args, 2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = MT2(t_model, alg_cfg["ema_factor"])
    elif args.alg == "PI":  # PI Model
        from lib2.algs.pimodel import PiModel
        ssl_obj = PiModel()
    elif args.alg == "supervised":
        pass
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    path = os.path.join('TensorBoard', "Cluster_LENET",
                        "{}_{}_ood_{}_lr_{}".format(args.dataset, args.alg, args.ood_ratio, args.meta_lr),
                        TIMESTAMP)
    writer = SummaryWriter(path)

    u_label = u_train_dataset.dataset['labels']
    emd_label = np.load('output/emd/vgg19_kmean_20_{}.npy'.format(args.ood_ratio))
    iteration = -1
    maximum_val_acc = 0
    s = time.time()

    ood_result = []
    in_result = []
    iidx = int(len(u_train_dataset) * (1 - args.ood_ratio))
    for l_data, u_data, v_data in zip(l_loader, u_loader, val_loader_in_train):
        model.train()
        iteration += 1
        l_input, target, _ = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        u_input, dummy_target, idx = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
        val_input, val_target, _ = v_data
        val_input, val_target = val_input.to(device).float(), val_target.to(device).long()
        coef = 1

        weight = torch.tensor(np.zeros([len(u_input), 1]), dtype=torch.float32, device="cuda", requires_grad=True)
        opt_w = optim.Adam([weight], lr=args.meta_lr)

        with higher.innerloop_ctx(model, optimizer) as (meta_model, diffopt):
            outputs_hat = meta_model(u_input)
            ssl_loss_meta = ssl_obj(u_input, outputs_hat.detach(), meta_model) * weight
            cls_loss_meta = F.cross_entropy(meta_model(l_input), target, reduction="none", ignore_index=-1).mean()
            loss_meta = cls_loss_meta + ssl_loss_meta.mean() * coef
            meta_model.zero_grad()
            diffopt.step(loss_meta)
            val_loss_meta = F.cross_entropy(meta_model(val_input), val_target, reduction="none", ignore_index=-1).mean()
            opt_w.zero_grad()
            val_loss_meta.backward()
            opt_w.step()
        weight.data = torch.clamp(weight, min=0.0000000001, max=1)
        w = weight.clone().detach()
        #### training
        outputs_u = model(u_input)
        ssl_loss = (ssl_obj(u_input, outputs_u.detach(), model) * w).mean()
        cls_loss = F.cross_entropy(model(l_input), target, reduction="none", ignore_index=-1).mean()
        loss = cls_loss + ssl_loss * coef
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        writer.add_scalars('data/loss_group', {'loss': loss,
                                               'CE loss': cls_loss,
                                               'ssl_loss': ssl_loss}, iteration)
        w_in = w[(idx < iidx)].data.cpu().numpy().mean()
        w_ood = w[(idx >= iidx)].data.cpu().numpy().mean()
        ood_result.append(w_ood)
        in_result.append(w_in)

        if args.alg == "MT" or args.alg == "ICT":
            # parameter update with exponential moving average
            ssl_obj.moving_average(model.parameters())
        # display
        if iteration == 1 or (iteration % 100) == 0:
            wasted_time = time.time() - s
            rest = (shared_cfg["iteration"][args.dataset] - iteration) / 100 * wasted_time / 60
            print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, w_ood : {:.2f}, w_in : {:.2f}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
                iteration, shared_cfg["iteration"][args.dataset], cls_loss.item(), ssl_loss.item(), w_ood, w_in, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]))
            s = time.time()
        if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"][args.dataset]:
            with torch.no_grad():
                model.eval()
                # model.update_test_stats(False)
                print()
                print("### validation ###")
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(val_loader):
                    input, target, _ = data
                    input, target = input.to(device).float(), target.to(device).long()
                    w_l = torch.ones(len(input), device='cuda')
                    output = model(input)
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                acc = sum_acc / float(len(val_dataset))
                print("varidation accuracy : {}".format(acc))
                writer.add_scalar('data/val_acc', acc, iteration)
                # test
                if maximum_val_acc < acc:
                    print("### test ###")
                    maximum_val_acc = acc
                    sum_acc = 0.
                    s = time.time()
                    for j, data in enumerate(test_loader):
                        input, target, _ = data
                        input, target = input.to(device).float(), target.to(device).long()
                        w_l = torch.ones(len(input), device='cuda')
                        output = model(input)
                        pred_label = output.max(1)[1]
                        sum_acc += (pred_label == target).float().sum()
                    test_acc = sum_acc / float(len(test_dataset))
                    print("test accuracy : {}".format(test_acc))
                    writer.add_scalar('data/test_acc', test_acc, iteration)
                    # torch.save(model.state_dict(), path + 'best_model.pth')
            model.train()
            s = time.time()
    t_acc = np.around(test_acc.cpu().data.numpy(), 4)*100
    all_acc.append(t_acc)
    print("test acc : {}".format(test_acc))
    condition["test_acc"] = test_acc.item()
    writer.close()
    # np.save('output/weight/{}_L2RW_MNIST_ood_w_{}_{}'.format(args.alg, args.ood_ratio, p), ood_result)
    # np.save('output/weight/{}_L2RW_MNIST_in_w_{}_{}'.format(args.alg, args.ood_ratio, p), in_result)
print("test acc all: ", all_acc)
print("mean: ", np.mean(all_acc), "std: ", np.std(all_acc))
