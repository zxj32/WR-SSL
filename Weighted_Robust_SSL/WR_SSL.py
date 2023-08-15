#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
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

from lib2.meta_model import LeNet_B, LeNet_B_D
from lib2 import transform
from config import config
from plot import plot_w
import copy

# seed = 123
# torch.manual_seed(seed)
# np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="VAT", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="mnist_idx", type=str, help="dataset name : [svhn, cifar10, mnist]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="baseline", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.5, type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default='boundary', type=str, help="number of labeled set")
parser.add_argument("--w_norm", default=1, type=int, help="normal initial ")
parser.add_argument("--meta_lr", default=0.1, type=float, help="learning rate for w")
parser.add_argument("--meta_val_batch", default=256, type=int, help="batch for meta val")
parser.add_argument("--w_clip", default=1e-5, type=float, help="batch for meta val")
parser.add_argument("--val_num", default=5000, type=int, help="val data size")
parser.add_argument("--w_initial", default=1, type=float, help="0.01, 1")
## high order approximation
parser.add_argument("--Neumann", default=5, type=int, help="inverse Hession approximation")
parser.add_argument("--inner_loop_g", default=3, type=int, help="inner_loop_gradients steps")
## reduce overfitting
parser.add_argument("--L1_trade_off", default=0, type=float, help="0, 1e-08")
parser.add_argument("--cluster", default=0, type=int, help="cluster or not")
parser.add_argument("--K", default=20, type=int, help="number of cluster")
## speed up
parser.add_argument("--last", default=0, type=int, help="only update last layer")
parser.add_argument("--update_iter", default=1, type=int, help="update times")

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

l_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "l_train")
# u_train_dataset = dataset_cfg["dataset"](args.root,  args.n_label, "u_train")
u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_ood_{}".format(args.ood_ratio))
val_dataset = dataset_cfg["dataset"](args.root, args.n_label, "val_{}".format(args.val_num))
test_dataset = dataset_cfg["dataset"](args.root, args.n_label, "test")

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
        l_train_dataset, 64, drop_last=True,
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
    u_train_dataset, 256, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"][args.dataset] * 256)
)

val_loader_in_train = DataLoader(val_dataset, args.meta_val_batch, drop_last=True,
                                 sampler=RandomSampler(len(val_dataset), shared_cfg["iteration"][args.dataset] * args.meta_val_batch))

val_loader = DataLoader(val_dataset, 256, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 256, shuffle=False, drop_last=False)

l_loader_pretrain = DataLoader(
        l_train_dataset, 64, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), 50)
    )



print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

all_acc = []
for Q in range(5):

    if args.em > 0:
        print("entropy minimization : {}".format(args.em))
        exp_name += "em_"
    condition["entropy_maximization"] = args.em

    if args.alg == "PI":
        model = LeNet_B_D(10, 0.1).cuda()
    else:
        model = LeNet_B(10).cuda()
    optimizer = optim.Adam(model.params(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000, 0.5)

    trainable_paramters = sum([p.data.nelement() for p in model.params()])
    print("trainable parameters : {}".format(trainable_paramters))

    if args.alg == "VAT":  # virtual adversarial training
        from lib2.algs.vat import VAT5

        ssl_obj = VAT5(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
    elif args.alg == "PL":  # pseudo label
        from lib2.algs.pseudo_label import PL3

        ssl_obj = PL3(alg_cfg["threashold"])
    elif args.alg == "MT":  # mean teacher
        from lib2.algs.mean_teacher import MT3

        t_model = LeNet_B(10).cuda()
        # t_model = wrn.WRN(args, 2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = MT3(t_model, alg_cfg["ema_factor"])
    elif args.alg == "PI":  # PI Model
        from lib2.algs.pimodel import PiModel3

        ssl_obj = PiModel3()
    elif args.alg == "ICT":  # interpolation consistency training
        from lib2.algs.ict import ICT

        t_model = LeNet_B(10).cuda()
        # t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
        t_model.load_state_dict(model.state_dict())
        ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
    elif args.alg == "MM":  # MixMatch
        from lib2.algs.mixmatch import MixMatch

        ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
    elif args.alg == "supervised":
        pass
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    path = os.path.join('TensorBoard', "Cluster",
                        "{}_{}_labeled_{}_ood_{}_lr_{}".format(args.dataset, args.alg, args.n_label, args.ood_ratio, args.meta_lr),
                        TIMESTAMP)
    writer = SummaryWriter(path)
    writer.add_text('Text', str(args), 0)

    if args.ood_ratio == 0.5:
        emd_label = np.load('output/emd/vgg19_kmean_20.npy')
        ood_idx = [8, 14, 18, 16, 2, 11, 4, 19, 7, 0, 12, 3, 9]
        in_idx = [5, 6, 17, 10, 1, 13, 15]
    elif args.ood_ratio == 0.75:
        emd_label = np.load('output/emd/vgg19_kmean_20_{}.npy'.format(args.ood_ratio))
        ood_idx = [0, 1, 2, 3, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
        in_idx = [4, 5, 12, 8]
    elif args.ood_ratio == 0.25:
        emd_label = np.load('output/emd/vgg19_kmean_20_{}.npy'.format(args.ood_ratio))
        ood_idx = [2, 6, 7, 9, 13, 16, 17, 18]
        in_idx = [15, 19, 4, 10, 5, 8, 3, 14, 0, 11, 12, 1]
    u_label = u_train_dataset.dataset['labels']

    if args.cluster:
        weight = np.ones([20, 1])
    else:
        weight = np.ones([len(u_label), 1])
    weight = torch.tensor(weight * args.w_initial, dtype=torch.float32, device="cuda", requires_grad=True)
    opt_w = optim.Adam([weight], lr=args.meta_lr)

    u_label = u_train_dataset.dataset['labels']

    iteration = -1
    maximum_val_acc = 0
    s = time.time()

    ood_result = []
    in_result = []

    for l_data, u_data, v_data in zip(l_loader, u_loader, val_loader_in_train):
        model.train()
        iteration += 1
        l_input, target, _ = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        u_input, dummy_target, idx = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

        val_input, val_target, _ = v_data
        val_input, val_target = val_input.to(device).float(), val_target.to(device).long()
        w_l = torch.ones(len(l_input), device='cuda')
        w_v = torch.ones(len(val_input), device='cuda')

        if (iteration % args.update_iter) == 0:
            if args.cluster:
                w_u = weight[emd_label[idx]]
            else:
                w_u = weight[idx].squeeze()


            for h in range(args.inner_loop_g):
                ssl_loss = ssl_obj(u_input, model(u_input, w_u).detach(), model, w_u)
                cls_loss = F.cross_entropy(model(l_input, w_l), target, reduction="none", ignore_index=-1).mean()
                loss = cls_loss + ssl_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            val_loss_meta = F.cross_entropy(model(val_input, w_v), val_target, reduction="none", ignore_index=-1).mean()
            v1 = torch.autograd.grad(val_loss_meta, model.params(), retain_graph=True)
            f = torch.autograd.grad(loss, model.params(), retain_graph=True, create_graph=True)
            p = copy.deepcopy(v1)

            for j in range(args.Neumann):
                p = list(p)
                v1 = list(v1)
                temp1 = torch.autograd.grad(f, model.params(), retain_graph=True, grad_outputs=v1)
                temp1 = list(temp1)
                for k in range(len(v1)):
                    v1[k] -= alg_cfg["lr"] * temp1[k]
                for k in range(len(v1)):
                    p[k] -= v1[k]
            p = tuple(p)
            v = torch.autograd.grad(loss, model.params(), retain_graph=True, create_graph=True)
            d_lambda = torch.autograd.grad(v, weight, create_graph=True, grad_outputs=p)
            # print(d_lambda)
            with torch.no_grad():
                weight += args.meta_lr * d_lambda[0]
            weight.data = torch.clamp(weight, min=0.0000000001, max=1)

        if args.cluster:
            w_up = weight[emd_label[idx]]
        else:
            w_up = weight[idx].squeeze()

        w = w_up.clone().detach()
        #### training
        ssl_loss = ssl_obj(u_input, model(u_input, w).detach(), model, w)
        cls_loss = F.cross_entropy(model(l_input, w_l), target, reduction="none", ignore_index=-1).mean()
        loss = cls_loss + ssl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_ood = weight[ood_idx].data.cpu().numpy().mean()
        w_in = weight[in_idx].data.cpu().numpy().mean()
        ood_result.append(w_ood)
        in_result.append(w_in)
        writer.add_scalars('data/weight_group', {'w ood': w_ood,
                                                 'w in': w_in}, iteration)
        # writer.add_scalars('data/gradient_group', {'min_g': grads_w.min(),
        #                                            'max_g': grads_w.max()}, iteration)
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
            ww = np.zeros_like(emd_label, dtype=float)
            for k in range(len(emd_label)):
                w_weight = weight.data.cpu().numpy()
                ww[k] = w_weight[emd_label[k]]
            plot_w(ww, u_label, path + 'w_{}'.format(iteration), iteration)
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
                    output = model(input, w_l)
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
                        output = model(input, w_l)
                        pred_label = output.max(1)[1]
                        sum_acc += (pred_label == target).float().sum()
                    test_acc = sum_acc / float(len(test_dataset))
                    print("test accuracy : {}".format(test_acc))
                    writer.add_scalar('data/test_acc', test_acc, iteration)
                    torch.save(model.state_dict(), path + 'best_model.pth')
            model.train()
            s = time.time()

    t_acc = np.around(test_acc.cpu().data.numpy(), 4)*100
    all_acc.append(t_acc)
    print("test acc : {}".format(test_acc))
    condition["test_acc"] = test_acc.item()
    writer.close()

print("test acc all: ", all_acc)
print("mean: ", np.mean(all_acc), "std: ", np.std(all_acc))
