# WR-SSL
This is a PyTorch implementation of the uncertainty-GNN model as described in our paper:
 
Xujiang Zhao, Killamsetty Krishnateja, Rishabh Iyer, Feng Chen. [How Out-of-Distribution Data Hurts Semi-Supervised Learning], ICDM 2022 

----------------------------------------------------------------------------------------
We first take a look at the structure of our code and datasets. The folder structure as following:

	+Weighted_Robust_SSL
	     +data
                +twomoons
                +mnist_100
                +cifar10_class6
	     lib2
                +algs
		   mean_teacher.py
		   vat.py
		   pimodel.py
		   pseudo_label.py
                +datasets
		   cifar10.py
		   mnist.py
		   synthetic.py
                transform.py
                meta_model.py
             WR_SSL.py
	     L2RW.py
	     train_DS3L.py
	     train_MWN.py
	     train_baseline_ssl.py
	     build_dataset.py
	     build_ood_data.py
	     config.py
             weight_batch_norm.py

    1.  Weighted Robust_SSL_Framework: It contains the implementation of our methods: WR-SSL and baseline: VAT, MT, PL, PI, L2RW-SSL, WMN-SSL, DS3L.
    
    2.  dataset: We consider four datasets: twomoons[1], MNIST, FashionMNIST[2], CIFAR10

Our code is written by Python3.6.10 We assume your Operating System is GNU/Linux-based. However, if you have MacOS or MacBook, it will be okay. If you use Windows, we recommend that you use PyCharm to run our code. The dependencies of our programs is Python3.6.10.

Notice: Since our code depends on Python3.6.10, for people who are not familiar with GNU/Linux environment, it may be difficult to run our code. However, we will provide our pip command if the paper got accepted.

----------------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. It has two steps:

    1.  install Python3.6.10
    
    2.  install numpy(1.15.2), Pytorch(1.4.0), torchvision(0.5.0), tensorboardX(2.0) scikit-learn(0.22.2), scipy(1.4.1).
    
    (Notice: Our code is based on open-source Pytorch implementation: github.com/perrying/realistic-ssl-evaluation-pytorch)

After set up above 2 steps, you are ready to run our code. For example, to generate the original dataset (save in "data/"), please try to run:

    python build_dataset.py -d=mnist -n=100
    
    python build_dataset.py -d=fashion_mnist -n=100
    
    python build_dataset.py -d=cifar10 -n=4000

to add OODs in unlabeled set (save in "data/mnist/"), please try to run:

    python build_ood_data.py

to run our method (--Neumann: inverse Hession approximation (P), 'inner_loop_g' inner loop gradients steps):

    python WR_SSL.py  --alg=VAT --ood_ratio=0.5 --Neumann=5 --inner_loop_g=3 --L1_trade_off=1e-08 

ro tun baseline

    python train_baseline_ssl.py --alg=PL --ood_ratio=0.5          (for traditional SSL: VAT, PI, PL, MT)
    
    python train_DS3L.py --alg=PL --ood_ratio=0.5	           (for robust SSL: DS3L-SSL)
    
    python train_MWN.py --alg=PL --ood_ratio=0.5		   (for robust SSL: MWN-SSL)
    
    python train_L2RW.py --alg=PL --ood_ratio=0.5		   (for robust SSL: L2RW-SSL)

If there is no error display, then we are done for this section.

----------------------------------------------------------------------------------------


References:

[1] https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

[2] https://github.com/zalandoresearch/fashion-mnist

----------------------------------------------------------------------------------------
