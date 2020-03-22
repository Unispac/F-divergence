import numpy as np
import torch
from torch import nn
from config import *
import torch.nn.functional as F
import sys
import pickle
import resnet
import random
import argparse
import time
from util import get_training_data
from util import get_testing_data
from util import get_normalize_layer
from torch.optim.lr_scheduler import StepLR


################# test ###################################
def test(myNet,test_data,noise_sd,batch_size=256):
    positive = 0
    n_item = 0
    myNet.eval()
    for (x,gt) in test_data:
        x = (x + torch.randn_like(x) * noise_sd).to(device)
        gt = gt.to(device)
        y = myNet(x)
        y = torch.argmax(y,dim=1)
        this_batch_size = len(y)
        for i in range(this_batch_size):
            if y[i]==gt[i]:
                positive+=1
            n_item+=1
    return positive/n_item
#############################################################

################# train ###################################
def train(myNet,train_data,test_data,args): 
    
    ######### Optimizer ######################################
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    opt_SGD= torch.optim.SGD(myNet.parameters(), learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    
    lr_scheduler = StepLR(opt_SGD, step_size=args.lr_step_size, gamma=args.gamma)

    loss_f = F.cross_entropy
    ##########################################################
    
    n_batch = len(train_data)
    n_epoch = args.epochs

    for i in range(n_epoch):
        lr_scheduler.step(i)
        myNet.train()
        time_st = time.time()
        for j,(x,gt) in enumerate(train_data):
            x = (x + torch.randn_like(x) * args.noise_sd).to(device)
            gt = gt.to(device)
            y = myNet(x)
            loss = loss_f(y,gt)
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            sys.stdout.write("\r Training : [Epoch] : %d / %d , [Iter] : %d / %d, [Loss] : %f" % (i+1,n_epoch,j+1,n_batch,loss))
            sys.stdout.flush()
        print("\n Time : %f \n" % (time.time()-time_st))
        lr_scheduler.step()
        ac = test(myNet,test_data,noise_sd=args.noise_sd,batch_size=args.batch)
        print("\n Accuracy : %f \n" % ac)
        with open("logs.txt",'a+') as log :
            log.write("\n [Epoch] : %d / %d , [Accuracy] : %f \n" % (i+1,n_epoch,ac))
        
#############################################################



if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage='Need Help ? ',description=' Help Info .')
    parser.add_argument('--arch', type=str, default="resnet-110",choices=ARCHITECTURES)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N',
                        help='batchsize (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--noise_sd', default=0.0, type=float,
                        help="standard deviation of Gaussian noise for data augmentation")
    
    args = parser.parse_args()


    print("reading data ... ")
    train_loader = get_training_data(batch_size=args.batch)
    test_loader = get_testing_data(batch_size=args.batch)

    print("initializing model ... ")
    print(args.arch)
    if args.arch == 'resnet-110':
        myNet = resnet.resnet(depth=110,num_classes=10)
        data_normalizer = get_normalize_layer('cifar10')
        myNet = torch.nn.Sequential(data_normalizer,myNet)
    else:
        print("[Error] : Invalid Architecture")
        exit(0)
    
    print("device : ",device)
    myNet = torch.nn.DataParallel(myNet)
    myNet.to(device)

    
    print("noise - standard deviation : ",args.noise_sd)

    print("start to train ...")
    train(myNet,train_data=train_loader,test_data=test_loader,args=args)

    model_name = "cifar_resnet_noise="+str(args.noise_sd)+".pytorch"
    print("[Done]...")
    torch.save(myNet.module.state_dict(), "./checkpoint/"+model_name)
