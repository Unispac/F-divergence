import torch
import os
from config import *
import resnet
from util import get_testing_data
from util import get_normalize_layer
import argparse
from core import Smooth

def test(test_data,batch_size,model,sample_num,enable_abstain=False,alpha=None):
    n_points = 0
    n_positive = 0
    n_tot_points = len(test_data)
    for j,(x,gt) in enumerate(test_data):
        n_points+=1
        top = model.predict(x,sample_num,batch_size,enable_abstain,alpha)
        if top == gt:
            n_positive+=1
        print("testing : %d/%d , ACC : %f"%(n_points,n_tot_points,n_positive/n_points))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='Need Help ? ',description=' Help Info .')
    parser.add_argument('--arch', type=str, default="resnet-110",choices=ARCHITECTURES)
    parser.add_argument('--batch', default=1000, type=int, metavar='N',
                        help='batchsize (default: 1000)')
    parser.add_argument('--noise_sd', required=True, type=float,
                        help="standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--model_path',required=True,type=str,help="file path of model checkpoint")
    parser.add_argument('--sample_num',default=1000,type=int,help="how many samples for prediction")
    parser.add_argument('--enable_abstain',action='store_true',help='whether abstain is allowed')
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    args = parser.parse_args()

    ############ data init ###################
    print("reading data ... ")
    test_loader = get_testing_data(batch_size=1) # each time, we perform sampling for one point.
    ##########################################

    ############ model init ##################
    print("initializing model ... ")
    print("arch : ",args.arch)
    if args.arch == 'resnet-110':
        myNet = resnet.resnet(depth=110,num_classes=10)
        data_normalizer = get_normalize_layer('cifar10')
        myNet = torch.nn.Sequential(data_normalizer,myNet)
    else:
        print("[Error] : Invalid Architecture")
        exit(0)
    print("checkpoint : ",args.model_path)
    model_dict = torch.load(args.model_path)
    myNet.load_state_dict(model_dict)
    print("device : ",device)
    myNet = torch.nn.DataParallel(myNet)
    myNet.to(device)

    print("noise level : ",args.noise_sd)
    print("initializing smooth model ...")
    myNet = Smooth(base_classifier=myNet,num_classes=n_class,sigma=args.noise_sd)
    ############################################

    ########## test ##############
    test(test_data=test_loader,batch_size=args.batch,model=myNet,\
        sample_num=args.sample_num,enable_abstain=args.enable_abstain,alpha=args.alpha)
    ##############################

