import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import random
from docopt import docopt
import timeit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim

import sys
import os
sys.path.append('..')

from models import deeplab_resnet
from dataloader.datasets import DAVIS2016, YTB_VOS, ECSSD, MSRA10K
from tools.loss import cross_entropy_loss_weighted, cross_entropy_loss
from evaluation.evaluate import test_model
from tools.utils import vis

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#davis_path = '/home/hakjine/datasets/DAVIS/DAVIS-2016/DAVIS'
davis_path = '/data/hakjin-workspace/DAVIS/DAVIS-2016/DAVIS'
davis_im_path = os.path.join(davis_path, 'JPEGImages/480p')
davis_gt_path = os.path.join(davis_path, 'Annotations/480p')
#vos_path = '/home/hakjine/datasets/Youtube-VOS/train/'
vos_path = '/data/hakjin-workspace/Youtube-VOS/'
vos_im_path  = os.path.join(vos_path, 'JPEGImages')
vos_gt_path  = os.path.join(vos_path, 'Annotations')
#ECSSD_path = '../data/ECSSD'
ECSSD_path= '/data/hakjin-workspace/ECSSD/'
MSRA10K_path = '/data/hakjin-workspace/MSRA10K/'
#MSRA10K_path = '../data/MSRA10K'

start = timeit.timeit()
docstr = """Train ResNet-DeepLab on segmentation datasets in pytorch using VOC12 pretrained initialization 

Usage: 
    train.py [options]

Options:
    --NoLabels=<int>            The number of labels in training data, foreground and background  [default: 2]
    --lr=<float>                Learning Rate [default: 0.001]
    -b, --batchSize=<int>       Num sample per batch [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu=<int>                 GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 30000]
"""

args = docopt(docstr, version='v0.9')
print(args)

gpu_num = int(args['--gpu'])
num_labels = int(args['--NoLabels'])
#torch.cuda.set_device(gpu_num)
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_num)


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.branch.parameters())
    b.append(model.fuse.parameters())
    b.append(model.predict.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


if not os.path.exists('../data/snapshots'):
    os.makedirs('../data/snapshots')


model = deeplab_resnet.Res_Deeplab_4chan(num_labels)

#saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
#saved_state_dict = torch.load('../data/MS_DeepLab_resnet_trained_VOC.pth')
saved_state_dict = torch.load('/data/hakjin-workspace/MS_DeepLab_resnet_trained_VOC.pth')
for i in saved_state_dict:
    i_parts = i.split('.')
    #if i_parts[1]=='layer5':
    #    saved_state_dict[i] = model.state_dict()[i]
    #if i_parts[1] == 'conv1':
    #    saved_state_dict[i] = torch.cat((saved_state_dict[i], torch.FloatTensor(64, 1, 7, 7).normal_(0,0.0001)), 1)
model_dict = model.state_dict()
saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
model_dict.update(saved_state_dict)

model.load_state_dict(model_dict)

"""
saved_state_dict = torch.load('../data/snapshots/box-12000.pth')
model.load_state_dict(saved_state_dict)
"""


max_iter = int(args['--maxIter']) 
batch_size = int(args['--batchSize'])
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.cuda()
#model.eval() # use_global_stats = True


db_davis_train = DAVIS2016(train=True,root=davis_path, aug=True)
db_ytb_train = YTB_VOS(train=True, root=vos_path, aug=True)
db_ECSSD = ECSSD(root=ECSSD_path, aug=True)
db_MSRA10K = MSRA10K(root=MSRA10K_path, aug=True)
db_train = ConcatDataset([db_davis_train, db_ytb_train, db_ECSSD, db_MSRA10K])

train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)

optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
optimizer.zero_grad()


losses = []
acc = []
numerics = {'loss':losses, 'acc': acc}
import json
iter = 0
save_path = '../data/snapshots'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
for epoch in range(0, 20):
    for ii, sample in enumerate(train_loader):
        iter += 1
        images, mask, label = sample[0], sample[1], sample[2]
        images = images.cuda()
        mask = mask.cuda()

        out = model(images, mask)
        loss = cross_entropy_loss_weighted(out, label.cuda())
        numerics['loss'].append(float(loss.data.cpu().numpy()))
        loss.backward()

        if iter %1 == 0:
            print('iter = ',iter, 'of',max_iter,'completed, loss = ', (loss.data.cpu().numpy()))

    
        #if iter % 5 == 0:
        #    vis(images[0], mask[0], label[0], out[0])

        optimizer.step()
        lr_ = lr_poly(base_lr,iter,max_iter,0.9)
        print('(poly lr policy) learning rate',lr_)
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        optimizer.zero_grad()

        if iter == 10000:
            lr_ *= 10

        if iter == 20000:
            lr_ *= 50

        if iter % 1000 == 0 and iter!=0:
            print('taking snapshot ...')
            iou = test_model(model, save=True)
            numerics['acc'].append(iou)
            torch.save(model.state_dict(),'../data/snapshots/box-'+str(iter)+'.pth')
            with open('../data/losses/box-'+str(iter)+'.json', 'w') as f:
                json.dump(numerics, f)

        if iter == max_iter:
            break

end = timeit.timeit()
print('time taken ', end-start)
