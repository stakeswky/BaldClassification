# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import ImageData
from efficientnet_pytorch import EfficientNet
from path import MODEL_PATH, DataID, DATA_PATH
from flyai.utils.log_helper import train_log
import numpy as np
from scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=20, type=int,
                    help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("-lr", "--LR", default=0.05, type=float,
                    help="learning rate")

args = parser.parse_args()

model_path = os.path.join(MODEL_PATH, args.EXP)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')




class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        
        data_helper = DataHelper()
        data_helper.download_from_ids(DataID)

    def deal_with_data(self):

        # pass

        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(240),
                transforms.RandomHorizontalFlip(),
			        transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
        ])
        val_trainsform = transforms.Compose([
            transforms.Resize((240,240)),
	        transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        return train_transform, val_trainsform

    def cross_entropy_loss(self,preds, target, reduction):
        logp = F.log_softmax(preds, dim=1)
        loss = torch.sum(-logp * target, dim=1)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
        
    def onehot_encoding(self,labels, n_classes):
        return torch.zeros(labels.size(0), n_classes).to(labels.device).scatter_(
            dim=1, index=labels.view(-1, 1), value=1)
    
    def label_smoothing(self,preds, targets,epsilon=0.1):

        n_classes = preds.size(1)
        device = preds.device
    
        onehot = self.onehot_encoding(targets, n_classes).float().to(device)
        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
        device) * epsilon / n_classes
        loss = self.cross_entropy_loss(preds, targets, reduction="mean")
        return loss
    
    
    
    
    
    
    
    
    
    def train(self):

        # pass
        df = pd.read_csv(os.path.join(DATA_PATH, DataID, 'train.csv'))
        image_path_list = df['image_path'].values
        label_list = df['label'].values

        # 划分训练集和校验集
        all_size = len(image_path_list)
        train_size = int(all_size * 0.9)
        train_image_path_list = image_path_list[:train_size]
        train_label_list = label_list[:train_size]
        val_image_path_list = image_path_list[train_size:]
        val_label_list = label_list[train_size:]
        print(
            'train_size: %d, val_size: %d' % (len(train_image_path_list),
                                              len(val_image_path_list)))
        train_transform, val_trainsform = self.deal_with_data()
        train_data = ImageData(train_image_path_list, train_label_list,
                               train_transform)
        val_data = ImageData(val_image_path_list, val_label_list,
                             val_trainsform)
        train_loader = DataLoader(train_data, batch_size=args.BATCH,
                                  num_workers=0, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.BATCH,
                                num_workers=0, shuffle=False)
        model = EfficientNet.from_pretrained('efficientnet-b1')
        
        model.fc = nn.Linear(1280, 2)
        if use_gpu:
            model.to(DEVICE)
        criteration = nn.CrossEntropyLoss()
        criteration.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.LR,
                                    momentum=0.9, weight_decay=5e-4)

        if args.SCHE == "cos":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=5,
                                                             eta_min=4e-08)
        elif args.SCHE == "red":
             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1,
               patience=3, verbose=False, threshold=0.0001
            )
        else:
            sys.exit(-1)
        max_correct = 0
        
        #scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
        
        for epoch in range(args.EPOCHS):
           
            #scheduler_warmup.step(epoch)
            model.train()
            correct = 0
            # Train losses
            train_losses = []
            for img, label in train_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                output = model(img)
                #loss = criteration(output, label)
                loss = self.label_smoothing(output, label,epsilon=0.1)
                loss.backward()
                optimizer.step()
                # Train Metric
                train_pred = output.detach().cpu().max(1, keepdim=True)[1]
                correct += train_pred.eq(label.detach().cpu().
                                         view_as(train_pred)).sum().item()
                train_losses.append(loss.item())
                del train_pred
                # print("Epoch {}, Loss {:.4f}".format(epoch, loss.item()))
            del img, label

            #  Train loss curve
            train_avg_loss = np.mean(train_losses)
            
            acc = 100 * correct / len(train_image_path_list)
            
            scheduler_warmup.step_ReduceLROnPlateau(train_avg_loss)


            if epoch % 1 == 0 or epoch == args.EPOCHS - 1:
                correct = 0
                with torch.no_grad():
                    model.eval()
                    # Val losses
                    val_losses = []
                    for val_img, val_label in val_loader:
                        val_img = val_img.to(DEVICE),
                        val_label = val_label.to(DEVICE)
                        val_output = model(val_img[0])
                        loss = criteration(val_output, val_label)
                        val_pred = val_output.detach().cpu().\
                            max(1, keepdim=True)[1]
                        correct += val_pred.eq(val_label.detach().cpu().
                                               view_as(val_pred)).\
                            sum().item()
                        val_losses.append(loss.item())
                        del val_img, val_label, val_output, val_pred

                #  Val loss curve
                val_avg_loss = np.mean(val_losses)
                
                val_acc = 100 * correct / len(val_image_path_list)
                

                if (correct > max_correct):
                    max_correct = correct
                    torch.save(model, MODEL_PATH + '/' + "best.pth")
                print("Epoch {},  Accuracy {:.0f}%".format(
                    epoch, 100 * correct / len(val_image_path_list)))
                
                
               # LR curve
                
            train_log(train_loss=train_avg_loss, train_acc=acc, val_loss=val_avg_loss,val_acc=val_acc)

        


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()