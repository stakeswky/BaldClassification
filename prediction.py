# -*- coding: utf-8 -*
import os
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from albumentations.pytorch.functional import img_to_tensor
from flyai.framework import FlyAI
from torch.autograd import Variable
import torchvision.transforms as transforms
from path import MODEL_PATH, DATA_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        # pass
        # print(MODEL_PATH+'/'+'best.pth')
        model = torch.load(MODEL_PATH + '/' + "best.pth")
        self.model = model.to(device)
        
    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/cloudy/00000.jpg"}
        :return: 模型预测成功中户 {"label": 0}
        '''
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        #print(image_path)
        normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(240),
        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ])
        img = Image.open(image_path).convert('RGB')
        img=transform_test(img)
        model = self.model
        model.eval()
        with torch.no_grad():
           img=img.unsqueeze(0)
           bs, ncrops, c, h, w = img.size()                        
           img_ = img.view(-1, c, h, w)
           img=Variable(img_.cuda())
           out=model(img)
           out = out.view(bs, ncrops, -1).mean(1)
           _,predicted=torch.max(out.data,1)
           pred=predicted.data.cpu().numpy().tolist()
        pred=" ".join('%s' %id for id in pred)
        pred=int(pred)
        return {"label": pred}