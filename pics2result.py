import torch
#from d2l import torch as d2l
from torchvision import transforms
import os
import torch.nn.functional as F
#from DataUtil import *
from PIL import Image

import sys

sys.path.append('model/')
# DenseNet GooLeNet ResNet
# 使用pytorch框架加载保存好的模型，已经训练好的模型放在OneDrive中

#net1 = torch.load('model/densenet_model.pt')
net1 = torch.load('model/densenet_model.pt', map_location='cpu')
#net2 = torch.load('model/goolenet_model.pt')
net2 = torch.load('model/goolenet_model.pt', map_location='cpu')
#net3 = torch.load('model/resnet_model.pt')
net3 = torch.load('model/resnet_model.pt', map_location='cpu')
#net = torch.load("model/stacking.pt")
net = torch.load("model/stacking.pt", map_location='cpu')


def predict(img_path):
    trans = [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    trans = transforms.Compose(trans)

    img_list = os.listdir(img_path)
    predict_result = [0, 0, 0, 0]
    for img in img_list:
        path = os.path.join(img_path, img)
        # print(path)
        img = Image.open(path).convert('RGB')
        img = trans(img)
        img = img.unsqueeze(0)
        #img = img.to(d2l.try_gpu())

        net1.eval()
        net2.eval()
        net3.eval()
        net.eval()
        with torch.no_grad():
            outputs1 = net1(img)
            outputs2 = net2(img)
            outputs3 = net3(img)
        _, predict1 = torch.max(outputs1, 1)
        _, predict2 = torch.max(outputs2, 1)
        _, predict3 = torch.max(outputs3, 1)
        res = torch.tensor([predict1[0].item(), predict2[0].item(), predict3[0].item()])
        res = res.float()
        res = res.unsqueeze(0)
        #res = res.to(d2l.try_gpu())
        with torch.no_grad():
            outputs = net(res)
        outputs = F.softmax(outputs, dim=1)
        values, predict = torch.max(outputs, 1)
        predict_result[predict[0].item()] += 1
        print(f'\r{predict_result}', end='')
    print(f'\r                                   \r', end='')
    return {"IPMN": predict_result[0], "NET": predict_result[1], "PDAC": predict_result[2], "SCN": predict_result[3]}


rootDir = 'intermediate/pics/'


def inp(item):  # A B C
    try:
        img_path = os.path.join(rootDir, f"{item}/")
        result = predict(img_path)
        return {"status": True, "message": result}
    except Exception as e:
        return {"status": False, "message": e}


if __name__ == "__main__":
    print("pics->result")
    for item in os.listdir(rootDir):
        inp(item)