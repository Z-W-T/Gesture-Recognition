import cv2
import os
import numpy as np
from PIL import Image
import json
import dataset.HaGRID.DataSet
import torch
import torchvision
from torchvision import transforms as T
root = "./dataset/HaGRID/"
targets = {
    1: "call",
    2: "dislike",
    3: "fist",
    4: "four",
    5: "like",
    6: "mute",
    7: "ok",
    8: "one",
    9: "palm",
    10: "peace",
    11: "rock",
    12: "stop",
    13: "stop_inverted",
    14: "three",
    15: "two_up",
    16: "two_up_inverted",
    17: "three2",
    18: "peace_inverted",
    19: "no_gesture"
}

# dataset = dataset.HaGRID.DataSet.HaGRIDDataset("dataset/HaGRID/subsample/","dataset/HaGRID/ann_subsample/", transforms = None)
# img = dataset[0][0]
# img = img.detach().numpy()
# print(img.shape)
# img = cv2.resize(img.transpose(1,2,0),(img.shape[2],img.shape[1]))
# # print(img.shape)
# # cv2.imshow("test",img)
# # cv2.waitKey(0)
# # imgtest = Image.open("dataset/HaGRID/subsample/call/01898f3e-8422-4e6a-a056-30206f905640.jpg")
# # convert2tensor = T.ToTensor()
# # imgtest = convert2tensor(imgtest)
# # print(imgtest.shape, img.shape)
# dict = dataset[0][1]
# i=0
# for box in dict["boxes"]:
#     print(box[0],box[1],box[2],box[3])
#     x1 = int(box[0])
#     y1 = int(box[1])
#     x2 = int(box[2])
#     y2 = int(box[3])
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), thickness=3)
#     cv2.putText(img, targets[int(dict["labels"][i])], (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

# cv2.imshow("result",img)
# cv2.waitKey(0)

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrain = True)
model.eval()
x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
predictions = model(x)
