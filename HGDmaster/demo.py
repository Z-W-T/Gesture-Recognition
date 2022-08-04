import os
import cv2
import time
import torch
import logging
import argparse
import numpy as np
import random
import sys
import torchvision
sys.path.append("C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID")
sys.path.append("C:\\Users\\DELL\\Desktop\\gesture\\HGDmaster\\detector\\detection")
sys.path.insert(0,"C:\\Users\\DELL\\Desktop\\gesture\\HGDmaster\\detector\\detection")
import DataSet


from torch import Tensor
from PIL import Image, ImageOps
from typing import Optional, Tuple
from torchvision import transforms as T
from torchvision.transforms import functional as f
from detector.ssd_mobilenetv3 import SSDMobilenet
from detector.model import TorchVisionModel
from detector.detection import utils
from detector.detection.engine import train_one_epoch, evaluate



logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

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


class Demo:

    @staticmethod
    def preprocess(img: np.ndarray, device: str) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        # cv2.imshow('Original Image', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((320, 320))
        image1 = np.asarray(image)
        # cv2.imshow('Processed Image', image1)

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)

    @staticmethod
    def train(model: TorchVisionModel, device: str):
        #取数据
        dataset = DataSet.HaGRIDDataset("C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\subsample","C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\ann_subsample", None)
        dataset_test = DataSet.HaGRIDDataset("C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\subsample","C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\ann_subsample", None)

        #划分训练集和测试集
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        #构建dataloader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4,
                                                    collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                    collate_fn=utils.collate_fn)

        #设置网络训练参数
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.Adam(params,lr = 0.005, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
        num_epochs = 5
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations

            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        
        torch.save(model.state_dict(), "detector/SSDLite_self.pth")

    @staticmethod
    def run(detector: TorchVisionModel, num_hands: int = 2, threshold: float = 0.5, device: str = "cpu") -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        detector : TorchVisionModel
            Detection model
        num_hands:
            Min hands to detect
        threshold : float
            Confidence threshold
        """

        # cap = cv2.VideoCapture("C:\\Users\\DELL\\Desktop\\gesture\\HGDmaster\\TestVedio0.mp4")

        t1 = cnt = 0
        # while cap.isOpened():
        #     delta = (time.time() - t1)
        #     t1 = time.time()

        #     ret, frame = cap.read()
        #     if ret:
        for k in range(len(targets)-1):
            indexlist = list(sorted(os.listdir(os.path.join("C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\subsample", targets[k+1]))))
            # for j in random.sample(range(0,len(indexlist)),5):
            for j in range(5,10):
                cnt = 0
                img_path = os.path.join("C:\\Users\\DELL\\Desktop\\gesture\\dataset\\HaGRID\\subsample", targets[k+1], indexlist[j])
                frame = Image.open(img_path).convert("RGB")
                frame = np.asarray(frame)
                processed_frame, size, padded_size = Demo.preprocess(frame, device)
                with torch.no_grad():
                    output = detector(processed_frame)[0]
                # print(output["boxes"].shape,output["scores"].shape,output["labels"].shape)
                _, indices = torch.sort(output["scores"], descending=True)
                boxes = output["boxes"][indices]
                scores = output["scores"][indices]
                labels = output["labels"][indices]
                for i in range(min(num_hands, len(boxes))):
                    # if scores[i] > 0 and targets[int(labels[i])] != "no_gesture":
                    if scores[i] > threshold:
                        cnt += 1
                        width, height = size
                        padded_width, padded_height = padded_size
                        scale = max(width, height) / 320

                        padding_w = abs(padded_width - width) // (2 * scale)
                        padding_h = abs(padded_height - height) // (2 * scale)

                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                        cv2.putText(frame, targets[int(labels[i])], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

                # fps = 1 / delta
                fps = 10
                # cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cv2.putText(frame, f"Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                # cnt += 1
                frame = Image.fromarray(frame)
                width, height = frame.size
                frame = ImageOps.pad(frame,(max(width, height),max(width, height)))
                
                frame = frame.resize((800, 800))
                frame = np.asarray(frame)
                cv2.imshow('Frame', frame)

                key = cv2.waitKey(1000)
                if key == ord('q'):
                    return
                # else:
                #     cap.release()
                #     cv2.destroyAllWindows()



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def _load_model(model_path: str, device: str, train: str) -> TorchVisionModel:
    """
    Load model
    Parameters
    ----------
    model_path: str
        Model Path
    device: str
        Device cpu or cuda
    """
    ssd_mobilenet = SSDMobilenet(num_classes=len(targets) + 1)
    if not os.path.exists(model_path):
        logging.info(f"Model not found {model_path}")
        raise FileNotFoundError  
    ssd_mobilenet.load_state_dict(model_path, map_location=device)
    # torch.save(ssd_mobilenet,"detector/SSDLite_net.pth")
    ssd_mobilenet.to(device)
    if train:
        ssd_mobilenet.train()
    else:
        ssd_mobilenet.eval()
    return ssd_mobilenet


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier...")

    parser.add_argument(
        "-p",
        "--path_to_model",
        required=True,
        type=str,
        help="Path to model"
    )

    parser.add_argument(
        "-d",
        "--device",
        required=False,
        default="cpu",
        type=str,
        help="Device"
    )

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == '__main__':
    #pragma optimize("", off)
    train = True
    args = parse_arguments()
    model = _load_model(os.path.expanduser(args.path_to_model), args.device, train)
    if train:
        Demo.train(model, args.device)
    else:
        if model is not None:
            Demo.run(model, num_hands=2, threshold=0.05, device = args.device)
