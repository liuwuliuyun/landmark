import torch
import cv2
import numpy as np
import tqdm
import time
from dataset import GeneralDataset
from utils import args
from models.models import resnet18


def show_image(input_image, coord):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = (input_image * std + mean) * 255.0
    for i in range(98):
        x = coord[2*i]
        y = coord[2*i+1]
        cv2.circle(input_image, (int(x), int(y)), 1, (0, 255, 0))
    cv2.imshow('test_pic', input_image)
    cv2.waitKey()
    cv2.destroyWindow('test_pic')


def test(arg):
    # load test dataset
    testset = GeneralDataset(dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
    # load trained model
    weight_path = './weights/resnet18_2000.pth'
    model = resnet18().cuda()
    model.load_state_dict(torch.load(weight_path), strict=True)
    model.eval()
    # start testing
    with torch.no_grad():
        for data in dataloader:
            # start = time.time()
            input_image, coord, _, _ = data
            input_image.cuda().float()
            estimated_coord = model(input_image)
            input_image.transpose((1, 2, 0))
            input_image.detach().cpu.squeeze().numpy()
            estimated_coord.detach().cpu().squeeze().numpy()
            show_image(input_image, estimated_coord)


if __name__ == '__main__':
    test(args)