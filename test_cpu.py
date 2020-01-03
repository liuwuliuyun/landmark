import torch
import cv2
import numpy as np
import time
from dataset import GeneralDataset
from utils import args
from models.models import resnet18


def show_image(image, coord):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    res_image = ((image * std + mean) * 255.0).astype(np.uint8).copy()
    for i in range(98):
        x = coord[2*i]
        y = coord[2*i+1]
        # print('x is {} y is {}'.format(x, y))
        cv2.circle(res_image, (int(x), int(y)), 3, (0, 255, 255))
    cv2.imshow('final_pic', res_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test(arg):
    # load test dataset
    testset = GeneralDataset(dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
    # load trained model
    weight_path = '.\\weights\\resnet18_2000.pth'
    model = resnet18()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.eval()
    # start testing
    with torch.no_grad():
        for data in dataloader:
            # start = time.time()
            input_image, coord, _, _ = data
            input_image = input_image.float()
            estimated_coord = model(input_image)
            input_image = input_image.squeeze().numpy()
            estimated_coord = estimated_coord.squeeze().numpy()
            input_image = input_image.transpose((1, 2, 0))
            show_image(input_image, estimated_coord)


if __name__ == '__main__':
    test(args)
