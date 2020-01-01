from .visual import *

import cv2
import time
import random
import numpy as np


def get_annotations_list(dataset, split, ispdb=False):
    annotations = []
    annotation_file = open(dataset_route[dataset] + dataset + '_' + split + '_annos.txt')
    for line in range(dataset_size[dataset][split]):
        annotations.append(annotation_file.readline().rstrip().split())
    annotation_file.close()

    return annotations


def convert_img_to_gray(img):
    if img.shape[2] == 1:
        return img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return gray
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    else:
        raise Exception("img shape wrong!\n")


def get_random_transform_param(split, bbox):
    translation, trans_dir, rotation, scaling, flip, gaussian_blur = 0, 0, 0, 1., 0, 0
    if split in ['train']:
        random.seed(time.time())
        translate_param = int(args.trans_ratio * abs(bbox[2] - bbox[0]))
        translation = random.randint(-translate_param, translate_param)
        trans_dir = random.randint(0, 3)  # LU:0 RU:1 LL:2 RL:3
        rotation = random.uniform(-args.rotate_limit, args.rotate_limit)
        scaling = random.uniform(1-args.scale_ratio, 1+args.scale_ratio)
        flip = random.randint(0, 1)
        gaussian_blur = random.randint(0, 1)
    return translation, trans_dir, rotation, scaling, flip, gaussian_blur


def further_transform(pic, bbox, flip, gaussian_blur):
    if flip == 1:
        pic = cv2.flip(pic, 1)
    if abs(bbox[2] - bbox[0]) < 120 or gaussian_blur == 0:
        return pic
    else:
        return cv2.GaussianBlur(pic, (5, 5), 1)


def get_affine_matrix(crop_size, rotation, scaling):
    center = (crop_size / 2.0, crop_size / 2.0)
    return cv2.getRotationMatrix2D(center, rotation, scaling)


def pic_normalize(pic):  # for accelerate, now support gray pic only
    pic = np.float32(pic)
    mean, std = cv2.meanStdDev(pic)
    pic_channel = 1 if len(pic.shape) == 2 else 3
    for channel in range(0, pic_channel):
        if std[channel][0] < 1e-6:
            std[channel][0] = 1
    pic = (pic - mean) / std
    return np.float32(pic)


def get_cropped_coords(crop_matrix, coord_x, coord_y):
    coord_x, coord_y = np.array(coord_x), np.array(coord_y)
    temp_x = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
    temp_y = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
    out = np.zeros(2*98)
    out[:2*98:2] = temp_x
    out[1:2*98:2] = temp_y
    return np.array(out)


def get_item_from(dataset, annotation):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pic = cv2.imread(dataset_route[dataset]+annotation[-1])
    pic = convert_img_to_gray(pic) if not args.RGB else pic
    coord_x = list(map(float, annotation[:2*kp_num[dataset]:2]))
    coord_y = list(map(float, annotation[1:2*kp_num[dataset]:2]))
    bbox = np.array(list(map(int, annotation[-11:-7])))
    position_before = np.float32([[int(bbox[0]), int(bbox[1])], [int(bbox[0]), int(bbox[3])], [int(bbox[2]), int(bbox[3])]])
    position_after = np.float32([[0, 0],
                                 [0, args.crop_size - 1],
                                 [args.crop_size - 1, args.crop_size - 1]])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    pic_crop = cv2.warpAffine(pic, crop_matrix, (args.crop_size, args.crop_size))
    pic_crop = (pic_crop / 255.0 - mean) / std
    coord_cropped = get_cropped_coords(crop_matrix, coord_x, coord_y)

    # for i in range(98):
    #     x = coord_cropped[i*2]
    #     y = coord_cropped[i*2 + 1]
    #     cv2.circle(pic_crop, (int(x), int(y)), 1, (0, 255, 0))
    # cv2.imshow('pic', pic_crop)
    # cv2.waitKey()
    # cv2.destroyWindow('pic')
    return pic_crop.transpose((2, 0, 1)), coord_cropped, bbox, annotation[-1]
