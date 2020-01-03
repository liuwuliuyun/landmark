import torch
import cv2
import numpy as np
import time
import tqdm
from dataset import GeneralDataset
from sklearn.metrics import auc
from utils.dataset_info import *
from models.models import resnet18


def show_image(image, coord):
    image = image.squeeze().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    res_image = ((image * std + mean) * 255.0).astype(np.uint8).copy()
    for i in range(98):
        x = coord[2*i]
        y = coord[2*i+1]
        # print('x is {} y is {}'.format(x, y))
        cv2.circle(res_image, (int(x), int(y)), 3, (0, 255, 255))
    cv2.imshow('landmark detection results', res_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def calc_normalize_factor(dataset, gt_coords_xy, normalize_way='inter_pupil'):
    if normalize_way == 'inter_ocular':
        error_normalize_factor = np.sqrt(
            (gt_coords_xy[0][lo_eye_corner_index_x[dataset]] - gt_coords_xy[0][ro_eye_corner_index_x[dataset]]) *
            (gt_coords_xy[0][lo_eye_corner_index_x[dataset]] - gt_coords_xy[0][ro_eye_corner_index_x[dataset]]) +
            (gt_coords_xy[0][lo_eye_corner_index_y[dataset]] - gt_coords_xy[0][ro_eye_corner_index_y[dataset]]) *
            (gt_coords_xy[0][lo_eye_corner_index_y[dataset]] - gt_coords_xy[0][ro_eye_corner_index_y[dataset]]))
        return error_normalize_factor
    elif normalize_way == 'inter_pupil':
        if l_eye_center_index_x[dataset].__class__ != list:
            error_normalize_factor = np.sqrt(
                (gt_coords_xy[0][l_eye_center_index_x[dataset]] - gt_coords_xy[0][r_eye_center_index_x[dataset]]) *
                (gt_coords_xy[0][l_eye_center_index_x[dataset]] - gt_coords_xy[0][r_eye_center_index_x[dataset]]) +
                (gt_coords_xy[0][l_eye_center_index_y[dataset]] - gt_coords_xy[0][r_eye_center_index_y[dataset]]) *
                (gt_coords_xy[0][l_eye_center_index_y[dataset]] - gt_coords_xy[0][r_eye_center_index_y[dataset]]))
            return error_normalize_factor
        else:
            length = len(l_eye_center_index_x[dataset])
            l_eye_x_avg, l_eye_y_avg, r_eye_x_avg, r_eye_y_avg = 0., 0., 0., 0.
            for i in range(length):
                l_eye_x_avg += gt_coords_xy[0][l_eye_center_index_x[dataset][i]]
                l_eye_y_avg += gt_coords_xy[0][l_eye_center_index_y[dataset][i]]
                r_eye_x_avg += gt_coords_xy[0][r_eye_center_index_x[dataset][i]]
                r_eye_y_avg += gt_coords_xy[0][r_eye_center_index_y[dataset][i]]
            l_eye_x_avg /= length
            l_eye_y_avg /= length
            r_eye_x_avg /= length
            r_eye_y_avg /= length
            error_normalize_factor = np.sqrt((l_eye_x_avg - r_eye_x_avg) * (l_eye_x_avg - r_eye_x_avg) +
                                             (l_eye_y_avg - r_eye_y_avg) * (l_eye_y_avg - r_eye_y_avg))
            return error_normalize_factor


def inverse_affine(arg, pred_coords, bbox):
    import copy
    pred_coords = copy.deepcopy(pred_coords)
    for i in range(kp_num[arg.dataset]):
        pred_coords[2 * i] = bbox[0] + pred_coords[2 * i]/(arg.crop_size-1)*(bbox[2] - bbox[0])
        pred_coords[2 * i + 1] = bbox[1] + pred_coords[2 * i + 1]/(arg.crop_size-1)*(bbox[3] - bbox[1])
    return pred_coords


def calc_error_rate_i(dataset, pred_coords, gt_coords_xy, error_normalize_factor):
    temp, error = (pred_coords - gt_coords_xy)**2, 0.
    for i in range(kp_num[dataset]):
        error += np.sqrt(temp[2*i] + temp[2*i+1])
    return error/kp_num[dataset]/error_normalize_factor


def calc_auc(dataset, split, error_rate, max_threshold):
    error_rate = np.array(error_rate)
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_rate < threshold[i]) * 1.0 / dataset_size[dataset][split]
    return auc(threshold, accuracys) / max_threshold, accuracys


def evaluate(arg):
    print('*****  Normal Evaluating  *****')
    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Epoch of the model: ' + str(arg.eval_epoch) + '\n' +
          '# Normalize way:      ' + arg.norm_way + '\n' +
          '# Max threshold:      ' + str(arg.max_threshold) + '\n')
    # load test dataset
    testset = GeneralDataset(dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
    # load trained model
    print('Loading network ...')
    weight_path = '.\\weights\\resnet18_2000.pth'
    model = resnet18()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.eval()
    print('Loading network done!\nStart testing ...')

    # start testing
    error_rate = []
    time_records = []
    failure_count = 0
    max_threshold = arg.max_threshold

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):

            input_image, coord_ground_truth, bbox, file_name = data
            bbox = bbox.squeeze().numpy()
            input_image = input_image.float()
            # TODO: use coord_xy by get_item in dataload.py. Current use face_size normalization other two is not right
            error_normalize_factor = calc_normalize_factor(arg.dataset, coord_ground_truth.numpy(), arg.norm_way) \
                if arg.norm_way in ['inter_pupil', 'inter_ocular'] else (bbox[2] - bbox[0])
            start = time.time()
            estimated_coord = model(input_image)
            time_records.append(time.time() - start)
            estimated_coord = estimated_coord.squeeze().numpy()
            error_rate_i = calc_error_rate_i(arg.dataset, estimated_coord, coord_ground_truth.squeeze().numpy(), error_normalize_factor)
            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve, auc_record = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n# AUC:          {:.4f}\n# Error Rate:   {:.2f}%\n# Failure Rate: {:.2f}%\n'.format(
        area_under_curve, error_rate, failure_rate))
    print('Average speed: {:.2f}FPS'.format(1. / np.mean(np.array(time_records))))

if __name__ == '__main__':
    evaluate(args)