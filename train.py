import torch
import os
import torch.nn as nn
import tqdm
import platform
import torchvision.transforms as transforms
from dataset import GeneralDataset
from utils import args
from models.resnet import resnet18
from torch.utils.tensorboard import SummaryWriter

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.resume_folder):
    os.mkdir(args.resume_folder)


def train(arg):
    print('*****  Normal Training  *****')
    print('Training parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# PDB:                ' + str(arg.PDB) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    if 'Windows' in platform.platform():
        model = resnet18()
    else:
        model = resnet18().cuda()
    trainset = GeneralDataset(dataset=arg.dataset)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)

    print('Start training ...')
    writer = SummaryWriter()
    num_step = 0
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                                 num_workers=1, pin_memory=True)
        for data in tqdm.tqdm(dataloader):
            input_images, coord, _, _ = data
            if 'Windows' in platform.platform():
                input_images = input_images.float()
                coord = coord.float()
            else:
                input_images = input_images.cuda().float()
                coord = coord.cuda().float()
            estimated_coord = model(input_images)
            loss = criterion(estimated_coord, coord)
            loss_ = loss.item()
            loss = torch.log(1 + loss)
            loss.backward()
            optimizer.step()
            num_step += 1
            if num_step % 10 == 0:
                writer.add_scalar('loss', loss_, num_step)
        print('epoch: {} | loss: {}'.format(epoch, loss.item()))
        if (epoch + 1) % arg.save_interval == 0:
            torch.save(model.state_dict(), arg.save_folder + 'resnet18_' + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    train(args)
