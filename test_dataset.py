import torch
import ipdb
from dataset import GeneralDataset
from utils import *


arg = args
trainset = GeneralDataset(dataset=arg.dataset)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                                 num_workers=arg.workers, pin_memory=True)
for i in dataloader:
    ipdb.set_trace()