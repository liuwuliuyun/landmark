import torch
import ipdb
from dataset import GeneralDataset
from utils import args


arg = args
trainset = GeneralDataset(dataset=arg.dataset)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                                 num_workers=1, pin_memory=True)
for i in dataloader:
    ipdb.set_trace()
    # i -> List of 6 items
    # i[0] batch_size*256*256 [Grey Scale Head Images]
    # i[1] batch_size*196 [Coords of X and Y]
    # i[2] batch_size*13*64*64 [Heatmaps]
    # i[3] batch_size*196 [Coords of X and Y]
    # i[4] batch_size*4 [bboxes]
    # i[5] [list of original image filenames]