# Facial Landmark Detection
## Network
### Resnet-18
#### Env Install
Python 3.7.4 for cpu testing

Python 3.5.2 for gpu testing
```shell script
# For cpu testing and evaluation
pip3 install ./requirements_cpu.txt
# For gpu training
pip3 install ./requirements_gpu.txt
```
#### Training and testing instructions
```shell script
cd this_dir
# For training, model will save to ./weights
# All training arguments can be found at ./utils/args.py
python3 train.py --dataset_route path_to_dataset_root --dataset 'WFLW' --split 'train'
# For testing on cpu
python3 test_cpu.py --dataset_route path_to_dataset_root --dataset 'WFLW' --split 'test'
# For evaluation on cpu
python3 evaluation_cpu.py --dataset_route path_to_dataset_root --dataset 'WFLW' --split 'test'
```
#### Training Dataset
[WLFW Dateset](https://wywu.github.io/projects/LAB/WFLW.html)
#### Architecture
1. Increase input size to 256*256
2. Remove avepooling layer
3. Change output size to 192 (x1, y1, ... , x98, y98)
#### Training Setup and testing results
##### Experiment 1

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 2000 | 4 | 2e-6 | 0 | Adam | None |

Testing result

| Mean Error | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| 13.80% | 59.28% | 0.1629 |

##### Experiment 2


Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 4000 | 4 | 1e-6 | 0 | Adam | None |

Testing result (Ongoing)

| Mean Error | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| None | None | None |


##### Baseline (by Dongfeng Yu)
Training loss changed to log(1 + L2loss)

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 512 | 4 | 2e-6 | 0 | Adam | None |

Testing result (Ongoing)

| Mean Error | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| 10.67% | 38.56% | 0.2735 |


### MobileNet-V3 (Large)
#### Architecture
1. Increase input size to 256*256
2. Remove avepooling layer
3. Change output size to 192 (x1, y1, ... , x98, y98)
#### Training Setup and testing results
##### Experiment 1
Training loss changed to log(1 + L2loss)

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 2000 | 4 | 2e-6 | 0 | SGD | 1000, 1500 |

Testing result

| Mean Error | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| None | None | None |



#### TODO list

- [x] Find out dataset defintion -- output sample images to 256*256 -- label landmarks and output heatmap (Finished 2019.12.26)
- [x] Defind model and loss -- model backbone resnet-18 -- l2 loss (Finished 2019.12.30)
- [x] Training and testing process -- with tensorboard (Finished 2020.01.02)
- [ ] Testing on 300w
- [ ] Quantilization model
- [ ] MNN Infer

#### References
1. Training and testing code are modified from [Look_At_Boundary_PyTorch](https://github.com/facial-landmarks-localization-challenge/Look_At_Boundary_PyTorch)
2. Original paper and code [2018CVPR](https://github.com/wywu/LAB)