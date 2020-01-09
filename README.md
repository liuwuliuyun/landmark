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
cd ROOT_DIR_OF_PRJ
# For training, model will save to ./weights
# All training arguments can be found at ./utils/args.py
python3 train.py --dataset_route path_to_dataset_root \
                 --dataset 'WFLW' \
                 --split 'train'
# For testing on cpu
python3 test_cpu.py --dataset_route path_to_dataset_root \
                    --dataset 'WFLW' \
                    --split 'test'
# For evaluation on cpu
python3 evaluation_cpu.py --dataset_route path_to_dataset_root \
                          --dataset 'WFLW' \ 
                          --split 'test'\ 
                          --eval_model_path = model_path \
                          --norm_way = 'face_size'
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

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- | :---:|
| 13.80% | 59.28% | 0.1629 | Face Size |

##### Experiment 2


Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 2700 | 4 | 1e-6 | 0 | Adam | None |

Testing result

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- | :---:|
| 14.04% | 59.92% | 0.1606 | Face Size |


##### Baseline (by Dongfeng Yu)
Training loss changed to log(1 + L2loss)

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 512 | 4 | 2e-6 | 0 | Adam | None |

Testing result

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- |:---:|
| 10.67% | 38.56% | 0.2735 | Face Size |


### MobileNet-V2
#### Architecture
1. Increase input size to 256*256
3. Change output size to 192 (x1, y1, ... , x98, y98)
#### Training Setup and testing results
##### Experiment 1
Training loss changed to log(1 + L2loss)

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 992 | 4 | 2e-6 | 0 | Adam | None |

Testing result (mbv2_992.pth)

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- |:---:|
| 10.70% | 44.48% | 0.2299 | Face Size |

Testing result (mbv2_512.pth)

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- |:---:|
| 10.45% | 42.00% | 0.2427 | Face Size |


### ShuffleNet-V2
#### Architecture
1. Increase input size to 256*256
3. Change output size to 192 (x1, y1, ... , x98, y98)
#### Training Setup and testing results
##### Experiment 1
Training loss changed to log(1 + L2loss)

Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 700 | 4 | 2e-6 | 0 | Adam | None |

Testing result

| Mean Error | Failure Rate | AUC | Normalized Way |
| :-------------: | :----------: | --- |:---:|
| 10.83% | 45.04% | 0.2241 | Face Size |


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