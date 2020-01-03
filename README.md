# Facial Landmark Detection
## Network
### Resnet-18
#### Training Dataet
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
| 2000 | 4 | 2e-6 | 5e-5 | Adam | 1000, 1500 |

Testing result

| Mean Error Rate | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| 13.80% | 59.28% | 0.1629 |

##### Experiment 2


Training setup

| Training epochs | Training batchsize |  LR  | Weight decay | Opt | Step Value |
| :-------------: | :----------------: | ---  | :----------: | --- | :---------: |
| 4000 | 4 | 1e-6 | 5e-6 | Adam | 2000, 3000 |

Testing result (Ongoing)

| Mean Error Rate | Failure Rate | AUC |
| :-------------: | :----------: | --- |
| None | None | None |


#### TODO list
- [x] Find out dataset defintion -- output sample images to 256*256 -- label landmarks and output heatmap (Finished 2019.12.26)
- [x] Defind model and loss -- model backbone resnet-18 -- l2 loss (Finished 2019.12.30)
- [x] Training and testing process -- with tensorboard (Finished 2020.01.02)
- [ ] Testing on 300w
- [ ] Quantilization model
- [ ] MNN Infer
