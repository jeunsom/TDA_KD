# TDA_KD
Topological Knowledge Distillation for Wearable Sensor Data

## Overview
This is re-implementation of the Topological Knowledge Distillation loss described in:
E. S. Jeon, H. Choi, A. Shukla, Y. Wang, M. P. Buman, and P. Turaga,
“Topological knowledge distillation for wearable sensor data,” in Asilo-
mar Conference on Signals, Systems, and Computers, 2022, proceedings
Forthcoming.

## Requirements
* pytorch>=1.4.0
* python>=3.6.0

## Time Series Data Classification
We use time series data classification as an example with a simple architecture. In order to reproduce the results described on the paper, please modify the hyperparameters. The users can also change the data to other dataset at their interest.
To run the code, please create persistence image via PI.ipynb.

## Sample
python3 train_ad.py --epochs 200 --teacher wrn163 --teacher-checkpoint Teaimg/wrn/wrn163_0_00_ep68_val81.3741_best.pth.tar --teacher2 wrn1631 --teacher-checkpoint2 Teasig/wrn/wrn163_0_org03_ep35_val74.1546_best.pth.tar --student wrn1611 --student-checkpoint Stu/wrn1611_161sig_161_099_id0_00s_ep60_val74.5035_best.pth.tar --cuda 1 --dataset pamap --batch_size 64 --sbj 0 --trial 163m_161_099_id0_00_std --save_weight 0 --seed 1234

