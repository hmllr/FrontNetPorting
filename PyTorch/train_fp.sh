#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

#CUDA_VISIBLE_DEVICES=1

python3 ETH.py --regime regime.json --epochs 10 --gray 1 --tensorboard 1 --quantize --load-trainset "/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/train_vignette4.pickle" --load-testset "/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/test_vignette4.pickle" --save-model "Models/FindNetGray3232_3264_64128_quant8fromMaxPool2.pt" --load-model "../FindNetMaxPool2x2/Models/FindNetGray3232_3264_64128_maxpool2.pt"

