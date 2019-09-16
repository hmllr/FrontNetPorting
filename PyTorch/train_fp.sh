#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

#CUDA_VISIBLE_DEVICES=1

python3 ETH.py --regime regime.json --epochs 1 --gray 1 --load-trainset "/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/train_vignette4.pickle" --quantize --load-testset "/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/test_vignette4.pickle" --load-model "Models/DronetGray-098.pt" --save-model "Models/DronetGray.pt" 

