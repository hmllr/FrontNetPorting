import pandas as pd
import numpy as np
import random
import logging
import sys
import os
sys.path.append("../")
import config
# try to remove ros because we need cv2 and the python2/3 paths get messed up with ros
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
sys.path.append("../DataProcessing/")
import cv2
from ImageTransformer import ImageTransformer

for root, dirs, files in os.walk("test_himax_hanna/"):
    for file in files:
        if file.endswith('.csv'):
            X = np.genfromtxt(root +'/' + file, delimiter=',')
            X = np.reshape(X, (244, 324))
            #cv2.imshow(root+file,X.astype("uint8"))
            #cv2.waitKey(0)
            X_con = np.concatenate((X,np.reshape(X[:,-1],(244,-1))),axis=1)
            X_diff = np.diff(X_con)
            X_sum = np.sum(abs(X_diff), axis=0)
            Xmax = np.argmax(X_sum)
            print(root, file, Xmax)
            X_shift = np.concatenate((X[:,Xmax:],X[:,:Xmax]), axis=1)
            cv2.imwrite(root +'/fixed_' + os.path.splitext(file)[0] + '.pgm',X_shift.astype("uint8"))
            




