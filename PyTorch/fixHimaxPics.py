# file to fix shifted himax pictures. 
# Sometimes there is somehow a shift of a few bytes in the circular buffer that makes the images appear as if the rightmost part was on the left or vice versa.
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

for root, dirs, files in os.walk("train_himax_hanna/Dataset_Himax_NoHead/take15"):
    for file in files:
        if file.endswith('.csv'):
            print(root, file)
            try:
	            X = np.genfromtxt(root +'/' + file, delimiter=',')
	            X = np.reshape(X, (244, 324))
	            #cv2.imshow(root+file,X.astype("uint8"))
	            #cv2.waitKey(0)
	            # we have to put the first column at the end to easily compute all differences in the next step
	            X_con = np.concatenate((X,np.reshape(X[:,0],(244,-1))),axis=1)
	            # we look for the largest difference in between two columns
	            X_diff = np.diff(X_con)
	            X_sum = np.sum(abs(X_diff), axis=0)
	            Xmax = np.argmax(X_sum)
	            print(Xmax)
	            # and then put the top line of the shifted slice (meaning the pixels from the last picture) at the bottom 
	            # to have the lines not shifted (the wrong bytes are now where they are missing, but as some frames are dropped it would not be better to fix in between pictures)
	            X_temp = np.concatenate((X[1:,:Xmax+1],X[0:1,:Xmax+1]), axis=0)
	            # and then put the slice at the right place
	            X_shift = np.concatenate((X[:,Xmax+1:],X_temp), axis=1)
	            cv2.imwrite(root +'/fixed_' + os.path.splitext(file)[0] + '.pgm',X_shift.astype("uint8"))
            except Exception as e:
            	print(e)




