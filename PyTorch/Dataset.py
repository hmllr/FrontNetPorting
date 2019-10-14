import torch
from torch.utils import data
import numpy as np
import cv2
import sys
sys.path.append("../DataProcessing/")
from ImageTransformer import ImageTransformer


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data, labels, train=False, isClassifier=False):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        length = len(self.data)
        self.list_IDs = range(0, length)
        self.train = train
        self.it = ImageTransformer()
        self.isClassifier = isClassifier


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def augmentGamma(self, X):
      X = X.cpu().numpy()
      h, w = X.shape[1:3]
      X = np.reshape(X, (h, w)).astype("uint8")
      # gamma correction augmentation
      gamma = np.random.uniform(0.6, 1.4)
      table = self.it.adjust_gamma(gamma)
      X = cv2.LUT(X, table)
      X = np.reshape(X, (1, h, w))
      X = torch.from_numpy(X).float().round()

      return X

  def augmentDR(self, X):
      X = X.cpu().numpy()
      h, w = X.shape[1:3]
      X = np.reshape(X, (h, w)).astype("uint8")
      # dynamic range augmentation
      dr = np.random.uniform(0.4, 0.8)  # dynamic range
      lo = np.random.uniform(0, 0.3)
      hi = min(1.0, lo + dr)
      # maps all values in [0, 255*lo] to 0, the ones in [255*hi,255] to 255 
      # and interpolates the ones in between to stretch over [0,255]
      X = np.interp(X/255.0, [0, lo, hi, 1], [0, 0, 1, 1])
      X = 255 * X
      X = np.reshape(X, (1, h, w))
      X = torch.from_numpy(X).float().round()

      return X

  '''def brightenDR(self, X):
      X = X.cpu().numpy()
      h, w = X.shape[1:3]
      X = np.reshape(X, (h, w)).astype("uint8")
      # dynamic range augmentation
      dr = 0.3  # dynamic range
      lo = 0
      hi = min(1.0, lo + dr)
      # maps all values in [0, 255*lo] to 0, the ones in [255*hi,255] to 255 
      # and interpolates the ones in between to stretch over [0,255]
      X = np.interp(X/255.0, [0, lo, hi, 1], [0, 0, 1, 1])
      X = 255 * X
      X = np.reshape(X, (1, h, w))
      X = torch.from_numpy(X).float().round()

      return X'''


  def __getitem__(self, index):
        'Generates one sample of data'
        ID = index

        X = self.data[ID]
        y = self.labels[ID]

        # to get a more diverse training set we augment it using 
        # (a) flip random images
        # (b) randomly augment the dynamic range (simulates different light conditions/exposures)
        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [2])
                if self.isClassifier == False:
                  y[1] = -y[1]  # Y
                  y[3] = -y[3]  # Relative YAW
                  #cv2.imshow("flipped", np.reshape(X.numpy().astype("uint8"),(60,108)))
                  #cv2.waitKey(0)

            if X.shape[0] == 1:
               # if np.random.choice([True, False]):
                #    X = self.augmentGamma(X)
                if np.random.choice([True, False]):#,p=[0.8,0.2]):
                    X = self.augmentDR(X)
        '''else:
          X = self.brightenDR(X)'''
        return X, y
