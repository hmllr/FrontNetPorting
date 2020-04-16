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

class DataProcessor:

    @staticmethod
    def ProcessTrainData(trainPath, image_height, image_width, isGray = False, isExtended=False, isClassifier=False, fromPics=False, picsPath=None, isCombined=False, noPose=False, fromCaltech=False, head=False, onlyHimax=False):
        """Reads the .pickle file and converts it into a format suitable fot training

            Parameters
            ----------
            trainPath : str
                The file location of the .pickle
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """
        if onlyHimax:
            noPose = True
            x_train = []
            y_train = []
            size = 0
            countAugmented = 0
            countDarioSamples = 0
            for root, dirs, files in os.walk(picsPath):
                for file in files:
                    isAugnmented = False
                    isValidation = False
                    if file.endswith('.pgm'):
                        X = cv2.imread(root +'/'+ file, 0)
                        x_train.append(X)
            x_train = np.asarray(x_train)
            x_train = np.reshape(x_train, (-1, image_height, image_width, 1))
            x_train = np.swapaxes(x_train, 1, 3)
            x_train = np.swapaxes(x_train, 2, 3)
            size = np.shape(x_train)
            logging.info('[DataProcessor] train pics number: ' + str(size) + ' head: ' + str(head))
            n_val = int(float(size[0]) * 0.2)
            print(n_val)
            ix_val, ix_tr = np.split(np.random.permutation(size[0]), [n_val])
            x_validation = x_train[ix_val, :]
            x_train = x_train[ix_tr, :]

        elif fromCaltech:
            noPose = True
            train_set = pd.read_pickle(trainPath).values
            train_set = train_set[0:60000]
            logging.info('[DataProcessor] train caltech shape: ' + str(train_set.shape))
            size = len(train_set[:, 0])
            n_val = int(float(size) * 0.2)
            #n_val = 13000

            np.random.seed()
            # split between train and test sets:
            x_train = train_set[:, 0]
            x_train = np.vstack(x_train[:]).astype(np.float32)
            if isGray == True:
                x_train = np.reshape(x_train, (-1, image_height, image_width, 1))
            else:
                x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

            x_train= np.swapaxes(x_train, 1, 3)
            x_train = np.swapaxes(x_train, 2, 3)

            y_train_headlabels = train_set[:, 1]

            ix_val, ix_tr = np.split(np.random.permutation(train_set.shape[0]), [n_val])
            #print(train_set.shape[0], n_val, ix_val)
            x_validation = x_train[ix_val, :]
            x_train = x_train[ix_tr, :]
            y_validation_headlabels = y_train_headlabels[ix_val]
            y_train_headlabels = y_train_headlabels[ix_tr]

            shape_ = len(x_train)
            # randomize order
            sel_idx = random.sample(range(0, shape_), k=(size-n_val))
            #sel_idx = random.sample(range(0, shape_), k=50000)
            x_train = x_train[sel_idx, :]
            y_train_headlabels = y_train_headlabels[sel_idx]

        elif fromPics == False:

            train_set = pd.read_pickle(trainPath).values
            #train_set = train_set[0:50000]
            train_set = train_set[0:63721]

            logging.info('[DataProcessor] train dario shape: ' + str(train_set.shape))
            size = len(train_set[:, 0])
            n_val = int(float(size) * 0.2)
            #n_val = 13000

            np.random.seed()
            # split between train and test sets:
            x_train = train_set[:, 0]
            x_train = np.vstack(x_train[:]).astype(np.float32)
            if isGray == True:
                x_train = np.reshape(x_train, (-1, image_height, image_width, 1))
            else:
                x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

            x_train= np.swapaxes(x_train, 1, 3)
            x_train = np.swapaxes(x_train, 2, 3)

            y_train = train_set[:, 1]
            y_train = np.vstack(y_train[:]).astype(np.float32)

            ix_val, ix_tr = np.split(np.random.permutation(train_set.shape[0]), [n_val])
            #print(train_set.shape[0], n_val, ix_val)
            x_validation = x_train[ix_val, :]
            x_train = x_train[ix_tr, :]
            y_validation = y_train[ix_val, :]
            y_train = y_train[ix_tr, :]

            shape_ = len(x_train)
            # randomize order
            sel_idx = random.sample(range(0, shape_), k=(size-n_val))
            #sel_idx = random.sample(range(0, shape_), k=50000)
            x_train = x_train[sel_idx, :]
            y_train = y_train[sel_idx, :]
            #for i in range(0,400):
            #    print(y_train[i*100])
            #    img = np.reshape(x_train[i*100],(image_height,image_width))
            #    cv2.imshow('train',img.astype("uint8"))
            #    cv2.waitKey(0)
        else:
            noPose = True
            x_train =[]
            x_train_fix = [] # don't want to have the augmented images from the same in x_val and x_train
            x_val_fix = []
            size = 0
            countAugmented = 0
            countDarioSamples = 0
            for root, dirs, files in os.walk(picsPath):
                for file in files:
                    isAugnmented = False
                    isValidation = False
                    if file.endswith('.csv'):
                        pass
                        ''' Not used at the moment as shift in images and fixed are stored as pgm
                        X = np.genfromtxt(root +'/' + file, delimiter=',')
                        X = np.reshape(X, (244, 324))
                        #cv2.imshow('train' + str(size),X.astype("uint8"))
                        #cv2.waitKey(0)
                        X = cv2.resize(X, (config.input_width, config.input_height), cv2.INTER_AREA)
                        X = X.astype("uint8")
                        #print(X)
                        #cv2.imshow('test2',X)
                        #cv2.waitKey(0)
                        #X = np.reshape(X, (-1, 60,108, 1))
                        #X = np.swapaxes(X, 1, 3)
                        #X = np.swapaxes(X, 2, 3)
                        x_train.append(X)
                        size+=1'''
                    if file.endswith('.pgm'):
                        X = cv2.imread(root +'/'+ file, 0)
                        #cv2.imshow('train' + str(size),X.astype("uint8"))
                        #cv2.waitKey(0)
                        try:
                            w = 324
                            h = 244
                            X = np.reshape(X, (h, w))
                            augment = True
                            if augment and (not "Udacity" in root):
                                countAugmented += 1
                                augmentation_factor_heads = 20
                                augmentation_factor_noheads = 30
                                if countAugmented%5:
                                    isAugnmented = True
                                    X_to_append = X[np.random.randint(0,5):np.random.randint(209,244),np.random.randint(0,20):np.random.randint(304,324)]
                                    #cv2.imshow('testrotorig',X)
                                    #cv2.waitKey(0)
                                    if head:
                                        augmentation_factor = augmentation_factor_heads
                                        for i in range(0,augmentation_factor_heads - 1):
                                            X_to_append = cv2.resize(X_to_append, (config.input_width, config.input_height), cv2.INTER_AREA)
                                            X_to_append = X_to_append.astype("uint8")
                                            x_train_fix.append(X_to_append)
                                            #size+=1
                                            X_to_append = X[np.random.randint(0,1+i):np.random.randint(243-2*i,244),np.random.randint(0,1+2*i):np.random.randint(323-2*i,324)]
                                    else:
                                        augmentation_factor = augmentation_factor_noheads
                                        for i in range(0,augmentation_factor_noheads - 1):
                                            cut = i
                                            X_to_append = cv2.resize(X_to_append, (config.input_width, config.input_height), cv2.INTER_AREA)
                                            X_to_append = X_to_append.astype("uint8")
                                            #cv2.imshow('testrot_res',X_to_append)
                                            #cv2.waitKey(0)
                                            x_train_fix.append(X_to_append)
                                            #size+=1
                                            center = (w/2,h/2)
                                            angle = np.random.randint(0,int(i/1)+1)
                                            if i%2:
                                                angle = 360 - angle  
                                            M = cv2.getRotationMatrix2D(center, angle, 1)
                                            X_to_append = cv2.warpAffine(X, M, (w, h)) 
                                            #cv2.imshow('testrot',X_to_append)
                                            #cv2.waitKey(0)
                                            X_to_append = X_to_append[np.random.randint(10+cut,11+2*cut):np.random.randint(233-2*cut,234-cut),np.random.randint(10+cut,11+2*cut):np.random.randint(313-2*cut,314-cut)]
                                            #cv2.imshow('testrotcrop',X_to_append)
                                            #cv2.waitKey(0)
                                else:
                                    isValidation = True

                        except ValueError as e:
                            #print(e)
                            if (not "Hollywood" in root):
                                countDarioSamples += 1
                                #if countDarioSamples%10:
                                #    continue
                                X = np.reshape(X, (config.input_height, config.input_width))
                            else:
                                X = np.reshape(X, (config.input_height, config.input_width))
                        #cv2.imshow('train' + str(size),X.astype("uint8"))
                        #cv2.waitKey(0)
                        X = cv2.resize(X, (config.input_width, config.input_height), cv2.INTER_AREA)
                        X = X.astype("uint8")
                        #print(X)
                        #cv2.imshow('test2',X)
                        #cv2.waitKey(0)
                        #X = np.reshape(X, (-1, 60,108, 1))
                        #X = np.swapaxes(X, 1, 3)
                        #X = np.swapaxes(X, 2, 3)
                        if isAugnmented:
                            x_train_fix.append(X)
                        elif isValidation:
                            x_val_fix.append(X)
                        else:
                            x_train.append(X)
                            size+=1
                            #print(size)
            # Add random images for no head learning - ATTENTION make sure it is labeled as no head!
            #num_randimg = 1000
            #for x in range(1,num_randimg):
            #    x_train.append(np.random.randint(max(20,255*x/num_randimg),size=np.shape(X)).astype("uint8"))
            #    x_train.append(np.random.randint(1+255.*x/num_randimg)*np.ones(np.shape(X)).astype("uint8"))
            #size += 2*num_randimg - 2
            print("DarioSamples : ", countDarioSamples)
            x_train = np.asarray(x_train)
            x_train = np.reshape(x_train, (-1, image_height, image_width, 1))
            x_train = np.swapaxes(x_train, 1, 3)
            x_train = np.swapaxes(x_train, 2, 3)
            x_train_fix = np.asarray(x_train_fix)
            x_train_fix = np.reshape(x_train_fix, (-1, image_height, image_width, 1))
            x_train_fix = np.swapaxes(x_train_fix, 1, 3)
            x_train_fix = np.swapaxes(x_train_fix, 2, 3)
            x_val_fix = np.asarray(x_val_fix)
            x_val_fix = np.reshape(x_val_fix, (-1, image_height, image_width, 1))
            x_val_fix = np.swapaxes(x_val_fix, 1, 3)
            x_val_fix = np.swapaxes(x_val_fix, 2, 3)
            print(np.shape(x_train), np.shape(x_train_fix), np.shape(x_val_fix))
            logging.info('[DataProcessor] train pics number: ' + str(size) + ' ' + str(countAugmented) + '*' + str(augmentation_factor) + ' head: ' + str(head))
            n_val = int(float(size) * 0.2)
            print(n_val)
            ix_val, ix_tr = np.split(np.random.permutation(size), [n_val])
            x_validation = x_train[ix_val, :]
            #print(np.shape(x_validation), np.shape(x_val_fix), np.size(x_validation), np.size(x_train))
            #print(np.size(x_validation) != 0)
            #print(np.size(x_val_fix) != 0)
            # Dirty, dirty hack - FIXME TODO remove!!!! Just don't want the only head images in the validation set right now...
            #if np.size(x_validation) != 0 and np.size(x_val_fix) != 0:
            #    x_validation = np.concatenate((x_validation,x_val_fix))
            #elif np.size(x_validation) == 0:
            #    x_validation = x_val_fix
                #print("val shape", np.shape(x_validation))
            x_validation = x_val_fix
            x_train = x_train[ix_tr, :]
            #print(np.shape(x_train), np.shape(x_train_fix), np.size(x_train_fix), np.size(x_train))
            #print(np.size(x_train) != 0)
            #print(np.size(x_train_fix) != 0)
            if np.size(x_train_fix) != 0 and np.size(x_train) != 0:
                x_train = np.concatenate((x_train, x_train_fix))
            elif np.size(x_train) == 0:
                x_train = x_train_fix

            shape_ = len(x_train)

            sel_idx = random.sample(range(0, shape_), k=(shape_))
            #sel_idx = random.sample(range(0, shape_), k=50000)
            x_train = x_train[sel_idx, :]

        if isCombined:
            if fromPics == True:
                # all zeros/ones for head/no head
                if head:
                    y_train = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_train),4)) , np.reshape(np.ones(len(x_train)),(len(x_train),-1))), axis=1).astype(np.float32)
                    y_validation = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_validation),4)),np.reshape(np.ones(len(x_validation)),(len(x_validation),-1))), axis=1).astype(np.float32)
                else:
                    y_train = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_train),4)) , np.reshape(np.zeros(len(x_train)),(len(x_train),-1))), axis=1).astype(np.float32)
                    y_validation = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_validation),4)),np.reshape(np.zeros(len(x_validation)),(len(x_validation),-1))), axis=1).astype(np.float32)
            elif fromCaltech:
                #FIXME maybe choose a different fake gt to plot nicely. Concat head/no head labels at the end
                y_train = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_train),4)) , np.reshape(y_train_headlabels,(len(x_train),-1))), axis=1).astype(np.float32)
                y_validation = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_validation),4)),np.reshape(y_validation_headlabels,(len(x_validation),-1))), axis=1).astype(np.float32)
            else:
                # concat ones at the end, for classifier results
                y_train = np.concatenate((y_train,np.reshape(np.ones(len(x_train)),(len(x_train),-1))), axis=1).astype(np.float32)
                y_validation = np.concatenate((y_validation,np.reshape(np.ones(len(x_validation)),(len(x_validation),-1))), axis=1).astype(np.float32)
            if noPose:
               # add label for ignoring pose loss
               y_train = np.concatenate((y_train , np.reshape(np.ones(len(x_train)),(len(x_train),-1))), axis=1).astype(np.float32)
               y_validation = np.concatenate((y_validation , np.reshape(np.ones(len(x_validation)),(len(x_validation),-1))), axis=1).astype(np.float32) 
            else:
                # add label for considering pose loss
               y_train = np.concatenate((y_train , np.reshape(np.zeros(len(x_train)),(len(x_train),-1))), axis=1).astype(np.float32)
               y_validation = np.concatenate((y_validation , np.reshape(np.zeros(len(x_validation)),(len(x_validation),-1))), axis=1).astype(np.float32) 

        elif isClassifier == True:
            if head == True:
                y_train = np.ones(len(x_train))
                y_validation = np.ones(len(x_validation))
            else:
                y_train = np.zeros(len(x_train))
                y_validation = np.zeros(len(x_validation))

        if isExtended == True:
            z_train = train_set[:, 2]
            z_train = np.vstack(z_train[:]).astype(np.float32)
            z_validation = z_train[ix_val, :]
            z_train = z_train[ix_tr, :]
            z_train = z_train[sel_idx, :]
            return [x_train, x_validation, y_train, y_validation, z_train, z_validation]

        return [np.asarray(x_train), x_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestData(testPath, image_height, image_width, isGray = False, isExtended=False, isClassifier=False, fromPics=False, picsPath=None, isCombined=False, fromCaltech=False, head=True):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """
        if fromCaltech:
            noPose = True
            test_set = pd.read_pickle(testPath).values
            logging.info('[DataProcessor] test caltech shape: ' + str(test_set.shape))
            size = len(test_set[:, 0])

            np.random.seed()
            # split between test and test sets:
            x_test = test_set[:, 0]
            x_test = np.vstack(x_test[:]).astype(np.float32)
            if isGray == True:
                x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
            else:
                x_test = np.reshape(x_test, (-1, image_height, image_width, 3))

            x_test= np.swapaxes(x_test, 1, 3)
            x_test = np.swapaxes(x_test, 2, 3)

            y_test_headlabels = test_set[:, 1]


        elif fromPics == False:
            noPose = True
            head = True
            test_set = pd.read_pickle(testPath).values
            #test_set = test_set[0:50]
            logging.info('[DataProcessor] test shape: ' + str(test_set.shape))

            x_test = test_set[:, 0]
            x_test = np.vstack(x_test[:]).astype(np.float32)
            x_test_rand = []
            if isGray == True:
                x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
                # FIXME just here to see what happens if we crop the image a bit
                CropTest = False
                if CropTest:
                    for X in x_test:
                        X = np.reshape(X,(image_height, image_width))
                        X = cv2.resize(X[random.randint(0,10):random.randint(50,60), random.randint(0,15):random.randint(93,108)], (config.input_width, config.input_height), cv2.INTER_AREA)
                        x_test_rand.append(X)
                    x_test = x_test_rand
                    x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
            else:
                x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
            x_test = np.swapaxes(x_test, 1, 3)
            x_test = np.swapaxes(x_test, 2, 3)
            y_test = test_set[:, 1]
            y_test = np.vstack(y_test[:]).astype(np.float32)
        else:
            noPose = False
            x_test =[]
            size = 0
            for root, dirs, files in os.walk(picsPath):
                for file in files:
                    if file.endswith('.csv'):
                        pass
                        ''' Not used at the moment as shift in images and fixed are stored as pgm
                        X = np.genfromtxt(picsPath + file, delimiter=',')
                        X = np.reshape(X, (244, 324))
                        #cv2.imshow('test' + str(size),X.astype("uint8"))
                        #cv2.waitKey(0)
                        X = cv2.resize(X, (config.input_width, config.input_height), cv2.INTER_AREA)
                        X = X.astype("uint8")
                        #print(X)
                        #cv2.imshow('test' + str(size),X)
                        #cv2.waitKey(0)
                        x_test.append(X)
                        size+=1'''
                    if file.endswith('.pgm'):
                        #print(file)
                        X = cv2.imread(root +'/'+ file, 0)
                        try:
                            X = np.reshape(X, (244, 324))
                            X = X[30:210:3,0:324:3]
                            #X = X[0:200,20:300]
                        except ValueError:
                            X = np.reshape(X, (config.input_height, config.input_width))
                        #cv2.imshow('test' + str(size),X.astype("uint8"))
                        #cv2.waitKey(0)
                        X = cv2.resize(X, (config.input_width, config.input_height), cv2.INTER_AREA)
                        X = X.astype("uint8")
                        #print(X)
                        #cv2.imshow('test' + str(size),X)
                        #cv2.waitKey(0)
                        x_test.append(X)
                        size+=1
            x_test = np.asarray(x_test)
            x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
            x_test = np.swapaxes(x_test, 1, 3)
            x_test = np.swapaxes(x_test, 2, 3)
            logging.info('[DataProcessor] test pics number: ' + str(size))

        if isExtended ==True:
            z_test = test_set[:, 2]
            z_test = np.vstack(z_test[:]).astype(np.float32)
            return [x_test, y_test, z_test]
        if isCombined:
            # put ones/zeros for classifier result to end of pose gt (if no head, no pose)
            if fromPics == True:
                #y_test = np.zeros((len(x_test),5)).astype(np.float32)
                if head:
                    y_test = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_test),4)),np.reshape(np.ones(len(x_test)),(len(x_test),-1))), axis=1).astype(np.float32)
                else:
                    y_test = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_test),4)),np.reshape(np.zeros(len(x_test)),(len(x_test),-1))), axis=1).astype(np.float32)
            elif fromCaltech:
                #FIXME maybe choose a different fake gt to plot nicely. Concat head/no head labels at the end
                y_test = np.concatenate((np.array((0,0,0,0))*np.ones((len(x_test),4)) , np.reshape(y_test_headlabels,(len(x_test),-1))), axis=1).astype(np.float32)
            else:
                y_test = np.concatenate((y_test,np.reshape(np.ones(len(x_test)),(len(x_test),-1))), axis=1).astype(np.float32)
            if noPose:
               # add label for ignoring pose loss
               y_test = np.concatenate((y_test , np.reshape(np.ones(len(x_test)),(len(x_test),-1))), axis=1).astype(np.float32)
            else:
                # add label for considering pose loss
               y_test = np.concatenate((y_test , np.reshape(np.zeros(len(x_test)),(len(x_test),-1))), axis=1).astype(np.float32)
        elif isClassifier == True:
            if head == True:
                y_test = np.ones(len(x_test))
            else:
                y_test = np.zeros(len(x_test))


        return [x_test, y_test]

    @staticmethod
    def ProcessInferenceData(images, image_height, image_width, isGray=False):
        """Converts a list of images into a format suitable fot inference

            Parameters
            ----------
            images : list
                list of images
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB

            Returns
            -------
            list
                list of video frames and list of labels (poses, which are garbage)
            """

        x_test = np.stack(images, axis=0).astype(np.float32)
        if isGray == True:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
        else:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = [0, 0, 0, 0] * len(x_test)
        y_test = np.vstack(y_test[:]).astype(np.float32)
        y_test = np.reshape(y_test, (-1, 4))


        return [x_test, y_test]

    @staticmethod
    def CreateGreyPickle(trainPath, image_height, image_width, file_name):
        """Converts Dario's RGB dataset to a gray + vignette dataset

            Parameters
            ----------
            images : list
                list of images
            image_height : int
                Please...
            image_width : int
                Please...
            file_name : str
                name of the new .pickle

            """
        train_set = pd.read_pickle(trainPath).values
        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))

        # split between train and test sets:
        x_train = train_set[:, 0]
        x_train = np.vstack(x_train[:])
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

        it = ImageTransformer()

        x_train_grey = []
        sigma = 50
        mask = it.ApplyVignette(image_width, image_width, sigma)

        for i in range(len(x_train)):
            gray_image = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY)
            gray_image = gray_image * mask[24:84, 0:108]
            gray_image = gray_image.astype(np.uint8)
            x_train_grey.append(gray_image)

        y_train = train_set[:, 1]

        df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train})
        df.to_pickle(file_name)







