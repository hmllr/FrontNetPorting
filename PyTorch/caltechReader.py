#### Downloads caltech dataset and stores a himax similar train and test set in pickle format #####
# labels are head/no head - where it is assumed that that pedestrians have a head.
# as the used resolution is quite low we filter out small pedestrians and only use pictures without any as no head and 
# with bigger size than min_height_pedestrian/min_width_pedestrian as head
# (the 'posv' field means position visible), and its format is something like starting x,y and then height,width
import dbcollection as dbc
import json
import os
import cv2
import pandas as pd
import sys
sys.path.append("../DataProcessing/")
from ImageEffects import ImageEffects
sys.path.append("../")
import config

min_height_pedestrian = 300
min_width_pedestrian = 60

# FIXME duplicate, in DatasetCreator as well
def SaveToDataFrame(x_dataset, y_dataset, datasetName):

    if len(x_dataset) == len(y_dataset):
        print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
        df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
    else:
        if len(y_dataset) == 2:
            print("dataset ready x:{} y:{} z:{}".format(len(x_dataset), len(y_dataset[0]), len(y_dataset[1])))
            df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z':y_dataset[1]})
        elif len(y_dataset) == 3:
            print("dataset ready x:{} y:{} z:{} w:{}".format(len(x_dataset), len(y_dataset[0]), len(y_dataset[1]), len(y_dataset[2])))
            df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z':y_dataset[1], 'w':y_dataset[2]})
        # add your own line if you have more tracking markers.....
        
    print("dataframe ready")
    df.to_pickle(datasetName)

debug = False
# load the dataset
caltech_ped_clean = dbc.load('caltech_pedestrian', 'detection_clean')
first_time = False # no nead to transform if already done
gammaLUT = ImageEffects.GetGammaLUT(0.6)
vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)
trainset = [0,1,2,3,4,5,6,7,8]
testset = [9,10]
names = ["train", "test"]
for dataset in names:
    if dataset == "train":
        sets = trainset
    elif dataset == "test":
        sets = testset
    else:
        print("Error, please define which sets belong to your dataset")
    counthead = 0
    countnohead = 0
    countskip = 0
    x_dataset = []
    y_dataset = []
    for setNumber in sets:
        for folderNumber in range(0,100):
                #try:
                for root, dirs, files in os.walk('/home/hanna/dbcollection/downloads/caltech_pedestrian/extracted_data/set' + "{:02d}".format(setNumber) +'/V' + "{:03d}".format(folderNumber) +'/images'):
                    print("converting set" + "{:02d}".format(setNumber) +" folder V" + "{:03d}".format(folderNumber))
                    for file in files:
                        if file.endswith('.jpg'):
                            # if we only want every 30th image uncomment here (as it is a 30Hz recording they are very similar otherwise)
                            #if int(os.path.splitext(file)[0][1:])%30:
                            #    continue
                            if debug:
                                print(file)
                                print(root +'/'+ file)
                            cv_image = cv2.imread(root +'/'+ file)
                            if debug:
                                print(cv_image.shape)
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                            cv_image = cv2.LUT(cv_image, gammaLUT)
                            cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
                            cv_image = cv2.resize(cv_image, (config.himax_width, config.himax_height), cv2.INTER_AREA)
                            # take rectangular out of square mask because himax camera is a square, so the effect we want to reproduce is as well
                            cv_image = cv_image *  vignetteMask[40:284, 0:324]
                            # there is no use in keeping floats as the himax values are ints as well
                            cv_image = cv2.convertScaleAbs(cv_image)
                            cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_NEAREST)
                            if first_time:
                                cv2.imwrite(root +'/himax_alike_' + os.path.splitext(file)[0] + '.pgm',cv_image.astype("uint8"))
                            
                            number = int(os.path.splitext(file)[0][1:])
                            if debug:
                                print("{:05d}".format(number))
                            person = False
                            with open('/home/hanna/dbcollection/downloads/caltech_pedestrian/extracted_data/set' + "{:02d}".format(setNumber) +'/V' + "{:03d}".format(folderNumber) +'/annotations/I' + "{:05d}".format(number) + '.json') as json_file:
                                data = json.load(json_file)
                                skip = False
                                for p in data:
                                    try:
                                        height = p['posv'][3]
                                        width = p['posv'][2]
                                    except:
                                        skip = True
                                        height = width = 0 # corrupt data - don't use
                                    if  p['lbl'] == 'person' and (height > min_height_pedestrian or width > min_width_pedestrian):
                                        if debug:
                                            print("{:05d}".format(number))
                                            print(height)
                                            #cv2.imshow("person", cv_image)
                                            #cv2.waitKey(0)
                                        person = True
                                        break
                                    else:
                                        skip = True # don't want ambigous pics in dataset
                            # order of else-if important (person and skip might both be True if there are multiple people)
                            if person:
                                y_dataset.append(1)
                                counthead += 1
                                x_dataset.append(cv_image)
                            elif skip:
                                countskip += 1
                            else:
                                countnohead += 1
                                if countnohead%10 == 0:
                                    y_dataset.append(0)
                                    x_dataset.append(cv_image)
                #except Exception as e:
                #    print(e)
                #    break # arrived at set/folder number higher than we have folders
    print("Head pics: ", counthead, "NoHead pics: ", countnohead, "skipped pics: ", countskip)

    SaveToDataFrame(x_dataset,y_dataset, dataset + "_caltech.pickle")