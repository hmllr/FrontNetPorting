#### Generates Head/NoHead samples from hollywood dataset (https://www.di.ens.fr/willow/research/headdetection/) #####
# There are annotation files in xml format that tell us where the heads are - 
# so for the head samples we crop randomly around the head (in a way we have a good chance it fits in the image)
# for the no head images we try to fit a randomly but more likely as big as possible sample in the left/right/bottom/top part of the image with no head.

import json
import os
import cv2
import pandas as pd
import numpy as np
import sys
import xml.etree.ElementTree as ET
sys.path.append("../DataProcessing/")
from ImageEffects import ImageEffects
sys.path.append("../")
import config

debug = False



def DEBUGprint(*objects):
    if debug:
        print(objects)

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

# FIXME duplicate, in DatasetCreator as well
def make_himax_similar(img, input_size=False):
    # FIXME duplicate code - in DatasetCreator as well
    img = cv2.LUT(img, gammaLUT)
    img = cv2.GaussianBlur(img,(5,5),0)
    if input_size:
        img = cv2.resize(img, (config.input_width, config.input_height), cv2.INTER_NEAREST)
        # take rectangular out of square mask because himax camera is a square, so the effect we want to reproduce is as well
        # take the rectangular at the bottom middle as the image might be cropped there (see config.h on pulp)
        img = img *  vignetteMaskWeak[224:284, 158:266]
        img = cv2.convertScaleAbs(img)
    else:
        img = cv2.resize(img, (config.himax_width, config.himax_height), cv2.INTER_AREA)
        # take rectangular out of square mask because himax camera is a square, so the effect we want to reproduce is as well
        img = img *  vignetteMask[40:284, 0:324]
        # there is no use in keeping floats as the himax values are ints as well
        img = cv2.convertScaleAbs(img)
        img = cv2.resize(img, (config.input_width, config.input_height), cv2.INTER_NEAREST)
    return img

def crop_to_prop_to_input(img):
    height, width = np.shape(img)
    crop_xmin = min(abs(int(np.random.normal(0, 3*(width - config.input_width)))), width - config.input_width)
    crop_xmax = crop_xmin + config.input_width + width - min(abs(int(np.random.normal(0,3*width))), width)
    prop_height = int(float(crop_xmax - crop_xmin)/float(config.input_width)*float(config.input_height))
    crop_ymin = np.random.randint(0, height - prop_height)
    crop_ymax = crop_ymin + prop_height
    img = img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
    return img

gammaLUT = ImageEffects.GetGammaLUT(0.6)
vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)
vignetteMaskWeak = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width, sigma=300)
if debug:
    cv2.imshow("vignette", vignetteMask)
    cv2.waitKey(0)
    cv2.imshow("vignetteWeak", vignetteMaskWeak)
    cv2.waitKey(0)

frame_rate = 10 # how many pics to skip before taking next sample
dark_frame_max = 5
clip_border = 5
base_path = "/home/hanna/Documents/ETH/masterthesis/Hollywood/HollywoodHeads/HollywoodHeads/"
annotations_path = base_path + "Annotations/"
imgs_path =  base_path + "JPEGImages/"
trainsplit_path = base_path + "Splits/train.txt"
valsplit_path = base_path + "Splits/val.txt"
testsplit_path = base_path + "Splits/test.txt"
converted_head_imgs_path = base_path + "PGMImages/Head/"
converted_nohead_imgs_path = base_path + "PGMImages/NoHead/"
# load the dataset

with open(trainsplit_path) as fp:
    img_name = fp.readline()[:-1]
    count_line = 0
    while img_name:
        img_path = imgs_path + img_name + ".jpeg"
        DEBUGprint(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #DEBUGprint(img)
        DEBUGprint(np.shape(img))
        img_height, img_width = np.shape(img)
        clip_top = clip_bottom = clip_left = clip_right = clip_border
        for i in range(0,img_height-1):
            #DEBUGprint("SUM", np.sum(img[1,:]))
            if np.sum(img[0,:]) < img_height*dark_frame_max:
                img = img[1:,:]
                clip_top += 1
            else:
                break
        for i in range(0,img_width-1):
            if np.sum(img[:,0]) < img_width*dark_frame_max:
                img = img[:,1:]
                clip_left += 1
            else:
                break
        img_height, img_width = np.shape(img)
        # there is no use in keeping black pictures
        if img_height < config.input_height or img_width < config.input_width:
            img_name = fp.readline()[:-1]
            continue

        for i in range(0,img_height):
            if np.sum(img[-1,:]) < img_height*dark_frame_max:
                img = img[:-1,:]
                clip_bottom += 1
            else:
                break
        for i in range(0,img_width):
            if np.sum(img[:,-1]) < img_width*dark_frame_max:
                img = img[:,:-1]
                clip_right += 1
            else:
                break

        img = img[clip_border:-clip_border, clip_border:-clip_border]

        img_height, img_width = np.shape(img)


        all_xmax = all_ymax = 0
        all_xmin = img_width
        all_ymin = img_height
        DEBUGprint(img_width, img_height)
        #cv2.imshow(img_name, img)
        #cv2.waitKey(0)
        tree = ET.parse(annotations_path + img_name + ".xml")
        root = tree.getroot()
        count = 0
        for head in root.findall('object'):
            head_too_big = False
            try:
                difficult = head.find('difficult').text
                DEBUGprint("Difficult ",difficult)
                if difficult == '1':
                    continue
            except:
                pass # just continue if difficult attribute is missing
            for element in head.findall('bndbox'):

                xmin = max(0,int(float(element.find('xmin').text)) - clip_left)
                ymin = max(0,int(float(element.find('ymin').text)) - clip_top)
                xmax = max(0,int(float(element.find('xmax').text)) - clip_left)
                ymax = max(0,int(float(element.find('ymax').text)) - clip_top)
                all_xmin = min(all_xmin, xmin)
                all_ymin = min(all_ymin, ymin)
                all_xmax = max(all_xmax, xmax)
                all_ymax = max(all_ymax, ymax)
                DEBUGprint(xmin, xmax, ymin, ymax)
                DEBUGprint(all_xmin, all_xmax, all_ymin, all_ymax)

                # crop image for head img
                head_width = xmax - xmin
                head_height = ymax - ymin
                #DEBUGprint(head_width, head_height)
                if head_height == 0 or head_width == 0:
                    continue
                head_img_xmin = max(0, xmin - head_width/2 - np.random.randint(0,4*head_width))
                head_img_xmax = min(img_width, xmax + head_width/2 + np.random.randint(0,4*head_width)) 
                crop_width = head_img_xmax - head_img_xmin
                crop_height = int(crop_width*float(config.input_height)/float(config.input_width))
                if crop_height <= head_height:
                    head_too_big = True
                    continue
                head_img_ymin = max(0, ymin - np.random.randint(0,crop_height - head_height))
                head_img_ymax = head_img_ymin + crop_height
                DEBUGprint(head_img_xmin, head_img_xmax, head_img_ymin, head_img_ymax, crop_width, crop_height)
                if head_img_ymax > img_height:
                    head_img_ymax = img_height
                    head_img_ymin = img_height - crop_height
                    if head_img_ymin < 0:
                        head_too_big = True
                        continue
                DEBUGprint(int(head_img_xmin), int(head_img_xmax), int(head_img_ymin), int(head_img_ymax), crop_width, crop_height)
                img_head_crop = img[int(head_img_ymin):int(head_img_ymax), int(head_img_xmin):int(head_img_xmax)]
                #DEBUGprint(img_head_crop)
                img_head_crop = make_himax_similar(img_head_crop)
                if debug:
                    cv2.imshow(img_name + "head", img_head_crop)
                    cv2.waitKey(0)
                cv2.imwrite(converted_head_imgs_path + img_name + "head" + str(count) + ".pgm", img_head_crop)
                count += 1


        if not head_too_big:
            #crop image for no head
            DEBUGprint("config HW: ", config.input_height, config.input_width)
            try:
                img_nohead_crop_left = img[:,:int(all_xmin)]
                height, width = np.shape(img_nohead_crop_left)
                DEBUGprint("HW:", height, width)
                if height > config.input_height and width > config.input_width:
                    img_nohead_crop_left = crop_to_prop_to_input(img_nohead_crop_left)
                    img_nohead_crop_left = make_himax_similar(img_nohead_crop_left, input_size=True)
                    if debug:
                        cv2.imshow(img_name + "left", img_nohead_crop_left)
                        cv2.waitKey(0)
                    cv2.imwrite(converted_nohead_imgs_path + img_name + "nohead_left" + ".pgm", img_nohead_crop_left)
            except Exception as e:
                DEBUGprint(e)
            try:
                img_nohead_crop_right = img[:,int(all_xmax):]
                height, width = np.shape(img_nohead_crop_right)
                DEBUGprint("HW:", height, width)
                if height > config.input_height and width > config.input_width:
                    img_nohead_crop_right = crop_to_prop_to_input(img_nohead_crop_right)
                    img_nohead_crop_right = make_himax_similar(img_nohead_crop_right, input_size=True)
                    if debug:
                        cv2.imshow(img_name +"right", img_nohead_crop_right)
                        cv2.waitKey(0)
                    cv2.imwrite(converted_nohead_imgs_path + img_name + "nohead_right" + ".pgm", img_nohead_crop_right)
            except Exception as e:
                DEBUGprint(e)
            try:
                img_nohead_crop_bottom = img[int(all_ymax):,:]
                height, width = np.shape(img_nohead_crop_bottom)
                DEBUGprint("HW:", height, width)
                if height > config.input_height and width > config.input_width:
                    img_nohead_crop_bottom = crop_to_prop_to_input(img_nohead_crop_bottom)
                    img_nohead_crop_bottom = make_himax_similar(img_nohead_crop_bottom, input_size=True)
                    if debug:
                        cv2.imshow(img_name + "bottom", img_nohead_crop_bottom)
                        cv2.waitKey(0)
                    cv2.imwrite(converted_nohead_imgs_path + img_name + "nohead_bottom" + ".pgm", img_nohead_crop_bottom)
            except Exception as e:
                DEBUGprint(e)
            try:
                img_nohead_crop_top = img[:int(all_ymin),:]
                height, width = np.shape(img_nohead_crop_top)
                DEBUGprint("HW:", height, width)
                if height > config.input_height and width > config.input_width:
                    img_nohead_crop_top = crop_to_prop_to_input(img_nohead_crop_top)
                    img_nohead_crop_top = make_himax_similar(img_nohead_crop_top, input_size=True)
                    if debug:
                        cv2.imshow(img_name + "top", img_nohead_crop_top)
                        cv2.waitKey(0)
                    cv2.imwrite(converted_nohead_imgs_path + img_name + "nohead_top" + ".pgm", img_nohead_crop_top)
            except Exception as e:
                DEBUGprint(e)


        for i in range(0,frame_rate):
            img_name = fp.readline()[:-1]
            DEBUGprint(img_name)
            if img_name:
                count_line += 1
                if count_line % 1000 == 0:
                    print("Processed " + str(count_line) + " Files")
            else:
                break
print("Finished")
'''
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

    SaveToDataFrame(x_dataset,y_dataset, dataset + "_caltech.pickle")'''