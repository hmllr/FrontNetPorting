from __future__ import print_function
from PreActBlockSimple import PreActBlockSimple
from PreActBlock import PreActBlock
from FrontNet import FrontNet
from Dronet import Dronet
from FindNet import FindNet


from DataProcessor import DataProcessor
from ModelTrainerETH import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import nemo

import logging

import sys
sys.path.append("../")
import config


def Parse(parser):

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # [NeMO] Model saving/loading improved for convenience
    parser.add_argument('--save-model', default=None, type=str,
                        help='for saving the model')
    parser.add_argument('--load-model', default=None, type=str,
                        help='for loading the model')

    parser.add_argument('--load-trainset', default=None, type=str,
                        help='for loading the train dataset')

    parser.add_argument('--load-testset', default=None, type=str,
                        help='for loading the test dataset')

    # [NeMO] If `quantize` is False, the script operates like the original PyTorch example
    parser.add_argument('--quantize', default=False, action="store_true",
                        help='for loading the model')
    # [NeMO] The training regime (in JSON) used to store all NeMO configuration.
    parser.add_argument('--regime', default=None, type=str,
                        help='for loading the model')
    parser.add_argument('--gray', default=None, type=int,
                        help='for choosing the model')
    parser.add_argument('--tensorboard', default=None, type=int,
                        help='for enabling visualization during training')
    parser.add_argument('--predictonly', default=None, type=int,
                        help='for only predicting(need --load-model)')
    parser.add_argument('--pic', default=None, type=str,
                        help='for loading a picture')
    parser.add_argument('--load-trainpics-nohead', default=None, type=str,
                        help='for loading no head train pictures')
    parser.add_argument('--load-trainpics-head', default=None, type=str,
                        help='for loading head train pictures')
    parser.add_argument('--load-testpics-head', default=None, type=str,
                        help='for loading head test pictures')
    parser.add_argument('--load-testpics-nohead', default=None, type=str,
                        help='for loading no head test pictures')
    parser.add_argument('--singlepic', default=None, type=int,
                        help='for testing single picture')
    parser.add_argument('--load-traincaltech', default=None, type=str,
                        help='for loading train caltech (as pickle)')
    parser.add_argument('--load-testcaltech', default=None, type=str,
                        help='for loading test caltech (as pickle)')
    parser.add_argument('--load-data', default=None, type=str,
                        help='for loading whole data (as pickle)')
    args = parser.parse_args()

    return args

def HelperConcat2(base, extend):
    try:
        #print(np.shape(base), np.shape(extend))
        return np.concatenate((base,extend))
    except Exception as e:
        print(e)
        return extend

def LoadData(args):

    if args.load_data is None:
        if args.gray is not None:
            x_train = y_train = x_validation = y_validation = x_test = y_test = None
            # load training set
            #   load head pose labeled data ("darios dataset")
            if args.load_trainset is not None:
                [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
                    args.load_trainset, 60, 108, isGray=True, isClassifier=True, fromPics=False)
                print("train shape", np.shape(x_train))
            #   load no head pictures from DroNet and recorded by me (Hanna) 
            if args.load_trainpics_nohead is not None:
                [x_train_nohead, x_validation_nohead, y_train_nohead, y_validation_nohead] = DataProcessor.ProcessTrainData(
                    None, 60, 108, isGray=True, isClassifier=True, onlyHimax=True, picsPath=args.load_trainpics_nohead, head=False)
                x_train = HelperConcat2(x_train, x_train_nohead)
                y_train = HelperConcat2(y_train, y_train_nohead)
                x_validation = HelperConcat2(x_validation, x_validation_nohead)
                y_validation = HelperConcat2(y_validation, y_validation_nohead)
                print("train shape", np.shape(x_train))
            #   load head pictures recorded by me (Hanna)
            if args.load_trainpics_head is not None: 
                [x_train_head_pics, x_validation_head_pics, y_train_head_pics, y_validation_head_pics] = DataProcessor.ProcessTrainData(
                    None, 60, 108, isGray=True, isClassifier=True, onlyHimax=True, picsPath=args.load_trainpics_head, head=True)
                x_train = HelperConcat2(x_train, x_train_head_pics)
                y_train = HelperConcat2(y_train, y_train_head_pics)
                x_validation = HelperConcat2(x_validation, x_validation_head_pics)
                y_validation = HelperConcat2(y_validation, y_validation_head_pics)
                print("train shape", np.shape(x_train))
            #   load head/no head pictures from caltech pedestrian dataset (preprocessed in caltechReader.py)
            if args.load_traincaltech is not None:
                [x_train_caltech, x_validation_caltech, y_train_caltech, y_validation_caltech] = DataProcessor.ProcessTrainData(
                    args.load_traincaltech, 60, 108, isGray=True, isClassifier=True, fromCaltech=True)
                x_train = HelperConcat2(x_train, x_train_caltech)
                y_train = HelperConcat2(y_train, y_train_caltech)
                x_validation = HelperConcat2(x_validation, x_validation_caltech)
                y_validation = HelperConcat2(y_validation, y_validation_caltech)
            # load test set
            #   load head pose labeled data ("darios dataset")
            if args.load_testset is not None:
                [x_test, y_test] = DataProcessor.ProcessTestData(args.load_testset, 60, 108, isGray=True, isClassifier=True, fromPics=False)
            #   load head pictures from Hollywood and recorded by me (Hanna) 
            if args.load_testpics_head is not None:
                [x_test_head, y_test_head] = DataProcessor.ProcessTestData(
                    args.load_testset, 60, 108, isGray=True, isClassifier=True, fromPics=True, picsPath=args.load_testpics_head, head=True)
                x_test = HelperConcat2(x_test, x_test_head)
                y_test = HelperConcat2(y_test, y_test_head)
            #   load no head test pictures from Hollywood and recorded by me (Hanna) 
            if args.load_testpics_nohead is not None:
                [x_test_nohead, y_test_nohead] = DataProcessor.ProcessTestData(
                    args.load_testset, 60, 108, isGray=True, isClassifier=True, fromPics=True, picsPath=args.load_testpics_nohead, head=False)
                x_test = HelperConcat2(x_test, x_test_nohead)
                y_test = HelperConcat2(y_test, y_test_nohead)
            #   load head/no head pictures from caltech pedestrian dataset (preprocessed in caltechReader.py)
            if args.load_testcaltech is not None:
                [x_test_caltech, y_test_caltech] = DataProcessor.ProcessTestData(
                    args.load_testcaltech, 60, 108, isGray=True, isClassifier=True, fromCaltech=True)
                x_test = HelperConcat2(x_test, x_test_caltech)
                y_test = HelperConcat2(y_test, y_test_caltech)
            debug = False
            if debug:
                x_test = np.concatenate((10*np.ones(np.shape(x_test)), x_test)).astype(np.float32)
                y_test = np.concatenate((np.zeros(np.shape(y_test)), y_test)).astype(np.float32)
            print("train shape", np.shape(x_train))

            # concatenate train/validation sets
            '''if args.load_traincaltech is not None:
                x_train = np.concatenate((x_train_head, x_train_head_pics, x_train_nohead, x_train_caltech))
                y_train = np.concatenate((y_train_head, y_train_head_pics, y_train_nohead, y_train_caltech)).astype(np.float32)
                x_validation = np.concatenate((x_validation_head, x_validation_head_pics, x_validation_nohead, x_validation_caltech))
                y_validation = np.concatenate((y_validation_head, y_validation_head_pics, y_validation_nohead, y_validation_caltech)).astype(np.float32)
            else:
                x_train = np.concatenate((x_train_head, x_train_head_pics, x_train_nohead))
                y_train = np.concatenate((y_train_head, y_train_head_pics, y_train_nohead)).astype(np.float32)
                x_validation = np.concatenate((x_validation_head, x_validation_head_pics, x_validation_nohead))
                y_validation = np.concatenate((y_validation_head, y_validation_head_pics, y_validation_nohead)).astype(np.float32)
                
            # concatenate test sets (try/except just for debug/test reasons)
            if args.load_testcaltech is not None:
                x_test = np.concatenate((x_test_head, x_test_nohead, x_test_caltech))
                y_test = np.concatenate((y_test_head, y_test_nohead, y_test_caltech)).astype(np.float32)
            else:
                try:
                    x_test = np.concatenate((x_test_head, x_test_nohead))
                    y_test = np.concatenate((y_test_head, y_test_nohead)).astype(np.float32)
                except:
                    # just for testing that plain/random images are recognized as no-head
                    #x_test = np.concatenate((10*np.ones(np.shape(x_test_nohead)), x_test_nohead)).astype(np.float32)
                    #y_test = np.concatenate((np.zeros(np.shape(y_test_nohead)), y_test_nohead)).astype(np.float32)
                    x_test = x_test_nohead.astype(np.float32)
                    y_test = y_test_nohead.astype(np.float32)'''

            np.save("trainset_classifier_x", x_train)
            np.save("trainset_classifier_y", y_train)

            np.save("validationset_classifier_x", x_validation)
            np.save("validationset_classifier_y", y_validation)

            np.save("testset_classifier_x", x_test)
            np.save("testset_classifier_y", y_test)
    else:
            x_train = np.load(args.load_data + "trainset_classifier_x.npy")
            y_train = np.load(args.load_data + "trainset_classifier_y.npy")

            x_validation = np.load(args.load_data + "validationset_classifier_x.npy")
            y_validation = np.load(args.load_data + "validationset_classifier_y.npy")

            x_test = np.load(args.load_data + "testset_classifier_x.npy")
            y_test = np.load(args.load_data + "testset_classifier_y.npy")

    print("Train size: ", np.shape(x_train), "Validation size: ", np.shape(x_validation), np.shape(y_validation), "Test size: ", np.shape(x_test))

    training_set = Dataset(x_train.astype(np.float32), y_train.astype(np.float32), True, isClassifier=True) 
    #validation_set = Dataset(x_validation.astype(np.float32), y_validation.astype(np.float32), isClassifier=True)
    validation_set = Dataset(x_validation[:,:,:,:].astype(np.float32), y_validation[:].astype(np.float32), isClassifier=True)
    test_set = Dataset(x_test.astype(np.float32), y_test.astype(np.float32), isClassifier=False)

    # Parameters
    num_workers = 6
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': num_workers}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': num_workers}
    test_loader = data.DataLoader(test_set, **params)

    return train_loader, validation_loader, test_loader

def ExportONXX(model, model_inner, val_loader, validate):
    #print(model)
    #model_inner = model
    #print("before export")
    #model.graph.print_jit_graph()
    nemo.utils.export_onnx("frontnet/model_int.onnx", model, model_inner, (1, 60, 108), perm=None)
    #print("After export")
    #model.graph.print_jit_graph()
    b_in, b_out, acc = nemo.utils.get_intermediate_activations(model_inner, validate, val_loader)
    if acc != None:
        logging.info("After integerize: %.2f%%" % (100*acc[0]))

    try:
        os.makedirs('frontnet/golden')
    except Exception:
        pass

    from collections import OrderedDict
    dory_dict = OrderedDict([])
    #for key, value in b_in.items():
    #    print(key, value)
    # save super-node outputs as CSV files as golden reference
    bidx = 0
    #for x in model_inner.named_modules():
    #    print(x)
    for n,m in model_inner.named_modules():
        try:
            #print("n:%s" % n)
            #print("bidx:%d" % bidx)
            actbuf = b_in[n][0][bidx].permute((1,2,0))
        except RuntimeError:
            actbuf = b_in[n][0][bidx]
        np.savetxt("frontnet/golden/golden_input_%s.txt" % n, actbuf.cpu().numpy().flatten(), header="input (shape %s)" % (list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')
    for n,m in model_inner.named_modules():
        try:
            actbuf = b_out[n][bidx].permute((1,2,0))
        except RuntimeError:
            actbuf = b_out[n][bidx]
        np.savetxt("frontnet/golden/golden_%s.txt" % n, actbuf.cpu().numpy().flatten(), header="%s (shape %s)" % (n, list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')
    args = Parse(parser)

    torch.manual_seed(args.seed)

    # [NeMO] Setup of console logging.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="log.txt",
                        filemode='w')


    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    if args.singlepic==None:
        train_loader, validation_loader, test_loader = LoadData(args)

    # [NeMO] Loading of the JSON regime file.
    regime = {}
    if args.regime is None:
        print("ERROR!!! Missing regime JSON.")
        raise Exception
    else:
        with open(args.regime, "r") as f:
            rr = json.load(f)
        for k in rr.keys():
            try:
                regime[int(k)] = rr[k]
            except ValueError:
                regime[k] = rr[k]

    if args.gray is not None:
        model = FindNet(PreActBlockSimple, [1, 1, 1], True, isClassifier=True)
    else:
        model = FindNet(PreActBlockSimple, [1, 1, 1], False)

    # [NeMO] This used to preload the model with pretrained weights.
    if args.load_model is not None:
        ModelManager.Read(args.load_model, model)

    if args.singlepic:
        #X = cv2.imread(args.pic,0)
        X = np.genfromtxt(args.pic, delimiter=',')
        print(np.size(X))

        X = np.reshape(X, (244, 324))
        print(np.size(X))
        print(X)
        cv2.imshow('test',X.astype("uint8"))
        cv2.waitKey(0)
        X = cv2.resize(X, (config.input_width, config.input_height), cv2.INTER_AREA)
        X = X.astype("uint8")
        print(X)
        cv2.imshow('test2',X)
        cv2.waitKey(0)
        X = np.reshape(X, (-1, 60,108, 1))
        X = np.swapaxes(X, 1, 3)
        X = np.swapaxes(X, 2, 3)
        model.eval()
        with torch.no_grad():
            print(model(torch.from_numpy(X).float().round()))
    else:
        trainer = ModelTrainer(model, args, regime)
        if args.quantize:
            trainer.Quantize(validation_loader, test_loader, train_loader)
            print(model)
            ExportONXX(model, model, validation_loader, trainer.ValidateSingleEpoch)

        if args.predictonly is None:
            if args.tensorboard is not None:
                import torchvision
                from torch.utils.tensorboard import SummaryWriter
                images, labels = next(iter(train_loader))
                grid = torchvision.utils.make_grid(images)
                tb = SummaryWriter(comment="FindNetGray_classifier_2res")#(comment="COBNRLMPL2BNRELUL2(64x64/2)BNRLL3APFCRLFCRL")
                tb.add_image('images', grid)
                #tb.add_graph(model, images)

                trainer.Train(train_loader, validation_loader, tb)
            else:
                trainer.Train(train_loader, validation_loader)
        trainer.Predict(test_loader)
        
        if args.save_model is not None:
            #torch.save(trainer.model.state_dict(), args.save_model)
            ModelManager.Write(trainer.GetModel(), 100, args.save_model)
        trainer.Predict(test_loader)
        print(model)
if __name__ == '__main__':
    main()
