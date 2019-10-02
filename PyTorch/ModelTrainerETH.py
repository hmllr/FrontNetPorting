import torch.nn as nn
import torch
import numpy as np
from ValidationUtils import RunningAverage
from ValidationUtils import MovingAverage
from DataVisualization import DataVisualization
from EarlyStopping import EarlyStopping
from ValidationUtils import Metrics
import nemo
import logging
from collections import OrderedDict

class ModelTrainer:
    def __init__(self, model, args, regime):
        self.num_epochs = args.epochs
        self.args = args
        self.model = model
        self.regime = regime
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logging.info("[ModelTrainer] " + device)
        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.criterion_class = nn.BCELoss()
        if self.args.quantize:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=float(regime['lr']), weight_decay=float(regime['weight_decay']))
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.relax = None
        self.folderPath = "Models/"
        self.classifierLossFactor = 1
        #self.train_losses_log = []
        #self.val_losses_log = []

    def GetModel(self):
        return self.model

    def Quantize(self, validation_loader):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: Before quantization process: %f" % acc)

        # [NeMO] This call "transforms" the model into a quantization-aware one, which is printed immediately afterwards.
        self.model = nemo.transform.quantize_pact(self.model)
        logging.info("[ModelTrainer] Model: %s", self.model)
        # [NeMO] NeMO re-training usually converges better using an Adam optimizer, and a smaller learning rate
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.regime['lr']),
        #                              weight_decay=float(self.regime['weight_decay']))

        # [NeMO] DNNs that do not employ batch normalization layers nor have clipped activations (e.g. ReLU6) require
        # an initial calibration to transfer to a quantization-aware version. This is used to calibrate the scaling
        # parameters of quantization-aware activation layers to the point defined by the maximum activation value
        # seen during a validation run. DNNs that employ BN or ReLU6 (or both) do not require this operation, as their
        # activations are already statistically bounded in terms of dynamic range.
        # logging.info("[ModelTrainer] Gather statistics for non batch-normed activations")

        # dict = {"layer1.conv1": "layer1.bn1",
        # "layer1.conv2": "layer1.bn2",
        # "layer2.conv2": "layer2.bn2"}
        # self.model.fold_bn(OrderedDict(dict))
        # dict = {"layer1.conv1": "layer1.bn2",
        #     "layer1.conv2": "layer2.bn1",
        #     "layer1.shortcut": "layer2.bn1",
        #     "layer2.conv1": "layer2.bn2",
        #     "layer2.conv2": "layer3.bn1",
        #     "layer2.shortcut": "layer3.bn1",
        #     "layer3.conv1": "layer3.bn2"}
        # self.model.fold_bn(OrderedDict(dict))

        # first block folding

        '''self.model.fold_bn_withinv(bn_dict={
            "conv": "layer1.bn1",
            "layer1.conv1": "layer1.bn2",
        },
            bn_inv_dict={
                "layer1.shortcut": "layer1.bn1",
            })

        # second block folding
        self.model.fold_bn_withinv(bn_dict={
            "layer1.conv2": "layer2.bn1",
            "layer1.shortcut": "layer2.bn1",
            "layer2.conv1": "layer2.bn2",
        },
            bn_inv_dict={
                "layer2.shortcut": "layer2.bn1",
            })

        # third block folding
        self.model.fold_bn_withinv(bn_dict={
            "layer2.conv2": "layer3.bn1",
            "layer2.shortcut": "layer3.bn1",
            "layer3.conv1": "layer3.bn2",
        },
            bn_inv_dict={
                "layer3.shortcut": "layer3.bn1",
            })
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: After BN folding: %f" % acc)'''


        self.model.set_statistics_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        self.model.unset_statistics_act()
        self.model.reset_alpha_act()
        logging.info("[ModelTrainer] statistics %.2f" % acc)

        precision_rule = self.regime['relaxation']

        # [NeMO] Change precision and reset weight clipping parameters
        self.model.change_precision(bits=8)
        self.model.reset_alpha_weights()
        # [NeMO] Export legacy-style INT-16 weights. Clipping parameters are changed!
        self.model.export_weights_legacy_int8()
        # [NeMO] Re-check validation accuracy
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: After export: %f" % acc)


        # [NeMO] The evaluation engine performs a simple grid search to decide, among the possible quantization configurations,
        # which one is the most promising step for the relaxation procedure. It uses an internal heuristic binning validation
        # results in top-bin (high accuracy), middle-bin (reduced accuracy, but not garbage) and bottom-bin (garbage results).
        # It typically selects a step from the middle-bin to maximize training speed without sacrificing the final results.
        # evale = nemo.evaluation.EvaluationEngine(self.model, precision_rule=precision_rule,
        #                                          validate_fn=self.ValidateSingleEpoch,
        #                                          validate_data=validation_loader)
        # # while evale.step():
        #     valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
        #         validation_loader)
        #     acc = torch.tensor(float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi))
        #     evale.report(acc)
        #     logging.info("[ModelTrainer] %.1f-bit W, %.1f-bit x: %.2f" % (
        #         evale.wgrid[evale.idx], evale.xgrid[evale.idx], acc))
        #Wbits, xbits = evale.get_next_config(upper_threshold=0.97)
        # Wbits = 16
        # xbits = 16
        # precision_rule['0']['W_bits'] = min(Wbits, precision_rule['0']['W_bits'])
        # precision_rule['0']['x_bits'] = min(xbits, precision_rule['0']['x_bits'])
        # logging.info("[ModelTrainer] Choosing %.1f-bit W, %.1f-bit x for first step" % (
        #     precision_rule['0']['W_bits'], precision_rule['0']['x_bits']))
        #
        # # [NeMO] The relaxation engine can be stepped to automatically change the DNN precisions and end training if the final
        # # target has been achieved.
        # self.relax = nemo.relaxation.RelaxationEngine(self.model, optimizer, criterion=None, trainloader=None,
        #                                          precision_rule=precision_rule, reset_alpha_weights=False,
        #                                          min_prec_dict=None, evaluator=evale)



    def TrainSingleEpoch(self, training_generator):

        self.model.train()
        train_loss_x = MovingAverage()
        train_loss_y = MovingAverage()
        train_loss_z = MovingAverage()
        train_loss_phi = MovingAverage()
        train_loss_class = MovingAverage()

        i = 0
        for batch_samples, batch_targets in training_generator:

            batch_targets = batch_targets.to(self.device)
            batch_samples = batch_samples.to(self.device)
            outputs = self.model(batch_samples)
            if self.model.isCombined:
                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss_class = self.criterion_class(outputs[4], (batch_targets[:, 4]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi + self.classifierLossFactor*loss_class
            elif self.model.isClassifier:
                loss = self.criterion_class(outputs.reshape(-1), batch_targets.float().reshape(-1))
            else:
                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.model.isCombined:
                train_loss_x.update(loss_x)
                train_loss_y.update(loss_y)
                train_loss_z.update(loss_z)
                train_loss_phi.update(loss_phi)
                train_loss_class.update(loss_class)
                #self.train_losses_log.append(loss_x)

                if (i + 1) % 100 == 0:
                    logging.info("[ModelTrainer] Step [{}]: Average train loss {}, {}, {}, {}, {}".format(i+1, train_loss_x.value, train_loss_y.value, train_loss_z.value,
                                                               train_loss_phi.value, train_loss_class.value))
                i += 1
            
            elif self.model.isClassifier:
                if (i + 1) % 100 == 0:
                    logging.info("[ModelTrainer] Step [{}]: Average train loss {}".format(i+1, loss))

                i += 1

            else:
                train_loss_x.update(loss_x)
                train_loss_y.update(loss_y)
                train_loss_z.update(loss_z)
                train_loss_phi.update(loss_phi)
                #self.train_losses_log.append(loss_x)

                if (i + 1) % 100 == 0:
                    logging.info("[ModelTrainer] Step [{}]: Average train loss {}, {}, {}, {}".format(i+1, train_loss_x.value, train_loss_y.value, train_loss_z.value,
                                                               train_loss_phi.value))
                i += 1

        if self.model.isCombined:
            return train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value, train_loss_class.value
        elif self.model.isClassifier:
            return loss
        else:
            return train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value


    def printConfusionMatrix(self):
        print("TP:" + str(self.TP))
        print("FP:" + str(self.FP))
        print("TN:" + str(self.TN))
        print("FN:" + str(self.FN))
        
    def updateConfusionMatrix(self, predictions, gt):
        count = 0
        for pred in predictions:
            if pred > 0.5:
                if gt[count] > 0.5:
                    self.TP += 1
                else: 
                    self.FP += 1
            else:
                if gt[count] > 0.5:
                    self.FN += 1
                else:
                    self.TN += 1
            count += 1
    def initConfusionMatrix(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def ValidateSingleEpoch(self, validation_generator):

        self.model.eval()
        valid_loss = RunningAverage()
        valid_loss_x = RunningAverage()
        valid_loss_y = RunningAverage()
        valid_loss_z = RunningAverage()
        valid_loss_phi = RunningAverage()
        valid_loss_class = RunningAverage()

        y_pred = []
        gt_labels = []
        self.initConfusionMatrix()
        with torch.no_grad():
            for batch_samples, batch_targets in validation_generator:
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)

                if self.model.isCombined:
                    loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                    loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                    loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                    loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                    loss_class = self.criterion_class(outputs[4], (batch_targets[:, 4]).view(-1, 1))
                    loss = loss_x + loss_y + loss_z + loss_phi + self.classifierLossFactor*loss_class
                    self.updateConfusionMatrix(outputs[4], (batch_targets[:, 4]).view(-1, 1))
                    valid_loss.update(loss)
                    valid_loss_x.update(loss_x)
                    valid_loss_y.update(loss_y)
                    valid_loss_z.update(loss_z)
                    valid_loss_phi.update(loss_phi)
                    valid_loss_class.update(loss_class)
                    #self.val_losses_log.append(loss_x)

                    outputs = torch.stack(outputs, 0)
                    outputs = torch.squeeze(outputs)
                    outputs = torch.t(outputs)
                elif self.model.isClassifier:
                    loss = self.criterion_class(outputs.reshape(-1), batch_targets.float().reshape(-1))
                    self.updateConfusionMatrix(outputs.reshape(-1), batch_targets.float().reshape(-1))
                else:
                    loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                    loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                    loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                    loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                    loss = loss_x + loss_y + loss_z + loss_phi

                    valid_loss.update(loss)
                    valid_loss_x.update(loss_x)
                    valid_loss_y.update(loss_y)
                    valid_loss_z.update(loss_z)
                    valid_loss_phi.update(loss_phi)
                    #self.val_losses_log.append(loss_x)

                    outputs = torch.stack(outputs, 0)
                    outputs = torch.squeeze(outputs)
                    outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

            if self.model.isCombined:
                self.printConfusionMatrix()
                logging.info("[ModelTrainer] Average validation loss {}, {}, {}, {}, {}".format(valid_loss_x.value, valid_loss_y.value,
                                                                      valid_loss_z.value,
                                                                      valid_loss_phi.value,
                                                                      valid_loss_class.value))
                return valid_loss_x.value, valid_loss_y.value, valid_loss_z.value, valid_loss_phi.value, valid_loss_class.value, y_pred, gt_labels
            elif self.model.isClassifier:
                self.printConfusionMatrix()
                logging.info("[ModelTrainer] Average validation loss {}".format(loss))
                return loss, y_pred, gt_labels
            else:
                logging.info("[ModelTrainer] Average validation loss {}, {}, {}, {}".format(valid_loss_x.value, valid_loss_y.value,
                                                                      valid_loss_z.value,
                                                                      valid_loss_phi.value))
                return valid_loss_x.value, valid_loss_y.value, valid_loss_z.value, valid_loss_phi.value, y_pred, gt_labels
                

    def Train(self, training_generator, validation_generator, tb=None):

        metrics = Metrics()
        early_stopping = EarlyStopping(patience=10, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
                                                                    patience=5, verbose=False,
                                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                    min_lr=0.1e-6, eps=1e-08)
        loss_epoch_m1 = 1e3

        for epoch in range(1, self.args.epochs + 1):
            logging.info("[ModelTrainer] Starting Epoch {}".format(epoch))

            change_prec = False
            ended = False
            # if self.args.quantize:
            #     change_prec, ended = self.relax.step(loss_epoch_m1, epoch, None)
            # if ended:
            #     break

            if self.model.isCombined:

                train_loss_x, train_loss_y, train_loss_z, train_loss_phi, train_loss_class = self.TrainSingleEpoch(training_generator)

                valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_loss_class, y_pred, gt_labels = self.ValidateSingleEpoch(
                    validation_generator)

                valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi + self.classifierLossFactor*valid_loss_class
                scheduler.step(valid_loss)

                gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
                y_pred = torch.tensor(y_pred, dtype=torch.float32)
                MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
                                                   [train_loss_x, train_loss_y, train_loss_z, train_loss_phi, train_loss_class],
                                                   [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_loss_class])
                train_total_loss = train_loss_x + train_loss_y + train_loss_z + train_loss_phi + self.classifierLossFactor*train_loss_class
                valid_total_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi + self.classifierLossFactor*valid_loss_class
                if tb != None:
                    AddLosssesTB(self, tb, epoch, train_loss_x, train_loss_y, train_loss_z, train_total_loss, valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_total_loss)
                    try:
                        tb.add_scalar('Lossclass/train', train_loss_class, epoch)
                        tb.add_scalar('Lossclass/valid', valid_loss_class, epoch)
                    except Exception as e:
                        print(e)
                logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
                logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
                logging.info('[ModelTrainer] Validation r_score: {}'.format(r_score))

            elif self.model.isClassifier:
                train_total_loss = self.TrainSingleEpoch(training_generator)
                valid_total_loss, y_pred, gt_labels = self.ValidateSingleEpoch(validation_generator)
                if tb != None:
                    try:
                        tb.add_scalar('TotalLoss/train', train_total_loss, epoch)
                        tb.add_scalar('TotalLoss/valid', valid_total_loss, epoch)
                    except Exception as e:
                        print(e)
            else:
                train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(training_generator)

                valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                    validation_generator)

                valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
                scheduler.step(valid_loss)

                gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
                y_pred = torch.tensor(y_pred, dtype=torch.float32)
                MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
                                                   [train_loss_x, train_loss_y, train_loss_z, train_loss_phi],
                                                   [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])
                train_total_loss = train_loss_x + train_loss_y + train_loss_z + train_loss_phi
                valid_total_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
                if tb != None:
                    AddLosssesTB(self, tb, epoch, train_loss_x, train_loss_y, train_loss_z, train_total_loss, valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_total_loss)

                logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
                logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
                logging.info('[ModelTrainer] Validation r_score: {}'.format(r_score))                


            checkpoint_filename = self.folderPath + self.model.name + '-{:03d}.pt'.format(epoch)
            early_stopping(valid_total_loss, self.model, epoch, checkpoint_filename)
            if early_stopping.early_stop:
                logging.info("[ModelTrainer] Early stopping")
                break
        if tb != None:
            tb.close()
        if self.model.isClassifier == False:        
            MSEs = metrics.GetMSE()
            MAEs = metrics.GetMAE()
            r_score = metrics.Getr2_score()
            y_pred_viz = metrics.GetPred()
            gt_labels_viz = metrics.GetLabels()
            train_losses_x, train_losses_y, train_losses_z, train_losses_phi, valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi = metrics.GetLosses()

            DataVisualization.desc = "Train_"
            DataVisualization.PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi)
            DataVisualization.PlotMSE(MSEs)
            DataVisualization.PlotMAE(MAEs)
            DataVisualization.PlotR2Score(r_score)

            DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
            DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
            #DataVisualization.PlotLossTest(self.train_losses_log, self.val_losses_log) #to display single and not averaged training/validation losses, uncomment there as well. Think about enough RAM
            DataVisualization.DisplayPlots()

    def PerdictSingleSample(self, test_generator):

        iterator = iter(test_generator)
        batch_samples, batch_targets = iterator.next()
        index = np.random.choice(np.arange(0, batch_samples.shape[0]), 1)
        x_test = batch_samples[index]
        y_test = batch_targets[index]
        self.model.eval()

        logging.info('[ModelTrainer] GT Values: {}'.format(y_test.cpu().numpy()))
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.model(x_test)

        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs)
        outputs = torch.t(outputs)
        outputs = outputs.cpu().numpy()
        logging.info('[ModelTrainer] Prediction Values: {}'.format(outputs))
        return x_test[0].cpu().numpy(), y_test[0], outputs


    def InferSingleSample(self, frame):

        shape = frame.shape
        if len(frame.shape) == 3:
            frame = np.reshape(frame, (1, shape[0], shape[1], shape[2]))

        frame = np.swapaxes(frame, 1, 3)
        frame = np.swapaxes(frame, 2, 3)
        frame = frame.astype(np.float32)
        frame = torch.from_numpy(frame)
        self.model.eval()

        with torch.no_grad():
            frame = frame.to(self.device)
            outputs = self.model(frame)

        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs)
        outputs = torch.t(outputs)
        outputs = outputs.cpu().numpy()
        return outputs

    def Predict(self, test_generator):

        metrics = Metrics()
        if self.model.isCombined:
            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_loss_class, y_pred, gt_labels = self.ValidateSingleEpoch(
                test_generator)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
                                               [0, 0, 0, 0, 0],
                                               [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_loss_class])

            y_pred_viz = metrics.GetPred()
            gt_labels_viz = metrics.GetLabels()

            DataVisualization.desc = "Test_"
            DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
            DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
            DataVisualization.DisplayPlots()
            logging.info('[ModelTrainer] Test MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Test MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Test r_score: {}'.format(r_score))
        elif self.model.isClassifier:
            valid_loss, y_pred, gt_labels  = self.ValidateSingleEpoch(test_generator)
            logging.info('[ModelTrainer] Test loss: {}'.format(valid_loss))
        else:
            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                test_generator)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
                                               [0, 0, 0, 0],
                                               [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

            y_pred_viz = metrics.GetPred()
            gt_labels_viz = metrics.GetLabels()

            DataVisualization.desc = "Test_"
            DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
            DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
            DataVisualization.DisplayPlots()
            logging.info('[ModelTrainer] Test MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Test MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Test r_score: {}'.format(r_score))

    def Infer(self, live_generator):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            live_generator)

        return y_pred

    def AddLosssesTB(self, tb, epoch, train_loss_x, train_loss_y, train_loss_z, train_total_loss, valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, valid_total_loss):
        try:
            tb.add_scalar('Lossx/train', train_loss_x, epoch)
            tb.add_scalar('Lossy/train', train_loss_y, epoch)
            tb.add_scalar('Lossz/train', train_loss_z, epoch)
            tb.add_scalar('Lossphi/train', train_loss_phi, epoch)
            tb.add_scalar('TotalLoss/train', train_total_loss, epoch)
            tb.add_scalar('Lossx/valid', valid_loss_x, epoch)
            tb.add_scalar('Lossy/valid', valid_loss_y, epoch)
            tb.add_scalar('Lossz/valid', valid_loss_z, epoch)
            tb.add_scalar('Lossphi/valid', valid_loss_phi, epoch)
            tb.add_scalar('TotalLoss/valid', valid_total_loss, epoch)
        except Exception as e:
            print(e)
        '''
        tb.add_scalar('ReLuStats/ReLu1/max',self.model.relu1.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLu1/mean',self.model.relu1.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLu1/var',self.model.relu1.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLu2/max',self.model.relu2.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLu2/mean',self.model.relu2.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLu2/var',self.model.relu2.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLu3/max',self.model.relu3.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLu3/mean',self.model.relu3.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLu3/var',self.model.relu3.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLu4/max',self.model.relu4.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLu4/mean',self.model.relu4.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLu4/var',self.model.relu4.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLu5/max',self.model.relu5.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLu5/mean',self.model.relu5.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLu5/var',self.model.relu5.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer1/max',self.model.layer1.relu2.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer1/mean',self.model.layer1.relu2.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer1/var',self.model.layer1.relu2.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer2/max',self.model.layer2.relu2.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer2/mean',self.model.layer2.relu2.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer2/var',self.model.layer2.relu2.get_statistics()[2] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer3/max',self.model.layer3.relu2.get_statistics()[0] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer3/mean',self.model.layer3.relu2.get_statistics()[1] , epoch)
        tb.add_scalar('ReLuStats/ReLuLayer3/var',self.model.layer3.relu2.get_statistics()[2] , epoch)
        '''
        #tb.add_scalar('Number Correct', total_correct, epoch)
        #tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
        
        #tb.add_histogram('conv.bias', self.model.conv.bias, epoch)
        try:
            tb.add_histogram('conv.weight', self.model.conv.weight, epoch)
            tb.add_histogram(
                'conv.weight.grad'
                ,self.model.conv.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
        try:
            tb.add_histogram('layer1/conv1.weight', self.model.layer1.conv1.weight, epoch)
            tb.add_histogram(
                'layer1/conv1.weight.grad'
                ,self.model.layer1.conv1.weight.grad
                ,epoch)
            tb.add_histogram('layer1/conv2.weight', self.model.layer1.conv2.weight, epoch)
            tb.add_histogram(
                'layer1/conv2.weight.grad'
                ,self.model.layer1.conv2.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
        try:
            tb.add_histogram('layer2/conv1.weight', self.model.layer2.conv1.weight, epoch)
            tb.add_histogram(
                'layer2/conv1.weight.grad'
                ,self.model.layer2.conv1.weight.grad
                ,epoch)
            tb.add_histogram('layer2/conv2.weight', self.model.layer2.conv2.weight, epoch)
            tb.add_histogram(
                'layer2/conv2.weight.grad'
                ,self.model.layer2.conv2.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
        try:
            tb.add_histogram('layer3/conv1.weight', self.model.layer3.conv1.weight, epoch)
            tb.add_histogram(
                'layer3/conv1.weight.grad'
                ,self.model.layer3.conv1.weight.grad
                ,epoch)
            tb.add_histogram('layer3/conv2.weight', self.model.layer3.conv2.weight, epoch)
            tb.add_histogram(
                'layer3/conv2.weight.grad'
                ,self.model.layer3.conv2.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
        try:
            tb.add_histogram('fc1.weight', self.model.fc1.weight, epoch)
            tb.add_histogram(
                'fc1.weight.grad'
                ,self.model.fc1.weight.grad
                ,epoch)
            tb.add_histogram('fc2.weight', self.model.fc2.weight, epoch)
            tb.add_histogram(
                'fc2.weight.grad'
                ,self.model.fc2.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
        try:
            tb.add_histogram('fc_x.weight', self.model.fc_x.weight, epoch)
            tb.add_histogram(
                'fc_x.weight.grad'
                ,self.model.fc_x.weight.grad
                ,epoch)
            tb.add_histogram('fc_y.weight', self.model.fc_y.weight, epoch)
            tb.add_histogram(
                'fc_y.weight.grad'
                ,self.model.fc_y.weight.grad
                ,epoch)
            tb.add_histogram('fc_z.weight', self.model.fc_z.weight, epoch)
            tb.add_histogram(
                'fc_z.weight.grad'
                ,self.model.fc_z.weight.grad
                ,epoch)
            tb.add_histogram('fc_phi.weight', self.model.fc_phi.weight, epoch)
            tb.add_histogram(
                'fc_phi.weight.grad'
                ,self.model.fc_phi.weight.grad
                ,epoch)
        except Exception as e:
            print(e)
