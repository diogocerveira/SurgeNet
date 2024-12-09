import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import utils
import torch
import numpy as np
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import machine
import seaborn as sns
import pickle
import time
import pandas as pd

class Teacher():
  ''' Each Teacher teaches (action = train) 'k_folds' models from 1 dataset, besides saving
      multiple checkpoint models along the way (for each fold)
      It also can evaluate (action = eval) the models.
      In summary, a Teacher is a Trainer, a Validater and a Tester
      In - split and batched dataset from a Cataloguer
      Action - teach, eval
      Pipeline: (train -> validate) x n_epochs -> test
  '''
  def __init__(self, trainer, validater, tester, criterion, n_classes, classWeights, labelToClass):
    if criterion == "crossEntropy":
      self.criterion = nn.CrossEntropyLoss(weight=classWeights)
    else:
      raise ValueError("Invalid criterion chosen (crossEntropy, )")
    self.trainer = trainer
    self.validater = validater
    self.tester = tester

    self.highScore = -np.inf
    self.bestState = {}

    
  def teach(self, model, trainloader, validloader, n_epochs, fold, path_resume, path_checkpoints, writer):
    ''' Iterate through epochs of model learning with training and validation
        In: Untrained model, data, etc
        Out: Trained model (state_dict)
    '''
    earlyStopper = EarlyStopper()
    optimizer = optim.SGD(model.parameters(), lr=self.trainer.lr, momentum=self.trainer.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.stepSize, gamma=self.trainer.gamma) # updates the optimizer by the gamma factor after the step_size
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    betterState, valid_minLoss = {}, np.inf
    self.trainer.earlyStopper.reset()
    if path_resume: # Loading checkpoint before resuming training
      model, optimizer, startEpoch, cpLoss = self.load_checkpoint(model, optimizer, path_resume)
    else:
      startEpoch = 0

    for epoch in range(startEpoch, n_epochs):
      print(f"     Epoch {epoch + 1}")
      train_loss, train_score = self.trainer.train(model, trainloader, self.criterion, optimizer, epoch, self.device)
      valid_loss, valid_score = self.validater.validate(model, validloader, self.criterion, epoch, self.device)

      writer.add_scalar("Loss/train", train_loss, epoch + 1)
      writer.add_scalar("Loss/valid", valid_loss, epoch + 1)
      writer.add_scalar("Acc/train", train_score, epoch + 1)
      writer.add_scalar("Acc/valid", valid_score, epoch + 1)
      print(f"\nTrain Loss: {train_loss:4f}\tValid Loss: {valid_loss:4f}")
      scheduler.step()

      if earlyStopper.early_stop(valid_loss):
        print(f"\n* Stopped early *\n")
        break
      if valid_loss <= valid_minLoss:
        print(f"\n* New best model (valid loss): {valid_minLoss:.4f} --> {valid_loss:.4f} *\n")
        valid_minLoss = valid_loss
        betterState = model.state_dict()
      if ((epoch + 1) % 5) == 0:  # Save checkpoint every 5 epochs
        print(f"\n* Checkpoint reached ({epoch + 1}/{n_epochs}) *\n")
        path_checkpoint = os.path.join(path_checkpoints, f"checkpoint_fold{fold + 1}_epoch{epoch + 1}.pth")
        machine.save_checkpoint(model, optimizer, epoch, valid_loss, path_checkpoint)
    return betterState

  def evaluate(self, model, testloader, testkey, path_load=None, path_save="test_results.pk1", return_extra=None):
    
    if path_load:
      with open(path_load, 'rb') as f:  # Load the DataFrame
        test_bundle = pickle.load(f)
    else:
      t1 = time.time()
      preds, targets, sampleNames = self.tester.test(model, testloader, return_extra)
      t2 = time.time()
      print(f"Testing took {t2 - t1:.2f} seconds")
      self.tester.test_metrics.display()
      # print(sampleNames[0], preds[0], targets[0])
      # torch.nn.functional.one_hot(tensor, n_classes=-1)
      test_bundle = self.get_bundle(preds, targets, sampleNames)
      with open(path_save, 'wb') as f: # Save the DataFrame
        pickle.dump(test_bundle, f)

    testsAvailable = {
      "acpref1com": self.tester.acpref1com,
      "timing": self.tester.phaseTiming,
      "chart": self.tester.phaseSegmentationChart
    }
    if testkey in testsAvailable:  # if provided a testkey, do that test
      testsAvailable[testkey](test_bundle)
    else: # do them all
      for action in testsAvailable.values():
        action(test_bundle)


  def get_bundle(self, sampleNames, preds, targets):
    df = pd.DataFrame({
      "SampleName": sampleNames,
      "Preds": preds,
      'Targets': targets
    })
    # assuming frames in the structure of videoName_relativeTime
    df[['Video', 'Timestamp']] = df["SampleName"].str.split('_', expand=True) # expand creates two columns, instead of just leaving data as a split list
    df["Timestamp"] = df["Timestamp"].astype(int)
    df = df.sort_values(by=['Video', 'Timestamp']) # Sort the DataFrame by VIDEO and TIME
    #df['AbsoluteTime'] = df.groupby('Video').cumcount() # Convert Relative Time to Absolute Time
    return df

class Trainer():
  ''' Part of the teacher that knows how to train models based on data
  '''
  def __init__(self, metricName, computeFrequency, updateFrequency, learningRate, momentum, stepSize, gamma, device):
    self.device = device
    self.train_metric = utils.RunningMetric(metricName, computeFrequency, device, updateFrequency)
    self.learningRate = learningRate
    self.momentum = momentum
    self.stepSize = stepSize
    self.gamma = gamma
  def train(self, model, trainloader, criterion, optimizer, epoch):
    ''' In: model, data, criterion (loss function), optimizer
        Out: train_loss, train_score
        Note - inputs can be samples (pics) or feature maps
               Targets == labels
               outputs can be predictions or logits?
    '''
    self.train_metric.reset()  # resets states, typically called between epochs
    runningLoss = 0.0
    model.train().to(self.device); print("\tTraining...")
    for batch, data in enumerate(trainloader, 0):   # start at batch 0
      inputs, targets = data[0].to(self.device), data[1].to(self.device) # data is a list of [inputs, labels]
      print("inputs: ", inputs.shape, "\ttargets: ", targets.shape, '\n')
      optimizer.zero_grad() # reset the parameter gradients before backward step
      outputs = model(inputs).squeeze(0).permute(1, 0) # forward pass
      print("\noutputs: ", outputs.shape)
      loss = criterion(outputs, targets) # calculate loss  # -1 for getting last stage guesses
      loss.backward() # backward pass
      optimizer.step()    # a single optimization step
      runningLoss += loss.item() # accumulate loss
      self.train_metric.update(outputs, targets)
      if batch % self.train_metric.period_compute == 0:
        print(f"\t  T [E{epoch + 1} B{batch + 1}]   scr: {self.train_metric.score():.4f}")
    return runningLoss / len(trainloader), self.train_metric.score()

class Validater():
  ''' Part of the teacher that knows how to validate a run of model training
  '''
  def __init__(self, metricName, computeFrequency, updateFrequency, device):
    self.device = device
    self.valid_metric = utils.RunningMetric(metricName, computeFrequency // 2, device)
  def validate(self, model, validloader, criterion, epoch):
    ''' In: model, data, criterion (loss function)
        Out: valid_loss, valid_score
    '''
    self.valid_metric.reset() # Running metric, e.g. accuracy
    runningLoss = 0.0
    model.eval(), print("\n\tValidating...")
    with torch.no_grad(): # don't calculate gradients (for learning only)
      for batch, data in enumerate(validloader, 0):
        inputs, targets = data[0].to(self.device), data[1].to(self.device)
        outputs = model(inputs).squeeze(0).permute(1, 0)
        # print(labels[:5], outputs[:5])
        loss = criterion(outputs, targets)
        runningLoss += loss.item()
        self.valid_metric.update(outputs, targets)  # updates with data from new batch
        if batch % self.valid_metric.period_compute == 0:
          print(f"\t  V [E{epoch + 1} B{batch + 1}]   scr: {self.valid_metric.score():.4f}")
    return runningLoss / len(validloader), self.valid_metric.score()

class Tester():
  ''' Part of the teacher that knows how to test a model knowledge in multiple ways
  '''
  def __init__(self, metricSwitches, n_classes, labelToClass, agg, device):
    self.test_metrics = MetricBunch(metricSwitches, n_classes, labelToClass, agg, device)
  
  def test(self, model, testloader, return_extra):
    ''' Test the model - return Preds, labels and sampleNames
    '''
    self.test_metrics.reset()
    preds = []
    targets = []
    sampleNames = []
    model.eval().to(self.device); print("\n\tTesting...")
    print(return_extra)
    with torch.no_grad():
      for batch, data in enumerate(testloader):
        if return_extra["ids"]:
          inputs, targets, sampleNames = data[0].to(self.device), data[1].to(self.device), data[2]
          sampleNames.append(sampleNames)
        else:
          inputs, targets = data[0].to(self.device), data[1].to(self.device)
        outputs = model(inputs).squeeze(0).permute(1, 0)
        _, batchPreds = torch.max(outputs, 1)  # get labels with max prediction values
        preds.append(batchPreds)
        targets.append(targets)
        self.test_metrics.update(outputs, targets)
        if batch % self.test_metric.accuracy.period_compute == 0:
          print(f"\t  T [B{batch + 1}]   scr: {self.test_metrics.accuracy.score():.4f}")
    preds = torch.cat(preds).cpu().tolist()  # flattens the list of tensor
    targets = torch.cat(targets).cpu().tolist()
    sampleNames = [os.path.splitext(os.path.basename(imgn))[0] for batch in sampleNames for imgn in batch]
    return preds, targets, sampleNames

  def acpref1com(self, preds, targets):
    def score_log(BM, writer, path_logs, fold):
      ''' Log metrics to TensorBoard
      '''
      writer.add_scalar(f"accuracy/test", BM.get_accuracy(), fold)
      writer.add_scalar(f"precision/test", BM.get_precision(), fold)
      writer.add_scalar(f"recall/test", BM.get_recall(), fold)
      writer.add_scalar(f"f1score/test", BM.get_f1score(), fold)
      image_cf = plt.imread(BM.get_confusionMatrix(path_logs))
      tensor_cf = torch.tensor(image_cf).permute(2, 0, 1)
      writer.add_image(f"confusion_matrix/test", tensor_cf, fold)
      writer.close()
    


class EarlyStopper:
  ''' Controls the validation loss progress to keep it going down and improve times
  '''
  def __init__(self, patience=1, minDelta=0):
    self.patience = patience
    self.minDelta = minDelta  # difference in decrease, 0 for no margin
    self.valid_minLoss = np.inf
    self.counter = 0

  def reset(self):
    self.valid_minLoss = np.inf
    self.counter = 0
  def early_stop(self, valid_loss):
    if valid_loss < self.valid_minLoss:
      self.valid_minLoss = valid_loss
      self.counter = 0  # reset warning/patience counter if loss decreases betw epochs
    elif valid_loss - self.valid_minLoss >= self.minDelta:
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  

  
class StillMetric:
  ''' For now uses torcheval metrics
      Performs metric update and computation based on a frequency provided
  '''
  def __init__(self, metricName, n_classes, device, agg="micro"):
    if metricName == "accuracy":
      self.metric = MulticlassAccuracy(average=agg, num_classes=n_classes, device=device)
    elif metricName == "precision":
      self.metric =  MulticlassPrecision(average=agg, num_classes=n_classes, device=device)
    elif metricName == "recall":
      self.metric = MulticlassRecall(average=agg, num_classes=n_classes, device=device)
    elif metricName == "f1score":
      self.metric = MulticlassF1Score(average=agg, um_classes=n_classes, device=device)
    elif metricName == "confusionMatrix":
      self.metric = MulticlassConfusionMatrix(average=agg, num_classes=n_classes, device=device)
    else:
      raise ValueError("Invalid Metric Name")
  def reset(self):
    self.metric.reset()
  def score(self):
    return self.metric.compute().item()
  
class RunningMetric(StillMetric):
  ''' Extends StillMetric to updates on the run
  '''
  def __init__(self, metricName, n_classes, device, agg, computeFrequency, updateFrequency=1):
    super().__init__(metricName, n_classes, device, agg)
    self.computeFrequency = computeFrequency  # num of batches between computations
    self.updateFrequency = updateFrequency  # num of batches between updates
    
  # def running(self, outputs, labels, batch, epoch):
  #   if (batch + 1) % self.updateFrequency == 0:
  #     self.metric.update(outputs, labels)
  #   if (batch + 1) % self.computeFrequency == 0:
  #     metric_result = self.metric.compute().item()
  #     print(f"\t  [E{epoch + 1} B{batch + 1}]   scr: {metric_result:.4f}")
  #     return metric_result

  def update(self, outputs, targets):
    self.metric.update(outputs, targets)


class MetricBunch:
  ''' Gets a bunch of still metrics at cheap price
      Accuracy locked at agg==micro for now
  '''
  def __init__(self, metricSwitches, n_classes, labelToClass, agg, device):  # metricSwitches is a boolean dict switch for getting metrics
    self.metricSwitches = metricSwitches
    self.labelToClass = labelToClass
    self.agg = agg
    
    if self.metricSwitches["accuracy"]:
      self.accuracy = RunningMetric("accuracy", n_classes, device, agg="micro")
    if self.metricSwitches["precision"]:
      self.precision = RunningMetric("precision", n_classes, device, agg=agg)
    if self.metricSwitches["recall"]:
      self.recall = RunningMetric("recall", n_classes, device, agg=agg)
    if self.metricSwitches["f1score"]:
      self.f1score = RunningMetric("f1score", n_classes, device, agg=agg)
    if self.metricSwitches["confusionMatrix"]:
      self.confusionMatrix = RunningMetric("confusionMatrix", n_classes, device, agg=agg)

  def reset(self):
    if self.accuracy: self.accuracy.reset()
    if self.precision: self.precision.reset()
    if self.recall: self.recall.reset()
    if self.f1score: self.f1score.reset()
    if self.confusionMatrix: self.confusionMatrix.reset()
    
  def update(self, outputs, targets):
    if self.accuracy: self.accuracy.update(outputs, targets)
    if self.precision: self.precision.update(outputs, targets)
    if self.recall: self.recall.update(outputs, targets)
    if self.f1score: self.f1score.update(outputs, targets)
    if self.confusionMatrix: self.confusionMatrix.update(outputs, targets)

  def display(self, metricSwitches={"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1, "confusionMatrix": 1}):
    if metricSwitches["accuracy"]:
      print(f"Accuracy: {self.get_accuracy():.4f}")
    if metricSwitches["precision"]:
      print(f"Precision: {self.get_precision():.4f}")
    if metricSwitches["recall"]:
      print(f"Recall: {self.get_recall():.4f}")
    if metricSwitches["f1score"]:
      print(f"F1 Score: {self.get_f1score():.4f}")
    if metricSwitches["confusionMatrix"]:
      plt.show(self.get_confusionMatrix())

  def get_accuracy(self):
    try:
      return self.accuracy.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_precision(self):
    try:
      return self.precision.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_recall(self):
    try:
      return self.recall.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_f1score(self):
    try:
      return self.f1score.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_confusionMatrix(self, path_to=''):
    try:
      fig, ax = plt.subplots(figsize=(10, 7))
      sns.heatmap(self.confusionMatrix.compute().cpu(), annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=self.labelToClass.values(), yticklabels=self.labelToClass.values(), ax=ax)
      ax.set_xlabel('Predicted')
      ax.set_ylabel('True')
      ax.set_title('Confusion Matrix')
      if path_to:
        path_img = os.path.join(path_to, os.path.basename(path_to) + ".png")
        plt.savefig(path_img)
        return path_img
    except:
      print("! Dataset group lacks representation of all classes !")
      return path_to
    
  


