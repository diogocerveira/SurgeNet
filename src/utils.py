import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

from torchvision.transforms import v2 as transforms
import os
import datetime
from collections import defaultdict
import torch.nn as nn

def modeFilter(preds, windowSize):
  def mode(sequence):
    # not handling ties yet
    counter = defaultdict(int)
    for j in sequence:
      counter[j] += 1
    return max(counter.items(), key=lambda x: x[1])[0]
  # preds should be cl  asses
  padded = np.pad(preds, (windowSize // 2), mode="edge")  # repeats edge values
  smoothed = []

  for i in range(len(preds)):
    window = padded[i:i + windowSize]
    smoothed.append(mode(window)) 
  return np.array(smoothed)

def choose_in_path(path):
  searching = True
  to = path
  
  while searching:
    
    ldir = os.listdir(to)
    print('\n', ldir)
    choice = int(input("\nWhere to go? "))
    # print(ldir[choice])
    if choice == -1:
      to = os.path.dirname(to)
      continue
    else:
      try:
        to = os.path.join(to, ldir[choice])
        if os.path.isfile(to):
          return to
      except:
        stop = input("Bad path! - do you wanna stop? (y/n) ")
        if stop == 'y':
          return path
        continue
 
class Pathtaker():
  ''' Handles paths to the local everywhere
  '''
  def __init__(self, path_root, datasetId, date):
    self.path_root = path_root
    for attrKey, attrValue in self._setup_paths(datasetId, date):
      setattr(self, attrKey, attrValue)

  def _setup_paths(self, datasetId, date):
    ''' Make sure paths/files exist before setting them as attrs (in constructor)
        And returns dict items
    '''
    paths_data = self._get_paths_data(datasetId)
    paths_logs = self._get_paths_logs(datasetId, date)
    paths_all = {**paths_data, **paths_logs} # merges dicts
    os.makedirs(paths_all["path_dataset"], exist_ok=True)
    os.makedirs(paths_all["path_run"], exist_ok=True)
    for k, v in paths_all.items():
      # print(v, '\n')
      if os.path.splitext(v)[1] in (".csv", ".yml"):  # If it's a file
        # print("ITS A FILE")
        os.makedirs(os.path.dirname(v), exist_ok=True)  # Create only the parent directory
        if not os.path.exists(v):  # Only create the file if it doesn't exist
          open(v, 'w', newline='').close()
      else:  # Otherwise, it's a directory path
        os.makedirs(v, exist_ok=True)

    return paths_all.items()
    # with open(path_config, "w") as config:
    #   config.write("learning_rate: \nbatch_size: \nepochs: \n")

  def _get_paths_data(self, datasetId):
    ''' Get dictionary with paths for everything data'''
    paths_data = {}
    paths_data["path_dataset"] = os.path.join(self.path_root, "data", datasetId)
    paths_data["path_samples"] = os.path.join(paths_data["path_dataset"], "samples")
    paths_data["path_labels"] = os.path.join(paths_data["path_dataset"], "labels.csv")
    paths_data["path_annots"] = os.path.join(paths_data["path_dataset"], "annotations")
    return paths_data


  def _get_paths_logs(self, datasetId, date):
    ''' Get dictionary with paths to everything related to a run'''
    paths_logs = {}
    paths_logs["path_logs"] = os.path.join(self.path_root, "logs")
    paths_logs["path_run"] = os.path.join(paths_logs["path_logs"], f"{datasetId.split('-')[0]}-{date}")
    paths_logs["path_runInfo"] = os.path.join(paths_logs["path_run"], "run-info.yml")
    paths_logs["path_eval"] = os.path.join(paths_logs["path_run"], "eval")
    paths_logs["path_aprfc"] = os.path.join(paths_logs["path_eval"], "aprfc")
    paths_logs["path_phaseCharts"] = os.path.join(paths_logs["path_eval"], "phase-charts")
    paths_logs["path_phaseTiming"] = os.path.join(paths_logs["path_eval"], "phase-timing")
    paths_logs["path_train"] = os.path.join(paths_logs["path_run"], "train")
    paths_logs["path_predictions"] = os.path.join(paths_logs["path_train"], "predictions")
    paths_logs["path_predictions"] = os.path.join(paths_logs["path_train"], "predictions")
    paths_logs["path_events"] = os.path.join(paths_logs["path_train"], "events")
    paths_logs["path_checkpoints"] = os.path.join(paths_logs["path_train"], "checkpoints")
    paths_logs["path_process"] = os.path.join(paths_logs["path_run"], "process")
    paths_logs["path_modeFilter"] = os.path.join(paths_logs["path_process"], "mode-filter")
    paths_logs["path_features"] = os.path.join(paths_logs["path_process"], "features")
    return paths_logs

def get_metrics(classWeights, device, computePeriod, n_classes, idxToClass, agg):
  criterion = nn.CrossEntropyLoss(weight=classWeights)
  trainAcc = RunningMetric(MulticlassAccuracy(device=device), computePeriod)
  validAcc = RunningMetric(MulticlassAccuracy(device=device), computePeriod // 2)
  testBM = MetricBunch({"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1,
                         "confusionMatrix": 1}, n_classes, idxToClass, agg, device)
  return criterion, trainAcc, validAcc, testBM
