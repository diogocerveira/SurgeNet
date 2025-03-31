import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import os
from collections import defaultdict
import torch.nn as nn
import datetime
import yaml

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
 


class Classroom():
  ''' Handles paths to the learning environment "local everywhere" (originally the pathtaker)
  '''
  def __init__(self, path_root, TRAIN, DATA, id_classroom=None):
    self.path_root = path_root

    classroomStatus = ''
    if id_classroom: # user choice
      if not os.path.exists(os.path.join(self.path_root, "logs", id_classroom)):
        raise ValueError(f"Chosen classroom {id_classroom} does not exist!")
      classroomStatus = "Chosen"
    else: # automatic choice (create new classroom if unique learning conditions)
      id_classroom = self.match_learningEnvironment(TRAIN, DATA)
      if id_classroom:  # found match
        classroomStatus = "Matched"
      else:
        classroomStatus = "New"
        id_classroom = f"{DATA['id_dataset'].split('-')[0]}-{datetime.datetime.now().strftime('%y%m')}-1" # create new classroom  
        while id_classroom in os.listdir(os.path.join(self.path_root, "logs")): # number new classrooms of the month by creation order
          id_classroom = id_classroom[:-1] + str(int(id_classroom[-1]) + 1)  # increment last digit
    
    for attrKey, attrValue in self._setup_paths(DATA["id_dataset"], id_classroom, TRAIN, DATA):
      setattr(self, attrKey, attrValue) # set all paths as attributes
    self.id = id_classroom
    self.status = classroomStatus
    
  def match_learningEnvironment(self, TRAIN, DATA):
    path_logs = os.path.join(self.path_root, "logs")
    classrooms = [c for c in os.listdir(path_logs)]
    for id_classroom in classrooms:
      with open(os.path.join(path_logs, id_classroom, "learning-conditions.yml"), 'r') as f:
        learningConditions = yaml.safe_load(f)
        # if new learning parameters match a previous classroom with same "subject" (id_dataset)
        if learningConditions["TRAIN"] == TRAIN and learningConditions["DATA"] == DATA:
          return id_classroom
    return None

  def _setup_paths(self, id_dataset, id_classroom, TRAIN, DATA):
    ''' Make sure paths/files exist before setting them as attrs (in constructor)
        And returns dict items
    '''
    paths_data = self._get_paths_data(id_dataset)  # first word is the datasetID
    paths_logs = self._get_paths_logs(id_classroom)
    paths_all = {**paths_data, **paths_logs} # merges dicts
    os.makedirs(paths_all["path_dataset"], exist_ok=True)
    os.makedirs(paths_all["path_classroom"], exist_ok=True)
    for _, path in paths_all.items():
      # print(path, '\n')
      if os.path.splitext(path)[1] in (".csv", ".yml"):  # If it's a file
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create only the parent directory
        if not os.path.exists(path):  # Only create the file if it doesn't exist
          open(path, 'w', newline='').close()
        # if file is learning-conditions.yml, write the learning conditions
        if os.path.basename(path) == "learning-conditions.yml":
          with open(path, "w") as f:
            yaml.dump({"TRAIN": TRAIN, "DATA": DATA}, f)
      else:  # Otherwise, it's a directory path
        os.makedirs(path, exist_ok=True)

    return paths_all.items()
    # with open(path_config, "w") as config:
    #   config.write("learning_rate: \nbatch_size: \nepochs: \n")

  def _get_paths_data(self, id_dataset):
    ''' Get dictionary with paths for everything data'''
    paths_data = {}
    paths_data["path_dataset"] = os.path.join(self.path_root, "data", id_dataset)
    paths_data["path_samples"] = os.path.join(paths_data["path_dataset"], "samples")
    paths_data["path_labels"] = os.path.join(paths_data["path_dataset"], "labels")
    paths_data["path_annots"] = os.path.join(paths_data["path_dataset"], "annotations")
    return paths_data
  def _get_paths_logs(self, id_classroom):
    ''' Get dictionary with paths to everything related to a run'''
    paths_logs = {}
    paths_logs["path_logs"] = os.path.join(self.path_root, "logs")
    paths_logs["path_classroom"] = os.path.join(paths_logs["path_logs"], id_classroom)
    paths_logs["path_classroomInfo"] = os.path.join(paths_logs["path_classroom"], "learning-conditions.yml")
    paths_logs["path_eval"] = os.path.join(paths_logs["path_classroom"], "eval")
    paths_logs["path_aprfc"] = os.path.join(paths_logs["path_eval"], "aprfc")
    paths_logs["path_phaseCharts"] = os.path.join(paths_logs["path_eval"], "phase-charts")
    paths_logs["path_phaseTiming"] = os.path.join(paths_logs["path_eval"], "phase-timing")
    paths_logs["path_train"] = os.path.join(paths_logs["path_classroom"], "train")
    paths_logs["path_predictions"] = os.path.join(paths_logs["path_train"], "predictions")
    paths_logs["path_events"] = os.path.join(paths_logs["path_train"], "events")
    paths_logs["path_checkpoints"] = os.path.join(paths_logs["path_train"], "checkpoints")
    paths_logs["path_process"] = os.path.join(paths_logs["path_classroom"], "process")
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
