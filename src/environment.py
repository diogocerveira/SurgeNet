import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import os
from collections import defaultdict
import torch.nn as nn
import datetime
import yaml
import shutil
from src import teaching
import torch
import pickle

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
  def __init__(self, path_root, TRAIN, DATA, MODEL, id_classroom=None, path_data=None):
    self.path_root = path_root
    self.path_data = path_data if path_data is not None else path_root
    self.path_logs = os.path.join(self.path_root, "logs")
    os.makedirs(self.path_logs, exist_ok=True)

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
        os.makedirs(os.path.join(self.path_root, "logs", id_classroom), exist_ok=True)

    # create classroom path before checking for student profile (necessary)
    self.path_classroom = os.path.join(self.path_root, "logs", id_classroom)
    self.id = id_classroom
    self.status = classroomStatus
  
    id_student = self.match_studentProfile(MODEL)
    if id_student:  # found match
      studentStatus = "Matched"
    else:
      studentStatus = "New"
      num = 1
      while True:
        id_student = self.get_modelId(MODEL)
        if id_student not in os.listdir(self.path_classroom):
          break
        num += 1
        print("diff model with same name")
      os.makedirs(os.path.join(self.path_classroom, id_student), exist_ok=True)

    self.path_student = os.path.join(self.path_classroom, id_student)
    
    self.studentId = id_student
    self.studentStatus = studentStatus

    for attrKey, attrValue in self._setup_paths(DATA["id_dataset"], id_student, TRAIN, DATA, MODEL):
      setattr(self, attrKey, attrValue) # set all paths as attributes

    # Update settings dict if a classroom is chosen
    if self.status == "Chosen":
      self.TRAIN = TRAIN
      self.DATA = DATA
    else:
      with open(self.path_learningConditions, 'r') as file:
        loaded_data = yaml.safe_load(file)
        self.TRAIN = loaded_data["TRAIN"]
        self.DATA = loaded_data["DATA"]
    
    self.DATA["labelType"] = tuple(DATA["labelType"].split('-'))  # e.g. ('single', 'head1', 'phase,etc')
    
    self.path_spaceFeats = MODEL["path_spaceFeats"]

  def get_modelId(self, MODEL):
    # learnMode = {"space": "SP", "time": "TP", "spatio-temporal": "ST"}
    spaceTransferMode = {None: None, "feat-xtract": "FX", "fine-tune": "FT", "l4-fine-tune": "pFT"}
    spaceArch = {"resnet50": "RN50"}
    timeTransferMode = {None: None, "feat-xtract": "FX", "fine-tune": "FT"}
    timeArch = {"tecno": "TECNO", "multiTecno": "mTECNO", "phatima": "PHA", "dTsNet": "DTS", "pTsNet": "PTS"}
    classifierArch = {None: None, "one-linear": "1L", "two-linear": "2L"}
    domain = MODEL["domain"].split('-')  # e.g. ['space', 'time'] or ['space', time']
    modelId = []
    
    if "space" in domain:
      modelId.append(spaceTransferMode[MODEL["spaceTransferMode"]])
      modelId.append(spaceArch[MODEL["spaceArch"]])
    if "time" in domain:
      modelId.append(timeTransferMode[MODEL["timeTransferMode"]])
      modelId.append(timeArch[MODEL["timeArch"]])
     

    modelId.append(classifierArch[MODEL["classifierArch"]])
    # print(modelId)
    # drop modelId None values
    modelId = [m for m in modelId if m is not None]
    return '-'.join(modelId)
  
  def match_learningEnvironment(self, TRAIN, DATA):
    classrooms = [c for c in os.listdir(self.path_logs)
                  if not c.startswith(('.', '_'))
                  and os.path.isdir(os.path.join(self.path_logs, c))
                  and "learning-conditions.yml" in os.listdir(os.path.join(self.path_logs, c))]
    for id_classroom in classrooms:
      with open(os.path.join(self.path_logs, id_classroom, "learning-conditions.yml"), 'r') as f:
        learningConditions = yaml.safe_load(f)
        # if new learning parameters match a previous classroom with same "subject" (id_dataset)
        if learningConditions["TRAIN"] == TRAIN and learningConditions["DATA"] == DATA:
          return id_classroom
    return None
  
  def match_studentProfile(self, MODEL):
    students = [m for m in os.listdir(self.path_classroom) if os.path.isdir(os.path.join(self.path_classroom, m))] # list of students in the classroom

    for id_student in students:
      # print(f"Checking student {id_student} in classroom {self.id}")
      if not os.path.exists(os.path.join(self.path_classroom, id_student, "student-profile.yml")):
        print(f"Student profile for {id_student} does not exist in classroom {self.id}")
        continue  # skip if student profile does not exist
      with open(os.path.join(self.path_classroom, id_student, "student-profile.yml"), 'r') as f:
        studentProfile = yaml.safe_load(f)
        # if new learning parameters match a previous classroom with same "subject" (id_dataset)
        try:
          if studentProfile["MODEL"] == MODEL:
            return id_student
        except Exception as e:
          print(f"Error reading student profile for {id_student}: {e}")
          pass
    return None

  def _setup_paths(self, id_dataset, id_student, TRAIN, DATA, MODEL):
    ''' Make sure paths/files exist before setting them as attrs (in constructor)
        And returns dict items
    '''
    paths_data = self._get_paths_data(id_dataset, DATA["labelType"])  # first word is the datasetID
    paths_logs = self._get_paths_logs(id_student)
    paths_all = {**paths_data, **paths_logs} # merges dicts
    for _, path in paths_all.items():
      # print(path, '\n')
      if os.path.splitext(path)[1] in (".csv", ".yml"):  # If it's a file
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create only the parent directory
        if not os.path.exists(path):  # Only create the file if it doesn't exist
          open(path, 'w', newline='').close()
        if os.path.basename(path) == "learning-conditions.yml" and self.status == "New": # if file is learning-conditions.yml, write the learning conditions
          with open(path, "w") as f:
            yaml.dump({"TRAIN": TRAIN, "DATA": DATA}, f)
        if os.path.basename(path) == "student-profile.yml" and self.studentStatus == "New": # if file is student-profile.yml, write the model characteristics
          with open(path, "w") as f:
            yaml.dump({"MODEL": MODEL}, f)
            yaml.dump({"VIDEO_SPLITS": {}}, f)  # Initialize with empty splits
      
      else:  # Otherwise, it's a directory path
        os.makedirs(path, exist_ok=True)

    return paths_all.items()
    # with open(path_config, "w") as config:
    #   config.write("learning_rate: \nbatch_size: \nepochs: \n")

  def _get_paths_data(self, id_dataset, labelType):
    ''' Get dictionary with paths for everything data'''
    paths_data = {}
    paths_data["path_dataset"] = os.path.join(self.path_data, "data", id_dataset)
    paths_data["path_samples"] = os.path.join(paths_data["path_dataset"], "samples")
    paths_data["path_labels"] = os.path.join(paths_data["path_dataset"], "labels", f"{labelType}.csv")
    paths_data["path_annots"] = os.path.join(paths_data["path_dataset"], "annotations")
    return paths_data
  def _get_paths_logs(self, id_student):
    ''' Get dictionary with paths to everything related to a run'''
    paths_logs = {}
    paths_logs["path_logs"] = self.path_logs
    paths_logs["path_learningConditions"] = os.path.join(self.path_classroom, "learning-conditions.yml")
    
    paths_logs["path_studentProfile"] = os.path.join(self.path_student, "student-profile.yml")
    paths_logs["path_eval"] = os.path.join(self.path_student, "eval")
    paths_logs["path_aprfc"] = os.path.join(paths_logs["path_eval"], "aprfc")
    paths_logs["path_ribbons"] = os.path.join(paths_logs["path_eval"], "ribbons")
    paths_logs["path_timings"] = os.path.join(paths_logs["path_eval"], "timings")
    paths_logs["path_train"] = os.path.join(self.path_student, "train")
    paths_logs["path_preds"] = os.path.join(paths_logs["path_eval"], "preds")
    paths_logs["path_events"] = os.path.join(paths_logs["path_train"], "events")
    paths_logs["path_state"] = os.path.join(paths_logs["path_train"], "state")
    paths_logs["path_process"] = os.path.join(self.path_student, "process")
    paths_logs["path_modedPreds"] = os.path.join(paths_logs["path_process"], "moded-preds")
    paths_logs["path_feats"] = os.path.join(paths_logs["path_process"], "feats")
    return paths_logs

  def update_fileIds(self):
      """Update filenames inside directories to start with parent-dir + student-classroom IDs"""
      
      for directory in [
        self.path_ribbons,
        self.path_timings,
        self.path_preds,
        self.path_state,
        self.path_modedPreds,
        self.path_feats
      ]:
          if not os.path.exists(directory):
              continue

          parent_name = os.path.basename(directory)
          for filename in os.listdir(directory):
              old_path = os.path.join(directory, filename)
              if not os.path.isfile(old_path):
                  continue

              parts = filename.split('_')
              # print(parts)
              # Prepend parent dir name and student-classroom IDs
              parts = [parts[0]] + parts[2:]  # Remove the first part (which is the old ID)
              new_filename = "_".join(parts)
              # parts[0] = f"{parent_name}_{self.studentId}-{self.id}"
              # new_filename = '_'.join(parts[1:])
              new_path = os.path.join(directory, new_filename)
              os.rename(old_path, new_path)


  def rename_directories(path_logs, renameMap=None):
    ''' Rename directories to match the new naming convention
        This is a one-time operation to update old directory names
    '''
    # Mapping of old names to new names
    renameMap = {
      "aprfc": "aprfc",
      "phase-charts": "ribbons",
      "phase-timing": "timings",
      "predictions": "preds",
      "mode-filter": "moded-preds",
      "features": "feats",
      "events": "events",
      "models": "state",
    }

    workspace_path = "logs"  # Replace with your path if needed

    for root, dirs, files in os.walk(path_logs, topdown=False):
      for directory in dirs:
        if directory in renameMap:
          old_path = os.path.join(root, directory)
          new_path = os.path.join(root, renameMap[directory])
          os.rename(old_path, new_path)


def build_globalCM(path_preds, path_to, n_classes, labelType, headType, labelToClassMap, DEVICE="cpu"):
  overallBundle = {"Pred": [], "Target": []}

  # collect predictions and targets
  for file in os.listdir(path_preds):
    if file.endswith(".pt"):
      with open(os.path.join(path_preds, file), 'rb') as f:
        test_bundle = pickle.load(f)
        overallBundle["Pred"].extend(test_bundle["Pred"])
        overallBundle["Target"].extend(test_bundle["Target"])

  # debug: check if we got anything
  print(f"Collected {len(overallBundle['Pred'])} predictions and {len(overallBundle['Target'])} targets")

  if len(overallBundle["Pred"]) == 0 or len(overallBundle["Target"]) == 0:
    print("WARNING: No predictions or targets found. Confusion matrix cannot be updated.")
    return None

  # map string labels to integers
  pred_indices = torch.tensor(expand_labels(overallBundle["Pred"], labelToClassMap), device=DEVICE)
  target_indices = torch.tensor(expand_labels(overallBundle["Target"], labelToClassMap), device=DEVICE)

  # debug: check tensor shapes
  print(f"Pred tensor shape: {pred_indices.shape}, Target tensor shape: {target_indices.shape}")

  cm = teaching.StillMetric("confusionMatrix", n_classes, DEVICE, labelType, headType, labelToClassMap)
  cm.metric.update(pred_indices, target_indices)

  path_to = os.path.join(path_to, "cm_global")
  _ = cm.score(path_to)
 
def expand_labels(entries, labelToClassMap):
  expanded = []
  for e in entries:
    if "," in e:  # multi-label case
      parts = [p.strip() for p in e.split(",") if p.strip()]
    else:         # single-label case (may contain "-" but it's still one class)
      parts = [e.strip()]
    expanded.extend([labelToClassMap[p] for p in parts])
  return expanded