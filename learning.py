import time
t1 = time.time()
import os
import torch
import numpy as np
import datetime

import torch.nn as nn
from torchvision import datasets # for the CIFAR10 image dataset

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from torchvision.models import resnet50, ResNet50_Weights
from torcheval.metrics import MulticlassAccuracy
import argparse
import yaml
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle

from src import catalogue, machine, teaching, utils
t2 = time.time()

def learning(**the):
  ''' Note: trying out "the" instead of "kwargs"/"config" for readability purposes '''
  # Setup Parameters
  DATE = datetime.datetime.now().strftime('%y%m-%d%H')  # for organising model/logs saving
  RUN = DATE if "train" in the["GENERAL"]["actions"] else the["GENERAL"]["load_run"]  # if training create new run, else load existing
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # where to save torch tensors (e.g. during training)
  Ptk = utils.Pathtaker(the["GENERAL"]["path_root"], the["DATA"]["datasetId"], RUN)
  
  # catalogue.py (dataset, data handling)
  Ctl = catalogue.Cataloguer(Ptk.path_dataset, the["DATA"])
  if "process" in the["GENERAL"]["actions"]:
    if the["PROCESS"]["sample"]:  # video to image frames
      Ctl.sample(the["PROCESS"]["path_rawVideos"], Ptk.path_samples, samplerate=the["PROCESS"]["samplerate"], processing="LDSS", filter_annotated=the["PROCESS"]["filter_annotated"])
      return True
    if the["PROCESS"]["resize"]:  # video to image frames
      Ctl.resize_frames(Ptk.path_samples, Ptk.path_samples, dim_to=(480, 270))
      return True
    if the["PROCESS"]["label"]:  # label (csv file) image frames
      Ctl.label(Ptk.path_samples, Ptk.path_labels)
  Ctl.build_dataset(the["DATA"]["datasetId"], the["DATA"]["preprocessing"], the["DATA"]["return_extradata"])
  
  # teaching.py (metrics, train validation, test)
  Tch = teaching.Teacher(the["TRAIN"], the["EVAL"], Ctl, DEVICE)

  print(f"\n[run] {the['DATA']['datasetId']}-{RUN} | [device] {DEVICE}")
  print(f"[dataset] {the['DATA']['datasetId']} | {Ctl.N_CLASSES} classes")
  print(f"[preprocessing] {Ctl.preprocessing}\t | {'Balanced' if Ctl.BALANCED else 'Unbalanced'}"); print(Ctl.labelToClass)
  print(f"{Ctl.DATASET_SIZE} datapoints | Batch size of {the['TRAIN']['HYPER']['batchSize']} | {the['TRAIN']['k_folds']} folds | {the['TRAIN']['HYPER']['n_epochs']} epochs\n")
  highScore = 0.0

  # Index split of the dataset (virtual)
  for fold, splits in enumerate(Ctl.split(the["TRAIN"]["k_folds"])):
    # iterating folds (trio tuples of idxs [0] and videoIds [1])
    ((train_idxs, valid_idxs, test_idxs), (train_videoIds, valid_videoIds, test_videoIds)) = splits
    if fold in the["TRAIN"]["skipFolds"]: continue
    print(f"Fold {fold + 1} split:\ttrain {train_videoIds}\n\t\tvalid {valid_videoIds}\n\t\ttest {test_videoIds}")
    logInfo = f"{the['DATA']['datasetId'].split('-')[0]}-{DATE}-{fold + 1}"  # model name for logging/saving
    writer = SummaryWriter(os.path.join(Ptk.path_events, logInfo)) # object for logging stats

    # Create the model
    spaceinator = machine.Spaceinator(the["MODEL"], DEVICE, N_CLASSES=Ctl.N_CLASSES)
    timeinator = machine.Timeinator(the["MODEL"], spaceinator.FEATURE_SIZE, Ctl.N_CLASSES)
    if the["MODEL"]["domain"] == "spatial":
      model = machine.SurgeNet(Ctl.N_CLASSES, [spaceinator])
      batchMode = "images"
    elif the["MODEL"]["domain"] == "temporal":
      spaceinator.load(Ptk.path_features)  # loads into Cataloguer dataset
      batchMode = "features"
      # Get actual data from the split indexs, collate and batch it to a dataloader trio
      model = machine.SurgeNet(Ctl.N_CLASSES, [timeinator])
    elif the["MODEL"]["domain"] == "full":
      batchMode = "images"
      model = machine.SurgeNet(Ctl.N_CLASSES, [spaceinator, timeinator])
    else:
      raise ValueError("Invalid domain choice!")
    trainloader, validloader, testloader = Ctl.batch(
        the["TRAIN"]["HYPER"]["batchSize"], batchMode, train_idx=train_idxs, valid_idx=valid_idxs, test_idx=test_idxs)
    

    # Feature extraction
    if the["PROCESS"]["fx_spatial"] and "process" in the["GENERAL"]["actions"]:
      t1 = time.time()
      spaceinator.export_features(DataLoader(Ctl.dataset, batch_size=the["TRAIN"]["HYPER"]["batchSize"]), Ptk.path_features, fold, DEVICE)
      t2 = time.time()
      print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")

    # Training
    if "train" in the["GENERAL"]["actions"]:
      betterState, valid_minLoss = Tch.teach(model, trainloader, validloader, the["TRAIN"]["HYPER"]["n_epochs"], logInfo,
          Ptk.path_checkpoints, writer)
      if the["TRAIN"]["save_betterModel"]:
        machine.save_model(Ptk.path_models, betterState, logInfo, DATE, 0.0, title="fold", fold=fold + 1)
      
    if "eval" in the["GENERAL"]["actions"]:
      if the["EVAL"]["eval_from"] == "predictions":
        ## path_pred = utils.choose_in_path(Ptk.path_predictions)
        with open(the["path_pred"], 'rb') as f:  # Load the DataFrame
          test_bundle = pickle.load(f)
      elif the["TEST"]["eval_from"] == "model":
        if "train" in the["GENERAL"]["actions"]: # use model just trained
          model.load_state_dict(betterState, strict=False)
        else: # load model from 
          ## path_model = utils.choose_in_path(Ptk.path_models)
          stateDict = torch.load(the["EVAL"]["path_model"], weights_only=False, map_location=DEVICE)
          model.load_state_dict(stateDict, strict=False)
        t1 = time.time()
        path_export = os.path.join(Ptk.path_predictions, f"{logInfo}_preds")
        test_bundle  = Tch.tester.test(model, testloader, extradata=the["DATA"]["return_extradata"], path_export=path_export)
        t2 = time.time()
        print(f"Testing took {t2 - t1:.2f} seconds")
      else:
        raise ValueError("Invalid evalFrom choice!")
      
      # FILTERING
      if the["PROCESS"]["modeFilter"] and "process" in the["actions"]:
        if the["PROCESS"]["path_bundle"]:
          ## path_bundle = utils.choose_in_path(Ptk.path_modeFilter)
          with open(the["PROCESS"]["path_bundle"], 'rb') as f:
            test_bundle = pickle.load(f)
        else:
          try:
            preds = np.array(test_bundle["Preds"])
            modePreds = utils.modeFilter(preds, 10)
            # torch.save(modePreds, Ptk.path_modeFilter)
            test_bundle["Preds"] = modePreds
            if the["PROCESS"]["export_modedPreds"]:
              path_modeFilter = os.path.join(Ptk.path_modeFilter, os.path.splitext(the["PROCESS"]["path_bundle"])[0] + "_moded")
              with open(path_modeFilter, 'wb') as f:
                pickle.dump(test_bundle, f)
          except Exception as e:
            print(f"No predictions (test_bundle) provided for processing: {e}")
      
      Tch.evaluate(test_bundle, the["EVAL"], fold + 1, writer, DATE, Ctl.N_CLASSES, Ctl.labelToClass,
          the["DATA"]["extradata"], Ptk.path_eval, Ptk.path_aprfc, Ptk.path_phaseCharts, the["TEST"]["path_preds"])

    writer.close()
    break # == 1 folc (debug)
 
if __name__ == "__main__":
  # Load default parameters from config.yml
  with open(os.path.join("settings", "config-default.yml"), "r") as file:
    config = yaml.safe_load(file)

  # Set up argparse fir runtime overriding
  parser = argparse.ArgumentParser(description="Override config parameters at runtime")
  parser.add_argument("--action", type=str, help="Action to perform T-rainE-val process)")
  parser.add_argument("--teachkey", type=str, help="Model to teach (spatial, temporal, medianFilter)")
  parser.add_argument("--evalkey", type=str, help="Tests to perform (apref1coma, phaseTiming, phaseChart)")
  parser.add_argument("--n_epochs", type=int, help="Number of epochs")
  parser.add_argument("--dataset", type=str, help="Dataset name")
  parser.add_argument("--fx_batchSize", type=int, help="Batch size for general training")
  args = parser.parse_args()  # parse input arguments

  # Override YAML config with argparse arguments
  for key, value in vars(args).items(): # Use vars(args) to convert the Namespace to a dictionary
    if value is not None:  # Only override if argument is provided
      config[key] = value

  np.random.seed(config["GENERAL"]["seed"]);
  torch.manual_seed(config["GENERAL"]["seed"]);
  random.seed(config["GENERAL"]["seed"])
  learning(**config)
