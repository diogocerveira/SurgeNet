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
import torchvision.models.feature_extraction as fxs
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


from src import catalogue, machine, teaching, utils, train, validate, test, phase_metric
t2 = time.time()

def learning(**the):
  ''' Trying out 'the' instead of 'kwargs'/'config' for readability purposes '''
  # Setup Parameters
  date = datetime.datetime.now().strftime('%Y%m%d-%H')  # for organising model/logs saving
  device = "cuda" if torch.cuda.is_available() else "cpu" # where to save torch tensors (e.g. during training)
  PTK = utils.Pathtaker(the["path_root"], the["datakey"], the["datasetName"], the["date"])

  # Data
  CTL = catalogue.Cataloguer(PTK.paths_dataset, the["datakey"], the["prepkey"], datapointExtras=the["datapointExtras"], update_labels=False, featureLayer=None)
  
  # TEACHER (Train/Valid/Test and Metrics)
  trainer = teaching.Trainer()
  validater = teaching.Validater()
  tester = teaching.Tester(the["metrics"]["test"], )
  TCH = teaching.Teacher(trainer, validater, tester, the["criterion"], CTL.classWeights, CTL.n_classes, CTL.labelToClass, device)

  # STUDENT (model prototype)
  fextractor = machine.Fextractor(
      CTL.n_classes, os.path.join(PTK.path_models, the["modelName"]), the["fx_modelType"], the["fx_preweights"])
  guesser = machine.Guesser(fextractor.featureSize, CTL.n_classes, the["n_blocks"], the["n_filters"], the["filterTime"])
  refiner = machine.Improver(CTL.n_classes, the["n_stages"], the["n_blocks"], the["n_filters"], the["filter_size"])

  print(f"\nData location: {the['path_root']} | Hardware device: {device}")
  print(f"Dataset ({the["datakey"]}): {the["datasetName"]} | {CTL.n_classes} classes"); print(CTL.labelToClass)
  print(f"{CTL.datasetSize} samples | FX Batch size of {the['fx_batchSize']} | {TCH.k_folds} folds | {the['n_epochs']} epochs\n")

  if the["action"] == "process":
    if the["processkey"] == "sample":
      return True
    elif the["processkey"] == "modeFilter":
      return True
    else:
      raise ValueError("Invalid processkey!")

  # elif the["processkey"] == "fextract":
  #   t1 = time.time()
  #   fextractor.export(DataLoader(dataset=CTL.dataset, batch_size=the["fx_batchSize"]), path_feature, device)
  #   t2 = time.time()
  #   print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")
  # STUDENT (model)
  if the["modelkey"] == "spatial":
    model = machine.SurgeNet(CTL.n_classes, fextractor)
  elif the["modelkey"] == "temporal":
    model = machine.SurgeNet(CTL.n_classes, fextractor, guesser, refiner)
  else:
    raise ValueError("Invalid trainkey!")
 
    
  if the["action"] == "train":
    TCH.teach(model, trainloader, validloader, the["n_epochs"], fold, the["path_resume"],
        PTK.path_checkpoints, writer)
    
  if the["action"] == "eval":
    stateDict = torch.load(path_model, weights_only=False, map_location=device)
    model.load_state_dict(stateDict, strict=False)
    TCH.evaluate(model, testloader, teskey, path_load, path_save, return_extra, fold)

  # "Classroom" with 'k_folds' students (models), all learning from the same teacher one at a time
  for fold, splits in enumerate(CTL.split(the["k_folds"])):
    # iterating through each fold, which has the idxs [0] and corresponding video names [1]
    ((train_idxs, valid_idxs, test_idxs), (train_vidNames, valid_vidNames, test_vidNames)) = splits
    if fold in the["skipFolds"]: continue # after a break (e.g. crash) allows to skip folds and restart from a checkpoint
    print(f"Fold {fold + 1} split:\ttrain {train_vidNames}\n\t\tvalid {valid_vidNames}\n\t\ttest {test_vidNames}")
    writer = SummaryWriter(os.path.join(PTK.path_events, str(f"fold{fold + 1}"))) # object for logging stats
    trainloader, validloader, testloader = CTL.batch(the["batchSize"], train_idxs, valid_idxs, test_idxs)


    


    
    score_test = test_bm.get_accuracy()
    utils.score_log(test_bm, writer, path_logs, fold + 1)
    utils.model_save(path_checkpoints, better_fold_state, datasetName, date, score_test, title="fold", fold=fold + 1)
    if score_test > highScore:
      print(f"\nNew best model (test acc): {highScore:.4f} --> {score_test:.4f}\n")
      bestState = better_fold_state
      highScore = score_test
    
    model.load_state_dict(better_fold_state, strict=False)

    # # break # == 1 folc (debug)

  utils.model_save(path_models, bestState, datasetName, date, highScore)

 
if __name__ == "__main__":
  # Load default parameters from config.yml
  with open(os.path.join("settings", "config-default.yml"), "r") as file:
    config = yaml.safe_load(file)

  # Set up argparse fir runtime overriding
  parser = argparse.ArgumentParser(description="Override config parameters at runtime")
  parser.add_argument("--action", type=str, help="Action to perform (train, eval, process)")
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

  np.random.seed(config["seed"]);
  torch.manual_seed(config["seed"]);
  random.seed(config["seed"])

  learning(**config)