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
import pickle


from src import catalogue, machine, teaching, utils
t2 = time.time()

def learning(**the):
  ''' Note: trying out "the" instead of "kwargs"/"config" for readability purposes '''
  # Setup Parameters
  date = datetime.datetime.now().strftime('%y%m%d%H')  # for organising model/logs saving
  device = "cuda" if torch.cuda.is_available() else "cpu" # where to save torch tensors (e.g. during training)
  PTK = utils.Pathtaker(the["path_root"], the["datakey"], the["run"])
  
  # catalogue.py (dataset, data handling)
  CTL = catalogue.Cataloguer(PTK.path_dataset, the["datakey"], the["preprocessing"],
      extradata=the["extradata"], updateLabels=False)
  
  # teaching.py (metrics, train, validation, test)
  train_metric = teaching.RunningMetric(the["train_metric"], CTL.dataset.n_classes, device, the["agg"], the["computeFreq"], the["updateFreq"])
  valid_metric = teaching.RunningMetric(the["train_metric"], CTL.dataset.n_classes, device, the["agg"], the["computeFreq"] // 2, the["updateFreq"])
  test_metrics = teaching.MetricBunch(the["test_metrics"], CTL.dataset.n_classes, CTL.labelToClass, the["agg"], the["computeFreq"], device)
  trainer = teaching.Trainer(the["hyperparameters"], train_metric, device)
  validater = teaching.Validater(valid_metric, device)
  tester = teaching.Tester(test_metrics, device)
  TCH = teaching.Teacher(trainer, validater, tester, the["criterion"], CTL.classWeights, PTK.path_eval)

  print(f"\nData location: {the['path_root']} | Device: {device}")
  print(f"Dataset ({the['datakey']}): {the['datasetName']} | {CTL.n_classes} classes"); print(CTL.labelToClass)
  print(f"{CTL.datasetSize} samples | FX Batch size of {the['fx_batchSize']} | {the['k_folds']} folds | {the['n_epochs']} epochs\n")

  # Virtual split of the dataset by the indexs 
  for fold, splits in enumerate(CTL.split(the["k_folds"])): # indexs dataset split (k_folds == 1? for no cross validation)
    # iterating through each fold, which has trio tuple of idxs [0] and corresponding trio for the video ids [1]
    ((train_idxs, valid_idxs, test_idxs), (train_vidNames, valid_vidNames, test_vidNames)) = splits
    if fold in the["skipFolds"]: continue # after a break (e.g. crash) allows to skip folds and restart from a checkpoint
    print(f"Fold {fold + 1} split:\ttrain {train_vidNames}\n\t\tvalid {valid_vidNames}\n\t\ttest {test_vidNames}")
    writer = SummaryWriter(os.path.join(PTK.path_events, str(f"fold{fold + 1}"))) # object for logging stats
    bestState = None # for saving the best model state

    # path_result = ''
    spaceinator = machine.Spatinator(CTL.n_classes, the["fx_model"], the["fx_weights"])
    timeinator = machine.Timeinator(spaceinator.featureSize, CTL.n_classes, the["n_stages"], the["n_resBlocks"], the["n_filters"], the["filterTime"])
    
    if the["modelkey"] == "spatial":
      model = machine.SurgeNet(CTL.n_classes, [spaceinator])
      batchMode = "images"
    if the["modelkey"] == "temporal":
      spaceinator.load(PTK.path_features)  # loads into Cataloguer dataset
      batchMode = "features"
      # Get actual data from the split indexs, collate and batch it to a dataloader trio
      model = machine.SurgeNet(CTL.n_classes, [timeinator])
    if the["modelkey"] == "full":
      batchMode = "images"
      model = machine.SurgeNet(CTL.n_classes, [spaceinator, timeinator])
    trainloader, validloader, testloader = CTL.batch(
          the["batchSize"], batch_mode=batchMode, train_idx=train_idxs, valid_idx=valid_idxs, test_idx=test_idxs)
    
    if "train" in the["actions"]:
      bestState = TCH.teach(model, trainloader, validloader, the["modelkey"], the["n_epochs"], fold, the["path_resume"],
          PTK.path_checkpoints, writer)
      
    if the["action"] == "process":
      if "sample" in the["processkeys"]:
        pass
      if "fx-spatial" in the["processkeys"]:
        t1 = time.time()
        spaceinator.export_features(DataLoader(CTL.dataset, batch_size=the["batchSize"]), PTK.path_features, device)
        t2 = time.time()
        print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")
      if "mode-filter" in the["processkeys"]:
        if "train" in the["actions"]:
        # preds = torch.load(os.path.join(PTK.path_predictions, fold))
        # path_result = os.path.join(PTK.path_eval, f"results_fold{fold + 1}")
        with open(path_result, 'rb') as f:  # Load the DataFrame
          pundle = pickle.load(f)
        preds = np.array(pundle["Preds"])
        modePreds = utils.modeFilter(preds, 10)
        # torch.save(modePreds, PTK.path_modeFilter)
        pundle["Preds"] = modePreds
        path_result = os.path.splitext(path_result)[0] + "_moded"
        with open(path_result, 'wb') as f:
          pickle.dump(pundle, f)
      else:
        raise ValueError("No valid processkeys!")
    
    if "eval" in the["actions"]:
      if the["eval_in"] == "predictions":
        preds = torch.load(the["path_prediction"], map_location=device)
        TCH.evaluate(preds, the["evalkeys"], fold + 1, writer, date, CTL.n_classes, CTL.labelToClass, model,
          the["extradata"], PTK.path_eval, PTK.path_aprfc, PTK.path_phaseCharts, the["path_result"])
      elif the["eval_in"] == "model":
        if "train" in the["actions"]:
          model.load_state_dict(bestState, strict=False)
        elif the["path_model"] is not None:
          stateDict = torch.load(the["path_model"], weights_only=False, map_location=device)
          model.load_state_dict(stateDict, strict=False)
        else:
          raise ValueError("Invalid path_model choice!")
      else:
        raise ValueError("Invalid eval_in choice!")
      
      TCH.evaluate(testloader, the["evalkeys"], fold + 1, writer, date, CTL.n_classes, CTL.labelToClass, model,
          the["extradata"], PTK.path_eval, PTK.path_aprfc, PTK.path_phaseCharts, the["path_prediction"])

      


 
    
    # score_test = test_bm.get_accuracy()
    # utils.score_log(test_bm, writer, path_logs, fold + 1)
    # utils.model_save(path_checkpoints, better_fold_state, datasetName, date, score_test, title="fold", fold=fold + 1)
    # if score_test > highScore:
    #   print(f"\nNew best model (test acc): {highScore:.4f} --> {score_test:.4f}\n")
    #   bestState = better_fold_state
    #   highScore = score_test
    
    # model.load_state_dict(better_fold_state, strict=False)

    # # break # == 1 folc (debug)

  # utils.model_save(path_models, bestState, datasetName, date, highScore)

 
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
