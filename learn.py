import time
t1 = time.time()
import os
import torch
import numpy as np
import datetime

import torch.nn as nn
from torchvision import datasets # for the CIFAR10 image dataset

from sklearn.model_selection import KFold, GroupKFold, train_test_split
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torcheval.metrics import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import yaml
import random
import torchvision.models.feature_extraction as fxs

from src import data, utils, net, train, validate, test, phase_metric
t2 = time.time()

def learn(**the):
  # I'm trying out 'the' instead of 'kwargs' for readability purposes
  # ---------------------------- #
  # dataset_name = "LDSS"
  # dataset_name = "CIFAR10"
  dataset_name = "LDSS_short"
  # ---------------------------- #

  # Setup Parameters
  device = "cuda" if torch.cuda.is_available() else "cpu"
  np.random.seed(the["seed"])
  torch.manual_seed(the["seed"])
  random.seed(the["seed"])
  datakey = "local" if (dataset_name == "LDSS" or dataset_name == "LDSS_short") else "external" # if its local ldss, there's group splitting
  date = datetime.datetime.now().strftime('%Y%m%d-%H')
  path_dataset, path_logs, path_events, path_checkpoints, path_config, path_models = utils.get_paths(the["path_root"], datakey, dataset_name, date).values()
  print(f"\nData location: {the['path_root']} | Hardware device: {device}")

  # Model/Data
  model_weights = ResNet50_Weights.DEFAULT if the["fx_weights"] == "default" else None # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
  k_folds = 7 if (dataset_name == "LDSS" or dataset_name == "LDSS_short") else 5 # CIFAR10 has 60000 images (10000 for test)
  preprocessing = utils.get_preprocessing()
  dataset, idxToClass, n_classes, datasetSize, classWeights = data.get_dataset(path_dataset, datakey=datakey, transform=preprocessing, update_labels=False, return_extra=the["return_extra"])
  print(f"Dataset ({datakey}): {dataset_name} | {n_classes} classes"); print(idxToClass)
  print(f"{datasetSize} samples | Batch size of {the["size_batch"]} | {k_folds} folds | {the["n_epochs"]} epochs\n")

  # Metrics
  criterion = nn.CrossEntropyLoss(weight=classWeights)
  agg = "macro" # accuracy is locked to micro (see utils.py)
  period_compute, period_update = 100, 1  # num batches between metric computations
  metric_train, metric_valid = MulticlassAccuracy(device=device), MulticlassAccuracy(device=device)
  RM_train = utils.RunningMetric(metric_train, period_compute)
  RM_valid = utils.RunningMetric(metric_valid, period_compute // 2)
  BM_test = utils.BunchMetrics({"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1,
                          "confusionMatrix": 1}, n_classes, idxToClass, agg, device)
                          

  # path_resume = os.path.join(path_checkpoints, f"checkpoint_epoch1.pth")
  # ----------------- #
  ## LEARN ##
  high_score, better_overall_state = -np.inf, {}
  for fold, ((train_idx, valid_idx, test_idx), (train_videos, valid_videos, test_videos)) in enumerate(data.splits(dataset, datakey, k_folds)):
    if fold in skip_folds: continue # if something crashes allows to skip and restart from certain point
    print(f"Fold {fold + 1} split:\ttrain {train_videos}\n\t\tvalid {valid_videos}\n\t\ttest {test_videos}")
    
    writer = SummaryWriter(os.path.join(path_events, str(f"fold{fold + 1}"))) # initialize SummaryWriter

    ## CONSTRUCTION SITE
    path_features = os.path.join(path_dataset, "features", f"fold{fold + 1}")
    path_model_fx = os.path.join(path_models, "LDSS-20241104-00--0.837800.pth")
    state_dict = torch.load(path_model_fx, weights_only=False, map_location=device)
    model_fx = net.get_resnet50(n_classes, model_weights=model_weights, state_dict=state_dict, extract_features=True)
    last_layer = fxs.get_graph_node_names(model_fx)[0][-2]  # last is fc converted to identity so -2
    # print(last_layer)  # get the last layer
    nodes_to_extract = [last_layer]
    os.makedirs(path_features, exist_ok=True)
    n_stages = 2
    n_filters = 64
    filter_size = 3
    n_blocks = 2

  
    fextractor = net.Fextractor(model_fx, writer, nodes_to_extract, mode="offline")

    ### TO EXPORT FEATURES
    # t1 = time.time()
    # fextractor.export(DataLoader(dataset=dataset, batch_size=size_batch), path_features, device)
    # t2 = time.time()
    # print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!!")
    # continue
    ###

    fextractor.load(last_layer, path_features)
    trainloader, validloader, testloader = data.load_features(fextractor.features, last_layer, dataset, size_batch, train_idx, valid_idx, test_idx)
    # for i in trainloader:
      # print(i)
      # assert 1 == 0

    features_size = fextractor.features_size
    guesser = net.Guesser(features_size, n_classes, n_blocks, n_filters, filter_size)
    improver = net.Improver(n_classes, n_stages, n_blocks, n_filters, filter_size)
    
    # model = net.get_resnet50(n_classes, device, model_weights)
    model = net.SurgeNet(n_classes, fextractor, guesser, improver)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # updates the optimizer by the gamma factor after the step_size
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # break

  # Practice (Full run of Train and Validation)
    better_fold_state = utils.practice(model, trainloader, validloader, criterion, optimizer,
                                      scheduler, ES, n_epochs, fold, writer, path_checkpoints,
                                      RM_train, RM_valid, device, path_resume)
    
    # EVAL
    test.test(model, testloader, device, BM_test)  # local test results saved on BM_test
    # BM_test.display() # problematic
    score_test = BM_test.get_accuracy()
    utils.score_log(BM_test, writer, path_logs, fold + 1)
    utils.model_save(path_checkpoints, better_fold_state, dataset_name, date, score_test, title="fold", fold=fold + 1)
    if score_test > high_score:
      print(f"\nNew best model (test acc): {high_score:.4f} --> {score_test:.4f}\n")
      better_overall_state = better_fold_state
      high_score = score_test
    
    model.load_state_dict(better_fold_state, strict=False)

    # # break # == 1 folc (debug)

  utils.model_save(path_models, better_overall_state, dataset_name, date, high_score)

if __name__ == "__main__":
  # Load default parameters from config.yml
  with open(os.path.join("config", "config.yml"), "r") as file:
    config = yaml.safe_load(file)

  # Set up argparse fir runtime overriding
  parser = argparse.ArgumentParser(description="Override config parameters at runtime")
  parser.add_argument("--action", type=str, help="Action to perform (train/eval)")
  parser.add_argument("--lr", type=float, help="Learning rate")
  parser.add_argument("--n_epochs", type=int, help="Number of epochs")
  parser.add_argument("--dataset", type=str, help="Dataset name")
  parser.add_argument("--fx_batchSize", type=int, help="Batch size for feature extractor training")
  args = parser.parse_args()  # parse input arguments

  # Override YAML config with argparse arguments
  for key, value in vars(args).items(): # Use vars(args) to convert the Namespace to a dictionary
    if value is not None:  # Only override if argument is provided
      config[key] = value

  learn(**config)