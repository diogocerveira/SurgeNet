import time
t1 = time.time()
import os
import torch
import numpy as np
import argparse
import yaml
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle
from src import classroom, dataset, machine, teaching
t2 = time.time()

def learning(**the):
  ''' Note: trying out "the" instead of "kwargs"/"config" for readability purposes '''
  # classroom.py (learning environment, paths)
  Clr = classroom.Classroom(the["path_root"], the["TRAIN"], the["DATA"], the["id_classroom"])
  print(f"A. [classroom] {Clr.id} ({Clr.status})\n")

  # catalogue.py (dataset, data handling)
  Dtk = dataset.Cataloguer(the["DATA"], Clr)
  if "process" in the["actions"]:
    if the["PROCESS"]["sample"]:  # video to image frames
      Dtk.sample(the["PROCESS"]["path_rawVideos"], the["PROCESS"]["path_sample"], samplerate=the["PROCESS"]["samplerate"], sampleFormat=the["PROCESS"]["sampleFormat"], processing="", filter_annotated=the["PROCESS"]["filter_annotated"])
      print(f"   [sample] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["resize"]:  # video to image frames
      #Dtk.resize_frames(the["PROCESS"]["path_sample"], f"{the['PROCESS']['path_sample'][:-4]}-025", dim_to=(480, 270))
      Dtk.resize_frames(the["PROCESS"]["path_sample"], "/home/spaceship/Desktop/Diogo/surgenet/data/LDSS-local/png-025", dim_to=(480, 270))
      print(f"   [resize] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["label"]:  # label (csv file) image frames
      Dtk.label(Clr.path_samples, Clr.path_labels, the["PROCESS"]["labelType"])
      print(f"   [label] {Clr.path_labels}")
  Dtk.build_dataset(the["DATA"]["id_dataset"], the["DATA"]["preprocessing"], the["DATA"]["recycle_itemIds"])
  print(f"B. [dataset] {the['DATA']['id_dataset']} (#{Dtk.DATASET_SIZE})")
  print(f"   [preprocessing] {Dtk.preprocessing}")
  print(f"   [classes] #{Dtk.N_CLASSES} ({'Balanced' if Dtk.BALANCED else 'Unbalanced'}) {list(Dtk.labelToClass.values())}\n")

  # teaching.py (metrics, train validation, test)
  Tch = teaching.Teacher(the["TRAIN"], the["EVAL"], Dtk, the["device"])
  print(f"C. [training] {the['TRAIN']['train_point']} on [device] {the['device']}")
  print(f"   [folds] #{the['TRAIN']['k_folds']} [epochs] #{the['TRAIN']['HYPER']['n_epochs']} [batches] #{the['TRAIN']['HYPER']['batchSize']}\n")
  highScore = 0.0
  # Index split of the dataset (virtual)
  for fold, splits in enumerate(Dtk.split(the["TRAIN"]["k_folds"])):
    # iterating folds (trio tuples of idxs [0] and videoIds [1])
    ((train_idxs, valid_idxs, test_idxs), (train_videoIds, valid_videoIds, test_videoIds)) = splits
    if fold in the["TRAIN"]["skipFolds"]:
      continue
    print(f"  Fold {fold + 1} splits:\n\ttrain {train_videoIds}\n\tvalid {valid_videoIds}\n\ttest {test_videoIds}")
    # Create the model
    spaceinator = machine.Spaceinator(the["MODEL"], the["device"], N_CLASSES=Dtk.N_CLASSES)
    timeinator = machine.Timeinator(the["MODEL"], spaceinator.FEATURE_SIZE, Dtk.N_CLASSES)
    model = machine.PhaseNet(the["MODEL"]["domain"], spaceinator, timeinator, Dtk.N_CLASSES)

    trainloader, validloader, testloader = Dtk.batch(
        the["TRAIN"]["HYPER"]["batchSize"], batchMode, train_idx=train_idxs, valid_idx=valid_idxs, test_idx=test_idxs)
    
    modelId = f"{Clr.id}-{fold + 1}"  # model name for logging/saving
    writer = SummaryWriter(os.path.join(Clr.path_events, modelId)) # object for logging stats


    # PROCESS - Feature extraction
    if the["PROCESS"]["fx_spatial"] and "process" in the["actions"]:
      t1 = time.time()
      spaceinator.export_features(DataLoader(Dtk.dataset, batch_size=the["TRAIN"]["HYPER"]["batchSize"]), Clr.path_features, fold, ["flatten"], the["device"])
      t2 = time.time()
      print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")

    # TRAIN
    if "train" in the["actions"]:
      betterState, valid_minLoss = Tch.teach(model, trainloader, validloader, the["TRAIN"]["HYPER"]["n_epochs"], modelId,
          Clr.path_checkpoints, writer)
      if the["TRAIN"]["save_betterModel"]:
        machine.save_model(Clr.path_models, betterState, modelId, CLASSROOM, 0.0, title="fold", fold=fold + 1)
    
    # EVALUATE 
    if "eval" in the["actions"]:
      if the["EVAL"]["eval_from"] == "predictions":
        ## path_pred = utils.choose_in_path(Clr.path_predictions)
        with open(the["path_pred"], 'rb') as f:  # Load the DataFrame
          test_bundle = pickle.load(f)
      elif the["TEST"]["eval_from"] == "model":
        if "train" in the["actions"]: # use model just trained
          model.load_state_dict(betterState, strict=False)
        else: # load model from 
          ## path_model = utils.choose_in_path(Clr.path_models)
          stateDict = torch.load(the["EVAL"]["path_model"], weights_only=False, map_location=the["device"])
          model.load_state_dict(stateDict, strict=False)
        t1 = time.time()
        path_export = os.path.join(Clr.path_predictions, f"{modelId}_preds")
        test_bundle  = Tch.tester.test(model, testloader, extradata=the["DATA"]["return_extradata"], path_export=path_export)
        t2 = time.time()
        print(f"Testing took {t2 - t1:.2f} seconds")
      else:
        raise ValueError("Invalid evalFrom choice!")
      
      # PROCESS - Filterting
      if the["PROCESS"]["modeFilter"] and "process" in the["actions"]:
        if the["PROCESS"]["path_bundle"]:
          ## path_bundle = utils.choose_in_path(Clr.path_modeFilter)
          with open(the["PROCESS"]["path_bundle"], 'rb') as f:
            test_bundle = pickle.load(f)
        else:
          try:
            preds = np.array(test_bundle["Preds"])
            modePreds = classroom.modeFilter(preds, 10)
            # torch.save(modePreds, Clr.path_modeFilter)
            test_bundle["Preds"] = modePreds
            if the["PROCESS"]["export_modedPreds"]:
              path_modeFilter = os.path.join(Clr.path_modeFilter, os.path.splitext(the["PROCESS"]["path_bundle"])[0] + "_moded")
              with open(path_modeFilter, 'wb') as f:
                pickle.dump(test_bundle, f)
          except Exception as e:
            print(f"No predictions (test_bundle) provided for processing: {e}")
      
      Tch.evaluate(test_bundle, the["EVAL"], fold + 1, writer, CLASSROOM, Dtk.N_CLASSES, Dtk.labelToClass,
          the["DATA"]["extradata"], Clr.path_eval, Clr.path_aprfc, Clr.path_phaseCharts, the["TEST"]["path_preds"])

    writer.close()
    break # == 1 folc (debug)
 
if __name__ == "__main__":
  # Load default parameters from config.yml
  with open(os.path.join("settings", "config-default.yml"), "r") as file:
    config = yaml.safe_load(file)

  # Set up argparse fir CLASSROOMtime overriding
  parser = argparse.ArgumentParser(description="Override config parameters at CLASSROOMtime")
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

  np.random.seed(config["seed"]);
  torch.manual_seed(config["seed"]);
  random.seed(config["seed"])
  if not config["actions"]:
    print("\nNo action specified. Exiting...\n")
    exit()
  if config["device"] == "auto" or not config["device"]: # where to save torch tensors (e.g. during training)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
  print()
  learning(**config)