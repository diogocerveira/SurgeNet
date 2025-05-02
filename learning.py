import time
import os
import torch
import numpy as np
import argparse
import yaml
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle
from src import dataset, environment, machine, teaching

def learning(**the):
  ''' Note: trying out "the" instead of "kwargs"/"config" for readability purposes '''
   # classroom.py (learning environment, paths)
  Csr = environment.Classroom(the["path_root"], the["TRAIN"], the["DATA"], the["MODEL"], id_classroom=the["id_classroom"])
  print(f"A. [classroom] {Csr.id} ({Csr.status})\n   [model] {Csr.studentId} ({Csr.studentStatus})")
  print(f"   For {tuple(the['actions'])}\n")
  # dataset.py (dataset, data handling)
  Ctl = dataset.Cataloguer(Csr.DATA, Csr.path_annots)
  if "process" in the["actions"]:
    if the["PROCESS"]["sample"]:  # video to image frames
      Ctl.sample(the["PROCESS"]["path_rawVideos"], the["PROCESS"]["path_sample"], samplerate=the["PROCESS"]["samplerate"], sampleFormat=the["PROCESS"]["sampleFormat"], processing="", filter_annotated=the["PROCESS"]["filter_annotated"])
      print(f"   [sample] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["resize"]:  # video to image frames
      #Ctl.resize_frames(the["PROCESS"]["path_sample"], f"{the['PROCESS']['path_sample'][:-4]}-025", dim_to=(480, 270))
      Ctl.resize_frames(the["PROCESS"]["path_sample"], "/home/spaceship/Desktop/Diogo/surgenet/data/LDSS-local/png-025", dim_to=(480, 270))
      print(f"   [resize] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["label"]:  # label (csv file) image frames
      Ctl.label(Csr.path_samples, Csr.path_labels, Csr.DATA["labelType"])
      print(f"   [label] {Csr.path_labels}")
  dset = Ctl.build_dataset(Csr.path_samples, Csr.path_labels) # for now the dataset is still used from inside the cataloguer
  print(f"B. [dataset] {the['DATA']['id_dataset']} (#{Ctl.DATASET_SIZE})")
  print(f"   [preprocessing] {Ctl.preprocessing}")
  print(f"   [classes] #{Ctl.N_CLASSES} ({'Balanced' if Ctl.BALANCED else 'Unbalanced'}) {list(Ctl.labelToClass.values())}\n")
  
  # teaching.py (metrics, train validation, test)
  Tch = teaching.Teacher(Csr.TRAIN, the["EVAL"], Ctl, the["device"])
  print(f"C. [training] {Csr.TRAIN['train_point']} on [device] {the['device']}")
  print(f"   [folds] #{Csr.TRAIN['k_folds']} [epochs] #{Csr.TRAIN['HYPER']['n_epochs']} [batch size] {Csr.TRAIN['HYPER']['batchSize']}\n")
  highScore = 0.0

  # Index split of the dataset (virtual)
  for fold, splits in enumerate(Ctl.split(Csr.TRAIN["k_folds"])):
    # iterating folds (trio tuples of idxs [0] and videoIds [1])
    ((train_idxs, valid_idxs, test_idxs), (train_videoIds, valid_videoIds, test_videoIds)) = splits
    if fold in Csr.TRAIN["skipFolds"]:
      continue
    print(f"  Fold {fold + 1} splits:\n\ttrain {train_videoIds}\n\tvalid {valid_videoIds}\n\ttest {test_videoIds}\n")
    # Create the model
    spaceinator = machine.Spaceinator(the["MODEL"], Ctl.dataset.N_CLASSES)
    print(f"    Feature size of {spaceinator.featureSize} at node '{spaceinator.featureNode}'\n")
    timeinator = machine.Timeinator(the["MODEL"], spaceinator.featureSize, Ctl.dataset.N_CLASSES)
    model = machine.PhaseNet(the["MODEL"]["domain"], spaceinator, timeinator, Csr.id, fold + 1)

    dset.path_spaceFeatures = os.path.join(Csr.path_exportedFeatures, f"spacefmaps_f{fold + 1}.pt")
    trainloader, validloader, testloader = Ctl.batch(dset, model.inputType, Csr.TRAIN["HYPER"]["batchSize"], Csr.TRAIN["HYPER"]["clipSize"], train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs)
   
    Tch.writer = SummaryWriter(log_dir=os.path.join(Csr.path_events, model.id.split('_')[1])) # object for logging stats

    # TRAIN
    if "train" in the["actions"]:
      trainedModel, valid_maxScore, betterState = Tch.teach(model, trainloader, validloader, Csr.TRAIN["HYPER"]["n_epochs"], Csr.path_states, Csr.TRAIN["path_resumeModel"])
      if Csr.TRAIN["save_betterState"]:
        path_betterState = os.path.join(Csr.path_states, f"{trainedModel.id}-{valid_maxScore:.2f}")
        torch.save(betterState, path_betterState)
      if Csr.TRAIN["save_lastState"]:
        path_lastState = os.path.join(Csr.path_states, f"{trainedModel.id}-{trainedModel.valid_score:.2f}")
        torch.save(trainedModel.state_dict(), path_lastState)

    # PROCESS - Feature extraction
    if the["PROCESS"]["fx_spatial"] and "process" in the["actions"]:
      t1 = time.time()
      if "train" in the["actions"]:
        spaceinator.load_state_dict(trainedModel.state_dict(), strict=False)
      path_export = os.path.join(Csr.path_features, f"spacefmaps_{model.id.split('_')[1]}.pt")
      spaceinator.export_features(DataLoader(Ctl.dataset, batch_size=Csr.TRAIN["HYPER"]["batchSize"]), path_export, the["PROCESS"]["featureLayer"], the["device"])
      t2 = time.time()
      print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")
    
    # EVALUATE 
    if "eval" in the["actions"]:
      if the["EVAL"]["eval_from"] == "predictions":
        ## path_pred = utils.choose_in_path(Csr.path_predictions)
        try:
          with open(the["path_pred"], 'rb') as f:  # Load the DataFrame
            test_bundle = pickle.load(f)
        except Exception as e:
          print(f"No predictions (test_bundle) provided for evaluation: {e}\n")
          break
      elif the["EVAL"]["eval_from"] == "model":
        try:
          if "train" in the["actions"]: # use model just trained
            model.load_state_dict(betterState, strict=False)
          else: # load model from 
            if the["EVAL"]["path_model"]: # chose before running
              stateDict = torch.load(the["EVAL"]["path_model"], weights_only=False, map_location=the["device"])
              Tch.evaluate(test_bundle, the["EVAL"], fold + 1, Csr, Ctl.N_CLASSES, Ctl.labelToClass,
                Csr.DATA["extradata"], Csr.path_eval, Csr.path_aprfc, Csr.path_phaseCharts, the["TEST"]["path_preds"])
              print("Breaking testing loop on 'Fold 1' because a path_model was provided without 'train' action.")
              break
            else: # choose from the states folder (corresponding to fold)
              stateDict = teaching.get_testState(Csr.path_states, model.id)
              # path_model = environment.choose_in_path(Csr.path_models)
              # stateDict = torch.load(path_model, weights_only=False, map_location=the["device"])
            model.load_state_dict(stateDict, strict=False)
        except Exception as e:
          print(f"Error loading model state for testing: {e}\n")
          break
        t1 = time.time()
        path_export = os.path.join(Csr.path_predictions, f"preds_{model.id.split('_')[1]}.pt")
        test_bundle  = Tch.tester.test(model, testloader, the["EVAL"]["export_testBundle"], path_export=path_export)
        t2 = time.time()
        print(f"Testing took {t2 - t1:.2f} seconds")
        print(test_bundle.head())
      else:
        raise ValueError("Invalid evalFrom choice!")
      
      # PROCESS - Filterting
      if the["PROCESS"]["apply_modeFilter"] and "process" in the["actions"]:
        if the["PROCESS"]["path_bundle"]:
          ## path_bundle = utils.choose_in_path(Csr.path_modeFilter)
          with open(the["PROCESS"]["path_bundle"], 'rb') as f:
            test_bundle = pickle.load(f)
        else:
          try:
            preds = np.array(test_bundle["Preds"])
            modePreds = environment.modeFilter(preds, 10)
            # torch.save(modePreds, Csr.path_modeFilter)
            test_bundle["Preds"] = modePreds
            if the["PROCESS"]["export_modedPreds"]:
              path_modeFilter = os.path.join(Csr.path_modeFilter, os.path.splitext(the["PROCESS"]["path_bundle"])[0] + "_moded")
              with open(path_modeFilter, 'wb') as f:
                pickle.dump(test_bundle, f)
          except Exception as e:
            print(f"No predictions (test_bundle) provided for processing: {e}")
            continue
      
      Tch.evaluate(test_bundle, the["EVAL"]["eval_tests"], model.id, Ctl.labelToClass, Csr)

    Tch.writer.close()
    print(f"\n\t = = = = = \t = = = = =\n\t\t Fold {fold + 1} done!\n\t = = = = = \t = = = = =\n")
    # break # == 1 folc (debug)
  os.system(f"tensorboard --logdir={Csr.path_events}")
 
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