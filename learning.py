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
import gc

def learning(**the):
  ''' Note: trying out "the" instead of "kwargs"/"config" for readability purposes '''
   # classroom.py (learning environment, paths)
  Csr = environment.Classroom(the["path_root"], the["TRAIN"], the["DATA"], the["MODEL"], id_classroom=the["id_classroom"], path_data=the["path_data"])
  print(f"A. [classroom] {Csr.id} ({Csr.status})\n   [model] {Csr.studentId} ({Csr.studentStatus})")
  print(f"   For {tuple(the['actions'])}\n")
  # Csr.update_fileIds()
  
  # dataset.py (dataset, data handling)
  Ctl = dataset.Cataloguer(Csr.DATA, Csr.path_annots)
  if "process" in the["actions"]:
    if the["PROCESS"]["sample"]:  # video to image frames
      Ctl.sample(the["PROCESS"]["path_rawVideos"], the["PROCESS"]["path_sample"], samplerate=the["PROCESS"]["samplerate"], sampleFormat=the["PROCESS"]["sampleFormat"], processing="", filter_annotated=the["PROCESS"]["filter_annotated"])
      print(f"   [sample] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["resize"]:  # video to image frames
      #Ctl.resize_frames(the["PROCESS"]["path_sample"], f"{the['PROCESS']['path_sample'][:-4]}-025", dim_to=(480, 270))
      Ctl.resize_frames(Csr.path_samples, "/home/spaceship/Desktop/Diogo/surgenet/data/LDSS-local/png-025", dim_to=(480, 270))
      print(f"   [resize] {the['PROCESS']['path_sample']} ({the['PROCESS']['samplerate']} fps)")
    if the["PROCESS"]["crop"]:
      Ctl.crop_frames(Csr.path_samples, Csr.path_samples)
      print(f"   [crop] {Csr.path_samples}")
    if the["PROCESS"]["label"]:  # label (csv file) image frames
      Ctl.label(Csr.path_samples, Csr.path_labels, Csr.DATA["labelType"])
      print(f"   A new labels.csv was created {Csr.path_labels}\n")

  dset = Ctl.build_dataset(Csr.DATA["id_dataset"].split('-')[1], Csr.path_samples, Csr.path_labels, headType=the["MODEL"]["headType"]) # for now the dataset is still used from inside the cataloguer
  print(f"B. [dataset] {the['DATA']['id_dataset']} (#{len(dset)})")
  print(f"   [label type] {'-'.join(list(dset.labelType))}")
  print(f"   [preprocessing] {Ctl.preprocessingType}")
  print(f"   [classes] #{dset.n_classes} ({'Balanced' if dset.BALANCED else 'Unbalanced'}) {list(dset.labelToClassMap.keys())}")
  print(f"   [weights] {dset.classWeights}")
  print(f"   [phases] #{len(dset.phases)} ", dset.phases, '\n')

  # teaching.py (metrics, train validation, test)
  Tch = teaching.Teacher(Csr.TRAIN, the["EVAL"], dset, the["device"])
  print(f"C. [training] {Csr.TRAIN['train_point']} on [device] {the['device']}")
  print(f"   [folds] #{Csr.TRAIN['k_folds']} [epochs] #{Csr.TRAIN['HYPER']['n_epochs']} [batch sizes] s{Csr.TRAIN['HYPER']['spaceBatchSize']} / t{Csr.TRAIN['HYPER']['timeBatchSize']}")
  print(f"   [criterions] {Tch.criterions} w/ weights {Tch.criterionWeights}")
  print(f"   [headtype] {dset.headType} ")
  if len(dset.headType) == 2 and dset.headType[0] == "multi" and dset.headType[1] == "phase":
    for i, head in enumerate(dset.headSplits):
      print(f"      Head {i} classes: {[dset.classToLabelMap[c] for c in head]}")
  print()
  highScore = 0.0
  # print(Ctl.get_mean_std(dset))

  # Index split of the dataset (virtual)
  for fold, splits in enumerate(Ctl.split(Csr.TRAIN["k_folds"], dset)):
    # iterating folds (trio tuples of idxs [0] and videoIds [1])
    ((train_idxs, valid_idxs, test_idxs), (train_videoIds, valid_videoIds, test_videoIds)) = splits
    with open(Csr.path_studentProfile) as f:
      # Add fold info to the student profile yaml file
      data = yaml.safe_load(f)
      data["VIDEO_SPLITS"][f"Fold{fold + 1}"] = {"train": ','.join(train_videoIds), "valid": ','.join(valid_videoIds), "test": ','.join(test_videoIds)}
    with open(Csr.path_studentProfile, 'w') as f:
      yaml.dump(data, f)
    if fold in Csr.TRAIN["skipFolds"]:
      continue
    print(f"  Fold {fold + 1} splits:\n\ttrain {train_videoIds}\n\tvalid {valid_videoIds}\n\ttest {test_videoIds}\n")

    # Create the model
    spaceinator = machine.Spaceinator(the["MODEL"], dset.n_classes)
    print(f"    Feature size of {spaceinator.featureSize} at node '{spaceinator.featureNode}'")
    timeinator = machine.Timeinator(the["MODEL"], spaceinator.featureSize, dset.n_classes)
    print(f"    Feature size of {timeinator.featureSize} at node '{timeinator.featureNode}'\n")
    classifier = machine.Classinator(the["MODEL"], spaceinator.featureSize, timeinator.featureSize, dset.n_classes)
    model = machine.Phasinator(the["MODEL"]["domain"], spaceinator, timeinator, classifier, f"{Csr.studentId}_f{fold + 1}").to(the["device"])

    path_spaceFeats, featuresId = teaching.get_path_spaceFeat(Csr.path_spaceFeats, f"f{fold + 1}")
    trainloader, validloader, testloader = Ctl.batch(dset, model.inputType, Csr.TRAIN["HYPER"], path_spaceFeats, Csr.DATA["multiClips"], Csr.DATA["multiClipStride"], train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs)

    if "train" in the["actions"] or "eval" in the["actions"]:
      Tch.writer = SummaryWriter(log_dir=os.path.join(Csr.path_events, model.id.split('_')[1])) # object for logging stats
   
    # TRAIN
    if "train" in the["actions"]:
      model, valid_maxScore, betterState = Tch.teach(model, trainloader, validloader, Csr.TRAIN["HYPER"]["n_epochs"], Csr.path_state, dset.labels, Csr.TRAIN["path_resumeModel"])
      if Csr.TRAIN["save_betterState"]:
        model.id = f"{model.id}-{valid_maxScore:.2f}" # update model id with the best validation score
        torch.save(betterState, os.path.join(Csr.path_state, f"state_{model.id.split('_')[1]}.pt"))
      if Csr.TRAIN["save_lastState"]:
        model.id = f"{model.id}-{model.valid_lastScore:.2f}"
        torch.save(model.state_dict(), os.path.join(Csr.path_state, f"state_{model.id.split('_')[1]}.pt"))

    # PROCESS - Feature extraction
    if the["PROCESS"]["fx_spatial"] and "process" in the["actions"]:
      print(f"D. [Processing] Extract Spatial Features")
      t1 = time.time()
      if "train" in the["actions"]:
        spaceinator.load_state_dict(model.state_dict(), strict=False)
        fx_stateId = model.id
      elif the["PROCESS"]["fx_load_models"]:
        fx_stateDict, fx_stateId = teaching.get_testState(Csr.path_state, model.id)
        spaceinator.load_state_dict(fx_stateDict, strict=False)
      path_export = os.path.join(Csr.path_feats, f"feats_{fx_stateId.split('_')[1]}.pt")
      spaceinator.export_features(DataLoader(dset, batch_size=Csr.TRAIN["HYPER"]["spaceBatchSize"]), path_export, the["PROCESS"]["featureLayer"], the["device"])
      t2 = time.time()
      print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")
      # clear GPU memory
      
    if the["PROCESS"]["fx_temporal"] and "process" in the["actions"]:
      print(f"D. [Processing] Extract Temporal Features")
      t1 = time.time()
      if "train" in the["actions"]:
        timeinator.load_state_dict(model.state_dict(), strict=False)
        fx_stateId = model.id
      elif the["PROCESS"]["fx_load_models"]:
        fx_stateDict, fx_stateId = teaching.get_testState(Csr.path_state, model.id)
        timeinator.load_state_dict(fx_stateDict, strict=False)
      path_export = os.path.join(Csr.path_feats, f"feats_{fx_stateId.split('_')[1]}.pt")
      timeinator.export_features(DataLoader(dset, batch_size=Csr.TRAIN["HYPER"]["timeBatchSize"]), path_export, the["PROCESS"]["featureLayer"], the["device"])
      t2 = time.time()
      print(f"Exporting features took {(t2 - t1) // 3600} hours and {(((t2 - t1) % 3600) / 60):.1f} minutes!")
      # clear GPU memory
      

    # EVALUATE 
    if "eval" in the["actions"]:
      if the["EVAL"]["eval_from"] == "predictions":
        print(f"E. [Evaluating] from preds {Csr.path_preds}")
        predId = teaching.get_pred(Csr.path_preds, fold + 1)
        path_pred = os.path.join(Csr.path_preds, f"{predId}.pt")
        try:
          with open(path_pred, 'rb') as f:  # Load the DataFrame
            test_bundle = pickle.load(f)
        except Exception as e:
          print(f"No predictions (test_bundle) provided for evaluation at fold {fold}: {e}\n")
          break
      elif the["EVAL"]["eval_from"] == "model":
        try:
          if "train" in the["actions"]: # use model just trained
            print(f"E. [Evaluating] from trained model {model.id}")
            pass # model already loaded
          else: # load model from 
            if the["EVAL"]["path_model"]: # chose before running NOT WORKING
              print(f"E. [Evaluating] from saved model {the['EVAL']['path_model']}")
              assert 1 == 0, "Choosing a specific model for testing is not yet implemented!"
              test_stateDict, test_stateId = torch.load(the["EVAL"]["path_model"], weights_only=False, map_location=the["device"])
              Tch.evaluate(test_bundle, the["EVAL"], fold + 1, Csr, Ctl.N_CLASSES, Csr.DATA["extradata"],
                  Csr.path_eval, Csr.path_aprfc, Csr.path_ribbons, the["TEST"]["path_preds"])
              print("Breaking testing loop on 'Fold 1' because a path_model was provided without 'train' action.")
              break
            else: # choose from the states folder (corresponding to fold)
              test_stateDict, modelId = teaching.get_testState(Csr.path_state, model.id)
              print(f"E. [Evaluating] from saved model {modelId}")
              # path_model = environment.choose_in_path(Csr.path_state)
              # stateDict = torch.load(path_model, weights_only=False, map_location=the["device"])
            model.load_state_dict(test_stateDict, strict=False)
            model.id = modelId
        except Exception as e:
          print(f"Error loading model state for testing: {e}\n")
          break
        t1 = time.time()
        path_export = os.path.join(Csr.path_preds, f"preds_{model.id}")
        testloader.dataset.set_preprocessing(("test", dset.preprocessingType))  # set preprocessing for test set
        test_bundle, test_score  = Tch.tester.test(model, testloader, dset.labels, the["EVAL"]["export_testBundle"], path_export=path_export)
        t2 = time.time()
        print(f"\nTest Score: {test_score:.04f} (took {t2 - t1:.2f}s)\n")
        # print(test_bundle.head())
      else:
        raise ValueError("Invalid evalFrom choice!")
      
      # PROCESS - Filterting
      if the["PROCESS"]["apply_modeFilter"] and "process" in the["actions"]:
        print(f"\nF. [Processing] Applying mode filter to preds {Csr.path_preds}")
        ## path_bundle = utils.choose_in_path(Csr.path_modeFilter)
        assert the["EVAL"]["eval_from"] == "predictions", "Mode filtering only works when evaluating from predictions"
        assert not test_bundle.empty, "No test bundle selected for mode filtering and evaluating"
        try:
          preds = np.array(test_bundle["Pred"])
          mode = 200
          modePreds = environment.modeFilter(preds, mode)
          # torch.save(modePreds, Csr.path_modedPreds)
          test_bundle["Pred"] = modePreds
          if the["PROCESS"]["export_modedPreds"]:
            predId = teaching.get_pred(Csr.path_preds, fold + 1)
            path_modedPred = os.path.join(Csr.path_modedPreds, f"moded{mode}-{os.path.splitext(predId)[0]}.pt")
            with open(path_modedPred, 'wb') as f:
              pickle.dump(test_bundle, f)
            Tch.tester.exportPrefix = f"moded{mode}-"
        except Exception as e:
          print(f"No predictions (test_bundle) provided for processing: {e}")
          continue
      Tch.evaluate(test_bundle, the["EVAL"]["eval_tests"], model.id.split("_")[1], Csr)

    try: # not defined if only processing
      Tch.writer.close()
    except:
      pass
    print(f"\n\t = = = = = \t = = = = =\n\t\t Fold {fold + 1} done!\n\t = = = = = \t = = = = =\n")
    # break # == 1 folc (debug)
  if "process" in the["actions"] and the["PROCESS"]["build_globalCM"]:
    print(f"G. [Processing] Generating all-fold confusion matrix {Csr.path_preds}")
    path_preds = Csr.path_modedPreds if the["PROCESS"]["apply_modeFilter"] and "process" in the["actions"] else Csr.path_preds
    environment.build_globalCM(path_preds, Csr.path_aprfc, dset.n_classes, dset.labelType, dset.headType, dset.labelToClassMap, prefix=Tch.tester.exportPrefix)

  print(f"\n\n ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨ ¨\n\n")
  # os.system(f"tensorboard --logdir={Csr.path_classroom}")
 

if __name__ == "__main__":
  # Load default parameters from config.yml

  # configIds = ["1cfg_SP1-RN50.yml", "2cfg_SP1-FX-RN50.yml", "3cfg_SP1-l4FT-RN50.yml",
  #              "4cfg_SP1-FT-RN50.yml", "5cfg_SP1-l4FT-RN50-aug.yml",
  #              "6cfg_SP1-l4FT-RN50-dynaug.yml", "7cfg_SP1-l4FT-dTsRN50-dynaug.yml",
  #              "8cfg_SP1-l4FT-pTsRN50-dynaug.yml", "9cfg_MP2-l4FT-RN50-PTS-1L-dynaug.yml",
  #              "10cfg_MP1-l4FT-RN50-PTS-2L-dynaug.yml", "11cfg_SP1-tecno.yml",
  #              "12cfg_MP2-tecno.yml", "13cfg_MP1-mTecno.yml"]
  configIds = ["1cfg_SP1-RN50.yml"]
  for configId in configIds:
    with open(os.path.join("settings", configId), "r") as file:
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
      config["actions"] = []  # default actions if none specified
    if config["device"] == "auto" or not config["device"]: # where to save torch tensors (e.g. during training)
      config["device"] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running {configId}...\n")
    learning(**config)