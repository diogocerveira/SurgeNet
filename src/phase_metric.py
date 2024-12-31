import time
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import pickle
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
import sys
import torchvision.models.feature_extraction as fxs

from src import catalogue, machine, utils
# from model_practice import path_root, path_logs, device

def get_bundle(final_imageNames, final_preds, final_targets):
  df = pd.DataFrame({
    'ImageName': final_imageNames,
    'Pred': final_preds,
    'GT': final_targets
  })
  # assuming frames in the structure of videoName_relativeTime
  df[['Video', 'Timestamp']] = df['ImageName'].str.split('_', expand=True) # expand creates two columns, instead of just leaving data as a split list
  df["Timestamp"] = df["Timestamp"].astype(int)
  df = df.sort_values(by=['Video', 'Timestamp']) # Sort the DataFrame by VIDEO and TIME
  #df['AbsoluteTime'] = df.groupby('Video').cumcount() # Convert Relative Time to Absolute Time

  return df


def _phase_lengths_i1(df, n_classes, idx_to_class, samplerate=1):
  outText = []
  lp, lt = len(df["Pred"]), len(df["GT"])
  # n_classes = (max(df["GT"]) + 1)  # assuming regular int encoding
  # n_classes = len(preds[0]) # assuming one hot encoding
  assert lp == lt, "Targets and Preds Size Incompatibility"
  videos = df["Video"].unique()
  phases_preds_sum = np.zeros(n_classes)
  phases_gt_sum = np.zeros(n_classes)

  for idx, video in enumerate(videos):
    print()
    df_filtered = df[df["Video"] == video][["Pred", "GT"]]
    
    classes_video = df_filtered["GT"].unique()
    print(classes_video, end='')
    # Reset for each video
    phases_preds_video = np.zeros(n_classes, dtype=int) # index: the class; value: counts
    
    for _, sample in df_filtered.iterrows():  # Iterate over rows
      pred_class = sample["Pred"]
      
      if pred_class == sample["GT"]:  # Compare the values
        phases_preds_video[pred_class] += 1  # Increment the corresponding class count

    phases_preds_sum += phases_preds_video
    phases_gt_video = df_filtered["GT"].value_counts().reindex(range(n_classes), fill_value=0) # GT video counts
    # print(phases_gt_video)
    # ensures every class is accounted for in cumulative count, even with no occurrences
    phases_gt_sum += phases_gt_video.reindex(range(n_classes), fill_value=0).values

    # Optionally print counts for each video
    # print("phases_preds counts+:", phases_preds_video, '\t', sum(phases_preds_video))
    # print("phases_gt counts:", phases_gt_video.to_numpy(), '\t', sum(phases_gt_video.to_numpy()))
    
    mix = list(zip(phases_preds_video / (60 * samplerate), phases_gt_video / (60 * samplerate)))
    ot = f"\nVideo {video} Phase Report\n" + '\n'.join(
          f"{idx_to_class[i]:<12} | TP {mix[i][0]:<8.2f}min | T {mix[i][1]:<8.2f}min | AE {(mix[i][0] - mix[i][1]):<8.2f}min"
          for i in range(n_classes)
      )
    outText.append(ot)
    print(ot)
  # Average Overall Results
  phases_preds_avg = phases_preds_sum / len(videos)
  phases_gt_avg = phases_gt_sum / len(videos)
  mix = list(zip(phases_preds_avg / (60 * samplerate), phases_gt_avg / (60 * samplerate)))
  
  otavg = "\n\nOverall Average Phase Report\n" + '\n'.join(
        f"{idx_to_class[i]:<12} | TP {mix[i][0]:<8.2f}min | T {mix[i][1]:<8.2f}min | AE {(mix[i][0] - mix[i][1]):<8.2f}min"
        for i in range(n_classes)
    )
  print(otavg)
  outText.append(otavg)
# phase_metric([1, 1, 2, 3, 3, 4, 0, 5], [0, 0, 0, 2, 3, 4, 5, 1])
  return outText

def _phase_lengths_i2(df, n_classes, idx_to_class, samplerate=1):
  outText = []
  lp, lt = len(df["Pred"]), len(df["GT"])
  # n_classes = (max(df["GT"]) + 1)  # assuming regular int encoding
  # n_classes = len(preds[0]) # assuming one hot encoding
  assert lp == lt, "Targets and Preds Size Incompatibility"
  videos = df["Video"].unique()
  phases_preds_sum = np.zeros(n_classes)
  phases_gt_sum = np.zeros(n_classes)
  phases_diff_sum = np.zeros(n_classes)

  for idx, video in enumerate(videos):
    print()
    df_filtered = df[df["Video"] == video][["Pred", "GT"]]
    
    classes_video = pd.Series(df_filtered["GT"].unique())
    # pd.set_option("display.max_rows", 100)
    # print(df_filtered["GT"].iloc[:100])
    print(classes_video.reindex(range(n_classes), fill_value=-1).values)
    # Reset for each video
    
    phases_preds_video = df_filtered["Pred"].value_counts().reindex(range(n_classes), fill_value=0) # GT video counts
    phases_gt_video = df_filtered["GT"].value_counts().reindex(range(n_classes), fill_value=0) # GT video counts
    # ensures every class is accounted for in cumulative count, even with no occurrences
    phases_preds_sum += phases_preds_video.values
    phases_gt_sum += phases_gt_video.values
    phases_diff_sum += np.abs(phases_preds_video.values - phases_gt_video.values)
    # print(phases_diff_sum)
    
    # Optionally print counts for each video
    print("phases_preds counts:", phases_preds_video.values, '\t', sum(phases_preds_video))
    print("phases_gt counts:", phases_gt_video.to_numpy(), '\t', sum(phases_gt_video.to_numpy()))
    
    mix = list(zip(phases_preds_video / (60 * samplerate), phases_gt_video / (60 * samplerate), phases_diff_sum))
    ot = f"\nVideo {video} Phase Report\n" + f"{'':<12} |   T Pred    |    T GT     ||  Diff\n" + '\n'.join(
          f"{idx_to_class[i]:<12} | {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][0] - mix[i][1]):<8.2f}min"
          for i in range(n_classes)
      )
    outText.append(ot)
    print(ot)
  # Average Overall Results
  phases_preds_avg = phases_preds_sum / len(videos) / (60 * samplerate)
  phases_gt_avg = phases_gt_sum / len(videos) / (60 * samplerate)
  phases_diff_avg = phases_diff_sum / len(videos) / (60 * samplerate)
  mix = list(zip(phases_preds_avg, phases_gt_avg, phases_diff_avg))
  
  otavg = "\n\nOverall Average Phase Report\n" + f"{'':<12} | T Avg Pred  |  T Avg GT   || Total AE\n" + '\n'.join(
          f"{idx_to_class[i]:<12} | {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][2]):<8.2f}min"
          for i in range(n_classes)
    )
  print(otavg)
  outText.append(otavg)
# phase_metric([1, 1, 2, 3, 3, 4, 0, 5], [0, 0, 0, 2, 3, 4, 5, 1])
  return outText

def phase_lengths(df, n_classes, idx_to_class, samplerate=1):
  return _phase_lengths_i2(df, n_classes, idx_to_class, samplerate=1)

def phase_graph(df, path_out, date, outText):
    # df = pd.DataFrame({
    #     'ImageName': ['vid1_1', 'vid1_3', 'vid1_2', 'vid2_20', 'vid2_23', 'vid2_40', 'vid2_51', 'vid2_52'],
    #     'GT': [0, 1, 1, 2, 0, 1, 1, 1],  # Example target labels
    #     'Pred': [0, 1, 0, 2, 1, 4, 3, 2],  # Example predictions (e.g., class indices)

    # })
    # Prepare for plotting
    videos = df['Video'].unique()
    num_videos = len(videos)
    print(f'\n\nNumber of videos: {num_videos}')
    color_map = {0: 'springgreen', 1: 'goldenrod', 2: 'tomato', 3: 'mediumslateblue', 4: 'plum', 5: 'deepskyblue'}
    plt.rcParams['font.family'] = 'monospace'
    # for saving all videos to same img
    # fig, axes = plt.subplots(num_videos, 1, figsize=(9, 5), sharex=True) # Create subplots
    # if num_videos == 1: axes = [axes] # prevent errors if only 1 video
    t1 = time.time()
    # Plotting
    for idx, video in enumerate(videos):

      data_video = df[df['Video'] == video][["GT", "Pred"]]
      fig, (ax, ax_text) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(9, 6))
        
      # ax = axes[idx]  # for saving all videos to same image
      
      # Prepare the bar data
      for j, stage in enumerate(data_video.columns):
        start_positions = np.arange(len(data_video))
        class_values = data_video[stage].values
        ax.broken_barh(list(zip(start_positions, np.ones(len(start_positions)))), (j - 0.4, 0.8), facecolors=[color_map[val] for val in class_values])

      # Add vertical separators
      # ax.axvline(x=len(data_video), color='grey', linestyle='--', linewidth=0.5)

      # Set labels
      ax.set_yticks(np.arange(len(data_video.columns)))  # Middle of the two rows
      ax.set_yticklabels(data_video.columns)
      ax.set_xticks([])
      ax.set_xlabel("Time Steps")
      ax.set_title(f"Video: {video}")

      # Add the summary paragraph below the graph
      ax_text.axis('off')  # Hide axis for the text box
      ot = outText[idx].replace('\t', '     ').replace('\u200b', '')  # Removes hidden tabs and zero-width spaces

      # print(ot)
      ax_text.text(0, 0.5, ot, transform=ax_text.transAxes, fontsize=11, 
                  verticalalignment='center', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    
      # Create custom legend
      handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[i]) for i in color_map.keys()]
      labels = [f'{i}' for i in color_map.keys()]
      ax.legend(handles, labels, loc='upper right', title='Classes')

      plt.tight_layout()
      # plt.show(block=True)
      plt.savefig(os.path.join(path_out, "phasegraphs", f"{date}_vid{idx + 1}.png"))
      plt.close(fig)

    plt.figure()
    fig, ax_text = plt.subplots(figsize=(6, 2))
    ax_text.axis('off')  # Hide axis for the text box
    ot = outText[-1].replace('\t', '      ').replace('\u200b', '')  # Removes hidden tabs and zero-width spaces
    # print(ot)
    ax_text.text(0, 0.5, ot, transform=ax_text.transAxes, fontsize=11, 
                verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.tight_layout()
    plt.show(block=True)
    plt.savefig(os.path.join(path_out, f"{date}_oa_avg_phase_rep.png"))
    plt.close()

    t2 = time.time()
    print(f"Video graphing took {t2 - t1:.2f} seconds")
    print(f"\nSaved phase metric graphs to {path_out}")

    
    # time.sleep(5)  # Keeps the script alive for a few seconds

    # Create custom legend for all videos in one image
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[i]) for i in color_map.keys()]
    # labels = [f'{i}' for i in color_map.keys()]
    # fig.legend(handles, labels, loc='upper right', title='Classes')
    # plt.tight_layout()
    # # plt.show(block=True)
    # plt.savefig(path_out)
    # print(f"\nSaved phases graphs to {path_out}")

def phase_metric():

  path_root = "."
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dataset_name = "LDSS_short"
  date = "20241119-12"
  datakey = "local"
  path_logs = os.path.join(path_root, "logs", dataset_name, f"run_{date}")
  path_dataset = os.path.join(path_root, "data", datakey, dataset_name)
  path_events, path_checkpoints, path_config = utils.setup_logs(path_logs) # subfolders
  # path_model = utils.choose_in_path(path_root)
  path_models = os.path.join(path_root, "models") # best models of the run
  path_model = f"./models/{dataset_name}-{date}--0.373134.pth"
  # path_out = os.path.join(path_logs, "phasegraphs")
  # print(f"\nChosen {path_model} model\n")
  model_weights = ResNet50_Weights.DEFAULT
  np.set_printoptions(threshold=50)

  return_extra = {"ids": True, "ts": False, "vids": False}  
  size_batch = 128
  k_folds = 7 if (dataset_name == "LDSS" or dataset_name == "LDSS_short") else 5
  preprocessing = utils.get_preprocessing()
  dataset = catalogue.get_dataset(path_dataset, datakey=datakey, transform=preprocessing, return_extra=return_extra)
  idx_to_class = dataset.idx_to_class
  n_classes = len(idx_to_class.keys()); print(n_classes, "classes")

  n_stages = 2
  n_filters = 64
  filter_size = 3
  n_blocks = 2

  skip_folds = [1, 2, 3, 4, 5, 6]
  path_resume = None

  for fold, ((_, _, test_idx), (train_videos, valid_videos, test_videos)) in enumerate(catalogue.splits(dataset, datakey, k_folds)):  
    if fold in skip_folds: continue # if something crashes allows to skip and restart from certain point
    print(f"Fold {fold + 1} split:\ttrain {train_videos}\n\t\tvalid {valid_videos}\n\t\ttest {test_videos}")
    # test_idx = np.arange(len(dataset))  # whole dataset
    writer = SummaryWriter(os.path.join(path_events, str(f"fold{fold + 1}"))) # initialize SummaryWriter
    # print(f"Fold 1 split: \ttest {test_videos}\n")
    path_features = os.path.join(path_dataset, "features", f"fold{fold + 1}")
    path_model_fx = os.path.join(path_models, "LDSS-20241104-00--0.837800.pth")
    state_dict = torch.load(path_model_fx, weights_only=False, map_location=device)
  
    model_fx = machine.get_resnet50(n_classes, model_weights=model_weights, state_dict=state_dict, extract_features=True)
    last_layer = fxs.get_graph_node_names(model_fx)[0][-2]  # last is fc converted to identity so -2
    fextractor = machine.Fextractor(mode="offline")
    fextractor.load(last_layer, path_features)
    features_size = fextractor.features_size
    guesser = machine.Guesser(features_size, n_classes, n_blocks, n_filters, filter_size)
    improver = machine.Improver(n_classes, n_stages, n_blocks, n_filters, filter_size)
    # _, _, testloader = data.load_data(dataset, size_batch, test_idx=test_idx, datakey=datakey)
    _, _, testloader = catalogue.load_features(fextractor.features, last_layer, dataset, size_batch, test_idx=test_idx)
    
    # model = net.get_resnet50(n_classes, state_dict=torch.load(path_model, weights_only=False, map_location=device))
    model = machine.SurgeNet(n_classes, fextractor, guesser, improver)

    load = False
    if not load:

      BM_test = utils.BunchMetrics({"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1,
                        "confusionMatrix": 1}, n_classes, idx_to_class, "macro", device)
    
      t1 = time.time()
      test.test(model, testloader, device, BM_test) 
      t2 = time.time()
      print(f"Testing took {t2 - t1:.2f} seconds")
      score_test = BM_test.get_accuracy()
      print("Model acc: ", score_test)

      t1 = time.time()
      final_imageNames, final_preds, final_targets = test.test(model, testloader, device, return_extra=return_extra)
      t2 = time.time()
      print(f"Testing took {t2 - t1:.2f} seconds")
      # print(final_imageNames[0], final_preds[0], final_targets[0])
      # torch.nn.functional.one_hot(tensor, n_classes=-1)
      final_bundle = get_bundle(final_imageNames, final_preds, final_targets)
      with open('dataframe.pkl', 'wb') as f: # Save the DataFrame
        pickle.dump(final_bundle, f)

    with open('dataframe.pkl', 'rb') as f:  # Load the DataFrame
      final_bundle = pickle.load(f)

    # print(list(final_bundle.groupby("Video")))
    outText = phase_lengths(final_bundle, n_classes, idx_to_class)
    phase_graph(final_bundle, path_logs, date, outText)

if __name__ == "__main__":
  phase_metric()