import os
from torch.utils.data import Dataset  # for the custom dataset (derived class)
import pandas as pd
import json
import chardet
import random
import imageio as iio
import torch
from torchvision import datasets
import numpy as np
import PIL as Image
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from torchvision import datasets
from torch.utils.data import DataLoader
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## SETUP ##
seed = 53
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
## SETUP ##

class Showcaser:
  def __init__(self):
    pass
  def show(self, imgs, im_type=torch.Tensor):
    if not isinstance(imgs, list):
      imgs = [imgs]
    size_imgs = len(imgs); print(size_imgs)
    cols = 5
    rows = 7 
    for img in imgs:  
      fig, axes = plt.subplots(rows, cols, figsize=(8, 11))
      for i, ax in enumerate(axes.flat):  # Iterate over the tensors and plot each one
        if i < size_imgs:
          if im_type == torch.Tensor:
            if i.dim() == 1:
              print("Tensor is 1D.")
              i = i.view(1, -1) # Reshape the tensor to 2D for visualization
            img = imgs[i].permute(1, 2, 0).numpy()  # Change shape from (C, H, W) to (H, W, C)
          ax.imshow(img)
        ax.axis('off')  # Hide axes
      plt.show()
      plt.savefig(".")


class SurgeryStillsDataset(Dataset):
  ''' Custom dataset of frames from sampled surgery videos
  '''
  def __init__(self, path_inputs, path_labels, path_annots, transform=None, return_extra={"ids": False, "ts": False, "vids": False}, update_labels=False):
    '''
      Args:
        path_labels (str): path to csv with annotations
        path_inputs (str): path to folder with all images
        path_annots (str): path to raw annotations (to extract classes)
        transform (callable, optional): transform a sample
    '''
    self.path_inputs = path_inputs
    self.path_labels = path_labels
    self.path_annots = path_annots
    if update_labels:
      self.update_labels_csv()
    self.labels = self.get_labels(path_labels) # frame_id, video, frame_label
    # print(self.labels.index)
    # print(self.labels.index)
    self.transform = transform
    self.return_extra = return_extra
    # NOT WORKING
    # Filter labels to include only rows where the frame file exists
    # self.labels = self.filter_valid_frames(self.labels) # Useful when using short version of the dataset with the full labels.csv
    # print("self_labels\n", self.labels)

    self.idx_to_class = self.get_idx_to_class() # gets dict for getting class name from int value
    self.video_ids = self.labels["video_id"].unique()  # Extract unique video identifiers
    self.video_to_group = {video_id: idx for idx, video_id in enumerate(self.video_ids)}
    self.n_classes = len(self.idx_to_class.keys())
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    ''' Converts idx to a list if it is a tensor (useful for batch processing) '''
    # print(f"Sending an idx of type {type(idx)}")
    if torch.is_tensor(idx):
      idx = idx.tolist()
    elif isinstance(idx, np.ndarray):
      idx = idx.tolist()
    elif isinstance(idx, slice):
      idx = list(range(*idx.indices(len(self.labels)))) # Convert slice to list
      
    extra = []
    
    if isinstance(idx, list):  # For batch processing
      # print(self.labels.iloc[idx[0], 0].split('_')[0])
      path_imgs = [os.path.join(self.path_inputs, self.labels.loc[i, "video_id"], self.labels.loc[i, "frame"] + ".png") for i in idx]
      
      if self.return_extra["ids"]:
        extra.append([self.labels.loc[i, "frame"] for i in idx]) 
      if self.return_extra["ts"]:
        extra.append([[self.labels.loc[i, "frame"] for i in idx].split('_')[1] for i in idx]) 
      if self.return_extra["vids"]:
        extra.append([[self.labels.loc[i, "frame"] for i in idx].split('_')[0] for i in idx])

      imgs = [iio.imread(p) for p in path_imgs]
      labels = self.labels.loc[idx, "class"].tolist()
      if self.transform:
          imgs = [self.transform(img) for img in imgs]
      imgs = torch.stack(imgs)  # Stack list of tensors into a single tensor
      labels = torch.tensor(labels)  # Convert list of labels to tensor

    else:  # For single index (using same return variables for these have only one element)
      path_imgs = os.path.join(self.path_inputs, self.labels.loc[idx, "video_id"], self.labels.loc[idx, "frame"] + ".png")

      if self.return_extra["ids"]:
        extra.append(self.labels.loc[idx, "frame"]) 
      if self.return_extra["ts"]:
        extra.append(self.labels.loc[idx, "frame"].split('_')[1]) 
      if self.return_extra["vids"]:
        extra.append(self.labels.loc[idx, "frame"].split('_')[0])

      imgs = iio.imread(path_imgs)
      labels = self.labels.loc[idx, "class"]
      if self.transform:
          imgs = self.transform(imgs)
      labels = torch.tensor(labels)  # Single label as tensor

    return imgs, labels, *extra
  
  def get_labels(self, path_labels):
    df = pd.read_csv(path_labels)
    # print(df.columns)
    df.reset_index(drop=True, inplace=True) # Create a new integer index and drop the old index
    # (Optional) Create a new column for video_id based on the frame column
    df['video_id'] = df['frame'].str.split('_', expand=True)[0]
    # print(df.columns)
    # self.write_labels_csv(df, "deg.csv")
    return df
  
  def write_labels_csv(self, df, path_to):
    df.to_csv(path_to, index=False)

  def update_labels_csv(self):
    # Read the labels CSV
    df_original = pd.read_csv("./data/local/LDSS/labels.csv")

    # Create a set of all available images from the dataset folder
    samples_available = set()
    for video_folder in os.listdir(self.path_inputs):
      video_path = os.path.join(self.path_inputs, video_folder)
      if os.path.isdir(video_path):  # Check if it's a folder
        for sample in os.listdir(video_path):
          samples_available.add(os.path.splitext(sample)[0])
    # Filter the DataFrame to only include samples present in the dataset folder
    df_filtered = df_original[df_original["frame"].isin(samples_available)]
    df_filtered.to_csv(self.path_labels, index=False)
    print(f"New labels.csv saved to: {self.path_labels}")


  def detect_encoding(self, file_path):
    ''' Extract and return the file encoding'''
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

  def parse_class(self, protoclass):
    ''' Transform "End of (x) y" protoclass into (x, y) -> specific for this way o annotation '''
    for l in protoclass:
      if l.isdigit():
        return (l, protoclass[protoclass.find(l) + 3:])

  def get_idx_to_class(self):
    ''' Extract existing classes from the video annotations '''

    def get_sigla(string):
      # get the first letter of each word in a string in uppercase
      return "".join([word[0].upper() for word in string.split()])

    path_annot_random = os.path.join(self.path_annots, random.choice(os.listdir(self.path_annots))) # get random annotation to extract all classes (assuming they all have - they DON'T!!!)
    # print(path_annot_random)
    if os.path.exists(path_annot_random):
      encoding = self.detect_encoding(path_annot_random)
      with open(path_annot_random, 'r', encoding=encoding) as f:
        annot_random = json.load(f)

        protoclasses = [m["content"] for m in annot_random]
        idx_to_class = {int(self.parse_class(p)[0]): self.parse_class(p)[0] + get_sigla(self.parse_class(p)[1]) for p in protoclasses}
        idx_to_class[0] = "0ther"

      return idx_to_class

  def get_grouping(self, idx=None):
    ''' Returns a list of numeric labels corresponding to the video groups '''
    if idx is None:
        idx = slice(None)  # This is equivalent to selecting all rows
    elif isinstance(idx, tuple):
        idx = slice(*idx)  # Convert tuple to slice
    return [self.video_to_group[video_id] for video_id in self.labels["video_id"][idx]]
  
  """ DISABLED - NOT WORKING
  def filter_valid_frames(self, labels):
    ''' Filters the labels DataFrame to only include rows where the corresponding frame exists '''
    valid_labels = labels[labels["frame"].apply(self.is_valid_frame)]
    return valid_labels.reset_index(drop=True)

  def is_valid_frame(self, frame_id):
      ''' Checks if the frame exists in the dataset '''
      frame_path = os.path.join(self.path_inputs, frame_id.split('_')[0], f"{frame_id}.png")  # Adjust if necessary
      # if os.path.exists(frame_path): print("frame_path", frame_path)
      return os.path.exists(frame_path)
  """
  def idx_to_video(self, list_of_idxs):
    ''' Extract video IDs (check for overlap in sets) - specific to each labels.csv implementation
        List of idxs [[tr_i, va_i, te_i], ...] --> List of videos [[vidsx, vidsy, vidsz], ...]
    '''
    # Load the CSV and extract video_id
    # AFFECTed THE ORIGINAL
    # df = self.labels
    # df.drop(["frame", "class"], axis=1, inplace=True)  # Optional: Drop unnecessary columns if not needed
    
    list_out = []

    for idxs in list_of_idxs: # for each fold indexs
      list_vids = []
      for i, split in enumerate(idxs):  # for each split  (train, valid, test)
        videos_in_split = self.labels.loc[split, 'video_id'].unique()  # Get unique video_ids for this subset
        videos_in_split = [v[:4] for v in videos_in_split]
        # print(f"split {i}\n", videos_in_split, '\n')
        list_vids.append(videos_in_split)
      # Check for overlaps between splits (train, validation, test)
      train_videos, valid_videos, test_videos = list_vids
      
      # Asserts to check that no videos overlap between sets
      assert not any(set(train_videos) & set(valid_videos)), "Overlap between train and validation sets"
      assert not any(set(train_videos) & set(test_videos)), "Overlap between train and test sets"
      assert not any(set(valid_videos) & set(test_videos)), "Overlap between validation and test sets"
      
      list_out.append(list_vids)  # Store results for each fold

    if len(list_of_idxs) == 1:
      return list_out[0]
    # print(list_out)
    return list_out
  
  def get_class_weights(self):
    # print(self.labels.columns)
    cw = self.labels["class"].value_counts() / self.__len__()
    return torch.tensor(cw.values, dtype=torch.float32)

def get_dataset(path_dataset, datakey="local", transform=None, return_extra={"ids": False, "ts": False, "vids": False}, update_labels=False):
  if datakey == "local":
    path_inputs = os.path.join(path_dataset, "inputs")
    path_labels = os.path.join(path_dataset, "labels.csv")
    path_annots = os.path.join(path_dataset, "annotations")
    return SurgeryStillsDataset(path_inputs, path_labels, path_annots, transform, return_extra, update_labels)
  if datakey == "external":
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def splits(dataset, datakey, k_folds):
  ''' returns ( [])
  '''
  # Practice / Test Split
  list_of_idxs = []
  if datakey == "local":
    kf_outer = GroupKFold(n_splits=k_folds)
    groups_outer = dataset.get_grouping()
    kf_outer_split = kf_outer.split(np.arange(len(dataset)), groups=groups_outer)  # train/valid and test split - fold split
    for tv_idx, test_idx in kf_outer_split:
      # print('\n', tv_idx, test_idx, '\n')
      # tv_subset = torch.utils.data.Subset(dataset, tv_idx)
      groups_inner = dataset.get_grouping(tv_idx)
      # print(groups_inner)
      kf_inner = GroupKFold(n_splits=k_folds - 1)
      train_idx_local, valid_idx_local = next(kf_inner.split(tv_idx, groups=groups_inner))  # train and valid split - practice split
      # print("trainloc", train_idx_local, '\n', "validloc", valid_idx_local)
      train_idx = tv_idx[train_idx_local] # the inner splits idx don't match the outer directly
      valid_idx = tv_idx[valid_idx_local] # because idxs for some video were removed (need to map back)
      list_of_idxs.append([train_idx, valid_idx, test_idx])
      # print('\n', [train_idx, valid_idx, test_idx], '\n')

    videos_split = dataset.idx_to_video(list_of_idxs)
    # print(videos_split[0])
    # print(list_of_idxs[0])
    return tuple(zip(list_of_idxs, videos_split))

  # Train / Valid Split
  elif datakey == "external":
    indices = np.arange(len(dataset[0]))
    train_idx, valid_idx = train_test_split(indices, train_size = (k_folds - 1) / (k_folds), random_state=random_state)
    # print(train_idx, valid_idx)
    # print(dataset[1])
    list_of_idxs.append([train_idx, valid_idx, dataset[1]])
    return tuple(zip(list_of_idxs, [[["no"], ["groups"], ["here"]]]))

  # print(f"Train idxs: {train_idx}\nValid idxs: {valid_idx}\nTest idxs: {test_idx}")
  # return train_idx, valid_idx, test_idx

def load_data(dataset, batch_size, train_idx=None, valid_idx=None, test_idx=None, datakey="local"):
  if datakey == "local":
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    validloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    testloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
  elif datakey == "external":
    # test_idx actually carries the test part of the dataset when using CIFAR-10
    trainloader = DataLoader(dataset=dataset[0], batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    validloader = DataLoader(dataset=dataset[0], batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    testloader = DataLoader(dataset=test_idx, batch_size=batch_size, shuffle=True)
    # print(next(trainloader).shape)
  return trainloader, validloader, testloader

def load_features(features, layer, dataset, batch_size, train_idx=None, valid_idx=None, test_idx=None, return_extra=None):
  # batch size is "one video"
  loader = [[], [], []] 
  for i, idx in enumerate([train_idx, valid_idx, test_idx]):
    df = dataset.labels.iloc[idx].copy()
    # print(df.columns)
    df["timestamp"] = df["frame"].str.split('_', expand=True)[1].astype(int)  # Ensure numeric sorting    
    df = df.sort_values(by=["video_id", "timestamp"])
    # print(df)
    for vid, group in df.groupby("video_id"):
      try:
        # Accumulate features, labels and frame_nmes as tensors for each video
        features_tensor = torch.stack([features[frame["frame"]][layer] for _, frame in group.iterrows()])
        labels_tensor = torch.tensor(group["class"].values)
        frames_tensor = [frame["frame"] for _, frame in group.iterrows()]

        # Ensure the channel dimension is present
        # print(features_tensor.shape)
        features_tensor = features_tensor.permute(1, 0).unsqueeze(0)  # to [batch_size, features_size, n_frames]
        print(features_tensor.shape, labels_tensor.shape, len(frames_tensor))
        # Store tuple of tensors (features, labels, frame indices) for this video in loader[i]
        loader[i].append((features_tensor, labels_tensor, frames_tensor))
      except KeyError as e: 
        print(f"KeyError: {e}")
        break
      except ValueError as e:
        print(f"ValueError: {e}")
        break
      except Exception as e:
        print(f"Unexpected error: {e}")
        break
      # v = vid
  # print(loader[2][v][0])
  # assert 1 == 0

  return loader # trainloader, validloader, testloader
