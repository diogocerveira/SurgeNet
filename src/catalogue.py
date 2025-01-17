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
from torchvision.transforms import v2 as transforms
import sys
import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from . import utils

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
  def __init__(self, dataScope, path_samples, path_labels, path_annots, extradata, transform=None, update_labels=False):
    '''
      Args:
        path_labels (str): path to csv with annotations
        path_samples (str): path to folder with all images
        path_annots (str): path to raw annotations (to extract classes)
        transform (callable, optional): transform a sample
    '''
    self.dataScope = dataScope
    self.path_samples = path_samples
    self.path_labels = path_labels
    self.path_annots = path_annots
    if update_labels:
      self.update_labels_csv()
    self.labels = self._get_labels(path_labels) # frame_id, video, frame_label
    # print(self.labels.index)
    self.extradata = extradata
    # print(type(extradata))
    self.transform = transform
    # NOT WORKING
    # Filter labels to include only rows where the frame file exists
    # self.labels = self.filter_valid_frames(self.labels) # Useful when using short version of the dataset with the full labels.csv
    # print("self_labels\n", self.labels)

    self.videoNames = self.labels["videoId"].unique()  # Extract unique video identifiers
    self.videoToGroup = {videoId: idx for idx, videoId in enumerate(self.videoNames)}
    
    self.n_classes = len(self.labels["class"].unique())
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
      path_imgs = [os.path.join(self.path_samples, self.labels.loc[i, "videoId"], self.labels.loc[i, "frame"] + ".png") for i in idx]
      
      if self.extradata["id"]:
        extra.append([self.labels.loc[i, "frame"] for i in idx]) 
      if self.extradata["timestamp"]:
        extra.append([[self.labels.loc[i, "frame"] for i in idx].split('_')[1] for i in idx]) 
      if self.extradata["video"]:
        extra.append([[self.labels.loc[i, "frame"] for i in idx].split('_')[0] for i in idx])

      imgs = [iio.imread(p) for p in path_imgs]
      labels = self.labels.loc[idx, "class"].tolist()
      if self.transform:
          imgs = [self.transform(img) for img in imgs]
      imgs = torch.stack(imgs)  # Stack list of tensors into a single tensor
      labels = torch.tensor(labels)  # Convert list of labels to tensor

    else:  # For single index (using same return variables for these have only one element)
      path_imgs = os.path.join(self.path_samples, self.labels.loc[idx, "videoId"], self.labels.loc[idx, "frame"] + ".png")

      if self.extradata["id"]:
        extra.append(self.labels.loc[idx, "frame"]) 
      if self.extradata["timestamp"]:
        extra.append(self.labels.loc[idx, "frame"].split('_')[1]) 
      if self.extradata["video"]:
        extra.append(self.labels.loc[idx, "frame"].split('_')[0])

      imgs = iio.imread(path_imgs)
      labels = self.labels.loc[idx, "class"]
      if self.transform:
          imgs = self.transform(imgs)
      labels = torch.tensor(labels)  # Single label as tensor

    return imgs, labels, *extra
  
  def _get_labels(self, path_labels):
    try:
      print(path_labels)
      labels = pd.read_csv(path_labels)
      # print(df.columns)
      labels.reset_index(drop=True, inplace=True) # Create a new integer index and drop the old index
      # (Optional) Create a new column for videoId based on the frame column
      labels['videoId'] = labels['frame'].str.split('_', expand=True)[0]
      # print(df.columns)
      # self.write_labels_csv(df, "deg.csv")
      return labels
    except:
      raise ValueError("Invalid path or labels.csv file")
  
  def write_labels_csv(self, df, path_to):
    df.to_csv(path_to, index=False)

  def update_labels_csv(self):
    # Read the labels CSV
    df_original = pd.read_csv("./data/local/LDSS/labels.csv")

    # Create a set of all available images from the dataset folder
    samplesAvailable = set()
    for videoFolder in os.listdir(self.path_samples):
      path_video = os.path.join(self.path_samples, videoFolder)
      if os.path.isdir(path_video):  # Check if it's a folder
        for sample in os.listdir(path_video):
          samplesAvailable.add(os.path.splitext(sample)[0])
    # Filter the DataFrame to only include samples present in the dataset folder
    df_filtered = df_original[df_original["frame"].isin(samplesAvailable)]
    df_filtered.to_csv(self.path_labels, index=False)
    print(f"New labels.csv saved to: {self.path_labels}")
  """
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

  def get_labelToClass(self):
    ''' Extract existing classes from the video annotations '''

    def get_sigla(string):
      # get the first letter of each word in a string in uppercase
      return "".join([word[0].upper() for word in string.split()])

    path_randomAnnot = os.path.join(self.path_annots, random.choice(os.listdir(self.path_annots))) # get random annotation to extract all classes (assuming they all have - they DON'T!!!)
    # print(path_randomAnnot)
    if os.path.exists(path_randomAnnot):
      encoding = self.detect_encoding(path_randomAnnot)
      with open(path_randomAnnot, 'r', encoding=encoding) as f:
        randomAnnot = json.load(f)

        protoclasses = [m["content"] for m in randomAnnot]
        labelToClass = {int(self.parse_class(p)[0]): self.parse_class(p)[0] + get_sigla(self.parse_class(p)[1]) for p in protoclasses}
        labelToClass[0] = "0ther"

      return labelToClass
  """
  def get_grouping(self, idx=None):
    ''' Returns a list of numeric labels corresponding to the video groups '''
    if idx is None:
        idx = slice(None)  # This is equivalent to selecting all rows
    elif isinstance(idx, tuple):
        idx = slice(*idx)  # Convert tuple to slice
    return [self.videoToGroup[videoId] for videoId in self.labels["videoId"][idx]]
  
  """ DISABLED - NOT WORKING
  def filter_valid_frames(self, labels):
    ''' Filters the labels DataFrame to only include rows where the corresponding frame exists '''
    valid_labels = labels[labels["frame"].apply(self.is_valid_frame)]
    return valid_labels.reset_index(drop=True)

  def is_valid_frame(self, frame_id):
      ''' Checks if the frame exists in the dataset '''
      frame_path = os.path.join(self.path_samples, frame_id.split('_')[0], f"{frame_id}.png")  # Adjust if necessary
      # if os.path.exists(frame_path): print("frame_path", frame_path)
      return os.path.exists(frame_path)
  """
  def get_idxToVideo(self, listOfIdxs):
    ''' Extract video IDs (check for overlap in sets) - specific to each labels.csv implementation
        List of idxs [[tr_i, va_i, te_i], ...] --> List of videos [[vidsx, vidsy, vidsz], ...]
    '''
    # Load the CSV and extract videoId
    # AFFECTed THE ORIGINAL
    # df = self.labels
    # df.drop(["frame", "class"], axis=1, inplace=True)  # Optional: Drop unnecessary columns if not needed
    
    list_out = []

    for idxs in listOfIdxs: # for each fold indexs
      vidsList = []
      for i, split in enumerate(idxs):  # for each split  (train, valid, test)
        vidsInSplit = self.labels.loc[split, 'videoId'].unique()  # Get unique videoNames for this subset
        vidsInSplit = [v[:4] for v in vidsInSplit]
        # print(f"split {i}\n", vidsInSplit, '\n')
        vidsList.append(vidsInSplit)
      # Check for overlaps between splits (train, validation, test)
      train_videos, valid_videos, test_videos = vidsList
      
      # Asserts to check that no videos overlap between sets
      assert not any(set(train_videos) & set(valid_videos)), "Overlap between train and validation sets"
      assert not any(set(train_videos) & set(test_videos)), "Overlap between train and test sets"
      assert not any(set(valid_videos) & set(test_videos)), "Overlap between validation and test sets"
      
      list_out.append(vidsList)  # Store results for each fold

    if len(listOfIdxs) == 1:
      return list_out[0]
    # print(list_out)
    return list_out
  
  def get_classWeights(self):
    # print(self.labels.columns)
    cw = self.labels["class"].value_counts() / self.__len__()
    return torch.tensor(cw.values, dtype=torch.float32)
"""
def get_preprocessing(preprocessing):
  if preprocessing == "basic":
    transform = transforms.Compose([
      transforms.Resize((224, 224), antialias=True),
      transforms.ToImage(), # only for v2
      transforms.ToDtype(torch.float32, scale=True),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
      # transforms.CenterCrop(),
      # transforms.ToTensor(),
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
  return transform

def get_augmentation():
  return transforms.Compose([
    # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),   # (mean) (std) for each channel New = (Prev - mean) / stf
    transforms.RandomRotation((-90, 90)),
    transforms.ColorJitter(10, 2)
    ])
"""
class VideoFeaturesDataset(SurgeryStillsDataset):
  def __init__(self, parent, path_features, featureLayer):
    attrToInherit = ["dataScope", "labels", "metadata", "labelToClass", "n_classes"]
    for attr in attrToInherit:
      if hasattr(parent, attr):
        setattr(self, attr, getattr(parent, attr))
    self.featuresDict = torch.load(path_features)
    



class Cataloguer():
  ''' Each Cataloguer organises/handles 1 dataset (local - LDSS - or external - CIFAR10)
      In - digital rawdata from pc paths
      Action - sample, train, eval
      Pipeline: [sample] -> [label] -> [preprocess] -> [batch]
      Out - split and batched dataset to a Teacher
  '''
  def __init__(self, path_dataset, DATA):
   
    self.path_samples = os.path.join(path_dataset, "samples")
    self.path_labels = os.path.join(path_dataset, "labels.csv")
    self.path_annots = os.path.join(path_dataset, "annotations")
    self.dataset = self._get_dataset(DATA["datasetId"], DATA["preprocessing"],
        DATA["return_extradata"], DATA["update_labels"])

    print(f"[{DATA['preprocessing']}] Preprocessing")
   
    self.labelToClass = self._get_labelToClass()   # gets dict for getting class name from int value
    self.n_classes = self.dataset.n_classes
    self.datasetSize = self._get_datasetSize()
    self.classWeights = self._get_classWeights()
    print(self.classWeights)

    self.features = None
    self.featureLayer = None

  def update_features(self, features, featureLayer):
    self.features = features
    self.featureLayer = featureLayer

  def split(self, k_folds, dataset=None):
    ''' returns ([])
    '''
    dataset = self.dataset
    # Practice / Test Split
    listOfIdxs = []
    if dataset.dataScope == "local":
      outerKF = GroupKFold(n_splits=k_folds)
      outerGroups = dataset.get_grouping()
      outerKFSplit = outerKF.split(np.arange(len(dataset)), groups=outerGroups)  # train/valid and test split - fold split
      for tv_idx, test_idx in outerKFSplit:
        # print('\n', tv_idx, test_idx, '\n')
        # tv_subset = torch.utils.data.Subset(dataset, tv_idx)
        innerGroups = dataset.get_grouping(tv_idx)
        # print(innerGroups)
        innerKF = GroupKFold(n_splits=k_folds - 1)
        train_localIdxs, valid_localIdxs = next(innerKF.split(tv_idx, groups=innerGroups))  # train and valid split - practice split
        # print("trainloc", train_localIdxs, '\n', "validloc", valid_localIdxs)
        train_idx = tv_idx[train_localIdxs] # the inner splits idx don't match the outer directly
        valid_idx = tv_idx[valid_localIdxs] # because idxs for some video were removed (need to map back)
        listOfIdxs.append([train_idx, valid_idx, test_idx])
        # print('\n', [train_idx, valid_idx, test_idx], '\n')

      videosSplit = dataset.get_idxToVideo(listOfIdxs)
      # print(videosSplit[0])
      # print(listOfIdxs[0])
      return tuple(zip(listOfIdxs, videosSplit))

    # Train / Valid Split
    elif dataset.dataScope == "external":
      indices = np.arange(len(dataset[0]))
      train_idx, valid_idx = train_test_split(indices, train_size = (k_folds - 1) / (k_folds), random_state=randomState)
      # print(train_idx, valid_idx)
      # print(dataset[1])
      listOfIdxs.append([train_idx, valid_idx, dataset[1]])
      return tuple(zip(listOfIdxs, [[["no"], ["groups"], ["here"]]]))

    # print(f"Train idxs: {train_idx}\nValid idxs: {valid_idx}\nTest idxs: {test_idx}")
    # return train_idx, valid_idx, test_idx

  
  def batch(self, batchSize, batchMode, train_idx=None, valid_idx=None, test_idx=None):
    ''' helper for loading data, features or stopping when fx_mode == 'export'
    '''
    if batchMode == "images":
      return self._batch_images(batchSize, train_idx, valid_idx, test_idx)
    elif batchMode == "features":
      return self._batch_features(train_idx, valid_idx, test_idx)
    else:
      raise ValueError("Invalid Batch Mode!")
  
  def _batch_images(self, batchSize, train_idx=None, valid_idx=None, test_idx=None):
    loaders = {}
    if self.dataset.dataScope == "local":
      if train_idx:
        loaders["trainloader"] = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
      if valid_idx:
        loaders["validloader"] = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
      if test_idx.any():
        loaders["testloader"] = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
    elif self.dataset.dataScope == "external":
      # test_idx actually carries the test part of the dataset when using CIFAR-10
      if train_idx:
        loaders["trainloader"] = DataLoader(dataset=self.dataset[0], batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
      if valid_idx:
        loaders["validloader"] = DataLoader(dataset=self.dataset[0], batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
      if test_idx:
        loaders["testloader"] = DataLoader(dataset=test_idx, batch_size=batchSize, shuffle=True)
      # print(next(trainloader).shape)
    return loaders
  
  def _collate_features(batch):
    # Pad features to the same length
    maxLen = max(feat.size(-1) for feat, _, _ in batch)
    paddedFeatures = [torch.nn.functional.pad(feat, (0, maxLen - feat.size(-1))) for feat, _, _ in batch]
    features = torch.stack(paddedFeatures)
    labels = [label for _, label, _ in batch]
    frameNames = [frames for _, _, frames in batch]
    return features, labels, frameNames
  def _batch_features2(self, batchSize, train_idx=None, valid_idx=None, test_idx=None):
    if train_idx.any():
      trainloader = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    if valid_idx.any():
      validloader = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))
    if test_idx.any():
      testloader = DataLoader(dataset=self.dataset, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
    return trainloader, validloader, testloader
  def _batch_features(self, train_idx=None, valid_idx=None, test_idx=None):
    # batch size is "one video"
    loader = [[], [], []] 
    for i, idx in enumerate([train_idx, valid_idx, test_idx]):
      df = self.dataset.labels.iloc[idx].copy()
      # print(df.columns)
      df["timestamp"] = df["frame"].str.split('_', expand=True)[1].astype(int)  # Ensure numeric sorting    
      df = df.sort_values(by=["videoId", "timestamp"])
      # print(df)
      for vid, group in df.groupby("videoId"):
        try:
          # print(self.features[list(self.features.keys())[-1]])
          # Accumulate features, labels and frame_nmes as tensors for each video
          feats = torch.stack([self.features[frame["frame"]][self.featureLayer] for _, frame in group.iterrows()])
          labels = torch.tensor(group["class"].values)
          frameNames = [frame["frame"] for _, frame in group.iterrows()]

          # Ensure the channel dimension is present
          # print(features.shape)
          feats = feats.permute(1, 0).unsqueeze(0)  # to [batchSize, features_size, n_frames]
          # print(feats.shape, labels.shape, len(frameNames))
          # Store tuple of tensors (features, labels, frame indices) for this video in loader[i]
          loader[i].append((feats, labels, frameNames))
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

  def _get_dataset(self, datasetId, preprocessing=None, extradata={}, update_labels=False):
    ''' Get the Dataset objects according to the dataScope'''
    transform = self._get_preprocessing(preprocessing);
    dataScope = datasetId.split('-')[1]
    if dataScope == "local":
      dataset = SurgeryStillsDataset(dataScope, self.path_samples, self.path_labels,
          self.path_annots, extradata, transform, update_labels)    
    elif dataScope == "external":
      train_dataset = datasets.CIFAR10(root=f"./data/{dataScope}", train=True, download=True, transform=preprocessing)
      test_dataset = datasets.CIFAR10(root=f"./data/{dataScope}", train=False, download=True, transform=preprocessing)
      dataset = (train_dataset, test_dataset)
    else:
      raise ValueError("Invalid dataScope!")
    return dataset
  
  def _get_datasetSize(self):
    if self.dataset.dataScope == "local":
      return len(self.dataset)
    elif self.dataset.dataScope == "external":
      return len(self.dataset[0]) + len(self.dataset[1])
    
  def _get_classWeights(self):
    ''' Get class distribution form the Dataset object'''
    # print(self.labels.columns)
    cw = self.dataset.labels["class"].value_counts() / self.datasetSize
    return torch.tensor(cw.values, dtype=torch.float32)
  
  def _get_labelToClass(self):
    ''' Get a map from labels (int) to classes (string) '''
    if self.dataset.dataScope == "local":
      def get_sigla(string):
        # get the first letter of each word in a string in uppercase
        return "".join([word[0].upper() for word in string.split()])
      def detect_encoding(file_path):
        ''' Extract and return the file encoding'''
        with open(file_path, 'rb') as f:
          raw_data = f.read()
          result = chardet.detect(raw_data)
          return result['encoding']
      def parse_class(protoclass):
        ''' Transform "End of (x) y" protoclass into (x, y) -> specific for this way o annotation '''
        for l in protoclass:
          if l.isdigit():
            return (l, protoclass[protoclass.find(l) + 3:])

      path_randomAnnot = os.path.join(self.path_annots, random.choice(os.listdir(self.path_annots))) # get random annotation to extract all classes (assuming they all have - they DON'T!!!)
      # print(path_randomAnnot)
      if os.path.exists(path_randomAnnot):
        encoding = detect_encoding(path_randomAnnot)
        with open(path_randomAnnot, 'r', encoding=encoding) as f:
          randomAnnot = json.load(f)
          protoclasses = [m["content"] for m in randomAnnot]
          labelToClass = {int(parse_class(p)[0]): parse_class(p)[0] + get_sigla(parse_class(p)[1]) for p in protoclasses}
          labelToClass[0] = "0ther"
        return labelToClass
      
    elif self.dataset.dataScope == "external":
      return {i: c for c, i in self.dataset[1].class_to_idx.items()};
    else:
      raise ValueError("Invalid dataScope!")
  
  def _get_preprocessing(self, preprocessing):
    ''' Get torch group of tranforms directly applied to torch Dataset object'''
    if preprocessing == "basic":
      transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToImage(), # only for v2
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        # transforms.CenterCrop(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif preprocessing == "aug":
      transform = transforms.Compose([
      # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
      transforms.Resize((32, 32)),
      transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),   # (mean) (std) for each channel New = (Prev - mean) / stf
      transforms.RandomRotation((-90, 90)),
      transforms.ColorJitter(10, 2)
      ])
    else:
      raise ValueError("Invalid preprocessing key")
    return transform