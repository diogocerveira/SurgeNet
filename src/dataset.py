import os
from torch.utils.data import Dataset  # for the custom dataset (derived class)
import pandas as pd
import json
import chardet
import random
# import imageio.v3 as iio
import torch
from torchvision import datasets
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import time
from torchvision.transforms import v2 as transforms
import shutil
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import csv
import cv2
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SampledVideosDataset(Dataset):
  ''' Custom dataset of frames from sampled surgery videos
  '''
  def __init__(self, DATA_SCOPE, labelType, path_samples, path_labels, transform):
    '''
      Args:
        path_labels (str): path to csv with annotations
        path_samples (str): path to folder with all images
        path_annots (str): path to raw annotations (to extract classesInt)
        transform (callable, optional): transform a sample
    '''
    self.DATA_SCOPE = DATA_SCOPE
    self.labelType = labelType
    self.path_samples = path_samples
    self.datatype = "png"
    self.path_labels = path_labels
    self.labels = self._get_labels(path_labels) # frameId, videoId, frame_label
    # print(self.labels.index)
    self.transform = transform
    # NOT WORKING
    # Filter labels to include only rows where the frame file exists
    # self.labels = self.filter_valid_frames(self.labels) # Useful when using short version of the dataset with the full labels.csv
    # print("self_labels\n", self.labels)
    self.path_spaceFeatures = None

    self.videoNames = self.labels["videoId"].unique()  # Extract unique video identifiers
    self.videoToGroup = {videoId: idx for idx, videoId in enumerate(self.videoNames)}
    
    self.labelToClassMap = self.get_labelToClassMap(self.labels)
    self.n_classes = len(self.labelToClassMap)
    self.classWeights = self.get_labelWeights(self.labels)
    print(f"Class weights: {self.classWeights}\n")
     # check if the class distribution is balanced
    if all([self.classWeights[i] == self.classWeights[i + 1] for i in range(len(self.classWeights) - 1)]):
      self.BALANCED = True
    else:
      self.BALANCED = False

  def __len__(self):
    length = 0
    # count number of files (images) in the samples folder directories (sampled video folders)
    for video in self.videoNames:
      # print(video)
      length += len([vid for vid in os.listdir(os.path.join(self.path_samples, video)) if not vid.startswith(('.', '_')) and not os.path.isdir(os.path.join(self.path_samples, video, vid))])
    if length != len(self.labels):
      raise ValueError(f"Number of files in samples folder ({length}) and labels.csv ({len(self.labels)}) do not match!") 
    return length
  
  def __getitem__(self, idx):
    """Retrieve a single item from the dataset based on an index"""
    path_img = os.path.join(self.path_samples, self.labels.loc[idx, "videoId"],
      self.labels.loc[idx, "frameId"] + '.' + self.datatype)
    # Read images and labels
    img = self.transform(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB))
    label = self.labels.loc[idx, "frameLabels"]
    print("WTF")
    # print(f"img: {img.shape}, label: {label}, idx: {torch.tensor(idx)}")  # WTFFFF does not print
    return (img, label, torch.tensor(idx))
  def __getitems__(self, idxs):
    """Retrieve items from the dataset based on index or list of indices."""
    # Convert dset idxs to list
    if isinstance(idxs, slice):  # Convert slice to list
      idxs = list(range(*idxs.indices(len(self.labels))))
    elif torch.is_tensor(idxs) or isinstance(idxs, np.ndarray):
      idxs = torch.tolist()
    # print(idxs[:4])
    # each idx is a int tensor
    path_imgs = [os.path.join(self.path_samples, self.labels.loc[i, "videoId"],
        self.labels.loc[i, "frameId"] + '.' + self.datatype) for i in idxs]
    # Read images and labels
    # imgs = [iio.imread(p) for p in path_imgs]
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in path_imgs]
    labels = self.labels.loc[idxs, "frameLabels"].tolist()
    # convert natural language multi label to numeric single
    labels = [[self.labelToClassMap[single] for single in multi.split(',')] for multi in labels]
    if self.transform:
      imgs = [self.transform(img) for img in imgs]
    else:
      imgs = [torch.tensor(img) for img in imgs]
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels).view(-1) if self.labelType.split('-')[0] == "single" else torch.tensor(labels)
    idxs = torch.tensor(idxs)
    print("idxs: ", idxs[:5])
    print("labels: ", labels[:5], '\n')
    # print(f"imgs: {imgs.shape}, labels: {labels}, idxs: {idxs}")
    return list(zip(imgs, labels, idxs))
  
  def _get_labels(self, path_labels):
    try:
      # print(path_labels, "\n\n", flush=True)
      labels = pd.read_csv(path_labels)
      # print(df.columns)
      labels.reset_index(drop=True, inplace=True) # Create a new integer index and drop the old index
      # (Optional) Create a new column for videoId based on the frame column
      labels['videoId'] = labels['frameId'].str.split('_', expand=True)[0]
      # print(df.columns)
      # self._write_labels_csv(df, "deg.csv")
      return labels
    except:
      raise ValueError("\nInvalid path_labels/labels.csv file (might suggest inexistent or a problematic dataset)")
  def _update_labels(self, path_labels):
    self.labels = self._get_labels(path_labels)
  
  def _write_labels_csv(self, df, path_to):
    df.to_csv(path_to, index=False)

  def _get_grouping(self, idx=None):
    ''' Returns a list of numeric labels corresponding to the video groups '''
    if idx is None:
        idx = slice(None)  # This is equivalent to selecting all rows
    elif isinstance(idx, tuple):
        idx = slice(*idx)  # Convert tuple to slice
    return [self.videoToGroup[videoId] for videoId in self.labels["videoId"][idx]]
  
  def _get_idxToVideo(self, listOfIdxs):
    ''' Extract video IDs (check for overlap in sets) - specific to each labels.csv implementation
        List of idxs [[tr_i, va_i, te_i], ...] --> List of videos [[vidsx, vidsy, vidsz], ...]
    '''
    # Load the CSV and extract videoId
    # AFFECTed THE ORIGINAL
    # df = self.labels
    # df.drop(["frameId", "frameLabels"], axis=1, inplace=True)  # Optional: Drop unnecessary columns if not needed
    
    list_out = []

    for idxs in listOfIdxs: # for each fold indexs
      vidsList = []
      for i, split in enumerate(idxs):  # for each split  (train, valid, test)
        vidsInSplit = self.labels.loc[split, 'videoId'].unique()  # Get unique videoNames for this subset
        vidsInSplit = [video[:4] for video in vidsInSplit]
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
      
  def get_labelWeights(self, labels):
    ''' Get class distribution form the Dataset object'''
    labels_counts = Counter()
    for multi in labels["frameLabels"]:
      single = [l.strip() for l in multi.split(',')] # comma separated labels
      labels_counts.update(single)
    cw = pd.Series(labels_counts) / self.__len__()  # normalize by the number of frames

    return torch.tensor(cw.values, dtype=torch.float32)
  
  def get_labelToClassMap(self, labels):
    ''' Convert to list of natural language labels into list of int number labels, saving a map class->label in classToLabelMap dict map '''
    uniqueLabels = set()
    # print(labels[:5])
    for multi in set(labels["frameLabels"]):
      for single in multi.split(','):
        uniqueLabels.add(single)
    labelToClassMap = {l:i for i, l in enumerate(sorted(list(uniqueLabels)))}
    classes = [tuple([labelToClassMap[single] for single in multi.split(',')]) for multi in labels["frameLabels"]]
    # print(classes[:5])
    self.classToLabelMap = {i: l for i, l in enumerate(sorted(list(uniqueLabels)))}
    print(self.classToLabelMap)
    # print(classes[:5])
    return labelToClassMap

class Cataloguer():
  ''' Each Cataloguer organises/handles 1 dataset (local - LDSS - or external - CIFAR10)
      In - digital rawdata from pc paths
      Action - sample, train, eval
      Pipeline: [sample] -> [label] -> [preprocess] -> [batch]
      Out - split and batched dataset to a Teacher
  '''
  def __init__(self, DATA, path_annots):
    self.path_annots = path_annots
    self.preprocessingType = DATA["preprocessing"]
    self.labelType = DATA["labelType"]
    self.predicateLabelsPossible = DATA["predicateLabels"]
    self.objectLabelsPossible = DATA["objectLabels"]
    # updated when dataset is built
    self.features = None  # Default dict with for each frameId: {layerX: tensor, layerY: tensor, ...}

  def sample(self, path_rawVideos, path_to, samplerate=1, sampleFormat="png", processing="LDSS", filter_annotated=True):
    ''' Sample rawdata videos into image frames while saving the annotations in its splitset csv''' 
    # Generator function to yield frames and their corresponding data
    path_videos = os.path.join(path_rawVideos, "videos")
    path_annots = os.path.join(path_rawVideos, "annotations")
    availableVideos = sorted([vid for vid in os.listdir(path_videos) if not vid.startswith(('.', '_')) and not os.path.isdir(os.path.join(path_videos, vid))])
    print(f"\n{os.path.splitext(path_videos)[0]} for {len(availableVideos)} videos!")
    print(f"Sample rate of {samplerate} fps")
    skip = 0
    for videoId in availableVideos:
      if skip > 0:
        skip -= 1
        print(videoId)
        continue
      #videoId = os.path.splitext(videoId)[0]  # remove extension
      path_video = os.path.join(path_videos, videoId)
      #metadata = iio.immeta(path_video, plugin="pyav")  
      #fps = metadata["fps"]
      #nframes = int(metadata["duration"] * fps)
      #print(f"\niio - Sampling {videoId} ({nframes} frames at {fps} fps), period of {int(fps / samplerate)}")
      
      # print(f"FPS: {fps}, Nframes: {nframes}, Period: {period}")
      if filter_annotated:
        path_annot = os.path.join(path_annots, f"{os.path.splitext(videoId)[0]}.json")
        if not os.path.exists(path_annot):
          continue
      try:
        videoId = videoId[:4]  # keep only 1st 4 chars of the video name
      except:
        videoId = os.path.splitext(videoId)[0]  # if smaller name only remove ext
      shutil.copy(path_annot, os.path.join(self.path_annots, f"{videoId}.json"))
      path_frames = os.path.join(path_to, videoId)
      os.makedirs(path_frames, exist_ok=True)
      
      start_time = time.time()
      cap = cv2.VideoCapture(path_video)  # Initialize video capture object
      fps = cap.get(cv2.CAP_PROP_FPS)
      nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      period = round(fps / samplerate)
      print(f"\ncv2 - Sampling {videoId} ({nframes} frames at {fps} fps), period of {period} frames")
      # Get the first frame to determine the size
      ret, firstFrame = cap.read()
      if ret:
        factor = 1 / 4
        height, width = firstFrame.shape[:2]  # Get the original dimensions of the frame
        newWidth = int(width * factor)
        newHeight = int(height * factor)
        print(f"Pre-resizing video {videoId} to {newWidth}x{newHeight} (factor of {factor})")
      for i in range(0, nframes, period):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Set the current frame position
        ret, frameImg = cap.read()
        if ret:
          frameId = f"{videoId}_{int(i / fps * 1000)}"  # remember basename is the file name itself / converting index to its timestamp (ms)
          path_frame = os.path.join(path_frames, f"{frameId}.{sampleFormat}")
          resizedFrameImg = cv2.resize(frameImg, (newWidth, newHeight))
          cv2.imwrite(path_frame, resizedFrameImg)
      cap.release()  # Release the video capture object after processing
      print(f"Frame writing took: {time.time() - start_time} sec")
      print("B. Sampling finished!\n")
  
  def resize_frames(self, path_from, path_to, dim_to=(480, 270), change=0.25):
    # Ensure the change percentage is between 0 and 100
    if change < 0 or change > 100:
        raise ValueError("change percentage must be between 0 and 100")
    
    for videoFolder in os.listdir(path_from):
      print(f"Resizing video {videoFolder}\n")
      path_from_video = os.path.join(path_from, videoFolder)
      if os.path.isdir(path_from_video):  # Process only directories
        path_to_video = os.path.join(path_to, videoFolder)  # Make a corresponding output directory for this subfolder
        os.makedirs(path_to_video, exist_ok=True)
        for frame in os.listdir(path_from_video):
          path_from_frame = os.path.join(path_from_video, frame)
          if path_from_frame.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
              img = cv2.imread(path_from_frame)
              if img is None:
                print(f"Error reading {path_from_frame}")
                continue
              if not dim_to:
                height, width = img.shape[:2]
                new_width = int(width * (change / 100))
                new_height = int(height * (change / 100))
                dim_to = (new_width, new_height)
              resizedImg = cv2.resize(img, dim_to)
              path_to_frame = os.path.join(path_to_video, frame)
              cv2.imwrite(path_to_frame, resizedImg)
            except Exception as e:
              print(f"Error processing {path_from_frame}: {e}")
      print("C. Resizing finished!\n")
  
  def label(self, path_from, path_to, labelType):
    ''' folders of sampled videos -> labels.csv
    '''
    totalFrameIds = []
    totalFrameLabels = []
    for videoId in os.listdir(path_from): # sampled videos (folder of frames)
      path_annot = os.path.join(self.path_annots, f"{videoId}.json")
      path_video = os.path.join(path_from, videoId)
      if os.path.exists(path_annot):  # if video is annotated (to make a better check)
        phaseDelimiters = self._get_phaseDelimiters(path_annot)

        if labelType.split('-')[1] == "phase":
          get_frameLabels = self._impl_label_phase
        elif labelType.split('-')[1] == "time-to-next-phase":
          get_frameLabels = self._impl_label_timeToNextPhase
        # print("phaseDelimiters: ", phaseDelimiters)
        # get framesId from video folder that are not directories and do not start with . or _
        framesId = [os.path.splitext(frame)[0] for frame in os.listdir(path_video) if not frame.startswith(('.', '_')) and not os.path.isdir(os.path.join(path_video, frame))]
        totalFrameIds.extend(framesId)
        
        framesTimestamps = [int(frameId.split('_')[1]) for frameId in framesId]  # convert to int
        frameLabels = get_frameLabels(framesTimestamps, phaseDelimiters)  # assume natural language
        # print(frameLabels)
        totalFrameLabels.extend(frameLabels)
        # print(len(totalFrameClasses), len(totalFrameIds))
      elif videoId.startswith(('.', '_')):  # hidden file are disregarded
        print(f"   Skipping hidden file {videoId}\n")
        continue
      else:
        print(f"   Warning: Skipping annotation for {videoId} ({path_annot} not found)")

    # Prepare CSV file
    with open(path_to, 'w', newline='') as file_csv:  # newline='' maximizes compatibility with other os
      fieldNames = ['frameId', 'frameLabels']
      writer = csv.DictWriter(file_csv, fieldnames=fieldNames)
      if os.stat(path_to).st_size == 0: # Check if the file is empty to write the header
        writer.writeheader()
      # Batch write frames and annotations
      for frameId, frameLabel in zip(totalFrameIds, totalFrameLabels):
        writer.writerow({'frameId': frameId, 'frameLabels': frameLabel})

  def _get_phaseDelimiters(self, path_annot):
    ''' extract annotated video frames from annotation file as (timestamp, label))
    '''
    with open(path_annot, 'r', encoding=self._detect_encoding(path_annot)) as f:
      annot = json.load(f)  # parsed into list of dicts=annotations
      timestamps = [a["playback_timestamp"] for a in annot]
      protolabels = [a["content"] for a in annot]
      labels = self._parse_protolabels(protolabels)
      # labels should not have Other class, these are not annotated
    return list(zip(timestamps, labels)) # ordered!
  
  def _detect_encoding(self, path_file):
    ''' Extract and return annotation file encoding'''
    with open(path_file, 'rb') as f:
      rawData = f.read()
      result = chardet.detect(rawData)
      return result['encoding']
    
  def _parse_protolabels(self, protolabels):
    ''' (list) protolabels -> label
        e.g. ["End of (1) yellow xox and action", "Beg of (2) swallow"] into ["Y,X,A", "S, OtherObject"]
        # Defines type of labelling as single class (e.g. "Y-X-Z") or multi class (e.g. "Y,X,Z")
    '''
    labels = []
    for proto in protolabels:
      found = []
      
      for label in self.predicateLabelsPossible:
        if label.lower() in proto.lower():
          found.append(self._get_sigla(label))
          break  # assuming only 1 predicate per label
          # print(label)
      else:  # if no predicate found, append "Other"
        found.append("0therPred")  # sort to have a consistent order
      for label in self.objectLabelsPossible:
        if label.lower() in proto.lower():
          found.append(self._get_sigla(label))
          break  # assuming only 1 object per label
          # print(label)
      else:
        found.append("0TherObj")
      
      # found = sorted(found)  # sort the labels to have a consistent order
      if self.labelType.split('-')[0] == "single":
        found = '-'.join(found) # labels as strings like A-B-C
      elif self.labelType.split('-')[0] == "multi":
        found = ','.join(found) # labels as strings like A,B,C
      else:
        raise ValueError("Invalid labelType domain!")
      labels.append(found)
      # print(found)
    return labels

  def _get_sigla(self, text):
    # get the first letter of each word in a string in uppercase
    return ''.join([word[0].upper() + word[1:3] for word in text.split()])
  
  def _impl_label_phase(self, framesTimestamps, phaseDelimiters):
    # phaseDelimiter is of type (timestamp, label)
    framesLabel = []
    # print(phaseDelimiters)
    # print(framesTimestamps)
    # print(list(range(0, len(phaseDelimiters), 2)))
    for ts in framesTimestamps:
      for i in range(0, len(phaseDelimiters), 2): # step of 2 because boundary annotated frames are in pairs (beginning/resume+end/pause)
        if (ts >= phaseDelimiters[i][0]) and (ts <= phaseDelimiters[i + 1][0]):
          framesLabel.append(phaseDelimiters[i][1])
          # print("ts: ", ts)
          # print(phaseDelimiters[i][0], "  ", phaseDelimiters[i + 1][0], " : ", phaseDelimiters[i][1])
          break
      else: # appending anything if not a labelled phase
        if self.labelType.split('-')[0] == "single":
          otherLabel = "0therPred-0TherObj"  # if no phase, append both Other labels
        elif self.labelType.split('-')[0] == "multi":
          otherLabel = "0therPred,0TherObj"
        else:
          raise ValueError("Invalid labelType domain!")
        framesLabel.append(otherLabel)  # if no phase, append both Other labels

      # print("label(s): ", phaseDelimiters[i][1], '\n')
    return framesLabel
  
  def _get_preprocessing(self, preprocessingType):
    ''' Get torch group of tranforms directly applied to torch Dataset object'''
    if preprocessingType == "basic":
      transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToImage(), # only for v2
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.CenterCrop(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif preprocessingType == "aug":
      transform = transforms.Compose([
      # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
      transforms.Resize((224, 224), antialias=True),
      transforms.ToImage(), # only for v2
      transforms.ToDtype(torch.float32, scale=True),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # (mean) (std) for each channel New = (Prev - mean) / stf
      transforms.RandomRotation((-90, 90)),
      transforms.ColorJitter(10, 2)
      ])
    elif not preprocessingType:
      # make identity transform
      transform = transforms.Compose([
        transforms.ToImage(), # only for v2
        transforms.ToDtype(torch.float32, scale=True),
      ])
    else:
      raise ValueError("Invalid preprocessingType key")
    return transform
  
  def build_dataset(self, DATA_SCOPE, path_samples=None, path_labels=None):
    ''' Get the Dataset objects according to the DATA_SCOPE'''
    transform = self._get_preprocessing(self.preprocessingType)

    if DATA_SCOPE == "local":
      dataset = SampledVideosDataset(DATA_SCOPE, self.labelType, path_samples, path_labels, transform)    
    elif DATA_SCOPE == "external":
      train_dataset = datasets.CIFAR10(root=f"./data/{DATA_SCOPE}", train=True, download=True, transform=transform)
      test_dataset = datasets.CIFAR10(root=f"./data/{DATA_SCOPE}", train=False, download=True, transform=transform)
      dataset = (train_dataset, test_dataset)
    else:
      raise ValueError("Invalid DATA_SCOPE!")
    return dataset
  
  def split(self, k_folds, dataset):
    ''' returns ([])
    '''
    # Practice / Test Split
    listOfIdxs = []
    try:
      if dataset.DATA_SCOPE == "local":
        outerKF = GroupKFold(n_splits=k_folds)
        outerGroups = dataset._get_grouping()
        outerKFSplit = outerKF.split(np.arange(len(dataset)), groups=outerGroups)  # train/valid and test split - fold split
        for tv_idx, test_idxs in outerKFSplit:
          # print('\n', tv_idx, test_idxs, '\n')
          # tv_subset = torch.utils.data.Subset(dataset, tv_idx)
          innerGroups = dataset._get_grouping(tv_idx)
          # print(innerGroups)
          innerKF = GroupKFold(n_splits=k_folds - 1)
          train_localIdxs, valid_localIdxs = next(innerKF.split(tv_idx, groups=innerGroups))  # train and valid split - practice split
          # print("trainloc", train_localIdxs, '\n', "validloc", valid_localIdxs)
          train_idxs = tv_idx[train_localIdxs] # the inner splits idx don't match the outer directly
          valid_idxs = tv_idx[valid_localIdxs] # because idxs for some video were removed (need to map back)
          listOfIdxs.append([train_idxs, valid_idxs, test_idxs])
          # print('\n', [train_idxs, valid_idxs, test_idxs], '\n')

        videosSplit = dataset._get_idxToVideo(listOfIdxs)
        # print(videosSplit[0])
        # print(listOfIdxs[0])
        return tuple(zip(listOfIdxs, videosSplit))
    except:
      print("Error in split() for local dataset, assuming external dataset split instead")
    # Train / Valid Split
      indices = np.arange(len(dataset[0]))
      train_idxs, valid_idxs = train_test_split(indices, train_size = (k_folds - 1) / (k_folds), random_state=53)
      # print(train_idxs, valid_idxs)
      # print(dataset[1])
      listOfIdxs.append([train_idxs, valid_idxs, dataset[1]])
      return tuple(zip(listOfIdxs, [[["no"], ["groups"], ["here"]]]))

    # print(f"Train idxs: {train_idxs}\nValid idxs: {valid_idxs}\nTest idxs: {test_idxs}")
    # return train_idxs, valid_idxs, test_idxs

  def batch(self, dset, inputType, size_batch, size_clip=30, train_idxs=None, valid_idxs=None, test_idxs=None):
    ''' helper for loading data, features or stopping when fx_mode == 'export'
    '''
    # print(inputType)
    if inputType == "images":
      return self._batch_images(dset, size_batch, train_idxs, valid_idxs, test_idxs)
    elif inputType == "fmaps":
      return self._batch_clips(dset, size_batch, size_clip, train_idxs, valid_idxs, test_idxs)
    else:
      raise ValueError("Invalid Batch Mode!")
    
  def _batch_images(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    loaders = {}
    try:
      if dset.DATA_SCOPE == "local":
        if np.any(train_idxs):
          loaders["trainloader"] = DataLoader(dataset=dset, batch_size=size_batch, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(train_idxs))
        if np.any(valid_idxs):
          loaders["validloader"] = DataLoader(dataset=dset, batch_size=size_batch, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(valid_idxs))
        if np.any(test_idxs):
          loaders["testloader"] = DataLoader(dataset=dset, batch_size=size_batch, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(test_idxs))
    except:
      print("Error in batch() for local dataset, assuming external dataset batch instead")
      # test_idxs actually carries the test part of the dataset when using CIFAR-10
      if np.any(train_idxs):
        loaders["trainloader"] = DataLoader(dataset=dset[0], batch_size=size_batch, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(train_idxs))
      if np.any(valid_idxs):
        loaders["validloader"] = DataLoader(dataset=dset[0], batch_size=size_batch, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(valid_idxs))
      if np.any(test_idxs):
        loaders["testloader"] = DataLoader(dataset=test_idxs, batch_size=size_batch, num_workers=2, shuffle=True)
      # print(next(trainloader).shape)
    return loaders.values()

  def _batch_clips(self, dset, size_batch, size_clip, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(dset.path_spaceFeatures), f"No exported space features found in {dset.path_spaceFeatures}!"
    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False)
    firstKey = next(iter(featuresDict))
    featureLayer = list(featuresDict[firstKey].keys())[-1]  # get the last layer of features
    # for now only allow the last layer of features
    loader = [[], [], []]
    # size_clip = 30
    for split, idxs in enumerate([train_idxs, valid_idxs, test_idxs]):

      feats = torch.stack([featuresDict[i][featureLayer] for i in idxs]).squeeze(1) # to [n_frames, features_size] 
      labels = dset.labels["frameLabels"].values[idxs] # use class column as np array for indexing labels
      clips, clipLabels, clipIdxs = self._clip_dataitems(feats, labels, idxs, size_clip)  # clips is now [n_clips, n_features, features_size]
      
      # the dataset zips the tensors together to be feed to a dataloader (unzips them)
      featset = TensorDataset(clips, clipLabels, clipIdxs)
      # print(featset[0][0].shape, featset[0][1].shape, featset[0][2].shape)
      loader[split] = DataLoader(featset, batch_size=size_batch, num_workers=4, shuffle=False)
    exampleBatch = next(iter(loader[0]))
    print(f"\tBatch example shapes:\n\t[space features] {exampleBatch[0].shape}\n\t[labels] {exampleBatch[1].shape}\n\t[idxs] {exampleBatch[2].shape}\n")
    return loader
  
  def _clip_dataitems(self, feats, labels, idxs, size_clip=3):
    # split into groups (clips) with size "size_clip" 
    
    # if not divisible by size_clip, pad the last clip
    if feats.shape[0] % size_clip != 0:
      pad_size = size_clip - (feats.shape[0] % size_clip)
      tensorPadding = torch.zeros(pad_size, feats.shape[1], dtype=feats.dtype, device=feats.device)
      labelPadding = np.zeros(pad_size, dtype=np.int64)  # Assuming labels are int64
      idxPadding = np.array([-1] * pad_size, dtype=np.int64)  # Assuming idxs are int64
      feats = torch.cat([feats, tensorPadding], dim=0)
      labels = np.concatenate([labels, labelPadding])
      idxs = np.concatenate([idxs, idxPadding])
    # print(f"feats shape: {feats.shape}, labels shape: {labels.shape}, idxs shape: {idxs.shape}")
    # print(feats.shape, labels.shape, idxs.shape)
    clips = feats.view(-1, feats.shape[1], size_clip)  # reshape to [n_clips, features_size, n_features==size_clip]
    clipLabels = torch.tensor(labels).view(-1, size_clip)  # reshape to [n_clips, n_features]
    clipIdxs = torch.tensor(idxs).view(-1, size_clip)  # reshape to [n_clips, n_features]
    # print(f"clips shape: {clips.shape}, clipLabels shape: {clipLabels.shape}, clipIdxs shape: {clipIdxs.shape}")
    return clips, clipLabels, clipIdxs
    

    clips = torch.stack(clips).permute(0, 2, 1)  # now tensor: [n_clips, features_size, n_features==size_clip]
    print(clips.shape)

class Showcaser:
  def __init__(self):
    pass
  def show(self, imgs, imDomain=torch.Tensor):
    if not isinstance(imgs, list):
      imgs = [imgs]
    size_imgs = len(imgs); print(size_imgs)
    cols = 5
    rows = 7 
    for img in imgs:  
      fig, axes = plt.subplots(rows, cols, figsize=(8, 11))
      for i, ax in enumerate(axes.flat):  # Iterate over the tensors and plot each one
        if i < size_imgs:
          if imDomain == torch.Tensor:
            if i.dim() == 1:
              print("Tensor is 1D.")
              i = i.view(1, -1) # Reshape the tensor to 2D for visualization
            img = imgs[i].permute(1, 2, 0).numpy()  # Change shape from (C, H, W) to (H, W, C)
          ax.imshow(img)
        ax.axis('off')  # Hide axes
      plt.show()
      plt.savefig(".")