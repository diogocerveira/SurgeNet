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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SampledVideosDataset(Dataset):
  ''' Custom dataset of frames from sampled surgery videos
  '''
  def __init__(self, DATA_SCOPE, path_samples, path_labels, transform, recycle_itemIds):
    '''
      Args:
        path_labels (str): path to csv with annotations
        path_samples (str): path to folder with all images
        path_annots (str): path to raw annotations (to extract classesInt)
        transform (callable, optional): transform a sample
    '''
    self.DATA_SCOPE = DATA_SCOPE
    self.path_samples = path_samples
    self.datatype = "png"
    self.path_labels = path_labels
    self.labels = self._get_labels(path_labels) # frameId, video, frame_label
    # print(self.labels.index)
    self.recycle_itemIds = recycle_itemIds
    # print(type(recycle_itemIds))
    self.transform = transform
    # NOT WORKING
    # Filter labels to include only rows where the frame file exists
    # self.labels = self.filter_valid_frames(self.labels) # Useful when using short version of the dataset with the full labels.csv
    # print("self_labels\n", self.labels)
    self.path_spaceFeatures = None

    self.videoNames = self.labels["videoId"].unique()  # Extract unique video identifiers
    self.videoToGroup = {videoId: idx for idx, videoId in enumerate(self.videoNames)}
    
    self.N_CLASSES = len(self.labels["class"].unique())

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
      self.labels.loc[idx, "frameId"] + ".png")
    # Read images and labels
    img = self.transform(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB))
    label = self.labels.loc[idx, "class"]
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
    labels = self.labels.loc[idxs, "class"].tolist()
    if self.transform:
      imgs = [self.transform(img) for img in imgs]
    else:
      imgs = [torch.tensor(img) for img in imgs]
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    idxs = torch.tensor(idxs)
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
      # self.write_labels_csv(df, "deg.csv")
      return labels
    except:
      raise ValueError("\nInvalid path_labels/labels.csv file (might suggest inexistent or a problematic dataset)")
  def update_labels(self, path_labels):
    self.labels = self._get_labels(path_labels)
  
  def write_labels_csv(self, df, path_to):
    df.to_csv(path_to, index=False)

  def get_grouping(self, idx=None):
    ''' Returns a list of numeric labels corresponding to the video groups '''
    if idx is None:
        idx = slice(None)  # This is equivalent to selecting all rows
    elif isinstance(idx, tuple):
        idx = slice(*idx)  # Convert tuple to slice
    return [self.videoToGroup[videoId] for videoId in self.labels["videoId"][idx]]
  
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


class Cataloguer():
  ''' Each Cataloguer organises/handles 1 dataset (local - LDSS - or external - CIFAR10)
      In - digital rawdata from pc paths
      Action - sample, train, eval
      Pipeline: [sample] -> [label] -> [preprocess] -> [batch]
      Out - split and batched dataset to a Teacher
  '''
  def __init__(self, DATA, path_annots):
    self.path_annots = path_annots
    self.id_dataset = DATA["id_dataset"]
    self.preprocessing = DATA["preprocessing"]
    self.recycle_itemIds = DATA["recycle_itemIds"]
    # updated when dataset is built
    self.dataset = None
    self.labelToClass = None
    self.N_CLASSES = None
    self.DATASET_SIZE = None
    self.classWeights = None
    self.BALANCED = None
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
    ''' Label sampled videos'''
    # Helper functions
    def detect_encoding(path_to_frame):
      ''' Extract and return annotation file encoding'''
      with open(path_to_frame, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']
    def extract_class(protoclass):
      ''' Transform "End of (x) y" protoclass into (x, y) -> specific for this way of annotation'''
      for l in protoclass:
        if l.isdigit():
          return (int(l), protoclass[protoclass.find(l) + 3:])
    totalFramesId = []
    totalFramesClass = []
    for sampledVideoId in os.listdir(path_from):
      path_annot = os.path.join(self.path_annots, f"{sampledVideoId}.json")
      path_video = os.path.join(path_from, sampledVideoId)
      if os.path.exists(path_annot):  # if video is annotated (to make a better check)
        with open(path_annot, 'r', encoding=detect_encoding(path_annot)) as f:
          annots = json.load(f)  # parsed into list of dicts=annotations
          labelsNL = [m["content"] for m in annots]
          timestamps = [a["playback_timestamp"] for a in annots]
        labelsInt = [extract_class(p)[0] for p in labelsNL]
        annotatedPoints = list(zip(timestamps, labelsInt))
        # print("annotatedPoints: ", annotatedPoints)
        framesId = [os.path.splitext(frame)[0] for frame in os.listdir(path_video) if not frame.startswith(('.', '_')) and not os.path.isdir(os.path.join(path_video, frame))]
        # print(framesId)
        
        framesTimestamps = [int(frameId.split('_')[1]) for frameId in framesId]  # convert to int
        # print(len(framesTimestamps))
        if labelType == "phase-class":
          get_class = self._get_phaseClass
        elif labelType == "time-to-next-phase":
          get_class = self._get_timeToNextPhase
        totalFramesClass.extend(get_class(framesTimestamps, annotatedPoints))
        totalFramesId.extend(framesId)
        # print(len(totalFramesClass), len(totalFramesId))
      elif sampledVideoId.startswith('.'):  # hidden file are disregarded
        # print(f"   Skipping hidden file {sampledVideoId}\n")
        continue
      else:
        print(f"   Warning: {path_annot} not found, skipping annotation for {sampledVideoId}")
      # Prepare CSV file
      assert len(totalFramesClass) == len(totalFramesId), f"Number of frames ({len(totalFramesClass)}) and labels ({len(totalFramesId)}) do not match! Aborting labeling."
      with open(path_to, 'w', newline='') as file_csv:  # newline='' maximizes compatibility with other os
        fieldNames = ['frameId', 'class']
        writer = csv.DictWriter(file_csv, fieldnames=fieldNames)
        if os.stat(path_to).st_size == 0: # Check if the file is empty to write the header
          writer.writeheader()
        # Batch write frames and annotations
        for frameId, sampleClass in zip(totalFramesId, totalFramesClass):
          writer.writerow({'frameId': frameId, 'class': sampleClass})
  def _label_timeSincePhaseStart(self, framesTimestamps, annotatedPoints):
    framesClass = []
    annotPhaseBegs = [annotatedPoints[i] for i in range(0, len(annotatedPoints), 2)]  # step of 2 to get only the beginning of each phase
    for ts in framesTimestamps:
      for bTC in annotPhaseBegs:  # get the time until next phase (bigger ts) beginning
        if bTC[0] < ts:
          framesClass.append(int((ts - bTC[1]) / 1000))  # get difference in seconds
          break
      else:
        framesClass.append(0)
    return framesClass

  def _get_phaseClass(self, framesTimestamps, annotatedPoints):
    framesClass = []
    for ts in framesTimestamps:
      for i in range(0, len(annotatedPoints), 2): # step of 2 because boundary frames are in pairs (beginning+end)
        if (ts >= annotatedPoints[i][0]) and (ts < annotatedPoints[i + 1][0]):
          framesClass.append(annotatedPoints[i][1])
          break
      else:
        framesClass.append(0)  # if no phase, append class 0
    return framesClass
  def _get_timeToNextPhase(self, framesTimestamps, annotatedPoints):
    framesClass = []
    annotPhaseBegs = [annotatedPoints[i] for i in range(0, len(annotatedPoints), 2)]  # step of 2 to get only the beginning of each phase
    for ts in framesTimestamps:
      for bTC in annotPhaseBegs:  # get the time until next phase (bigger ts) beginning
        if bTC[0] >= ts:
          framesClass.append(int((bTC[1]-ts) / 1000))  # get difference in seconds
          break
      else:
        framesClass.append(0)
    return framesClass
  def _get_datasetSize(self):
    if self.dataset.DATA_SCOPE == "local":
      return len(self.dataset)
    elif self.dataset.DATA_SCOPE == "external":
      return len(self.dataset[0]) + len(self.dataset[1])
  def _get_classWeights(self):
    ''' Get class distribution form the Dataset object'''
    # print(self.labels.columns)
    cw = self.dataset.labels["class"].value_counts() / self.DATASET_SIZE
    return torch.tensor(cw.values, dtype=torch.float32)  
  def _get_labelToClass(self):
    ''' Get a map from labels (int) to classesInt (string) '''
    if self.dataset.DATA_SCOPE == "local":
      def get_sigla(string):
        # get the first letter of each word in a string in uppercase
        return "".join([word[0].upper() for word in string.split()])
      def detect_encoding(path_to_frame):
        ''' Extract and return the file encoding'''
        with open(path_to_frame, 'rb') as f:
          raw_data = f.read()
          result = chardet.detect(raw_data)
          return result['encoding']
      def extract_class(protoclass):
        ''' Transform "End of (x) y" protoclass into (x, y) -> specific for this way o annotation '''
        for l in protoclass:
          if l.isdigit():
            return (l, protoclass[protoclass.find(l) + 3:])

      path_randomAnnot = os.path.join(self.path_annots, random.choice(os.listdir(self.path_annots))) # get random annotation to extract all classesInt (assuming they all have - they DON'T!!!)
      # print(path_randomAnnot)
      if os.path.exists(path_randomAnnot):
        encoding = detect_encoding(path_randomAnnot)
        with open(path_randomAnnot, 'r', encoding=encoding) as f:
          randomAnnot = json.load(f)
          classesNL = [m["content"] for m in randomAnnot]
          labelToClass = {int(extract_class(p)[0]): extract_class(p)[0] + get_sigla(extract_class(p)[1]) for p in classesNL}
          
          if 0 in self.dataset.labels["class"].unique():
            labelToClass[0] = "0ther" 
        # filter out non-existing classes in the real (e.g. partial, lite versions) dataset of items
        assert len(labelToClass) == len(self.dataset.labels["class"].unique()), f"Number of classes ({len(labelToClass)}) and labels ({len(self.dataset.labels['class'].unique())}) do not match! Aborting labeling."
        # labelToClass = {int(k): v for k, v in labelToClass.items() if k in self.dataset.labels["class"].unique()}
        
        return labelToClass
      
    elif self.dataset.DATA_SCOPE == "external":
      return {i: c for c, i in self.dataset[1].class_to_idx.items()};
    else:
      raise ValueError("Invalid DATA_SCOPE!")
  def _get_preprocessing(self, preprocessing):
    ''' Get torch group of tranforms directly applied to torch Dataset object'''
    if preprocessing == "basic":
      transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToImage(), # only for v2
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.CenterCrop(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif preprocessing == "aug":
      transform = transforms.Compose([
      # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
      transforms.Resize((224, 224), antialias=True),
      transforms.ToImage(), # only for v2
      transforms.ToDtype(torch.float32, scale=True),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # (mean) (std) for each channel New = (Prev - mean) / stf
      transforms.RandomRotation((-90, 90)),
      transforms.ColorJitter(10, 2)
      ])
    elif not preprocessing:
      # make identity transform
      transform = transforms.Compose([
        transforms.ToImage(), # only for v2
        transforms.ToDtype(torch.float32, scale=True),
      ])
    else:
      raise ValueError("Invalid preprocessing key")
    return transform
  
  def build_dataset(self, path_samples, path_labels):
    ''' Get the Dataset objects according to the DATA_SCOPE'''
    transform = self._get_preprocessing(self.preprocessing)
    DATA_SCOPE = self.id_dataset.split('-')[1]
    if DATA_SCOPE == "local":
      dataset = SampledVideosDataset(DATA_SCOPE, path_samples, path_labels, transform, self.recycle_itemIds)    
    elif DATA_SCOPE == "external":
      train_dataset = datasets.CIFAR10(root=f"./data/{DATA_SCOPE}", train=True, download=True, transform=self.preprocessing)
      test_dataset = datasets.CIFAR10(root=f"./data/{DATA_SCOPE}", train=False, download=True, transform=self.preprocessing)
      dataset = (train_dataset, test_dataset)
    else:
      raise ValueError("Invalid DATA_SCOPE!")
    
    self.dataset = dataset  # temporarily
    self.labelToClass = self._get_labelToClass()   # gets dict for getting class name from int value
    self.N_CLASSES = dataset.N_CLASSES
    self.DATASET_SIZE = self._get_datasetSize()
    self.classWeights = self._get_classWeights()
     # check if the class distribution is balanced
    if all([self.classWeights[i] == self.classWeights[i + 1] for i in range(len(self.classWeights) - 1)]):
      self.BALANCED = True
    else:
      self.BALANCED = False
    return dataset
  
  def split(self, k_folds, dataset=None):
    ''' returns ([])
    '''
    dataset = self.dataset
    # Practice / Test Split
    listOfIdxs = []
    if dataset.DATA_SCOPE == "local":
      outerKF = GroupKFold(n_splits=k_folds)
      outerGroups = dataset.get_grouping()
      outerKFSplit = outerKF.split(np.arange(len(dataset)), groups=outerGroups)  # train/valid and test split - fold split
      for tv_idx, test_idxs in outerKFSplit:
        # print('\n', tv_idx, test_idxs, '\n')
        # tv_subset = torch.utils.data.Subset(dataset, tv_idx)
        innerGroups = dataset.get_grouping(tv_idx)
        # print(innerGroups)
        innerKF = GroupKFold(n_splits=k_folds - 1)
        train_localIdxs, valid_localIdxs = next(innerKF.split(tv_idx, groups=innerGroups))  # train and valid split - practice split
        # print("trainloc", train_localIdxs, '\n', "validloc", valid_localIdxs)
        train_idxs = tv_idx[train_localIdxs] # the inner splits idx don't match the outer directly
        valid_idxs = tv_idx[valid_localIdxs] # because idxs for some video were removed (need to map back)
        listOfIdxs.append([train_idxs, valid_idxs, test_idxs])
        # print('\n', [train_idxs, valid_idxs, test_idxs], '\n')

      videosSplit = dataset.get_idxToVideo(listOfIdxs)
      # print(videosSplit[0])
      # print(listOfIdxs[0])
      return tuple(zip(listOfIdxs, videosSplit))

    # Train / Valid Split
    elif dataset.DATA_SCOPE == "external":
      indices = np.arange(len(dataset[0]))
      train_idxs, valid_idxs = train_test_split(indices, train_size = (k_folds - 1) / (k_folds), random_state=53)
      # print(train_idxs, valid_idxs)
      # print(dataset[1])
      listOfIdxs.append([train_idxs, valid_idxs, dataset[1]])
      return tuple(zip(listOfIdxs, [[["no"], ["groups"], ["here"]]]))

    # print(f"Train idxs: {train_idxs}\nValid idxs: {valid_idxs}\nTest idxs: {test_idxs}")
    # return train_idxs, valid_idxs, test_idxs

  def batch(self, dset, inputType, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    ''' helper for loading data, features or stopping when fx_mode == 'export'
    '''
    # print(inputType)
    if inputType == "images":
      return self._batch_images(dset, size_batch, train_idxs, valid_idxs, test_idxs)
    elif inputType == "fmaps":
      return self._batch_clips(dset, size_batch, train_idxs, valid_idxs, test_idxs)
    else:
      raise ValueError("Invalid Batch Mode!")
  def _batch_images(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    loaders = {}
    if self.dataset.DATA_SCOPE == "local":
      if np.any(train_idxs):
        loaders["trainloader"] = DataLoader(dataset=dset, batch_size=size_batch, sampler=torch.utils.data.SubsetRandomSampler(train_idxs))
      if np.any(valid_idxs):
        loaders["validloader"] = DataLoader(dataset=dset, batch_size=size_batch, sampler=torch.utils.data.SubsetRandomSampler(valid_idxs))
      if np.any(test_idxs):
        loaders["testloader"] = DataLoader(dataset=dset, batch_size=size_batch, sampler=torch.utils.data.SubsetRandomSampler(test_idxs))
    elif self.dataset.DATA_SCOPE == "external":
      # test_idxs actually carries the test part of the dataset when using CIFAR-10
      if np.any(train_idxs):
        loaders["trainloader"] = DataLoader(dataset=dset[0], batch_size=size_batch, sampler=torch.utils.data.SubsetRandomSampler(train_idxs))
      if np.any(valid_idxs):
        loaders["validloader"] = DataLoader(dataset=dset[0], batch_size=size_batch, sampler=torch.utils.data.SubsetRandomSampler(valid_idxs))
      if np.any(test_idxs):
        loaders["testloader"] = DataLoader(dataset=test_idxs, batch_size=size_batch, shuffle=True)
      # print(next(trainloader).shape)
    return loaders.values()

  def _batch_clips(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(dset.path_spaceFeatures), f"No exported space features found in {dset.path_spaceFeatures}!"
    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False)
    firstKey = next(iter(featuresDict))
    featureLayer = list(featuresDict[firstKey].keys())[-1]  # get the last layer of features
    # for now only allow the last layer of features
    loader = [[], [], []]
    size_clip = 3
    for split, idxs in enumerate([train_idxs, valid_idxs, test_idxs]):

      feats = torch.stack([featuresDict[i][featureLayer] for i in idxs]).squeeze(1) # to [n_frames, features_size] 
      labels = dset.labels["class"].values[idxs] # use class column as np array for indexing labels
      clips, clipLabels, clipIdxs = self._clip_dataitems(feats, labels, idxs, size_clip)  # clips is now [n_clips, n_features, features_size]
      
      # the dataset zips the tensors together to be feed to a dataloader (unzips them)
      featset = TensorDataset(clips, clipLabels, clipIdxs)
      # print(featset[0][0].shape, featset[0][1].shape, featset[0][2].shape)
      loader[split] = DataLoader(featset, batch_size=size_batch, shuffle=False)
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
      

  def _old_batch_features(self, path_spaceFeatures, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(path_spaceFeatures), f"No exported space features found in {path_spaceFeatures}!"
    # batch size is "one video"
    loader = [[], [], []] 
    for i, idx in enumerate([train_idxs, valid_idxs, test_idxs]):
      df = self.dataset.labels.iloc[idx].copy()
      # print(df.columns)
      df["timestamp"] = df["frame"].str.split('_', expand=True)[1].astype(int)  # Ensure numeric sorting    
      df = df.sort_values(by=["videoId", "timestamp"])
      # print(df)
      for vid, group in df.groupby("videoId"):
        try:
          # print(self.features[list(self.features.keys())[-1]])
          # Accumulate features, labels and frame_nmes as tensors for each video
          feats = torch.stack([self.features[frame["frame"]][self.FEATURE_LAYER] for _, frame in group.iterrows()])
          labels = torch.tensor(group["class"].values)
          frameNames = [frame["frame"] for _, frame in group.iterrows()]

          # Ensure the channel dimension is present
          # print(features.shape)
          feats = feats.permute(1, 0).unsqueeze(0)  # to [size_batch, features_size, n_frames]
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
        # video = vid
    # print(loader[2][video][0])
    # assert 1 == 0

    return loader # trainloader, validloader, testloader
    

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