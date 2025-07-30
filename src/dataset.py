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
from sklearn.model_selection import KFold, GroupKFold, train_test_split, LeaveOneGroupOut
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
    self.labels = self._get_labels(path_labels) # frameId, videoInt, frame_label
    # print(self.labels.index)
    self.transform = transform
    # NOT WORKING
    # Filter labels to include only rows where the frame file exists
    # self.labels = self.filter_valid_frames(self.labels) # Useful when using short version of the dataset with the full labels.csv
    # print("self_labels\n", self.labels)
    self.path_spaceFeatures = None
    self.path_timeFeatures = None

    self.videoNames = self.labels["videoInt"].unique()  # Extract unique video identifiers
    self.videoToGroup = {videoInt: idx for idx, videoInt in enumerate(self.videoNames)}
    
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
    path_img = os.path.join(self.path_samples, self.labels.loc[idx, "videoInt"],
      self.labels.iloc[idx, "frameId"] + '.' + self.datatype)
    # Read images and labels
    img = torch.tensor(self.transform(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)))
    label = torch.tensor(self.labels.loc[idx, "frameLabels"])
    if self.labelType.split('-')[0] == "multi":
      target = self._multiHot([label], num_classes=self.n_classes)

    # print(f"img: {img.shape}, label: {label}, idx: {torch.tensor(idx)}")  # WTFFFF does not print
    return (img, target, torch.tensor(idx))
  def __getitems__(self, idxs):
    """Retrieve items from the dataset based on index or list of indices."""
    # Convert dset idxs to list
    if isinstance(idxs, slice):  # Convert slice to list
      idxs = list(range(*idxs.indices(len(self.labels))))
    elif torch.is_tensor(idxs) or isinstance(idxs, np.ndarray):
      idxs = idxs.tolist()
    # print(idxs[:4])
    # each idx is a int tensor
    path_imgs = [os.path.join(self.path_samples, self.labels.iloc[i]["videoInt"],
        self.labels.iloc[i]["frameId"] + '.' + self.datatype) for i in idxs]
    # Read images and labels
    # imgs = [iio.imread(p) for p in path_imgs]
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in path_imgs]
    labels = self.labels.iloc[idxs]["frameLabels"].tolist()
    self.export_images(imgs[:5], prefix='before_transform')
    if self.transform:
      imgs = [self.transform(img) for img in imgs]
    else:
      imgs = [torch.tensor(img) for img in imgs]
    imgs = torch.stack(imgs)
    targets = self._get_targets(labels, self.labelType.split('-')[0])
    idxs = torch.tensor(idxs)

    self.export_images(imgs[:5], prefix='after_transform')
    
    # print("idxs: ", idxs[:5])
    # print(self.labels.iloc[idxs[:5]])
    # print("targets: ", targets[:5], '\n')
    # print(f"imgs: {imgs.shape}, targets: {targets}, idxs: {idxs}")
    return list(zip(imgs, targets, idxs))
  
  def export_images(self, imgs, output_dir='exported_images', prefix='img'):
    """Export images to a specified directory with a prefix."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(imgs):
      img_path = os.path.join(output_dir, f"{prefix}_{i}.png")
      # if numpy array
      if isinstance(img, np.ndarray):
        plt.imsave(img_path, img)
      elif isinstance(img, torch.Tensor):
        if img.dim() == 3:
          img = img.permute(1, 2, 0)
          plt.imsave(img_path, img.numpy())
    
      print(f"Exported {img_path}")

    


  def _get_targets(self, labels, labelTypeLeft):
    """Convert labels to numeric targets based on the labelType."""
    # print(self.labelToClassMap)
    if labelTypeLeft == "single":
      # Assume labels like ['cutting', 'suturing', ...]
      targets = [self.labelToClassMap[label.strip()] for label in labels]
      targets = torch.tensor(targets, dtype=torch.long)
    elif labelTypeLeft == "multi":
      # Assume labels like ['cutting,suturing', 'suturing', ...]
      label_indices = [[self.labelToClassMap[lbl.strip()] for lbl in multi.split(',')] for multi in labels]
      targets = self._multiHot(label_indices, self.n_classes)
    else:
      raise ValueError("Invalid labelType domain!")
    return targets

  def _multiHot(self, seq, n_classes):
    hotSeq = torch.zeros(len(seq), n_classes, dtype=torch.float32)
    for i, indices in enumerate(seq):
      hotSeq[i, indices] = 1.0
    return hotSeq

  def _get_labels(self, path_labels):
    try:
      # print(path_labels, "\n\n", flush=True)
      labels = pd.read_csv(path_labels)
      # print(df.columns)
      # labels.reset_index(drop=True, inplace=True) # Create a new integer index and drop the old index
      # (Optional) Create a new column for videoInt based on the frame column
      labels['videoInt'] = labels['frameId'].str.split('_', expand=True)[0]
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
    return [self.videoToGroup[videoInt] for videoInt in self.labels.iloc[idx]["videoInt"]]
  
  def _get_idxToVideo(self, listOfIdxs):
    ''' Extract video IDs (check for overlap in sets) - specific to each labels.csv implementation
        List of idxs [[tr_i, va_i, te_i], ...] --> List of videos [[vidsx, vidsy, vidsz], ...]
    '''
    # Load the CSV and extract videoInt
    # AFFECTed THE ORIGINAL
    # df = self.labels
    # df.drop(["frameId", "frameLabels"], axis=1, inplace=True)  # Optional: Drop unnecessary columns if not needed
    
    list_out = []

    for idxs in listOfIdxs: # for each fold indexs
      vidsList = []
      for i, split in enumerate(idxs):  # for each split  (train, valid, test)
        vidsInSplit = self.labels.iloc[split]["videoInt"].unique()  # Get unique videoNames for this subset
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
    ''' Get inverse class weights from dataset labels '''
    labels_counts = Counter()
    for multi in labels["frameLabels"]:
        single = [l.strip() for l in multi.split(',')]  # comma separated labels
        labels_counts.update(single)
    # Convert to frequencies
    cw = pd.Series(labels_counts) / self.__len__()  # class frequencies
    inv_cw = 1.0 / cw  # Inverse frequency weights
    # Normalize weights so sum equals number of classes (optional)
    inv_cw = inv_cw / inv_cw.sum() * len(inv_cw)
    return torch.tensor(inv_cw.values, dtype=torch.float32)
  
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
  def __init__(self, DATA, path_annots, DEVICE):
    self.path_annots = path_annots
    self.preprocessingType = DATA["preprocessing"]
    self.labelType = DATA["labelType"]
    self.predicateLabelsPossible = DATA["predicateLabels"]
    self.objectLabelsPossible = DATA["objectLabels"]
    self.DEVICE = DEVICE
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
    for videoInt in availableVideos:
      if skip > 0:
        skip -= 1
        print(videoInt)
        continue
      #videoInt = os.path.splitext(videoInt)[0]  # remove extension
      path_video = os.path.join(path_videos, videoInt)
      #metadata = iio.immeta(path_video, plugin="pyav")  
      #fps = metadata["fps"]
      #nframes = int(metadata["duration"] * fps)
      #print(f"\niio - Sampling {videoInt} ({nframes} frames at {fps} fps), period of {int(fps / samplerate)}")
      
      # print(f"FPS: {fps}, Nframes: {nframes}, Period: {period}")
      if filter_annotated:
        path_annot = os.path.join(path_annots, f"{os.path.splitext(videoInt)[0]}.json")
        if not os.path.exists(path_annot):
          continue
      try:
        videoInt = videoInt[:4]  # keep only 1st 4 chars of the video name
      except:
        videoInt = os.path.splitext(videoInt)[0]  # if smaller name only remove ext
      shutil.copy(path_annot, os.path.join(self.path_annots, f"{videoInt}.json"))
      path_frames = os.path.join(path_to, videoInt)
      os.makedirs(path_frames, exist_ok=True)
      
      start_time = time.time()
      cap = cv2.VideoCapture(path_video)  # Initialize video capture object
      fps = cap.get(cv2.CAP_PROP_FPS)
      nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      period = round(fps / samplerate)
      print(f"\ncv2 - Sampling {videoInt} ({nframes} frames at {fps} fps), period of {period} frames")
      # Get the first frame to determine the size
      ret, firstFrame = cap.read()
      if ret:
        factor = 1 / 4
        height, width = firstFrame.shape[:2]  # Get the original dimensions of the frame
        newWidth = int(width * factor)
        newHeight = int(height * factor)
        print(f"Pre-resizing video {videoInt} to {newWidth}x{newHeight} (factor of {factor})")
      for i in range(0, nframes, period):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Set the current frame position
        ret, frameImg = cap.read()
        if ret:
          frameId = f"{videoInt}_{int(i / fps * 1000)}"  # remember basename is the file name itself / converting index to its timestamp (ms)
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
    
  def crop_frames(self, path_from, path_to, cropWindow=None):

    for videoFolder in sorted(os.listdir(path_from)):
      path_from_video = os.path.join(path_from, videoFolder)
      # print(videoFolder)
      if os.path.isdir(path_from_video) and not videoFolder.startswith(('.', '_')):  # Process only directories
        print(f"Cropping video {videoFolder}\n")
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
              if not cropWindow:
                height, width = img.shape[:2]
                cropWindow = (width // 8, 0, width - 2 * (width // 8), height)

              x, y, w, h = cropWindow
              croppedImg = img[y:y+h, x:x+w]
              path_to_frame = os.path.join(path_to_video, frame)
              cv2.imwrite(path_to_frame, croppedImg)
            except Exception as e:
              print(f"Error processing {path_from_frame}: {e}")

    print("D. Resizing finished!\n")
  
  def label(self, path_from, path_to, labelType):
    ''' folders of sampled videos -> labels.csv
    '''
    totalFrameIds = []
    totalFrameLabels = []
    for videoInt in os.listdir(path_from): # sampled videos (folder of frames)
      path_annot = os.path.join(self.path_annots, f"{videoInt}.json")
      path_video = os.path.join(path_from, videoInt)
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
      elif videoInt.startswith(('.', '_')):  # hidden file are disregarded
        print(f"   Skipping hidden file {videoInt}\n")
        continue
      else:
        print(f"   Warning: Skipping annotation for {videoInt} ({path_annot} not found)")

    # Prepare CSV file
    with open(path_to, 'w', newline='') as file_csv:  # newline='' maximizes compatibility with other os
      fieldNames = ['frameId', 'frameLabels']
      writer = csv.DictWriter(file_csv, fieldnames=fieldNames)
      if os.stat(path_to).st_size == 0: # Check if the file is empty to write the header
        writer.writeheader()
      # Batch write frames and annotations
      sortedZip = sorted( # sort by videoInt and then timestamp
          zip(totalFrameIds, totalFrameLabels),
          key=lambda x: (x[0].split('_')[0], int(x[0].split('_')[1]))
          )  # sort by frameId
      for frameId, frameLabel in sortedZip:
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
    # print(phaseDelimiters, '\n', len(phaseDelimiters))
    # print(framesTimestamps)
    # print(list(range(0, len(phaseDelimiters), 2)))
    # Fix odd number of delimiters
    try:
      if phaseDelimiters[0][1] != phaseDelimiters[1][1]:
        phaseDelimiters.insert(0, (0, phaseDelimiters[0][1]))
        # print(phaseDelimiters[0])
        # print(len(phaseDelimiters))
      if phaseDelimiters[-1][1] != phaseDelimiters[-2][1]:
        phaseDelimiters.append((max(framesTimestamps) + 1, phaseDelimiters[-1][1]))
        # print(phaseDelimiters[-1])
        # print(len(phaseDelimiters))
        # print(phaseDelimiters)
      # print(phaseDelimiters[0][1], "  ", phaseDelimiters[1][1])
      # print(phaseDelimiters[-1][1], "  ", phaseDelimiters[-2][1])
    except IndexError:
      raise ValueError("Phase delimiters list is empty or has only one element, cannot label frames.")
    assert len(phaseDelimiters) % 2 == 0, "Phase delimiters should be in pairs (beginning/resume+end/pause)"
    # assert 1 == 0
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
          # innerKF = LeaveOneGroupOut()

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

  def batch(self, dset, inputType, HYPER, train_idxs=None, valid_idxs=None, test_idxs=None):
    ''' helper for loading data, features or stopping when fx_mode == 'export'
    '''
    inputTypeLeft, inputTypeRight = inputType.split('-')
    # print(inputType)
    if inputTypeLeft == "images": # regular resnet training on images
      return self._batch_images(dset, HYPER["spaceBatchSize"], train_idxs, valid_idxs, test_idxs)
    elif inputTypeLeft == "fmaps":
      if inputTypeRight == "framed":  # classifier training on space features
        return self._batch_framed(dset, HYPER["spaceBatchSize"], train_idxs, valid_idxs, test_idxs)
      elif inputTypeRight == "clipped": # temporal model training on space features
        return self._batch_clips(dset, HYPER["timeBatchSize"], HYPER["clipSize"], train_idxs, valid_idxs, test_idxs)
      elif inputTypeRight == "video": # temporal model training on single videos as each padded "batch"
        return self._batch_videos(dset, HYPER["timeBatchSize"], train_idxs, valid_idxs, test_idxs)
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

  def _batch_framed(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(dset.path_spaceFeatures), f"No exported space features found in {dset.path_spaceFeatures}!"
    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False)
    firstKey = next(iter(featuresDict))
    featureLayer = list(featuresDict[firstKey].keys())[-1]  # get the last layer of features
    print(f"Using feature layer: {featureLayer}")

    # for now only allow the last layer of features
    loader = [[], [], []]
    # size_clip = 30
    for split, idxs in enumerate([train_idxs, valid_idxs, test_idxs]):

      feats = torch.stack([featuresDict[i][featureLayer] for i in idxs]).squeeze(1) # to [n_frames, features_size] 
      labels = dset.labels.iloc[idxs]["frameLabels"].tolist() # use class column as np array for indexing labels
      targets = dset._get_targets(labels, dset.labelType.split('-')[0])  # convert to targets
      idxs = torch.tensor(idxs, dtype=torch.int64)  # convert to tensor
      # print(type(targets), targets.dtype)
      # print(targets[:5])  # show a few samples

      # move to cpu for DataLoader
      feats = feats.cpu()
      targets = targets.cpu()
      idxs = idxs.cpu()
      # the dataset zips the tensors together to be feed to a dataloader (unzips them)
      featset = TensorDataset(feats, targets, idxs)
      # print(featset[0][0].shape, featset[0][1].shape, featset[0][2].shape)
      loader[split] = DataLoader(featset, batch_size=size_batch, num_workers=2, shuffle=True)
    exampleBatch = next(iter(loader[0]))
    print(f"\tBatch example shapes:\n\t[space features] {exampleBatch[0].shape}\n\t[labels] {exampleBatch[1].shape}\n\t[idxs] {exampleBatch[2].shape}\n")
    return loader
  
  def _batch_videos(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(dset.path_spaceFeatures), f"No exported space features found in {dset.path_spaceFeatures}!"
    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False, map_location=self.DEVICE)
    firstKey = next(iter(featuresDict))
    availableLayers = list(featuresDict[firstKey].keys())
    featureLayer = availableLayers[-1]  # or some index/selection logic
    print(f"  Available feature layers: {availableLayers} / Using last layer: {featureLayer}\n")
    
    groupedIdxs, splitVidsInt = self.get_vidsIdxsAndSplits(dset, train_idxs, valid_idxs, test_idxs)
    maxFrames = max(len(frameIdxs) for frameIdxs in groupedIdxs.values())

    loader = [[], [], []]
    for split, vidsInt in splitVidsInt.items():
      # Prepare padded data for all videos
      splitFeats = []
      splitTargets = []
      splitIdxs = []
      for vid in vidsInt:
        if vid in groupedIdxs:
          frameIdxs = groupedIdxs[vid]
          n_frames = len(frameIdxs)
          # Get features, targets, and indices for this video
          vidFeats = torch.stack([featuresDict[i][featureLayer] for i in frameIdxs]).squeeze(1)
          vidLabels = dset.labels.iloc[frameIdxs]["frameLabels"].tolist()
          vidTargets = dset._get_targets(vidLabels, dset.labelType.split('-')[0]).to(self.DEVICE)
          vidIdxs = torch.tensor(frameIdxs, dtype=torch.int64, device=self.DEVICE)

          # Create padding
          featureSize = vidFeats.shape[1]
          padFrames = maxFrames - n_frames
          if padFrames > 0:
            featPadding = torch.zeros(padFrames, featureSize, device=self.DEVICE)  # Pad features with zeros
            vidFeats = torch.cat([vidFeats, featPadding], dim=0)
            # Pad targets based on label type
            if len(vidTargets.shape) == 1:  # single label
              targetPadding = torch.full((padFrames,), -1, dtype=torch.int64, device=self.DEVICE)
            else:  # multi-label
              targetPadding = torch.zeros(padFrames, vidTargets.shape[1], dtype=vidTargets.dtype, device=self.DEVICE)
              targetPadding[:, 1] = -1
            vidTargets = torch.cat([vidTargets, targetPadding], dim=0)
            idxsPadding = torch.full((padFrames,), -1, dtype=torch.int64, device=self.DEVICE)  # Pad indices with -1
            vidIdxs = torch.cat([vidIdxs, idxsPadding], dim=0)
            # print(f"Padding video {vid} with {padFrames} frames")
          splitFeats.append(vidFeats)
          splitTargets.append(vidTargets)
          splitIdxs.append(vidIdxs)
      
      # Stack all videos lists into tensors
      feats = torch.stack(splitFeats).cpu()  # [num_videos, maxFrames, featureSize]
      targets = torch.stack(splitTargets).cpu()  # [num_videos, maxFrames]
      idxs = torch.stack(splitIdxs).cpu()  # [num_videos, maxFrames]

      # Create TensorDataset
      featset = TensorDataset(feats, targets, idxs)
      # Create DataLoader - now each batch contains size_batch videos
      loader[split] = DataLoader(featset, batch_size=size_batch, num_workers=2, shuffle=True)
    
    # Show example batch
    if loader[0]:
      exampleBatch = next(iter(loader[0]))
      print(f"\tBatch example shapes:\n\t[space features] {exampleBatch[0].shape}\n\t[labels] {exampleBatch[1].shape}\n\t[idxs] {exampleBatch[2].shape}")
      print(f"  Batch contains {exampleBatch[0].shape[0]} video(s), each with up to {exampleBatch[0].shape[1]} frames\n")
    
    return loader
  def get_vidsIdxsAndSplits(self, dset, train_idxs=None, valid_idxs=None, test_idxs=None):
    '''
    Returns:
      - groupedIdxs: dict {videoInt: [frameIdxs]} — all videos with their frame indices
      - splitVidsInt: dict {0|1|2: set(videoIds)} — videoIds per split (0=train, 1=valid, 2=test)
    '''
    groupedIdxs = {}
    splitVidsInt = {0: set(), 1: set(), 2: set()}
    idxMap = {0: train_idxs, 1: valid_idxs, 2: test_idxs}

    for split, idxs in idxMap.items():
      if idxs is None or len(idxs) == 0:
        continue

      groups = dset._get_grouping(idxs)
      for idx, videoInt in zip(idxs, groups):
        groupedIdxs.setdefault(videoInt, []).append(idx)
        splitVidsInt[split].add(videoInt)

    return groupedIdxs, splitVidsInt


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
      labels = dset.labels.iloc[idxs]["frameLabels"].tolist() # use class column as np array for indexing labels
      targets = dset._get_targets(labels, dset.labelType.split('-')[0])  # convert to targets
      idxs = torch.tensor(idxs, dtype=torch.int64)  # convert to tensor
      # print(type(targets), targets.dtype)
      # print(targets[:5])  # show a few samples

      clips, clipTargets, clipIdxs = self._clip_dataitems(feats, targets, idxs, dset.labelType.split('-')[0], size_clip)  # clips is now [n_clips, n_features, features_size]
      
      # move to cpu for DataLoader
      clips = clips.cpu()
      clipTargets = clipTargets.cpu()
      clipIdxs = clipIdxs.cpu()
      # the dataset zips the tensors together to be feed to a dataloader (unzips them)
      featset = TensorDataset(clips, clipTargets, clipIdxs)
      # print(featset[0][0].shape, featset[0][1].shape, featset[0][2].shape)
      loader[split] = DataLoader(featset, batch_size=size_batch, num_workers=2, shuffle=False)
    exampleBatch = next(iter(loader[0]))
    print(f"\tBatch example shapes:\n\t[space features] {exampleBatch[0].shape}\n\t[labels] {exampleBatch[1].shape}\n\t[idxs] {exampleBatch[2].shape}\n")
    return loader
  
  def _clip_dataitems(self, feats, targets, idxs, labelTypeLeft, size_clip=3):
    # split into groups (clips) with size "size_clip" 
    
    # if not divisible by size_clip, pad the last clip
    if feats.shape[0] % size_clip != 0:
      pad_size = size_clip - (feats.shape[0] % size_clip)
      featsPadding = torch.zeros(pad_size, feats.shape[1], dtype=feats.dtype, device=feats.device)
      if labelTypeLeft == "single":
        targetPadding = torch.zeros(pad_size, dtype=targets.dtype, device=targets.device)
      else:
        targetPadding = torch.zeros(pad_size, targets.shape[1], dtype=targets.dtype, device=targets.device)
      idxPadding = torch.full((pad_size,), -1, dtype=idxs.dtype, device=idxs.device)
      feats = torch.cat([feats, featsPadding], dim=0)
      targets = torch.cat([targets, targetPadding], dim=0)
      idxs = torch.cat([idxs, idxPadding], dim=0)
    # print(f"feats shape: {feats.shape}, targets shape: {targets.shape}, idxs shape: {idxs.shape}")
    # print(feats.shape, targets.shape, idxs.shape)
    clips = feats.view(-1, feats.shape[1], size_clip)  # reshape to [n_clips, features_size, n_features==size_clip]
    if labelTypeLeft == "single":
      clipTargets = targets.view(-1, size_clip)         # [n_clips, size_clip]
    else:  # multi
      clipTargets = targets.view(-1, size_clip, targets.shape[1])  # [n_clips, size_clip, n_classes]
    clipIdxs = idxs.view(-1, size_clip)  # reshape to [n_clips, n_features]
    # print(f"clips shape: {clips.shape}, clipTargets shape: {clipTargets.shape}, clipIdxs shape: {clipIdxs.shape}")
    return clips, clipTargets, clipIdxs

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