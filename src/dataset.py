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
from PIL import Image
import torchvision.transforms.functional as TF
import imageio.v3 as iio

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
    self.path_timeFeatures = None

    self.videoNames = self.labels["videoId"].unique()  # Extract unique video identifiers
    self.videoToGroup = {videoId: idx for idx, videoId in enumerate(self.videoNames)}
    
    self.labelToClassMap = self.get_labelToClassMap(self.labels)
    self.n_classes = len(self.labelToClassMap)
    self.phases = sorted(set(self.labels["frameLabels"]))
    print(f"Phases (#{len(self.phases)}): ", self.phases)
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
      self.labels.iloc[idx, "frameId"] + '.' + self.datatype)
    # Read images and labels
    img = torch.tensor(self.transform(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)))
    label = torch.tensor(self.labels.loc[idx, "frameLabels"])
    if self.labelType.split('-')[0] == "multi":
      target = self._multiHot([label], num_classes=self.n_classes)
    print("WARNING!! Wrong pipe\n\n")
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
    path_imgs = [os.path.join(self.path_samples, self.labels.iloc[i]["videoId"],
        self.labels.iloc[i]["frameId"] + '.' + self.datatype) for i in idxs]
    # Read images and labels
    # imgs = [iio.imread(p) for p in path_imgs]
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in path_imgs]
    labels = self.labels.iloc[idxs]["frameLabels"].tolist()
    # self.export_images_raw(imgs[:5], prefix='before_transform')
    if self.transform:
      imgs = [self.transform(img) for img in imgs]
    else:
      imgs = [torch.tensor(img) for img in imgs]
    # print(f"Image shape: {imgs[0].shape}")  # should be [3, 224, 224]
    # self.export_images_tensor(imgs[:5], prefix='after_transform')
    imgs = torch.stack(imgs)
    targets = self._get_targets(labels, self.labelType.split('-')[0])
    idxs = torch.tensor(idxs)
    # assert 1 == 0
    # print("idxs: ", idxs[:5])
    # print(self.labels.iloc[idxs[:5]])
    # print("targets: ", targets[:5], '\n')
    # print(f"imgs: {imgs.shape}, targets: {targets}, idxs: {idxs}")
    return list(zip(imgs, targets, idxs))


  def export_images_raw(self, imgs, prefix="raw", outdir="./debug_images"):
    os.makedirs(outdir, exist_ok=True)
    for i, img in enumerate(imgs):
      path = os.path.join(outdir, f"{prefix}_{i}.png")
      iio.imwrite(path, np.asarray(img))  # assumes RGB
  def export_images_tensor(self, tensors, prefix="tensor", outdir="./debug_images"):
    normValues = ((0.416, 0.270, 0.271), (0.196, 0.157, 0.156))
    os.makedirs(outdir, exist_ok=True)
    
    for i, tensor in enumerate(tensors):
      img = tensor.clone()
      # Per-tensor min-max normalize for visualization
      img -= img.min()
      img /= img.max()
      img = (img * 255).to(torch.uint8)
      img = TF.to_pil_image(img)
      
      # other
      # img = tensor.clone().clamp(min=0)  # zero out all negative values
      # img = img / img.max()              # normalize [0, 1]
      # img = (img * 255).to(torch.uint8)
      # img = TF.to_pil_image(img)

      img.save(os.path.join(outdir, f"{prefix}_{i}.png"))

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
      labels = pd.read_csv(path_labels)

      labels['videoId'] = labels['frameId'].str.split('_', expand=True)[0]
      labels['timestamp'] = labels['frameId'].str.split('_', expand=True)[1].astype(int)

      # Debug: Print max timestamp per video
      # max_timestamps = labels.groupby('videoId')['timestamp'].max()
      # print("[DEBUG] Max timestamp per videoId:\n", max_timestamps, "\n", flush=True)
      # Normalize timestamp
      labels['normalizedTimestamp'] = labels.groupby('videoId')['timestamp'].transform(lambda x: x / x.max())
      # Debug: Show one example comparison
      # print("[DEBUG] Example (raw vs normalized):\n", labels[['videoId', 'timestamp', 'normalizedTimestamp']].head(), "\n", flush=True)
      return labels
    except Exception as e:
      print(f"[ERROR] Failed during label processing: {e}", flush=True)
      raise

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
    return [self.videoToGroup[videoId] for videoId in self.labels.iloc[idx]["videoId"]]
  
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
        vidsInSplit = self.labels.iloc[split]["videoId"].unique()  # Get unique videoNames for this subset
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
      single = [l.strip() for l in multi.split(',')]
      labels_counts.update(single)  # keys are strings

    # Map to class indices
    counts_array = np.zeros(self.n_classes)
    for label, count in labels_counts.items():
      class_idx = self.labelToClassMap[label]
      counts_array[class_idx] = count

    # Convert to frequencies
    cw = counts_array / self.__len__()  # class frequency
    inv_cw = 1.0 / cw
    inv_cw = inv_cw / inv_cw.sum() * self.n_classes  # normalize

    return torch.tensor(inv_cw, dtype=torch.float32)

  
  def get_labelToClassMap(self, labels):
    ''' Convert to list of natural language labels into list of int number labels, saving a map class->label in classToLabelMap dict map '''
    uniqueLabels = set()
    # print(labels[:5])
    for multi in set(labels["frameLabels"]):
      for single in multi.split(','):
        uniqueLabels.add(single)
    # print(uniqueLabels)
    # print(sorted(list(uniqueLabels)))
    labelToClassMap = {l:i for i, l in enumerate(sorted(list(uniqueLabels)))}
    # print(labelToClassMap)
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
    self.actionLabelsPossible = DATA["actionLabels"]
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
      sortedZip = sorted( # sort by videoId and then timestamp
          zip(totalFrameIds, totalFrameLabels),
          key=lambda x: (x[0].split('_')[0], int(x[0].split('_')[1]))
          )  # sort by frameId
      for frameId, frameLabel in sortedZip:
        writer.writerow({'frameId': frameId, 'frameLabels': frameLabel})
        # print(frameLabel, end='\n')

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
      
      for label in self.actionLabelsPossible:
        if label.lower() in proto.lower():
          found.append(self._get_sigla(label))
          break  # assuming only 1 action per label
          # print(label)
      else:  # if no action found, append "Other"
        found.append("0Act")  # sort to have a consistent order
      for label in self.objectLabelsPossible:
        if label.lower() in proto.lower():
          found.append(self._get_sigla(label))
          break  # assuming only 1 object per label
          # print(label)
      else:
        found.append("0Obj")
      found = sorted(found)  # sort the labels to have a consistent order
      if self.labelType.split('-')[0] == "single":
        found = '-'.join(found) # labels as strings like A-B-C
      elif self.labelType.split('-')[0] == "multi":
        found = ','.join(found) # labels as strings like A,B,C
      else:
        raise ValueError("Invalid labelType domain!")
      labels.append(found)
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
          otherLabel = "0Act-0Obj"  # if no phase, append both Other labels
        elif self.labelType.split('-')[0] == "multi":
          otherLabel = "0Act,0Obj"
        else:
          raise ValueError("Invalid labelType domain!")
        framesLabel.append(otherLabel)  # if no phase, append both Other labels

      # print("label(s): ", phaseDelimiters[i][1], '\n')
    return framesLabel
  
  def _get_preprocessing(self, preprocessingType, normValues=None):
    ''' Get torch group of tranforms directly applied to torch Dataset object'''
    normValues = ((0.416, 0.270, 0.271), (0.196, 0.157, 0.156))
    if preprocessingType == "basic":
      transform = transforms.Compose([
        transforms.ToImage(), # only for v2
        transforms.Resize((224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(normValues[0], normValues[1])
      ])
    elif preprocessingType == "resnet":
      transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((224, 224), antialias=True),  # squishes if needed
        transforms.ToDtype(torch.float32, scale=True),                          # [0, 255] → [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet channel stats
                            std=[0.229, 0.224, 0.225])
      ])
    elif preprocessingType == "augmentation":
      transform = transforms.Compose([
        # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
        transforms.ToImage(), # only for v2
        transforms.Resize((224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(normValues[0], normValues[1]),
        transforms.RandomRotation((-90, 90)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        # transforms.RandomVerticalFlip(p=0.5),    # 50% chance to flip vertically
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomCrop((224, 224), padding=4, padding_mode='reflect'),  # pad and crop to 224x224
        # transforms.ColorJitter(10, 2)
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

  def get_mean_std(self, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.
    std = 0.
    total = 0

    for imgs, _, _ in loader:
      # imgs: [B, C, H, W]
      B = imgs.size(0)
      imgs = imgs.view(B, imgs.size(1), -1)  # [B, C, H*W]
      mean += imgs.mean(2).sum(0)  # → [C]
      std += imgs.std(2).sum(0)
      total += B

    mean /= total
    std /= total
    return mean.tolist(), std.tolist()


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

  def batch(self, dset, inputType, HYPER, multiclips=False, multiClipStride=1, train_idxs=None, valid_idxs=None, test_idxs=None):
    ''' helper for loading data, features or stopping when fx_mode == 'export'
    '''
    inputTypeLeft, inputTypeRight = inputType.split('-')
    # print(inputType)
    if inputTypeLeft == "images": # regular resnet training on images
      return self._batch_dataset(dset, HYPER["spaceBatchSize"], train_idxs, valid_idxs, test_idxs)
    elif inputTypeLeft == "fmaps":
      if inputTypeRight == "frame":  # classifier training on space features
        return self._batch_frameFeats(dset, HYPER["spaceBatchSize"], train_idxs, valid_idxs, test_idxs)
      elif inputTypeRight == "clip" and not multiclips: # temporal model training on space features
        return self._batch_clipFeats(dset, HYPER["timeBatchSize"], HYPER["clipSize"], train_idx=train_idxs, valid_idx=valid_idxs, test_idx=test_idxs)
      elif inputTypeRight == "clip" and multiclips:  # temporal model training on space features
        assert HYPER["clipSize"] > 0, "clipSize must be greater than 0 for multiclips!"
        return self._batch_multiClipFeats(dset, HYPER["timeBatchSize"], HYPER["clipSize"], multiClipStride, train_idxs, valid_idxs, test_idxs)

    else:
      raise ValueError("Invalid Batch Mode!")
    
  def _batch_dataset(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
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

  def _batch_frameFeats(self, dset, size_batch, train_idxs=None, valid_idxs=None, test_idxs=None):
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

  def _batch_clipFeats(self, dset, size_batch, clip_size=0, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert os.path.exists(dset.path_spaceFeatures), f"No exported space features found in {dset.path_spaceFeatures}!"
    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False, map_location=self.DEVICE)
    firstKey = next(iter(featuresDict))
    availableLayers = list(featuresDict[firstKey].keys())
    featureLayer = availableLayers[-1]
    print(f"  Available feature layers: {availableLayers} / Using last layer: {featureLayer}\n")

    # Get per-video frame indices + split grouping
    groupedIdxs, splitVidsInt = self.get_vidsIdxsAndSplits(dset, train_idxs, valid_idxs, test_idxs)
    # Compute global max length if using full videos
    maxFrames = max(len(frames) for frames in groupedIdxs.values()) if clip_size == 0 else None
    loader = [[], [], []]
    for split, vidsInt in splitVidsInt.items():
      splitFeats, splitTargets, splitIdxs = [], [], []

      for vid in vidsInt:
        frameIdxs = groupedIdxs[vid]
        n_frames = len(frameIdxs)
        # Load data
        vidFeats = torch.stack([featuresDict[i][featureLayer] for i in frameIdxs]).squeeze(1)  # [T, F]
        vidLabels = dset.labels.iloc[frameIdxs]["frameLabels"].tolist()
        vidTargets = dset._get_targets(vidLabels, dset.labelType.split('-')[0]).to(self.DEVICE)
        vidIdxs = torch.tensor(frameIdxs, dtype=torch.int64, device=self.DEVICE)
        # Determine padding length
        if clip_size == 0:
          pad_len = maxFrames - n_frames
        else:
          pad_len = (clip_size - (n_frames % clip_size)) % clip_size
        # Pad features
        if pad_len > 0:
          featureDim = vidFeats.shape[1]
          vidFeats = torch.cat([vidFeats, torch.zeros(pad_len, featureDim, device=self.DEVICE)], dim=0)
          if len(vidTargets.shape) == 1:
            targetPadding = torch.full((pad_len,), -1, dtype=torch.int64, device=self.DEVICE)
          else:
            targetPadding = torch.zeros(pad_len, vidTargets.shape[1], dtype=vidTargets.dtype, device=self.DEVICE)
            targetPadding[:, 1] = -1
          vidTargets = torch.cat([vidTargets, targetPadding], dim=0)
          vidIdxs = torch.cat([vidIdxs, torch.full((pad_len,), -1, dtype=torch.int64, device=self.DEVICE)], dim=0)
        # Store either whole video or clips
        if clip_size == 0:
          splitFeats.append(vidFeats)          # [T, F]
          splitTargets.append(vidTargets)      # [T] or [T, C]
          splitIdxs.append(vidIdxs)            # [T]
        else:
          num_clips = vidFeats.shape[0] // clip_size
          F = vidFeats.shape[1]
          C = vidTargets.shape[1] if len(vidTargets.shape) > 1 else None
          splitFeats.extend(vidFeats.view(num_clips, clip_size, F))
          if C: # multi-label case
            splitTargets.extend(vidTargets.view(num_clips, clip_size, C))
          else:
            splitTargets.extend(vidTargets.view(num_clips, clip_size))
          splitIdxs.extend(vidIdxs.view(num_clips, clip_size))
      # Stack and create loader
      feats = torch.stack(splitFeats).cpu()
      targets = torch.stack(splitTargets).cpu()
      idxs = torch.stack(splitIdxs).cpu()

      featset = TensorDataset(feats, targets, idxs)
      loader[split] = DataLoader(featset, batch_size=size_batch, num_workers=2, shuffle=True)
    if loader[0]:
      exampleBatch = next(iter(loader[0]))
      print(f"\tBatch example shapes:\n\t[space features] {exampleBatch[0].shape}\n\t[labels] {exampleBatch[1].shape}\n\t[idxs] {exampleBatch[2].shape}")
      if clip_size == 0:
        print(f"    Batch contains {exampleBatch[0].shape[0]} video(s), each with up to {exampleBatch[0].shape[1]} frames\n")
      else:
        print(f"    Batch contains {exampleBatch[0].shape[0]} clip(s), each with {clip_size} frames\n")
    return loader

  def _batch_multiClipFeats(self, dset, size_batch, clip_size, stride=None, train_idxs=None, valid_idxs=None, test_idxs=None):
    assert clip_size > 0, "clip_size must be > 0 in multi-clip mode"
    stride = stride or clip_size
    assert os.path.exists(dset.path_spaceFeatures), f"No features at {dset.path_spaceFeatures}!"

    featuresDict = torch.load(dset.path_spaceFeatures, weights_only=False, map_location=self.DEVICE)
    featureLayer = list(featuresDict[next(iter(featuresDict))].keys())[-1]
    print(f"  [multiClip] Using feature layer: {featureLayer}\n")

    groupedIdxs, splitVidsInt = self.get_vidsIdxsAndSplits(dset, train_idxs, valid_idxs, test_idxs)
    loader = [[], [], []]

    for split, vidsInt in splitVidsInt.items():
      featsList, targetsList, idxsList = [], [], []

      for vid in vidsInt:
        frameIdxs = groupedIdxs[vid]
        n_frames = len(frameIdxs)
        if n_frames < clip_size:
          continue  # skip short videos

        vidFeats = torch.stack([featuresDict[i][featureLayer] for i in frameIdxs]).squeeze(1)
        vidLabels = dset.labels.iloc[frameIdxs]["frameLabels"].tolist()
        vidTargets = dset._get_targets(vidLabels, dset.labelType.split('-')[0]).to(self.DEVICE)
        vidIdxs = torch.tensor(frameIdxs, dtype=torch.int64, device=self.DEVICE)

        for start in range(0, n_frames - clip_size + 1, stride):
          end = start + clip_size
          featsList.append(vidFeats[start:end])
          targetsList.append(vidTargets[start:end])
          idxsList.append(vidIdxs[start:end])

      if not featsList:
        loader[split] = None
        continue

      feats = torch.stack(featsList).cpu()
      targets = torch.stack(targetsList).cpu()
      idxs = torch.stack(idxsList).cpu()
      dataset = TensorDataset(feats, targets, idxs)
      loader[split] = DataLoader(dataset, batch_size=size_batch, num_workers=2, shuffle=True)

    if loader[0]:
      xb = next(iter(loader[0]))
      print(f"\t[multiClip] Example batch:\n\t[features] {xb[0].shape} | [labels] {xb[1].shape} | [idxs] {xb[2].shape}")
      print(f"\t{xb[0].shape[0]} clips × {clip_size} frames\n")

    return loader


  def get_vidsIdxsAndSplits(self, dset, train_idxs=None, valid_idxs=None, test_idxs=None):
    '''
    Returns:
      - groupedIdxs: dict {videoId: [frameIdxs]} — all videos with their frame indices
      - splitVidsInt: dict {0|1|2: set(videoIds)} — videoIds per split (0=train, 1=valid, 2=test)
    '''
    groupedIdxs = {}
    splitVidsInt = {0: set(), 1: set(), 2: set()}
    idxMap = {0: train_idxs, 1: valid_idxs, 2: test_idxs}
    for split, idxs in idxMap.items():
      if idxs is None or len(idxs) == 0:
        continue
      groups = dset._get_grouping(idxs)
      for idx, videoId in zip(idxs, groups):
        groupedIdxs.setdefault(videoId, []).append(idx)
        splitVidsInt[split].add(videoId)
    return groupedIdxs, splitVidsInt

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