import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import os
from collections import defaultdict, OrderedDict
import torchvision.models.feature_extraction as fx
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import copy
import torchvision.models.feature_extraction as fxs
import numpy as np
import math

class Phasinator(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] 1st guesses
      Outputs [output_stage_id, batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, modelDomain, spaceinator, timeinator, classroomId, fold):
    super().__init__()
    self.valid_lastScore = -np.inf
    
    # the modular approach allows to have modules with no conditional on the forward pass
    if modelDomain == "spatial":
      classifier = nn.Linear(spaceinator.featureSize, spaceinator.n_classes)
      inators = [spaceinator, classifier]
      inputType = "images-framed"
    elif modelDomain == "temporal":
      classifier = nn.Linear(timeinator.featureSize, timeinator.n_classes)
      inators = [timeinator, classifier]
      inputType = "fmaps-clipped"
    elif modelDomain == "temporal-frames":
      classifier = nn.Linear(timeinator.featureSize, timeinator.n_classes)
      inators = [timeinator, classifier]
      inputType = "fmaps-clipped"
    elif modelDomain == "tecno-like":
      inators = [timeinator]
      inputType = "fmaps-clipped"
    elif modelDomain == "full":
      classifier = nn.Linear(timeinator.featureSize, timeinator.n_classes)
      inators = [spaceinator, timeinator, classifier]
      inputType = "images-framed"
    elif modelDomain == "spaceClassifier":
      inators = [nn.Linear(spaceinator.featureSize, spaceinator.n_classes)]
      inputType = "fmaps-framed"
    elif modelDomain == "timeClassifier":
      inators = [nn.Linear(timeinator.featureSize, timeinator.n_classes)]
      inputType = "fmaps-clipped"
    else:
      raise ValueError("Invalid domain choice!")
    self.model = nn.Sequential(*inators)
    self.inputType = inputType
    self.id = f"{modelDomain}phase-{classroomId}_f{fold}" # model name for logging/saving

  def forward(self, x_batch):
    # print("Input shape: ", x_batch.shape)
    x_batch = self.model(x_batch)
    # print("Output shape: ", x_batch.shape)
    return x_batch

class Spaceinator(nn.Module):
  ''' 2D Convolutional network object, able to be trained for feature extraction,
      _features them or load presaved ones
  '''
  def __init__(self, MODEL, n_classes):
    super().__init__()
    self.domain = MODEL["domain"]
    self.n_classes = n_classes
    arch = MODEL["spaceArch"]
    preweights = self._get_preweights(MODEL["spacePreweights"])
    transferMode = MODEL["spaceTransferLearning"]
    self.model = self._get_model(arch, preweights, transferMode, MODEL["path_spaceModel"])  # to implement my own in model.py 
    self.featureSize = self.model.fc.in_features
    self.featureNode = fxs.get_graph_node_names(self.model)[0][-2] # -2 is the flatten node (virtual layer)

    if self.domain == "temporal" and MODEL["path_spaceModel"]:
      self.exportedFeatures = torch.load(MODEL["path_spaceModel"], weights_only=False, map_location="cpu")
    
    # print(self.model)
    # print(fxs.get_graph_node_names(self.model)[0])
    
  def forward(self, x):
    x = self.model(x)
    # print("After Spaceinator Shape: ", x.shape)
    return x
  
  def _get_model(self, arch, preweights, transferMode, path_model):
    # Choose model architecture - backbones
    if arch == "resnet50":
      model = resnet50(weights=preweights)
    elif arch == "resnet100":
      pass
    else:
      raise ValueError("Invalid space architecture choice!")
    
    # transfer learning mode
    n_features = model.fc.in_features  # 2048 for resnet50
    if transferMode == "fixed-fx":
      for param in model.parameters():  # freeze all layers
        param.requires_grad = False
        # parameters of newly constructed modules have requires_grad=True by default
    elif transferMode == "fine-tune":  
      pass
    else:
      raise ValueError("Invalid transfer learning mode!")
    # remove final layer if passing features to another model (not learning them)
    model.fc = nn.Identity()
    model.fc.in_features = n_features # otherwise Identity does not have in_features attribute
    # if self.domain == "spatial":
    #   model.fc = nn.Linear(n_features, self.n_classes)
    # elif self.domain == "temporal" or self.domain == "full":
    #   model.fc = nn.Identity()  # Remove classification layer, assumed already trained or not the end of the network
    #   model.fc.in_features = n_features # ?
    
    # If provided with a model state, load it
    stateDict = None
    if path_model and os.path.isfile(path_model):
      try:
        print(f"    Loading state from 'path_spaceModel' to {arch} in {transferMode} mode.")
        stateDict = torch.load(path_model, weights_only=False, map_location="cpu")
      except Exception as e:
        print(f"     Error loading model from {path_model}: {e}")
    else:
      print(f"    No 'path_spaceModel' found. Using pretrained {arch} in {transferMode} mode.")
    if stateDict:  # Load saved model if provided
      model.load_state_dict(stateDict, strict=False)

    return model
  
  def _get_preweights(self, preweights):
    if preweights == "default":
      return ResNet50_Weights.DEFAULT # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    elif not preweights:
      return None
    else:
      raise ValueError("Invalid preweight choice!")
  def get_featureSize(self, x):
    # to implement
    pass

  def log_features(self, writer, features, layer, im_id, fold):
    # print(features.shape)
    if features.dim() == 4:  # If it's a 4D tensor (N, C, H, W)
      for i in range(features.shape[0]):  # Iterate through each image in the batch
        writer.add_images(f"{layer}/{im_id}", features[i], fold)
    elif features.dim() == 3:  # If it's a 3D tensor (C, H, W)
      for i in range(features.shape[0]):
        writer.add_image(f"{layer}/{im_id}/fm{i}", features[i].unsqueeze(0), fold)
        break
    elif features.dim() == 1:  # If it's a 1D tensor (features)
      writer.add_image(f"{layer}/{im_id}", features.unsqueeze(0).unsqueeze(0), fold)  # Add row col dims

  def _extract(self, images, nodesToExtract):
    # Extract features at specified nodes
    return fx.create_feature_extractor(self.model, return_nodes=nodesToExtract)(images)

  def export_features(self, dataloader, path_export, nodesToExtract, device):
    # _features features maps to external file
    self.model.fc = nn.Identity()  # Remove classification layer
    out_features = defaultdict(dict)  # default: ddict of features (at multiple nodes/layers) for each frame
    self.model.eval().to(device) 
    with torch.no_grad():
      for batch, data in enumerate(dataloader):
        images, ids = data[0].to(device), [int(id) for id in data[2]]
        features = self._extract(images, nodesToExtract)
        for layer in features:
          # print(f"{layer}: {features[layer].shape}")
          for imageId, ftrs in zip(ids, features[layer]): # Save features from chosen nodes/layers for each image ID
            out_features[imageId][layer] = ftrs.unsqueeze(0).to(device)
            # print(list(out_features.items()))
            # break
    # { "image_id": {layer: feature, ... }, ... }
    torch.save(out_features, path_export)
    # print(out_features)
    print(f"* Exported space features as {os.path.basename(path_export)} *\n")
  
  @staticmethod
  def get_defaultDict(): # torch.save() can't pickle whatever if nested defaultdict uses lambda (dedicated function instead) 
    return defaultdict(None)

class Timeinator(nn.Module):
  ''' 1D Convolutional network object, able to be trained for feature extraction,
      _features them or load presaved ones
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()
    self.domain = MODEL["domain"]
    self.n_classes = n_classes
    arch = MODEL["timeArch"]
    preweights = None
    transferMode = None
    self.model = self._get_model(arch, MODEL, in_channels, n_classes)  # to implement my own in model.py 
    self.out_featureSize = self.model.out_featureSize
    self.featureNode = fxs.get_graph_node_names(self.model)[0][-2] # -2 is the flatten node (virtual layer)

    if self.domain == "classifier" and MODEL["path_timeModel"]:
      self.exportedFeatures = torch.load(MODEL["path_spaceModel"], weights_only=False, map_location="cpu")
  
  def forward(self, x):
    x = self.model(x)
    # print("After Timeinator Shape: ", x.shape)
    return x

  def _extract(self, images, nodesToExtract):
    # Extract features at specified nodes
    return fx.create_feature_extractor(self.model, return_nodes=nodesToExtract)(images)
  def _get_model(self, arch, MODEL, in_channels, n_classes):
    # Choose model architecture - backbones
    if arch == "phatima":
      model = Phatima(MODEL, in_channels, n_classes)
    elif arch == "tecno":
      model = Tecno(MODEL, in_channels, n_classes)
    else:
      raise ValueError("Invalid time architecture choice!")
    return model
  def export_features(self, dataloader, path_export, nodesToExtract, device):
    # _features features maps to external file
    self.model.fc = nn.Identity()  # Remove classification layer
    out_features = defaultdict(dict)  # default: ddict of features (at multiple nodes/layers) for each frame
    self.model.eval().to(device) 
    with torch.no_grad():
      for batch, data in enumerate(dataloader):
        images, ids = data[0].to(device), [int(id) for id in data[2]]
        features = self._extract(images, nodesToExtract)
        for layer in features:
          # print(f"{layer}: {features[layer].shape}")
          for imageId, ftrs in zip(ids, features[layer]): # Save features from chosen nodes/layers for each image ID
            out_features[imageId][layer] = ftrs.unsqueeze(0).to(device)
            # print(list(out_features.items()))
            # break
    # { "image_id": {layer: feature, ... }, ... }
    torch.save(out_features, path_export)
    # print(out_features)
    print(f"* Exported space features as {os.path.basename(path_export)} *\n")


class Phatima(nn.Module):
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()

    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"] 
    N_BLOCKS = self.min_blocks_for_rf(17776, 4)
    N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.n_classes = n_classes

    stage1 = _TCStage(in_channels, N_FILTERS, N_BLOCKS, N_FILTERS, TF_FILTER, dilation=1)
    stage2 = _TCStage(N_FILTERS, N_FILTERS * 2, N_BLOCKS, N_FILTERS * 2, TF_FILTER, dilation=2)
    stage3 = _TCStage(N_FILTERS * 2, N_FILTERS * 4, N_BLOCKS, N_FILTERS * 4, TF_FILTER, dilation=4)
    stage4 = _TCStage(N_FILTERS * 4, N_FILTERS * 8, N_BLOCKS, N_FILTERS * 8, TF_FILTER, dilation=8)

    self.stages = nn.ModuleList([stage1, stage2, stage3, stage4])
    self.out_featureSize = N_FILTERS * 8 # (batchSize, n_classes)

  def forward(self, x):
    # print("In Shape: ", x.shape)
    x = x.transpose(1, 2)  # [batch_size, in_length, in_channels] -> [batch_size, in_channels, in_length]
    # x = self.conv1x1_in(x)
    for stage in self.stages:
      x = stage(x) # + x # add identity residual connection
    x = x.transpose(1, 2)
    # print("Out Shape: ", x.shape)
    return x
  def min_blocks_for_rf(self, L, k):
    a = math.ceil(math.log2(((L - 1) / (2 * (k - 1))) + 1))
    return int(a)
    
class Tecno(nn.Module):
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()

    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"] 
    self.N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.n_classes = n_classes

    self.conv1x1_in = nn.Conv1d(in_channels, self.N_FILTERS, kernel_size=1)
    self.predStage = copy.deepcopy(_TCStage(in_channels, N_BLOCKS, N_FILTERS, TF_FILTER, n_classes))  # stage for prediction
    self.refStages = nn.ModuleList([copy.deepcopy(_TCStage(n_classes, N_BLOCKS, N_FILTERS, TF_FILTER, n_classes)) for s in range(N_STAGES - 1)])

    self.stages = nn.ModuleList([copy.deepcopy(_TCStage(self.N_FILTERS, self.N_FILTERS, N_BLOCKS, self.N_FILTERS, TF_FILTER)) for s in range(N_STAGES)])
    self.out_featureSize = self.n_classes # (batchSize, n_classes)

  def forward(self, x):
    # print("Before predStage Shape: ", x.shape)
    x = x.transpose(1, 2)  # [batch_size, in_length, in_channels] -> [batch_size, in_channels, in_length]
    # print("After transpose Shape: ", x.shape)
    x = self.predStage(x) # first pred logits
    # print("After predStage Shape: ", x.shape)
    x_acm = x.unsqueeze(0)  # add dimension for accumulation stage results
    for stage in self.refStages:
      x = stage(F.softmax(x, dim=1)) # get probabilities from logits
      x_acm = torch.cat((x_acm, x.unsqueeze(0)), dim=0) # accumulate stage results
    # print("After conv1x1_out Shape: ", x.shape)
    # print(list(x_acm)[0], list(x_acm)[-1])  # print accumulated results for debugging
    return x # x_acm for more in depth (returning last state guess)

class _TCStage(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks, n_filters, tf_filter, dilation=1):
    super().__init__()
    # self.conv1x1_res = nn.Conv1d(in_channels, out_channels, kernel_size=1) # adapts fv channel dimension
    # self.blocks = nn.ModuleList([copy.deepcopy(_TCResBlock(n_filters, n_filters, tf_filter, 2**b)) for b in range(n_blocks)])
    self.ProjectionResBlock = _ProjectionResBlock(in_channels, n_filters, tf_filter, dilation)
    self.IdentityResBlock = _IdentityResBlock(n_filters, n_filters, tf_filter, dilation)
    self.blocks = nn.ModuleList([self.ProjectionResBlock] + [copy.deepcopy(self.IdentityResBlock) for _ in range(n_blocks - 1)])
    # self.conv1x1_out = nn.Conv1d(n_filters, out_channels, kernel_size=1)
  def forward(self, x):
    # print("Before conv1x1_in Shape: ", x.shape)
    # x = self.conv1x1_in(x)
    # print("After conv1x1_in Shape: ", x.shape)
    for block in self.blocks:
      x = block(x)
    # print("After blocks Shape: ", x.shape)
    # x = self.conv1x1_out(x)
    # print("After conv1x1_out Shape: ", x.shape)
    return x
  
class _IdentityResBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, kernelSize, dilation):
    super().__init__()
    self.convDilated = nn.Conv1d(in_channels, out_channels, kernel_size=kernelSize, padding=dilation, dilation=dilation)
    # self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.dropout = nn.Dropout()
    self.batchNorm = nn.BatchNorm1d(out_channels)  # Batch Normalization for stability
  def forward(self, x):
    # x becomes the residual
    # out = self.convDilated(x)
    # out = self.batchNorm(out)
    # out = F.relu(out)
    out = self.convDilated(x)
    out = self.batchNorm(out)
    out = F.relu(out)
    # out = self.dropout(out)
    # out = F.relu(self.convDilated(x))
    # out = self.conv1x1(out)
    # out = self.dropout(out)
    return out + x

class _ProjectionResBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, kernelSize, dilation):
    super().__init__()
    self.conv1x1_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.convDilated = nn.Conv1d(in_channels, out_channels, kernel_size=kernelSize, padding=dilation, dilation=dilation)
    self.dropout = nn.Dropout()
    self.batchNorm = nn.BatchNorm1d(out_channels)  # Batch Normalization for stability
  
  def forward(self, x):
    out = self.convDilated(x)
    out = self.batchNorm(out)
    out = F.relu(out)
    # out = self.dropout(out)
    x = self.conv1x1_proj(x)  # project input to output channels
    return out + x

class Classifier(nn.Module):
  def __init__(self, FEATURE_SIZE, N_CLASSES):
    super().__init__()
    self.fc = nn.Linear(FEATURE_SIZE, N_CLASSES) # 2048 for resnet50
    self.softmax = nn.Softmax(dim=1) # dim 0 is batch position and dim 1 is the logits
    self.model = nn.Sequential(self.fc, self.softmax)

  def forward(self, x):
    return self.model(x)