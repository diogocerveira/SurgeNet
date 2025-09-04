import gc
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
from pathlib import Path

class Phasinator(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] 1st guesses
      Outputs [output_stage_id, batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, domain, spaceinator, timeinator, classinator, id):
    super().__init__()
    self.valid_lastScore = -np.inf
    # inputType = MODEL["inputType"]
    # the modular approach allows to have modules with no conditional on the forward pass
    if domain == "space":
      backbone = [spaceinator]
      head = [classinator]
      inputType = ("images", "frame")
      arch = (spaceinator.arch, classinator.arch)
     
    elif domain == "space-time":
      if timeinator.arch == "dTsNet" or timeinator.arch == "pTsNet":
        backbone = [spaceinator, timeinator]
        head = [classinator]
        inputType = ("images", "frame")
        arch = (spaceinator.arch, timeinator.arch, classinator.arch)
        spaceinator.feed_ts = True  # Enable time feature feeding
        print(f"[DEBUG] Spaceinator will feed timestamps: {spaceinator.feed_ts}")
      else:
        assert 1 == 0, "Not implemented yet!"
    elif domain == "time":
      if timeinator.arch == "phatima":
        backbone = [timeinator]
        head = [classinator]
        assert 1 == 0, "Not implemented yet!"
      elif timeinator.arch == "tecno" or timeinator.arch == "multiTecno":
        backbone = []
        head = [timeinator]
      
      inputType = ("fmaps", "clip")
      arch = (timeinator.arch,)
    elif domain == "full":
      backbone = [spaceinator]
      head = [timeinator]
      inputType = ("images", "frame")
      arch = "full"
      assert 1 == 0, "Not implemented yet!"
    elif domain == "linear":
      backbone = []
      head = [classinator]
      inputType = ("images", "frame")
      arch = "linear"
      assert 1 == 0, "Not implemented yet!"
    else:
      raise ValueError("Invalid domain choice!")
    
    self.inators = nn.Sequential(*(backbone + head))
    self.domain = domain
    self.arch = arch
    self.inputType = inputType
    self.id = id # model name for logging/saving
    self.headType = classinator.heads

  def forward(self, x_batch):
    # print("Input shape: ", x_batch.shape)
    # print(f"[DEBUG Phasinator] input type: {type(x_batch)}")
    # if isinstance(x_batch, tuple):
    #   print(f"[DEBUG Phasinator] tuple lengths: {len(x_batch)}")
    #   print(f"[DEBUG Phasinator] first element shape: {x_batch[0].shape}")
    #   print(f"[DEBUG Phasinator] second element shape: {x_batch[1].shape}")
    # else:
    #   print(f"[DEBUG Phasinator] input shape: {x_batch.shape}")
    x_batch = self.inators(x_batch)
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
    self.arch = MODEL["spaceArch"]
    preweights = self._get_preweights(MODEL["spacePreweights"])
    transferMode = MODEL["spaceTransferMode"]
    self.model = self._get_model(self.arch, preweights, transferMode, MODEL["path_spaceModel"])  # to implement my own in model.py 
    
    self.featureSize = self.model.fc.in_features
    self.featureNode = fxs.get_graph_node_names(self.model)[0][-2] # -2 is the flatten node (virtual layer)
    self.neuron = nn.Linear(1, 1)

    if self.domain == "time" and MODEL["path_spaceModel"]:
      self.exportedFeats = torch.load(MODEL["path_spaceModel"], weights_only=False, map_location="cpu")

    self.feed_ts = False
    # print(self.model)
    # print(fxs.get_graph_node_names(self.model)[0])
    
  def forward(self, x):
    if self.feed_ts:
      x, timestamps = x  # Unpack: x → [B, C, H, W], timestamps → [B]
      # print(f"[DEBUG] Duration values shape before unsqueeze: {timestamps.shape}", flush=True)
    # print(f"Spaceinator input shape: {x.shape}", flush=True)
    x = self.model(x)  # → [B, 2048]
    # print("spaceinator output shape: ", x.shape)
    # print(f"[DEBUG] Feature map after model: {x.shape}", flush=True)
    if self.feed_ts:      # Add duration as new feature
      x = (x, timestamps)
      # print(f"[DEBUG] Concatenated features + duration: {x.shape}", flush=True)
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
    if transferMode == "feat-xtract":
      for param in model.parameters():  # freeze all layers
        param.requires_grad = False
        # parameters of newly constructed modules have requires_grad=True by default
    elif transferMode == "fine-tune":
      pass
    elif transferMode == "l4-fine-tune":
      for name, param in model.named_parameters():
        # print(f"    {name}: {param.requires_grad}")
        if "layer4" in name or "fc" in name:
          continue
        param.requires_grad = False
    elif not transferMode:
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
    if preweights == "image-net":
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
    self.model.eval()
    with torch.no_grad():
      for batch, data in enumerate(dataloader):
        images, ids = data[0].to(device), [int(id) for id in data[2]]
        features = self._extract(images, nodesToExtract)
        for layer in features:
          # print(f"{layer}: {features[layer].shape}")
          for imageId, ftrs in zip(ids, features[layer]): # Save features from chosen nodes/layers for each image ID
            out_features[imageId][layer] = ftrs.unsqueeze(0).cpu()
            # print(list(out_features.items()))
            # break
    # { "image_id": {layer: feature, ... }, ... }
    print("\nExporting features...")
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
    self.arch = MODEL["timeArch"]
    preweights = None
    transferMode = None

    self.heads = [int(head) for head in MODEL["headType"].split("-")] if MODEL["headType"] else [n_classes]
    if self.arch:
      assert sum(self.heads) == n_classes, "Incompatible head configuration"
    self.model = self._get_model(self.arch, MODEL, in_channels, self.heads)  # to implement my own in model.py 

    try:
      self.featureSize = self.model.featureSize
      self.featureNode = fxs.get_graph_node_names(self.model)[0][-1]
    except:
      self.featureSize = None
      self.featureNode = None

    if self.domain == "class" and MODEL["path_timeModel"]:
      self.exportedFeats = torch.load(MODEL["path_spaceModel"], weights_only=False, map_location="cpu")
    # print(self.model)
    # print(fxs.get_graph_node_names(self.model)[0])
  
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
      model = Phatima(MODEL, in_channels, n_classes[0])
    elif arch == "tecnoOG":
      model = TecnoOG(MODEL, in_channels, n_classes[0])
    elif arch == "tecno2":
      model = Tecno2(MODEL, in_channels, n_classes[0])
    elif arch == "multiTecno":
      assert len(n_classes) > 1, "Multi-label architecture requires multiple classes values"
      model = MultiLabelTecno(MODEL, in_channels, n_classes)
    elif arch == "dTsNet" or arch == "pTsNet":
      model = TsNet(MODEL, in_channels)
    elif not arch:
      model = nn.Identity()
    else:
      raise ValueError("Invalid time architecture choice!")
    return model
  def export_features(self, dataloader, path_export, nodesToExtract, device):
    # _features features maps to external file
    self.model.fc = nn.Identity()  # Remove classification layer
    out_features = defaultdict(dict)  # default: ddict of features (at multiple nodes/layers) for each frame
    self.model.eval()
    with torch.no_grad():
      for batch, data in enumerate(dataloader):
        images, ids = data[0].to(device), [int(id) for id in data[2]]
        features = self._extract(images, nodesToExtract)
        for layer in features:
          # print(f"{layer}: {features[layer].shape}")
          for imageId, ftrs in zip(ids, features[layer]): # Save features from chosen nodes/layers for each image ID
            out_features[imageId][layer] = ftrs.unsqueeze(0).cpu()
            # print(list(out_features.items()))
            # break
    # { "image_id": {layer: feature, ... }, ... }
    print("\nExporting features...")
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

    stage1 = _PhatimaStage(in_channels, N_FILTERS, N_BLOCKS, N_FILTERS, TF_FILTER, dilation=1)
    stage2 = _PhatimaStage(N_FILTERS, N_FILTERS * 2, N_BLOCKS, N_FILTERS * 2, TF_FILTER, dilation=2)
    stage3 = _PhatimaStage(N_FILTERS * 2, N_FILTERS * 4, N_BLOCKS, N_FILTERS * 4, TF_FILTER, dilation=4)
    stage4 = _PhatimaStage(N_FILTERS * 4, N_FILTERS * 8, N_BLOCKS, N_FILTERS * 8, TF_FILTER, dilation=8)

    self.stages = nn.ModuleList([stage1, stage2, stage3, stage4])
    self.featureSize = N_FILTERS * 8 # (batchSize, n_classes)

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

class _PhatimaStage(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks, n_filters, tf_filter, dilation=1):
    super().__init__()
    self.conv1x1_in = nn.Conv1d(in_channels, n_filters, kernel_size=1) # adapts fv channel dimension
    # self.blocks = nn.ModuleList([copy.deepcopy(_TCResBlock(n_filters, n_filters, tf_filter, 2**b)) for b in range(n_blocks)])
    self.ProjectionResBlock = _ProjectionResBlock(in_channels, n_filters, tf_filter, dilation)
    self.IdentityResBlock = _IdentityResBlock(n_filters, n_filters, tf_filter, dilation)
    self.blocks = nn.ModuleList([self.ProjectionResBlock] + [copy.deepcopy(self.IdentityResBlock) for _ in range(n_blocks - 1)])
    self.conv1x1_out = nn.Conv1d(n_filters, out_channels, kernel_size=1)
  def forward(self, x):
    # print("Before conv1x1_in Shape: ", x.shape)
    x = self.conv1x1_in(x)
    # print("After conv1x1_in Shape: ", x.shape)
    for block in self.blocks:
      x = block(x)
    # print("After blocks Shape: ", x.shape)
    x = self.conv1x1_out(x)
    # print("After conv1x1_out Shape: ", x.shape)
    return x    


class Tecno2(nn.Module):
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()

    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"] 
    self.N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.n_classes = n_classes # actually they're phases
    self.bneck_channels = MODEL["n_visuals"]
    self.predStage = _TecnoStageV2(in_channels, self.bneck_channels, n_classes, N_BLOCKS, self.N_FILTERS, TF_FILTER)  # stage for prediction
    self.refStages = nn.ModuleList([copy.deepcopy(_TecnoStage(n_classes, n_classes, N_BLOCKS, self.N_FILTERS, TF_FILTER)) for s in range(N_STAGES - 1)])
    self.featureSize = n_classes # (batchSize, n_classes)

  def forward(self, x):
    # x: [B, T, Cfeat]  OR  [B, Cclips, Tclip, Cfeat]
    if x.dim() == 4:                      # clip mode
      B, N, T, C = x.shape
      x = x.reshape(B, N*T, C)    # [B, T_total, Cfeat]
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    elif x.dim() == 3:                    # full-video mode
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    else:
      raise ValueError(f"bad input shape {x.shape}")

    # x = x.transpose(1, 2)             # [B, C, T] for conv layers
    # prediction head
    headOut = self.predStage(x)           # [B, C, T]
    outputs = [headOut.permute(0, 2, 1)]  # [B, T, C]
    # refinement heads
    for stage in self.refStages:
      headOut = stage(F.softmax(headOut, dim=1))   # still [B, C, T]
      outputs.append(headOut.permute(0, 2, 1)) # convert to [B, T, C]
    return tuple(outputs)   # (head0, head1, ...)

class TecnoOG(nn.Module):
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()

    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"] 
    self.N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.n_classes = n_classes # actually they're phases
    self.bneck_channels = MODEL["n_visuals"]
    self.predStage = _TecnoStage(in_channels, n_classes, N_BLOCKS, self.N_FILTERS, TF_FILTER)  # stage for prediction
    self.refStages = nn.ModuleList([copy.deepcopy(_TecnoStage(n_classes, n_classes, N_BLOCKS, self.N_FILTERS, TF_FILTER)) for s in range(N_STAGES - 1)])
    self.featureSize = n_classes # (batchSize, n_classes)

  def forward(self, x):
    # x: [B, T, Cfeat]  OR  [B, Cclips, Tclip, Cfeat]
    if x.dim() == 4:                      # clip mode
      B, N, T, C = x.shape
      x = x.reshape(B, N*T, C)    # [B, T_total, Cfeat]
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    elif x.dim() == 3:                    # full-video mode
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    else:
      raise ValueError(f"bad input shape {x.shape}")

    # x = x.transpose(1, 2)             # [B, C, T] for conv layers
    # prediction head
    headOut = self.predStage(x)           # [B, C, T]
    outputs = [headOut.permute(0, 2, 1)]  # [B, T, C]
    # refinement heads
    for stage in self.refStages:
      headOut = stage(F.softmax(headOut, dim=1))   # still [B, C, T]
      outputs.append(headOut.permute(0, 2, 1)) # convert to [B, T, C]
    return tuple(outputs)   # (head0, head1, ...)

class MultiLabelTecno(nn.Module):
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()
    assert len(n_classes) > 1, "More than one n_classes values is required"
    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"] 
    self.N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.n_classes = n_classes
    self.predStage = _MultiLabelTecnoStage(in_channels, n_classes, N_BLOCKS, self.N_FILTERS, TF_FILTER)  # stage for prediction

    self.refStageLeft = nn.ModuleList([copy.deepcopy(_TecnoStage(n_classes[0], n_classes[0], N_BLOCKS, self.N_FILTERS, TF_FILTER)) for s in range(N_STAGES - 1)])
    self.refStageRight = nn.ModuleList([copy.deepcopy(_TecnoStage(n_classes[1], n_classes[1], N_BLOCKS, self.N_FILTERS, TF_FILTER)) for s in range(N_STAGES - 1)])
    self.featureSize = n_classes # (batchSize, n_classes)

  def forward(self, x):
    # x: [B, T, Cfeat]  OR  [B, Cclips, Tclip, Cfeat]
    if x.dim() == 4:                      # clip mode
      B, N, T, C = x.shape
      x = x.reshape(B, N*T, C)    # [B, T_total, Cfeat]
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    elif x.dim() == 3:                    # full-video mode
      x = x.permute(0, 2, 1).contiguous()     # [B, Cfeat, T_total]
    else:
      raise ValueError(f"bad input shape {x.shape}")
    # x: [B, T, C] -> convs need [B, C, T]
    # x = x.transpose(1, 2)

    # stage 0 (prediction)
    predHeadOuts = self.predStage(x)                     # ( [B,C1,T], [B,C2,T] )
    leftHead  = [predHeadOuts[0].permute(0, 2, 1)]       # [B,T,C1]
    rightHead = [predHeadOuts[1].permute(0, 2, 1)]       # [B,T,C2]

    # refinement: chain from the previous stage of the SAME head
    predLeft, predRight = predHeadOuts                     # logits
    for stageLeft, stageRight in zip(self.refStageLeft, self.refStageRight):
      curLeft  = stageLeft(F.softmax(predLeft,  dim=1))   # -> logits [B,C1,T]
      curRight = stageRight(F.softmax(predRight, dim=1))  # -> logits [B,C2,T]
      leftHead.append(curLeft.permute(0, 2, 1))          # [B,T,C1]
      rightHead.append(curRight.permute(0, 2, 1))        # [B,T,C2]

    return (tuple(leftHead), tuple(rightHead))           # len==2 (heads); each is tuple over stages


class _TecnoStageV2(nn.Module):
  def __init__(self, in_channels, bneck_channels, out_channels, n_blocks, n_filters, tf_filter, dilation=1):
    super().__init__()
    self.conv1x1_in = nn.Conv1d(in_channels, bneck_channels, kernel_size=1) # adapts fv channel dimension
    self.bneckBlock = _ProjectionResBlock(bneck_channels, n_filters, tf_filter, 1)
    self.blocks = nn.ModuleList([copy.deepcopy(_IdentityResBlock(n_filters, n_filters, tf_filter, 2**b)) for b in range(1, n_blocks)])
    self.conv1x1_out = nn.Conv1d(n_filters, out_channels, kernel_size=1)
  def forward(self, x):
    # print("Before conv1x1_in Shape: ", x.shape)
    x = self.conv1x1_in(x)
    # print("After conv1x1_in Shape: ", x.shape)
    x = self.bneckBlock(x)
    # print("After bneckBlock Shape: ", x.shape)
    for block in self.blocks:
      x = block(x)
    # print("After blocks Shape: ", x.shape)
    x = self.conv1x1_out(x)
    # print("After conv1x1_out Shape: ", x.shape)
    return x

class _TecnoStage(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks, n_filters, tf_filter, dilation=1):
    super().__init__()
    self.conv1x1_in = nn.Conv1d(in_channels, n_filters, kernel_size=1) # adapts fv channel dimension
    self.blocks = nn.ModuleList([copy.deepcopy(_IdentityResBlock(n_filters, n_filters, tf_filter, 2**b)) for b in range(n_blocks)])
    self.conv1x1_out = nn.Conv1d(n_filters, out_channels, kernel_size=1)

  def forward(self, x):
    # print("Before conv1x1_in Shape: ", x.shape)
    x = self.conv1x1_in(x)
    # print("After conv1x1_in Shape: ", x.shape)
    for block in self.blocks:
      x = block(x)
    # print("After blocks Shape: ", x.shape)
    x = self.conv1x1_out(x)
    # print("After conv1x1_out Shape: ", x.shape)
    return x

class _MultiLabelTecnoStage(nn.Module):
  def __init__(self, in_channels, out_channels, n_blocks, n_filters, tf_filter, dilation=1):
    super().__init__()
    assert len(out_channels) > 1, "More than one output channel values is required"
    self.conv1x1_in = nn.Conv1d(in_channels, n_filters, kernel_size=1) # adapts fv channel dimension
    self.blocks = nn.ModuleList([copy.deepcopy(_IdentityResBlock(n_filters, n_filters, tf_filter, 2**b)) for b in range(n_blocks)])
    self.conv1x1_outLeft = nn.Conv1d(n_filters, out_channels[0], kernel_size=1)
    self.conv1x1_outRight = nn.Conv1d(n_filters, out_channels[1], kernel_size=1)
  def forward(self, x):
    # print("Before conv1x1_in Shape: ", x.shape)
    x = self.conv1x1_in(x)
    # print("After conv1x1_in Shape: ", x.shape)
    for block in self.blocks:
      x = block(x)
    # print("After blocks Shape: ", x.shape)
    x1 = self.conv1x1_outLeft(x)
    x2 = self.conv1x1_outRight(x)
    # print("After conv1x1_out Shape: ", x.shape)
    return (x1, x2) # double output

class _IdentityResBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, kernelSize, dilation):
    super().__init__()
    padding = (kernelSize - 1) * dilation // 2
    self.convDilated = nn.Conv1d(in_channels, out_channels, kernel_size=kernelSize, padding=padding, dilation=dilation)
    self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.dropout = nn.Dropout()
    # self.batchNorm = nn.BatchNorm1d(out_channels)  # Batch Normalization for stability
  def forward(self, x):
    # x becomes the residual
    out = self.convDilated(x)
    # out = self.batchNorm(out)
    out = F.relu(out)
    # out = self.dropout(out)
    out = self.conv1x1(out)
    out = self.dropout(out)
    return out + x

class _ProjectionResBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, kernelSize, dilation):
    super().__init__()
    padding = (kernelSize - 1) * dilation // 2
    self.conv1x1_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.convDilated = nn.Conv1d(in_channels, out_channels, kernel_size=kernelSize, padding=padding, dilation=dilation)
    self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.dropout = nn.Dropout()
    # self.batchNorm = nn.BatchNorm1d(out_channels)  # Batch Normalization for stability
  
  def forward(self, x):
    out = self.convDilated(x)
    # out = self.batchNorm(out)
    out = F.relu(out)
    out = self.dropout(out)
    x = self.conv1x1_proj(x)  # project input to output channels
    return out + x
  
class TsNet(nn.Module):
  def __init__(self, MODEL, in_channels):
    super().__init__()
    self.projChannels = MODEL["proj_channels"] if MODEL["proj_channels"] else 5
    
    if MODEL["timeArch"] == "pTsNet":
      self.tsFeed = nn.Sequential(
        nn.Linear(1, self.projChannels),
        nn.ReLU(),
        nn.Linear(self.projChannels, 1)
      )
    elif MODEL["timeArch"] == "dTsNet":
      self.tsFeed = nn.Identity()
    else:
      raise ValueError("Invalid TsNet type!")
    self.featureSize = 1 + in_channels  # assuming input features are of size 2048 (e.g., from ResNet50)
  def forward(self, x):
    x, ts = x  # Unpack: x → [B, F], ts → [B]
    # print("x.shape: ", x.shape, "timestamps.shape: ", ts.shape)
    ts = ts.unsqueeze(1)  # [B] → [B, 1]
    # print("unsqueezed.shape: ", ts.shape)
    ts = self.tsFeed(ts)
    # print("ts.shape: ", ts.shape)
    x = torch.cat((x, ts), dim=1)  # [B, F] + [B, 1] → [B, F+1]
    # print("After concat shape: ", x.shape)
    return x
  

class Classinator(nn.Module):
  def __init__(self, MODEL, spaceFeatureSize, timeFeatureSize, N_CLASSES):
    super().__init__()
    self.FEATURE_SIZE = spaceFeatureSize if MODEL["domain"] in ["space", "spacetime"] else timeFeatureSize
    self.arch = MODEL["classifierArch"]
    self.heads = [int(head) for head in MODEL["headType"].split("-")] if MODEL["headType"] else [N_CLASSES]
    if self.arch:
      assert sum(self.heads) == N_CLASSES, "Incompatible head configuration"
    self.model = self._get_model(self.arch, MODEL, self.FEATURE_SIZE, self.heads)  # to implement my own in model.py

  def forward(self, x):
    # print("Before Classinator Shape: ", x.shape)
    x = self.model(x)
    # print("After Classinator Shape: ", x.shape)
    return x

  def _extract(self, images, nodesToExtract):
    # Extract features at specified nodes
    return fx.create_feature_extractor(self.model, return_nodes=nodesToExtract)(images)
  def _get_model(self, arch, MODEL, in_channels, n_classes):
    # Choose model architecture - backbones
    if arch == "one-linear":
      assert len(n_classes) == 1, "Single head classifier expects single integer for n_classes"
      model = SingleHeadLinearClassifier(in_channels, n_classes[0], MODEL)
    elif arch == "two-linear":
      assert len(n_classes) == 2, "Two head classifier expects tuple of size two for n_classes"
      model = TwoHeadLinearClassifier(in_channels, n_classes, MODEL)
    elif not arch:
      model = nn.Identity()
    else:
      raise ValueError("Invalid classifier architecture choice!")
    return model
  
class SingleHeadLinearClassifier(nn.Module):
  def __init__(self, featureSize, n_classes, MODEL=None):
    super().__init__()
    self.linear = nn.Linear(featureSize, n_classes)

  def forward(self, x):
    x = self.linear(x)  # → [B, N_CLASSES]
    return x
  
class TwoHeadLinearClassifier(nn.Module):
  def __init__(self, featureSize, n_classes, MODEL=None):
    super().__init__()
    n_classes1, n_classes2 = n_classes # tuple of size two
    self.linear1 = nn.Linear(featureSize, n_classes1)
    self.linear2 = nn.Linear(featureSize, n_classes2)

  def forward(self, x):
    x1 = self.linear1(x)  # → [B, N_CLASSES]
    x2 = self.linear2(x)
    return (x1, x2)