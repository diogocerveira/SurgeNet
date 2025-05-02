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

class PhaseNet(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] 1st guesses
      Outputs [output_stage_id, batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, modelDomain, spaceinator, timeinator, classroomId, fold):
    super().__init__()
    self.valid_score = 0.0

    # the modular approach allows to have modules with no conditional on the forward pass
    if modelDomain == "spatial":
      inators = [spaceinator]
      inputType = "images"
    elif modelDomain == "temporal":
      inators = [timeinator]
      inputType = "fmaps"
    elif modelDomain == "full":
      inators = [spaceinator, timeinator]
      inputType = "images"
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
      Modes - classify: train and classify in runtime
              _features: train and _features features in runtime
              load: load presaved features and forward them
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
    # print(self.model.parameters())
    # print(train_nodes == eval_nodes, end='\t'); print(train_nodes)
    
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
    if self.domain == "spatial":
      model.fc = nn.Linear(n_features, self.n_classes)
    elif self.domain == "temporal" or self.domain == "full":
      model.fc = nn.Identity()  # Remove classification layer, assumed already trained or not the end of the network
      model.fc.in_features = n_features # ?
    
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
  
  # def classify(self, N_CLASSES):
    # self.model = get_resnet50(N_CLASSES, model_weights=modelPreWeights, extract_features=False)  # Keeps last classifier layer

  def load(self, path_from, layer=None):
    layer = self.featureNode
    # print(layer, path_from)
    features = torch.load(os.path.join(path_from, os.listdir(path_from)[0]), weights_only=False, map_location="cpu") # single item on the dir [0]
    # print(features.keys())
    
    for i in features.keys():
      # print(i)
      random_f = features[i]
      # print(random_f.keys())
      # print(random_f)
      # break
    example_feature = random_f[layer] # defaultdict of defaultdicts
    self.features = features
    self.featureSize = example_feature.size(0) # assuming every value has the same dim
    print("\nE.g. Feature: ", example_feature.squeeze(-1).squeeze(-1))
    print("Features Size: ", self.featureSize, '\n')

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
          print(f"{layer}: {features[layer].shape}")
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
  ''' Receives [batch_size, in_channels == feature size, in_length == length of clip of vid] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, in_channels, n_classes):
    super().__init__()

    N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"]
    N_KERNELS = MODEL["n_kernels"]
    TF_FILTER = MODEL["tf_filter"]
    N_CLASSES = n_classes
    self.conv1x1_in = nn.Conv1d(in_channels, N_KERNELS, kernel_size=1)
    self.stages = nn.ModuleList([copy.deepcopy(_TCStage(N_BLOCKS, N_KERNELS, TF_FILTER)) for s in range(N_STAGES)])
    self.conv1x1_out = nn.Conv1d(N_KERNELS, N_CLASSES, kernel_size=1)
    self.featureSize = N_KERNELS

  def forward(self, x):
    # print("Before Timeinator Shape: ", x.shape)
    x = self.conv1x1_in(x)  # adapt number of channels (features) to the number of kernels per filter
    # print("After conv1x1_in Shape: ", x.shape)
    x_acm = x.unsqueeze(0)  # add dimension for accumulation stage results
    for stage in self.stages:
      x = stage(x)  # apply the stage (blocks of dilated convolutions) - keeps the same number of channels
      # x = F.softmax(self.conv_out(x), dim=1)  # get predictions from logits
      x_acm = torch.cat((x_acm, x.unsqueeze(0)), dim=0) # accumulate stage results
    x = self.conv1x1_out(x)  # adapt number of channels (features) to the number of classes
    # print("After conv1x1_out Shape: ", x.shape)
    return x # x_accm for more in depth (returning last state guess)

class _TCStage(nn.Module):
  def __init__(self, n_blocks, n_kernels, tf_filter):
    super().__init__()
    # self.conv1x1_in = nn.Conv1d(in_channels, n_kernels, kernel_size=1) # adapts fv channel dimension
    self.blocks = nn.ModuleList([copy.deepcopy(_TCResBlock(n_kernels, n_kernels, tf_filter, 2**b)) for b in range(n_blocks)])
    # self.conv1x1_out = nn.Conv1d(n_kernels, out_channels, kernel_size=1)
  def forward(self, x):
    # x = self.conv1x1_in(x)
    for block in self.blocks:
      x = block(x)
    # x = self.conv1x1_out(x)
    return x
  
class _TCResBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, kernelSize, dilation):
    super().__init__()
    self.convDilated = nn.Conv1d(in_channels, out_channels, kernel_size=kernelSize, padding=dilation, dilation=dilation)
    self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time frame 1: mix learned features
    self.dropout = nn.Dropout()
  
  def forward(self, x):
    # x becomes the residual
    out = F.relu(self.convDilated(x))
    # out = self.conv1x1(x)
    # out = self.dropout(x)
    return out + x

class Classifier(nn.Module):
  def __init__(self, FEATURE_SIZE, N_CLASSES):
    super().__init__()
    self.fc = nn.Linear(FEATURE_SIZE, N_CLASSES) # 2048 for resnet50
    self.softmax = nn.Softmax(dim=1) # dim 0 is batch position and dim 1 is the logits
    self.model = nn.Sequential(self.fc, self.softmax)

  def forward(self, x):
    return self.model(x)

"""
class Guesser(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] 1st guesses
  '''
  def __init__(self, featureSize, N_CLASSES, N_BLOCKS, N_KERNELS, kernelSize):
    super().__init__()
    self.N_BLOCKS = N_BLOCKS
    self.conv1x1_in = nn.Conv1d(featureSize, N_KERNELS, kernel_size=1) # adapts fv channel dimension
    
    ## ALternative Guesser: Using different blocks from Predictor
    # self.dconv1 = nn.ModuleList((
    #   nn.Conv1d(N_KERNELS, N_KERNELS, kernelSize, padding=2**(N_BLOCKS - 1 - b), dilation=2**(N_BLOCKS - 1 - b))
    #             for b in range(N_BLOCKS) ))
    # self.dconv2 = nn.ModuleList((
    #   nn.Conv1d(N_KERNELS, N_KERNELS, kernelSize, padding=2**b, dilation=2**b)
    #             for b in range(N_BLOCKS )))
    # self.conv_fusion = nn.ModuleList((  # merge two previous
    #   nn.Conv1d(2 * N_KERNELS, N_KERNELS, kernel_size=1) 
    #             for _ in range(N_BLOCKS) ))
    
    ## Same Guesser: Same blocks as improver
    self.blocks = nn.ModuleList([copy.deepcopy(_TCResBlock(N_KERNELS, N_KERNELS, kernelSize, 2**b)) for b in range(N_BLOCKS)])
    # self.dropout = nn.Dropout()
    self.conv_out = nn.Conv1d(N_KERNELS, N_CLASSES, kernel_size=1)

  def forward(self, x):
    x = self.conv1x1_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.conv_out(x)
    return x

class Refiner(nn.Module):
  ''' Receives [batch_size, N_CLASSES, in_length] 1st guesses
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, N_CLASSES, N_STAGES, N_BLOCKS, N_KERNELS, kernelSize):
    super().__init__()
    self.N_STAGES = N_STAGES
    self.conv1x1_in = nn.Conv1d(N_CLASSES, N_KERNELS, kernel_size=1)
    self.blocks = nn.ModuleList([copy.deepcopy(_TCResBlock(N_KERNELS, N_KERNELS, kernelSize, 2**b)) for b in range(N_BLOCKS)])
    self.conv_out = nn.Conv1d(N_KERNELS, N_CLASSES, kernel_size=1)
  
  def forward(self, x):
    x = self.conv1x1_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.conv_out(x)
    return x
"""

'''
class FirstNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)   # 3 channel image input, 6 output features maps (from 6 different filters/kernels?), 5x5 square kernel
    self.pool = nn.MaxPool2d(2, 2)    # 2x2 pooling window with stride 2
    self.conv2 = nn.Conv2d(6, 16, 5)
    # CAREFUL!! Depends on the input size of the image (58x58 for 3x244x244 input)
# INPUT
    self.fc1 = nn.Linear(16 * 58 * 58, 120)
# INPUT
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    # print(x.shape)  # Add this to see the size of x after conv layers and pooling
    x = torch.flatten(x, 1)  # flatten starting from dimension 1 (batch dimension excluded)
    # print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
'''
