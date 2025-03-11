import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet
import os
from collections import defaultdict, OrderedDict
import torchvision.models.feature_extraction as fx
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import copy
import torchvision.models.feature_extraction as fxs
from . import catalogue
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader

class SurgeNet(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] 1st guesses
      Outputs [output_stage_id, batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, N_CLASSES, inators):
    super().__init__()
    self.N_CLASSES = N_CLASSES
    self.inators = inators  # separate model list composing the overall model
    self.model = nn.Sequential(*inators)
    
  def forward(self, x):
    print("Input shape: ", x.shape)
    x = self.model(x)
    print("Output shape: ", x.shape)
    return x
  def classify(self, x):
    return False
  def dry_run(self, layer, in_shape=[2, 3, 244, 244]):
    rnd_in = torch.randn(*in_shape)
    with torch.no_grad():
      out = layer(rnd_in)
    print(out.shape)
    return out


class Spaceinator(nn.Module):
  ''' 2D Convolutional network object, able to be trained for feature extraction,
      _features them or load presaved ones
      Modes - classify: train and classify in runtime
              _features: train and _features features in runtime
              load: load presaved features and forward them
  '''
  def __init__(self, MODEL, DEVICE, N_CLASSES=None):
    super().__init__()
    self.classifier = True if N_CLASSES else False
    self.N_CLASSES = N_CLASSES
    SPACE_ARCH = MODEL["spaceArch"]
    SPACE_WEIGHTS = self._get_preweights(MODEL["spaceWeights"])
    path_model = MODEL["path_spaceModel"]
    STATE_DICT = torch.load(path_model, weights_only=False, map_location=DEVICE) if path_model else None
    self.FEATURE_SIZE = 2048 if (SPACE_ARCH == "resnet50" or SPACE_ARCH == "resnet50-custom") else None # default for resnet50
    self.features = None

    self.model = self._get_model(SPACE_ARCH, SPACE_WEIGHTS, STATE_DICT, N_CLASSES)  # to implement my own in model.py    
    # print(self.model)
    # print(self.model.parameters())
    self.FEATURE_NODE = fxs.get_graph_node_names(self.model)[0][-2] # -2 is the flatten node (virtual layer)
    train_nodes, eval_nodes = fx.get_graph_node_names(self.model)
    # print(train_nodes == eval_nodes, end='\t'); print(train_nodes)
    # print(list(self.model.children()))
    print("Feature size: ", self.FEATURE_SIZE, "\tnode: ", self.FEATURE_NODE)

  def forward(self, x):
    x = self.model(x)
    print("After Spaceinator Shape: ", x.shape)
    return x
  def get_featureSize(self, x):
    pass
  def _get_model(self, SPACE_ARCH, SPACE_WEIGHTS, STATE_DICT, N_CLASSES):
    if SPACE_ARCH == "resnet50":
      model = resnet50(weights=SPACE_WEIGHTS)
      if self.classifier:
        model.fc = nn.Linear(self.FEATURE_SIZE, N_CLASSES)
      else: # no classification layer
        model.fc = nn.Identity()
    elif SPACE_ARCH == "resnet50-custom" and STATE_DICT:
      model = resnet50(weights=SPACE_WEIGHTS)
      if self.classifier:
        model.fc = nn.Linear(self.FEATURE_SIZE, N_CLASSES)
      else: # no classification layer
        model.fc = nn.Identity()
      model.load_state_dict(STATE_DICT, strict=False)
    else:
      raise ValueError("Invalid space architecture &/or path to fx_model choice!")
    return model

  def _get_preweights(self, weights):
    if weights == "default":
      return ResNet50_Weights.DEFAULT # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    elif not weights:
      return None
    else:
      raise ValueError("Invalid preweight choice!")
  
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
    layer = self.FEATURE_NODE
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
    self.FEATURE_SIZE = example_feature.size(0) # assuming every value has the same dim
    print("\nE.g. Feature: ", example_feature.squeeze(-1).squeeze(-1))
    print("Features Size: ", self.FEATURE_SIZE, '\n')

  def _extract(self, images, nodesToExtract):
    # Extract features at specified nodes
    return fx.create_feature_extractor(self.model, return_nodes=nodesToExtract)(images)

  def export_features(self, dataloader, path_to, fold, nodesToExtract, device="cpu"):
    # _features features maps to external file
    assert os.path.isdir(path_to), "Invalid features export path!"
    self.model.fc = nn.Identity()  # Remove classification layer
    out_features = defaultdict(self.getdd)  # default: ddict of features (at multiple nodes/layers) for each frame
    self.model.eval().to(device) 
    with torch.no_grad():
      for bi, batch in enumerate(dataloader):
        # print('\n', "batch", bi)
        images, ids = batch[0].to(device), batch[2]
        features = self._extract(images, nodesToExtract)
        for layer in features:
          print(features[layer].shape)
          for im_id, fs in zip(ids, features[layer]): # Save features from chosen nodes/layers for each image ID
            # out_features[im_id][layer] = fs.cpu()
            out_features[im_id][layer] = fs.cpu() if fs.size(0) > 1 else fs.unsqueeze(0).cpu()
            # print(list(out_features.items()))
            # break
    print(os.path.join(path_to, f"fmaps_{fold + 1}.pt"))
    torch.save(out_features, os.path.join(path_to, f"fmaps_{fold + 1}.pt"))
  
  @staticmethod
  def getdd(): # torch.save() can't pickle whatever if nested defaultdict uses lambda (dedicated function instead) 
    return defaultdict(None)

class Timeinator(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] further guesses
  '''
  def __init__(self, MODEL, FEATURE_SIZE, N_CLASSES):
    super().__init__()

    self.N_STAGES = MODEL["n_stages"]
    N_BLOCKS = MODEL["n_blocks"]
    N_FILTERS = MODEL["n_filters"]
    TF_FILTER = MODEL["tf_filter"]
    self.conv1x1_in = nn.Conv1d(FEATURE_SIZE, N_FILTERS, kernel_size=1)
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(N_FILTERS, N_FILTERS, TF_FILTER, 2**b)) for b in range(N_BLOCKS)])
    self.conv_out = nn.Conv1d(N_FILTERS, N_CLASSES, kernel_size=1)

  def forward(self, x):
    x = self.conv1x1_in(x)
    x_acm = x.unsqueeze(0)  # add dimension for accumulation stage results
    for _ in range(self.N_STAGES):
      for block in self.blocks:
        x = block(x)
      x = F.softmax(self.conv_out(x), dim=1)  # get predictions from logits
      x_acm = torch.cat((x_acm, x.unsqueeze(0)), dim=0)
    return x.squeeze(0).permute(1, 0) # x_accm for more in depth (returning last state guess)

class _TCBlock(nn.Module):
  # Dilated Residual Block of Layers
  def __init__(self, in_channels, out_channels, filterSize, dilation):
    super().__init__()
    self.dconv = nn.Conv1d(in_channels, out_channels, kernel_size=filterSize, padding=dilation, dilation=dilation)
    self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time frame 1
    self.dropout = nn.Dropout()
  
  def forward(self, x):
    res = self.conv1x1(x) # residual
    x = F.relu(self.dconv(x))
    x = self.conv1x1(x)
    x = self.dropout(x)
    return x + res


"""
class Guesser(nn.Module):
  ''' Receives [batch_size, in_channels, in_length] feature tensors
      Outputs [batch_size, N_CLASSES, in_length] 1st guesses
  '''
  def __init__(self, FEATURE_SIZE, N_CLASSES, N_BLOCKS, N_FILTERS, filterSize):
    super().__init__()
    self.N_BLOCKS = N_BLOCKS
    self.conv1x1_in = nn.Conv1d(FEATURE_SIZE, N_FILTERS, kernel_size=1) # adapts fv channel dimension
    
    ## ALternative Guesser: Using different blocks from Predictor
    # self.dconv1 = nn.ModuleList((
    #   nn.Conv1d(N_FILTERS, N_FILTERS, filterSize, padding=2**(N_BLOCKS - 1 - b), dilation=2**(N_BLOCKS - 1 - b))
    #             for b in range(N_BLOCKS) ))
    # self.dconv2 = nn.ModuleList((
    #   nn.Conv1d(N_FILTERS, N_FILTERS, filterSize, padding=2**b, dilation=2**b)
    #             for b in range(N_BLOCKS )))
    # self.conv_fusion = nn.ModuleList((  # merge two previous
    #   nn.Conv1d(2 * N_FILTERS, N_FILTERS, kernel_size=1) 
    #             for _ in range(N_BLOCKS) ))
    
    ## Same Guesser: Same blocks as improver
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(N_FILTERS, N_FILTERS, filterSize, 2**b)) for b in range(N_BLOCKS)])
    # self.dropout = nn.Dropout()
    self.conv_out = nn.Conv1d(N_FILTERS, N_CLASSES, kernel_size=1)

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
  def __init__(self, N_CLASSES, N_STAGES, N_BLOCKS, N_FILTERS, filterSize):
    super().__init__()
    self.N_STAGES = N_STAGES
    self.conv1x1_in = nn.Conv1d(N_CLASSES, N_FILTERS, kernel_size=1)
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(N_FILTERS, N_FILTERS, filterSize, 2**b)) for b in range(N_BLOCKS)])
    self.conv_out = nn.Conv1d(N_FILTERS, N_CLASSES, kernel_size=1)
  
  def forward(self, x):
    x = self.conv1x1_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.conv_out(x)
    return x
"""

def save_model(path_models, modelState, datasetName, date, test_score, path_model=None, title='', fold=''):
  if path_model:
    torch.save(modelState, path_model)
  else:
    # print(path_models, datasetName, date, test_score)
    model_name = f"{datasetName}-{date}-{title}{fold}-{test_score:4f}.pth"
    path_model = os.path.join(path_models, model_name)
    torch.save(modelState, path_model)

def save_checkpoint(model, optimizer, epoch, loss, path_checkpoint):
  checkpoint = {
    'epoch': epoch, # Current epoch
    'model_state_dict': model.state_dict(), # Model weights
    'optimizer_state_dict': optimizer.state_dict(), # Optimizer state
    'valid_loss': loss  # Current loss value
  }
  torch.save(checkpoint, path_checkpoint)
  # print(f"Checkpoint saved at {path_checkpoint}")

def load_checkpoint(model, optimizer, path_checkpoint):
  print(f"Retrieving checkpoint for model of type {model}")
  checkpoint = torch.load(path_checkpoint)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  print(f"Checkpoint loaded: Epoch {epoch}, Loss {loss}")
  return model, optimizer, epoch, loss

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
