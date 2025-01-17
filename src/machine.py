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
      Outputs [output_stage_id, batch_size, n_classes, in_length] further guesses
  '''
  def __init__(self, n_classes, spatinator, tempinator):
    super().__init__()
    self.n_classes = n_classes
    self.Spaceinator = spatinator
    self.Timeinator = tempinator
  def forward(self, x):
    # print("In Shape: ", x.shape)
    x = self.Spaceinator(x)  # running offline the features are fed as data
    # print("FEX Shape: ", x.shape)
    x = self.Timeinator(x)
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
  def __init__(self, n_classes, path_spaceModel=None, spaceModelType=None, spacePreweights=None, nodesToExtract={}):
    super().__init__()

    spaceWeights = self._get_preweights(spacePreweights)
    self.spaceModel = self._get_model(spaceModelType, spaceWeights, torch.load(path_spaceModel, weights_only=False))  # to implement my own in model.py
    # fx_stateDict = torch.load(path_model, weights_only=False)
    self.featureLayer = fxs.get_graph_node_names(self.spaceModel)[0][-2]  # last is fc converted to identity so -2 for flattemed feature vectors
    print(self.featureLayer)  # get the last layer
    self.nodesToExtract = [self.featureLayer]
    if spaceModelType == "resnet50-custom":
      self.featureSize = 2048 # default for resnet50
    self.features = None
    # train_nodes, eval_nodes = fx.get_graph_node_names(self.model)
    # print(train_nodes == eval_nodes, end='\t'); print(train_nodes)
    # print(list(model.children()))
  def forward(self, x):
    return self.spaceModel(x)
  def get_featureSize(self, x):
    pass
  
  def _get_preweights(self, weights):
    if weights == "default":
      return ResNet50_Weights.DEFAULT # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    elif weights == None:
      return None
    else:
      raise ValueError("Invalid preweight choice!")
  
  def _get_model(self, spaceArch, spaceWeights, state_dict):
    if spaceArch == "resnet50":
      return resnet50(weights=spaceWeights)
    elif spaceArch == "resnet50_custom":
      model = resnet50(weights=spaceWeights)
      model.fc = nn.Identity()
      model.load_state_dict(state_dict, strict=False)
      return model
    else:
      raise ValueError("Invalid fx_modelType choice!")
  def _extract(self, images, nodes=None):
    # Extract features at specified nodes
    if not nodes:
      nodes = self.nodesToExtract
    return fx.create_feature_extractor(self.model, return_nodes=nodes)(images)
  
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
  
  # def classify(self, n_classes):
    # self.model = get_resnet50(n_classes, model_weights=modelPreWeights, extract_features=False)  # Keeps last classifier layer

  def load(self, path_from, layer=None):
    layer = self.featureLayer
    
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

  def export_features(self, dataloader, path_to, fold, device="cpu"):
    # _features features maps to external file
    assert os.path.isdir(path_to), "Invalid features export path!"
    out_features = defaultdict(Spaceinator.getdd)  # default: ddict of features (at multiple nodes/layers) for each frame
    self.model.eval().to(device) 
    with torch.no_grad():
      for bi, batch in enumerate(dataloader):
        # print('\n', "batch", bi)
        images, ids = batch[0].to(device), batch[2]
        features = self._extract(images)
        for layer in features:
          # print(features[layer].shape)
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
      Outputs [batch_size, n_classes, in_length] further guesses
  '''
  def __init__(self, featureSize, n_classes, n_stages, n_blocks, n_filters, filterSize):
    super().__init__()
    self.n_stages = n_stages
    self.n_blocks = n_blocks
    self.conv1x1_in = nn.Conv1d(featureSize, n_filters, kernel_size=1)
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(n_filters, n_filters, filterSize, 2**b)) for b in range(n_blocks)])
    self.conv_out = nn.Conv1d(n_filters, n_classes, kernel_size=1)

  def forward(self, x):
    x = self.conv1x1_in(x)
    x_acm = x.unsqueeze(0)  # add dimension for accumulation stage results
    for _ in range(self.n_stages):
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
    self.conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1) # filter with time size 1
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
      Outputs [batch_size, n_classes, in_length] 1st guesses
  '''
  def __init__(self, featureSize, n_classes, n_blocks, n_filters, filterSize):
    super().__init__()
    self.n_blocks = n_blocks
    self.conv1x1_in = nn.Conv1d(featureSize, n_filters, kernel_size=1) # adapts fv channel dimension
    
    ## ALternative Guesser: Using different blocks from Predictor
    # self.dconv1 = nn.ModuleList((
    #   nn.Conv1d(n_filters, n_filters, filterSize, padding=2**(n_blocks - 1 - b), dilation=2**(n_blocks - 1 - b))
    #             for b in range(n_blocks) ))
    # self.dconv2 = nn.ModuleList((
    #   nn.Conv1d(n_filters, n_filters, filterSize, padding=2**b, dilation=2**b)
    #             for b in range(n_blocks )))
    # self.conv_fusion = nn.ModuleList((  # merge two previous
    #   nn.Conv1d(2 * n_filters, n_filters, kernel_size=1) 
    #             for _ in range(n_blocks) ))
    
    ## Same Guesser: Same blocks as improver
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(n_filters, n_filters, filterSize, 2**b)) for b in range(n_blocks)])
    # self.dropout = nn.Dropout()
    self.conv_out = nn.Conv1d(n_filters, n_classes, kernel_size=1)

  def forward(self, x):
    x = self.conv1x1_in(x)
    for block in self.blocks:
      x = block(x)
    x = self.conv_out(x)
    return x

class Refiner(nn.Module):
  ''' Receives [batch_size, n_classes, in_length] 1st guesses
      Outputs [batch_size, n_classes, in_length] further guesses
  '''
  def __init__(self, n_classes, n_stages, n_blocks, n_filters, filterSize):
    super().__init__()
    self.n_stages = n_stages
    self.conv1x1_in = nn.Conv1d(n_classes, n_filters, kernel_size=1)
    self.blocks = nn.ModuleList([copy.deepcopy(_TCBlock(n_filters, n_filters, filterSize, 2**b)) for b in range(n_blocks)])
    self.conv_out = nn.Conv1d(n_filters, n_classes, kernel_size=1)
  
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

def get_model(modelkey, n_classes, Spaceinator, guesser, refiner):
  if modelkey == "spatial":
    model = SurgeNet(n_classes, Spaceinator)
  elif modelkey == "temporal":
    model = SurgeNet(n_classes, Spaceinator, guesser, refiner)
  else:
    raise ValueError("Invalid modelkey!")
  return model

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
