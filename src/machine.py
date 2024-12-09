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
  def __init__(self, modelkey, n_classes, fextractor, guesser, improver):
    super().__init__()
    self.modelkey = modelkey  # defines operating mode: fextract, classify2d, classify3d
    self.n_classes = n_classes
    # assert feature_extractor.compatible
    # fextractor_in
    self.fextractor = fextractor
    if modelkey == ""
    # guesser_in = [o.shape[1] for o in self.dry_run(self.fextractor._extract()).values()]
    self.guesser = guesser
    # improver_in
    self.improvers = nn.ModuleList([copy.deepcopy(improver) for _ in range(improver.n_stages)])
    # improver_out
  def forward(self, x):
    print("In Shape: ", x.shape)
    x = self.fextractor(x)  # running offline the features are fed as data
    print("FEX Shape: ", x.shape)
  
    if self.modelkey == "temporal":
      x = self.guesser(x) # 1st stage
      print("Guesser Shape: ", x.shape)
      x_acc = x.unsqueeze(0)  # add dimension for accumulation guesser/improvers guess results
      for improver in self.improvers: # next stages
        x = improver(F.softmax(x, dim=1))
        print("Refiner Shape: ", x.shape)
        print("x": x)
        x_acc = torch.cat((x_acc, x.unsqueeze(0)), dim=0)
        print("x_acc: ", x_acc.shape)

    # temporary output calculation:
    print("Out Shape: ", x.shape)
    return x # x_acc for more in depth (returning last state guess)
  
  def classify(self, x):

  def dry_run(self, layer, in_shape=[2, 3, 244, 244]):
    rnd_in = torch.randn(*in_shape)
    with torch.no_grad():
      out = layer(rnd_in)
    print(out.shape)
    return out


class Fextractor(nn.Module):
  ''' 2D Convolutional network object, able to be trained for feature extraction,
      export them or load presaved ones
      Modes - classify: train and classify in runtime
              export: train and export features in runtime
              load: load presaved features and forward them
  '''
  def __init__(self, n_classes, path_model=None, fx_modelType=None, preweights=None, nodesToExtract={}):
    super().__init__()

    fxWeights = self._get_preweights(preweights)
    model = self._get_model(fx_modelType, fxWeights, stateDict)  # to implement my own in model.py
    if fxkey == "fextract":
      # model = torch.nn.Sequential(*list(model.children())[:-1]) # remove last layer - FCL for classification
      model.fc = nn.Identity()  # "remove" last layer
    else: # classify aswell
      model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes)  # adjust out from 1000 (pretraining) to 6/10
    if state_dict:
      # print(list(fx.get_graph_node_names(model)[1]))
      model.load_state_dict(state_dict, strict=False)
      # print(list(fx.get_graph_node_names(model)[1]))


    fx_stateDict = torch.load(path_model, weights_only=False)
    fx_model = get_resnet50(n_classes, model_weights=modelPreWeights, state_dict=fx_stateDict, extract_features=True)
    fx_featureLayer = fxs.get_graph_node_names(fx_model)[0][-2]  # last is fc converted to identity so -2 for flattemed feature vectors
  # print(fx_featureLayer)  # get the last layer
    nodesToExtract = [fx_featureLayer]
    fx = Fextractor(fx_model, writer, nodesToExtract, mode=mode)
    fx.load(fx_featureLayer, path_feature)  # updates features attribute
    featureSize = fextractor.featureSize
    return fx, featureSize, fx_featureLayer
  else:
    raise ValueError("No valid fx_mode specificied (classify, export, load)")
  if model:
    assert type(model) == resnet.ResNet or type(model) == torch.nn.modules.container.Sequential, "FeatureExtractor only accepting resnets for now"
  # print(type(model))
    model.fc = nn.Identity() # 'remove' last classifier layer

    self.nodesToExtract = nodesToExtract
    self.features = {}
    self.featureSize = None
    self.featureLayer = fxs.get_graph_node_names(fx_model)[0][-2]  # last is fc converted to identity so -2 for flattemed feature vectors
    # train_nodes, eval_nodes = fx.get_graph_node_names(model)
    # print(train_nodes == eval_nodes)
    # print(train_nodes)
    # print(list(model.children()))
  def forward(self, x):
    if self.mode == "online":
      print("\nONLINE MODE!! CAREFUL!\n")
      return self.model(x)
    return x

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
  
  def classify(self, n_classes):
    self.model = get_resnet50(n_classes, model_weights=modelPreWeights, extract_features=False)  # Keeps last classifier layer

  def load(self, layer, path_from):
    features = torch.load(os.path.join(path_from, os.listdir(path_from)[0]), weights_only=False) # single item on the dir [0]
    
    for i in features.keys():
      random_f = features[i]
      # print(random_f)
      break
    example_feature = random_f[layer] # defaultdict of defaultdicts
    self.features = features
    self.featureSize = example_feature.size(0) # assuming every value has the same dim
    print("\nE.g. Feature: ", example_feature.squeeze(-1).squeeze(-1))
    print("Features Size: ", self.featureSize, '\n')

  def export(self, dataloader, path_to, device="cpu"):
    # export features maps to external file
    assert os.path.isdir(path_to), "Invalid features export path!"
    out_features = defaultdict(Fextractor.getdd)  # default: ddict of features (at multiple nodes/layers) for each frame
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
    print(os.path.join(path_to, f"fmaps_fold{path_to[-1]}.pt"))
    torch.save(out_features, os.path.join(path_to, f"fmaps_fold{path_to[-1]}.pt"))
  
  @staticmethod
  def getdd(): # torch.save() can't pickle whatever if nested defaultdict uses lambda (dedicated function instead) 
    return defaultdict(None)

  def _get_preweights(self, weights):
    if weights == "default":
      return ResNet50_Weights.DEFAULT # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    elif weights == None:
      return None
    else:
      raise ValueError("Invalid preweight choice!")
  
  def _get_model(self, fx_modelType, preweigts):
    if fx_modelType == "resnet50":
      return resnet50(weights=fxWeights)
    elif fx_modelType == "resnet50_custom":
      model = resnet50(weights=fxWeights)
      model.load_state_dict(state_dict, strict=False)
      return model

    else:
      raise ValueError("Invalid fx_modelType choice!")

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


def model_save(path_models, modelState, datasetName, date, test_score, path_model=None, title='', fold=''):
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