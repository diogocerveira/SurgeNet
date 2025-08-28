import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy, TopKMultilabelAccuracy, MulticlassPrecision, BinaryPrecision, MulticlassRecall, BinaryRecall, MulticlassF1Score, BinaryF1Score, MulticlassConfusionMatrix, BinaryConfusionMatrix
from src import machine
import torch
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnchoredOffsetbox
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re, itertools, colorsys
import seaborn as sns
import pickle
import time
import pandas as pd
import gc

class Teacher():
  ''' Each Teacher teaches (action = train) 'k_folds' models from 1 dataset, besides saving
      multiple checkpoint models along the way (for each fold)
      It also can evaluate (action = eval) the models.
      In summary, a Teacher is a Trainer, a Validater and a Tester
      In - split and batched dataset from a Cataloguer
      Action - teach, eval
      Pipeline: (train -> validate) x n_epochs -> test
  '''
  def __init__(self, TRAIN, EVAL, dataset, DEVICE):
    self.dataset = dataset
    self.N_PHASES = dataset.n_classes
    self.PHASES = dataset.phases
    self.labelType = dataset.labelType
    self.headType = dataset.headType
    self.headSplits = dataset.headSplits
    self.DATASET_SIZE = dataset.__len__()
    train_metric, valid_metric, test_metrics = self._get_metrics(TRAIN, EVAL, self.N_PHASES, dataset.labelToClassMap, dataset.labelType, dataset.headType, DEVICE)
    self.criterions = self._get_criterion(TRAIN["criterionId"], dataset.classWeights, DEVICE)
    self.criterionWeights = dataset.classWeights

    self.trainer = Trainer(train_metric, self.criterions, TRAIN, self.labelType, self.headType, self.headSplits, DEVICE)
    self.validater = Validater(valid_metric, self.criterions, self.labelType, self.headType, self.headSplits, DEVICE)
    self.tester = Tester(test_metrics, dataset.labels, self.PHASES, dataset.labelToClassMap, self.labelType, self.headType, self.headSplits, DEVICE)

    self.save_checkpoints = TRAIN["save_checkpoints"]
    self.highScore = -np.inf
    self.bestState = {}
    self.minDelta = TRAIN["HYPER"]["minDelta"]  # minimal delta for early stopping
    self.patience = TRAIN["HYPER"]["patience"]  # patience for early stopping

  def teach(self, model, trainloader, validloader, n_epochs, path_states, labels, path_resume=None):
    ''' Iterate through folds and epochs of model learning with training and validation
        In: Untrained model, data, etc
        Out: Trained model (state_dict)
    '''
    print("\nTeaching model: ", model.id)
    optimizer = self._get_optimizer(model, self.trainer.optimizerId, self.trainer.learningRate, self.trainer.momentum)
    scheduler = self._get_scheduler(optimizer, self.trainer.schedulerId, self.trainer.stepSize, self.trainer.gamma)

    earlyStopper = EarlyStopper(self.patience, self.minDelta)

    betterState, valid_minLoss = {}, np.inf
    earlyStopper.reset()
    if path_resume: # Loading checkpoint before resuming training
      model, optimizer, startEpoch, cpLoss = self.load_checkpoint(model, optimizer, path_resume)
    else:
      startEpoch = 0
    self.dataset.strength = 0.0
    prev_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(startEpoch, n_epochs):
      if model.domain != "time" and earlyStopper.counter == 2:  # one epoch after decrease the learning rate, increase augmentation strength
        self.dataset.strength = min(1.0, round(self.dataset.strength + 0.2, 2))  # increase strength every 5 epochs to max at 1.0
        print(f"\n* Dataset augmentation strength increased to {self.dataset.strength} *\n")

      print(f"-------------- Epoch {epoch + 1} --------------")
      print("     T"), torch.cuda.empty_cache(), gc.collect()
      trainloader.dataset.set_preprocessing(("train", self.dataset.preprocessingType), self.dataset.strength)
      train_loss, train_score = self.trainer.train(model, trainloader, labels, optimizer)
      print("     V"), torch.cuda.empty_cache(), gc.collect()
      validloader.dataset.set_preprocessing(("valid", self.dataset.preprocessingType), self.dataset.strength)
      valid_loss, valid_score = self.validater.validate(model, validloader, labels)
      torch.cuda.empty_cache(), gc.collect()
      self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
      self.writer.add_scalar("Loss/valid", valid_loss, epoch + 1)
      self.writer.add_scalar("Acc/train", train_score, epoch + 1)
      self.writer.add_scalar("Acc/valid", valid_score, epoch + 1)
      print(f"Train Loss: {train_loss:4f}\tValid Loss: {valid_loss:4f}")

      scheduler.step(valid_loss) # Adjust learning rate based on validation loss stagnation
      
      if (valid_minLoss - valid_loss) > earlyStopper.minDelta:  # if valid loss decreased by more than minDelta == good
        print(f"\n* New best model (valid loss): {valid_minLoss:.4f} --> {valid_loss:.4f} *\n")
        valid_minLoss = valid_loss
        valid_maxScore = valid_score
        betterState = model.state_dict()

      if earlyStopper.early_stop(valid_loss): # same as prev if but for a [patience] number of epochs in a row
        print(f"\n* Stopped early *\n")
        break
      
      lr = optimizer.param_groups[0]["lr"]
      if lr != prev_lr:
        print(f"\n* Learning rate changed to {lr:.6f} *")
      prev_lr = lr

      if ((epoch + 1) % 5) == 0:  # Save checkpoint every 5 epochs
        if self.save_checkpoints:
          print(f"\n* Checkpoint reached ({epoch + 1}/{n_epochs}) *\n")
          path_checkpoint = os.path.join(path_states, f"{model.id}_cp{epoch + 1}.pth")
          self.save_checkpoint(model, optimizer, epoch, valid_loss, path_checkpoint)
      model.valid_lastScore = valid_score

      # clear GPU memory
      torch.cuda.synchronize()               # wait for kernels to finish
      torch.cuda.empty_cache()               # release cached blocks to the driver
      gc.collect()                           # reclaim Python refs

    return model, valid_maxScore, betterState

  def evaluate(self, test_bundle, eval_tests, modelId, Csr):

    if eval_tests["aprfc"]:
      self.tester._aprfc(test_bundle, self.writer, modelId, Csr.path_aprfc)
      # clear GPU memory
      torch.cuda.empty_cache()
      gc.collect()
    if eval_tests["phaseTiming"]:
      outText  = self.tester._phase_timing(test_bundle, Csr.path_timings, modelId)
      # clear GPU memory
      torch.cuda.empty_cache()
      gc.collect()
      if eval_tests["phaseChart"]:
        self.tester._phase_graph(test_bundle, Csr.path_ribbons, modelId, outText)
        # clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

  def _get_optimizer(self, model, optimizerId, lr, momentum):
    if optimizerId == "adam":
      return optim.Adam(model.parameters(), lr=lr)
    elif optimizerId == "sgd":
      return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
      raise ValueError("Invalid optimizer chosen (adam, sgd)")
  def _get_scheduler(self, optimizer, schedulerId, stepSize, gamma):
    if schedulerId == "step":
      scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=gamma)
    elif schedulerId == "cosine":
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stepSize, eta_min=gamma)
    elif schedulerId == "plateau":
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='min',       # monitor loss (want to minimize)
        factor=0.1,       # scale lr by this factor (10x smaller)
        patience=3,       # wait 3 epochs without improvement, after a 4th change
    )
    else:
      raise ValueError("Invalid scheduler chosen (step, cosine)")
    return scheduler
  
  def _get_metrics(self, TRAIN, EVAL, N_PHASES, labelToClassMap, labelType, headType, DEVICE):

    train_metric = RunningMetric(TRAIN["train_metric"], N_PHASES, DEVICE, EVAL["agg"], EVAL["computeRate"], labelType, headType, labelToClassMap, EVAL["updateRate"])
    valid_metric = RunningMetric(TRAIN["valid_metric"], N_PHASES, DEVICE, EVAL["agg"], EVAL["computeRate"], labelType, headType, labelToClassMap, EVAL["updateRate"])
    test_metrics = MetricBunch(EVAL["test_metrics"], N_PHASES, labelToClassMap, EVAL["agg"], EVAL["computeRate"], DEVICE, labelType, headType)
    return train_metric, valid_metric, test_metrics

  def _get_criterion(self, CRITERION_ID, classWeights, DEVICE):
    assert len(classWeights) == len(self.headType), (
        f"Number of classWeights ({len(classWeights)}) does not match number of heads ({len(self.headType)})"
    )
    if CRITERION_ID == "cross-entropy":
      assert int(self.labelType[2][-1]) == 1 or (self.labelType[0] == "single"), "CrossEntropyLoss is for top1 tasks"
      return [nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE) for _ in range(len(self.headType))]
    elif CRITERION_ID == "weighted-cross-entropy":
      assert int(self.labelType[2][-1]) == 1 or (self.labelType[0] == "single"), "CrossEntropyLoss is for top1 tasks"
      return [nn.CrossEntropyLoss(weight=classWeights[i].to(DEVICE), ignore_index=-1).to(DEVICE) for i in range(len(self.headType))]
    if CRITERION_ID == "binary-cross-entropy-logits":
      # classCounts = 1 / classWeights # inverse frequency
      assert self.labelType[0] == "multi" and int(self.labelType[2][-1]) > 1, "BCEWithLogitsLoss is for multi-label tasks with top-k k>1"
      return [nn.BCEWithLogitsLoss(pos_weight=classWeights[i].to(DEVICE)).to(DEVICE) for i in range(len(self.headType))]
    else:
      raise ValueError("Invalid criterion chosen (crossEntropy, )")

  def save_checkpoint(self, model, optimizer, epoch, loss, path_checkpoint):
    checkpoint = {
      'epoch': epoch, # Current epoch
      'model_state_dict': model.state_dict(), # Model preweights
      'optimizer_state_dict': optimizer.state_dict(), # Optimizer state
      'valid_loss': loss  # Current loss value
    }
    torch.save(checkpoint, path_checkpoint)
    # print(f"Checkpoint saved at {path_checkpoint}")

  def load_checkpoint(self, model, optimizer, path_checkpoint):
    print(f"Retrieving checkpoint for model of type {model}")
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Epoch {epoch}, Loss {loss}")
    return model, optimizer, epoch, loss

def nextBatch(data, labels, model, labelType, batchStats, DEVICE):
  try:
    # print(data[0].shape, data[1], data[2].shape)
    inputs = data[0].to(DEVICE) # data is a list of [inputs, labels]
    targets = [headTargets.to(DEVICE) for headTargets in data[1]] # list of lists of target tensors
    idxs = data[2].to(DEVICE)  # sample indices in the dataset

    # print("BEFORE inputs: ", inputs.shape, "\ttargets: ", [tgt.shape for tgt in targets], "\t[idxs]: ", idxs.shape, '\n\n')
    
    # Add timestamps as input if the case
    if model.domain == "space-time" and ("dTsNet" in model.arch or "pTsNet" in model.arch):
      normTs = torch.tensor(labels.loc[idxs.cpu().numpy(), 'normalizedTimestamp'].values,
          dtype=torch.float32).to(DEVICE)
      inputs = [inputs, normTs]  # ← tuple of tensors
    else:
      inputs = [inputs]

    mask = torch.zeros(1) # when no mask is needed, return dummy
    # Mask and De-clipping (B, T, C -> (B*T, C))
    if model.inputType[1] == "clip":  # only clipped input needs masking
      if labelType[0] == "single":
        targets = [tgt.reshape(-1) for tgt in targets]
        mask = targets[0] != -1
      else:  # multi-label
        targets = [tgt.reshape(-1, tgt.shape[-1]) for tgt in targets]
        mask = (targets[0] != -1).all(-1)  # True for samples where all columns are valid
      idxs = idxs.reshape(-1)
      targets = tuple([tgt[mask] for tgt in targets])
      idxs = idxs[mask]

    
    # if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
    #   print(f"\n\t{'Example Sample #':<15}", idxs[0].cpu().numpy())
      # print(f"{'inputs shape:':<15}", [f"I{i} {inp.shape}" for i, inp in enumerate(inputs)])
      # print(f"{'targets shape:':<15}", [f"H{i} {tgt.shape}" for i, tgt in enumerate(targets)])
    #   for i, tgt in enumerate(targets):
    #     print(f"  tgt0 H{i:<9} {tgt[0]}")
    #   print(f"{'mask shape:':<15}", mask.shape)
  except Exception as e:
    print(f"Error processing batch {batchStats[0]} data: {e}\n")
    raise
  
  return inputs, targets, idxs, mask

def nextOutputs(model, inputs, mask, labelType, batchStats):
  try:
    if model.domain == "space-time" and ("dTsNet" in model.arch or "pTsNet" in model.arch):
      outputs = model(inputs) # forward pass (list of head outputs)
    else:
      outputs = model(inputs[0])
    # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
    if isinstance(outputs, torch.Tensor):
      outputs = (outputs,)  # make it a tuple for uniform processing

    # Declip and apply mask
    if model.inputType[1] == "clip":  # only clipped input needs masking
      if labelType[0] == "multi" and labelType[2][-1] == '1': # multi label with topk1 (2 heads)
        outputs = (
          [o.reshape(-1, o.shape[-1])[mask] for o in outputs[0]],  # head 0, all stages
          [o.reshape(-1, o.shape[-1])[mask] for o in outputs[1]]   # head 1, all stages
        )
        # if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
        #   print(f"{'outputs shape:':<15}", [f"H{i} {[outStage.shape for outStage in outHead]}" for i, outHead in enumerate(outputs)])
          # for i, outHead in enumerate(outputs):
          #   print([f"  out0 H{i:<9}{outStage[0].detach().cpu().numpy()}" for outStage in outHead])
      else:
        outputs = [out.reshape(-1, out.shape[-1])[mask] for out in outputs]
        # if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
        #   print(f"{'outputs shape:':<15}", [f"H{i} {out.shape}" for i, out in enumerate(outputs)])
          # for i, out in enumerate(outputs):
          #   print(f"  out0 H{i:<9} {out[0].detach().cpu().numpy()}")
    else:
      # if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
      #   print(f"{'outputs shape:':<15}", [f"H{i} {out.shape}" for i, out in enumerate(outputs)])
        # for i, out in enumerate(outputs):
        #   print(f"  out0 H{i:<9} {out[0].detach().cpu().numpy()}")
        pass

  except Exception as e:
    print(f"Error during model forward pass for batch {batchStats[0]}: {e}\n")
    raise

  return outputs

def nextLoss(criterions, outputs, targets, labelType, arch, batchStats):
  try:
    if labelType[0] == "single" and len(targets) == 1 and len(outputs) > 1:
      tgt = targets[0]
      criterion = criterions[0]
      loss = sum(criterion(out, tgt) for out in outputs)
    elif "multiTecno" in arch:  # special case for multi-head multi-label TecnoNet
      loss  = sum(criterions[0](out, targets[0]) for out in outputs[0])  # all stages, head 0
      loss += sum(criterions[1](out, targets[1]) for out in outputs[1])  # all stages, head 1
    else:
      loss = sum(criterion(out, tgt) for criterion, (out, tgt) in zip(criterions, zip(outputs, targets)))


    # if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
      # print(criterions)
      # print(f"{'loss:':<15}{loss.item():.4f}")
  except Exception as e:
    print(f"Error while computing loss for batch {batchStats[0]}: {e}\n")
    raise
  return loss

def nextPreds(outputs, targets, labelType, headType, headSplits, arch, metric, batchStats):
  try:
    try:
      k = int(labelType[2][-1])  
      if labelType[0] == "single":
        k = 1
    except:
      k = 1

    if "multiTecno" in arch: # multi tecno has outputs per head and then per stage, we only want last stage
      outputs = (outputs[0][-1], outputs[1][-1])
    if "tecno" in arch:
      outputs = (outputs[-1],)
    # always build preds from top-k
    topkPreds = [torch.topk(out, k=k, dim=1).indices for out in outputs]
    preds = [torch.zeros_like(out) for out in outputs]
    for i in range(len(preds)):
      preds[i].scatter_(1, topkPreds[i], 1)

    # -------------------------------------------
    # (1) SINGLE LABEL + SINGLE TASK
    if labelType[0] == "single" and len(preds) == 1:
      mergedPreds = preds[0].argmax(dim=1)  # vector of class indices [B]
      mergedTargets = targets[0].to(torch.int64)
      metric.update(mergedPreds, mergedTargets) # single-label metrics

    # (2) SINGLE LABEL + MULTI HEAD
    elif labelType[0] == "single" and len(preds) > 1:
      assert 1 == 0, "Single-label multi-head not implemented yet"
      # merge into full class space
      total_classes = sum(len(split) for split in headSplits)
      B = preds[0].shape[0]
      merged_full = torch.zeros(B, total_classes, device=preds[0].device)
      for head_idx, split in enumerate(headSplits):
        merged_full[:, split] = preds[head_idx]
      mergedPreds = merged_full.argmax(dim=1)
      # merge targets similarly for metric
      global_targets = torch.zeros(B, total_classes, device=mergedPreds.device)
      for head_idx, split in enumerate(headSplits):
        global_targets[:, split] = targets[head_idx]
      metric.update(mergedPreds, global_targets.argmax(dim=1).to(torch.int64)) # single-label metrics

    # (3) MULTI LABEL + SINGLE TASK
    elif labelType[0] == "multi" and len(preds) == 1:
      mergedPreds = preds[0]  # multi-hot [B, n_classes] (possibly top-k)
      mergedTargets = targets[0].to(torch.int64)  # multi-hot [B, n_classes]
      metric.update(mergedPreds, mergedTargets) # multi-label metrics

    # (4) MULTI LABEL but SINGLE LABEL in head each + MULTI TASK
    elif labelType[0] == "multi" and len(preds) > 1:
      total_classes = sum(len(split) for split in headSplits) # assuming non-overlapping splits
      B = preds[0].shape[0] # batch size
      mergedPreds = torch.zeros(B, total_classes, device=preds[0].device)
      mergedTargets = torch.zeros(B, total_classes, device=preds[0].device)
      for head_idx, split in enumerate(headSplits):
        mergedPreds[:, split] = preds[head_idx]  # one-hot in full space [B, n_classes]
        mergedTargets[:, split] = targets[head_idx]  # multi-hot in full space [B, n_classes]
        # update metric per head immediately with preds, not merged

        # Convert head split tuple to tensor on the right device
        split_tensor = torch.tensor(split, device=preds[0].device)

        # Map local argmax indices to global indices
        headPreds_local = preds[head_idx].argmax(dim=1)   # [B]
        headTargets_local = targets[head_idx].argmax(dim=1)  # [B]

        headPreds_global = split_tensor[headPreds_local]      # [B]
        headTargets_global = split_tensor[headTargets_local]  # [B]

        metric.update(headPreds_global, headTargets_global) # NOW using global class indices
    else:
      raise ValueError(f"Invalid labelType {labelType}")

    if batchStats[0] % batchStats[1] == 0 or (batchStats[0] + 1) == batchStats[2]:
      # print(f"{'preds shape:':<15}", [f"H{i} {pred.shape}" for i, pred in enumerate(preds)])
      # for i, pred in enumerate(preds):
      #   print(f"  pred0 H{i:<8} {pred[0].detach().cpu().numpy()}")
      # if len(preds) > 1:
      #   print([f"{'mergedPred:':<15}{mergedPreds[0].detach().cpu().numpy()}"])
      #   print([f"{'mergedTarget:':<15}{mergedTargets[0].detach().cpu().numpy()}"])
      print(f"\t  [B{batchStats[0] + 1}]  scr: {metric.score():.4f}")
      
  except Exception as e:
    print(f"Error while originating batch {batchStats[0]} predictions: {e}\n")
    raise
  return mergedPreds, mergedTargets

class Trainer():
  ''' Part of the teacher that knows how to train models based on data
  '''
  def __init__(self, metric, criterions, TRAIN, labelType, headType, headSplits, DEVICE):
    self.train_metric = metric
    self.criterions = criterions
    self.labelType = labelType
    self.headType = headType
    self.headSplits = headSplits

    self.optimizerId = TRAIN["optimizerId"]
    self.learningRate = TRAIN["HYPER"]["learningRate"]
    self.momentum = TRAIN["HYPER"]["momentum"]
    self.schedulerId = TRAIN["schedulerId"]
    self.stepSize = TRAIN["HYPER"]["stepSize"]
    self.gamma = TRAIN["HYPER"]["gamma"]

    self.DEVICE = DEVICE

  def train(self, model, trainloader, labels, optimizer):
    ''' In: model, data, criterion (loss function), optimizer
        Out: train_loss, train_score
        Note - inputs can be samples (pics) or feature maps / outputs are logits and then pred/probabilities
    '''
    self.train_metric.reset()  # resets states, typically called between epochs
    runningLoss, totalSamples = 0.0, 0
    smartComputeRate = len(trainloader) // self.train_metric.computeRate if len(trainloader) // self.train_metric.computeRate else 1
    batchStats = [0, smartComputeRate, len(trainloader)]

    model.train()
    for batch, data in enumerate(trainloader, 0):   # start at batch 0
      batchStats[0] = batch
      inputs, targets, idxs, mask = nextBatch(data, labels, model, self.labelType, batchStats, self.DEVICE)
      
      optimizer.zero_grad() # reset the parameter gradients before backward step

      outputs = nextOutputs(model, inputs, mask, self.labelType, batchStats)  # outputs is a list of size n_heads

      loss = nextLoss(self.criterions, outputs, targets, self.labelType, model.arch, batchStats)
      loss.backward() # backward pass
      optimizer.step()    # a single optimization step

      preds, _ = nextPreds(outputs, targets, self.labelType, self.headType, self.headSplits, model.arch, self.train_metric, batchStats)  # preds is a list of size n_heads

      runningLoss += loss.item() * targets[0].numel() # all heads should have same number of samples
      totalSamples += targets[0].numel()

    return runningLoss / totalSamples, self.train_metric.score()

def check(x, y, C):
    assert torch.isfinite(x).all(), "NaN/Inf in inputs"
    ys = y if isinstance(y,(list,tuple)) else [y]
    for h,t in enumerate(ys):
        if t.dtype == torch.long:
            m, M = int(t.min()), int(t.max())
            assert 0 <= m, f"neg target in head {h}"
            assert M < C[h], f"target {M} >= C({C[h]}) in head {h}"

  
class Validater():
  ''' Part of the teacher that knows how to validate a run of model training
  '''
  def __init__(self, metric, criterions, labelType, headType, headSplits, DEVICE):
    self.valid_metric = metric
    self.criterions = criterions
    self.labelType = labelType
    self.headType = headType
    self.headSplits = headSplits
    self.DEVICE = DEVICE

  def validate(self, model, validloader, labels):
    ''' In: model, data, criterion (loss function)
        Out: valid_loss, valid_score
    '''
    self.valid_metric.reset() # Running metric, e.g. accuracy
    runningLoss, totalSamples = 0.0, 0
    smartComputeRate = len(validloader) // self.valid_metric.computeRate if len(validloader) // self.valid_metric.computeRate else 1
    batchStats = [0, smartComputeRate, len(validloader)]

    model.eval()
    with torch.no_grad(): # don't calculate gradients (for learning only)
      for batch, data in enumerate(validloader, 0):   # start at batch 0
        batchStats[0] = batch
        inputs, targets, idxs, mask = nextBatch(data, labels, model, self.labelType, batchStats, self.DEVICE)

        outputs = nextOutputs(model, inputs, mask, self.labelType, batchStats)  # outputs is a list of size n_heads

        loss = nextLoss(self.criterions, outputs, targets, self.labelType, model.arch, batchStats)
        # print(loss.item())

        preds, _ = nextPreds(outputs, targets, self.labelType, self.headType, self.headSplits, model.arch, self.valid_metric, batchStats)  # preds is a list of size n_heads

        runningLoss += loss.item() * targets[0].numel()
        totalSamples += targets[0].numel()

    return runningLoss / totalSamples, self.valid_metric.score()

class Tester():
  ''' Part of the teacher that knows how to test a model knowledge in multiple ways
  '''
  def __init__(self, metrics, labels, PHASES, labelToClassMap, labelType, headType, headSplits, DEVICE):
    self.test_metrics = metrics
    self.DEVICE = DEVICE
    self.labels = labels
    self.PHASES = PHASES
    self.labelToClassMap = labelToClassMap  # map from label to class index
    self.labelType = labelType
    self.headType = headType
    self.headSplits = headSplits
    self.colormap = self.build_colormap_from_phases(PHASES)
    self.exportPrefix = ''

  def test(self, model, testloader, labels, export_bundle, path_export=None):
    ''' Test the model - return Preds, labels and sampleIds
    '''
    self.test_metrics.reset()
    predsList = []
    targetsList = []
    sampleIdsList = []
    smartComputeRate = len(testloader) // self.test_metrics.computeRate  if len(testloader) // self.test_metrics.computeRate else 1
    batchStats = [0, smartComputeRate, len(testloader)]

    model.eval(); print("\n\tTesting...")
    with torch.no_grad():
      for batch, data in enumerate(testloader, 0):   # start at batch 0
        batchStats[0] = batch
        inputs, targets, idxs, mask = nextBatch(data, labels, model, self.labelType, batchStats, self.DEVICE)

        outputs = nextOutputs(model, inputs, mask, self.labelType, batchStats)  # outputs is a list of size n_heads

        preds, targets = nextPreds(outputs, targets, self.labelType, self.headType, self.headSplits, model.arch, self.test_metrics, batchStats)

        predsList.append(preds.cpu())
        targetsList.append(targets.cpu())
        sampleIdsList.append(idxs.cpu())

    predsList = torch.cat(predsList).tolist()  # flattens the list of tensor
    targetsList = torch.cat(targetsList).tolist()
    sampleIdsList = torch.cat(sampleIdsList).tolist()

    # print(f"Preds: {predsList}\n Targets: {targetsList}\n SampleIds: {sampleIdsList}")
    test_bundle = self._get_bundle(predsList, targetsList, sampleIdsList, self.labelType)
    if export_bundle:
      # print("here", path_export)
      test_bundle.to_csv(path_export + ".csv", index=False)
      with open(path_export + ".pt", 'wb') as f:
        pickle.dump(test_bundle, f)
    # print(len(predsList), len(targetsList), len(sampleIdsList))
    # print(f"SampleIds Type: {type(sampleIdsList[0])}, Content: {sampleIdsList[:5]}"
    return test_bundle, self.test_metrics.score()

  def _get_bundle(self, preds, targets, sampleIds, labelType):
    # for now requires ordered parameters
     # 1. Create base DataFrame
    bundle = pd.DataFrame({
        "Pred": preds,
        "Target": targets,
        "SampleId": sampleIds
    })
    # Convert preds and targets to phases
    # print(bundle.head(5))  # Show first 5 rows of the bundle
    classToLabelMap = {v: k for k, v in self.labelToClassMap.items()}
    # print(classToLabelMap)
    if labelType[0] == "multi":
      bundle["Pred"] = bundle["Pred"].apply(lambda x: self.multihotToLabel(x, classToLabelMap))
      bundle["Target"] = bundle["Target"].apply(lambda x: self.multihotToLabel(x, classToLabelMap))
    elif labelType[0] == "single":
      bundle["Pred"] = bundle["Pred"].apply(lambda x: classToLabelMap[x])
      bundle["Target"] = bundle["Target"].apply(lambda x: classToLabelMap[x])
    # print(bundle.head(5))  # Show first 5 rows of the bundle
    # 2. Extract Video and Timestamp from self.labels
    label_info = self.labels.copy()
    label_info[['Video', 'Timestamp']] = label_info['frameId'].str.split('_', expand=True)
    label_info['SampleId'] = label_info.index  # assume labels are indexed by sampleId

    # 3. Merge video info into bundle where sampleId >= 0
    bundle = bundle.merge(label_info[['SampleId', 'Video', 'Timestamp']], on='SampleId', how='left')
    # 4. Forward-fill Video/Timestamp for padding frames (SampleId == -1)
    bundle[['Video', 'Timestamp']] = bundle[['Video', 'Timestamp']].ffill()
    bundle["Timestamp"] = bundle["Timestamp"].astype(int)  # Convert Timestamp to int
    # 5. Sort by video and timestamp to ensure order
    bundle = bundle.sort_values(by=['Video', 'Timestamp']).reset_index(drop=True)
    # 6. Optional: assign absolute frame order
    bundle['AbsoluteFrameIndex'] = bundle.groupby('Video').cumcount()
    # print(bundle.head(5))  # Show first 5 rows of the bundle
    return bundle
  def multihotToLabel(self, multihot, classToLabelMap, sep=','):
    # print(f"Multihot: {multihot}")  # Debugging line to check multihot input
    multihot = np.array(multihot)  # ensure multihot is a numpy array
    indices = np.where(multihot == 1.0)[0]  # indices where class is active
    # print(f"Indices: {indices}")  # Debugging line to check indices
    labels = sorted([classToLabelMap[int(i)] for i in indices])
    # print(f"Labels: {labels}")  # Debugging line to check labels
    return sep.join(labels)

  def _aprfc(self, test_bundle, writer, modelId, path_aprfc):
    # self.test_metrics.display()
    fold = int(modelId[1]) # get fold number from modelId
    self.test_metrics.to("cpu")
    self.test_metrics.update_from_bundle(test_bundle, self.labelToClassMap, self.labelType)
    if self.test_metrics.accuracy:
      writer.add_scalar(f"accuracy/test", self.test_metrics.get_accuracy(), fold)
    if self.test_metrics.precision:
      writer.add_scalar(f"precision/test", self.test_metrics.get_precision(), fold)
    if self.test_metrics.recall:
      writer.add_scalar(f"recall/test", self.test_metrics.get_recall(), fold)
    if self.test_metrics.f1score:
      writer.add_scalar(f"f1score/test", self.test_metrics.get_f1score(), fold)
    if self.test_metrics.confusionMatrix:
      path_to = os.path.join(path_aprfc, f"{self.exportPrefix}cm_{modelId}.png")
      cm = self.test_metrics.get_confusionMatrix(path_to)
      if cm is not None:
        image_cf = plt.imread(cm)
        tensor_cf = torch.tensor(image_cf).permute(2, 0, 1)
        writer.add_image(f"confusion_matrix/test", tensor_cf, fold)

    writer.close()
    
  def _phase_timing_i2(self, df, path_phaseTiming, modelId, samplerate=1):
    # print(self.PHASES)
    phaseLabelToIntMap = {phase: i for i, phase in enumerate(self.PHASES)}
    phaseIntToLabelMap = {i: phase for i, phase in enumerate(self.PHASES)}
    print(phaseLabelToIntMap)
    outText = []
    lp, lt = len(df["Pred"]), len(df["Target"])
    # N_CLASSES = (max(df["Targets"]) + 1)  # assuming regular int encoding
    # N_CLASSES = len(preds[0]) # assuming one hot encoding
    assert lp == lt, "Targets and Preds Size Incompatibility"
    N_PHASES = len(self.PHASES)
    videos = df["Video"].unique()
    phases_preds_sum = np.zeros(N_PHASES)
    phases_gt_sum = np.zeros(N_PHASES)
    phases_diff_sum = np.zeros(N_PHASES)

    for idx, video in enumerate(videos):
      print()
      df_filtered = df[df["Video"] == video][["Pred", "Target"]].copy()
      # print(df_filtered.head(5))
      df_filtered["Target"] = df_filtered["Target"].apply(lambda x: phaseLabelToIntMap[x])
      df_filtered["Pred"] = df_filtered["Pred"].apply(lambda p: phaseLabelToIntMap[p] if p in self.PHASES else -1)
      # print("Filtered DataFrame:\n", df_filtered.head(5))
      # Count only valid classes (ignore -1)
      phases_preds_video = df_filtered[df_filtered["Pred"] != -1]["Pred"].value_counts().reindex(range(N_PHASES), fill_value=0)
      phases_gt_video = df_filtered["Target"].value_counts().reindex(range(N_PHASES), fill_value=0)

      phases_preds_sum += phases_preds_video.values
      phases_gt_sum += phases_gt_video.values
      phases_diff_sum += np.abs(phases_preds_video.values - phases_gt_video.values)
      # print(phases_diff_sum)
      
      # Optionally print counts for each video
      print("phases_preds counts:", phases_preds_video.values, '\t', sum(phases_preds_video))
      print("phases_gt counts:", phases_gt_video.values, '\t', sum(phases_gt_video))
      print(phaseIntToLabelMap)
      mix = list(zip(phases_preds_video / (samplerate), phases_gt_video / (samplerate), phases_diff_sum))
      ot = f"\nVideo {video[:4]} Phase Report\n" + f"{'':<10} | {'T Preds':^10} | {'T Targets':^10} || {'Error':^10}\n" + '\n'.join(
            f"{phaseIntToLabelMap[i]:<10} | {self.get_hourMinSec_from_seconds(mix[i][0]):^10} | {self.get_hourMinSec_from_seconds(mix[i][1]):^10} || {self.get_hourMinSec_from_seconds(mix[i][2]):^10}"
            for i in range(N_PHASES)
        )
      outText.append(ot)
      print(ot, '\n')
      self._export_textImg(ot, path_phaseTiming, f"{self.exportPrefix}timings_{modelId}_{video[:4]}")
    # Average Overall Results
    phases_preds_avg = phases_preds_sum / len(videos) / (samplerate)
    phases_gt_avg = phases_gt_sum / len(videos) / (samplerate)
    phases_diff_avg = phases_diff_sum / len(videos) / (samplerate)
    mix = list(zip(phases_preds_avg, phases_gt_avg, phases_diff_avg))

    otavg = f"\n\nOverall Average Phase Report\n{'':12}\t|{'  T Avg Preds':<12}|{'  T Avg Targets':<12}||{'  Total AE':<12}\n" + '\n'.join(
            f"{phaseIntToLabelMap[i]:<12}|{int(mix[i][0]):12}|{int(mix[i][1]):12}||{int(mix[i][2]):12}"
            for i in range(N_PHASES)
      )
    # print(otavg)
    # self._export_textImg(otavg, path_phaseTiming, f"{modelId}_overall-phaseTiming")
    outText.append(otavg)
  # phase_metric([1, 1, 2, 3, 3, 4, 0, 5], [0, 0, 0, 2, 3, 4, 5, 1])

    # accumulate results to csv file
    results_df = pd.DataFrame({
        "Phase": [phaseIntToLabelMap[i] for i in range(N_PHASES)],
        "Avg_Preds": phases_preds_avg,
        "Avg_Targets": phases_gt_avg,
        "Avg_Absolute_Error": phases_diff_avg
    })
    results_path = os.path.join(path_phaseTiming, f"{self.exportPrefix}timings_Summary.csv")

    # aggregate with existing results if any (csv)
    # Step 1: load existing
    if os.path.isfile(results_path):
      old_df = pd.read_csv(results_path)
      combined_df = pd.concat([old_df, results_df], ignore_index=True)
    else:
      combined_df = results_df
    # Step 2: aggregate by phase
    agg_df = combined_df.groupby("Phase", as_index=False).mean()
    # Step 3: save aggregated
    print(f"Saving aggregated phase timing results to {results_path}")
    agg_df.to_csv(results_path, index=False)
    return outText

  def _export_textImg(self, outText, path_to, name):
    from matplotlib import rcParams
    rcParams['font.family'] = 'monospace'
    # Clean text
    outText = outText.replace('\t', '    ').replace('\u200b', '')
    fig, ax = plt.subplots()
    ax.axis('off')
    # Create TextArea with monospaced font
    ta = TextArea(outText, textprops=dict(fontsize=10, linespacing=1.4))
    # Anchor it to top-left
    box = AnchoredOffsetbox(
      loc='upper left',
      child=ta,
      pad=0.5,
      frameon=True,
      bbox_to_anchor=(0, 1),
      bbox_transform=ax.transAxes,
      borderpad=0.2
    )
    ax.add_artist(box)
    save_path = os.path.join(path_to, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

  def _phase_timing(self, df, path_phaseTiming, modelId, samplerate=1):
    return self._phase_timing_i2(df, path_phaseTiming, modelId, samplerate=samplerate)

  def _phase_graph(self, df, path_phaseCharts, modelId, outText):
    # df = pd.DataFrame({
    #   'SampleName': ['vid1_1', 'vid1_3', 'vid1_2', 'vid2_20', 'vid2_23', 'vid2_40', 'vid2_51', 'vid2_52'],
    #   'Targets': [0, 1, 1, 2, 0, 1, 1, 1],  # Example target labels
    #   'Preds': [0, 1, 0, 2, 1, 4, 3, 2],  # Example predictions (e.g., class indices)
    # })
    # Prepare for plotting
    videos = df['Video'].unique()
    num_videos = len(videos)
    print(f'\n\nNumber of videos: {num_videos}')
    # color_map = {0: 'springgreen', 1: 'goldenrod', 2: 'tomato', 3: 'mediumslateblue', 4: 'plum', 5: 'deepskyblue'}
    
    plt.rcParams['font.family'] = 'monospace'
    # for saving all videos to same img
    # fig, axes = plt.subplots(num_videos, 1, figsize=(9, 5), sharex=True) # Create subplots
    # if num_videos == 1: axes = [axes] # prevent errors if only 1 video
    t1 = time.time()
    # Plotting
    for idx, video in enumerate(videos):
      data_video = df[df['Video'] == video][["Target", "Pred"]]
      fig, (ax, ax_text) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(9, 6))
        
      # ax = axes[idx]  # for saving all videos to same image
      # Prepare the bar data
      for j, stage in enumerate(data_video.columns):
        start_positions = np.arange(len(data_video))
        class_values = data_video[stage].values
        # print(class_values)
        ax.broken_barh(
            list(zip(start_positions, np.ones(len(start_positions)))),
            (j - 0.4, 0.8),
            facecolors=[self.colormap.get(val, "#bbbbbb") for val in class_values]
        )
      # Add vertical separators
      # ax.axvline(x=len(data_video), color='grey', linestyle='--', linewidth=0.5)
      # Set labels
      ax.set_yticks(np.arange(len(data_video.columns)))  # Middle of the two rows
      ax.set_yticklabels(data_video.columns)
      ax.set_xticks([])
      ax.set_xlabel("Time Steps")
      ax.set_title(f"Video: {video}")

      # Add the summary paragraph below the graph
      ax_text.axis('off')  # Hide axis for the text box
      ot = outText[idx].replace('\t', '     ').replace('\u200b', '')  # Removes hidden tabs and zero-width spaces

      # print(ot)
      ax_text.text(0, 0.5, ot, transform=ax_text.transAxes, fontsize=11, 
                  verticalalignment='center', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

      # Only phases actually present in this video's data
      used_set = set(map(str, data_video["Target"].values)) | set(map(str, data_video["Pred"].values))

      # Order legend entries by your global PHASES
      standard_in_video = [p for p in self.PHASES if p in used_set]
      nonstandard_in_video = [p for p in used_set if p not in self.PHASES]
      used_phases = standard_in_video + nonstandard_in_video
      handles = [plt.Rectangle((0, 0), 1, 1, color=self.colormap.get(p, "#bbbbbb")) for p in used_phases]
      labels = used_phases

      legend = fig.legend(
          handles, labels,
          loc='lower right',
          bbox_to_anchor=(0.98, 0.02),
          bbox_transform=fig.transFigure,
          borderaxespad=0.5,
          fontsize=9,
          title='Phases',
          frameon=True
      )

      plt.tight_layout()  # can keep to adjust spacing nicely
      # print(modelId)
      plt.savefig(os.path.join(path_phaseCharts, f"{self.exportPrefix}ribbons_{modelId}_{video[:4]}.png"), bbox_inches='tight')
      plt.close(fig)

    t2 = time.time()
    print(f"Video graphing took {t2 - t1:.2f} seconds")
    print(f"\nSaved phase metric graphs to {path_phaseCharts}")

  def get_phase_colormap_with_unknowns(self, df):
    known_set = set(self.PHASES)
    print(f"Known phases: {known_set}", self.PHASES)
    # Gather all predicted and target phases
    preds = df["Pred"].dropna().tolist()
    targets = df["Target"].dropna().tolist()
    all_phases = set(preds + targets)
    # Split known and unknown phases
    known = [p for p in all_phases if p in known_set]
    unknown = sorted([p for p in all_phases if p not in known_set])  # sort to keep color order consistent

    # Final ordered phase list
    full_phases = known + unknown
    n = len(full_phases)
    # Choose colormap based on count
    base_cmap = cm.get_cmap('Set3', n) if n <= 20 else cm.get_cmap('nipy_spectral', n)
    # Map each phase string to a hex color
    colormap = {
      phase: mcolors.to_hex(base_cmap(i))
      for i, phase in enumerate(full_phases)
    }
    return colormap, full_phases

  def _beautiful_palette(self, n_colors=25):
    """
    Returns a vibrant exotic palette with up to 25 colors, maximally distinct first.
    """
    palette = [
        "#aaffc3", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
        "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
        "#e6194b", "#808000", "#ffd8b1", "#000075", "#808080",
        "#ffffff", "#000000", "#ff69b4", "#7fff00", "#00ced1"
    ]
    # Cap at requested n_colors
    return palette[:n_colors]

  def build_colormap_from_phases(self, phases):
      """
      Build a fixed colormap for ALL unordered class pairs found in `phases`.
      Keys include both ',' and '-' separators and both orders (A,B) & (B,A).
      """
      # 1) collect atomic class tokens
      tokens = set()
      for ph in phases:
          for t in re.split(r'[,-]', str(ph)):
              t = t.strip()
              if t:
                  tokens.add(t)
      tokens = sorted(tokens)

      # 2) all unordered pairs
      pairs = list(itertools.combinations(tokens, 2))
      palette = self._beautiful_palette(len(pairs))

      # 3) map synonyms → same color
      cmap = {}
      for (a, b), col in zip(pairs, palette):
          for k in (f"{a},{b}", f"{b},{a}", f"{a}-{b}", f"{b}-{a}"):
              cmap[k] = col

      return cmap

  def get_hourMinSec_from_seconds(self, seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    h, m, s = '', '', ''
    if hours > 0:
      h = f"{int(hours)}h "
    if minutes > 0:
      m = f"{int(minutes)}m "
    if secs > 0 or (hours == 0 and minutes == 0):
      s = f"{int(secs)}s"
    return f"{h}{m}{s}".strip()

class EarlyStopper:
  ''' Controls the validation loss progress to keep it going down and improve times
  '''
  def __init__(self, patience=5, minDelta=0.001):
    self.patience = patience
    self.minDelta = minDelta  # percentual difference for a worthy increase in performance, 0 for no margin
    self.valid_minLoss = np.inf
    self.counter = 0

  def reset(self):
    self.valid_minLoss = np.inf
    self.counter = 0
  def early_stop(self, valid_loss):
    if (self.valid_minLoss - valid_loss) > self.minDelta:  # if loss decreased by more than minDelta
      self.valid_minLoss = valid_loss
      self.counter = 0  # reset warning/patience counter if loss decreases betw epochs
    else:
      self.counter += 1 # added 1 strike (5 strikes and you're out)
      if self.counter >= self.patience:
        return True
    return False
  def early_stop_opposite(self, valid_loss):
    if valid_loss < self.valid_minLoss:
      self.valid_minLoss = valid_loss
      self.counter = 0  # reset warning/patience counter if loss decreases betw epochs
    elif ((valid_loss - self.valid_minLoss) / valid_loss) >= self.minDelta:
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  
class StillMetric:
  ''' For now uses torcheval metrics
      Performs metric update and computation based on a frequency provided
  '''
  def __init__(self, metricName, n_classes, DEVICE, labelType, headType, labelToClass, agg="micro"):
    self.name = metricName
    self.labelToClass = labelToClass  # map from label to class index
    self.labelType = labelType  # single or multi
    self.topK = int(self.labelType[2][-1])
    self.headType = headType  # single or multi
    if metricName == "accuracy":
      if labelType[0] == "single" or (labelType[0] == "multi" and int(labelType[2][-1]) == 1):  # single-label single-head or multi-label multi-head with topK=1
        self.metric = MulticlassAccuracy(average="micro", num_classes=n_classes, device=DEVICE)
      else:
        self.metric = TopKMultilabelAccuracy(device=DEVICE, k=self.topK) # criteria="hamming"
    elif metricName == "precision":
      self.metric =  MulticlassPrecision(average=agg, num_classes=n_classes, device=DEVICE)
    elif metricName == "recall":
      self.metric = MulticlassRecall(average=agg, num_classes=n_classes, device=DEVICE)
    elif metricName == "f1score":
      self.metric = MulticlassF1Score(average=agg, num_classes=n_classes, device=DEVICE)
    elif metricName == "confusionMatrix":
      self.metric = MulticlassConfusionMatrix(num_classes=n_classes, device=DEVICE, normalize="true")
    else:
      raise ValueError("Invalid Metric Name")

  def reset(self):
    self.metric.reset()
  def score(self, path_to=None):
    if self.name == "confusionMatrix":
      cm = self.metric.compute().cpu().numpy()

      # # normalize by row (true labels)
      # cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)  # avoid division by zero

      # original labels
      labels = list(self.labelToClass.keys())

      # # move index 1 to the end
      # order = list(range(cm.shape[0]))  # assuming square matrix
      # col = order.pop(1)
      # order.append(col)

      # # reorder rows and columns
      # cm_reordered = cm[np.ix_(order, order)]
      # labels_reordered = [labels[i] for i in order]

      fig, ax = plt.subplots(figsize=(10, 7))
      sns.heatmap(
          cm,
          annot=True,
          fmt=".2f",
          cmap="rocket_r",
          vmin=0, vmax=1,
          xticklabels=labels,
          yticklabels=labels,
          ax=ax
      )
      ax.set_xlabel("Predicted")
      ax.set_ylabel("True")
      ax.set_title("Normalized Confusion Matrix")
      print(path_to)
      if path_to:
        path_img = os.path.join(path_to)
        plt.savefig(path_img)
        return path_img
    return self.metric.compute().item()
  
class RunningMetric(StillMetric):
  ''' Extends StillMetric to updates on the run
  '''
  def __init__(self, metricName, n_classes, DEVICE, agg, computeRate, labelType, headType, labelToClass, updateFreq=1):
    super().__init__(metricName, n_classes, DEVICE, labelType, headType, labelToClass, agg)
    self.computeRate = computeRate  # num of batches to show minus the last 
    self.updateFreq = updateFreq  # num of batches between update
  
  def update(self, outputs, targets):     
    if self.name == "confusionMatrix" and len(self.headType) == 1 and self.labelType[0] == "multi":
      # multi-label single-head with top-k predictions
      k = 1  # or infer from self.labelType[2][-1] if you want dynamic
      topk_preds = torch.topk(outputs, k=k, dim=1).indices  # [B, k]
      # Predicted class ids [B*k]
      preds_flat = topk_preds.flatten()
      # True label at each predicted class
      targets_flat = targets.gather(1, topk_preds).flatten()
      self.metric.update(preds_flat, targets_flat)
    else:
      self.metric.update(outputs, targets)

class MetricBunch:
  ''' Gets a bunch of still metrics at cheap price
      Accuracy locked at agg==micro for now
  '''
  def __init__(self, metricSwitches, n_classes, labelToClass, agg, computeRate, DEVICE, labelType, headType):  # metricSwitches is a boolean dict switch for getting metrics
    self.metricSwitches = metricSwitches
    self.labelToClass = labelToClass
    self.agg = agg
    self.computeRate = computeRate
    self.DEVICE = DEVICE

    if labelType[0] == "single" or (labelType[0] == "multi" and int(labelType[2][-1]) == 1):  # single-label single-head or multi-label multi-head
      self.accuracy = RunningMetric("accuracy", n_classes, DEVICE, "micro", computeRate, labelType, headType, labelToClass) if self.metricSwitches["accuracy"] else None
      self.precision = RunningMetric("precision", n_classes, DEVICE, agg, computeRate, labelType, headType, labelToClass) if self.metricSwitches["precision"] else None
      self.recall = RunningMetric("recall", n_classes, DEVICE, agg, computeRate, labelType, headType, labelToClass) if self.metricSwitches["recall"] else None
      self.f1score = RunningMetric("f1score", n_classes, DEVICE, agg, computeRate, labelType, headType, labelToClass) if self.metricSwitches["f1score"] else None
      self.confusionMatrix = RunningMetric("confusionMatrix", n_classes, DEVICE, agg, computeRate, labelType, headType, labelToClass) if self.metricSwitches["confusionMatrix"] else None
    elif labelType[0] == "multi" and int(labelType[2][-1]) > 1:  # multi-label single-head (for now)
      self.accuracy = RunningMetric("accuracy", n_classes, DEVICE, "micro", computeRate, labelType, headType, labelToClass) if self.metricSwitches["accuracy"] else None
      self.precision = None
      self.recall = None
      self.f1score = None
      self.confusionMatrix = None
    else:
      raise ValueError("Invalid labelType")
  def score(self):
     assert self.accuracy, "Accuracy metric not initialized! Can't compute Bunch score."
     return self.accuracy.score()
  def reset(self):
    if self.accuracy: self.accuracy.reset()
    if self.precision: self.precision.reset()
    if self.recall: self.recall.reset()
    if self.f1score: self.f1score.reset()
    if self.confusionMatrix: self.confusionMatrix.reset()
    
  def update(self, outputs, targets):
    if self.accuracy: self.accuracy.update(outputs, targets)
    if self.precision: self.precision.update(outputs, targets)
    if self.recall: self.recall.update(outputs, targets)
    if self.f1score: self.f1score.update(outputs, targets)
    if self.confusionMatrix: self.confusionMatrix.update(outputs, targets)

  def update_from_bundle(self, df, labelToClassMap, labelType):
    # Drop padding rows if they exist
    if "SampleId" in df.columns:
        df = df[df["SampleId"] >= 0].copy()
    assert len(df["Pred"]) == len(df["Target"]), "Pred and Target length mismatch in bundle DataFrame"

    if labelType[0] == "single":  
        # Pred/Target are strings like "Dis-Vau"
        y_pred = torch.tensor(
            [labelToClassMap[s] for s in df["Pred"].tolist()],
            dtype=torch.long,
            device=self.DEVICE
        )
        y_true = torch.tensor(
            [labelToClassMap[s] for s in df["Target"].tolist()],
            dtype=torch.long,
            device=self.DEVICE
        )
        self.reset()
        self.update(y_pred, y_true)

    elif labelType[0] == "multi" and labelType[2][-1] == '1':
       # topk=1 -> two heads, one class per head (order may vary)
      actions = {"0Act", "Dis", "Fix", "Rep"}
      objects = {"0Obj", "Vau", "Pro"}

      def split_pair(s):
        if not isinstance(s, str) or not s:
          return None, None
        parts = [p.strip() for p in s.replace("-", ",").split(",") if p.strip()]
        if len(parts) != 2:
          raise ValueError(f"Expected 2 tokens (action+object), got {s}")
        act, obj = None, None
        for p in parts:
          if p in actions:
            act = p
          elif p in objects:
            obj = p
          else:
            raise ValueError(f"Unknown class {p} in {s}")
        return act, obj

      preds_a, preds_o = zip(*[split_pair(s) for s in df["Pred"]])
      trues_a, trues_o = zip(*[split_pair(s) for s in df["Target"]])

      y_pred_a = torch.tensor([labelToClassMap[a] for a in preds_a],
                              dtype=torch.long, device=self.DEVICE)
      y_true_a = torch.tensor([labelToClassMap[a] for a in trues_a],
                              dtype=torch.long, device=self.DEVICE)

      y_pred_o = torch.tensor([labelToClassMap[o] for o in preds_o],
                              dtype=torch.long, device=self.DEVICE)
      y_true_o = torch.tensor([labelToClassMap[o] for o in trues_o],
                              dtype=torch.long, device=self.DEVICE)

      self.reset()
      self.update(y_pred_a, y_true_a)   # action head
      self.update(y_pred_o, y_true_o)   # object head

    elif labelType[0] == "multi" and int(labelType[2][-1]) > 1:
        # multi-label. Pred/Target are strings like "Class1,Class2"
        def to_ids(s):
            if not isinstance(s, str) or not s:
                return []
            return [labelToClassMap[c.strip()] for c in s.split(",") if c.strip()]

        pred_ids = [to_ids(s) for s in df["Pred"]]
        true_ids = [to_ids(s) for s in df["Target"]]

        C = len(labelToClassMap)
        Yhat = torch.zeros(len(pred_ids), C, dtype=torch.float32, device=self.DEVICE)
        Y    = torch.zeros(len(true_ids), C, dtype=torch.float32, device=self.DEVICE)

        for i, ids in enumerate(pred_ids):
            if ids: Yhat[i, ids] = 1.0
        for i, ids in enumerate(true_ids):
            if ids: Y[i, ids] = 1.0

        self.reset()
        self.update(Yhat, Y)
    else:
        raise ValueError("Invalid labelType")

  def to(self, device):
    # Move every underlying metric to device
    if self.accuracy: self.accuracy.metric.to(device)
    if self.precision: self.precision.metric.to(device)
    if self.recall: self.recall.metric.to(device)
    if self.f1score: self.f1score.metric.to(device)
    if self.confusionMatrix: self.confusionMatrix.metric.to(device)
    # Also update the DEVICE field so new tensors match
    self.DEVICE = device
    return self

  def display(self, metricSwitches={"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1, "confusionMatrix": 1}):
    if metricSwitches["accuracy"]:
      print(f"Accuracy: {self.get_accuracy():.4f}")
    if metricSwitches["precision"]:
      print(f"Precision: {self.get_precision():.4f}")
    if metricSwitches["recall"]:
      print(f"Recall: {self.get_recall():.4f}")
    if metricSwitches["f1score"]:
      print(f"F1 Score: {self.get_f1score():.4f}")
    if metricSwitches["confusionMatrix"]:
      plt.show(self.get_confusionMatrix())

  def get_accuracy(self):
    try:
      return self.accuracy.score()
    except:
      print("! Accuracy: Dataset group lacks representation of all classes !")
      return -1
  def get_precision(self):
    try:
      return self.precision.score()
    except:
      print("! Precision: Dataset group lacks representation of all classes !")
      return -1
  def get_recall(self):
    try:
      return self.recall.score()
    except:
      print("! Recall: Dataset group lacks representation of all classes !")
      return -1
  def get_f1score(self):
    try:
      return self.f1score.score()
    except:
      print("! F1Score: Dataset group lacks representation of all classes !")
      return -1
  def get_confusionMatrix(self, path_to):
    try:
      return self.confusionMatrix.score(path_to)
    except Exception as e:
      print("Confusion matrix could not be created:", e)
      print("Test Split probably lacks representation of all classes.")
      return None

  
def get_testState(path_states, modelId):
  # problem if 2 models with same id (e.g. last and best state)
  fold = modelId.split("_")[-1][1]  # get fold number from modelId
  for stateId in os.listdir(path_states):
    if fold in stateId:
      path_state = os.path.join(path_states, stateId)
      print(f"    Loading state from model {stateId}")
      stateDict = torch.load(path_state, weights_only=False)
      stateId = os.path.splitext(stateId)[0].split('_')[1]  # remove .pt and state_
      return stateDict, stateId
  else:
    raise ValueError(f"Model {modelId} not found in {path_states}")

def get_path_spaceFeat(path_features, featureName):
  # problem if 2 models with same id (e.g. last and best state)
  # print(featureName)
  for featureId in os.listdir(path_features):
    if featureName in featureId:
      path_features = os.path.join(path_features, featureId)
      print(f"    Found feature path! ({featureId})")
      return path_features, featureId
  else:
    print(f"    Feature path ({featureName}) not found in {path_features}\n")
    return None, None

def get_pred(path_preds, fold):
  for predId in os.listdir(path_preds):
    # print(predId)
    if str(fold) in predId and os.path.splitext(predId)[1] == '.pt':
      path_pred = os.path.join(path_preds, predId)
      print(f"    Found prediction path! ({predId})")
      return os.path.splitext(predId)[0]
  else:
    print(f"    Prediction path (fold{fold}) not found in {path_preds}\n")
    return None