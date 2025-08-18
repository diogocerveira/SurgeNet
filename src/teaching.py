import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy, MultilabelAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from src import machine
import torch
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnchoredOffsetbox
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
    self.N_PHASES = dataset.n_classes
    self.PHASES = dataset.phases
    self.labelType = dataset.labelType
    self.DATASET_SIZE = dataset.__len__()
    train_metric, valid_metric, test_metrics = self._get_metrics(TRAIN, EVAL, self.N_PHASES, dataset.labelToClassMap, dataset.labelType, DEVICE)
    criterion = self._get_criterion(TRAIN["criterionId"], dataset.classWeights, DEVICE)
    
    
    self.trainer = Trainer(train_metric, criterion, TRAIN, DEVICE)
    self.validater = Validater(valid_metric, criterion, DEVICE)
    self.tester = Tester(test_metrics, dataset.labels, self.PHASES, dataset.labelToClassMap, DEVICE)

    self.save_checkpoints = TRAIN["save_checkpoints"]
    self.highScore = -np.inf
    self.bestState = {}
    self.minDelta = TRAIN["HYPER"]["minDelta"]  # minimal delta for early stopping
    self.patience = TRAIN["HYPER"]["patience"]  # patience for early stopping

  def teach(self, model, trainloader, validloader, n_epochs, path_states, path_resume=None):
    ''' Iterate through folds and epochs of model learning with training and validation
        In: Untrained model, data, etc
        Out: Trained model (state_dict)
    '''
    if not list(model.parameters()):
      raise ValueError("Model parameters are empty/invalid. Check model initialization.")
    optimizer = self.trainer.optimizer
    scheduler = self.trainer.scheduler
    earlyStopper = EarlyStopper(self.patience, self.minDelta)

    betterState, valid_minLoss = {}, np.inf
    earlyStopper.reset()
    if path_resume: # Loading checkpoint before resuming training
      model, optimizer, startEpoch, cpLoss = self.load_checkpoint(model, optimizer, path_resume)
    else:
      startEpoch = 0

    for epoch in range(startEpoch, n_epochs):
      print(f"-------------- Epoch {epoch + 1} --------------")
      train_loss, train_score = self.trainer.train(model, trainloader, self.labelType, optimizer, epoch)
      valid_loss, valid_score = self.validater.validate(model, validloader, self.labelType, epoch)

      self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
      self.writer.add_scalar("Loss/valid", valid_loss, epoch + 1)
      self.writer.add_scalar("Acc/train", train_score, epoch + 1)
      self.writer.add_scalar("Acc/valid", valid_score, epoch + 1)
      print(f"Train Loss: {train_loss:4f}\tValid Loss: {valid_loss:4f}")
      scheduler.step(valid_loss) # Adjust learning rate based on validation loss

      if (valid_minLoss - valid_loss) > earlyStopper.minDelta:  # if valid loss decreased by more than minDelta == good
        print(f"\n* New best model (valid loss): {valid_minLoss:.4f} --> {valid_loss:.4f} *\n")
        valid_minLoss = valid_loss
        valid_maxScore = valid_score
        betterState = model.state_dict()

      if earlyStopper.early_stop(valid_loss): # same as prev if but for a [patience] number of epochs in a row
        print(f"\n* Stopped early *\n")
        break
        
      if ((epoch + 1) % 5) == 0:  # Save checkpoint every 5 epochs
        if self.save_checkpoints:
          print(f"\n* Checkpoint reached ({epoch + 1}/{n_epochs}) *\n")
          path_checkpoint = os.path.join(path_states, f"{model.id}_cp{epoch + 1}.pth")
          self.save_checkpoint(model, optimizer, epoch, valid_loss, path_checkpoint)
      model.valid_lastScore = valid_score

      # clear GPU memory
      torch.cuda.empty_cache()
      gc.collect()
    return model, valid_maxScore, betterState

  def evaluate(self, test_bundle, eval_tests, modelId, Csr):

    if eval_tests["aprfc"]:
      self.tester._aprfc(self.writer, modelId, Csr.path_events, Csr.path_aprfc)
      # clear GPU memory
      torch.cuda.empty_cache()
      gc.collect()
    if eval_tests["phaseTiming"]:
      outText  = self.tester._phase_timing(test_bundle, Csr.path_phaseTiming, modelId)
      # clear GPU memory
      torch.cuda.empty_cache()
      gc.collect()
      if eval_tests["phaseChart"]:
        self.tester._phase_graph(test_bundle, Csr.path_phaseCharts, modelId, outText)
        # clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

  def _get_metrics(self, TRAIN, EVAL, N_PHASES, labelToClass, labelType, DEVICE):
    
    train_metric = RunningMetric(TRAIN["train_metric"], N_PHASES, DEVICE, EVAL["agg"], EVAL["computeRate"], labelType, EVAL["updateRate"])
    valid_metric = RunningMetric(TRAIN["valid_metric"], N_PHASES, DEVICE, EVAL["agg"], EVAL["computeRate"], labelType, EVAL["updateRate"])
    test_metrics = MetricBunch(EVAL["test_metrics"], N_PHASES, labelToClass, EVAL["agg"], EVAL["computeRate"], DEVICE, labelType)
    return train_metric, valid_metric, test_metrics

  def _get_criterion(self, CRITERION_ID, classWeights, DEVICE):
    if CRITERION_ID == "cross-entropy":
      return nn.CrossEntropyLoss()
    elif CRITERION_ID == "weighted-cross-entropy":
      return nn.CrossEntropyLoss(weight=classWeights.to(DEVICE))
    elif CRITERION_ID == "binary-cross-entropy-logits":
      classCounts = 1 / classWeights # inverse frequency
      return nn.BCEWithLogitsLoss(pos_weight=classWeights.to(DEVICE))
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

class Trainer():
  ''' Part of the teacher that knows how to train models based on data
  '''
  def __init__(self, metric, criterion, TRAIN, DEVICE):
    self.train_metric = metric
    self.criterion = criterion

    self.learningRate = TRAIN["HYPER"]["learningRate"]
    self.MOMENTUM = TRAIN["HYPER"]["momentum"]
    self.STEP_SIZE = TRAIN["HYPER"]["stepSize"]
    self.GAMMA = TRAIN["HYPER"]["gamma"]

    self.optimizer = self._get_optimizer(TRAIN["optimizer"], self.learningRate, TRAIN["HYPER"]["momentum"])
    self.scheduler = self._get_scheduler(TRAIN["scheduler"], self.STEP_SIZE, self.GAMMA)
    self.DEVICE = DEVICE
  
  def _get_optimizer(self, optimizerId, lr, momentum):
  if optimizerId == "adam":
    return optim.Adam(lr=lr)
  elif optimizerId == "sgd":
    return optim.SGD(lr=lr, momentum=momentum)
  else:
    raise ValueError("Invalid optimizer chosen (adam, sgd)")
  def _get_scheduler(self, schedulerId, stepSize, gamma):
    if schedulerId == "step":
      scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=gamma)
    elif schedulerId == "cosine":
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stepSize, eta_min=gamma)
    elif schedulerId == "plateau":
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='min',       # monitor loss (want to minimize)
        factor=0.1,       # scale lr by this factor (10x smaller)
        patience=3,       # wait 3 epochs without improvement
        verbose=True      # log when lr changes
    )
    else:
      raise ValueError("Invalid scheduler chosen (step, cosine)")
    return scheduler

  def train(self, model, trainloader, labelType, optimizer, epoch):
    ''' In: model, data, criterion (loss function), optimizer
        Out: train_loss, train_score
        Note - inputs can be samples (pics) or feature maps / outputs are logits and then pred/probabilities
    '''
    self.train_metric.reset()  # resets states, typically called between epochs
    runningLoss, totalSamples = 0.0, 0
    model.train().to(self.DEVICE)
    for batch, data in enumerate(trainloader, 0):   # start at batch 0
      inputs, targets = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # data is a list of [inputs, labels]
      # print("inputs: ", inputs.shape, "\ttargets: ", targets.shape, '\n')
      # print(targets[:5])  # show a few samples
      optimizer.zero_grad() # reset the parameter gradients before backward step
      outputs = model(inputs) # forward pass
      # print(inputs[:5], targets[:5], outputs[:5])
      # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
      if model.inputType.split('-')[1] == "clipped":
        outputs = outputs.reshape(-1, outputs.shape[-1])
        if labelType.split('-')[0] == "single":
          targets = targets.reshape(-1)
          mask = targets != -1
          assert isinstance(self.criterion, torch.nn.CrossEntropyLoss), "Use CrossEntropyLoss for single-label tasks"
        else:  # multi
          targets = targets.reshape(-1, targets.shape[-1])
          mask = targets[:, 1] != -1  # assuming class=-1 means padding
          assert isinstance(self.criterion, torch.nn.BCEWithLogitsLoss), "Use BCEWithLogitsLoss for multi-label tasks"
        # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
        outputs = outputs[mask]
        targets = targets[mask]

      # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
      # print("outputs: ", outputs[:5], "\ttargets: ", targets[:5], '\n')  
      loss = self.criterion(outputs, targets) # calculate loss
      loss.backward() # backward pass
      optimizer.step()    # a single optimization step

      if labelType.split('-')[0] == "single":
        outputs = torch.argmax(outputs, dim=1)  # get labels with max prediction values
      else:
        outputs = torch.sigmoid(outputs)  # get probabilities for Multi label accuracy
      self.train_metric.update(outputs, targets)

      runningLoss += loss.item() * targets.numel() # accumulate loss by batch size (safer, though most batches are same size)
      totalSamples += targets.numel()  # numel() returns the total number of elements in the tensor
      if batch % self.train_metric.computeRate == 0:
        print(f"\t  T [E{epoch + 1} B{batch + 1}]   scr: {self.train_metric.score():.4f}")
    return runningLoss / totalSamples, self.train_metric.score()

class Validater():
  ''' Part of the teacher that knows how to validate a run of model training
  '''
  def __init__(self, metric, criterion, DEVICE):
    self.valid_metric = metric
    self.criterion = criterion
    self.DEVICE = DEVICE

  def validate(self, model, validloader, labelType, epoch):
    ''' In: model, data, criterion (loss function)
        Out: valid_loss, valid_score
    '''
    self.valid_metric.reset() # Running metric, e.g. accuracy
    runningLoss, totalSamples = 0.0, 0
    model.eval()
    with torch.no_grad(): # don't calculate gradients (for learning only)
      for batch, data in enumerate(validloader, 0):
        inputs, targets = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
        outputs = model(inputs)
        # print(inputs[:5], targets[:5], outputs[:5])
        # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
        if model.inputType.split('-')[1] == "clipped":
          outputs = outputs.reshape(-1, outputs.shape[-1])
          if labelType.split('-')[0] == "single":
            targets = targets.reshape(-1)
            mask = targets != -1
            assert isinstance(self.criterion, torch.nn.CrossEntropyLoss), "Use CrossEntropyLoss for single-label tasks"
          else:  # multi
            targets = targets.reshape(-1, targets.shape[-1])
            mask = targets[:, 1] != -1  # assuming class=-1 means padding
            assert isinstance(self.criterion, torch.nn.BCEWithLogitsLoss), "Use BCEWithLogitsLoss for multi-label tasks"
          # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
          outputs = outputs[mask]
          targets = targets[mask]

        # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
        loss = self.criterion(outputs, targets)
        
        runningLoss += loss.item() * targets.numel() # accumulate loss by batch size (safer, though most batches are same size)
        totalSamples += targets.numel()
        if labelType.split('-')[0] == "single":
          outputs = torch.argmax(outputs, dim=1)  # get labels with max prediction values
        else:
          outputs = torch.sigmoid(outputs)  # get probabilities for Multi label accuracy
        self.valid_metric.update(outputs, targets)  # updates with data from new batch
        if batch % self.valid_metric.computeRate == 0:
          print(f"\t  V [E{epoch + 1} B{batch + 1}]   scr: {self.valid_metric.score():.4f}")
    return runningLoss / totalSamples, self.valid_metric.score()

class Tester():
  ''' Part of the teacher that knows how to test a model knowledge in multiple ways
  '''
  def __init__(self, metrics, labels, PHASES, labelToClassMap, DEVICE):
    self.test_metrics = metrics
    self.DEVICE = DEVICE
    self.labels = labels
    self.PHASES = PHASES
    self.labelToClassMap = labelToClassMap  # map from label to class index
  
  def test(self, model, testloader, labelType, export_bundle, path_export=None):
    ''' Test the model - return Preds, labels and sampleIds
    '''
    self.test_metrics.reset()
    predsList = []
    targetsList = []
    sampleIdsList = []
    model.eval().to(self.DEVICE); print("\n\tTesting...")
    with torch.no_grad():
      for batch, data in enumerate(testloader):
        inputs, targets, sampleIds = data[0].to(self.DEVICE), data[1].to(self.DEVICE), data[2].to(self.DEVICE)
        outputs = model(inputs)
        # print(inputs[:5], '\n', targets[:5], '\n', outputs[:5])
        if model.inputType.split('-')[1] == "clipped":
          outputs = outputs.reshape(-1, outputs.shape[-1])
          if labelType.split('-')[0] == "single":
            targets = targets.reshape(-1)
            mask = targets != -1
          else:  # multi
            targets = targets.reshape(-1, targets.shape[-1])
            mask = targets[:, 1] != -1  # assuming class=-1 means padding
          sampleIds = sampleIds.reshape(-1)
          # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')
          outputs = outputs[mask]
          targets = targets[mask]
          sampleIds = sampleIds[mask]
          # print("outputs: ", outputs.shape, "\ttargets: ", targets.shape, '\n')

        # print("outputs.shape", outputs.shape)
        if labelType.split('-')[0] == "single":
          outputs = torch.argmax(outputs, dim=1)  # get labels with max prediction values
          preds = outputs
        else:
          outputs = torch.sigmoid(outputs)  # get probabilities
          topk = torch.topk(outputs, k=2, dim=1).indices
          preds = torch.zeros_like(outputs)
          preds.scatter_(1, topk, 1)
          # print(preds[:5], targets[:5])

        predsList.append(preds)
        targetsList.append(targets)
        sampleIdsList.append(sampleIds)
        # print("pred: ", outputs[:5], '\n', "target: ", targets[:5])
        # print("outputs.shape", outputs.shape)
        # print("targets.min()", targets.min().item(), "targets.max()", targets.max().item())
        self.test_metrics.update(outputs, targets)
        if batch % self.test_metrics.accuracy.computeRate == 0:
          print(f"\t  T [B{batch + 1}]   scr: {self.test_metrics.accuracy.score():.4f}")
    predsList = torch.cat(predsList).cpu().tolist()  # flattens the list of tensor
    targetsList = torch.cat(targetsList).cpu().tolist()
    sampleIdsList = torch.cat(sampleIdsList).cpu().tolist()

    # print(f"Preds: {predsList}\n Targets: {targetsList}\n SampleIds: {sampleIdsList}")
    test_bundle = self._get_bundle(predsList, targetsList, sampleIdsList, labelType)
    if export_bundle:
      # print("here", path_export)
      test_bundle.to_csv(path_export + ".csv", index=False)
      with open(path_export + ".pt", 'wb') as f:
        pickle.dump(test_bundle, f)
    # print(len(predsList), len(targetsList), len(sampleIdsList))
    # print(f"SampleIds Type: {type(sampleIdsList[0])}, Content: {sampleIdsList[:5]}")
    return test_bundle

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
    if labelType.split('-')[0] == "multi":
      bundle["Pred"] = bundle["Pred"].apply(lambda x: self.multihotToLabel(x, classToLabelMap))
      bundle["Target"] = bundle["Target"].apply(lambda x: self.multihotToLabel(x, classToLabelMap))
    elif labelType.split('-')[0] == "single":
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

  def _aprfc(self, writer, modelId, path_events, path_aprfc):
    # self.test_metrics.display()
    fold = int(modelId.split("_")[1][1]) # get fold number from modelId
    if self.test_metrics.accuracy:
      writer.add_scalar(f"accuracy/test", self.test_metrics.get_accuracy(), fold)
    if self.test_metrics.precision:
      writer.add_scalar(f"precision/test", self.test_metrics.get_precision(), fold)
    if self.test_metrics.recall:
      writer.add_scalar(f"recall/test", self.test_metrics.get_recall(), fold)
    if self.test_metrics.f1score:
      writer.add_scalar(f"f1score/test", self.test_metrics.get_f1score(), fold)
    if self.test_metrics.confusionMatrix:
      image_cf = plt.imread(self.test_metrics.get_confusionMatrix(path_events, path_aprfc))
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
      mix = list(zip(phases_preds_video / (60 * samplerate), phases_gt_video / (60 * samplerate), phases_diff_sum))
      ot = f"\nVideo {video[:4]} Phase Report\n" + f"{'':<12} \t|   T Preds   |  T Targets  ||  Diff\n" + '\n'.join(
            f"{phaseIntToLabelMap[i]:<12} \t| {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][0] - mix[i][1]):<8.2f}min"
            for i in range(N_PHASES)
        )
      outText.append(ot)
      print(ot, '\n')
      self._export_textImg(ot, path_phaseTiming, f"{modelId}_{video[:4]}-phaseTiming")
    # Average Overall Results
    phases_preds_avg = phases_preds_sum / len(videos) / (60 * samplerate)
    phases_gt_avg = phases_gt_sum / len(videos) / (60 * samplerate)
    phases_diff_avg = phases_diff_sum / len(videos) / (60 * samplerate)
    mix = list(zip(phases_preds_avg, phases_gt_avg, phases_diff_avg))
    
    otavg = "\n\nOverall Average Phase Report\n" + f"{'':<12} \t| T Avg Preds |T Avg Targets|| Total AE\n" + '\n'.join(
            f"{phaseIntToLabelMap[i]:<12} \t| {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][2]):<8.2f}min"
            for i in range(N_PHASES)
      )
    # print(otavg)
    # self._export_textImg(otavg, path_phaseTiming, f"{modelId}_overall-phaseTiming")
    outText.append(otavg)
  # phase_metric([1, 1, 2, 3, 3, 4, 0, 5], [0, 0, 0, 2, 3, 4, 5, 1])
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
    colormap, ordered_phases = self.get_phase_colormap_with_unknowns(df)
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
        ax.broken_barh(list(zip(start_positions, np.ones(len(start_positions)))), (j - 0.4, 0.8), facecolors=[colormap[val] for val in class_values])
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
    
    # Create custom legend handles and labels as before
    handles = [plt.Rectangle((0, 0), 1, 1, color=colormap[i]) for i in colormap.keys()]
    labels = [f'{i}' for i in colormap.keys()]

    # Legend outside axes, anchored to figure bottom-right
    legend = fig.legend(
        handles, labels,
        loc='lower right',                   # Align legend's lower right corner
        bbox_to_anchor=(0.98, 0.02),        # Just inside figure bottom right (x,y)
        bbox_transform=fig.transFigure,     # coords relative to figure, not axes
        borderaxespad=0.5,
        fontsize=9,
        title='Classes',
        frameon=True
    )
    plt.tight_layout()  # can keep to adjust spacing nicely
    plt.savefig(os.path.join(path_phaseCharts, f"{modelId}_{video[:4]}_phase-ev.png"), bbox_inches='tight')
    plt.close(fig)

    t2 = time.time()
    print(f"Video graphing took {t2 - t1:.2f} seconds")
    print(f"\nSaved phase metric graphs to {path_phaseCharts}")

  def get_phase_colormap_with_unknowns(self, df):
    known_set = set(self.PHASES)
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
  def __init__(self, metricName, N_PHASES, DEVICE, labelType, agg="micro"):
    self.name = metricName
  
    
    if metricName == "accuracy":
      if labelType.split('-')[0] == "single":
        self.metric = MulticlassAccuracy(average="micro", num_classes=N_PHASES, device=DEVICE)
      else:
        self.metric = MultilabelAccuracy(device=DEVICE) # criteria="hamming"
    elif metricName == "precision":
      self.metric =  MulticlassPrecision(average=agg, num_classes=N_PHASES, device=DEVICE)
    elif metricName == "recall":
      self.metric = MulticlassRecall(average=agg, num_classes=N_PHASES, device=DEVICE)
    elif metricName == "f1score":
      self.metric = MulticlassF1Score(average=agg, num_classes=N_PHASES, device=DEVICE)
    elif metricName == "confusionMatrix":
      self.metric = MulticlassConfusionMatrix(num_classes=N_PHASES, device=DEVICE)
    else:
      raise ValueError("Invalid Metric Name")

  def reset(self):
    self.metric.reset()
  def score(self):
    return self.metric.compute().item()
  
class RunningMetric(StillMetric):
  ''' Extends StillMetric to updates on the run
  '''
  def __init__(self, metricName, N_PHASES, DEVICE, agg, computeRate, labelType, updateFreq=1):
    super().__init__(metricName, N_PHASES, DEVICE, labelType, agg)
    self.computeRate = computeRate  # num of batches between computations
    self.updateFreq = updateFreq  # num of batches between updates
  
  def update(self, outputs, targets):      
    self.metric.update(outputs, targets)


class MetricBunch:
  ''' Gets a bunch of still metrics at cheap price
      Accuracy locked at agg==micro for now
  '''
  def __init__(self, metricSwitches, N_PHASES, labelToClass, agg, computeRate, DEVICE, labelType):  # metricSwitches is a boolean dict switch for getting metrics
    self.metricSwitches = metricSwitches
    self.labelToClass = labelToClass
    self.agg = agg
    
    if labelType.split('-')[0] == "single":
      self.accuracy = RunningMetric("accuracy", N_PHASES, DEVICE, "micro", computeRate, labelType) if self.metricSwitches["accuracy"] else None
      self.precision = RunningMetric("precision", N_PHASES, DEVICE, agg, computeRate, labelType) if self.metricSwitches["precision"] else None
      self.recall = RunningMetric("recall", N_PHASES, DEVICE, agg, computeRate, labelType) if self.metricSwitches["recall"] else None
      self.f1score = RunningMetric("f1score", N_PHASES, DEVICE, agg, computeRate, labelType) if self.metricSwitches["f1score"] else None
      self.confusionMatrix = RunningMetric("confusionMatrix", N_PHASES, DEVICE, agg, computeRate, labelType) if self.metricSwitches["confusionMatrix"] else None
    elif labelType.split('-')[0] == "multi":
      self.accuracy = RunningMetric("accuracy", N_PHASES, DEVICE, "micro", computeRate, labelType) if self.metricSwitches["accuracy"] else None
      self.precision = None
      self.recall = None
      self.f1score = None
      self.confusionMatrix = None
    else:
      raise ValueError(f"Invalid labelType {labelType} for metric bunch")
    
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
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_precision(self):
    try:
      return self.precision.score()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_recall(self):
    try:
      return self.recall.score()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_f1score(self):
    try:
      return self.f1score.score()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_confusionMatrix(self, path_to='', path_testBundle=''):
    try:
      fig, ax = plt.subplots(figsize=(10, 7))
      sns.heatmap(self.confusionMatrix.metric.compute().cpu(), annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=self.labelToClass.keys(), yticklabels=self.labelToClass.keys(), ax=ax)
      ax.set_xlabel('Predicted')
      ax.set_ylabel('True')
      ax.set_title('Confusion Matrix')
      if path_to:
        path_img = os.path.join(path_to, f"cm_{os.path.basename(path_testBundle)}.png")
        plt.savefig(path_img)
        return path_img
    except:
      print("! Dataset group lacks representation of all classes !")
      return path_to
  
def get_testState(path_states, modelId):
  # problem if 2 models with same id (e.g. last and best state)
  for stateId in os.listdir(path_states):
    if modelId in stateId:
      path_state = os.path.join(path_states, stateId)
      print(f"    Loading state from model {stateId}")
      stateDict = torch.load(path_state, weights_only=False)
      return stateDict, stateId
  else:
    raise ValueError(f"Model {modelId} not found in {path_states}")

def get_path_exportedfeatures(path_features, featureName):
  # problem if 2 models with same id (e.g. last and best state)
  for featureId in os.listdir(path_features):
    if featureName in featureId:
      path_features = os.path.join(path_features, featureId)
      print(f"    Found feature path! ({featureId})")
      return path_features, featureId
  else:
    print(f"Feature path ({featureName}) not found in {path_features}\n")
    return None, None