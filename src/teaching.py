import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from src import machine
import torch
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import pandas as pd

class Teacher():
  ''' Each Teacher teaches (action = train) 'k_folds' models from 1 dataset, besides saving
      multiple checkpoint models along the way (for each fold)
      It also can evaluate (action = eval) the models.
      In summary, a Teacher is a Trainer, a Validater and a Tester
      In - split and batched dataset from a Cataloguer
      Action - teach, eval
      Pipeline: (train -> validate) x n_epochs -> test
  '''
  def __init__(self, TRAIN, EVAL, Ctl, DEVICE):
    train_metric, valid_metric, test_metrics = self._get_metrics(TRAIN, EVAL, Ctl.dataset.N_CLASSES, Ctl.labelToClass, DEVICE)
    criterion = self._get_criterion(TRAIN["criterionId"], Ctl.classWeights, DEVICE)
    self.trainer = Trainer(train_metric, criterion, TRAIN["HYPER"], DEVICE)
    self.validater = Validater(valid_metric, criterion, DEVICE)
    self.tester = Tester(test_metrics, DEVICE)

    self.highScore = -np.inf
    self.bestState = {}
    
  def teach(self, model, trainloader, validloader, n_epochs, path_states, path_resume=None):
    ''' Iterate through folds and epochs of model learning with training and validation
        In: Untrained model, data, etc
        Out: Trained model (state_dict)
    '''
    if not list(model.parameters()):
      raise ValueError("Model parameters are empty/invalid. Check model initialization.")
    earlyStopper = EarlyStopper()
    optimizer = optim.SGD(model.parameters(), lr=self.trainer.learningRate, momentum=self.trainer.MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.STEP_SIZE, gamma=self.trainer.GAMMA) # updates the optimizer by the GAMMA factor after the step_size
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    betterState, valid_minLoss = {}, np.inf
    earlyStopper.reset()
    if path_resume: # Loading checkpoint before resuming training
      model, optimizer, startEpoch, cpLoss = self.load_checkpoint(model, optimizer, path_resume)
    else:
      startEpoch = 0

    for epoch in range(startEpoch, n_epochs):
      print(f"-------------- Epoch {epoch + 1} --------------")
      train_loss, train_score = self.trainer.train(model, trainloader, optimizer, epoch)
      # assert 1 == 0
      valid_loss, valid_score = self.validater.validate(model, validloader, epoch)

      self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
      self.writer.add_scalar("Loss/valid", valid_loss, epoch + 1)
      self.writer.add_scalar("Acc/train", train_score, epoch + 1)
      self.writer.add_scalar("Acc/valid", valid_score, epoch + 1)
      print(f"Train Loss: {train_loss:4f}\tValid Loss: {valid_loss:4f}")
      scheduler.step()

      if earlyStopper.early_stop(valid_loss):
        print(f"\n* Stopped early *\n")
        break
      if valid_loss <= valid_minLoss:
        print(f"\n* New best model (valid loss): {valid_minLoss:.4f} --> {valid_loss:.4f} *\n")
        valid_minLoss = valid_loss
        valid_maxScore = valid_score
        betterState = model.state_dict()
      if ((epoch + 1) % 5) == 0:  # Save checkpoint every 5 epochs
        print(f"\n* Checkpoint reached ({epoch + 1}/{n_epochs}) *\n")
        path_checkpoint = os.path.join(path_states, f"{model.id}_cp{epoch + 1}.pth")
        self.save_checkpoint(model, optimizer, epoch, valid_loss, path_checkpoint)
      model.valid_score = valid_score
    return model, valid_maxScore, betterState

  def evaluate(self, test_bundle, EVAL, fold, date, N_CLASSES, labelToClass, path_eval=None, path_aprfc=None, path_phaseCharts=None):
    # print(predictions[0], targets[0], imageIds[0])
    # torch.nn.functional.one_hot(tensor, N_CLASSES=-1)
    if EVAL["export_testBundle"]:
      path_to = os.path.join(path_eval, f"predictions_fold{fold}")
      with open(path_to, 'wb') as f: # Save the DataFrame
        pickle.dump(test_bundle, f)

    if EVAL["aprfc"]:
      self.tester._aprfc(self.writer, path_aprfc, fold, test_bundle)
    if EVAL["phaseTiming"]:
      outText = self.tester._phase_lengths(test_bundle, N_CLASSES, labelToClass)
      self.tester._aprfc(self.writer, path_aprfc, fold, test_bundle)
      # print(list(final_bundle.groupby("Video")))
      outText = self.tester._phase_lengths(test_bundle, N_CLASSES, labelToClass, test_bundle)
    if EVAL["phaseChart"]:
      self.tester._phase_graph(test_bundle, path_phaseCharts, fold, date, outText, test_bundle)

  def evaluate2(self, model, testloader, EVAL, extradata=None, path_to=None):
    # _, ext = os.path.splitext(path_from)
    # if ext == ".pkl": # pickle file for results
    #   with open(path_from, 'rb') as f:  # Load the DataFrame
    #     test_bundle = pickle.load(f)
    # elif ext == ".pth": # torch model to eval
    # model = torch.load(path_from, weights_only=False)
    t1 = time.time()
    preds, targets, extradata = self.tester.test(model, testloader, extradata)
    t2 = time.time()
    print(f"Testing took {t2 - t1:.2f} seconds")
    # self.tester.test_metrics.display()
    # print(sampleNames[0], preds[0], targets[0])
    # torch.nn.functional.one_hot(tensor, N_CLASSES=-1)
    test_bundle = self._get_bundle(preds, targets, extradata)
    if path_to:
      with open(path_to, 'wb') as f: # Save the DataFrame
        pickle.dump(test_bundle, f)

    self._do_tests(test_bundle, EVAL)
 

  def _do_tests(self, test_bundle, testkey):
    testsAvailable = {
      "acpref1com": self.tester.acpref1com,
      "timing": self.tester.phaseTiming,
      "chart": self.tester.phaseSegmentationChart
    }
    if testkey in testsAvailable:  # if provided a testkey, do that test
      testsAvailable[testkey](test_bundle)
    else: # do them all
      for action in testsAvailable.values():
        action(test_bundle)


  def _get_metrics(self, TRAIN, EVAL, N_CLASSES, labelToClass, DEVICE):
    train_metric = RunningMetric(TRAIN["train_metric"], N_CLASSES, DEVICE, EVAL["agg"], EVAL["computeRate"], EVAL["updateRate"])
    valid_metric = RunningMetric(TRAIN["valid_metric"], N_CLASSES, DEVICE, EVAL["agg"], EVAL["computeRate"] // 2, EVAL["updateRate"])
    test_metrics = MetricBunch(EVAL["test_metrics"], N_CLASSES, labelToClass, EVAL["agg"], EVAL["computeRate"], DEVICE)
    return train_metric, valid_metric, test_metrics

  def _get_criterion(self, CRITERION_ID, classWeights, DEVICE):
    if CRITERION_ID == "cross-entropy":
      return nn.CrossEntropyLoss()
    elif CRITERION_ID == "weighted-cross-entropy":
      return nn.CrossEntropyLoss(weight=classWeights.to(DEVICE))
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
  def __init__(self, metric, criterion, HYPER, DEVICE):
    self.train_metric = metric
    self.criterion = criterion
    self.learningRate = HYPER["learningRate"]
    self.MOMENTUM = HYPER["momentum"]
    self.STEP_SIZE = HYPER["stepSize"]
    self.GAMMA = HYPER["gamma"]
    self.DEVICE = DEVICE
    
  def train(self, model, trainloader, optimizer, epoch):
    ''' In: model, data, criterion (loss function), optimizer
        Out: train_loss, train_score
        Note - inputs can be samples (pics) or feature maps
               Targets == labels
               outputs can be predictions or logits?
    '''
    self.train_metric.reset()  # resets states, typically called between epochs
    runningLoss = 0.0
    model.train().to(self.DEVICE)
    for batch, data in enumerate(trainloader, 0):   # start at batch 0
      inputs, targets = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # data is a list of [inputs, labels]
      # print("inputs: ", inputs.shape, "\ttargets: ", targets.shape, '\n')
      optimizer.zero_grad() # reset the parameter gradients before backward step
      outputs = model(inputs) # forward pass
      # print("\noutputs: ", outputs.shape)
      loss = self.criterion(outputs, targets) # calculate loss  # -1 for getting last stage guesses
      loss.backward() # backward pass
      optimizer.step()    # a single optimization step
      runningLoss += loss.item() # accumulate loss
      self.train_metric.update(outputs, targets)
      if batch % self.train_metric.computeRate == 0:
        print(f"\t  T [E{epoch + 1} B{batch + 1}]   scr: {self.train_metric.score():.4f}")
    return runningLoss / len(trainloader), self.train_metric.score()

class Validater():
  ''' Part of the teacher that knows how to validate a run of model training
  '''
  def __init__(self, metric, criterion, DEVICE):
    self.valid_metric = metric
    self.criterion = criterion
    self.DEVICE = DEVICE

  def validate(self, model, validloader, epoch):
    ''' In: model, data, criterion (loss function)
        Out: valid_loss, valid_score
    '''
    self.valid_metric.reset() # Running metric, e.g. accuracy
    runningLoss = 0.0
    model.eval()
    with torch.no_grad(): # don't calculate gradients (for learning only)
      for batch, data in enumerate(validloader, 0):
        inputs, targets = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
        outputs = model(inputs)
        # print(labels[:5], outputs[:5])
        loss = self.criterion(outputs, targets)
        runningLoss += loss.item()
        self.valid_metric.update(outputs, targets)  # updates with data from new batch
        if batch % self.valid_metric.computeRate == 0:
          print(f"\t  V [E{epoch + 1} B{batch + 1}]   scr: {self.valid_metric.score():.4f}")
    return runningLoss / len(validloader), self.valid_metric.score()

class Tester():
  ''' Part of the teacher that knows how to test a model knowledge in multiple ways
  '''
  def __init__(self, metrics, DEVICE):
    self.test_metrics = metrics
    self.DEVICE = DEVICE
  
  def test(self, model, testloader, extradata=None, path_export=None):
    ''' Test the model - return Preds, labels and sampleNames
    '''
    self.test_metrics.reset()
    predsList = []
    targetsList = []
    sampleIdsList = []
    model.eval().to(self.DEVICE); print("\n\tTesting...")
    print(extradata)
    with torch.no_grad():
      for batch, data in enumerate(testloader):
        if extradata["id"]:
          inputs, targets, sampleNames = data[0].to(self.DEVICE), data[1].to(self.DEVICE), data[2]
          sampleIdsList.extend(sampleNames)
        else:
          inputs, targets = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
        outputs = model(inputs).squeeze(0).permute(1, 0)
        _, preds = torch.max(outputs, 1)  # get labels with max prediction values

        predsList.append(preds)
        targetsList.append(targets)
        self.test_metrics.update(outputs, targets)
        if batch % self.test_metrics.accuracy.computeRate == 0:
          print(f"\t  T [B{batch + 1}]   scr: {self.test_metrics.accuracy.score():.4f}")
    predsList = torch.cat(predsList).cpu().tolist()  # flattens the list of tensor
    targetsList = torch.cat(targetsList).cpu().tolist()
    # sampleIdsList = [imgId for batch in sampleIds for imgId in batch]
    test_bundle = self._get_bundle(predsList, targetsList, sampleIdsList)
    if path_export:
      with open(path_export, 'wb') as f:
        pickle.dump(test_bundle, f)
    # print(len(predsList), len(targetsList), len(sampleIdsList))
    # print(f"SampleIds Type: {type(sampleIdsList[0])}, Content: {sampleIdsList[:5]}")
    return test_bundle

  def _get_bundle(self, preds, targets, extradata):
    preBundle = {
      "Preds": preds,
      'Targets': targets,
      'SampleName': extradata
    }
    # if extradata:
    #   preBundle.update(extradata)
    # print(len(preds), len(targets), len(extradata))
    # print(f"Sample Names Type: {type(extradata[0])}, Content: {extradata[:5]}")
    bundle = pd.DataFrame(preBundle)
    # print(bundle.head())

    # assuming frames in the structure of videoName_relativeTime
    bundle[['Video', 'Timestamp']] = bundle["SampleName"].str.split('_', expand=True) # expand creates two columns, instead of just leaving data as a split list
    bundle["Timestamp"] = bundle["Timestamp"].astype(int)
    bundle = bundle.sort_values(by=['Video', 'Timestamp']) # Sort the DataFrame by VIDEO and TIME
    #bundle['AbsoluteTime'] = bundle.groupby('Video').cumcount() # Convert Relative Time to Absolute Time
    return bundle
  
  def _aprfc(self, writer, path_to, fold, path_aprfc):
    # self.test_metrics.display()
    writer.add_scalar(f"accuracy/test", self.test_metrics.get_accuracy(), fold)
    writer.add_scalar(f"precision/test", self.test_metrics.get_precision(), fold)
    writer.add_scalar(f"recall/test", self.test_metrics.get_recall(), fold)
    writer.add_scalar(f"f1score/test", self.test_metrics.get_f1score(), fold)
    image_cf = plt.imread(self.test_metrics.get_confusionMatrix(path_to, path_aprfc))
    tensor_cf = torch.tensor(image_cf).permute(2, 0, 1)
    writer.add_image(f"confusion_matrix/test", tensor_cf, fold)
    writer.close()
    
  def _phase_lengths_i2(self, df, N_CLASSES, labelToClass, samplerate=1):
    outText = []
    lp, lt = len(df["Preds"]), len(df["Targets"])
    # N_CLASSES = (max(df["Targets"]) + 1)  # assuming regular int encoding
    # N_CLASSES = len(preds[0]) # assuming one hot encoding
    assert lp == lt, "Targets and Preds Size Incompatibility"
    videos = df["Video"].unique()
    phases_preds_sum = np.zeros(N_CLASSES)
    phases_gt_sum = np.zeros(N_CLASSES)
    phases_diff_sum = np.zeros(N_CLASSES)

    for idx, video in enumerate(videos):
      print()
      df_filtered = df[df["Video"] == video][["Preds", "Targets"]]
      
      classes_video = pd.Series(df_filtered["Targets"].unique())
      # pd.set_option("display.max_rows", 100)
      # print(df_filtered["Targets"].iloc[:100])
      print(classes_video.reindex(range(N_CLASSES), fill_value=-1).values)
      # Reset for each video
      
      phases_preds_video = df_filtered["Preds"].value_counts().reindex(range(N_CLASSES), fill_value=0) # Targets video counts
      phases_gt_video = df_filtered["Targets"].value_counts().reindex(range(N_CLASSES), fill_value=0) # Targets video counts
      # ensures every class is accounted for in cumulative count, even with no occurrences
      phases_preds_sum += phases_preds_video.values
      phases_gt_sum += phases_gt_video.values
      phases_diff_sum += np.abs(phases_preds_video.values - phases_gt_video.values)
      # print(phases_diff_sum)
      
      # Optionally print counts for each video
      print("phases_preds counts:", phases_preds_video.values, '\t', sum(phases_preds_video))
      print("phases_gt counts:", phases_gt_video.to_numpy(), '\t', sum(phases_gt_video.to_numpy()))
      
      mix = list(zip(phases_preds_video / (60 * samplerate), phases_gt_video / (60 * samplerate), phases_diff_sum))
      ot = f"\nVideo {video} Phase Report\n" + f"{'':<12} |   T Preds    |    T Targets     ||  Diff\n" + '\n'.join(
            f"{labelToClass[i]:<12} | {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][0] - mix[i][1]):<8.2f}min"
            for i in range(N_CLASSES)
        )
      outText.append(ot)
      print(ot)
    # Average Overall Results
    phases_preds_avg = phases_preds_sum / len(videos) / (60 * samplerate)
    phases_gt_avg = phases_gt_sum / len(videos) / (60 * samplerate)
    phases_diff_avg = phases_diff_sum / len(videos) / (60 * samplerate)
    mix = list(zip(phases_preds_avg, phases_gt_avg, phases_diff_avg))
    
    otavg = "\n\nOverall Average Phase Report\n" + f"{'':<12} | T Avg Preds  |  T Avg Targets   || Total AE\n" + '\n'.join(
            f"{labelToClass[i]:<12} | {mix[i][0]:<8.2f}min | {mix[i][1]:<8.2f}min || {(mix[i][2]):<8.2f}min"
            for i in range(N_CLASSES)
      )
    print(otavg)
    outText.append(otavg)
  # phase_metric([1, 1, 2, 3, 3, 4, 0, 5], [0, 0, 0, 2, 3, 4, 5, 1])
    return outText

  def _phase_lengths(self, df, N_CLASSES, labelToClass, samplerate=1):
    return self._phase_lengths_i2(df, N_CLASSES, labelToClass, samplerate=1)

  def _phase_graph(self, df, path_out, fold, date, outText, path_phaseCharts):
      # df = pd.DataFrame({
      #     'SampleName': ['vid1_1', 'vid1_3', 'vid1_2', 'vid2_20', 'vid2_23', 'vid2_40', 'vid2_51', 'vid2_52'],
      #     'Targets': [0, 1, 1, 2, 0, 1, 1, 1],  # Example target labels
      #     'Preds': [0, 1, 0, 2, 1, 4, 3, 2],  # Example predictions (e.g., class indices)

      # })
      # Prepare for plotting
      videos = df['Video'].unique()
      num_videos = len(videos)
      print(f'\n\nNumber of videos: {num_videos}')
      color_map = {0: 'springgreen', 1: 'goldenrod', 2: 'tomato', 3: 'mediumslateblue', 4: 'plum', 5: 'deepskyblue'}
      plt.rcParams['font.family'] = 'monospace'
      # for saving all videos to same img
      # fig, axes = plt.subplots(num_videos, 1, figsize=(9, 5), sharex=True) # Create subplots
      # if num_videos == 1: axes = [axes] # prevent errors if only 1 video
      t1 = time.time()
      # Plotting
      for idx, video in enumerate(videos):

        data_video = df[df['Video'] == video][["Targets", "Preds"]]
        fig, (ax, ax_text) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(9, 6))
          
        # ax = axes[idx]  # for saving all videos to same image
        
        # Prepare the bar data
        for j, stage in enumerate(data_video.columns):
          start_positions = np.arange(len(data_video))
          class_values = data_video[stage].values
          ax.broken_barh(list(zip(start_positions, np.ones(len(start_positions)))), (j - 0.4, 0.8), facecolors=[color_map[val] for val in class_values])

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

      
        # Create custom legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[i]) for i in color_map.keys()]
        labels = [f'{i}' for i in color_map.keys()]
        ax.legend(handles, labels, loc='upper right', title='Classes')

        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(os.path.join(path_out, f"{date}_{os.path.basename(path_phaseCharts)}_{video[:4]}.png"))
        plt.close(fig)

      plt.figure()
      fig, ax_text = plt.subplots(figsize=(6, 2))
      ax_text.axis('off')  # Hide axis for the text box
      ot = outText[-1].replace('\t', '      ').replace('\u200b', '')  # Removes hidden tabs and zero-width spaces
      # print(ot)
      ax_text.text(0, 0.5, ot, transform=ax_text.transAxes, fontsize=11, 
                  verticalalignment='center', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
      plt.tight_layout()
      plt.show(block=True)
      plt.savefig(os.path.join(path_out, f"{date}_{os.path.basename(path_phaseCharts)}_oa-avg-phase-rep.png"))
      plt.close()

      t2 = time.time()
      print(f"Video graphing took {t2 - t1:.2f} seconds")
      print(f"\nSaved phase metric graphs to {path_out}")

class EarlyStopper:
  ''' Controls the validation loss progress to keep it going down and improve times
  '''
  def __init__(self, patience=3, minDelta=0.05):
    self.patience = patience
    self.minDelta = minDelta  # percentual difference in decrease, 0 for no margin
    self.valid_minLoss = np.inf
    self.counter = 0

  def reset(self):
    self.valid_minLoss = np.inf
    self.counter = 0
  def early_stop(self, valid_loss):
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
  def __init__(self, metricName, N_CLASSES, DEVICE, agg="micro"):
    if metricName == "accuracy":
      self.metric = MulticlassAccuracy(average="micro", num_classes=N_CLASSES, device=DEVICE)
    elif metricName == "precision":
      self.metric =  MulticlassPrecision(average=agg, num_classes=N_CLASSES, device=DEVICE)
    elif metricName == "recall":
      self.metric = MulticlassRecall(average=agg, num_classes=N_CLASSES, device=DEVICE)
    elif metricName == "f1score":
      self.metric = MulticlassF1Score(average=agg, num_classes=N_CLASSES, device=DEVICE)
    elif metricName == "confusionMatrix":
      self.metric = MulticlassConfusionMatrix(num_classes=N_CLASSES, device=DEVICE)
    else:
      raise ValueError("Invalid Metric Name")
  def reset(self):
    self.metric.reset()
  def score(self):
    return self.metric.compute().item()
  
class RunningMetric(StillMetric):
  ''' Extends StillMetric to updates on the run
  '''
  def __init__(self, metricName, N_CLASSES, DEVICE, agg, computeRate, updateFreq=1):
    super().__init__(metricName, N_CLASSES, DEVICE, agg)
    self.computeRate = computeRate  # num of batches between computations
    self.updateFreq = updateFreq  # num of batches between updates
    
  # def running(self, outputs, labels, batch, epoch):
  #   if (batch + 1) % self.updateFreq == 0:
  #     self.metric.update(outputs, labels)
  #   if (batch + 1) % self.computeRate == 0:
  #     metric_result = self.metric.compute().item()
  #     print(f"\t  [E{epoch + 1} B{batch + 1}]   scr: {metric_result:.4f}")
  #     return metric_result

  def update(self, outputs, targets):
    self.metric.update(outputs, targets)


class MetricBunch:
  ''' Gets a bunch of still metrics at cheap price
      Accuracy locked at agg==micro for now
  '''
  def __init__(self, metricSwitches, N_CLASSES, labelToClass, agg, computeRate, DEVICE):  # metricSwitches is a boolean dict switch for getting metrics
    self.metricSwitches = metricSwitches
    self.labelToClass = labelToClass
    self.agg = agg
    
    if self.metricSwitches["accuracy"]:
      self.accuracy = RunningMetric("accuracy", N_CLASSES, DEVICE, "micro", computeRate)
    if self.metricSwitches["precision"]:
      self.precision = RunningMetric("precision", N_CLASSES, DEVICE, agg, computeRate)
    if self.metricSwitches["recall"]:
      self.recall = RunningMetric("recall", N_CLASSES, DEVICE, agg, computeRate)
    if self.metricSwitches["f1score"]:
      self.f1score = RunningMetric("f1score", N_CLASSES, DEVICE, agg, computeRate)
    if self.metricSwitches["confusionMatrix"]:
      self.confusionMatrix = RunningMetric("confusionMatrix", N_CLASSES, DEVICE, agg, computeRate)

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
                  xticklabels=self.labelToClass.values(), yticklabels=self.labelToClass.values(), ax=ax)
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
  