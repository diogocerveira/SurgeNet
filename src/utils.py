import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from torchvision.transforms import v2 as transforms
import os
import datetime
from collections import defaultdict

from . import train, validate

class EarlyStopper:
  ''' Controls the validation loss progress to keep it going down and improve times
  '''
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta  # difference in decrease, 0 for no margin
    self.min_loss_valid = np.inf
    self.counter = 0

  def reset(self):
    self.min_loss_valid = np.inf
    self.counter = 0
  def early_stop(self, loss_valid):
    if loss_valid < self.min_loss_valid:
      self.min_loss_valid = loss_valid
      self.counter = 0  # reset warning/patience counter if loss decreases betw epochs
    elif loss_valid - self.min_loss_valid >= self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  

class RunningMetric:
  ''' Perform metric update and computation based on a frequency provided
  '''
  def __init__(self, metric, period_compute, period_update=1):
    self.metric = metric  # torcheval metric object
    self.period_compute = period_compute  # num of batches between computations
    # self.period_update = period_update  # num of batches between updates
    
  # def running(self, outputs, labels, batch, epoch):
  #   if (batch + 1) % self.period_update == 0:
  #     self.metric.update(outputs, labels)
  #   if (batch + 1) % self.period_compute == 0:
  #     metric_result = self.metric.compute().item()
  #     print(f"\t  [E{epoch + 1} B{batch + 1}]   scr: {metric_result:.4f}")
  #     return metric_result
  def update(self, outputs, labels):
    self.metric.update(outputs, labels)
  def score(self):
    return self.metric.compute().item()
  
class BunchMetrics:
  ''' Gets a bunch of metrics at cheap price
  '''
  def __init__(self, metric_switches, num_classes, idx_to_class, agg, device):  # metric_switches is a boolean dict switch for getting metrics
    self.accuracy = MulticlassAccuracy(average="micro", num_classes=num_classes, device=device) if metric_switches["accuracy"] else None
    self.precision = MulticlassPrecision(average=agg, num_classes=num_classes, device=device) if metric_switches["precision"] else None
    self.recall = MulticlassRecall(average=agg, num_classes=num_classes, device=device) if metric_switches["recall"] else None
    self.f1score = MulticlassF1Score(average=agg, num_classes=num_classes, device=device) if metric_switches["f1score"] else None
    self.confusionMatrix = MulticlassConfusionMatrix(num_classes=num_classes, device=device) if metric_switches["confusionMatrix"] else None
    self.metric_switches = metric_switches
    self.idx_to_class = idx_to_class

  def reset(self):
    if self.accuracy: self.accuracy.reset()
    if self.precision: self.precision.reset()
    if self.recall: self.recall.reset()
    if self.f1score: self.f1score.reset()
    if self.confusionMatrix: self.confusionMatrix.reset()
    
  def update(self, predictions, labels):
    if self.accuracy: self.accuracy.update(predictions, labels)
    if self.precision: self.precision.update(predictions, labels)
    if self.recall: self.recall.update(predictions, labels)
    if self.f1score: self.f1score.update(predictions, labels)
    if self.confusionMatrix: self.confusionMatrix.update(predictions, labels)

  def display(self, metric_switches={"accuracy": 1, "precision": 1, "recall": 1, "f1score": 1, "confusionMatrix": 1}):
    if metric_switches["accuracy"]:
      print(f"Accuracy: {self.get_accuracy():.4f}")
    if metric_switches["precision"]:
      print(f"Precision: {self.get_precision():.4f}")
    if metric_switches["recall"]:
      print(f"Recall: {self.get_recall():.4f}")
    if metric_switches["f1score"]:
      print(f"F1 Score: {self.get_f1score():.4f}")
    if metric_switches["confusionMatrix"]:
      plt.show(self.get_confusionMatrix())

  def get_accuracy(self):
    try:
      return self.accuracy.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_precision(self):
    try:
      return self.precision.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_recall(self):
    try:
      return self.recall.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_f1score(self):
    try:
      return self.f1score.compute().item()
    except:
      print("! Dataset group lacks representation of all classes !")
      return -1
  def get_confusionMatrix(self, path_to=''):
    try:
      fig, ax = plt.subplots(figsize=(10, 7))
      sns.heatmap(self.confusionMatrix.compute().cpu(), annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=self.idx_to_class.values(), yticklabels=self.idx_to_class.values(), ax=ax)
      ax.set_xlabel('Predicted')
      ax.set_ylabel('True')
      ax.set_title('Confusion Matrix')
      if path_to:
        path_img = os.path.join(path_to, os.path.basename(path_to) + ".png")
        plt.savefig(path_img)
        return path_img
    except:
      print("! Dataset group lacks representation of all classes !")
      return path_to
    
    
def get_preprocessing():
  return transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToImage(), # only for v2
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # transforms.CenterCrop(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def get_augmentation():
  return transforms.Compose([
    # from 0-255 Image/numpy.ndarray to 0.0-1.0 torch.FloatTensor
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),   # (mean) (std) for each channel New = (Prev - mean) / stf
    transforms.RandomRotation((-90, 90)),
    transforms.ColorJitter(10, 2)
    ])


def practice(model, trainloader, validloader, criterion, optimizer, scheduler,
             ES, n_epochs, fold, writer, path_checkpoints, RM_train, RM_valid,
             device, path_resume=None):
  ''' Iterate through epochs for model training and validation
  '''
  better_state, min_loss_valid = {}, np.inf
  ES.reset()
  if path_resume: # Loading checkpoint before resuming training
    model, optimizer, start_epoch, loss_cp = load_checkpoint(model, optimizer, path_resume)
  else:
    start_epoch = 0

  for epoch in range(start_epoch, n_epochs):
    print(f"     Epoch {epoch + 1}")
    loss_train, score_train = train.train(model, trainloader, criterion, optimizer, epoch, RM_train, device)
    loss_valid, score_valid = validate.validate(model, validloader, criterion, RM_valid, device, epoch)

    writer.add_scalar("Loss/train", loss_train, epoch + 1)
    writer.add_scalar("Loss/valid", loss_valid, epoch + 1)
    writer.add_scalar("Acc/train", score_train, epoch + 1)
    writer.add_scalar("Acc/valid", score_valid, epoch + 1)
    print(f"\nTrain Loss: {loss_train:4f}\tValid Loss: {loss_valid:4f}")
    scheduler.step()

    if ES.early_stop(loss_valid):
      print(f"\n* Stopped early *\n")
      break
    if loss_valid <= min_loss_valid:
      print(f"\n* New best model (valid loss): {min_loss_valid:.4f} --> {loss_valid:.4f} *\n")
      min_loss_valid = loss_valid
      better_state = model.state_dict()
    if ((epoch + 1) % 5) == 0:  # Save checkpoint every 5 epochs
      print(f"\n* Checkpoint reached ({epoch + 1}/{n_epochs}) *\n")
      path_checkpoint = os.path.join(path_checkpoints, f"checkpoint_fold{fold + 1}_epoch{epoch + 1}.pth")
      save_checkpoint(model, optimizer, epoch, loss_valid, path_checkpoint)

  return better_state


def choose_in_path(path):
  searching = True
  to = path
  
  while searching:
    
    ldir = os.listdir(to)
    print('\n', ldir)
    choice = int(input("\nWhere to go? "))
    # print(ldir[choice])
    if choice == -1:
      to = os.path.dirname(to)
      continue
    else:
      try:
        to = os.path.join(to, ldir[choice])
        if os.path.isfile(to):
          return to
      except:
        stop = input("Bad path! - do you wanna stop? (y/n) ")
        if stop == 'y':
          return path
        continue

def setup_logs(path_logs):
  path_events = os.path.join(path_logs, "events")
  path_checkpoints = os.path.join(path_logs, "checkpoints")
  path_config = os.path.join(path_logs, "hyper_config.json")
  path_phasegraphs = os.path.join(path_logs, "phasegraphs")
  os.makedirs(path_logs, exist_ok=True);
  os.makedirs(path_events, exist_ok=True)
  os.makedirs(path_checkpoints, exist_ok=True)
  os.makedirs(path_phasegraphs, exist_ok=True)
  with open(path_config, "w") as config:
    config.write("learning_rate: \nbatch_size: \nepochs: \n")
  return path_events, path_checkpoints, path_config

def score_log(BM, writer, path_logs, fold):
  ''' Log metrics to TensorBoard
  '''
  writer.add_scalar(f"accuracy/test", BM.get_accuracy(), fold)
  writer.add_scalar(f"precision/test", BM.get_precision(), fold)
  writer.add_scalar(f"recall/test", BM.get_recall(), fold)
  writer.add_scalar(f"f1score/test", BM.get_f1score(), fold)
  image_cf = plt.imread(BM.get_confusionMatrix(path_logs))
  tensor_cf = torch.tensor(image_cf).permute(2, 0, 1)
  writer.add_image(f"confusion_matrix/test", tensor_cf, fold)
  writer.close()

def model_save(path_models, model_state, dataset_name, date, score_test, path_model=None, title='', fold=''):
  if path_model:
    torch.save(model_state, path_model)
  else:
    # print(path_models, dataset_name, date, score_test)
    model_name = f"{dataset_name}-{date}-{title}{fold}-{score_test:4f}.pth"
    path_model = os.path.join(path_models, model_name)
    torch.save(model_state, path_model)

def save_checkpoint(model, optimizer, epoch, loss, path_checkpoint):
  checkpoint = {
    'epoch': epoch, # Current epoch
    'model_state_dict': model.state_dict(), # Model weights
    'optimizer_state_dict': optimizer.state_dict(), # Optimizer state
    'loss_valid': loss  # Current loss value
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