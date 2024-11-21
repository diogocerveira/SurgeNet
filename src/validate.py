import torch

def validate(model, validloader, criterion, RM, device, epoch):
  ''' Validate the model - return running loss and metric
  '''
  RM.metric.reset() # Running metric, e.g. accuracy
  loss_running = 0.0
  model.eval(), print("\n\tValidating...")
  with torch.no_grad(): # don't calculate gradients (for learning only)
    for batch, data in enumerate(validloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      outputs = model(inputs).squeeze(0).permute(1, 0)
      # print(labels[:5], outputs[:5])
      loss = criterion(outputs, labels)
      loss_running += loss.item()
      RM.update(outputs, labels)  # updates with data from new batch
      if batch % RM.period_compute == 0:
        print(f"\t  V [E{epoch + 1} B{batch + 1}]   scr: {RM.score():.4f}")
  return loss_running / len(validloader), RM.score()