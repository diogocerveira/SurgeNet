def train(model, trainloader, criterion, optimizer, epoch, RM, device):
  ''' Train the model - return running loss and metric
  '''
  RM.metric.reset()  # resets states, typically called between epochs
  loss_running = 0.0
  model.train().to(device); print("\tTraining...")
  for batch, data in enumerate(trainloader, 0):   # start at batch 0
    inputs, labels = data[0].to(device), data[1].to(device) # data is a list of [inputs, labels]
    print("inputs: ", inputs.shape, "\nlabels: ", labels.shape, '\n')
    optimizer.zero_grad() # reset the parameter gradients before backward step
    outputs = model(inputs).squeeze(0).permute(1, 0) # forward pass
    print("\noutputs: ", outputs.shape)
    loss = criterion(outputs, labels) # calculate loss  # -1 for getting last stage guesses
    loss.backward() # backward pass
    optimizer.step()    # a single optimization step
    loss_running += loss.item() # accumulate loss
    RM.update(outputs, labels)
    if batch % RM.period_compute == 0:
        print(f"\t  T [E{epoch + 1} B{batch + 1}]   scr: {RM.score():.4f}")
  return loss_running / len(trainloader), RM.score()