import torch
import os

def test(model, testloader, device, BM=None, return_extra={"ids": False, "ts": False, "vids": False}):
  ''' Test the model - updates metrics of BM
  '''
  
  final_image_ids = []
  final_preds = []
  final_targets = []
  model.eval().to(device); print("\n\tTesting...")

  if return_extra["ids"]: # make sure ids were retrieved from the dataset 
    with torch.no_grad():
      for data in testloader:
        images, labels, path_image = data[0].to(device), data[1].to(device), data[2]
        outputs = model(images).squeeze(0).permute(1, 0)
        _, preds = torch.max(outputs, 1)
        final_image_ids.append(path_image)
        final_preds.append(preds)
        final_targets.append(labels)

    final_image_ids = [os.path.splitext(os.path.basename(imgn))[0] for batch in final_image_ids for imgn in batch]
    final_preds = torch.cat(final_preds).cpu().tolist()  # flattens the list of tensor
    final_targets = torch.cat(final_targets).cpu().tolist()

    return final_image_ids, final_preds, final_targets
  
  else:
    print("\nNo return extra!")
    BM.reset()
    with torch.no_grad():
      for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images).squeeze(0).permute(1, 0)
        _, predictions = torch.max(outputs, 1)  # get labels with max prediction values
        final_preds.append(predictions)
        final_targets.append(labels)

    final_preds = torch.cat(final_preds)  # flattens the list of tensor
    final_targets = torch.cat(final_targets)

    BM.update(final_preds, final_targets)