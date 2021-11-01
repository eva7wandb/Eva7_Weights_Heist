from tqdm import tqdm
import torch.nn.functional as F

def train_model(model, device, train_loader, optimizer, epoch):
  train_batch_loss = []
  train_batch_acc = []

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    # Predict
    y_pred = model(data)
    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_batch_loss.append(loss.item())
    # Backpropagation
    loss.backward()
    optimizer.step()
    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_batch_acc.append(100*correct/processed)
  
  return train_batch_loss, train_batch_acc

