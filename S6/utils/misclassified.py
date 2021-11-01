import numpy as np
import matplotlib.pyplot as plt 
import torch 
import random

def get_incorrect_preds(model, test_dataloader):
  incorrect_examples = []
  pred_wrong = []
  true_wrong = []

  model.eval()
  for data,target in test_dataloader:
    data , target = data.cuda(), target.cuda()
    output = model(data)
    _, preds = torch.max(output,1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    preds = np.reshape(preds,(len(preds),1))
    target = np.reshape(target,(len(preds),1))
    data = data.cpu().numpy()
    for i in range(len(preds)):
        if(preds[i]!=target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            incorrect_examples.append(data[i])

  return true_wrong, incorrect_examples, pred_wrong

def plot_incorrect_preds(true,ima,pred,n_figures = 10):
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(14, 6), nrows = n_row, ncols=5)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        wrong = int(wrong)
        image = image.squeeze().numpy()
        im = ax.imshow(image, cmap='gray_r')
        ax.set_title(f'A: {correct} , P: {wrong}')
        ax.axis('off')
    plt.show()

