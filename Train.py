from __future__ import print_function
import torch
import torch.optim as optim
import Dataset
import torch.nn as nn
from GestureNet import GestureNet
import os

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  batch_size = 5
  step_size = 0.0001
  num_epochs = 50

  # data loading
  train_dir = './data/train'
  test_dir = './data/test'
  
  train_dataset = Dataset.RadarGesture(train_dir)
  test_dataset = Dataset.RadarGesture(test_dir)

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
  
  # model loading
  model = GestureNet().to(device)
  print(model)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
  total_step = len(train_loader)

  # training
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)

      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1) % 6 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        test_image, test_label = next(iter(test_loader))
        _, test_predicted = torch.max(model(test_image.to(device)).data, 1)
        print('Testing data: [Predicted: {} / Real: {}]'.format(test_predicted, test_label))

  if epoch+1 == num_epochs:
    torch.save(model.state_dict(), 'model.pth')
  else:
    torch.save(model.state_dict(), 'model-{:02d}_epochs.pth'.format(epoch+1))

  # test
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
  print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader)*batch_size, 100 * correct / total))

if __name__ == '__main__':
    main()
