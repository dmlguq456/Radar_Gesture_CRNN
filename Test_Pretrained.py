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

    # data loading
    test_dir = './data_5_5/test' 
    test_dataset = Dataset.RadarGesture(test_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
  
    # model loading
    model = GestureNet().to(device)
    print(model)
    model.load_state_dict(torch.load('model.pth'))

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
            print('Predicted:', predicted, 'Real:', labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader)*batch_size, 100 * correct / total))

if __name__ == '__main__':
    main()
