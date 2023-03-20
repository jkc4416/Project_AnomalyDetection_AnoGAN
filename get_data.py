import torch
from torchvision import datasets, transforms
import torchvision

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.ImageFolder(root='C:/Users/Administrator/PycharmProjects/AnoGAN_pytorch/dataset', transform=trans)
test_data = torchvision.datasets.ImageFolder(root='C:/Users/Administrator/PycharmProjects/AnoGAN_pytorch/dataset', transform=trans)

train_set = torch.utils.data.DataLoader(dataset = train_data, batch_size = 8, shuffle=True, num_workers=2)
test_set = torch.utils.data.DataLoader(dataset = test_data, batch_size = len(test_data))

print(train_set)