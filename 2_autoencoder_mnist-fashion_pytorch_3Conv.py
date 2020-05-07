import torch
import numpy as np
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import nn
import os

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 10
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.FashionMNIST('~/.pytorch/MNIST_Fashion_data/', download=True, transform=img_transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=1)

        # decoder
        self.decoder1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2)
        self.decoder3 = nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1)

    def forward(self, x):

        # add sequence of convolutional and max pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = F.tanh(self.decoder3(x))  
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
model = Net()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')