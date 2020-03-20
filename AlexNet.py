#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.transforms import ToPILImage
show=ToPILImage()
import numpy as np
import matplotlib.pyplot as plt
from DataLoader_ILSVRC import ILSVRC2012

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def imshow(img):
    cv2.imshow("",img)
    cv2.waitKey(20)
    # cv2.destroyAllWindows()

class LocalResponseNorm(nn.Module):
    __constants__ = ['size', 'alpha', 'beta', 'k']
    
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2.):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(input, self.size, self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)



class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.C1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.N1 = LocalResponseNorm()
        self.P1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.C2_1 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.C2_2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.N2 = LocalResponseNorm()
        self.P2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.C3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.C4_1 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.C4_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.C5_1 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.C5_2 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.P5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.F6 = nn.Linear(9216, 4096)
        self.D6 = nn.Dropout(0.5)
        self.F7 = nn.Linear(4096, 4096)
        self.D7 = nn.Dropout(0.5)
        self.F8 = nn.Linear(4096, num_classes)
        
    
    def forward(self, x):
        x = F.relu(self.C1(x))
        x = self.N1(x)
        x = self.P1(x)
        x1 = F.relu(self.C2_1(x[:, :48]))
        x2 = F.relu(self.C2_2(x[:, 48:]))
        x = self.N2(torch.cat([x1, x2], 1))
        x = self.P2(x)
        x = F.relu(self.C3(x))
        x1 = F.relu(self.C4_1(x[:, :192]))
        x2 = F.relu(self.C4_2(x[:, 192:]))
        x1 = F.relu(self.C5_1(x1))
        x2 = F.relu(self.C5_2(x2))
        x = self.P5(torch.cat([x1, x2], 1))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.F6(x))
        x = self.D6(x)
        x = F.relu(self.F7(x))
        x = self.D7(x)
        x = self.F8(x)
        
        return x

#
pretrained_weights = None
    
batch_size_train = 128
weight_decay = 0.0005
momentum = 0.9
learning_rate = np.array([0.01] * 30 + [0.001] * 25 + [0.0001] * 20 + [0.00001] * 15) * 0.01
        

trainset = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train', 'dirname_to_classname')
testset = ILSVRC2012('/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val', 'dirname_to_classname')
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)
valid_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_train, shuffle=False, num_workers=8)

num_classes = trainset.num_classes
##load data

net = AlexNet(num_classes).cuda()
if pretrained_weights != None:
    net.load_state_dict(torch.load(pretrained_weights))
    
loss_func = nn.CrossEntropyLoss()

train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()

for epoch in range(len(learning_rate)):
    time_s = time.time()
    print('Epoch : ', epoch + 1)
#    optimizer= optim.SGD(net.parameters(), lr=learning_rate[epoch], momentum=momentum, weight_decay=weight_decay)
    optimizer= optim.Adam(net.parameters(), lr=learning_rate[epoch])
    
    net.train()
    
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
         
        y_PD = net(img.cuda())
        loss = loss_func(y_PD, y_GT.long().cuda())
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print("Epoch {}, Training Data Num {}, Loss {}, Batch Accuracy {}%".format(epoch+1, (batch_idx + 1) * batch_size_train, loss.item(), np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))/len(y_GT)*100))
            print("labels(GT) = ", y_GT[:10].numpy())
            print("labels(PD) = ", np.argmax(y_PD.cpu().data.numpy()[:10], axis=1))
  
    net.eval()
    
    acc_train = 0
    loss_train = 0 
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
         
        y_PD = net(img.cuda())
        loss = loss_func(y_PD, y_GT.long().cuda())
        acc_train += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_train += loss.item()
    acc_train /= len(trainset)
    loss_train /= len(trainset) / batch_size_train
    train_loss_list.append(loss_train)
    train_accuracy_list.append(acc_train)
    print("Train Loss : ", loss_train, "Accuracy : %.2f%%" %(acc_train * 100))
        
    acc_test = 0
    loss_test = 0   
    for batch_idx, (img, y_GT) in enumerate(test_dataloader):
        img = img.permute(0, 3, 1, 2).float()
         
        y_PD = net(img.cuda())
        loss = loss_func(y_PD, y_GT.long().cuda())
        acc_test += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_test += loss.item()
    acc_test /= len(testset)
    loss_test /= len(testset) / batch_size_train
    test_loss_list.append(loss_test)
    test_accuracy_list.append(acc_test)
    print("Test Loss : ", loss_test, "Accuracy : %.2f%%" %(acc_test * 100))

    torch.save(net.state_dict(), 'weights/alexnet_pretrained_weights_{}.pth'.format(epoch+1))
    
    print("Time Elapsed : ", time.time() - time_s)
 
x = np.arange(len(learning_rate)) + 1
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.plot(x, train_accuracy_list)
plt.plot(x, test_accuracy_list)
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.savefig('Accuracy0.01.png') 
plt.show()   


plt.xlabel('epochs')
plt.ylabel('Loss')
plt.plot(x, train_loss_list)
plt.plot(x, test_loss_list)
plt.legend(['training loss', 'testing loss'], loc='upper right')
plt.savefig('Loss0.01.png')
plt.show()    
