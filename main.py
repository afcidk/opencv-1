import torch.nn as nn
import resource
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.multiprocessing import set_sharing_strategy
from torch import load
from lenet import LeNet
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from PyQt5 import QtWidgets, uic
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import events

class Ui(QtWidgets.QMainWindow):
    buttons = [
       "load_image", "color_conversion", "image_flipping",
       "blending", "global_threshold", "local_threshold",
       "gaussian", "sobel_x", "sobel_y", "magnitude", "rst",
       "show_train_image", "show_hyper", "train_1", "pt",
       "inference", "ok", "show_train_result", "cancel"]
    inputs = ["angle", "scale", "tx", "ty", "test_index"]

    def __init__(self):
        super(Ui, self).__init__()

        uic.loadUi('main_window.ui', self)
        self.get_widgets()
        self.get_input()
        self.bind_event()
        self.param_setup()
        self.torch_setup()
        self.show()

    def get_widgets(self):
        for btn in self.buttons:
            setattr(self, btn, self.findChild(QtWidgets.QPushButton, btn))
    
    def get_input(self):
        for inp in self.inputs:
            setattr(self, inp, self.findChild(QtWidgets.QLineEdit, inp))


    def bind_event(self):
        for btn in self.buttons:
            getattr(self, btn).clicked.connect(partial(
                getattr(events,  btn), 
                self))
    def param_setup(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.opt = "SGD"
        self.loss_list = []
        self.loss_epoch = []
        self.acc_train_epoch = []
        self.acc_test_epoch = []
        self.compose = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])


    def torch_setup(self):
        self.data_train = MNIST('./data/mnist',
                            train=True,
                            download=True,
                            transform=self.compose)
                            
        self.data_test = MNIST('./data/mnist',
                            train=False,
                            download=True,
                            transform=self.compose)
        self.data_train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.data_test_loader = DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)
        self.criterion = nn.CrossEntropyLoss()
        self.net = LeNet()
        self.optimizer = getattr(optim, self.opt)(self.net.parameters(), lr=self.learning_rate)

        try:
            self.net.load_state_dict(load('model_params.pkl'))
            self.loaded = True
            print("Loaded")
        except Exception as e:
            print(e)
            self.loaded = False
            print("Not loaded")

    def train(self, epoch):
        self.net.train()
        self.loss_list = []
        correct, total = 0, 0
        for i, (images, labels) in enumerate(self.data_train_loader):
            self.optimizer.zero_grad()
            output = self.net(images)
            loss = self.criterion(output, labels)
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
            total += images.size(0)
            self.loss_list.append(loss.detach().cpu().item())

            if i % 100 == 0:
                print(f'Train - Epoch {epoch}, Batch: {i}, Loss: {loss.detach().cpu().item()}')

            loss.backward()
            self.optimizer.step()
        self.acc_train_epoch.append(correct/total)
        self.loss_epoch.append(sum(self.loss_list)/len(self.loss_list))

    def test(self):
        self.net.eval()
        total_correct, avg_loss = 0, 0.0
        for i, (images, labels) in enumerate(self.data_test_loader):
            output = self.net(images)
            avg_loss += self.criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        avg_loss /= len(self.data_test)
        acc = float(total_correct)/len(self.data_test)
        self.acc_test_epoch.append(acc)

    def test_and_train(self, epoch):
        self.train(epoch)
        self.test()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
