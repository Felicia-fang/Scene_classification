import torch
import torch.nn as nn
import os
import cv2
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# START: OWN CODE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU id


class NN(nn.Module):  # Build the neural network
    def __init__(self, in_features, out_features):
        super(NN, self).__init__()

        self.fc = nn.Linear(in_features, out_features, True)
        self.dropout = nn.Dropout(p=0.7)  # dropout
        self.relu = nn.ReLU()

    def forward(self, ipt):
        opt = self.fc(ipt)
        opt = self.dropout(opt)
        opt = self.relu(opt)
        return opt


class BpNet(nn.Module):
    def __init__(self, num_classes=6):
        super(BpNet, self).__init__()

        self.nn1 = NN(in_features=65536, out_features=1024)
        self.nn2 = NN(in_features=1024, out_features=1024)
        self.nn3 = NN(in_features=1024, out_features=2048)
        self.net = nn.Sequential(self.nn1, self.nn2, self.nn3)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        # self.softmax == nn.Softmax(dim=1)

    def forward(self, ipt):
        opt = self.net(ipt)
        opt = self.fc(opt)
        return opt


# END: OWN CODE

def adjust_lr(optimizer, epoch):  # Learning rate adjustment function
    lr = 0.001
    # START: OWN CODE
    if epoch > 450:
        lr = lr / 1000000000
    elif epoch > 400:
        lr = lr / 100000000
    elif epoch > 350:
        lr = lr / 10000000
    elif epoch > 300:
        lr = lr / 1000000
    elif epoch > 250:
        lr = lr / 100000
    elif epoch > 200:
        lr = lr / 10000
    elif epoch > 150:
        lr = lr / 1000
    elif epoch > 100:
        lr = lr / 100
    elif epoch > 50:
        lr = lr / 10
    # END: OWN CODE
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(model, train_accuracy, train_loss, train_loader, optimizer, cuda_available=True):  # Training process
    model.train()
    # START: OWN CODE
    for i, (images, labels) in enumerate(train_loader):
        if cuda_available:
            images = Variable(images.cuda())
            images = images.reshape(-1, 65536)
            labels = Variable(labels.cuda())
        optimizer.zero_grad()
        result = model(images)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(result, labels)
        loss.backward()
        optimizer.step()
        # END: OWN CODE
        train_loss += loss.cpu().data.numpy() * images.size(0)
        _, prediction = torch.max(result.data, 1)
        train_accuracy += torch.sum(prediction == labels.data)

    train_accuracy = train_accuracy / len_train_set  # average train accuracy and loss
    train_loss = train_loss / len_train_set
    return train_accuracy, train_loss


def test(model, test_accuracy, test_loss, test_loader, cuda_available=True):  # Testing process
    model.eval()
    # START: OWN CODE
    for i, (images, labels) in enumerate(test_loader):
        if cuda_available:
            images = Variable(images.cuda())
            images = images.reshape(-1, 65536)
            labels = Variable(labels.cuda())
        predict = model(images)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(predict, labels)
        # END: OWN CODE
        test_loss += loss.cpu().data.numpy() * images.size(0)
        _, prediction = torch.max(predict.data, 1)
        test_accuracy += torch.sum(prediction == labels.data)

    test_accuracy = test_accuracy / len_test_set  # average test accuracy and loss
    test_loss = test_loss / len_train_set
    return test_accuracy, test_loss


# START: OWN CODE

class dataset(Dataset):

    def __init__(self, images, labels, transform=None):  # data_features,data_target->numpy array
        self.length = len(images)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            self.images[index] = self.transform(self.images[index])
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


# END: OWN CODE

def main():
    # START: OWN CODE
    log = "Scene_classification_log"  # log file
    isExists = os.path.exists(log)
    if not isExists:
        os.makedirs(log)
    log_file = open(log + '/' + log + '_log.txt', 'w')

    xx = []
    yy = []
    # END: OWN CODE
    for i in range(1, 7):
        for f in os.listdir(r"/home/Liruochen/Q1Q2Q3/archive/image/%s" % i):
            img = cv2.imread(r"/home/Liruochen/Q1Q2Q3/archive/image/%s/%s" % (i, f))
            image = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
            xx.append(((hist / 255).flatten()))
            yy.append((i - 1))
    # START: OWN CODE
    np.random.seed(116)  # shuffle
    np.random.shuffle(xx)
    np.random.seed(116)
    np.random.shuffle(yy)
    xx = np.array(xx)
    yy = np.array(yy)

    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.3,
                                                        random_state=1)  # 30%  split into trainset and testset

    train_set = dataset(images=X_train, labels=y_train)
    test_set = dataset(images=X_test, labels=y_test)
    global len_train_set
    global len_test_set
    len_train_set = len(train_set)
    len_test_set = len(test_set)
    print("len(train_set) : %d" % len_train_set)
    print("len(test_set) : %d" % len_test_set)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = BpNet(num_classes=6)  # initialize model
    # END: OWN CODE
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model.cuda()
    # START: OWN CODE
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  # optimizer

    num_epochs = 500
    best = 0.0
    for epoch in range(num_epochs):  # Training
        train_accuracy = 0.0
        train_loss = 0.0
        test_accuracy = 0.0
        test_loss = 0.0
        train_accuracy, train_loss = train(model, train_accuracy, train_loss, train_loader, optimizer, cuda_available)
        adjust_lr(optimizer, epoch)  # Ajust learning rate
        test_accuracy, test_loss = test(model, test_accuracy, test_loss, test_loader, cuda_available)

        if test_accuracy > best:
            torch.save(model.state_dict(), "scene_classification_{}.model".format(epoch))  # Save the weights
            print("Chekcpoint saved")
            best = test_accuracy

        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {} , TestLoss: {}".format(epoch,
                                                                                                       train_accuracy,
                                                                                                       train_loss,
                                                                                                       test_accuracy,
                                                                                                       test_loss))
        log_file.write("\n")
        log_file.write("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {} , TestLoss: {}".format(epoch,
                                                                                                                train_accuracy,
                                                                                                                train_loss,
                                                                                                                test_accuracy,
                                                                                                                test_loss))
        log_file.write("\n")
        log_file.write("\n")
        log_file.flush()


# END: OWN CODE

if __name__ == "__main__":
    main()