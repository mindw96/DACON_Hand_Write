import copy
import time

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, file_path_list, transform, labels=None):
        self.file_path_list = file_path_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.file_path_list.iloc[idx], cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)

        image = image.clone().detach()

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            return image, label.clone().detach()

        return image

    def __len__(self):
        return len(self.file_path_list)


def ACCURACY(true, pred):
    score = np.mean(true == pred)
    return score


class DataLoad():
    def __init__(self, file_dir='', batch_size=0):
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.file_dir = file_dir
        self.batch_size = batch_size
        self.dataset = pd.read_csv(self.file_dir + 'train/train_data.csv')
        self.trainset = self.dataset.sample(frac=0.8)
        self.validset = self.dataset.sample(frac=0.2)
        self.testset = pd.read_csv(self.file_dir + 'test/test/test_data.csv')

    def train_data_load(self):
        train_list = self.file_dir + 'train/' + self.trainset['filen_name']
        train_labels = self.trainset['label']
        train_dataset = CustomDataset(train_list, self.train_transform, labels=train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader

    def valid_data_load(self):
        valid_list = self.file_dir + 'train/' + self.validset['filen_name']
        valid_labels = self.validset['label']
        valid_dataset = CustomDataset(valid_list, self.test_transform, labels=valid_labels)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return valid_loader

    def test_data_load(self):
        test_list = self.file_dir + 'test/test/' + self.testset['file_name']
        test_dataset = CustomDataset(test_list, self.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return test_loader


def train(optimizer=None, epochs=0, model=None, train_loader=None, valid_loader=None, criterion=None):
    best_loss = 100.0
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []

    device = torch.device("cuda:0")
    model = model.to(device)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    since = time.time()

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Train
        train_batch_loss, valid_batch_loss, train_batch_acc, valid_batch_acc = 0.0, 0.0, 0.0, 0.0
        model.train()
        batch_index = 0

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            train_batch_loss += loss.item()
            train_batch_acc += acc

            batch_index += 1

        train_loss.append(train_batch_loss / batch_index)
        train_acc.append(train_batch_acc / batch_index)

        lr_scheduler.step()

        # Validation
        model.eval()

        batch_index = 0

        for image, label in valid_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                output = model(image)

                loss = criterion(output, label)

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            valid_batch_loss += loss.item()
            valid_batch_acc += acc

            batch_index += 1

        valid_loss.append(valid_batch_loss / batch_index)
        valid_acc.append(valid_batch_acc / batch_index)

        # 1 Epoch Result
        print('Train Acc: {:.2f} Valid Acc: {:.2f}'.format(train_acc[epoch] * 100, valid_acc[epoch] * 100))
        print('Train Loss: {:.4f} Valid Loss: {:.4f}'.format(train_loss[epoch], valid_loss[epoch]))

        # deep copy the model
        if valid_loss[epoch] < best_loss:
            best_idx = epoch
            best_loss = valid_loss[epoch]
            torch.save(model.state_dict(), 'SimpleCNN_check.pt')
            base_cnn_best_model_wts = copy.deepcopy(model.state_dict())
            print('==> best model saved - {} / {:.4f}'.format(best_idx + 1, best_loss))

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Loss: %d - %.4f' % (best_idx + 1, best_loss))

    # load best model weights
    model.load_state_dict(base_cnn_best_model_wts)
    torch.save(model.state_dict(), 'SimpleCNN_Final.pt')
    print('final model saved')

    return best_idx + 1, train_acc, train_loss, valid_acc, valid_loss


def final_train(optimizer=None, epochs=0, model=None, train_loader=None, valid_loader=None, criterion=None):
    train_loss, train_acc = [], []

    device = torch.device("cuda:0")
    model = model.to(device)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    since = time.time()

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Train
        train_batch_loss, train_batch_acc = 0.0, 0.0
        model.train()
        batch_index = 0

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            train_batch_loss += loss.item()
            train_batch_acc += acc

            batch_index += 1

        for image, label in valid_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                output = model(image)

                loss = criterion(output, label)

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            train_batch_loss += loss.item()
            train_batch_acc += acc

            batch_index += 1

        lr_scheduler.step()
        train_loss.append(train_batch_loss / batch_index)
        train_acc.append(train_batch_acc / batch_index)

        # 1 Epoch Result
        print('Train Acc: {:.2f}'.format(train_acc[epoch] * 100))
        print('Train Loss: {:.4f}'.format(train_loss[epoch]))

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    torch.save(model.state_dict(), 'Final_SimpleCNN.pt')
    print('final model saved')

    return model


def test(model=None, test_loader=None):
    device = torch.device("cuda:0")
    model = model.to(device)

    predict = []

    with torch.no_grad():
        for image in test_loader:
            image = image.to(device)
            output = model(image)

            pred = output.detach().cpu().numpy().argmax(-1)
            predict.append(pred.tolist())

    labels = np.array(predict).flatten().tolist()
    labels = sum(labels, [])

    submission = pd.read_csv('sample_submission.csv')  # sample submission 불러오기
    labels = pd.Series(labels)
    submission['label'] = labels

    submission.to_csv('submission.csv', index=False)


def draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss):
    print('best model : %d - %.2f / %.4f' % (best_idx, valid_acc[best_idx] * 100, valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()
