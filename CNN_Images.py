import torch
import torch.nn as nn
import numpy as np
import joblib
import torch.utils.data as Data
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random

def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image

def random_rotate(image):
    angle = np.random.randint(-30, 30)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image

def RandomGenerator(image):
    if random.random() > 0.5:
        image = random_rot_flip(image)
    elif random.random() <= 0.5:
        image = random_rotate(image)
    return image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#灰度图，channel为一
                out_channels=16,#输出channels自己设定
                kernel_size=3,#卷积核大小
                stride=1,
                padding=1
            ),
            nn.ReLU(),#激活函数，线性转意识到非线性空间
            nn.MaxPool2d(kernel_size=2)#池化操作，降维，取其2x2窗口最大值代表此窗口，因此宽、高减半，channel不变
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


        self.prediction = nn.Sequential(
            nn.Linear(128*5*5, 128),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )
        #前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output
class myDataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Output:
        img = images
        label = labels"""

    def __init__(self, X, Labels):
        self.X_train = X
        self.Labels_train = Labels
        # self.X_train = myData['Images_train']
        # self.Labels_train = myData['Labels_train']
        # self.X_val = myData['Images_val']
        # self.Labels_val = myData['Labels_val']
        # self.X_test = myData['Images_test']
        # self.Labels_test = myData['Labels_test']

    def __len__(self):
        return len(self.Labels_train)

    def __getitem__(self, i):
        img=self.X_train[i,:].reshape([80,80])
        label = self.Labels_train[i].astype(np.double)
        if label==0:
            label=[1,0]
        elif label==1:
            label=[0,1]
        # label = label.reshape(1,2)
        img=RandomGenerator(img)
        # img = (img - img.min()) / (img.max() - img.min())
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

def myCNN_Images(myData):
    dataset_train=myDataset(myData['Images_train'],myData['Labels_train'])
    dataset_val = myDataset(myData['Images_val'], myData['Labels_val'])
    dataset_test = myDataset(myData['Images_test'], myData['Labels_test'])
    batch_size=64
    lr=0.005
    epoch=20
    train_loader = Data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0 )
    val_loader = Data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = Data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    pin_memory = False
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    if train_on_gpu:
        pin_memory = True

    cnn = CNN()
    cnn.to(device)
    cnn.train()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()

    for epoch in range(epoch):
        for iter, (batch_img, batch_label) in enumerate(train_loader):
            batch_img=batch_img.to(device)
            batch_label=batch_label.to(device)
            output = cnn(batch_img)
            loss = loss_func(output,batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #打印操作，用测试集检验是否预测准确
            if iter % 50 == 0:
                cnn.eval()
                accuracy=0
                with torch.no_grad():
                    Val_Output=torch.ones(len(val_loader),2)
                    Pred_y=torch.ones(len(val_loader))
                    for id_val, (val_img, val_label) in enumerate(val_loader):
                        val_output = cnn(val_img.to(device))
                        Val_Output[id_val,:]=val_output
                        pred_y = torch.max(val_output, 1)[1].data.squeeze()
                        real_y = torch.max(val_label, 1)[1].data.squeeze()
                        accuracy = accuracy+float((pred_y == real_y).sum())
                        Pred_y[id_val]=pred_y.cpu().detach()
                    accuracy=accuracy/len(val_loader)
                    print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|val accuracy：%.4f" %accuracy)
                    lr_precision, lr_recall, _ = precision_recall_curve(myData['Labels_val'].tolist(),Val_Output[:, 1].cpu().detach().numpy().tolist())
                    lr_f1, lr_auc = f1_score(myData['Labels_val'].tolist(), Pred_y.numpy().tolist()), auc(lr_recall,lr_precision)
                    print("epoch:", epoch, "| Val F1 score :%.4f" % lr_f1, "|Val pr_auc：%.4f" %lr_auc)
                    cnn.train()
    # torch.save(cnn.state_dict(), './Images/CNN_epoch_' + str(epoch)
    #                + '_batchsize_' + str(batch_size) + '.pth')
    return cnn.cpu()

def test_MyCNN(model,myData):
    dataset_train = myDataset(myData['Images_train'], myData['Labels_train'])
    dataset_val = myDataset(myData['Images_val'], myData['Labels_val'])
    dataset_test = myDataset(myData['Images_test'], myData['Labels_test'])
    test_loader = Data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=0)
    cnn = model
    cnn.eval()
    accuracy = 0
    with torch.no_grad():
        Test_Output = torch.ones(len(test_loader), 2)
        Pred_y = torch.ones(len(test_loader))
        for id_test, (test_img, test_label) in enumerate(test_loader):
            test_output = cnn(test_img)
            Test_Output[id_test, :] = test_output
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            real_y = torch.max(test_label, 1)[1].data.squeeze()
            accuracy = accuracy + float((pred_y == real_y).sum())
            Pred_y[id_test] = pred_y.detach()
        accuracy = accuracy / len(test_loader)
        print("Test accuracy：%.4f" % accuracy)
        lr_precision, lr_recall, _ = precision_recall_curve(myData['Labels_test'].tolist(),Test_Output[:, 1].detach().numpy().tolist())
        lr_f1, lr_auc = f1_score(myData['Labels_test'].tolist(), Pred_y.numpy().tolist()), auc(lr_recall, lr_precision)
        print("Test F1 score :%.4f" % lr_f1, "|Val pr_auc：%.4f" % lr_auc)
        return Test_Output.detach().numpy(), Pred_y.numpy()

def CNNs_bagging(myData):
    X_train = myData['Images_train']
    Labels_train = myData['Labels_train']
    X_val = myData['Images_val']
    Labels_val = myData['Labels_val']
    X_test = myData['Images_test']
    Labels_test = myData['Labels_test']

    X_train_pos=X_train[Labels_train==1,:]
    X_train_neg = X_train[Labels_train == 0, :]
    Labels_train_pos = Labels_train[Labels_train==1]
    Labels_train_neg = Labels_train[Labels_train == 0]

    numpos=sum(Labels_train==1)
    numneg=sum(Labels_train==0)

    numclassifer=int(numneg/numpos)
    CNNs_bagging=[]
    for id_calssifer in range(numclassifer):
        torch.cuda.empty_cache()
        x_train=np.concatenate((X_train_pos,X_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos,:]),0)
        y_train=np.concatenate((Labels_train_pos,Labels_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos]),0)
        Data_mini={
            'Images_train':x_train,
            'Labels_train':y_train,
            'Images_val':X_val,
            'Labels_val':Labels_val,
            'Images_test':X_test,
            'Labels_test':Labels_test
        }
        cnn = myCNN_Images(Data_mini)
        CNNs_bagging.append(cnn)
    joblib.dump(CNNs_bagging, './Images/CNNs_bagging.pkl')

    return CNNs_bagging
