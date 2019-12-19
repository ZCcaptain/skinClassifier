from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # normalize
])

def default_loader(path):
    # img_pil =  Image.open(path)
    # img_pil = img_pil.resize((500,500))
    # img_tensor = preprocess(img_pil)
    img = cv2.imread(path)
    return img

#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        file_train = np.load('./data/files_train.npy')
        target_train = np.load('./data/target_train.npy')
        self.images = file_train
        self.target = target_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        file_train = np.load('./data/files_test.npy')
        target_train = np.load('./data/target_test.npy')
        self.images = file_train
        self.target = target_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


train_data  = trainset()
test_data = testset()
trainloader = DataLoader(train_data, batch_size=1,shuffle=True)
testloader = DataLoader(test_data, batch_size=1,shuffle=True)
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(type(images), type(labels), images.shape)
# count = 0
# print(len(images))
# for npimg, label in dataiter:
#     # print(type(npimg), npimg.shape)
#     print(label)
#     print(npimg.shape)
#     # for img in npimg:
#     #     plt.imshow(np.transpose(img, (1, 2, 0)))
#     # plt.show()
#     # 关闭当前显示的图像
#         # plt.pause(1)
#         # plt.close()
    
#     count += 1
# print(images, labels)
