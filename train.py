import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  model import Activation_Net,simpleNet,Net
from dataloader import trainset,testset
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

import insightface
import cv2
import numpy as np 




# model_ins = insightface.app.FaceAnalysis()
# ctx_id = -1
# model_ins.prepare(ctx_id = ctx_id, nms=0.4)

train_data  = trainset()
test_data = testset()
trainloader = DataLoader(train_data, batch_size=1,shuffle=True)
testloader = DataLoader(test_data, batch_size=1,shuffle=True)



# model = model.simpleNet(512, 300, 100, 10)
model = Activation_Net(512, 256, 128, 4)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
# model = Net()
if torch.cuda.is_available():
    model = model.cuda()
 
# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# classes = ['White', 'Black', 'Asian', 'Indian']
# model.train()
modelface = insightface.app.FaceAnalysis()
ctx_id = -1
modelface.prepare(ctx_id = ctx_id, nms=0.4)

for epoch in range(10):  # loop over the dataset multiple times

      running_loss = 0.0
      count = 0
      for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            
            img = torch.squeeze(inputs).numpy()
            faces = modelface.get(img)
            if len(faces) == 0:
                  continue
            
            inputs = torch.from_numpy((faces[0].embedding)).unsqueeze(0)
            print(inputs.shape)
            if torch.cuda.is_available():
                  inputs = inputs.cuda()
                  labels = labels.cuda()
      # forward + backward + optimize
            if inputs.shape != (1, 512):
                  print(inputs.size())
                  continue
            # print(inputs.size())
            outputs = model(inputs)
            # outputs = outputs.unsqueeze(0)
            # labels = labels.squeeze(0)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            count += 1
                  # print statistics
            running_loss += loss.item()
            if count % 1999 == 0:    # print every 2000 mini-batches
                  print('[%d, %5d] loss: %.9f' %(epoch + 1, count + 1, running_loss / 2000))
                  running_loss = 0.0

print('Finished Training')

# import matplotlib.pyplot as plt
 
# x = range(0, 60)
# plt.figure()
# plt.plot(x, average_loss_series)

torch.save(model, './model.pkl')

model.eval()
eval_loss = 0
eval_acc = 0
for data in testloader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_data)),
    eval_acc / (len(test_data))
))

