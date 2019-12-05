import os
import cv2
from pandas import Series,DataFrame 
import insightface
import cv2
import numpy as np 

valid_suffix = ['.jpg', '.jpeg', '.png']
invalid_suffix = ['.gif']
def rename(path):
    i = 0
    # '该文件夹下所有的文件（包括文件夹）'
    FileList = os.listdir(path)
    # '遍历所有文件'
    for files in FileList:
        # '原来的文件路径'
        oldDirPath = os.path.join(path, files)
        # '如果是文件夹则递归调用'
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
        # '文件名'
        fileName = os.path.splitext(files)[0]
        # '文件扩展名'
        fileType = os.path.splitext(files)[1]
        print(fileName, ' ', fileType)
        # '新的文件路径'
        # newDirPath = os.path.join(path, str(i) + fileType)
        # '重命名'
        # os.rename(oldDirPath, newDirPath)
        i += 1


def countRaceAndClean(base_path):
    i = 0
    count_dict = {}
    count = 0
    for i in [1, 2 , 3]:
        path = base_path + str(i)
        FileList = os.listdir(path)
        for files in FileList:
            file_name = os.path.join(path, files)
            print(file_name)
            # if os.path.isdir(oldDirPath):
            #     rename(oldDirPath)
            labels = os.path.splitext(files)[0].split('_')
            if len(labels) > 2:
                race = int(labels[2])
                if race not in [0, 1, 2, 3, 4]:
                    os.remove(file_name)
                    print(os.path.splitext(files)[0])
                else:
                    if race not in count_dict.keys():
                        count_dict[race] = 0
                    count_dict[race] += 1 
                    if race == 4:
                        os.remove(file_name)
            else:
                os.remove(file_name)
                count += 1
    print(count_dict)
    print('count = ', count)



def generateDateset(base_path):#2test 8 train
    train_count = 0
    test_count = 0
    count = 0
    count_dict = {}
    for i in [1, 2 , 3]:
        path = base_path + str(i)
        FileList = os.listdir(path)
        for files in FileList:
            file_name = os.path.join(path, files)
            labels = os.path.splitext(files)[0].split('_')
            race = int(labels[2])
            if race not in count_dict.keys():
                count_dict[race] = 0
            count_dict[race] += 1 
            count_dict[race] %= 10
            if count_dict[race] % 10 < 8:
                new_file_name = str(train_count) + '_' + str(race) + '.jpg'
                command = 'cp ' + str(file_name) +  '   ./train/' + new_file_name
                os.system(command)
                train_count += 1
            elif count_dict[race] % 10 <= 9:
                new_file_name = str(test_count) + '_' + str(race) + '.jpg'
                command = 'cp ' + str(file_name) +  '   ./test/' + new_file_name
                os.system(command)
                test_count += 1


def generateVectorDataset(path):
    FileList = os.listdir(path)
    np_x = []
    np_y = []
    model = insightface.app.FaceAnalysis()
    ctx_id = -1
    model.prepare(ctx_id = ctx_id, nms=0.4)
    for files in FileList:
        file_name = os.path.join(path, files)
        race = os.path.splitext(files)[0].split('_')[1]
        img = cv2.imread(file_name)
        faces = model.get(img)
        for idx, face in enumerate(faces):
                np_x.append(face.embedding)
                np_y.append(int(race))
                # box = face.bbox.astype(np.int).flatten()
                # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # cv2.imshow("fff", img)
                # cv2.waitKey(0)
            # c = np.concatenate((np_x,face.embedding),axis=0)
            

    np_y = np.array(np_y)
    np.save('./data/train/x.npy', np_x)
    np.save('./data/train/y.npy', np_y)



def generateFileList():
    train_path = './data/train'
    test_path = './data/test'
    train_list = []
    for path in (train_path, test_path):
        FileList = os.listdir(path)
        for files in FileList:
            file_name = os.path.join(path, files)
            train_list.append(file_name)
        if path is train_path:
            np.save( "./data/files_train.npy" ,train_list)
        else:
            np.save( "./data/files_test.npy" ,train_list )
        train_list = []
    

def generateTargetList():
    train_path = './data/train'
    test_path = './data/test'
    train_list = []
    for path in (train_path, test_path):
        FileList = os.listdir(path)
        
        for files in FileList:
            race = os.path.splitext(files)[0].split('_')[1]
            file_name = os.path.join(path, files)
            train_list.append(int(race))
        if path is train_path:
            np.save( "./data/target_train.npy" ,train_list)
        else:
            np.save( "./data/target_test.npy" ,train_list )
        train_list = []

# generateVectorDataset('./data/test')

# x = np.load('./data/target_train.npy')
# print(x)
# print(len(x))
# print(type(x))
# y = np.load('./data/train/y.npy')
# print(y)
# print(len(y))
# print(type(y))
generateFileList()
generateTargetList()