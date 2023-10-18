import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import LDA_SLIC
import sys,os
sys.path.append(os.pardir)
from global_module import network, train
from global_module.generate_pic import aa_and_each_accuracy, sampling,sampl,load_dataset, generate_png, generate_iter
from prettytable import PrettyTable
from global_module.Utils import fdssc_model, record, extract_samll_cubic
from torchsummary import summary
import seaborn as sns
import matplotlib.pyplot as plt
import tsne
# plt.switch_backend('Agg')
# class ConfusionMatrix(object):
#     """
#     注意,如果显示的图像不全,是matplotlib版本问题
#     本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
#     需要额外安装prettytable库
#     """
#     def __init__(self, num_classes: int, labels: list):
#         self.matrix = np.zeros((num_classes, num_classes))
#         self.num_classes = num_classes
#         self.labels = labels
#
#     def update(self, preds, labels):
#         for p, t in zip(preds, labels):
#             self.matrix[p, t] += 1
#
#     def summary(self):
#         # calculate accuracy
#         sum_TP = 0
#         for i in range(self.num_classes):
#             sum_TP += self.matrix[i, i]
#         acc = sum_TP / np.sum(self.matrix)
#         print("the model accuracy is ", acc)
#
#         # precision, recall, specificity
#         table = PrettyTable()
#         table.field_names = ["", "Precision", "Recall", "Specificity"]
#         for i in range(self.num_classes):
#             TP = self.matrix[i, i]
#             FP = np.sum(self.matrix[i, :]) - TP
#             FN = np.sum(self.matrix[:, i]) - TP
#             TN = np.sum(self.matrix) - TP - FP - FN
#             Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
#             Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
#             Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
#             table.add_row([self.labels[i], Precision, Recall, Specificity])
#         print(table)
#
#     def plot(self):
#         matrix = self.matrix
#         print(matrix)
#         plt.figure(figsize=(8, 8))
#         plt.imshow(matrix, cmap=plt.cm.Blues)
#
#         # 设置x轴坐标label
#         plt.xticks(range(self.num_classes), self.labels, rotation=45)
#         # 设置y轴坐标label
#         plt.yticks(range(self.num_classes), self.labels)
#         # 显示colorbar
#         plt.colorbar()
#         plt.xlabel('True Labels')
#         plt.ylabel('Predicted Labels')
#         plt.title('Confusion matrix')
#
#         # 在图中标注数量/概率信息
#         thresh = matrix.max() / 2
#         for x in range(self.num_classes):
#             for y in range(self.num_classes):
#                 # 注意这里的matrix[y, x]不是matrix[x, y]
#                 info = int(matrix[y, x])
#                 plt.text(x, y, info,
#                          verticalalignment='center',
#                          horizontalalignment='center',
#                          color="white" if info > thresh else "black")
#         plt.tight_layout()
#         sns.set(font_scale=0.1)
#         plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.enabled = False
# for Monte Carlo runs
seeds = [1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331]
#seeds = [1331, 1332, 1333, 1334, 1335]
# seeds=[1331]
ensemble = 1

day = datetime.datetime.now()  # 获取当前日期(精确到毫秒)
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

global Dataset  # UP,IN,KSC
dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
# print(data_hsi)
# print('..............',gt_hsi)



h,w,b=data_hsi.shape
# print(h)
xslice=data_hsi.reshape(h*w,b)
c=[]

for j in range(b):
    # print(j)
    xvar = xslice[:,j]
    # print(xvar)
    xvarm =np.mean(xvar)
    # print(xvarm)
    xvarm_std = np.std(xvar)
    # print(xvarm_std)
    # xvar1 = abs(xvar-xvarm)/xvarm_std
    # print(xvar1)
    # xvarm1 =np.mean(xvar1)
    # print('dsdada',xvarm1)
    # xvarm_std1=np.std(xvar1)
    # print('dsdada2', xvarm_std1)
    xvar=xvarm_std/xvarm
    # print(xvar*100)
    c.append(xvar)
    # if (xvar*100)>=25:
    # # xn = np.var(xvar)
    #     b.append(j)
# #     print(j)
c = np.argsort(c)
i=20
c =c[i:]
# print('....', c)
c=sorted(c)
data_hsi=data_hsi[:,:,c]

ls = LDA_SLIC.LDA_SLIC(data_hsi, np.reshape( gt_hsi,[h,w]))
data_hsi= ls.LDA_Process( np.reshape( gt_hsi,[h,w])).reshape(h,w,-1)
# identify = data_hsi
# xslice = identify.reshape(h*w, -1)
# k=[20,9376,2470,0,4643,895,6261,10548,4911,8867,898,97,318,16989,1425,71,1931]#in4
# k=[0,91,19429,98609,170,49432,90301,100799,49003,51815]#up4
# k=[0,106198,136103,153952,107895,93213,129529,212360,42318,212817,200487,82672,50700,76331]#ksc18
# k=[0,48911,39633,31856,487,933,305,338,19620,82710,63914,52609,55000,58916,61739,1133,95697]#sv
# k=[12197,13229,289832,17218,83990,98267,15263,29463,5802,18410,12715,50049,172657,187291,272682]#hs
# 光谱
# for i in range(len(k)):
#     xslice1=xslice[k[i],:]
#     # plt.figure()
#
#     plt.scatter(xslice1)
#     # plt.show()
#     # plt.plot(xslice1)
# plt.show()
# b0 = list(np.where(a == 2))[0]
# xslice1 = xsliceb[b0,:].reshape(100,-1)
# print(xslice1.shape)
# b1 = list(np.where(a == 15))[0]
# xslice2 = xsliceb[b1,:].reshape(100,-1)
#
# for j in range(100):
#     x=j
#     for i in range(46):
#         y=xslice1[j,i]
#         plt.scatter(x, y, c='red', s=1)  # s为点的大小
#     # slice1 = xslice1[:,j]
#     # plt.plot(slice1)
#     # plt.scatter(x, y,c='red', s=1)  # s为点的大小
# for j in range(100):
#     x1=j
#     for i in range(46):
#         y1=xslice2[j,i]
#         plt.scatter(x1, y1, c='b', s=1)  # s为点的大小
#     # y1=xslice2[j,1]
#     # slice1 = xslice1[:,j]
#     # plt.plot(slice1)
#     # plt.scatter(x1, y1,c='b', s=1)  # s为点的大小
# plt.figure()
# plt.show()
#
print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 1
PATCH_LENGTH = 4
# number of training samples per class
# lr, num_epochs, batch_size = 0.001, 200, 32
lr, num_epochs, batch_size = 0.001, 300 ,64
# lr, num_epochs, batch_size = 0.00050, 200, 16
# lr, num_epochs, batch_size = 0.0005, 200, 12
# net = network.DBDA_network_drop(BAND, CLASSES_NUM)
# net = network.DBDA_network_PReLU(BAND, CLASSES_NUM)
# net = network.DBMA_network(BAND, CLASSES_NUM)
# optimizer = optim.Adam(net.parameters(), lr=lr) #, weight_decay=0.0001)

loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]#200
INPUT_DIMENSION = data_hsi.shape[2]#200
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]#145*145
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_#(145,145,200)
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    print('iter:', index_iter)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    train_iter, valida_iter, test_iter, all_iter= generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                               whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    net = network.GDPA_LDSICS(BAND, CLASSES_NUM)
    # net = net.to(device)
    # summary(net, (1, 9, 9, BAND))
    SAVE_PATH3 = net.name + '.pth'
    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)  #  weight_decay=0.0001)
    tic1 = time.perf_counter()

    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    best_acc = 0
    # trainloss_txt ="E:\\新建文件夹\\ocean\\train\\up_trainLOSS.txt"
    for epoch in range(num_epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        # 原来耐心值5
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 11, eta_min=0.0, last_epoch=-1)#余先退火
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())  # 损失值计算

            optimizer.zero_grad()  # 梯度清零
            l.backward()  ##反向传播
            optimizer.step()  ##更新梯度
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)

        acc_sum, n = 0.0, 0

        with torch.no_grad():
            for X, y in valida_iter:
                test_l_sum, test_num = 0, 0
                X = X.to(device)
                y = y.to(device)
                net.eval()  # 评估模式, 这会关闭dropout
                y_hat = net(X)
                l = loss(y_hat, y.long())
                acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
                test_l_sum += l
                test_num += 1
                net.train()  # 改回训练模式
                n += y.shape[0]
        new_acc = acc_sum / n

        valida_acc = new_acc
        valida_loss = test_l_sum.cpu().numpy()
        loss_list.append(valida_loss)
        if valida_acc > best_acc:
            torch.save(net.state_dict(), SAVE_PATH3)
            best_acc = valida_acc

        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)
        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
              % (
              epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, best_acc, time.time() - time_epoch))
        # outp='epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f'% (epoch + 1,
        #                                 train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc)

        # with open(trainloss_txt,"a+") as f:
        #     f.write(outp+'\n')
        #     f.close
        early_num=20
        early_stopping = False
        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0: # and valida_acc > 0.9:
                torch.save(net.state_dict(), SAVE_PATH3)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(SAVE_PATH3))
                break
        else:
            early_epoch = 0
    #train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.perf_counter()

    # labels=['Alfalfa','C-notill','C-mintill','Corn','G-pasture','G-trees','G-p-mowed','H-windrowed','Oats','S-notill','S-mintill','S-clean','Wheat','Woods','B-G-T-Drivers','S-Steel-Towers']
    # confusion = ConfusionMatrix(num_classes=CLASSES_NUM, labels=labels)
    pred_test_fdssc = []
    tic2 = time.perf_counter()
    net.load_state_dict(torch.load(SAVE_PATH3))
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)

            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            # print(net(X))
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.perf_counter()

    collections.Counter(pred_test_fdssc)

    gt_test = gt[test_indices] - 1


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    torch.save(net.state_dict(), "./net/" + str(round(overall_acc_fdssc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc
#
print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'IN/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt')


# generate_png(net,Dataset, device,gt_hsi,data_hsi,image_x*image_y,BAND,PATCH_LENGTH)
generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)







