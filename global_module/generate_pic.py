import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from global_module.Utils import extract_samll_cubic
import torch.utils.data as Data
from sklearn import metrics
import sys,os
sys.path.append(os.pardir)
from sklearn import metrics, preprocessing
def load_dataset(Dataset):
    if Dataset == 'IN':
        mat_data = sio.loadmat('datasets/Indian_pines_corrected.mat')#图像数据
        mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')#标签数据
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PC':
        uPavia = sio.loadmat('datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.997
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'SV':
        SV = sio.loadmat('datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('datasets/KSC.mat')
        gt_KSC = sio.loadmat('datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'HS':
        HS = sio.loadmat('datasets/houston.mat')
        gt_HS = sio.loadmat('datasets/houston_gt_sum.mat')
        data_hsi = HS['hsi']
        gt_hsi = gt_HS['houston_gt_sum']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'BS':
        BS = sio.loadmat('datasets/Botswana.mat')
        gt_BS = sio.loadmat('datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.988
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampl(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)#标签16
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]#依次找出标签所对应的索引
        np.random.shuffle(indexes)#打乱索引
        labels_loc[i] = indexes
        # if proportion != 1:
        #     nb_val = max(int((1 - proportion) * len(indexes)), 3)
        # else:
        #     nb_val = 0
        train_num=100
        sample_num = train_num
        max_index = np.max(len(indexes)) + 1
        if sample_num > max_index:
            sample_num = 15
        else:
            sample_num = train_num
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:sample_num]#生成训练样本
        test[i] = indexes[sample_num:]#生成测试样本
    train_indexes = []
    test_indexes = []
    #生成训练和测试样本索引
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)#标签16
    for i in range(m):

        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]#依次找出标签所对应的索引
        np.random.shuffle(indexes)#打乱索引
        print(len(indexes))
        labels_loc[i] = indexes
        if proportion != 1:
            # nb_val = max(int((1 - proportion) * len(indexes)), 3)
            nb_val = max(np.ceil((1 - proportion) * len(indexes)).astype('int'),3)
        else:
            nb_val = 0
        # train_num=100
        # sample_num = train_num
        # max_index = np.max(len(indexes)) + 1
        # if sample_num > max_index:
        #     sample_num = 15
        # else:
        #     sample_num = train_num
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]#生成训练样本
        # print(',,,',len(train[i]))
        test[i] = indexes[nb_val:]#生成测试样本


    train_indexes = []
    test_indexes = []
    #生成训练和测试样本索引
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0
# def list_to_colormap(x_list):
#     y = np.zeros((x_list.shape[0], 3))
#     for index, item in enumerate(x_list):
#         if item == 0:
#             y[index] = np.array([0, 0,0]) / 255.
#         if item == 1:
#             y[index] = np.array([255, 255, 255]) / 255.
#         if item == 2:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 3:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 4:
#             y[index] = np.array([0, 0, 0]) / 255.
#         if item == 5:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 6:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 7:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 8:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 9:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 10:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 11:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 12:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 13:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 14:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 15:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 16:
#             y[index] = np.array([0,0,0]) / 255.
#         if item == 17:
#             y[index] = np.array([215, 255, 0]) / 255.
#         if item == 18:
#             y[index] = np.array([0, 255, 215]) / 255.
#         if item == -1:
#             y[index] = np.array([0, 0, 0]) / 255.
#     return y
#in
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([151, 30, 94]) / 255.
        if item == 1:
            y[index] = np.array([240, 249, 132]) / 255.
        if item == 2:
            y[index] = np.array([230, 239, 37]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([97, 228, 58]) / 255.
        if item == 5:
            y[index] = np.array([47, 147, 19]) / 255.
        if item == 6:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 7:
            y[index] = np.array([44, 206, 48]) / 255.
        if item == 8:
            y[index] = np.array([190, 183, 35]) / 255.
        if item == 9:
            y[index] = np.array([226, 220, 99]) / 255.
        if item == 10:
            y[index] = np.array([6, 66, 7]) / 255.
        if item == 11:
            y[index] = np.array([156, 245, 158]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([52, 177, 62]) / 255.
        if item == 14:
            y[index] = np.array([43, 85, 45]) / 255.
        if item == 15:
            y[index] = np.array([77, 85, 43]) / 255.
        if item == 16:
            y[index] = np.array([199, 199, 199]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y

# ksc
# def list_to_colormap(x_list):
#     y = np.zeros((x_list.shape[0], 3))
#     for index, item in enumerate(x_list):
#         if item == 0:
#             y[index] = np.array([23, 147, 2]) / 255.
#         if item == 1:
#             y[index] = np.array([176, 164, 89]) / 255.
#         if item == 2:
#             y[index] = np.array([175, 254, 224]) / 255.
#         if item == 3:
#             y[index] = np.array([26, 109, 14]) / 255.
#         if item == 4:
#             y[index] = np.array([117, 146, 20]) / 255.
#         if item == 5:
#             y[index] = np.array([112, 111, 61]) / 255.
#         if item == 6:
#             y[index] = np.array([0, 255, 0]) / 255.
#         if item == 7:
#             y[index] = np.array([188, 197, 109]) / 255.
#         if item == 8:
#             y[index] = np.array([164, 51, 139]) / 255.
#         if item == 9:
#             y[index] = np.array([253, 197,66]) / 255.
#         if item == 10:
#             y[index] = np.array([240, 240, 240]) / 255.
#         if item == 11:
#             y[index] = np.array([160, 86, 31]) / 255.
#         if item == 12:
#             y[index] = np.array([38, 89, 242]) / 255.
#         if item == 13:
#             y[index] = np.array([52, 177, 62]) / 255.
#         if item == 14:
#             y[index] = np.array([43, 85, 45]) / 255.
#         if item == 15:
#             y[index] = np.array([77, 85, 43]) / 255.
#         if item == 16:
#             y[index] = np.array([196, 196, 196]) / 255.
#         if item == 17:
#             y[index] = np.array([215, 255, 0]) / 255.
#         if item == 18:
#             y[index] = np.array([0, 255, 215]) / 255.
#         if item == -1:
#             y[index] = np.array([0, 0, 0]) / 255.
#     return y
# # up
# def list_to_colormap(x_list):
#     y = np.zeros((x_list.shape[0], 3))
#     for index, item in enumerate(x_list):
#         if item == 0:
#             y[index] = np.array([60, 60, 60]) / 255.
#         if item == 1:
#             y[index] = np.array([0, 100, 0]) / 255.
#         if item == 2:
#             y[index] = np.array([152, 129, 90]) / 255.
#         if item == 3:
#             y[index] = np.array([0, 200, 0]) / 255.
#         if item == 4:
#             y[index] = np.array([228, 228, 228]) / 255.
#         if item == 5:
#             y[index] = np.array([91, 69, 0]) / 255.
#         if item == 6:
#             y[index] = np.array([103, 88, 69]) / 255.
#         if item == 7:
#             y[index] = np.array([191, 0, 0]) / 255.
#         if item == 8:
#             y[index] = np.array([100, 100, 100]) / 255.
#         if item == 9:
#             y[index] = np.array([226, 220, 99]) / 255.
#         if item == 10:
#             y[index] = np.array([6, 66, 7]) / 255.
#         if item == 11:
#             y[index] = np.array([156, 245, 158]) / 255.
#         if item == 12:
#             y[index] = np.array([0, 128, 128]) / 255.
#         if item == 13:
#             y[index] = np.array([52, 177, 62]) / 255.
#         if item == 14:
#             y[index] = np.array([43, 85, 45]) / 255.
#         if item == 15:
#             y[index] = np.array([77, 85, 43]) / 255.
#         if item == 16:
#             y[index] = np.array([196, 196, 196]) / 255.
#         if item == 17:
#             y[index] = np.array([215, 255, 0]) / 255.
#         if item == 18:
#             y[index] = np.array([0, 255, 215]) / 255.
#         if item == -1:
#             y[index] = np.array([0, 0, 0]) / 255.
#     return y



def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    print('123456', train_data.shape)


    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print('y_val', np.unique(y_val))
    # print('y_test', np.unique(y_test))
    # print(y_val)
    # print(y_test)

    # K.clear_session()  # clear session before next loop

    # print(y1_train)
    #y1_train = to_categorical(y1_train)  # to one-hot labels
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)
    # print('x1_tensor_train', len(torch_dataset_train))
    # print('y1_tensor_train', y1_tensor_train.shape)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    print(all_tensor_data.shape)
    print( all_tensor_data_label.shape)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test

def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    # with torch.no_grad():
    #     for i in range(len(gt_hsi)):
    #         if i == 0:
    #             pred_test.extend([-1])
    #         else:
    #             X = all_iter[i].to(device)
    #             net.eval()  # 评估模式, 这会关闭dropout
    #             # print(net(X))
    #             pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

        # for X, y in all_iter:
        #     #for data, label in X, y:
        #     if y.item() != 0:
        #         # print(X)
        #         X = X.to(device)
        #         net.eval()  # 评估模式, 这会关闭dropout
        #         y_hat = net(X)
        #         # print(net(X))
        #         pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
        #     else:
        #         pred_test.extend([-1])
    for X, y in all_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        # print(net(X))
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            # x[i] = 16
            x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    #path = '../' + net.name
    path = '../GDPA_LDSICS'
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + net.name + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')
# def sampling1(proportion, ground_truth):
#     train = {}
#     test = {}
#     labels_loc = {}
#     m = max(ground_truth) + 1
#     for i in range(m):
#         indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i]
#
#         np.random.shuffle(indexes)
#         labels_loc[i] = indexes
#         if proportion != 1:
#             nb_val = max(int((1 - proportion) * len(indexes)), 3)
#         else:
#             nb_val = 0
#         # print(i, nb_val, indexes[:nb_val])
#         # train[i] = indexes[:-nb_val]
#         # test[i] = indexes[-nb_val:]
#         train[i] = indexes[:nb_val]
#         test[i] = indexes[nb_val:]
#     train_indexes = []
#     test_indexes = []
#     for i in range(m):
#         train_indexes += train[i]
#         test_indexes += test[i]
#     np.random.shuffle(train_indexes)
#     np.random.shuffle(test_indexes)
#     return train_indexes, test_indexes
#
#
# def generate_png(net, Dataset, device, gt_hsi, data_hsi, xy, band, PATCH_LENGTH):
#     X = data_hsi
#     # gt_re =  indian_pines_gt
#     y = gt_hsi
#
#     y = y.reshape(np.prod(y.shape[:2]), )
#     _, total_indices1 = sampling1(1, y)#全部索引
#     # print(len(total_indices1) )
#     gt_all1 = y[total_indices1] - 1
#
#     # X = X.reshape(np.prod(X.shape[:2]), np.prod(X.shape[2:]))
#
#     data = X.reshape(np.prod(X.shape[:2]), np.prod(X.shape[2:]))
#     data = preprocessing.scale(data)
#     # 沿着某个轴标准化数据集，以均值为中心，以分量为单位方差。
#     data_ = data.reshape(X.shape[0], X.shape[1], X.shape[2])
#     print(data_.shape)
#     whole_data = data_
#     padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
#                              'constant', constant_values=0)
#
#     all_data1 = extract_samll_cubic.select_small_cubic(xy, total_indices1, whole_data,
#                                                        PATCH_LENGTH, padded_data, band)
#
#     all_data1.reshape(all_data1.shape[0], all_data1.shape[1], all_data1.shape[2], band)
#     print(all_data1.shape)
#     print(len(gt_all1))
#     all_tensor_data1 = torch.from_numpy(all_data1).type(torch.FloatTensor).unsqueeze(1)
#     all_tensor_data_label1 = torch.from_numpy(gt_all1).type(torch.FloatTensor)
#     torch_dataset_all1 = Data.TensorDataset(all_tensor_data1, all_tensor_data_label1)
#
#     all_iter1 = Data.DataLoader(
#         dataset=torch_dataset_all1,  # torch TensorDataset format
#         batch_size=64,  # mini batch size
#         shuffle=False,  # 要不要打乱数据 (打乱比较好)
#         num_workers=0,  # 多线程来读数据
#     )
#
#     pred = []  # 10249    total_indices=[10249]        total_indices=21045    pred_test=21045
#     for X, y in all_iter1:
#         X = X.to(device)
#         net.eval()  # 评估模式, 这会关闭dropout
#         # print(net(X))
#         pred.extend(np.array(net(X).cpu().argmax(axis=1)))
#
#     y = gt_hsi.flatten()
#     print(len(y))
#     x_label = np.zeros(y.shape)
#     for i in range(len(y)):
#         if y[i] == 0:
#             y[i] = 17
#             # x[i] = 16
#             x_label[i] = 16
#         # else:
#         #     x_label[i] = pred_test[label_list]
#         #     label_list += 1
#     y = y[:] - 1
#     print('123456', len(pred))
#     print('123456', len(total_indices1))
#     x_label[total_indices1] = pred
#     x = np.ravel(x_label)
#
#     y_list = list_to_colormap(x)
#     y_gt = list_to_colormap(y)
#
#     y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
#     gt = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
#     path = '../CNN_GCN_A'
#     classification_map(y_re, gt, 300,
#                        path + '/classification_maps/' + Dataset + '_' + net.name + '.png')
#     classification_map(gt, gt, 300,
#                        path + '/classification_maps/' + Dataset + '_gt.png')
#     print('------Get classification maps successful-------')



