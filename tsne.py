import scipy.io as sio
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne(node_features,node_labels):

# node_features = sio.loadmat("indian_pines.mat")['outputs']
#   node_labels = sio.loadmat("E:/高光谱图像分类/ocean/Semi-supersvise CNN_GCN/Data/indian_pines_gt.mat")['indian_pines_gt'].reshape(-1)
    node_features=node_features
    # print(node_features.shape)
    node_labels =node_labels.reshape(-1)
    # print(node_labels.shape)
    ind = np.argwhere(node_labels!=0).reshape(-1)
    # print(ind.shape)
    node_features = node_features[ind,:]
    # print(node_features.shape)
    # print(node_features.shape)
    node_labels = node_labels[ind]

    num_classes = max(node_labels) # 9
    print(num_classes)
    c = ['chartreuse', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'skyblue', 'royalblue', 'fuchsia', 'tan', 'chocolate',
     'dodgerblue', 'yellowgreen']
    t_sne_embeddings = TSNE(n_components=2, perplexity=60, method='barnes_hut').fit_transform(node_features)

    fig = plt.figure(figsize=(10,8), dpi = 500)
    for class_id in range(16):
        print(class_id)
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=60, c=c[class_id],edgecolors='black', linewidths=0.2)
    plt.legend(loc = 1, prop = {'size':15})
    plt.xticks([])
    plt.yticks([])
    plt.show()


