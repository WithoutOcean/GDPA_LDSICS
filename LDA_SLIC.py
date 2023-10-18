import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
from sklearn import preprocessing
import cv2
import math





class LDA_SLIC(object):
    def __init__(self, data, labels):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels

    #
    def LDA_Process(self, curr_labels):
        '''
        :param curr_labels: height * width
        :return:
        '''
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis()  # n_components=self.n_component
        lda.fit(x, y - 1)
        X_new = lda.transform(self.x_flatt)
        print(X_new.shape)
        out = np.reshape(X_new, [self.height, self.width, -1])

        return out


