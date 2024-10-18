from keras_cv_attention_models import visualizing, test_images, resnest
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import math
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW
from keras_cv_attention_models import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from edafa import ClassPredictor
from tqdm import tqdm
import copy
import os
from keras.callbacks import ModelCheckpoint
from skimage.transform import radon, iradon
import tensorflow_model_optimization as tfmot
import random
from keras.models import Model
import time

size = 224
channel = 1
batch = 8
theta = np.linspace(0., 180., size, endpoint=False)

class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
       
        ## For normal case
        batch_x = [np.array(self.x[i]).reshape(size, size, channel) for i in indices]
        batch_y = [self.y[i] for i in indices]
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
   
def spatial_filtering(binary_image):

    result_image = np.copy(binary_image)
    rows, cols = binary_image.shape
    
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2),
                  (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]

    # directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    for i in range(rows):
        for j in range(cols):
            all_zero_neighbors = True
            if binary_image[i, j] != 0:
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < rows and 0 <= y < cols:
                        if binary_image[x, y] == 1:
                            all_zero_neighbors = False                    
                            break
                if all_zero_neighbors:
                    result_image[i, j] = 0
    return result_image

test_data = np.load('/home/cocel/sh/samsung/1106_new_data/data/test_data.npy').reshape(-1, size, size)
test_data_radon = np.load('/home/cocel/sh/samsung/1106_new_data/radon_test_data.npy').reshape(-1, 1, size, size)
test_label = np.load('/home/cocel/sh/samsung/1106_new_data/data/test_label.npy')

print(test_data_radon[0][0][10:20])
raise
# test_data_ = tf.image.per_image_standardization(test_data) * 0.23 + 0.49
# test_data_radon_ = tf.image.per_image_standardization(test_data_radon) * 0.23 + 0.49
# test_loader = Dataloader(test_data_, test_label, batch)
# test_loader_radon = Dataloader(test_data_radon_, test_label, batch)

mm = keras.models.load_model("/home/cocel/sh/samsung/model/test/26-0.8687.h5")
mm_radon = keras.models.load_model("/home/cocel/sh/samsung/model/test/efficient_radon_new_data.h5")

# mm.evaluate(test_loader)
# mm_radon.evaluate(test_loader_radon)
# raise
# True_value = np.argmax(test_label, axis = -1)

# predictions1 = np.argmax(mm.predict(test_loader), axis = -1)
# predictions2 = np.argmax(mm_radon.predict(test_loader_radon), axis = -1)

# confusion_matrix1 = metrics.confusion_matrix(True_value, predictions1)
# confusion_matrix2 = metrics.confusion_matrix(True_value, predictions2)

# cm_display1 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1, display_labels = ["100", "102", "110", "115", "120", "130", "138", "139", "140", "155", "156","200"])
# cm_display1.plot()
# plt.show()

# cm_display2 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2, display_labels = ["100", "102", "110", "115", "120", "130", "138", "139", "140", "155", "156","200"])
# cm_display2.plot()
# plt.show()


for index in range(len(test_label)):

    img_vis = test_data[index].copy()
    img_vis_radon = test_data_radon[index].copy()


    # test_data = np.load('mixed_test_data.npy')
    test_data_std = tf.image.per_image_standardization(test_data) * 0.23 + 0.49
    test_data_radon_std = tf.image.per_image_standardization(test_data_radon) * 0.23 + 0.49

    img = test_data_std[index]
    img_radon = test_data_radon_std[index][0]

    superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, img, img_vis, alpha=0.95, layer_name="auto")
    superimposed_img_radon, heatmap_radon, preds_radon = visualizing.make_and_apply_gradcam_heatmap(mm_radon, img_radon, img_vis_radon, alpha=0.2, layer_name="auto")
    
    if np.argmax(preds) != np.argmax(test_label[index]):
        save_name = './fault_image/original/'+ str(index) + '_' + str(np.argmax(test_label[index])) + '_' + str(np.argmax(preds)) + '.png'
        plt.imsave(save_name, superimposed_img)
    if np.argmax(preds_radon) != np.argmax(test_label[index]):
        save_name_radon = './fault_image/radon/'+ str(index) + '_' + str(np.argmax(test_label[index])) + '_' + str(np.argmax(preds_radon)) + '.png'
        plt.imsave(save_name_radon, superimposed_img_radon)    
    
    
    print(f'index {index} has been saved!')
    print(np.argmax(test_label[index]))
    print(np.argmax(preds))
    print(np.argmax(preds_radon))
    
# plt.subplot(2, 2, 1)
# plt.imshow(img)
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(superimposed_img)
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(img_radon)
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(superimposed_img_radon)
# plt.axis('off')

# plt.show()