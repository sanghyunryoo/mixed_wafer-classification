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
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

size = 52
batch = 8
channel = 1

theta = np.linspace(0., 180., size, endpoint=False)

def change_value_1_0(data):
    value_to_find = 1
    new_value = 0
    indices = np.where(data == value_to_find)
    data[indices] = new_value
    return data

def change_value_2_1(data):
    value_to_find = 2
    new_value = 1
    indices = np.where(data == value_to_find)
    data[indices] = new_value
    return data

def spatial_filtering(binary_image):

    result_image = np.copy(binary_image)
    rows, cols = binary_image.shape
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
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

arr0 = np.load('/home/cocel/sh/samsung/MixedWM38/arr_0.npy')
arr1 = np.load('/home/cocel/sh/samsung/MixedWM38/arr_1.npy')

arr0 = change_value_2_1(change_value_1_0(arr0))

new_arr0 = []
new_arr1 = []

multi_arr0 = []
multi_arr1 = []

for i in range(len(arr1)):
    if arr1[i].sum() == 1 or arr1[i].sum() == 0:
        new_arr1.append(arr1[i])
        new_arr0.append(arr0[i])
    else:
        multi_arr1.append(arr1[i])
        multi_arr0.append(arr0[i])
                
new_arr0 = np.array(new_arr0)
new_arr1 = np.array(new_arr1)

multi_arr0 = np.array(multi_arr0)
multi_arr1 = np.array(multi_arr1)

index_list = [0]
for i in range(1, len(arr1)-1):
    if np.mean(np.equal(arr1[i], arr1[i-1])) != 1.0:
        index_list.append(i)
index_list.append(len(arr1))
print(index_list)
      
        
X_train, y_train, X_test, y_test = iterative_train_test_split(arr0, arr1, test_size=0.4)
X_train, y_train, X_val, y_val = iterative_train_test_split(X_train, y_train, test_size=0.2)
       
single_X_train, single_X_test, single_y_train, single_y_test = train_test_split(new_arr0, new_arr1, test_size=0.4, stratify=new_arr1, random_state=42)
single_X_train, single_X_val, single_y_train, single_y_val = train_test_split(single_X_train, single_y_train, test_size=0.2, stratify=single_y_train, random_state=42)

multi_X_train, multi_y_train, multi_X_test, multi_y_test = iterative_train_test_split(multi_arr0, multi_arr1, test_size=0.4)
multi_X_train, multi_y_train, multi_X_val, multi_y_val = iterative_train_test_split(multi_X_train, multi_y_train, test_size=0.2)
raise
# np.save('mixed_train_data.npy', X_train)
# np.save('mixed_val_data.npy', X_val)
# np.save('mixed_test_data.npy', X_test)

# np.save('mixed_train_label.npy', y_train)
# np.save('mixed_val_label.npy', y_val)
# np.save('mixed_test_label.npy', y_test)

# np.save('single_train_data.npy', single_X_train)
# np.save('single_val_data.npy', single_X_val)
# np.save('single_test_data.npy', single_X_test)

# np.save('single_train_label.npy', single_y_train)
# np.save('single_val_label.npy', single_y_val)
# np.save('single_test_label.npy', single_y_test)

# np.save('multi_train_data.npy', multi_X_train)
# np.save('multi_val_data.npy', multi_X_val)
# np.save('multi_test_data.npy', multi_X_test)

# np.save('multi_train_label.npy', multi_y_train)
# np.save('multi_val_label.npy', multi_y_val)
# np.save('multi_test_label.npy', multi_y_test)

### Anomaly data generation
total_data = []
for i in tqdm(range(len(X_test))):
    total_data.append(cv2.bitwise_not(X_test[i]))
np.save('Anomaly_data_test.npy', np.array(total_data))
print('Anomaly Test Data Saved . . .')

print('Data Saving . . .')
time.sleep(2)
print('Data Loading . . .')
mixed_train_data = np.load('mixed_train_data.npy').reshape(-1, 1, size, size) # 22809
mixed_val_data = np.load('mixed_val_data.npy').reshape(-1, 1, size, size) # 7603
mixed_test_data = np.load('mixed_test_data.npy').reshape(-1, 1, size, size) # 7603

mixed_train_label = np.load('mixed_train_label.npy')
mixed_val_label = np.load('mixed_val_label.npy')
mixed_test_label = np.load('mixed_test_label.npy')

single_train_data = np.load('single_train_data.npy').reshape(-1, 1, size, size) # 22809
single_val_data = np.load('single_val_data.npy').reshape(-1, 1, size, size) # 7603
single_test_data = np.load('single_test_data.npy').reshape(-1, 1, size, size) # 7603

single_train_label = np.load('single_train_label.npy')
single_val_label = np.load('single_val_label.npy')
single_test_label = np.load('single_test_label.npy')

multi_train_data = np.load('multi_train_data.npy').reshape(-1, 1, size, size) # 22809
multi_val_data = np.load('multi_val_data.npy').reshape(-1, 1, size, size) # 7603
multi_test_data = np.load('multi_test_data.npy').reshape(-1, 1, size, size) # 7603

multi_train_label = np.load('multi_train_label.npy')
multi_val_label = np.load('multi_val_label.npy')
multi_test_label = np.load('multi_test_label.npy')

### Mixed
total_data = []
for i in tqdm(range(len(mixed_train_data))):
    radon_data = radon(mixed_train_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('mixed_train_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(mixed_val_data))):
    radon_data = radon(mixed_val_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('mixed_val_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(mixed_test_data))):
    radon_data = radon(mixed_test_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('mixed_test_data_radon.npy', np.array(total_data))

### Single
total_data = []
for i in tqdm(range(len(single_train_data))):
    radon_data = radon(single_train_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('single_train_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(single_val_data))):
    radon_data = radon(single_val_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('single_val_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(single_test_data))):
    radon_data = radon(single_test_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('single_test_data_radon.npy', np.array(total_data))

### Multi
total_data = []
for i in tqdm(range(len(multi_train_data))):
    radon_data = radon(multi_train_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('multi_train_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(multi_val_data))):
    radon_data = radon(multi_val_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('multi_val_data_radon.npy', np.array(total_data))

total_data = []
for i in tqdm(range(len(multi_test_data))):
    radon_data = radon(multi_test_data[i][0]*255., theta=theta)
    total_data.append(radon_data)
np.save('multi_test_data_radon.npy', np.array(total_data))

raise

train_data = np.load('mixed_train_data.npy')
val_data = np.load('mixed_val_data.npy')    
test_data = np.load('mixed_test_data.npy')
spatial_filter_test_data = test_data.copy()

train_label = np.load('mixed_train_label.npy')
val_label = np.load('mixed_val_label.npy')
test_label = np.load('mixed_test_label.npy')

# index = 0
# print(test_data[index][10])
# print(test_label[index])
# plt.subplot(1, 2, 1)
# plt.imshow(test_data[index])
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(radon(test_data[index], theta=theta))
# plt.axis('off')
# plt.show()
# raise
### Binary Split
binary_train_label = []
binary_val_label = []
binary_test_label = []

# for i in range(len(train_label)):
#     if np.sum(train_label[i]) == 1:
#         binary_train_label.append(0)
#     else:
#         binary_train_label.append(1)
# for i in range(len(val_label)):
#     if np.sum(val_label[i]) == 1:
#         binary_val_label.append(0)
#     else:
#         binary_val_label.append(1)
# for i in range(len(test_label)):
#     if np.sum(test_label[i]) == 1:
#         binary_test_label.append(0)
#     else:
#         binary_test_label.append(1)

for i in range(len(train_label)):
    binary_train_label.append(np.sum(train_label[i]))
for i in range(len(val_label)):
    binary_val_label.append(np.sum(val_label[i]))
for i in range(len(test_label)):
    binary_test_label.append(np.sum(test_label[i]))

binary_train_label = np.eye(5)[np.array(binary_train_label)]
binary_val_label = np.eye(5)[np.array(binary_val_label)]
binary_test_label = np.eye(5)[np.array(binary_test_label)]


single_train_data = []
single_val_data = []
single_test_data = []

single_train_label = []
single_val_label = []
single_test_label = []

for i in range(len(train_label)):
    if np.sum(train_label[i]) == 1:
        single_train_label.append(train_label[i])
        single_train_data.append(train_data[i])
for i in range(len(val_label)):
    if np.sum(val_label[i]) == 1:
        single_val_label.append(val_label[i])
        single_val_data.append(val_data[i])
for i in range(len(test_label)):
    if np.sum(test_label[i]) == 1:
        single_test_label.append(test_label[i])
        single_test_data.append(test_data[i])
        
single_train_data = np.array(single_train_data)
single_val_data = np.array(single_val_data)
single_test_data = np.array(single_test_data)

single_train_label = np.array(single_train_label)
single_val_label = np.array(single_val_label)
single_test_label = np.array(single_test_label)
       
def spatial_filtering(binary_image):

    result_image = np.copy(binary_image)
    rows, cols = binary_image.shape
    
    # directions = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2),
    #               (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    for i in range(rows):
        for j in range(cols):
            all_zero_neighbors = True
            if binary_image[i, j] != 0:
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < rows and 0 <= y < cols:
                        if binary_image[x, y] == 0.8952564:
                            all_zero_neighbors = False                    
                            break
                if all_zero_neighbors:
                    result_image[i, j] = 0.3594663
    return result_image

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
        # batch_x = [np.array(self.x[i]).reshape(size, size, channel) for i in indices]
        batch_x = [radon(np.array(self.x[i]), theta=theta).reshape(size, size, channel) for i in indices]
        batch_y = [self.y[i] for i in indices]
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
class Dataloader_radon(Sequence):

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
        # batch_x = [np.array(self.x[i]).reshape(size, size, channel) for i in indices]
        batch_x = [radon(np.array(self.x[i]), theta=theta).reshape(size, size, channel) for i in indices]
        batch_y = [self.y[i] for i in indices]
        return np.array(batch_x), np.array(batch_y)
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            

train_data = tf.image.per_image_standardization(train_data) * 0.23 + 0.49
val_data = tf.image.per_image_standardization(val_data) * 0.23 + 0.49
test_data = tf.image.per_image_standardization(test_data) * 0.23 + 0.49

train_loader = Dataloader(train_data, train_label, batch, shuffle=True)
val_loader = Dataloader(val_data, val_label, batch, shuffle=True)
test_loader = Dataloader(test_data, test_label, batch)

# test_loader_radon = Dataloader_radon(test_data, single_test_label, batch)

print('Data Load Done -------------------------------------------------\n')
model = efficientnet.EfficientNetV1B5(input_shape=(size, size, channel), dropout=0.2, num_classes=8, classifier_activation='sigmoid', pretrained="imagenet")
# model = maxvit.MaxViT_Base(input_shape=(size, size, channel), dropout=0.2, num_classes=8, classifier_activation='sigmoid', pretrained="imagenet") 
# model = levit.LeViT384(input_shape=(size, size, channel), num_classes=8, pretrained="imagenet", use_distillation=False, classifier_activation='softmax', dropout=0.2) 

model.compile(AdamW(learning_rate=0.0003, weight_decay=False),
              loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(AdamW(learning_rate=0.0003, weight_decay=False),
#               loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
#               metrics=['accuracy'])
print('Model Selected -------------------------------------------------\n')

MODEL_SAVE_FOLDER_PATH = '/home/cocel/sh/samsung/model/mixed_test/' + str(time.time()) + '/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

model.fit(train_loader, validation_data=val_loader, epochs=50, callbacks=[cb_checkpoint])

print('Train Done -------------------------------------------------\n')


# sm_classifier = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713152810.2051215/42-0.0198.h5") # Original 99.03%
sm_classifier = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713169001.696304/37-0.0149.h5") # Radon 99.43%

mlc_model = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713273151.8846905/29-0.0169.h5") # Radon

mlc_model.evaluate(test_loader)
raise
# mlc_model = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1712140684.5930784/35-0.0161.h5") # Original

# mcc_model = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713159552.0679545/41-0.6674.h5") # Original
mcc_model = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713160708.5963886/41-0.6587.h5") # Radon 99.29%
mcc_model2 = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713181730.2691243/19-0.6502.h5") # Radon 99.43%


multi_classifier = keras.models.load_model("/home/cocel/sh/samsung/model/mixed_test/1713191361.8346484/44-0.5650.h5")
multi_classifier.evaluate(test_loader)
raise


from keras.models import Sequential
from keras.layers import Dense


# X_train, X_val, y_train, y_val = train_test_split(val_data, val_label, test_size=0.3, shuffle=True)
# train_loader = Dataloader(X_train, y_train, batch, shuffle=True)
# val_loader = Dataloader(X_val, y_val, batch, shuffle=True)

from keras.models import Model
from keras.layers import Concatenate


import tensorflow as tf

def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''
    
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    
    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model

mlc_model = add_prefix(mlc_model, 'mlc_model_')
# 두 모델 합치기
concatenated = Concatenate()([mlc_model.output, multi_classifier.output])
x = Dense(10)(concatenated)
x = Dense(10)(x)
output = Dense(8)(x)

# 새로운 모델 정의
combined_model = Model(inputs=[mlc_model.input, multi_classifier.input], outputs=output)
combined_model.compile(AdamW(learning_rate=0.0003, weight_decay=False),
              loss='binary_crossentropy', metrics=['accuracy'])
MODEL_SAVE_FOLDER_PATH = '/home/cocel/sh/samsung/model/mixed_test/' + str(time.time()) + '/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)
combined_model.fit(train_loader, validation_data=val_loader, epochs=50, callbacks=[cb_checkpoint])


# classify = sm_classifier.predict(test_loader)

# a1 = 0
# b1 = 0
# a2 = 0
# b2 = 0
# for i in tqdm(range(len(classify))):
#     test_image = radon(test_data[i].numpy(), theta=theta)
    
#     if classify[i] < 0.5:
#         result = mlc_model.predict(tf.expand_dims(tf.expand_dims(test_image, 2), 0)) #* mlc_model.predict(tf.expand_dims(tf.expand_dims(test_image, 2), 0)) * mcc_model2.predict(tf.expand_dims(tf.expand_dims(test_image, 2), 0))
#         result = np.argmax(result)
#         if result == np.argmax(test_label[i]):
#             a1 += 1
#         else:
#             b1 += 1
#     else:
#         result = mlc_model.predict(tf.expand_dims(tf.expand_dims(test_image, 2), 0))
#         result = np.round(result)[0]
#         compare = np.array_equal(result, test_label[i])
        
#         if compare == True:
#             a2 += 1
#         else:
#             b2 += 1
            
            
            
# print(a1)
# print(b1)
# print(a2)
# print(b2)
 
# # print(a) # 7463 --> 7466 --> 
# # print(b) # 141  --> 138  --> 
# raise


binry_classifier.evaluate(test_loader)
raise

prediction2 = model2.predict(test_loader_radon) * model.predict(test_loader)
prediction2 = prediction2 > 0.5
prediction2 = prediction2.astype(int)

prediction = model.predict(test_loader)
prediction = prediction > 0.5
prediction = prediction.astype(int)

true = 0
false = 0

for i in range(len(test_label)):
    if np.all(np.equal(test_label[i], prediction2[i]) == True) == True:
        true += 1
    else:
        false += 1
        
print(true/len(test_label))
print(true)
print(false)

true = 0
false = 0
for i in range(len(test_label)):
    if np.all(np.equal(test_label[i], prediction[i]) == True) == True:
        true += 1
    else:
        false += 1
        
print(true/len(test_label))
print(true)
print(false)

# print(np.mean(np.equal(test_label, prediction)))
raise
model2.evaluate(test_loader_radon)
model.evaluate(test_loader)

raise
from sklearn.metrics import *

result = model.predict(test_loader) * model2.predict(test_loader_radon)
result = result > 0.5
result = result.astype(int)

print(accuracy_score(test_label, result,normalize=False))
print('average=None : ', precision_score(test_label, result, average=None))
print('average=\'macro\' : ', precision_score(test_label, result, average='macro'))
print('average=\'micro\' : ', precision_score(test_label, result, average='micro'))
print('average=\'weighted\' : ', precision_score(test_label, result, average='weighted'))
print('\n')
print(classification_report(test_label, result))