# loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
import warnings
import cv2

df=pd.read_pickle("/home/cocel/sh/samsung/wafer_dataset/WM811k_withlabel.pkl")
df.info()

data = np.array(df)

images_or = data[:, 1]
labels_or = data[:, 6]
train_test = data[:, 5]

train_labels = []
train_images = []

test_labels = []
test_images = []

for i in range(len(labels_or)):
    if train_test[i][0][0] == 'Training':
        if labels_or[i][0][0] == 'Center':
            train_labels.append(0)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Donut':
            train_labels.append(1)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Edge-Loc':
            train_labels.append(2)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Edge-Ring':
            train_labels.append(3)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Loc':
            train_labels.append(4)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Random':
            train_labels.append(5)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Scratch':
            train_labels.append(6)
            train_images.append(cv2.resize(images_or[i], (64, 64)))
    else:
        if labels_or[i][0][0] == 'Center':
            test_labels.append(0)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Donut':
            test_labels.append(1)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Edge-Loc':
            test_labels.append(2)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Edge-Ring':
            test_labels.append(3)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Loc':
            test_labels.append(4)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Random':
            test_labels.append(5)
            test_images.append(cv2.resize(images_or[i], (64, 64)))
        elif labels_or[i][0][0] == 'Scratch':
            test_labels.append(6)
            test_images.append(cv2.resize(images_or[i], (64, 64)))        

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


# value_to_find = 1
# new_value = 0

# indices = np.where(train_images == value_to_find)
# train_images[indices] = new_value

# indices = np.where(test_images == value_to_find)
# test_images[indices] = new_value

# np.save('wm811k_train_image.npy', train_images)
# np.save('wm811k_train_label.npy', train_labels)

# np.save('wm811k_test_image.npy', test_images)
# np.save('wm811k_test_label.npy', test_labels)


total_data = np.concatenate([train_images, test_images], 0)
label = np.concatenate([train_labels, test_labels], 0)

label = np.eye(7)[label]

import numpy as np

def stratified_split(images, labels, train_ratio=0.6, val_ratio=0.2):
    # Assuming labels are one-hot encoded
    n_classes = labels.shape[1]  # Number of classes
    n_samples = images.shape[0]  # Total number of images

    # Empty lists to hold training, validation, and testing data and labels
    train_images, val_images, test_images = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    # Process each class
    for i in range(n_classes):
        # Find indices where the label is 1 for the current class
        class_indices = np.where(labels[:, i] == 1)[0]

        # Randomly shuffle the class indices
        np.random.shuffle(class_indices)

        # Calculate split points
        n_train_samples = int(len(class_indices) * train_ratio)
        n_val_samples = int(len(class_indices) * val_ratio)

        # Split the indices into train, validation, and test indices
        train_indices = class_indices[:n_train_samples]
        val_indices = class_indices[n_train_samples:n_train_samples + n_val_samples]
        test_indices = class_indices[n_train_samples + n_val_samples:]

        # Append the split data and labels to their respective lists
        train_images.append(images[train_indices])
        train_labels.append(labels[train_indices])
        val_images.append(images[val_indices])
        val_labels.append(labels[val_indices])
        test_images.append(images[test_indices])
        test_labels.append(labels[test_indices])

    # Concatenate all class-specific splits into single arrays
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Optionally, shuffle the training, validation, and testing data
    train_perm = np.random.permutation(len(train_images))
    val_perm = np.random.permutation(len(val_images))
    test_perm = np.random.permutation(len(test_images))
    train_images = train_images[train_perm]
    train_labels = train_labels[train_perm]
    val_images = val_images[val_perm]
    val_labels = val_labels[val_perm]
    test_images = test_images[test_perm]
    test_labels = test_labels[test_perm]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

train_count = np.zeros((7,), dtype=int)

for index in range(len(label)):
    train_count[np.argmax(label[index])] += 1
    
print('total data distribution: ', train_count)
print('train: 60%   |   validation: 20%   |   test: 20%')
train_ratio = 0.6
start = 0
train_data, train_label, val_data, val_label, test_data, test_label = stratified_split(total_data, label)


# tr_mask = np.zeros((len(total_data),))

# for i in range(7):
#     tr_mask[start:int(start + train_count[i] * train_ratio)] = 1
#     start += train_count[i]

# tr_data = total_data[tr_mask > 0]
# tr_label = label[tr_mask > 0]
# val_test_data = total_data[tr_mask == 0]
# val_test_label = label[tr_mask == 0]

# test_ratio = 0.5
# test_mask = np.zeros((len(val_test_data),))
# test_count = np.zeros((7,), dtype=int)
# for index in range(len(val_test_label)):
#     test_count[np.argmax(val_test_label[index])] += 1

# start = 0
# for i in range(7):
#     test_mask[start:int(start + test_count[i] * test_ratio)] = 1
#     start += test_count[i]

# val_data = val_test_data[test_mask > 0]
# val_label = val_test_label[test_mask > 0]
# test_data = val_test_data[test_mask == 0]
# test_label = val_test_label[test_mask == 0]

# size = 64
# tr_data = tr_data.reshape(-1, 1, size, size)
# val_data = val_data.reshape(-1, 1, size, size)
# test_data = test_data.reshape(-1, 1, size, size)



np.save('train_data_wm811k.npy', train_data)
np.save('train_label_wm811k.npy', np.array(train_label))

np.save('val_data_wm811k.npy', val_data)
np.save('val_label_wm811k.npy', np.array(val_label))

np.save('test_data_wm811k.npy', test_data)
np.save('test_label_wm811k.npy', np.array(test_label))
print("    - [Finish] NPY file is saved.")




